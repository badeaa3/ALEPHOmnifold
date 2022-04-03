/*

Author: Anthony Badea
Date: April 2, 2022

Default event selections:
// passesTotalChgEnergyMin = TotalChgEnergy >= 15;
// passesNTrkMin = NTrk >= 5;
// passesSTheta = TMath::Abs(TMath::Cos(STheta)) <= .82;
// passesMissP = MissP < 20;
// passesNeuNch = (Neu+NTrk)>=13;
// if(nParticle < 4) continue;

*/

// root dependencies
#include "TTree.h"
#include "TFile.h"
#include "TVector3.h"
#include "TMath.h"
#include "TString.h"

// thrust code
#include "thrustTools.h"
#include "sphericityTools.h"

// c++ code
#include <vector>
#include <iostream>
#include <iomanip>

std::map<std::string, float> getTrackVariation(
  /* charged track selections */
  int nTPCcut, // 2PC paper value: 4
  float chargedTracksAbsCosThCut, // 0.94
  float ptCut, // 0.2
  float d0Cut, // 2
  float z0Cut, // 10
  /* neutral track selections */
  float ECut, // 0.4
  float neutralTracksAbsCosThCut // 0.98
) {
  return std::map<std::string, float> {
    {"nTPCcut", nTPCcut},
    {"chargedTracksAbsCosThCut", chargedTracksAbsCosThCut},
    {"ptCut", ptCut},
    {"d0Cut", d0Cut},
    {"z0Cut", z0Cut},
    {"ECut", ECut},
    {"neutralTracksAbsCosThCut", neutralTracksAbsCosThCut}
  };
}

std::map<std::string, float> getEventVariation(
  /* event selections */
  float TotalChgEnergyCut, // 2PC paper value: 15
  float NTrkCut, // 5
  float AbsCosSThetaCut, // 0.82
  float NeuNchCut // 13
) {
  return std::map<std::string, float> {
    {"TotalChgEnergyCut", TotalChgEnergyCut},
    {"NTrkCut", NTrkCut},
    {"AbsCosSThetaCut", AbsCosSThetaCut},
    {"NeuNchCut", NeuNchCut}
  };
}

void pbftp(double time_diff, int nprocessed, int ntotal) {
  double rate = (double)(nprocessed + 1) / time_diff;
  std::cout << "\r > " << nprocessed << " / " << ntotal
            << " | "   << std::fixed << std::setprecision(1) << 100 * (double)(nprocessed) / (double)(ntotal) << "%"
            << " | "   << std::fixed << std::setprecision(1) << rate << "Hz"
            << " | "   << std::fixed << std::setprecision(1) << time_diff / 60 << "m elapsed"
            << " | "   << std::fixed << std::setprecision(1) << (double)(ntotal - nprocessed) / (rate * 60) << "m remaining";
  std::cout << std::flush;
}

int main(int argc, char* argv[]) {

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% User Input %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::string inFileName = "";
  std::string outFileName = "";
  int nEvents = -1;
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "-i", 2) == 0) inFileName = argv[i + 1];
    if (strncmp(argv[i], "-o", 2) == 0) outFileName = argv[i + 1];
    if (strncmp(argv[i], "-n", 2) == 0) nEvents = std::stoi(argv[i + 1]);
  }
  if (inFileName == "") {
    std::cout << "No input file name provided. Exiting" << std::endl;
    return 0;
  }
  if (outFileName == "" && inFileName != "") {
    outFileName = inFileName;
    outFileName.erase(outFileName.length() - 5); // strip .root
    outFileName += "_thrust.root";
  }

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Input Data %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::unique_ptr<TFile> f (new TFile(inFileName.c_str(), "READ"));
  std::unique_ptr<TTree> t ((TTree*) f->Get("t"));

  // declare variables
  int maxPart = 500;
  int nParticle;
  bool passesNTupleAfterCut;
  int pwflag[maxPart];
  float theta[maxPart];
  float pt[maxPart];
  float d0[maxPart];
  float z0[maxPart];
  int ntpc[maxPart];
  float px[maxPart];
  float py[maxPart];
  float pz[maxPart];
  float pmag[maxPart];
  float mass[maxPart];

  // link memory
  t->SetBranchAddress("nParticle", &nParticle);
  t->SetBranchAddress("passesNTupleAfterCut", &passesNTupleAfterCut);
  t->SetBranchAddress("pwflag", &pwflag);
  t->SetBranchAddress("theta", &theta);
  t->SetBranchAddress("pt", &pt);
  t->SetBranchAddress("d0", &d0);
  t->SetBranchAddress("z0", &z0);
  t->SetBranchAddress("ntpc", &ntpc);
  t->SetBranchAddress("px", &px);
  t->SetBranchAddress("py", &py);
  t->SetBranchAddress("pz", &pz);
  t->SetBranchAddress("pmag", &pmag);
  t->SetBranchAddress("mass", &mass);

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Output File %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::unique_ptr<TFile> fout (new TFile(outFileName.c_str(), "RECREATE"));

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Track Variations %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::vector<std::map<std::string, float> > trackVariations; // vector of variations
  // nominal values
  trackVariations.push_back(getTrackVariation(4, 0.94, 0.2, 2, 10, 0.4, 0.98));
  // ntpc variations
  for (int i = 0; i <= 7; i++) {
    if (i != trackVariations.at(0)["nTPCcut"]) {
      trackVariations.push_back(getTrackVariation(i, 0.94, 0.2, 2, 10, 0.4, 0.98));
    }
  }

  // vectors for selected objects
  std::vector<int> selectedParts;
  std::vector<std::vector<float> > selectedPx, selectedPy, selectedPz;
  std::vector<std::vector<Short_t> > selectedPwflag;
  // event level quantities
  TVector3 thrust;
  std::unique_ptr<Sphericity> spher;

  // save variation definitions to a tree
  std::unique_ptr<TTree> varDefs (new TTree("TrackVariationDefinitions", ""));
  int nTPCcut;
  float chargedTracksAbsCosThCut, ptCut, d0Cut, z0Cut, ECut, neutralTracksAbsCosThCut;
  varDefs->Branch("nTPCcut", &nTPCcut);
  varDefs->Branch("chargedTracksAbsCosThCut", &chargedTracksAbsCosThCut);
  varDefs->Branch("ptCut", &ptCut);
  varDefs->Branch("d0Cut", &d0Cut);
  varDefs->Branch("z0Cut", &z0Cut);
  varDefs->Branch("ECut", &ECut);
  varDefs->Branch("neutralTracksAbsCosThCut", &neutralTracksAbsCosThCut);

  // push back for each variation and save to tree
  for (int iV = 0; iV < trackVariations.size(); iV++) {
    selectedParts.push_back(0);
    selectedPx.push_back(std::vector<float>());
    selectedPy.push_back(std::vector<float>());
    selectedPz.push_back(std::vector<float>());
    selectedPwflag.push_back(std::vector<Short_t>());
    nTPCcut = trackVariations.at(iV)["nTPCcut"];
    chargedTracksAbsCosThCut = trackVariations.at(iV)["chargedTracksAbsCosThCut"];
    ptCut = trackVariations.at(iV)["ptCut"];
    d0Cut = trackVariations.at(iV)["d0Cut"];
    z0Cut = trackVariations.at(iV)["z0Cut"];
    ECut = trackVariations.at(iV)["ECut"];
    neutralTracksAbsCosThCut = trackVariations.at(iV)["neutralTracksAbsCosThCut"];
    varDefs->Fill();
  }
  varDefs->Write();

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Event Variations %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::vector<std::map<std::string, float> > eventVariations; // vector of variations
  // nominal values
  eventVariations.push_back(getEventVariation(15, 5, 0.82, 13));
  // total charged energy variation
  eventVariations.push_back(getEventVariation(10, 5, 0.82, 13));

  // save variation definitions to a tree
  std::unique_ptr<TTree> evtVarDefs (new TTree("EventVariationDefinitions", ""));
  float TotalChgEnergyCut, AbsCosSThetaCut;
  int NTrkCut, NeuNchCut;
  evtVarDefs->Branch("TotalChgEnergyCut", &TotalChgEnergyCut);
  evtVarDefs->Branch("AbsCosSThetaCut", &AbsCosSThetaCut);
  evtVarDefs->Branch("NTrkCut", &NTrkCut);
  evtVarDefs->Branch("NeuNchCut", &NeuNchCut);
  for (int iV = 0; iV < eventVariations.size(); iV++) {
    TotalChgEnergyCut = eventVariations.at(iV)["TotalChgEnergyCut"];
    AbsCosSThetaCut = eventVariations.at(iV)["AbsCosSThetaCut"];
    NTrkCut = eventVariations.at(iV)["NTrkCut"];
    NeuNchCut = eventVariations.at(iV)["NeuNchCut"];
    evtVarDefs->Fill();
  }
  evtVarDefs->Write();

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Data Tree %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::unique_ptr<TTree> tout (new TTree("t", ""));
  std::vector<float> Thrust, TotalChgEnergy, STheta;
  std::vector<int> NTrk, Neu;
  std::vector<std::vector<bool> > passEventSelection(eventVariations.size()); // allocate memory here without push_back to avoid copying which confuses tree->Branch
  tout->Branch("Thrust", &Thrust);
  tout->Branch("TotalChgEnergy", &TotalChgEnergy);
  tout->Branch("NTrk", &NTrk);
  tout->Branch("Neu", &Neu);
  tout->Branch("STheta", &STheta);
  for (int iV = 0; iV < eventVariations.size(); iV++) {
    tout->Branch(("passEventSelection_" + std::to_string(iV)).c_str(), &passEventSelection.at(iV));
  }

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Event Loop %%%%%%%%%%%%%%%%%%%%%%%%%%#
  if (nEvents == -1) nEvents = t->GetEntries();
  // timekeeper
  std::chrono::time_point<std::chrono::system_clock> time_start = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds;

  for (int iE = 0; iE < nEvents; iE++ ) {

    // progressbar
    elapsed_seconds = (std::chrono::system_clock::now() - time_start);
    pbftp(elapsed_seconds.count(), iE + 1, nEvents);

    t->GetEntry(iE);

    // reset variables
    TotalChgEnergy.clear();
    NTrk.clear();
    Neu.clear();
    STheta.clear();
    Thrust.clear();
    for (int iV = 0; iV < trackVariations.size(); iV++) {
      selectedParts.at(iV) = 0;
      selectedPx.at(iV).clear();
      selectedPy.at(iV).clear();
      selectedPz.at(iV).clear();
      selectedPwflag.at(iV).clear();
      TotalChgEnergy.push_back(0);
      NTrk.push_back(0);
      Neu.push_back(0);
    }
    for (auto &ev : passEventSelection) ev.clear();

    // compute event selection variables
    for (int iP = 0; iP < nParticle; iP++) {

      // loop over variations
      for (int iV = 0; iV < trackVariations.size(); iV++) {

        // count charged tracks
        bool chargedTrackSelections =
          (pwflag[iP] >= 0 && pwflag[iP] <= 2)
          && TMath::Abs(cos(theta[iP])) <= trackVariations.at(iV)["chargedTracksAbsCosThCut"]
          && pt[iP] >= trackVariations.at(iV)["ptCut"]
          && TMath::Abs(d0[iP]) <= trackVariations.at(iV)["d0Cut"]
          && TMath::Abs(z0[iP]) <= trackVariations.at(iV)["z0Cut"]
          && ntpc[iP] >= trackVariations.at(iV)["nTPCcut"];
        if (chargedTrackSelections) {
          TotalChgEnergy.at(iV) += TMath::Sqrt(pmag[iP] * pmag[iP] + mass[iP] * mass[iP]);
          NTrk.at(iV) += 1;
        }

        // count neutral tracks
        bool neutralTrackSelections =
          (pwflag[iP] == 4 || pwflag[iP] == 5)
          && TMath::Sqrt(pmag[iP] * pmag[iP] + mass[iP] * mass[iP]) >= trackVariations.at(iV)["ECut"]
          && TMath::Abs(cos(theta[iP])) <= trackVariations.at(iV)["neutralTracksAbsCosThCut"];
        if (neutralTrackSelections) {
          Neu.at(iV) += 1;
        }

        // add to input list for thrust
        if (chargedTrackSelections || neutralTrackSelections) {
          selectedParts.at(iV) += 1;
          selectedPx.at(iV).push_back(px[iP]);
          selectedPy.at(iV).push_back(py[iP]);
          selectedPz.at(iV).push_back(pz[iP]);
          selectedPwflag.at(iV).push_back(pwflag[iP]);
        }
      }
    }

    // compute event level variables
    for (int iV = 0; iV < trackVariations.size(); iV++) {

      // sphericity
      spher = std::make_unique<Sphericity>(Sphericity(selectedParts.at(iV), selectedPx.at(iV).data(), selectedPy.at(iV).data(), selectedPz.at(iV).data(), selectedPwflag.at(iV).data(), false));
      STheta.push_back(spher->sphericityAxis().Theta());
      // thrust
      thrust = getThrust(selectedParts.at(iV), selectedPx.at(iV).data(), selectedPy.at(iV).data(), selectedPz.at(iV).data(), THRUST::OPTIMAL); //, false, false, pDataReader.weight);
      Thrust.push_back(thrust.Mag());

      // compute event selection passes
      for (int iEV = 0; iEV < eventVariations.size(); iEV++) {
        passEventSelection.at(iEV).push_back(
          passesNTupleAfterCut == 1
          && TotalChgEnergy.at(iV) >= eventVariations.at(iEV)["TotalChgEnergyCut"]
          && NTrk.at(iV) >= eventVariations.at(iEV)["NTrk"]
          && TMath::Abs(TMath::Cos(STheta.at(iV))) <= eventVariations.at(iEV)["AbsCosSThetaCut"]
          && (NTrk.at(iV) + Neu.at(iV)) >= eventVariations.at(iEV)["NeuNchCut"]
        );
      }
    }

    // fill output tree
    tout->Fill();

  }

  // write to output file
  tout->Write();

  return 1;
}