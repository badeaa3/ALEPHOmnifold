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

// thrust code
#include "thrustTools.h"
#include "sphericityTools.h"

// c++ code
#include <vector>
#include <iostream>

std::map<std::string, float> getVariation(
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

int main(int argc, char* argv[]) {

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% User Input %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::string inFileName = "/Users/anthonybadea/Documents/ALEPH/ALEPH/LEP1Data1992_recons_aftercut-MERGED.root";
  std::string outFileName = "thrust.root";

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

  // create output file
  std::unique_ptr<TFile> fout (new TFile(outFileName.c_str(), "RECREATE"));

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Variations %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::vector<std::map<std::string, float> > variations; // vector of variations
  variations.push_back(getVariation(4, 0.94, 0.2, 2, 10, 0.4, 0.98));
  variations.push_back(getVariation(3, 0.94, 0.2, 2, 10, 0.4, 0.98));

  // vectors for selected objects
  std::vector<int> selectedParts;
  std::vector<std::vector<float> > selectedPx, selectedPy, selectedPz;
  std::vector<std::vector<Short_t> > selectedPwflag;
  // event level quantities
  TVector3 thrust;
  std::unique_ptr<Sphericity> spher;

  // save variation definitions to a tree
  std::unique_ptr<TTree> varDefs (new TTree("VariationDefinitions", ""));
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
  for (int iV = 0; iV < variations.size(); iV++) {
    selectedParts.push_back(0);
    selectedPx.push_back(std::vector<float>());
    selectedPy.push_back(std::vector<float>());
    selectedPz.push_back(std::vector<float>());
    selectedPwflag.push_back(std::vector<Short_t>());
    nTPCcut = variations.at(iV)["nTPCcut"];
    chargedTracksAbsCosThCut = variations.at(iV)["chargedTracksAbsCosThCut"];
    ptCut = variations.at(iV)["ptCut"];
    d0Cut = variations.at(iV)["d0Cut"];
    z0Cut = variations.at(iV)["z0Cut"];
    ECut = variations.at(iV)["ECut"];
    neutralTracksAbsCosThCut = variations.at(iV)["neutralTracksAbsCosThCut"];
    varDefs->Fill();
  }
  varDefs->Write();

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Data Tree %%%%%%%%%%%%%%%%%%%%%%%%%%#
  std::unique_ptr<TTree> tout (new TTree("t", ""));
  std::vector<float> Thrust, TotalChgEnergy, STheta_copy;
  std::vector<int> NTrk, Neu;
  tout->Branch("Thrust", &Thrust);
  tout->Branch("TotalChgEnergy", &TotalChgEnergy);
  tout->Branch("NTrk", &NTrk);
  tout->Branch("Neu", &Neu);
  tout->Branch("STheta", &STheta_copy);
  tout->Branch("passesNTupleAfterCut", &passesNTupleAfterCut);

  // #%%%%%%%%%%%%%%%%%%%%%%%%%% Event Loop %%%%%%%%%%%%%%%%%%%%%%%%%%#
  int nEvents = 10; //t->GetEntries();
  for (int iE = 0; iE < nEvents; iE++ ) {

    t->GetEntry(iE);

    // reset variables
    TotalChgEnergy.clear();
    NTrk.clear();
    Neu.clear();
    STheta_copy.clear();
    Thrust.clear();
    for (int iV = 0; iV < variations.size(); iV++) {
      selectedParts.at(iV) = 0;
      selectedPx.at(iV).clear();
      selectedPy.at(iV).clear();
      selectedPz.at(iV).clear();
      selectedPwflag.at(iV).clear();
      TotalChgEnergy.push_back(0);
      NTrk.push_back(0);
      Neu.push_back(0);
    }

    // compute event selection variables
    for (int iP = 0; iP < nParticle; iP++) {

      // loop over variations
      for (int iV = 0; iV < variations.size(); iV++) {

        // count charged tracks
        bool chargedTrackSelections =
          (pwflag[iP] >= 0 && pwflag[iP] <= 2)
          && TMath::Abs(cos(theta[iP])) <= variations.at(iV)["chargedTracksAbsCosThCut"]
          && pt[iP] >= variations.at(iV)["ptCut"]
          && TMath::Abs(d0[iP]) <= variations.at(iV)["d0Cut"]
          && TMath::Abs(z0[iP]) <= variations.at(iV)["z0Cut"]
          && ntpc[iP] >= variations.at(iV)["nTPCcut"];
        if (chargedTrackSelections) {
          TotalChgEnergy.at(iV) += TMath::Sqrt(pmag[iP] * pmag[iP] + mass[iP] * mass[iP]);
          NTrk.at(iV) += 1;
        }

        // count neutral tracks
        bool neutralTrackSelections =
          (pwflag[iP] == 4 || pwflag[iP] == 5)
          && TMath::Sqrt(pmag[iP] * pmag[iP] + mass[iP] * mass[iP]) >= variations.at(iV)["ECut"]
          && TMath::Abs(cos(theta[iP])) <= variations.at(iV)["neutralTracksAbsCosThCut"];
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
    for (int iV = 0; iV < variations.size(); iV++) {
      // sphericity
      spher = std::make_unique<Sphericity>(Sphericity(selectedParts.at(iV), selectedPx.at(iV).data(), selectedPy.at(iV).data(), selectedPz.at(iV).data(), selectedPwflag.at(iV).data(), false));
      STheta_copy.push_back(spher->sphericityAxis().Theta());
      // thrust
      thrust = getThrust(selectedParts.at(iV), selectedPx.at(iV).data(), selectedPy.at(iV).data(), selectedPz.at(iV).data(), THRUST::OPTIMAL); //, false, false, pDataReader.weight);
      Thrust.push_back(thrust.Mag());
    }

    // fill output tree
    tout->Fill();
  }

  // write to output file
  tout->Write();

  return 1;
}