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

// c++ code
#include <vector>
#include <iostream>

int main(int argc, char* argv[]) {

  // parse input
  std::string inFileName = "/Users/anthonybadea/Documents/ALEPH/ALEPH/LEP1Data1992_recons_aftercut-MERGED.root";
  std::string outFileName = "thrust.root";

  // load input data
  std::unique_ptr<TFile> f (new TFile(inFileName.c_str(), "READ"));
  std::unique_ptr<TTree> t ((TTree*) f->Get("t"));

  // declare variables
  int maxPart = 500;
  int nParticle;
  bool passesNTupleAfterCut;
  float STheta;
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
  t->SetBranchAddress("STheta", &STheta);
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

  // create output tree
  std::unique_ptr<TFile> fout (new TFile(outFileName.c_str(), "RECREATE"));
  std::unique_ptr<TTree> tout (new TTree("t", ""));
  float Thrust, TotalChgEnergy, STheta_copy;
  int NTrk, Neu, passesNTupleAfterCut_copy;
  tout->Branch("Thrust", &Thrust, "Thrust/F");
  tout->Branch("TotalChgEnergy", &TotalChgEnergy, "TotalChgEnergy/F");
  tout->Branch("NTrk", &NTrk, "NTrk/I");
  tout->Branch("Neu", &Neu, "Neu/I");
  tout->Branch("STheta", &STheta_copy, "STheta/F");
  tout->Branch("passesNTupleAfterCut", &passesNTupleAfterCut_copy, "passesNTupleAfterCut/I");

  // variables used in loop
  int selectedParts;
  std::vector<float> selectedPx, selectedPy, selectedPz;
  TVector3 thrust;

  // charged track selections
  bool chargedTrackSelections;
  int nTPCcut = 4;
  float chargedTracksAbsCosThCut = 0.94; //maximum abs(cos(th)) of charged tracks
  float ptCut = 0.2;
  float d0Cut = 2;
  float z0Cut = 10;

  // neutral track selections
  bool neutralTrackSelections;
  float ECut = 0.4;
  float neutralTracksAbsCosThCut = 0.98; //maximum abs(cos(th)) of charged NeutralHadrons

  // loop over events
  int nEvents = t->GetEntries();
  for (int iE = 0; iE < nEvents; iE++ ) {

    t->GetEntry(iE);

    // reset variables
    TotalChgEnergy = 0;
    NTrk = 0;
    Neu = 0;
    STheta_copy = STheta;
    passesNTupleAfterCut_copy = passesNTupleAfterCut;
    selectedParts = 0;
    selectedPx.clear();
    selectedPy.clear();
    selectedPz.clear();

    // compute event selection variables
    for (int iP = 0; iP < nParticle; iP++) {

      // count charged tracks
      bool chargedTrackSelections =
        (pwflag[iP] >= 0 && pwflag[iP] <= 2)
        && TMath::Abs(cos(theta[iP])) <= chargedTracksAbsCosThCut
        && pt[iP] >= ptCut
        && TMath::Abs(d0[iP]) <= d0Cut
        && TMath::Abs(z0[iP]) <= z0Cut
        && ntpc[iP] >= nTPCcut;
      if (chargedTrackSelections) {
        TotalChgEnergy += TMath::Sqrt(pmag[iP] * pmag[iP] + mass[iP] * mass[iP]);
        NTrk += 1;
      }

      // count neutral tracks
      bool neutralTrackSelections =
        (pwflag[iP] == 4 || pwflag[iP] == 5)
        && TMath::Sqrt(pmag[iP] * pmag[iP] + mass[iP] * mass[iP]) >= ECut
        && TMath::Abs(cos(theta[iP])) <= neutralTracksAbsCosThCut;
      if (neutralTrackSelections) {
        Neu += 1;
      }

      // add to input list for thrust
      selectedParts += 1;
      selectedPx.push_back(px[iP]);
      selectedPy.push_back(py[iP]);
      selectedPz.push_back(pz[iP]);
    }

    // compute thrust
    thrust = getThrust(selectedParts, selectedPx.data(), selectedPy.data(), selectedPz.data(), THRUST::OPTIMAL); //, false, false, pDataReader.weight);

    // fill output tree
    Thrust = thrust.Mag();
    tout->Fill();
  }

  // write to output file
  tout->Write();

  return 1;
}
