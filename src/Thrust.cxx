
// root dependencies
#include "TTree.h"
#include "TFile.h"
#include "TVector3.h"

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
  float px[maxPart];
  float py[maxPart];
  float pz[maxPart];

  // link memory
  t->SetBranchAddress("nParticle", &nParticle);
  t->SetBranchAddress("px", &px);
  t->SetBranchAddress("py", &py);
  t->SetBranchAddress("pz", &pz);

  // create output tree
  std::unique_ptr<TFile> fout (new TFile(outFileName.c_str(), "RECREATE"));
  std::unique_ptr<TTree> tout (new TTree("t", ""));
  float Thrust;
  tout->Branch("Thrust", &Thrust, "Thrust/F");

  // variables used in loop
  TVector3 thrust;

  // loop over events
  int nEvents = t->GetEntries();
  for (int iE = 0; iE < nEvents; iE++ ) {

    // use input data
    t->GetEntry(iE);
    thrust = getThrust(nParticle, px, py, pz, THRUST::OPTIMAL); //, false, false, pDataReader.weight);

    // fill output tree
    Thrust = thrust.Mag();
    tout->Fill();
  }

  // write to output file
  tout->Write();

  return 1;
}
