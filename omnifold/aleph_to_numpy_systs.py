'''
Author: Anthony Badea
Date: January 30, 2022

A python script that takes in the files and outputs a numpy array with the thrust value, event selections, and event weights.

File names:
data_fileName = 'LEP1Data{}_recons_aftercut-MERGED.root'.format(year), with year = 1992, 1993, 1994, 1995
mc_fileName = 'alephMCRecoAfterCutPaths_1994.root'

Tree definitions:
- t = data or reco
- tgen = generation level + hadronic event selection
- tgenBefore = generation level without hadronic event selection

Notes:
- see to understand event/track selections https://www.dropbox.com/scl/fi/gqe7qgm4ygnr7xontuke4/ALEPH-Omnifold-073020.pdf?dl=0

Systematic uncertainty variation from https://arxiv.org/pdf/1906.00489.pdf:
The required number of hits a track leaves in the ALEPH time projection chamber was varied from 4 to 7. 
From this variation, the tracking uncertainty is estimated to be 0.7% in the lab coordinate analysis and 0.3% in the thrust coordinate analysis. 
The hadronic event selection was studied by changing the required charged energy in an event to be 10 instead of 15 GeV. 
... 
An additional systematic of 0.2%-10% (0.1%-0.5%) in the lab (thrust) coordinate analysis is included to quantify the residual 
uncertainty in the reconstruction effect correction factor derived from the pythia 6.1 archived MC sample, which is mainly from the 
limited size of the archived MC sample.
'''

import uproot
import numpy as np
import argparse

def main():

    # user options
    ops = options()

    # configurations
    output = aleph_to_numpy(ops.i)

    # print summary
    for key, val in output.items():
        print(f"{key} of size {val.shape}")

    # save to file
    np.savez(ops.o, **output)
    
def options():
    ''' argument parser to handle inputs from the user '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input file")
    parser.add_argument("-o", help="Output file name", default="test.npz")
    return parser.parse_args()

def aleph_to_numpy(fileName):

    # load the data file and infer type (data or mc)
    file = uproot.open(fileName)
    
    # list of relevenat event selections
    event_selections = [f'passEventSelection_{i}' for i in range(file["EventVariationDefinitions"].num_entries)]

    if "LEP1Data" in fileName: # data

        output = {f"{evsel}" : np.stack(np.array(file["t"][evsel])) for evsel in event_selections}
        output["thrust"] = np.stack(np.array(file["t"]["Thrust"]))
        return output

    else: # MC  

        output = {
            "t_thrust" : np.stack(np.array(file["t"]["Thrust"])), # reco (after detector simulation)
            "tgen_thrust" : np.stack(np.array(file["tgen"]["Thrust"])), # with hadronic event selection
            "tgenBefore_thrust" : np.stack(np.array(file["tgenBefore"]["Thrust"])) # without hadronic event selection
        }
        # construct the final event selection
        for evsel in event_selections:
            output[f"t_{evsel}"] = np.stack(np.array(file["t"][evsel]))
            output[f"tgen_{evsel}"] = np.stack(np.array(file["t"][evsel]))

        return output

if __name__ == "__main__":
    main()