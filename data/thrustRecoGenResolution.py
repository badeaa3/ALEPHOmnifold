import matplotlib.pyplot as plt
import uproot
import numpy as np
import sys
import argparse
import os

plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inFile", help="Input file", required=True)
    parser.add_argument("-o", "--outDir", help="Output directory for plots", required=True)
    ops = parser.parse_args()

    x = uproot.open(ops.inFile)

    recoEventSelection = np.stack(np.array(x['t']['passEventSelection_0']))
    t = np.stack(np.array(x['t']['Thrust']))
    tgen = np.stack(np.array(x['tgen']['Thrust']))

    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
    bins = np.linspace(-.3,.3,100)
    noTrkSel, bin_edges, _ = ax.hist(t[:,0] - tgen[:,0],bins=bins,histtype="step",label="no track selection", color="black", lw=1.5)
    withTrkSel = []
    for i in range(1,t.shape[1]):
        hist, bin_edges, _ = ax.hist(t[:,i] - tgen[:,0], bins=bins, histtype="step", label=f"v{i}", lw=1.5)
        withTrkSel.append(hist)
    ax.set_ylabel("Number of Events")
    ax.set_yscale("log")
    ax.legend(title="No Event Selection", loc="upper left", prop={'size': 8}, framealpha=0.0)
    for hist in withTrkSel:
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, noTrkSel/(hist+10**-50), 'o-', label = f'v{i}', lw=1.5, ms=1.5)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="gray",alpha=0.8)
    rx.set_xlabel("Reco - Gen Thrust")
    rx.set_ylabel("Ratio")
    rx.set_ylim(0,2)
    plt.savefig(os.path.join(ops.outDir,"thrust_resolution.pdf"),bbox_inches="tight")

    event_selections = [i for i in x['t'].keys() if 'passEventSelection' in i] + [None]
    for iE, eSel in enumerate(event_selections):
        if eSel == None:
            temp = np.stack(np.array(x['t']['passEventSelection_0']))
            recoEventSelection = np.ones((temp.shape[0],temp.shape[1])).astype(bool)
        else:
            recoEventSelection = np.stack(np.array(x['t'][eSel]))
        fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)
        noTrkSel, bin_edges, _ = ax.hist((t[:,0] - tgen[:,0])[recoEventSelection[:,0]],bins=bins,histtype="step",label="no track selection", color="black", lw=1.5)
        withTrkSel = []
        for i in range(1,t.shape[1]):
            hist, bin_edges, _ = ax.hist((t[:,i] - tgen[:,0])[recoEventSelection[:,i]], bins=bins, histtype="step", label=f"v{i}", lw=1.5)
            withTrkSel.append(hist)
        ax.set_ylabel("Number of Events")
        ax.set_yscale("log")
        ax.legend(title=f"Event Selection v{iE}" if eSel else "No Event Selection", loc="upper left", prop={'size': 8}, framealpha=0.0)
        for hist in withTrkSel:
            rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, noTrkSel/(hist+10**-50), 'o-', label = f'v{i}', lw=1.5, ms=1.5)
        rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="gray",alpha=0.8)
        rx.set_xlabel("Reco - Gen Thrust")
        rx.set_ylabel("Ratio")
        rx.set_ylim(0,2)
        plt.savefig(os.path.join(ops.outDir,f"thrust_resolution_{eSel}.pdf"),bbox_inches="tight")
    
if __name__ == "__main__":
    main()  
