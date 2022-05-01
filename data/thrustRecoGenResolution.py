import matplotlib.pyplot as plt
import uproot
import numpy as np
import sys

plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

def main():

    x = uproot.open(sys.argv[1])

    t = np.stack(np.array(x['t']['Thrust']))
    tgen = np.stack(np.array(x['tgen']['Thrust']))

    fig, [ax,rx] = plt.subplots(2,1,constrained_layout=False,sharey=False,sharex=True,gridspec_kw={"height_ratios": [3.5,1], 'hspace':0.0},)

    bins = np.linspace(-.3,.3,100)
    noTrkSel, bin_edges, _ = ax.hist(t[:,0] - tgen[:,0],bins=bins,histtype="step",label="no track selection", color="blue", lw=1.5)
    withTrkSel, bin_edges, _ = ax.hist(t[:,1] - tgen[:,0],bins=bins,histtype="step",label="with track selection", color="black", lw=1.5)
    ax.set_ylabel("Number of Events")
    ax.set_yscale("log")
    ax.legend(loc="upper left", prop={'size': 8}, framealpha=0.0)

    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2, noTrkSel/(withTrkSel+10**-50), 'o-', label = 'Ratio', color = 'black', lw=1.5, ms=1.5)
    rx.plot((bin_edges[:-1] + bin_edges[1:]) / 2,[1] * len((bin_edges[:-1] + bin_edges[1:]) / 2), ls="--",color="gray",alpha=0.8)
    rx.set_xlabel("Thrust Reco - Thrust Gen")
    rx.set_ylabel("Ratio")
    rx.set_ylim(0,2)
    
    plt.savefig("thrust_resolution.pdf",bbox_inches="tight")
    # plt.show()

if __name__ == "__main__":
    main()  