import numpy as np
#import aleph_to_numpy

#myoutMC = aleph_to_numpy.aleph_to_numpy("../../alephMCRecoAfterCutPaths_1994.root")
#myoutData = aleph_to_numpy.aleph_to_numpy("../../LEP1Data1994_recons_aftercut-MERGED.root")

myoutMC = np.load("/eos/home-a/abadea/data/aleph/thrust/20220514/alephMCRecoAfterCutPaths_1994_ThrustReprocess.npz",allow_pickle=True)
myoutData = np.load("/eos/home-a/abadea/data/aleph/thrust/20220514/LEP1Data1994_recons_aftercut-MERGED_ThrustReprocess.npz",allow_pickle=True)

#a = myoutMC['WHES_id']
#b = myoutMC['WHOES_id']
a = myoutMC["tgen_uniqueID"]
b = myoutMC["tgenBefore_uniqueID"]
intersect, ind_a, ind_b = np.intersect1d(a,b, return_indices=True)

pass_reco = np.zeros(len(b))
#pass_reco[ind_b] = myoutMC['reco_passselection'][ind_a]
pass_reco[ind_b] = myoutMC['t_passEventSelection_0'][:,0][ind_a] # pick up nominal event selection (passEventSelection_0) applied to nominal track selection (slice 0)

reco_vals = -999.*np.ones(len(b))
#reco_vals[ind_b] = myoutMC['reco_thrust'][ind_a]
reco_vals[ind_b] = myoutMC['t_thrust'][:,0][ind_a] # pick up nominal (slice 0) thrust value

#dict_keys(['truthWHES_thrust', 'truthWHES_passselection', 'truthWHES_eventweight', 'truthWOHES_thrust', 'truthWOHES_passselection', 'truthWOHES_eventweight', 'reco_passselection', 'reco_thrust', 'reco_eventweight', 'WHES_id', 'WHOES_id'])

# save MC
print(myoutMC['tgenBefore_thrust'].shape, ind_b.shape)
print(pass_reco.shape, myoutMC['tgenBefore_passEventSelection'][ind_b].shape, reco_vals.shape, np.concatenate(myoutMC['tgenBefore_thrust'][ind_b])[:4])
np.save("MC_pass_reco",pass_reco)
#np.save("MC_pass_truth",myoutMC['truthWOHES_passselection'])
np.save("MC_pass_truth",myoutMC['tgenBefore_passEventSelection'][ind_b])
np.save("MC_vals_reco",reco_vals)
#np.save("MC_vals_truth",myoutMC['truthWOHES_thrust'])
np.save("MC_vals_truth",np.concatenate(myoutMC['tgenBefore_thrust'][ind_b]))

# Save data
#np.save("data_pass_reco",myoutData['data_passselection'])
np.save("data_pass_reco",myoutData['t_passEventSelection_0'][:,0])
#np.save("data_vals_reco",myoutData['data_thrust'])
np.save("data_vals_reco",myoutData['t_thrust'][:,0])
