import numpy as np
import aleph_to_numpy
myoutMC = aleph_to_numpy.aleph_to_numpy("../../alephMCRecoAfterCutPaths_1994.root")
myoutData = aleph_to_numpy.aleph_to_numpy("../../LEP1Data1994_recons_aftercut-MERGED.root")

a = myoutMC['WHES_id']
b = myoutMC['WHOES_id']
intersect, ind_a, ind_b = np.intersect1d(a,b, return_indices=True)

pass_reco = np.zeros(len(b))
pass_reco[ind_b] = myoutMC['reco_passselection'][ind_a]

reco_vals = -999.*np.ones(len(b))
reco_vals[ind_b] = myoutMC['reco_thrust'][ind_a]

#dict_keys(['truthWHES_thrust', 'truthWHES_passselection', 'truthWHES_eventweight', 'truthWOHES_thrust', 'truthWOHES_passselection', 'truthWOHES_eventweight', 'reco_passselection', 'reco_thrust', 'reco_eventweight', 'WHES_id', 'WHOES_id'])

np.save("MC_pass_reco",pass_reco)
np.save("MC_pass_truth",myoutMC['truthWOHES_passselection'])
np.save("MC_vals_reco",reco_vals)
np.save("MC_vals_truth",myoutMC['truthWOHES_thrust'])
np.save("data_pass_reco",myoutData['data_passselection'])
np.save("data_vals_reco",myoutData['data_thrust'])
