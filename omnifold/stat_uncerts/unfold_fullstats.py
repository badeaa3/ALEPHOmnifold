import sys

'''
run like python unfold_fullstats.py
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from matplotlib import gridspec
import time

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

import os
#os.environ['CUDA_VISIBLE_DEVICES']="1"

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)

def reweight(events):
    f = model.predict(events, batch_size=5000)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights,posinf=1))


bins = {}

#thrust
bins[0] = np.linspace(0,0.5,20)

#logfile
logfile = open(sys.argv[1]+"_"+sys.argv[2]+"log.txt","w")

#Read in the data
theta_unknown_S = np.load("inputs/data_vals_reco.npy",allow_pickle=True)
pass_data = np.load("inputs/data_pass_reco.npy",allow_pickle=True)
theta_unknown_S = theta_unknown_S[pass_data]

#Read in the MC
theta0_S = np.load("inputs/MC_vals_reco.npy",allow_pickle=True)
theta0_G = np.load("inputs/MC_vals_truth.npy",allow_pickle=True)
weights_MC_sim = np.ones(len(theta0_S))
pass_reco = np.load("inputs/MC_pass_reco.npy",allow_pickle=True)
pass_truth = np.load("inputs/MC_pass_truth.npy",allow_pickle=True)
pass_fiducial = np.load("inputs/MC_pass_truth.npy",allow_pickle=True)

#Early stopping
earlystopping = EarlyStopping(patience=10,
                              verbose=True,
                              restore_best_weights=True)

#Now, for the unfolding!

nepochs = 500

starttime = time.time()

NNweights_step2 = np.ones(len(theta0_S))

#Set up the model
inputs = Input((1, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
mymodel = Model(inputs=inputs, outputs=outputs)

np.random.seed(int(sys.argv[1]) - int(sys.argv[1])%10)
dataw = np.ones(len(theta_unknown_S))
if (sys.argv[2]=="1"):
    dataw = np.random.poisson(1,len(theta_unknown_S))
if (sys.argv[2]=="2"):
    weights_MC_sim = np.random.poisson(1,len(weights_MC_sim))

for iteration in range(5):

    #Process the data
    print("on iteration=",iteration," processing data for step 1, time elapsed=",time.time()-starttime)
    logfile.write("on iteration="+str(iteration)+" processing data for step 1, time elapsed="+str(time.time()-starttime)+"\n")
    
    xvals_1 = np.concatenate([theta0_S[pass_reco==1],theta_unknown_S])
    yvals_1 = np.concatenate([np.zeros(len(theta0_S[pass_reco==1])),np.ones(len(theta_unknown_S))])
    weights_1 = np.concatenate([NNweights_step2[pass_reco==1]*weights_MC_sim[pass_reco==1],dataw])
    xvals_1 = (xvals_1 - np.mean(theta_unknown_S)) / np.std(theta_unknown_S)

    X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1,test_size=0.5)
    del xvals_1,yvals_1,weights_1
    gc.collect()

    Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
    Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)
    del w_train_1,w_test_1
    gc.collect()

    print("on iteration=",iteration," done processing data for step 1, time elapsed=",time.time()-starttime)
    print("data events = ",len(X_train_1[Y_train_1[:,0]==1]))
    print("MC events = ",len(X_train_1[Y_train_1[:,0]==0]))

    logfile.write("on iteration="+str(iteration)+" done processing data for step 1, time elapsed="+str(time.time()-starttime)+"\n")
    logfile.write("data events = "+str(len(X_train_1[Y_train_1[:,0]==1]))+"\n")
    logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==0]))+"\n")
    
    #Step 1
    print("on step 1, time elapsed=",time.time()-starttime)
    logfile.write("on step 1, time elapsed="+str(time.time()-starttime)+"\n")
    
    opt = tf.keras.optimizers.Adam(learning_rate=2e-6)
    mymodel.compile(loss=weighted_binary_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

    hist_s1 =  mymodel.fit(X_train_1,Y_train_1,
              epochs=nepochs,
              batch_size=50000,
              validation_data=(X_test_1,Y_test_1),
              callbacks=[earlystopping],
              verbose=1)

    print("done with step 1, time elapsed=",time.time()-starttime)
    logfile.write("done with step 1, time elapsed="+str(time.time()-starttime)+"\n")
    
    #Now, let's do some checking.

    ###
    # Loss
    ###

    fig = plt.figure(figsize=(7, 5)) 
    gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")
    ax0.minorticks_on()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(np.array(hist_s1.history['loss']),label="loss")
    plt.plot(np.array(hist_s1.history['val_loss']),label="val. loss",ls=":")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("OmniFold iteration "+str(iteration)+", step 1",loc="left",fontsize=20)
    plt.text(0.05, 1.15,'ALEPH', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
    plt.legend(frameon=False,fontsize=20)
    plt.locator_params(axis='x', nbins=5)
    plt.xscale("log")
    fig.savefig("storage_plots/"+sys.argv[1]+"_"+sys.argv[2]+"Iteration"+str(iteration)+"_Step1_loss.pdf",bbox_inches='tight')

    mypred = mymodel.predict((theta0_S - np.mean(theta_unknown_S)) / np.std(theta_unknown_S),batch_size=10000)
    mypred = mypred/(1.-mypred)
    mypred = mypred[:,0]
    mypred = np.squeeze(np.nan_to_num(mypred,posinf=1))

    ###
    # thrust
    ###

    fig = plt.figure(figsize=(7, 5)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")
    ax0.minorticks_on()
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=20)

    n_data,b,_=plt.hist(1.-theta_unknown_S,bins=bins[0],density=True,alpha=0.5,label="Data")
    n_MC,_,_=plt.hist(1.-theta0_S[pass_reco==1],bins=bins[0],weights=weights_MC_sim[pass_reco==1]*NNweights_step2[pass_reco==1],density=True,histtype="step",color="black",label="MC + step 2 (i-1)")
    n_Omni,_,_=plt.hist(1.-theta0_S[pass_reco==1],bins=bins[0],weights=weights_MC_sim[pass_reco==1]*NNweights_step2[pass_reco==1]*mypred[pass_reco==1],density=True,histtype="step",color="black",ls=":",label="MC + step 1 (i)")

    plt.ylabel("Normalized to unity",fontsize=20)
    plt.title("OmniFold iteration "+str(iteration)+", step 1",loc="left",fontsize=20)
    plt.text(0.05, 1.25,'ALEPH', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
    plt.legend(frameon=False,fontsize=15)
    plt.locator_params(axis='x', nbins=5)
    plt.yscale("log")

    ax1 = plt.subplot(gs[1])
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(direction="in",which="both")
    ax1.minorticks_on()
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.xlabel("Detector-level Thrust",fontsize=15)
    plt.ylabel("MC/data",fontsize=15)

    plt.plot(0.5*(b[0:-1]+b[1:]),n_MC/n_data,ls="--",color="black")
    plt.plot(0.5*(b[0:-1]+b[1:]),n_Omni/n_data,ls=":",color="black")

    fig.savefig("storage_plots/"+sys.argv[1]+"_"+sys.argv[2]+"Iteration"+str(iteration)+"_Step1_thrust.pdf",bbox_inches='tight')
    np.save("storage_files/"+sys.argv[1]+"_"+sys.argv[2]+"n_Omni_step1_thrust_iteration"+str(iteration),n_Omni)
    
    print("time for step 2, time elapsed=",time.time()-starttime)
    logfile.write("time for step 2, time elapsed="+str(time.time()-starttime)+"\n")
    
    xvals_2 = np.concatenate([theta0_G[pass_truth==1],theta0_G[pass_truth==1]])
    yvals_2 = np.concatenate([np.zeros(len(theta0_G[pass_truth==1])),np.ones(len(theta0_G[pass_truth==1]))])
    xvals_2 = (xvals_2 - np.mean(theta_unknown_S)) / np.std(theta_unknown_S)
    
    NNweights = mymodel.predict((theta0_S[pass_truth==1] - np.mean(theta_unknown_S)) / np.std(theta_unknown_S),batch_size=10000)
    NNweights = NNweights/(1.-NNweights)
    NNweights = NNweights[:,0]
    NNweights = np.squeeze(np.nan_to_num(NNweights,posinf=1))
    NNweights[pass_reco[pass_truth==1]==0] = 1.
    weights_2 = np.concatenate([NNweights_step2[pass_truth==1]*weights_MC_sim[pass_truth==1],NNweights*NNweights_step2[pass_truth==1]*weights_MC_sim[pass_truth==1]])

    X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2,test_size=0.5)
    del xvals_2,yvals_2,weights_2
    gc.collect()
    
    Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
    Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)
    del w_train_2,w_test_2
    gc.collect()

    print("on iteration=",iteration," done processing data for step 2, time elapsed=",time.time()-starttime)
    print("MC events = ",len(X_train_1[Y_train_1[:,0]==1]))
    print("MC events = ",len(X_train_1[Y_train_1[:,0]==0]))

    logfile.write("on iteration="+str(iteration)+" done processing data for step 2, time elapsed="+str(time.time()-starttime)+"\n")
    logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==1]))+"\n")
    logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==0]))+"\n")
    
    #step 2
    opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
    mymodel.compile(loss=weighted_binary_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])
    hist_s2 =  mymodel.fit(X_train_2,Y_train_2,
              epochs=nepochs,
              batch_size=100000,
              validation_data=(X_test_2,Y_test_2),
              callbacks=[earlystopping],
              verbose=1)

    print("on iteration=",iteration," finished step 2; time elapsed=",time.time()-starttime)
    logfile.write("on iteration="+str(iteration)+" finished step 2; time elapsed="+str(time.time()-starttime)+"\n")
    
    NNweights_step2_hold = mymodel.predict((theta0_G - np.mean(theta_unknown_S)) / np.std(theta_unknown_S),batch_size=10000)
    NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
    NNweights_step2_hold = NNweights_step2_hold[:,0]
    NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
    NNweights_step2_hold[pass_truth==0] = 1.
    NNweights_step2 = NNweights_step2_hold*NNweights_step2

    #Now, let's do some checking.

    ###
    # Loss
    ###

    fig = plt.figure(figsize=(7, 5)) 
    gs = gridspec.GridSpec(1, 1, height_ratios=[1]) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")
    ax0.minorticks_on()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(np.array(hist_s2.history['loss']),label="loss")
    plt.plot(np.array(hist_s2.history['val_loss']),label="val. loss",ls=":")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
    plt.text(0.05, 1.15,'ALEPH', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
    plt.legend(frameon=False,fontsize=20)
    plt.locator_params(axis='x', nbins=5)
    plt.xscale("log")
    fig.savefig("storage_plots/"+sys.argv[1]+"_"+sys.argv[2]+"Step2_loss.pdf",bbox_inches='tight')

    ###
    # Thrust
    ###

    fig = plt.figure(figsize=(7, 5)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")
    ax0.minorticks_on()
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=20)

    n_data,b,_=plt.hist(1.-theta0_G[pass_fiducial==1],bins=bins[0],weights=weights_MC_sim[pass_fiducial==1]*NNweights[pass_fiducial==1]*NNweights_step2[pass_fiducial==1]/NNweights_step2_hold[pass_fiducial==1],density=True,alpha=0.5,label="MC + step 1")
    n_MC,_,_=plt.hist(1.-theta0_G[pass_fiducial==1],bins=bins[0],weights=weights_MC_sim[pass_fiducial==1],density=True,histtype="step",color="black",label="MC")
    n_Omni_step2_pT,_,_=plt.hist(1.-theta0_G[pass_fiducial==1],bins=bins[0],weights=weights_MC_sim[pass_fiducial==1]*NNweights_step2[pass_fiducial==1],density=True,histtype="step",color="black",ls=":",label="MC + step 2")

    plt.ylabel("Normalized to unity",fontsize=20)
    plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
    plt.text(0.05, 1.25,'ALEPH', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
    plt.legend(frameon=False,fontsize=15)
    plt.locator_params(axis='x', nbins=5)
    plt.yscale("log")

    ax1 = plt.subplot(gs[1])
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params(direction="in",which="both")
    ax1.minorticks_on()
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.xlabel("Particle-level Thrust",fontsize=15)
    plt.ylabel("step 2/step 1",fontsize=15)

    plt.plot(0.5*(b[0:-1]+b[1:]),n_MC/n_data,ls="--",color="black")
    plt.plot(0.5*(b[0:-1]+b[1:]),n_Omni_step2_pT/n_data,ls=":",color="black")

    fig.savefig("storage_plots/"+sys.argv[1]+"_"+sys.argv[2]+"Iteration"+str(iteration)+"_Step2_thrust.pdf",bbox_inches='tight')
    np.save("storage_files/"+sys.argv[1]+"_"+sys.argv[2]+"n_Omni_step2_thrust_iteration"+str(iteration),n_Omni_step2_pT)
    
    print("done with the "+str(iteration)+"iteration, time elapsed=",time.time()-starttime)
    logfile.write("done with the "+str(iteration)+"iteration, time elapsed="+str(time.time()-starttime)+"\n")
    
    pass

tf.keras.models.save_model(mymodel,"models/"+sys.argv[1]+"_"+sys.argv[2])
logfile.close()
