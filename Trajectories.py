import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import pandas as pd
import os,sys,glob, h5py

from dlab import generalephys as ephys
from dlab.generalephys import placeAxesOnGrid, cleanAxes
from dlab import utils_pipeline as utils
from dlab import rf_analysis
from dlab import psth_and_raster as psth
from dlab import reliability_and_precision as rp
from dlab.continuous_traces import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr, zscore
from itertools import combinations 
from dlab import analysis_pipeline as analysis

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import warnings


def center(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xc = ss.fit_transform(X.T).T
    return Xc

def pca(tens):
    for i in range(min(tens.shape[0], tens.shape[1])-1):
        pca = PCA(n_components=i)
        p=pca.fit_transform(tens.T).T
        ev = (pca.explained_variance_ / sum(pca.explained_variance_))
    return p,ev


def trialave_trajectory_byoutcome(df,df_reaches,start=1,end=1):

    mouseid = df.mouse.unique()

    print('computing pca for mouse ' + str(mouseid))

    dfr_s = df_reaches[df_reaches.behaviors=='success']
    dfr_f = df_reaches[df_reaches.behaviors!='success']

    ave_reach,normedbins,ave_reach_  = analysis.reachave_tensor(df,dfr_s,start = start,end = end) 
    ave_reachf,normedbins_f,ave_reach_f  = analysis.reachave_tensor(df,dfr_f,start = start,end = end)

    trial_concat= np.concatenate((ave_reach,ave_reachf),axis=1)
    centr_traj = center(trial_concat)       
    traj,ev = pca(centr_traj)


##plot trajectory
    #fig, (ax1,ax2) = plt.subplots(2,constrained_layout=True)
    tm = int(np.shape(traj)[1]/2)

    ax = plt.axes(projection='3d')

    ax.plot3D(traj[0][tm:],traj[1][tm:],traj[2][tm:],'r',label='failure trials')
    ax.plot3D(traj[0][0],traj[1][0],traj[2][0],'p',label='start')
    ax.plot3D(traj[0][:tm],traj[1][:tm],traj[2][:tm],'b',label='success trials')
    ax.plot3D(traj[0][tm],traj[1][tm],traj[2][tm],'p',label='end')
    ax.legend()
    ax.set_title('neural trajectories: success and failure')

    #ax2.plot(ev[0:20])
    #ax2.set_title('explained variance')


