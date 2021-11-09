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
from dlab import analysis_pipeline as analysis
from dlab.continuous_traces import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr, zscore
from itertools import combinations 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import warnings



def trial_concat_pca(df,df_reaches):

#    mice = df.mouse.unique()
#    df_pcs = pd.DataFrame(data=mice,columns=['mouse'])

    ave_reach_s = []
    ave_reach_f = []
    normedbinssf = []
    trajs = []
    evs = []

    for mouseid in df.mouse.unique():

        dfr_ = df_reaches[df_reaches.mouse==mouseid]
        df_ = df[df.mouse==mouseid]
        
        dfr_s = dfr_[dfr_.behaviors=='success']
        dfr_f = dfr_[dfr_.behaviors!='success']

        ave_reach  = analysis.reachave_tensor(df_,dfr_s,start = 2,end = 4) 
        ave_reachf  = analysis.reachave_tensor(df_,dfr_f,start = 2,end = 4) 

        ave_reach_s.append(ave_reach)
        ave_reach_f.append(ave_reachf)

    trial_concat= np.concatenate((ave_reach_s,ave_reach_f),axis=1)
    centr_traj = center(trial_concat)       
    traj,ev = pca(centr_traj) # for all mice
    return traj, ev





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



