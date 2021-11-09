import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import pandas as pd
import os,sys,glob, h5py
from scipy.stats import sem 
from scipy import stats

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

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import warnings

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import ShuffleSplit


#TRIAL CONCAT TENSOR
def create_SVM_values(df,df_reaches,start,end,binsize):

    edges=np.arange(start,end,binsize)
    num_bins=edges.shape[0]-1
    reach = []

    byreach=np.zeros((np.shape(df_reaches)[0],num_bins))
    allunits = []
    trajs = []
    evs = []
    outcomes = []
    units = []
    edg = []
    means = []
    tensor = []
    each_unit = []
    behavior = []
    reach = []
    roll_sum = []
    each_unit_rolled = []

    for i,times in enumerate(df.times): #compare that unit's spike times to each reach max
        t = np.array(times) #for reach unit create an array of that unit's spike times
        for j,tmax in enumerate(df_reaches.rMax_t): #for each unit 
            a = tmax+start
            b = tmax+end
            byreach = np.zeros((np.shape(df_reaches)[0],num_bins))
    #        try:
            unit = df.index[i]
            units.append(unit)
            rea = df_reaches.behaviors[j]
            if rea == 'success':
                reach.append(1)
            else:
                reach.append(0)
            rd = np.array(t[(t >= a) & (t <= b)]) #find if that unit spiked within designated timeframes around reachmax
            edges=np.arange(a,b,binsize) #designated bins around this iteration of reachmax
            hist=np.histogram(rd,bins=edges)[0] #bin spikes into timeframe
            #s = pd.Series(hist)
            #window = int(1/binsize)
            #hist_rollsum = s.rolling(window).sum()
            #hist_rollsum = hist_rollsum.dropna()
            #roll_sum.append(hist_rollsum)
            byreach[j,:] = hist
        each_unit.append(byreach)
    #        byreach_rolled[j,:] = hist_rollsum
    #        except:
    #            print(str(j) + ' missed ')
            #tensor.append(hists)
        #each_unit_rolled.append(byreach_rolled)
        #roll_sum = []
        #hists = []

        behavior.append(reach)
        reach = []
    return each_unit,behavior,units

def get_SVM_scores(each_unit,behavior,param_grid):
    scores = []

    for i,unit in enumerate(each_unit):
        X = unit
        y = behavior[i]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        
        X_train_scaled = scale(X_train)
        X_test_scaled = scale(X_test)
        
        optimal_params = GridSearchCV(SVC(),param_grid,cv=5,scoring='accuracy')
        optimal_params.fit(X_train_scaled,y_train)
        print(optimal_params.best_params_)
        
        c = optimal_params.best_params_['C']
        gam = optimal_params.best_params_['gamma']
        
        clf_svm = SVC(C=c, gamma=gam)
        clf_svm.fit(X_train_scaled,y_train)
        score = clf_svm.score(X_test_scaled,y_test)
        scores.append(score)
    return scores,y_train,y_test

def get_SVM_scores_crossval(each_unit,behavior,param_grid):
    scores = []

    for i,unit in enumerate(each_unit):
        X = unit
        y = behavior[i]
        clf = svm.SVC(kernel='linear', C=0.5)
        cv = ShuffleSplit(n_splits=5, test_size=0.2)
        scores_ = cross_val_score(clf, X, y, cv=cv) 
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores_.mean(), scores_.std()))
        score = np.mean(scores_)
        scores.append(score)
    return scores

def get_indicative_neurons(df,df_reaches,scores):
    df['SVM_score'] = scores
    success = np.shape(df_reaches[df_reaches.behaviors=='success'])[0]
    failure = np.shape(df_reaches[df_reaches.behaviors!='success'])[0]

    Pprior_f = failure/(failure+success)
    Pprior_s = success/(failure+success)
    prior = np.max([Pprior_f,Pprior_s])

    indicative_units = []
    not_ind_units = []
    for i,score in enumerate(scores):
        if score> prior:
            indicative_units.append(i)
        else:
            not_ind_units.append(i)
    
    df_ind = df.iloc[indicative_units, :]
    df_ind['indicative'] = 'yes'
    df_noind = df.iloc[not_ind_units, :]
    df_noind['indicative'] = 'no'
    df = pd.concat([df_ind,df_noind])
    return df,indicative_units,not_ind_units,prior

def center(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xc = ss.fit_transform(X.T).T
    return Xc

def get_trials_byunit(df,df_reaches,start,end,binsize):
    edges=np.arange(start,end,binsize)
    num_bins=edges.shape[0]-1 #number of bins 
    byreach=np.zeros((np.shape(df_reaches)[0],num_bins))
    rastor = []
    units = []
    psths = []
    psths_std = []
    rastor_norm = []
    for i,times in enumerate(df.times): #compare that unit's spike times to each reach max
        t = np.array(times) #for reach unit create an array of that unit's spike times
        byreach=np.zeros((np.shape(df_reaches)[0],num_bins))
        byreach_norm=np.zeros((np.shape(df_reaches)[0],num_bins))
        for j,tmax in enumerate(df_reaches.rMax_t): #for each unit 
            a = tmax+start
            b = tmax+end
            unit = df.index[i]
            units.append(unit)
            #rea = dfr_s.behaviors[j]
            #reach.append(rea)
            #try:
            rd = np.array(t[(t >= a) & (t <= b)]) #find if that unit spiked within designated timeframes around reachmax
            edges=np.arange(a,b,binsize) #designated bins around this iteration of reachmax
            num_bins=edges.shape[0]-1 #number of bins 
            hist=np.histogram(rd,edges)[0] #bin spikes into timeframe
            normbins = (hist-min(hist))/max(hist) #per dailey
            #s = pd.Series(hist)
            #window = int(1/binsize)
            #hist_rollsum = s.rolling(window).sum()
            #hist_rollsum = hist_rollsum.dropna()
            byreach[j,:] = hist
            byreach_norm[j,:] = normbins
            #byreach = center(byreach)
            #except:
                #pass
        rastor.append(byreach) # tensor rastor - every trial response for every neuron
        rastor_norm.append(byreach_norm)
        psth = np.mean(byreach,axis=0)
        psth_std = sem(byreach,axis=0) # 
        psths.append(psth)
        psths_std.append(psth_std)
    return rastor,rastor_norm,psths,units,psths_std



def plot_outcome_units(df,df_reaches,indicative_units,save_path,start,end,binsize,save=False,norm=False):

    dfr_s = df_reaches[df_reaches.behaviors=='success']
    dfr_f = df_reaches[df_reaches.behaviors!='success']

    rastor_s,rastor_norm_s,psths_s,units_s,psths_std_s = get_trials_byunit(df,dfr_s,start=start,end=end,binsize=binsize)
    rastor_f,rastor_norm_f,psths_f,units_f,psths_std_f = get_trials_byunit(df,dfr_f,start=start,end=end,binsize=binsize)

    #smoothed indicative neurons
    edges=np.arange(start,end,binsize)
    edges = edges[1:]
    for i,unit in enumerate(df.index):
        fig, (ax1,ax2,ax3) = plt.subplots(3,figsize=(10,8),constrained_layout=True)
        #fig = plt.figure(figsize=(20, 5))
        s = gaussian_filter1d(psths_s[i], sigma=1)
        f = gaussian_filter1d(psths_f[i], sigma=1)
        es = psths_std_s[i]
        ef = psths_std_f[i]
        neuron = unit
        label = df.group[unit]
        #plt.plot(s,'b')
        #plt.plot(f,'r')
        #score = df.SVM_score[unit]
        ax1.errorbar(edges,s,es,ecolor='lightcoral',label='success')
        ax1.errorbar(edges,f,ef,ecolor='lightblue',label='failure')
        ax1.legend()
        ax1.axvline(0,c='black',ls='-')
        #ax1.set_title('psth unit: ' + str(unit) + str(label) + ' // SVMscore: ' +str(score))
        ax1.set_title('psth unit: ' + str(unit) + str(label) )

        if norm==True:
            #bins_s = rastor_s[i]
            #bins_f = stats.zscore(rastor_f[i],axis=1)
            print('normalization not working yet')
        else:
            bins_s = rastor_s[i]
            bins_f = rastor_f[i]

        max1 = np.max(bins_s)
        max2 = np.max(bins_f)
        vmax = np.max([max1,max2])
        g1 = sns.heatmap(bins_s,vmax=vmax,cmap='RdYlBu_r',xticklabels=False,ax=ax2)
        g1.axvline(150,c='black',ls='-')
        g1.set_ylabel('success trials')
        #ymax = np.max(rastor_s[unit].extend(rastor_f[unit]))
        #ax1.set_title('success ' + str(indicative_units[unit]))
        g2 = sns.heatmap(bins_f,vmax=vmax,cmap='RdYlBu_r',xticklabels=False,ax=ax3)
        g2.axvline(150,c='black',ls='-')
        g2.set_ylabel('failure trials')
        #ax2 = sns.heatmap(bins,cmap='RdYlBu_r',cbar=False)
        #ax3 = sns.heatmap(rastor_f[unit],cmap='RdYlBu_r',cbar=axcb)
        #ax2.set_title('failure ' + str(indicative_units[unit]))
        if save==True:
            fig.savefig(save_path + str(neuron))
        plt.show()
        plt.close()