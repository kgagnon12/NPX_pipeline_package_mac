import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import pandas as pd
import os,sys,glob, h5py, csv, time
import matplotlib.pyplot as plt
from dlab import utils_pipeline as utils
from dlab import depth_estimation as depth
import pandas as pd
import numpy as np
import glob
from dlab import generalephys as ephys
from dlab.generalephys import get_waveform_duration,get_waveform_PTratio,get_waveform_repolarizationslope,option234_positions
from scipy.cluster.vq import kmeans2
from dlab import sorting_quality_editing as sq
import seaborn as sns;sns.set_style("ticks")
import matplotlib.pyplot as plt
import h5py
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import os,time


####################### LOAD KILOSORT OUTPUTS ###############################################

def df_from_phy_multimouse(folder,expnum='1',recnum='1',**kwargs):
    # list all animals within folder, iterate through each animal and get cluster information
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]
        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                pass
    #        else:
    #            rec_folder = folder
    #            print(rec_folder)
        raw_path = os.path.join(rec_folder,'experiment1','recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            print('loading from '+raw_path)
        else:
            print('could not find data folder for '+raw_path)
    if 'cohort' in kwargs.keys():
        cohort = kwargs['cohort']
    else:
        cohort = None 
    if 'mouse' in kwargs.keys():
        mouse = kwargs['mouse']
    else:
        mouse = None              
    
    path = raw_path
    if os.path.isfile(os.path.join(raw_path,'spike_clusters.npy')) :                  
        units = ephys.load_phy_template(path)
        mouse = [];experiment=[];cell = [];ypos = [];xpos = [];waveform=[];template=[];structure=[];times=[]
        index = []; count = 1; cohort = []
        probe_id=[]
        if os.path.isfile(os.path.join(raw_path,'cluster_group.tsv')) : 
            for unit in units.index:
                if 'probe' in kwargs.keys():
                    probe_id.extend([kwargs['probe']])
                else:
                    probe_id.extend(['A'])
                if 'mouse' in kwargs.keys():
                    mouse.extend([kwargs['mouse']])
                else:
                    mouse.extend([int(mouse_)])
                if 'experiment' in kwargs.keys():
                    experiment.extend([kwargs['experiment']])
                else:
                    experiment.extend(['placeholder'])
                if 'cohort' in kwargs.keys():
                    cohort.extend([kwargs['cohort']])
                else:
                    cohort.extend([cohort_])

            df = units
            df['mouse'] = mouse
            df['experiment'] = experiment
            df['probe'] = probe_id
            #     df['structure'] = structure
            df['cell'] = units.index
            df['cohort'] = cohort
            df['times'] = units['times']
            df['ypos'] = units['ypos']
            df['xpos'] = units['xpos']
            #         df['depth'] = xpos
            df['waveform'] = units['waveform_weights']
            df['template'] = units['template']
        
            return df
    
        else:
            pass




# to load a single recording file from one recording, one mouse
def df_from_phy(folder,expnum='1',recnum='1',**kwargs):
    # if 'est' not in folder:
 #       base_folder = os.path.basename(folder)
 #       cohort_ = os.path.basename(base_folder).split('_')[-2]
 #       mouse_  = os.path.basename(base_folder).split('_')[-1]

        #traverse down tree to data
 #       if 'open-ephys-neuropix' in base_folder:
 #           try:
 #               rec_folder = glob.glob(folder+'/*')[0]
 #           except:
 #               print(base_folder)
 #               return None
 #       else:
 #           rec_folder = folder
 #       raw_path = os.path.join(rec_folder,'experiment'+str(expnum),'recording'+str(recnum),'continuous')
 #       if len(glob.glob(raw_path+'/*100.0*'))>0:
 #           raw_path = glob.glob(raw_path+'/*100.0*')[0]
 #           print('loading from '+raw_path)
 #       else:
            
 #           print('could not find data folder for '+raw_path)
    raw_path=folder
 
    if 'cohort' in kwargs.keys():
        cohort = kwargs['cohort']
    else:
        cohort = None 
    if 'mouse' in kwargs.keys():
        mouse = kwargs['mouse']
    else:
        mouse = None              
    # df = df_from_phy(raw_path,site_positions = ephys.option234_positions,cluster_file='KS2',cohort=cohort,mouse=mouse)
    
    path = raw_path
    #units = ephys.load_phy_template(path,cluster_file='KS2',site_positions=site_positions)
    units = ephys.load_phy_template(path,site_positions=site_positions)
    #structures is a dictionary that defines the bounds of the structure e.g.:{'v1':(0,850), 'hpc':(850,2000)}
    mouse = [];experiment=[];cell = [];ypos = [];xpos = [];waveform=[];template=[];structure=[];times=[]
    index = []; count = 1; cohort = []
    probe_id=[]
    depth=[];#print(list(nwb_data.keys()));print(list(nwb_data['processing'].keys()));

    for unit in units.index:
        if 'probe' in kwargs.keys():
            probe_id.extend([kwargs['probe']])
        else:
            probe_id.extend(['A'])
        if 'mouse' in kwargs.keys():
            mouse.extend([kwargs['mouse']])
        else:
            mouse.extend([mouse_])
        if 'experiment' in kwargs.keys():
            experiment.extend([kwargs['experiment']])
        else:
            experiment.extend(['placeholder'])
        if 'cohort' in kwargs.keys():
            cohort.extend([kwargs['cohort']])
        else:
            cohort.extend([cohort_])

    df = units
    df['mouse'] = mouse
    df['experiment'] = experiment
    df['probe'] = probe_id
    #     df['structure'] = structure
    df['cell'] = units.index
    df['cohort'] = cohort
    df['times'] = units['times']
    df['ypos'] = units['ypos']
    df['xpos'] = units['xpos']
    #         df['depth'] = xpos
    df['waveform'] = units['waveform_weights']
    df['template'] = units['template']
    return df









###################### get info for units in df ###############################################


def get_unit_info(df):
# get waveform
    wave_ = []
    for i,template in df.template.items():
        wave_.append(utils.get_peak_waveform_from_template(np.array(template)))
    df.waveform=wave_

# get firing rate
    f_ = []
    for i,times in df.times.items():
        try:
            rate = float(len(times)/(times[-1] - times[0]))
            if rate < 400:
                f_.append(rate)
            else:
                f_.append(0.)
        except:
            f_.append(0.)
    df['overall_rate']=f_

    return df









###################### batch depth estimation #################################################

def depth_estimation_multimouse(folder,expnum='1',recnum='1',**kwargs):
    # list all animals within folder, iterate through each animal and get cluster information
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]
        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                pass
    #        else:
    #            rec_folder = folder
    #            print(rec_folder)
        raw_path = os.path.join(rec_folder,'experiment1','recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            #print('loading from '+raw_path)
        else:
            print('could not find data folder for '+raw_path)
    if 'cohort' in kwargs.keys():
        cohort = kwargs['cohort']
    else:
        cohort = None 
    if 'mouse' in kwargs.keys():
        mouse = kwargs['mouse']
    else:
        mouse = None             
    
    path = raw_path
    if os.path.isfile(os.path.join(raw_path,'spike_clusters.npy')) :
        ephys_params = depth.createInputJson(output_file=os.path.join(raw_path,'depth'), 
                    npx_directory=raw_path, 
                    continuous_file = os.path.join(raw_path,'continuous.dat'),
                    extracted_data_directory=raw_path,
                    kilosort_output_directory=raw_path, 
                    kilosort_output_tmp=raw_path)
        ephys_params = ephys_params['ephys_params']
        fig_path = os.path.join(raw_path,'depth.png')
        surface_chan = depth.get_surface_channel(ephys_params,fig_path)
        print('mouse: ' +str(mouse_),' surface channel: '+str(surface_chan))
        #df = pd.DataFrame(columns = ['mouse','surface_channel'])
        #df['mouse'] = mouse_
        #df['surface_channel'] = surface_chan
    else:
        surface_chan = 'not found - check file path'

    return mouse_,surface_chan




def plot_results(chunk, 
                 power, 
                 in_range, 
                 values, 
                 nchannels, 
                 surface_chan, 
                 power_thresh, 
                 diff_thresh, 
                 figure_location):

    plt.figure(figsize=(5,10))
    plt.subplot(4,1,1)
    # plt.imshow(np.flipud((chunk).T), aspect='auto',vmin=-1000,vmax=1000)
    plt.imshow((chunk).T, aspect='auto',vmin=-1000,vmax=1000)
    plt.title('raw data (the chunk)')
    plt.xlabel('time (samples)')
    plt.ylabel('channels')
    

    plt.subplot(4,1,2)
    # plt.imshow(np.flipud(np.log10(power[in_range,:]).T), aspect='auto')
    plt.imshow(np.log10(power[in_range,:]).T, aspect='auto')
    plt.title('gamma power')
    plt.xlabel('time (samples)')
    plt.ylabel('channels')

    plt.subplot(4,1,3)
    plt.plot(values) 
    plt.plot([0,nchannels],[power_thresh,power_thresh],'--k') #kg input -0.1 for y axis
    plt.plot([surface_chan, surface_chan],[0.5, 2],'--r')
    plt.xlabel('channels')
    plt.ylabel('gamma power')
    
    plt.subplot(4,1,4)
    plt.plot(np.diff(values))
    plt.plot([0,nchannels],[diff_thresh,diff_thresh],'--k') #kg input -0.1 for y axis
    plt.plot([surface_chan, surface_chan],[diff_thresh, diff_thresh],'--r')
    plt.title(surface_chan)
    plt.xlabel('channels')
    plt.ylabel('power difference between channels')

    plt.show()
    plt.savefig(figure_location)
    plt.close()





def adjust_depth(df,df_surfchan,layer_thresh,positions,mouseid):
    surf_chan = df_surfchan
    if surf_chan == 'not found - check file path':
        print('no surface channel found for mouse ' + str(mouseid) + '-check file path')
        pass
    else:
        ypos = df.ypos
        surf_chan = int(df_surfchan)
        print('mouse ' + str(mouseid ) + ' surface channel: ' +str(surf_chan))
        surf_depth = positions[surf_chan][1]
        print('y-position of surface channel: '+str(surf_depth))
        depth = abs(surf_depth-ypos)
        df['depth'] = depth

        #assign layer to each unit
        #layer_split = surf_depth-layer_thresh
        df_L23 = df[df.depth<layer_thresh]
        df_L5 = df[df.depth>layer_thresh]
        df_L23['layer'] = 'L2-3'       
        df_L5['layer'] = 'L5'
        df = pd.concat([df_L23,df_L5])
    return df










########################## SORTING QUALITY #####################################################

# ===========================================================================
    ### utility code for batch sorting quality ###


def sorting_quality_multimouse(folder,expnum='1',recnum='1',channels=383,**kwargs):
    # list all animals within folder, iterate through each animal and get cluster information
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]
        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                pass
    #        else:
    #            rec_folder = folder
    #            print(rec_folder)
        raw_path = os.path.join(rec_folder,'experiment1','recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            print('loading from '+raw_path)
        else:
            print('could not find data folder for '+raw_path)
    if 'cohort' in kwargs.keys():
        cohort = kwargs['cohort']
    else:
        cohort = None 
    if 'mouse' in kwargs.keys():
        mouse = kwargs['mouse']
    else:
        mouse = None              

    directory = raw_path
    if os.path.isfile(os.path.join(raw_path,'cluster_group.tsv')) :
        channels2 = 'all'#[0,383]
        print(channels2)
        time_limits = None#[500.,600.]

        t0 = time.time()
        #quality = sq.masked_cluster_quality(directory,time_limits)
        #print('PCA quality took '+str(time.time()-t0)+' sec');t0 = time.time()
        isiV = sq.isiViolations(directory,time_limits)
        print('ISI quality took '+str(time.time()-t0)+' sec');t0 = time.time(); 
        #SN = sq.cluster_signalToNoise(directory,filename,time_limits, channels = 383)
        #print('SN quality took '+str(time.time()-t0)+' sec');t0 = time.time()

        cluster_groups = sq.read_cluster_groups_CSV(directory)  
        print(cluster_groups[2])

        print(isiV[0])

        cluster_group = []
        color = []
        for clu_id in isiV[0]:
            if clu_id in cluster_groups[0]:
                cluster_group.append('good')
                color.append(sns.color_palette()[1])
            else:
                if clu_id in cluster_groups[1]:
                    cluster_group.append('mua')
                    color.append(sns.color_palette()[0])
                else:
                    if clu_id in cluster_groups[2]:
                        cluster_group.append('unsorted')
                        color.append(sns.color_palette()[1])
                    else:
                        cluster_group.append('noise')
                        color.append(sns.color_palette()[1])
        mouse = []; cohort = []; trash = []; 
        for unit in cluster_group:
            if 'mouse' in kwargs.keys():
                mouse.extend([kwargs['mouse']])
            else:
                mouse.extend([int(mouse_)])
            if 'cohort' in kwargs.keys():
                cohort.extend([kwargs['cohort']])
            else:
                cohort.extend([cohort_])

        df = pd.DataFrame({
            'clusterID':isiV[0],
            'isi_violations':np.ones(len(isiV[1])) - isiV[1],
#            'sn_max':SN[1],
#            'sn_mean':SN[2],
#            'isolation_distance':quality[1],
#            'mahalanobis_contamination':np.ones(len(quality[2]))-quality[2],
#            'FLDA_dprime':quality[3]*-1,
            'cluster_group':cluster_group,
            'color':color})
        df['mouse'] = mouse
        df['cohort'] = cohort
        return df

    else:
        pass



def assign_sq_rank(df_sq):
    qual_rank = []

    for i in df_sq.linear_quality:
        if i>1.5: # i > 1.5
            qual_rank.append(1) 
        if i>1 and i<1.5: # 1 < i < 1.5
            qual_rank.append(2)
        if i > 0.5 and i < 1:
            qual_rank.append(3)
        if i > 0 and i < 0.5:
            qual_rank.append(4)
        if i < 0 and i > -0.5:
            qual_rank.append(5)
        if i < -0.5 and i > -1:
            qual_rank.append(6)
        if i <-1 and i > -1.5:
            qual_rank.append(7)
        if i <-1.5:
            qual_rank.append(8)
    df_sq['quality_rank'] = qual_rank
    return df_sq




def save_individual_mouse_pairplot(folder,df,df_sq,expnum='1',recnum='1',channels=383,**kwargs):
    # list all animals within folder, iterate through each animal and get cluster information
    if 'est' not in folder:
        base_folder = os.path.basename(folder)
        cohort_ = os.path.basename(base_folder).split('_')[-2]
        mouse_  = os.path.basename(base_folder).split('_')[-1]
        #traverse down tree to data
        if 'open-ephys-neuropix' in base_folder:
            try:
                rec_folder = glob.glob(folder+'/*')[0]
            except:
                pass
    #        else:
    #            rec_folder = folder
    #            print(rec_folder)
        raw_path = os.path.join(rec_folder,'experiment1','recording'+str(recnum),'continuous')
        if len(glob.glob(raw_path+'/*100.0*'))>0:
            raw_path = glob.glob(raw_path+'/*100.0*')[0]
            print('saving figure in '+raw_path)
        else:
            print('could not find data folder for '+raw_path)

    directory = raw_path
    df_sq = df_sq[df_sq.mouse==int(mouse_)]
    df_ = df[df.mouse==int(mouse_)]
    if os.path.isfile(os.path.join(raw_path,'cluster_group.tsv')) :
        fig = sns.pairplot(df_sq,
            diag_kind='kde',markers='o',hue='cluster_group')
        plt.title('sorting quality metric distributions for mouse' +str(mouse_))
        plt.show()
        fig.savefig(os.path.join(raw_path,'sorting_quality_metrics_distribution.png'))
        plt.close()

        fig = sns.jointplot(x = df_.overall_rate,y = df_.depth, hue = df_.cluster_group
           )
        plt.title('firing rate vs depth')
        plt.ylabel('depth (0 = deep)')
        plt.xlabel('firing rate')
        plt.show()
        plt.close()

        sns.catplot(data=df_, kind="swarm", x="waveform_class", y="overall_rate", hue="cluster_group")

        fig = sns.jointplot(x = df_.linear_quality,y = df_.depth, hue = df_.cluster_group)










#################### LOAD BEHAVIOR FILES ####################################################

def load_curated_behavior(folder):
    if folder[-4:-1] == '.cs':
        base_folder = os.path.basename(folder)
        base_folder = os.path.basename(folder[:-4])

        if folder[-6] == 'S':
            cohort_ = os.path.basename(base_folder).split('_')[-3]
            mouse_  = os.path.basename(base_folder).split('_')[-2]
        else:
            cohort_ = os.path.basename(base_folder).split('_')[-2]
            mouse_  = os.path.basename(base_folder).split('_')[-1]
        df = pd.read_csv(folder)
        df['mouse'] = int(mouse_)
        df['cohort'] = cohort_
        print('loaded ' + str(folder))
        return df



def load_timestamps(folder):
    base_folder = os.path.basename(folder)
    mouse_  = os.path.basename(base_folder).split('_')[-2]
    print(mouse_)
    ts = np.load(folder)
    print('loaded' + str(folder))
    return mouse_,ts

