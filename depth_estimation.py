import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
import pandas as pd
import os,sys,glob, h5py, csv, time
import matplotlib.pyplot as plt
from dlab import preprocessing_pipeline as preprocessing
from dlab import utils_pipeline as utils
import io
import json

from scipy.signal import welch
from scipy.ndimage.filters import gaussian_filter1d

from .common.utils import find_range, rms, printProgressBar
from .common.OEFileInfo import get_lfp_channel_order




###### depth adjustment ######


# creates parameters for each recording
def createInputJson(output_file, 
                    npx_directory=None, 
                    continuous_file = None,
                    extracted_data_directory=None,
                    kilosort_output_directory=None, 
                    kilosort_output_tmp=None, 
                    probe_type='3A'):

    if kilosort_output_directory is None \
         and extracted_data_directory is None \
         and npx_directory is None:
        raise Exception('Must specify at least one output directory')

    if probe_type == '3A':
        acq_system = '3a'
        reference_channels = [36, 75, 112, 151, 188, 227, 264, 303, 340, 379]
    else:
        acq_system = 'PXI'
        reference_channels = [191]

    if npx_directory is not None:
        settings_xml = os.path.join(npx_directory, 'settings.xml')
        if extracted_data_directory is None:
            extracted_data_directory = npx_directory + '_sorted'
        probe_json = os.path.join(extracted_data_directory, 'probe_info.json')
        settings_json = os.path.join(extracted_data_directory, 'open-ephys.json')
    else:
        if extracted_data_directory is not None:
            probe_json = os.path.join(extracted_data_directory, 'probe_info.json')
            settings_json = os.path.join(extracted_data_directory, 'open-ephys.json')
            settings_xml = None
        else:
            settings_xml = None
            settings_json = None
            probe_json = None
            extracted_data_directory = kilosort_output_directory

    if kilosort_output_tmp is None:
        kilosort_output_tmp = r"C:\data\kilosort" #kilosort_output_directory

    if continuous_file is None:
        continuous_file = os.path.join(kilosort_output_directory, 'continuous.dat')

    dictionary = \
    {

        "directories": {
            "extracted_data_directory": extracted_data_directory,
            "kilosort_output_directory": kilosort_output_directory,
            "kilosort_output_tmp": kilosort_output_tmp
        },

        "common_files": {
            "settings_json" : settings_json,
            "probe_json" : probe_json,
        },

        "waveform_metrics" : {
            "waveform_metrics_file" : os.path.join(kilosort_output_directory, 'waveform_metrics.csv')
        },

        "ephys_params": {
            "sample_rate" : 30000,
            "lfp_sample_rate" : 2500,
            "bit_volts" : 0.195,
            "num_channels" : 384,
            "reference_channels" : reference_channels,
            "vertical_site_spacing" : 10e-6,
            "ap_band_file" : continuous_file,
            "lfp_band_file" : continuous_file,
            "reorder_lfp_channels" : probe_type == '3A',
            "cluster_group_file_name" : 'cluster_group.tsv.v2'
        }, 

        "extract_from_npx_params" : {
            "npx_directory": npx_directory,
            "settings_xml": settings_xml,
            "npx_extractor_executable": r"C:\Users\svc_neuropix\Documents\GitHub\npxextractor\Release\NpxExtractor.exe",
            "npx_extractor_repo": r"C:\Users\svc_neuropix\Documents\GitHub\npxextractor"
        },

        "depth_estimation_params" : {
            "hi_noise_thresh" : 50.0,
            "lo_noise_thresh" : 3.0,
            "save_figure" : 1,
            "figure_location" : os.path.join(extracted_data_directory, 'probe_depth.png'),
            "smoothing_amount" : 5,
            "power_thresh" : 2.5,
            "diff_thresh" : -0.06,
            "freq_range" : [0, 10],
            "max_freq" : 150,
            "channel_range" : [374, 384],
            "n_passes" : 10,
            "air_gap" : 25,
            "time_interval" : 5,
            "skip_s_per_pass" : 10,
            "start_time" : 10
        }, 

        "median_subtraction_params" : {
            "median_subtraction_executable": "C:\\Users\\svc_neuropix\\Documents\\GitHub\\spikebandmediansubtraction\\Builds\\VisualStudio2013\\Release\\SpikeBandMedianSubtraction.exe",
            "median_subtraction_repo": "C:\\Users\\svc_neuropix\\Documents\\GitHub\\spikebandmediansubtraction\\",
        },

        "kilosort_helper_params" : {

            "matlab_home_directory": "C:\\Users\\svc_neuropix\\Documents\\MATLAB",
            "kilosort_repository": "C:\\Users\\svc_neuropix\\Documents\\GitHub\\kilosort2",
            "kilosort_version" : 2,
            "surface_channel_buffer" : 15,

            "kilosort2_params" :
            {
                "chanMap" : "'chanMap.mat'",
                "fshigh" : 150,
                "minfr_goodchannels" : 0.1,
                "Th" : '[10 4]',
                "lam" : 10,
                "AUCsplit" : 0.9,
                "minFR" : 1/50.,
                "momentum" : '[20 400]',
                "sigmaMask" : 30,
                "ThPre" : 8
            },

            "kilosort3_params" :
            {
                "chanMap" : "'chanMap.mat'",
                "fshigh" : 300,
                "minfr_goodchannels" : 0.1,
                "Th" : '[9 9]',
                "lam" : 10,
                "AUCsplit" : 0.8,
                "minFR" : 1/50.,
                "momentum" : '[20 400]',
                "sigmaMask" : 30,
                "ThPre" : 8,
                "sig" : 20,
                "nblocks" : 5
            }
        },

        "ks_postprocessing_params" : {
            "within_unit_overlap_window" : 0.000166,
            "between_unit_overlap_window" : 0.000166,
            "between_unit_overlap_distance" : 5
        },

        "mean_waveform_params" : {
        
            "mean_waveforms_file" : os.path.join(kilosort_output_directory, 'mean_waveforms.npy'),

            "samples_per_spike" : 82,
            "pre_samples" : 20,
            "num_epochs" : 1,
            "spikes_per_epoch" : 1000,
            "spread_threshold" : 0.12,
            "site_range" : 16
        },

        "noise_waveform_params" : {
            "classifier_path" : os.path.join(os.getcwd(), 'ecephys_spike_sorting', 'modules', 'noise_templates', 'rf_classifier.pkl'),
            "multiprocessing_worker_count" : 10
        },

        "quality_metrics_params" : {
            "isi_threshold" : 0.0015,
            "min_isi" : 0.000166,
            "num_channels_to_compare" : 7,
            "max_spikes_for_unit" : 500,
            "max_spikes_for_nn" : 10000,
            "n_neighbors" : 4,
            'n_silhouette' : 10000,
            "quality_metrics_output_file" : os.path.join(kilosort_output_directory, "metrics_test.csv"),
            "drift_metrics_interval_s" : 51,
            "drift_metrics_min_spikes_per_interval" : 10,

            "include_pc_metrics" : True
        }

    }

    with io.open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dictionary, ensure_ascii=False, sort_keys=True, indent=4))

    return dictionary




def get_surface_channel(ephys_params,raw_path):

    """
    Computes surface channel from LFP band data

    Inputs:
    ------
    ephys_params : dict from create_json function

    Outputs:
    -------
    output_dict : dict
        - surface_channel : channel at brain surface
        - air_channel : channel at agar / air surface (approximate)

    """

    start = time.time()

    numChannels = ephys_params['num_channels']

    rawDataAp = np.memmap(ephys_params['ap_band_file'], dtype='int16', mode='r')
    ap_data = np.reshape(rawDataAp, (int(rawDataAp.size/numChannels), numChannels))

    rawDataLfp = np.memmap(ephys_params['lfp_band_file'], dtype='int16', mode='r')
    lfp_data = np.reshape(rawDataLfp, (int(rawDataLfp.size/numChannels), numChannels))

    print('Computing surface channel...')

    #info_lfp = find_surface_channel(dataLfp, 
                                #ephys_params)
        # get surface channel

    nfft = 4096
    max_freq = 150
    freq_range = [0, 10]
    smoothing_amount = 5
    diff_thresh = -0.04
    power_thresh = 1.5
    air_gap = 100


    nchannels = ephys_params['num_channels']
    sample_frequency = ephys_params['lfp_sample_rate']

    #save_figure = params['save_figure']

    candidates = np.zeros((10,))

    for p in range(10):

        startPt = int(sample_frequency*5*p)
        endPt = startPt + int(sample_frequency)

        if ephys_params['reorder_lfp_channels']:
            channels = get_lfp_channel_order()
        else:
            channels = np.arange(nchannels).astype('int')

        chunk = np.copy(lfp_data[startPt:endPt,channels])

        for ch in np.arange(nchannels):
            chunk[:,ch] = chunk[:,ch] - np.median(chunk[:,ch])

        for ch in np.arange(nchannels):
            chunk[:,ch] = chunk[:,ch] - np.median(chunk[:,[370, 380][0]:[370, 380][1]],1)

        power = np.zeros((int(nfft/2+1), nchannels))

        for ch in np.arange(nchannels):

            #printProgressBar(p * nchannels + ch + 1, nchannels * n_passes)

            sample_frequencies, Pxx_den = welch(chunk[:,ch], fs=sample_frequency, nfft=nfft)
            power[:,ch] = Pxx_den

        in_range = find_range(sample_frequencies, 0, max_freq)

        mask_chans = ephys_params['reference_channels']

        in_range_gamma = find_range(sample_frequencies, freq_range[0],freq_range[1])
        
        mask_chans = np.array(mask_chans)
        values = np.log10(np.mean(power[in_range_gamma,:],0))
        values[mask_chans] = values[mask_chans-1]
        values = gaussian_filter1d(values,smoothing_amount)
        power_thresh = np.mean(values[379:384],0) # power thresh set by KG -- we assume these channels are outside of the brain for dailey-specific recordings
        surface_channels = np.where(1.5* (values[:-1] < power_thresh) )[0]

        if len(surface_channels > 0):
            candidates[p] = np.min(surface_channels)
        else:
            candidates[p] = nchannels

    surface_channel = np.median(candidates)
    #air_channel = np.min([surface_channel + air_gap, nchannels])

    #output_dict = {
    #    'surface_channel' : surface_channel,
    #    'air_channel' : air_channel
    #}

    preprocessing.plot_results(chunk, 
                power, 
                in_range, 
                values, 
                nchannels, 
                surface_channel, 
                power_thresh, 
                diff_thresh, 
                figure_location = raw_path)

    return surface_channel