## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
from scipy.signal import spectrogram
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
import re
from tqdm import tqdm, trange
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[...,time_vec<stim_time],axis=-1)[...,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    #mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    #array -= mean_baseline
    return array

# _____      _                  _     ____        _        
#| ____|_  _| |_ _ __ __ _  ___| |_  |  _ \  __ _| |_ __ _ 
#|  _| \ \/ / __| '__/ _` |/ __| __| | | | |/ _` | __/ _` |
#| |___ >  <| |_| | | (_| | (__| |_  | |_| | (_| | || (_| |
#|_____/_/\_\\__|_|  \__,_|\___|\__| |____/ \__,_|\__\__,_|
#                                                          

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/fastdata/lfp_analyses'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)
log_file_name = os.path.join(data_folder, 'file_list.txt')

if os.path.exists(log_file_name):
    file_list = open(log_file_name,'r').readlines()
    file_list = [x.rstrip() for x in file_list]
    basename_list = [os.path.basename(x) for x in file_list]


# Iterate through nodes in the HDF5 and find all with an STFT array
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    freq_vec = hf5.get_node('/stft','freq_vec')[:]
    time_vec = hf5.get_node('/stft','time_vec')[:]

    phase_node_path_list = \
            [x.__str__().split(" ")[0] for x in hf5.root.stft._f_walknodes() \
            if 'phase_array' in x.__str__()]
    phase_node_path_list.sort()

    # Extract all nodes with phase array
    node_path_list = [os.path.dirname(x) for x in phase_node_path_list]

    # Ask user to select files to perform anaysis on
    selected_files = easygui.multchoicebox(\
            msg = 'Please select files to run analysis on',
            choices = ['{}) '.format(num)+x[6:] \
                    for num,x in enumerate(node_path_list)])
    selected_file_inds = [int(x.split(')')[0]) for x in selected_files]

    phase_node_path_list = [phase_node_path_list[i] for i in selected_file_inds]
    node_path_list = [node_path_list[i] for i in selected_file_inds]

animal_name_date_list = [x.split('/')[2:4] for x in node_path_list]

# Match strings from animal_name_date_list with filenames in the log file
# This will be used from extracting firing rates and raw lfp from
# their respective HDF5 files
matched_filename_inds = [[num for num, filename in enumerate(file_list) \
            if (name_date_str[0] in filename) and (name_date_str[1] in filename)] \
            for name_date_str in animal_name_date_list]
matched_filename_list = [file_list[i[0]] for i in matched_filename_inds]

# Define variables to be maintained across files
initial_dir = dir() + ['initial_dir']

for this_node_num in tqdm(range(len(phase_node_path_list))):

    with tables.open_file(data_hdf5_path,'r') as hf5:
        #phase_array = hf5.get_node(node_path_list[this_node_num],
        #                        'phase_array')[:].swapaxes(0,1)
        amplitude_array = hf5.get_node(node_path_list[this_node_num],
                                'amplitude_array')[:].swapaxes(0,1)

    median_ampltiude_array = np.median(amplitude_array, axis = (1,2))
    normalized_median_amp_array = normalize_timeseries(
            median_ampltiude_array, time_vec, stim_time = 2)

    dat = ephys_data(os.path.dirname((matched_filename_list[this_node_num])))
    dat.firing_rate_params = dict(zip(\
        ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
        ('conv',1,250,1,1e-3,1e-3)))
    dat.extract_and_process()

    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    #                       

    animal_name = animal_name_date_list[this_node_num][0]
    date_str = animal_name_date_list[this_node_num][1]
    plot_dir = os.path.join(data_folder,animal_name,date_str)
    # If directory doesn't exist, make it 
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    #    shutil.rmtree(plot_dir)

    # Plot median spectrogram for all channels to make sure nothing is off
    # Median is preferred over mean to avoid effects by strong outliers
    # (which are quite common)
    firing_overview(
            data = median_ampltiude_array,
            t_vec = time_vec,
            y_values_vec = freq_vec,
            #subplot_labels = middle_channels_bool,
            cmap_lims = 'individual')
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'spectrogram_overview.png'))

    del amplitude_array, median_ampltiude_array

    # Plot normalized median spectrogram for all channels to make sure nothing is off
    firing_overview(
            data = normalized_median_amp_array,
            t_vec = time_vec,
            y_values_vec = freq_vec,
            #subplot_labels = middle_channels_bool,
            cmap_lims = 'individual')
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'normalized_spectrogram_overview.png'))

    del normalized_median_amp_array

    # Plot firing rates for all neurons
    firing_overview(dat.all_normalized_firing)
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'firing_rate_overview.png'))

    # Plot median LFP for every taste for all channels
    # Calculate clims
    firing_overview(np.median(dat.lfp_array, axis = 2).swapaxes(0,1), 
                        cmap = 'viridis',
                        #subplot_labels = middle_channels_bool, 
                        zscore_bool = True)
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'raw_lfp_overview.png'))


    plt.close('all')

