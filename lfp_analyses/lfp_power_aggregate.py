
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
from sklearn.utils import resample
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from scipy.stats import zscore

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

##################################################
## Define functions
##################################################

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    #mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    #array -= mean_baseline
    return array

def firing_overview(data, t_vec = None, y_values_vec = None,
                    interpolation = 'nearest',
                    cmap = 'jet',
                    min_val = None, max_val=None, 
                    subplot_labels = None):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron
    """
    if data.shape[-1] != len(time_vec):
        raise Exception('Time dimension in data needs to be'\
            'equal to length of time_vec')
    num_nrns = data.shape[0]

    if min_val is None:
        min_val = np.min(data,axis=None)
    elif max_val is None:
        max_val = np.max(data,axis=None)
    elif t_vec is None:
        t_vec = np.arange(data.shape[-1])
    elif y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])

    # Plot firing rates
    square_len = np.int(np.ceil(np.sqrt(num_nrns)))
    fig, ax = plt.subplots(square_len,square_len, sharex='all',sharey='all')
    
    nd_idx_objs = []
    for dim in range(ax.ndim):
        this_shape = np.ones(len(ax.shape))
        this_shape[dim] = ax.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to( 
                    np.reshape(
                        np.arange(ax.shape[dim]),
                        this_shape.astype('int')), ax.shape).flatten())
    
    if subplot_labels is None:
        subplot_labels = np.zeros(num_nrns)
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])
    for nrn in range(num_nrns):
        plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
        plt.gca().set_title('{}:{}'.format(int(subplot_labels[nrn]),nrn))
        plt.gca().pcolormesh(t_vec, y_values_vec,
                data[nrn,:,:],cmap=cmap,
                vmin = min_val, vmax = max_val)
    return ax

##################################################
## Read in data 
##################################################


# Define middle channels in board
middle_channels = np.arange(8,24)

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Pull out all terminal groups (leafs) under stft
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    freq_vec = hf5.get_node('/stft','freq_vec')[:]
    time_vec = hf5.get_node('/stft','time_vec')[:]

    amplitude_node_list = [x for x in hf5.root.stft._f_walknodes() \
            if 'amplitude_array' in x.__str__()]
    print('Extracting phase info')
    amplitude_array_list = [x[:] for x in tqdm(amplitude_node_list)]
    # Extract all nodes with phase array
    node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) for x in amplitude_node_list]
    # Pull channels selected for coherence analyses
    relative_channel_nums = [hf5.get_node(this_path,'relative_region_channel_nums')[:] \
            for this_path in node_path_list]
    # Pull parsed_lfp_channel from each array
    parsed_channel_list  = [hf5.get_node(path,'parsed_lfp_channels')[:] for path in node_path_list]

for this_node_num in tqdm(range(len(phase_node_list))):
# Plot 8 + 9
# Power in each band over all trials for both regions separately
# Sort trials by time of max power per taste, per region
    this_amplitude_array = amplitude_array_list[this_node_num]\
            [:,relative_channel_nums[this_node_num]]

mean_channel_amplitude_array_long = np.reshape(mean_channel_amplitude_array,
        (-1,np.prod(mean_channel_amplitude_array.shape[1:3]),
            *mean_channel_amplitude_array.shape[3:]))
        
# zscore array along trials for every timepoint
zscore_amplitude_array_long = np.array([[zscore(freq,axis=0) for freq in region]\
        for region in mean_channel_amplitude_array_long.swapaxes(1,2)])
#zscore_amplitude_array_long = zscore(mean_channel_amplitude_array_long,axis=1)
stim_time = 2
max_times = np.argmax(zscore_amplitude_array_long[...,time_vec>stim_time], axis=-1)
trials_per_taste = mean_channel_amplitude_array.shape[2]
trial_order = np.zeros(max_times.shape)
for region_num,region in enumerate(max_times):
    for freq_num,freq in enumerate(region):
        for taste in range(mean_channel_amplitude_array.shape[1]):
            trial_order[region_num,freq_num,
                    (taste*trials_per_taste):((taste+1)*trials_per_taste)] = \
                        taste*trials_per_taste + \
                        np.argsort(freq[(taste*trials_per_taste):((taste+1)*trials_per_taste)])

sorted_zscore_amplitude = np.array(
        [[freq[(len(freq_order) - 1) - freq_order] \
                for freq,freq_order in zip(region,region_order)]\
                for region,region_order in \
                zip(zscore_amplitude_array_long,trial_order.astype(int))])

# Cycle through max times for every band and region and sort trials within tastes
firing_overview(
            sorted_zscore_amplitude[0],
            t_vec = time_vec,subplot_labels = freq_vec)
fig = plt.gcf()
fig.set_size_inches(8,10)
fig.suptitle("_".join(animal_date_list))
fig.savefig(os.path.join(this_plot_dir,'RG0_Freq_power_trials'))

firing_overview(
            sorted_zscore_amplitude[1],
            t_vec = time_vec,subplot_labels = freq_vec)
fig = plt.gcf()
fig.set_size_inches(8,10)
fig.suptitle("_".join(animal_date_list))
fig.savefig(os.path.join(this_plot_dir,'RG1_Freq_power_trials'))

# Plot 10
# a-d) Mean spectrogram for each taste
# e) Mean spectrogram across all tastes
# Pre-determine color limits
mean_channel_trial_amplitude_array = np.mean(mean_channel_amplitude_array,axis=2)
min_val,max_val = mean_channel_trial_amplitude_array.min(),\
                    mean_channel_trial_amplitude_array.max()
fig, ax = plt.subplots(5,2,sharex='all',sharey='all')
for region_num, region in enumerate(mean_channel_trial_amplitude_array):
    for taste_num,taste in enumerate(region):
        ax[taste_num,region_num].pcolormesh(
                #time_vec,freq_vec,normalize_timeseries(taste,time_vec,2),
                time_vec,freq_vec,taste,
                cmap = 'jet', vmin= min_val, vmax = max_val)
ax[-1,0].pcolormesh(time_vec,freq_vec,
        #normalize_timeseries(np.mean(mean_trial_channel_amplitude_array,axis=(1))[0],time_vec,2),
        np.mean(mean_channel_amplitude_array,axis=(1))[0],
        cmap = 'jet', vmin= min_val, vmax = max_val)
ax[-1,1].pcolormesh(time_vec,freq_vec,
        #normalize_timeseries(np.mean(mean_trial_channel_amplitude_array,axis=(1))[1],time_vec,2),
        np.mean(mean_channel_amplitude_array,axis=(1))[1],
        cmap = 'jet', vmin= min_val, vmax = max_val)
ax[-1,0].set_title('Average spectrum')
ax[-1,1].set_title('Average spectrum')
ax[0,0].set_title('Region 0')
ax[0,1].set_title('Region 1')
fig.set_size_inches(8,10)
fig.suptitle("_".join(animal_date_list))
fig.savefig(os.path.join(this_plot_dir,'Average_Spectra'))

