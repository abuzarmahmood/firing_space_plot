"""
Script to load all files from Brad's paradigm for a single 
session and perform the following tests/visualisations:
    1) Are there differences in taste ITI LFPs between tastes
        for different bands
    2) Do changes in sickness persists into the taste delivery period
        and are they detectable in the LFP during the ITI period
    3) What are the dynamics of the LFP during the ITI periods
        and if sickness is noticeable in the LFP does it rebound
        towards health
"""

# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   

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
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import zscore
import glob
from collections import namedtuple
from scipy.signal import convolve


# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#

##############
# Define functions to extract data
#############

def get_spikes(hdf5_name):
    """
    Extract spike arrays from specified HD5 files
    """
    with tables.open_file(hdf5_name, 'r+') as hf5: 
    
        # Iterate through tastes and extract spikes from laser on and off conditions
        # If no array for laser durations, put everything in laser off
        
        dig_in_list = \
            [x for x in hf5.list_nodes('/spike_trains') if 'dig_in' in x.__str__()]
        
        spikes = np.asarray([dig_in.spike_array[:] for dig_in in dig_in_list])
        return spikes

def calc_normalized_firing(spike_array,kern_length):
    """
    spike_array :: N-D array with time as last dim
    """
    kern_shape = (*np.ones(len(spike_array.shape)-1),kern_length)
    kern_shape = [int(x) for x in kern_shape]
    firing_rates = convolve(spike_array, np.ones(kern_shape))
    return firing_rates

def get_parsed_lfp(hdf5_name):
    """
    Extract parsed lfp arrays from specified HD5 files
    """
    with tables.open_file(hdf5_name, 'r+') as hf5: 
        if 'Parsed_LFP' in hf5.list_nodes('/').__str__():
            lfp_nodes = [node for node in hf5.list_nodes('/Parsed_LFP')\
                    if 'dig_in' in node.__str__()]
            lfp_array = np.asarray([node[:] for node in lfp_nodes])
            all_lfp_array = \
                    lfp_array.\
                        swapaxes(1,2).\
                        reshape(-1, lfp_array.shape[1],\
                                lfp_array.shape[-1]).\
                        swapaxes(0,1)
        else:
            raise Exception('Parsed_LFP node absent in HDF5')
    return all_lfp_array

def get_whole_session_lfp(hdf5_name):
    with tables.open_file(hdf5_name, 'r+') as hf5: 
        whole_session_lfp_node = hf5.list_nodes('/Whole_session_raw_LFP') 
        whole_lfp = whole_session_lfp_node[0][:]
    return whole_lfp

def get_delivery_times(hdf5_name):
    delivery_times = \
            pd.read_hdf(hdf5_name,'/Whole_session_spikes/delivery_times')
    delivery_times['taste'] = delivery_times.index
    delivery_times = \
            pd.melt(delivery_times,
                    id_vars = 'taste',
                    var_name ='trial',
                    value_name='delivery_time')
    delivery_times.sort_values(by='delivery_time',inplace=True)
    # Delivery times are in 30kHz samples, convert to ms
    delivery_times['delivery_time'] = delivery_times['delivery_time'] // 30
    delivery_times['chronological'] = np.argsort(delivery_times.delivery_time)
    return delivery_times

# All HDF5 files need to be in the same folder
# Load files and make sure the order is right
dir_name = '/media/bigdata/brads_data/Brad_LFP_ITI_analyses/BS23'

final_confirmation = 'n'
while 'y' not in final_confirmation:
    hdf5_name = glob.glob(
            os.path.join(dir_name, '**.h5'))
    selection_list = ['{}) {} \n'.format(num,os.path.basename(file)) \
            for num,file in enumerate(hdf5_name)]
    selection_string = 'Please enter the number of the HDF5 files in the following'\
                ' order \n (as a comma separated string e.g. 2,1,3,4,0):'\
                '\n Day1 Saline \n Day1 LiCl \n Day1 Taste \n' \
                ' Day3 Saline \n Day3 Taste:\n{}'.\
                    format("".join(selection_list))
    file_selection = input(selection_string)
    file_order = [int(x) for x in file_selection.split(',')]
    # Check with user that the order is right
    final_list = [hdf5_name[x] for x in file_order]
    exp_list = ['Day1 Saline','Day1 LiCl','Day1 Taste',
                'Day3 Saline', 'Day3 Taste']
    final_selection_list = ['{}) {} \n'.format(exp,os.path.basename(file)) \
            for exp,file in zip(exp_list,final_list)]
    final_selection_string = 'Is this order correct (y/n): \n {}'.\
            format("".join(final_selection_list))
    final_confirmation = input(final_selection_string)

# Pull in spike trains from all sessions to calculate firing rates
# Replace list with

spike_tuple = namedtuple('SpikeTrains',['filename','spiketrains'])

all_spike_trains = [spike_tuple(file_name, get_spikes(file_name)) \
        for file_name in tqdm(final_list)]

# Elements 0,1,3 will be from the affective recording 
# So they're shaped a little weirdly
affective_recordings = [0,1,3]
fin_spike_trains = [spike_train.spiketrains[0,0] \
        if recording_num in affective_recordings \
        else spike_train.spiketrains \
        for recording_num, spike_train in enumerate(all_spike_trains)]

# Calculate firing rates for all spike_trains
fin_firing_rates = [calc_normalized_firing(x,250) for x in tqdm(fin_spike_trains)]

# Downsample spiketrains and delete originals
down_ratio = 25
down_firing_rate = [x[..., np.arange(0,x.shape[-1],down_ratio)] \
        for x in fin_firing_rates]

del fin_firing_rates

# Extract LFP from all sessions
whole_lfp = [np.squeeze(get_parsed_lfp(file_name))
        if recording_num in affective_recordings \
        else get_whole_session_lfp(file_name) \
        for recording_num, file_name in tqdm(enumerate(final_list))]

taste_whole_lfp = [whole_lfp[2],whole_lfp[-1]]
affective_whole_lfp = [whole_lfp[x] for x in [0,1,3]]

# Extract ITI's from taste sessions
taste_files = [final_list[2], final_list[-1]]
trial_time_data = [get_delivery_times(file_name) \
        for file_name in taste_files]
delivery_time_list = [x.delivery_time for x in trial_time_data]

# Define parameters to extract ITI data
time_before_delivery = 10 #seconds
padding = 1 #second before taste delivery won't be extracted
Fs = 1000 # Sampling frequency

# (trials x channels x time)
iti_array_list = np.asarray(\
        [[lfp_array[:,(x-(time_before_delivery*Fs)):(x-(padding*Fs))]\
        for x in delivery_times] \
        for delivery_times,lfp_array in \
        zip(delivery_time_list,taste_whole_lfp)])

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

# Bandpass filter lfp into relevant bands

#define bandpass filter parameters to parse out frequencies
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

band_freqs = [(1,4),
                (4,7),
                (7,12),
                (12,25),
                (25,50)]


affective_whole_bandpassed = \
            [np.asarray([
                    butter_bandpass_filter(
                        data = data, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])\
                for data in affective_whole_lfp]

taste_whole_bandpassed  = \
            [np.asarray([
                    butter_bandpass_filter(
                        data = data, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])\
                for data in taste_whole_lfp]

iti_lfp_bandpassed  = \
            [np.asarray([
                    butter_bandpass_filter(
                        data = data, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])\
                for data in iti_array_list]

# Calculate Hilbert and amplitude
affective_whole_hilbert = [hilbert(data) for data in tqdm(affective_whole_bandpassed)]
taste_whole_hilbert = [hilbert(data) for data in tqdm(taste_whole_bandpassed)]
iti_lfp_bandpassed = [hilbert(data) for data in tqdm(iti_lfp_bandpassed)]

whole_lfp_amplitude = np.abs(whole_lfp_hilbert)
iti_lfp_amplitude = np.abs(iti_lfp_hilbert)
affective_lfp_amplitude = np.abs(affective_lfp_hilbert)

