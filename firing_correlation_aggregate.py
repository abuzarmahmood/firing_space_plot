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
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

########################################################
# Define functions and relevant variables to extract data
########################################################

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),os.path.basename(path_to_node))

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Define middle channels in board
middle_channels = np.arange(8,24)

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

with tables.open_file(data_hdf5_path,'r') as hf5:
    firing_node_list  = [x for x in hf5.root.firing._f_walknodes() \
            if 'normalized_firing_array' in x.__str__()]
    print('Extracting firing info')
    firing_array_list = [x[:] for x in tqdm(firing_node_list )]
    # Extract all node names with firing array
    node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) for x in firing_node_list]
    # Pull electrode numbers from each array
    electrode_num_list  = [hf5.get_node(path,'unit_electrode')[:] for path in node_path_list]
    spike_array_list =  [hf5.get_node(path,'spike_array')[:] for path in node_path_list]

print('Calculating firing correlations')
#for this_node_num in tqdm(range(len(phase_node_list))):

this_node_num = 0

firing_array = firing_array_list[this_node_num].swapaxes(0,1)
# Should be a better wap to reshape spike_array
spike_array = spike_array_list[this_node_num].swapaxes(0,2).swapaxes(1,2)
electrode_nums = electrode_num_list[this_node_num]
middle_channels_bool = np.array([True if channel in middle_channels else False \
        for channel in electrode_nums ])
firing_array_split = [firing_array[middle_channels_bool], firing_array[~middle_channels_bool]]
spike_array_split = [spike_array[middle_channels_bool], spike_array[~middle_channels_bool]]
channel_num_split = \
        [electrode_nums[middle_channels_bool], electrode_nums[~middle_channels_bool]]
relative_channel_num_split = \
        [np.arange(len(electrode_nums))[middle_channels_bool], 
                np.arange(len(electrode_nums))[~middle_channels_bool]]
del firing_array


#  ____                    _       _   _             
# / ___|___  _ __ _ __ ___| | __ _| |_(_) ___  _ __  
#| |   / _ \| '__| '__/ _ \ |/ _` | __| |/ _ \| '_ \ 
#| |__| (_) | |  | | |  __/ | (_| | |_| | (_) | | | |
# \____\___/|_|  |_|  \___|_|\__,_|\__|_|\___/|_| |_|
#                                                    

# 1) Correlation of spike trains
#   a) Convolve spike trains
#   b) Convolve shuffled spike trains

## Have this stored somewhere
Fs = 1000
time_vec = np.arange(0,7)/Fs

# Truncate firing_array and spike_array to have data only after stimulus delivery

# Actually, don't truncate

# 2) Correlation of firing rates
