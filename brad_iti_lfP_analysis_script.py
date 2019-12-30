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

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#
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

def calc_normalized_firing(spike_array):
    """
    spike_array :: N-D array with time as last dim
    """


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


