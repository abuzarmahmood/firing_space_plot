# 1) Open single file for now
# 2) extract spiking for all tastes for on&off conditions
# 3) Compute firing rate for ENTIRE 7s (or whatever length of recording)
# 3.5) Maybe smooth firing rate
# 4) Project into 'n' dim space and reduce dimensions

######################### Import dat ish #########################
import tables
import easygui
import os
import numpy as np

import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import kruskal
import shutil

###################### Open file and extract data ################
dir_name = "/media/sf_shared_folder/jian_you_data/tastes_separately/file_1"
os.chdir(dir_name)
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r') 

## For extracting spike trains
dig_in = hf5.list_nodes('/spike_trains')
dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]
dig_in = list(filter(None, dig_in))

## Number of trials per taste for indexing when concatenating spike trains
taste_n = [dig_in[i].spike_array[:].shape[0] for i in range(len(dig_in))]
if np.std(taste_n) == 0:
    taste_n = taste_n[0]
else:
    taste_n = int(easygui.multenterbox('How many trails per taste??',fields = ['# of trials'])[0])

# Which trials did and did not have laser
off_trials = [np.where(dig_in[i].laser_durations[:] == 0)[0] for i in range(len(dig_in))]
on_trials = [np.where(dig_in[i].laser_durations[:] > 0)[0] for i in range(len(dig_in))]

# Get the spike array for each individual taste and put in list:
spikes = [dig_in[i].spike_array[:] for i in range(len(dig_in))]
off_spikes = [spikes[i][off_trials[i],:,:] for i in range(len(dig_in))] #Index trials with no laser
on_spikes = [spikes[i][on_trials[i],:,:] for i in range(len(dig_in))] #Index trials with laser

################### Convert spikes to firing rates ##################
off_firing = 

for l in range(len(off_spikes)):
    for i in range(off_spikes[0].shape[0]):
        for j in range(off_spikes[0].shape[1]):
            for k in range(off_spikes[0].shape[2]):
                off_firing[i, j, k] = np.mean(off_firing[i, j, step_size*k:step_size*k + bin_window_size])

