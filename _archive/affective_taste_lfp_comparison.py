# ==============================
# Setup
# ==============================

#Import necessary tools
import numpy as np
import tables
import easygui
import os
import glob
import matplotlib.pyplot as plt
import re
import sys
from tqdm import tqdm, trange
#Import specific functions in order to filter the data file
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import spectrogram
from scipy.stats import zscore

# ==============================
# Define Functions 
# ==============================

def get_filtered_electrode(data, low_pass, high_pass, sampling_rate):
    el = 0.195*(data)
    m, n = butter(
            2, 
            [2.0*int(low_pass)/sampling_rate, 2.0*int(high_pass)/sampling_rate], 
            btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

# ==============================
# Collect user input needed for later processing 
# ==============================
affective_dir = '/media/bigdata/brads_data/BS23_Saline_171217_094642'
taste_dir = '/media/bigdata/brads_data/BS23_4Tastes_171217_103315'

# Load data
raw_affective_files = np.sort(glob.glob(affective_dir + '/*amp*dat*'))
raw_taste_files = np.sort(glob.glob(taste_dir + '/*amp*dat*'))

raw_affective_data = np.array([np.fromfile(this_file, dtype = np.dtype('int16')) \
                            for this_file in tqdm(raw_affective_files)])
raw_taste_data = np.array([np.fromfile(this_file, dtype = np.dtype('int16')) \
                            for this_file in tqdm(raw_taste_files)])

# Downsample both by 30 to make timesteps 1ms
raw_affective_data = np.mean(raw_affective_data.reshape(\
                (raw_affective_data.shape[0],-1,30)),axis=-1)
raw_taste_data = np.mean(raw_taste_data.reshape(\
                (raw_taste_data.shape[0],-1,30)),axis=-1)

# Truncate to 20th value for downsamping below (good for upto 25Hz)
raw_affective_data = raw_affective_data[:,:(20*(raw_affective_data.shape[-1]//20))]
raw_taste_data = raw_taste_data[:,:(20*(raw_taste_data.shape[-1]//20))]

down_affective_data = np.mean(raw_affective_data.reshape(\
                (raw_affective_data.shape[0],-1,20)),axis=-1)
down_taste_data = np.mean(raw_taste_data.reshape(\
                (raw_taste_data.shape[0],-1,20)),axis=-1)

# Also pick half of channels to make spectrogram easier
down_affective_data = down_affective_data[np.arange(0, down_affective_data.shape[0],2)]
down_taste_data = down_taste_data[np.arange(0, down_taste_data.shape[0],2)]


# Plot color-matched heatmaps for both regions to compare amplitude
mean_val = np.mean(np.concatenate((down_affective_data,down_taste_data),axis=1),axis=None)
mean_std = np.std(np.concatenate((down_affective_data,down_taste_data),axis=1),axis=None)

fig,ax = plt.subplots(2,1, sharex = True)
ax[0].imshow(down_affective_data,vmin = mean_val - 3*mean_std, vmax = mean_val + 3*mean_std,
        interpolation = 'nearest', aspect = 'auto')
ax[1].imshow(down_taste_data,vmin = mean_val - 3*mean_std, vmax = mean_val + 3*mean_std,
        interpolation = 'nearest', aspect = 'auto')
plt.show()

# Plot overlayed error plots for mean traces from both regions
affective_mean = np.mean(down_affective_data,axis=0)
affective_std = np.std(down_affective_data,axis=0)
taste_mean = np.mean(down_taste_data,axis=0)
taste_std = np.std(down_taste_data,axis=0)

plt.figure()
plt.plot(affective_mean)
plt.plot(taste_mean)
plt.fill_between(range(len(affective_mean)),
                affective_mean - affective_std, affective_mean + affective_std) 
plt.fill_between(range(len(taste_mean)),
                taste_mean - taste_std, taste_mean + taste_std) 
plt.show()

# For completeness, plot spectrograms of both regions side by side
Fs = 50 
signal_window = 500 
window_overlap = 499
max_freq = 25

f_aff, t_aff, S_aff = spectrogram(x = down_affective_data, 
                                    fs = Fs,
                                    window = 'hanning',
                                    nperseg = signal_window,
                                    noverlap=signal_window-(signal_window-window_overlap)) 
f_taste, t_taste, S_taste = spectrogram(x = down_taste_data, 
                                    fs = Fs,
                                    window = 'hanning',
                                    nperseg = signal_window,
                                    noverlap=signal_window-(signal_window-window_overlap)) 

S_aff = S_aff[:,f_aff<max_freq]
S_taste = S_taste[:,f_taste<max_freq]
f_aff = f_aff[f_aff<max_freq]
f_taste = f_taste[f_taste<max_freq]

aff_power = np.abs(S_aff)
taste_power = np.abs(S_taste)


# Plot mean power for both recordings
min_val,max_val = np.min(np.mean(S_aff,axis=0)),\
                    np.max(np.mean(S_aff,axis=0))
fig, ax = plt.subplots(2,1,sharex = True)
ax[0].pcolormesh(t_aff,f_aff,np.mean(S_aff,axis=0),
                    cmap = 'jet', vmin = min_val, vmax = max_val)
ax[1].pcolormesh(t_taste,f_taste,np.mean(S_taste,axis=0), 
        cmap = 'jet', vmin = min_val, vmax = max_val)
plt.show()

# Plot mean power for both recordings
min_val,max_val = np.min(10*np.log10(np.mean(S_aff,axis=0))),\
                    np.max(10*np.log10(np.mean(S_aff,axis=0)))
fig, ax = plt.subplots(2,1,sharex = True)
ax[0].pcolormesh(t_aff,f_aff,10*np.log10(np.mean(S_aff,axis=0)),
                    cmap = 'jet', vmin = min_val, vmax = max_val)
ax[1].pcolormesh(t_taste,f_taste,10*np.log10(np.mean(S_taste,axis=0)), 
        cmap = 'jet', vmin = min_val, vmax = max_val)
plt.show()

# Concatenate and plot both zscores
mean_aff = np.mean(S_aff, axis=0)
mean_taste = np.mean(S_taste, axis=0)
total_S = np.concatenate((mean_aff,mean_taste),axis=-1)

plt.imshow(zscore(10*np.log10(total_S),axis=-1),interpolation='nearest',aspect='auto',cmap='jet')
plt.show()
