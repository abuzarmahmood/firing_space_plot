"""
Visualizing hypothesis whether lfp sepctrum varies trial to trial
in a similar manner to HMM states

!) Look at spectra for multiple trial and assess cleanliness of data
    whether trial-to-trial changes are visible
2) Visualize spiking and lfp spectra simultaensouly to see whether
    you can LOOK at the spectrum and firing changing at the same time
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

dat = \
ephys_data('/media/bigdata/Abuzar_Data/AM11/AM11_4Tastes_191030_114043_copy')
dat.firing_rate_params = dat.default_firing_params 

dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()
dat.get_lfps()

dat.get_region_units()

dat.get_stft()

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[...,time_vec<stim_time],axis=-1)[...,np.newaxis]
    array = array/mean_baseline
    return array

############################################################
## Trial-by-trial spectrogram
############################################################
# Select random subset of channels, and plot a few trials
# This can later be extended to look at BLA and GC distinctly and 
# check coordination between the 2 regions
# Only plot AFTER stimulus delivery (to minimize scratching artifacts)

amplitude_array_long = dat.amplitude_array.copy()
amplitude_array_long = amplitude_array_long.swapaxes(1,2)
amplitude_array_long = amplitude_array_long.reshape(\
        (-1, *amplitude_array_long.shape[2:]))
#normalized_amplitude_long = normalize_timeseries(\
#        amplitude_array_long, dat.time_vec, 2)
amplitude_array_long = amplitude_array_long[...,2000:]
#normalized_amplitude_long = normalized_amplitude_long[...,2000:]

trial_count = 5
channel_count = 6
random_trials = np.random.choice(np.arange(amplitude_array_long.shape[0]),
                                trial_count, replace = False)
random_channels = np.random.choice(np.arange(amplitude_array_long.shape[1]),
                                channel_count, replace = False)
random_channels = np.sort(random_channels)
plot_dat = amplitude_array_long[random_trials] 
plot_dat = plot_dat[:,random_channels]
plot_dat = plot_dat.swapaxes(0,1)
iters = np.ndindex(plot_dat.shape[:2])
fig, ax = plt.subplots(channel_count, trial_count)
for this_iter in iters:
    ax[this_iter].imshow(plot_dat[this_iter], aspect='auto')
fig.suptitle(f'Channels {random_channels}')
plt.show()

############################################################
## Trial-by-trial spectrogram + raster
############################################################
# Same stuff as above except now also include spiking for
# plotted trials

# Hopeing indices line up for lfp and spikes
spikes_long = np.array(dat.spikes)
spikes_long = spikes_long.reshape((-1,*spikes_long.shape[2:]))
spike_timelims = np.arange(2000,5000)

trial_count = 8
channel_count = 4
random_trials = np.random.choice(np.arange(amplitude_array_long.shape[0]),
                                trial_count, replace = False)
random_channels = np.random.choice(np.arange(amplitude_array_long.shape[1]),
                                channel_count, replace = False)
random_channels = np.sort(random_channels)
plot_dat = amplitude_array_long[random_trials] 
plot_dat = plot_dat[:,random_channels]
plot_dat = plot_dat.swapaxes(0,1)
# Color lims using Median Absolute Deviation
mad = np.median(np.abs(plot_dat - np.median(plot_dat,axis=None)))
#vmin,vmax = np.min(plot_dat, axis=None),np.max(plot_dat,axis=None)
v_multiplier = 10
vmin, vmax = -v_multiplier*mad, v_multiplier*mad
iters = np.ndindex(plot_dat.shape[:2])
fig, ax = plt.subplots(channel_count+1, trial_count) # +1 to have ax for spikes
for this_iter in iters:
    #ax[this_iter].imshow(zscore(plot_dat[this_iter],axis=-1), aspect='auto')
    ax[this_iter].imshow(plot_dat[this_iter], aspect='auto')
            #vmin = vmin, vmax = vmax)
for num, trial in enumerate(random_trials):
    inds = np.where(spikes_long[trial,...,spike_timelims].T)
    ax[-1,num].scatter(inds[1], inds[0], s=2)
fig.suptitle(f'Channels {random_channels}')
plt.show()

############################################################
## Trial-by-trial spectrogram for BLA + GC 
############################################################
