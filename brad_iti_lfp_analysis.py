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
import lspopt
import tables
import easygui
import scipy
from scipy.signal import spectrogram
from lspopt import spectrogram_lspopt
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from sklearn.utils import resample
from itertools import product
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA as pca
from sklearn.mixture import GaussianMixture as gmm
from matplotlib.colors import LogNorm

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extracting spikes, firing rates and raw_lfp

# Load affective and taste sessions
affective_dat = ephys_data('/media/bigdata/brads_data/Brad_LFP_ITI_analyses/BS23')
taste_dat = ephys_data('/media/bigdata/brads_data/Brad_LFP_ITI_analyses/BS23')

affective_dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))
taste_dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

affective_dat.get_spikes()
affective_dat.get_lfps()
affective_dat.all_lfp_array = np.squeeze(affective_dat.all_lfp_array)
# Have to do weird shit with indexing to make this work
affective_dat.spikes = [affective_dat.spikes[0][0][np.newaxis,:,:]]

# Get firing rates
affective_dat.get_firing_rates()
plt.imshow(affective_dat.all_normalized_firing[:,0],
        interpolation='bilinear',aspect='auto',cmap='jet');plt.show()

taste_dat.extract_and_process()
taste_dat.firing_overview(taste_dat.all_normalized_firing);plt.show()

# Extract and plot whole session spikes
with tables.open_file(taste_dat.hdf5_name,'r') as h5:
    whole_spikes = h5.root.Whole_session_spikes.all_spikes[:]
# Whole spikes is in 30000Khz, downsample to 1ms
whole_spikes[1] = [x//30 for x in whole_spikes[1]]

# Generate downsamples spike array
down_rate = 10
down_spike_times = [x//down_rate for x in whole_spikes[1]]
max_len = np.max(down_spike_times)
whole_spike_array = np.empty((np.max(whole_spikes[0])+1,max_len+1))
for spike in list(zip(whole_spikes[0],down_spike_times)):
    whole_spike_array[spike[0],spike[1]] = 1

window_size = 25 #Since already downsampled to 10ms bins
box_kern = np.ones((1,window_size))
box_kern = box_kern / np.sum(box_kern)
whole_firing = scipy.signal.convolve(whole_spike_array,box_kern)
normalized_whole_firing = [(x - np.min(x))/(np.max(x)-np.min(x)) for x in whole_firing]

plt.imshow(normalized_whole_firing,
        interpolation='bilinear',aspect='auto',cmap='jet');plt.show()

## Doesn't tell us much :p
# Plot LFPs
#mean_val = np.mean(affective_dat.all_lfp_array, axis = None)
#sd_val = np.std(affective_dat.all_lfp_array, axis = None)
#plt.imshow(affective_dat.all_lfp_array, vmin= mean_val - 2*sd_val,
#                    vmax= mean_val + 2*sd_val, cmap = 'viridis',
#                    interpolation='bilinear',aspect='auto');plt.show()
#
#taste_dat.all_lfp_array = np.squeeze(taste_dat.all_lfp_array)
#mean_val = np.mean(taste_dat.all_lfp_array, axis = None)
#sd_val = np.std(taste_dat.all_lfp_array, axis = None)
#taste_dat.firing_overview(taste_dat.all_lfp_array, min_val = mean_val - 2*sd_val,
#                    max_val = mean_val + 2*sd_val, cmap = 'viridis',
#                    time_step = 1,interpolation='bilinear');plt.show()
#

# Extract whole session LFPs
# This has already been bandpassed 1-300Hz
with tables.open_file(taste_dat.hdf5_name,'r') as hf5:
    whole_session_lfp_node = hf5.list_nodes('/Whole_session_raw_LFP') 
    whole_lfp = whole_session_lfp_node[0][:]

# Whole spectrogram
signal_window = 2000 
window_overlap = 1950
fbounds = (1,50)
Fs = 1000

f_whole,t_whole,whole_spectrograms= scipy.signal.spectrogram(
            scipy.signal.detrend(whole_lfp), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap), 
            mode='psd')

whole_spectrograms = whole_spectrograms[:,(f_whole>fbounds[0])*(f_whole<fbounds[1])]
f_whole= f_whole[(f_whole>fbounds[0])*(f_whole<fbounds[1])]

# mean affective spectrogram across channels
mean_whole_spectrogram = np.mean(whole_spectrograms,axis=0)

# Plot zscore log10 values
fig,ax = plt.subplots(2,1)
ax[0].imshow(10*np.log10(mean_whole_spectrogram),cmap='jet',
        interpolation='bilinear',aspect='auto',origin='below')
ax[0].title.set_text('log10 Power Whole Taste session')
ax[1].imshow(zscore(10*np.log10(mean_whole_spectrogram),axis=1),
        cmap='jet',interpolation='bilinear',aspect='auto',
        origin='below',vmin = -1,vmax=1)
ax[1].title.set_text('log10 Zscore power Whole Taste session')
plt.show()

# Extract ITIs from whole session lfp
# Extract taste delivery times
delivery_times = pd.read_hdf(taste_dat.hdf5_name,'/Whole_session_spikes/delivery_times')
delivery_times['taste'] = delivery_times.index
delivery_times = pd.melt(delivery_times,id_vars = 'taste',var_name ='trial',value_name='delivery_time')
delivery_times.sort_values(by='delivery_time',inplace=True)
# Delivery times are in 30kHz samples, convert to ms
delivery_times['delivery_time'] = delivery_times['delivery_time'] // 30
delivery_times['chronological'] = np.argsort(delivery_times.delivery_time)

# Define parameters to extract ITI data
time_before_delivery = 10 #seconds
padding = 1 #second before taste delivery won't be extracted

# (trials x channels x time)
iti_array = np.asarray([whole_lfp[:,(x-(time_before_delivery*Fs)):(x-(padding*Fs))]\
        for x in delivery_times.delivery_time])


# ____                  _                                           
#/ ___| _ __   ___  ___| |_ _ __ ___   __ _ _ __ __ _ _ __ ___  ___ 
#\___ \| '_ \ / _ \/ __| __| '__/ _ \ / _` | '__/ _` | '_ ` _ \/ __|
# ___) | |_) |  __/ (__| |_| | | (_) | (_| | | | (_| | | | | | \__ \
#|____/| .__/ \___|\___|\__|_|  \___/ \__, |_|  \__,_|_| |_| |_|___/
#      |_|                            |___/                         

# Mean spectrograms across all channels

# Spectrogram for affective period and ITIs 

# Affective spectrogram
f_aff,t_aff,affective_spectrograms= scipy.signal.spectrogram(
            scipy.signal.detrend(affective_dat.all_lfp_array), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap), 
            mode='psd')
affective_spectrograms = affective_spectrograms[:,(f_aff>fbounds[0])*(f_aff<fbounds[1])]
f_aff = f_aff[(f_aff>fbounds[0])*(f_aff<fbounds[1])]

# mean affective spectrogram across channels
mean_affective_spectrogram = np.mean(affective_spectrograms,axis=0)

# Plot zscore log10 values
fig,ax = plt.subplots(2,1)
ax[0].imshow(10*np.log10(mean_affective_spectrogram),origin='below',
        interpolation='bilinear',aspect='auto', cmap='jet')
ax[0].title.set_text('log10 Power Affective session')
ax[1].imshow(zscore(10*np.log10(mean_affective_spectrogram),axis=1),origin='below',
        interpolation='bilinear',aspect='auto', cmap='jet',
        vmin = -1,vmax=1)
ax[1].title.set_text('Zscore log10 power Affective session')
plt.show()

# ITI spectrograms
f_iti,t_iti,iti_spectrograms= scipy.signal.spectrogram(
            scipy.signal.detrend(iti_array), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap), 
            mode='psd')
iti_spectrograms = iti_spectrograms[:,:,(f_iti>fbounds[0])*(f_iti<fbounds[1])]
f_iti = f_iti[(f_iti>fbounds[0])*(f_iti<fbounds[1])]

# mean iti spectrogram across channels
mean_iti_spectrograms = np.mean(iti_spectrograms,axis=1)

# Roll open ITI spectrogram
mean_iti_spectrogram_long = mean_iti_spectrograms.swapaxes(0,1)
mean_iti_spectrogram_long = mean_iti_spectrogram_long.reshape(\
        mean_iti_spectrogram_long.shape[0], np.prod(mean_iti_spectrogram_long.shape[1:]))

# Plot average power for every trial
fig,ax = plt.subplots(2,1)
ax[0].imshow(10*np.log10(mean_iti_spectrogram_long),origin='below',
        interpolation='bilinear',aspect='auto', cmap='jet')
ax[0].title.set_text('log10 ITI power over Taste session')
ax[1].imshow(zscore(10*np.log10(mean_iti_spectrogram_long),axis=1),origin='below',
        interpolation='bilinear',aspect='auto', cmap='jet',vmin=-1,vmax=1)
ax[1].title.set_text('Zscore log10 ITI power over Taste session')
plt.show()

# Concatenate spectrograms
tot_spectrograms = np.concatenate((mean_affective_spectrogram,mean_iti_spectrogram_long),1) 

plt.imshow(zscore(10*np.log10(tot_spectrograms)),origin='below',
        interpolation='bilinear',aspect='auto', cmap='jet',vmin=-1,vmax=1)
plt.show()


# ____                  _                     
#| __ )  __ _ _ __   __| |_ __   __ _ ___ ___ 
#|  _ \ / _` | '_ \ / _` | '_ \ / _` / __/ __|
#| |_) | (_| | | | | (_| | |_) | (_| \__ \__ \
#|____/ \__,_|_| |_|\__,_| .__/ \__,_|___/___/
#                        |_|                  

# Extract bandpassed lfp
# Bandpass filter whole_lfp

define bandpass filter parameters to parse out frequencies
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


Fs = 1000 

# Bandpass lfp
whole_lfp_bandpassed = np.asarray([
                    butter_bandpass_filter(
                        data = whole_lfp, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])

iti_lfp_bandpassed = np.asarray([
                    butter_bandpass_filter(
                        data = iti_array, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])

affective_lfp_bandpassed = np.asarray([
                    butter_bandpass_filter(
                        data = affective_dat.all_lfp_array, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])

# Calculate Hilbert and amplitude
whole_lfp_hilbert = hilbert(whole_lfp_bandpassed)
iti_lfp_hilbert = hilbert(iti_lfp_bandpassed)
affective_lfp_hilbert = hilbert(affective_lfp_bandpassed)

whole_lfp_amplitude = np.abs(whole_lfp_hilbert)
iti_lfp_amplitude = np.abs(iti_lfp_hilbert)
affective_lfp_amplitude = np.abs(affective_lfp_hilbert)


# Plot to make sure it's working
tmax = 100000
plt.plot(whole_lfp_bandpassed[0,0,:tmax])
plt.plot(whole_lfp_ampltidue[0,0,:tmax])
plt.show()

# Downsample amplitude to make it more amenable to plotting
down_ratio = 10
whole_lfp_amplitude_down = whole_lfp_amplitude\
        [:,:,np.arange(0,whole_lfp_amplitude.shape[-1],down_ratio)]
iti_lfp_amplitude_down = iti_lfp_amplitude\
        [:,:,:,np.arange(0,iti_lfp_amplitude.shape[-1],down_ratio)]
mean_iti_lfp_amplitude_down = np.mean(iti_lfp_amplitude_down,axis=-1).swapaxes(1,2)
affective_lfp_amplitude_down = affective_lfp_amplitude\
        [:,:,np.arange(0,affective_lfp_amplitude.shape[-1],down_ratio)]

# Take average across channels
# Entire taste session lfp
this_dat = whole_lfp_amplitude_down 
mean_this_dat = np.mean(this_dat,axis=1)
std_this_dat = np.std(this_dat,axis=1)
fig,ax = plt.subplots(len(this_dat),1,sharex=True)
for ax_num,this_ax in enumerate(ax):
    this_ax.fill_between(x=range(this_dat[ax_num].shape[-1]),
            y1 = mean_this_dat[ax_num] - std_this_dat[ax_num],
            y2= mean_this_dat[ax_num] + std_this_dat[ax_num],
            alpha = 0.5,color='orange')
    this_ax.plot(range(len(mean_this_dat[ax_num])),mean_this_dat[ax_num])
plt.show()

# Taste session ITIs 
# Heatmaps
mean_channel_iti_lfp_amp_down = np.mean(iti_lfp_amplitude_down,axis=2)
zscore_channel_iti_lfp_amp_down = np.array([zscore(x,axis=None) \
        for x in mean_channel_iti_lfp_amp_down])
affective_dat.firing_overview(zscore_channel_iti_lfp_amp_down);plt.show()

# Error bars
this_dat = mean_iti_lfp_amplitude_down 
mean_this_dat = np.mean(this_dat,axis=1)
std_this_dat = np.std(this_dat,axis=1)
fig,ax = plt.subplots(len(this_dat),1,sharex=True)
for ax_num,this_ax in enumerate(ax):
    this_ax.fill_between(x=range(this_dat[ax_num].shape[-1]),
            y1 = mean_this_dat[ax_num] - std_this_dat[ax_num],
            y2= mean_this_dat[ax_num] + std_this_dat[ax_num],
            alpha = 0.5,color='orange')
    this_ax.plot(range(len(mean_this_dat[ax_num])),mean_this_dat[ax_num])
plt.show()

# Affective session 
this_dat = affective_lfp_amplitude_down 
mean_this_dat = np.mean(this_dat,axis=1)
std_this_dat = np.std(this_dat,axis=1)
fig,ax = plt.subplots(len(this_dat),1,sharex=True)
for ax_num,this_ax in enumerate(ax):
    this_ax.fill_between(x=range(this_dat[ax_num].shape[-1]),
            y1 = mean_this_dat[ax_num] - std_this_dat[ax_num],
            y2= mean_this_dat[ax_num] + std_this_dat[ax_num],
            alpha = 0.5,color='orange')
    this_ax.plot(range(len(mean_this_dat[ax_num])),mean_this_dat[ax_num])
plt.show()


#############
# Question:
# Is ITI LFP more similar to different parts of the affective period
# Or is ITI LFP power more similar to first or second half of LiCl affective recording
############

# Unroll iti_lfp_ampltiude
iti_lfp_amplitude_down_long = iti_lfp_amplitude_down.swapaxes(1,2).\
        reshape((iti_lfp_amplitude_down.shape[0],
                    iti_lfp_amplitude_down.shape[2],
                    -1))

# Side by side plots of ITI and Affective bands 
fig,ax = plt.subplots(len(this_dat),2,sharey='row')
this_dat = iti_lfp_amplitude_down_long 
mean_this_dat = np.mean(this_dat,axis=1)
std_this_dat = np.std(this_dat,axis=1)
for ax_num,this_ax in enumerate(ax[:,1]):
    this_ax.fill_between(x=range(this_dat[ax_num].shape[-1]),
            y1 = mean_this_dat[ax_num] - std_this_dat[ax_num],
            y2= mean_this_dat[ax_num] + std_this_dat[ax_num],
            alpha = 0.5,color='orange')
    this_ax.plot(range(len(mean_this_dat[ax_num])),mean_this_dat[ax_num])
this_dat = affective_lfp_amplitude_down 
mean_this_dat = np.mean(this_dat,axis=1)
std_this_dat = np.std(this_dat,axis=1)
for ax_num,this_ax in enumerate(ax[:,0]):
    this_ax.fill_between(x=range(this_dat[ax_num].shape[-1]),
            y1 = mean_this_dat[ax_num] - std_this_dat[ax_num],
            y2= mean_this_dat[ax_num] + std_this_dat[ax_num],
            alpha = 0.5,color='orange')
    this_ax.plot(range(len(mean_this_dat[ax_num])),mean_this_dat[ax_num])
ax[0,0].title.set_text('Affective')
ax[0,1].title.set_text('Taste ITI')
plt.show()



#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs


# Mean band spectrograms
# Average spectrogram power within single bands
mean_band_spectrograms = np.asarray(\
        [np.mean(mean_iti_spectrograms[:,(f_iti>band[0])*(f_iti<band[1]),:],axis=1) \
        for band in band_freqs])

# Check for outliers
zscore_trials_power = np.asarray([\
        zscore(band,axis = None) for band in mean_band_spectrograms])
mean_trials_power = np.mean(zscore_trials_power,axis=-1)
plt.plot(mean_trials_power.T,'x');plt.show()

bad_trial_threshold = 0.5 
bad_trials = np.unique(np.where(mean_trials_power > bad_trial_threshold)[1])


# Plot all bands for all 120 trials
fig,ax = plt.subplots(1,len(mean_band_spectrograms),sharey=True)
num_std = 3
for ax_num,this_ax in enumerate(ax):
    prio_mean = np.mean(mean_band_spectrograms[ax_num],axis=None)
    prio_std= np.std(mean_band_spectrograms[ax_num],axis=None)
    this_dat = zscore(np.ma.masked_greater(mean_band_spectrograms[ax_num],
                                    prio_mean + num_std*prio_std),axis=None)
    mean = np.mean(this_dat,axis=None)
    std= np.std(this_dat,axis=None)
    im = this_ax.pcolormesh(this_dat,
            vmin = mean-num_std*std, vmax = mean+num_std*std, cmap = 'viridis')
    #this_ax.plot(np.mean(this_dat,axis=-1),np.arange(this_dat.shape[0]),c='r')
    im.cmap.set_over('k')
fig.subplots_adjust(bottom = 0.2)
cbar_ax = fig.add_axes([0.15,0.1,0.7,0.02])
plt.colorbar(im, cax = cbar_ax,orientation = 'horizontal', pad = 0.2, extend='max')
plt.show()

# Average spectrogram for (taste x band)
taste_band_spectrograms = np.asarray([\
        mean_band_spectrograms[:,delivery_times.taste == taste,:] \
        for taste in np.sort(delivery_times.taste.unique())])
# Set points with values > 3*std as masked
masked_taste_band_spectrograms = np.asarray(\
        [np.ma.masked_greater(taste_band_spectrograms[:,band],
            np.mean(taste_band_spectrograms[:,band],axis=None)+3*\
                    np.std(taste_band_spectrograms[:,band],axis=None))\
        for band in range(taste_band_spectrograms.shape[1])]).swapaxes(0,1)
zscore_taste_band_spectrograms = np.asarray(\
        [zscore(masked_taste_band_spectrograms[:,band],axis=None) \
        for band in range(masked_taste_band_spectrograms.shape[1])]).swapaxes(0,1)

fig,ax = plt.subplots(taste_band_spectrograms.shape[0],
                        taste_band_spectrograms.shape[1])
for taste in range(taste_band_spectrograms.shape[0]):
    for band in range(taste_band_spectrograms.shape[1]):
        plt.sca(ax[taste,band])
        im = plt.imshow(zscore_taste_band_spectrograms[taste,band],
                interpolation='nearest',aspect='auto',vmin=-1,vmax=1)
        im.cmap.set_over('k')
fig.subplots_adjust(bottom = 0.2)
cbar_ax = fig.add_axes([0.15,0.1,0.7,0.02])
plt.colorbar(im, cax = cbar_ax,orientation = 'horizontal', pad = 0.2, extend='max')
plt.show()

#    _    _   _  _____     ___    
#   / \  | \ | |/ _ \ \   / / \   
#  / _ \ |  \| | | | \ \ / / _ \  
# / ___ \| |\  | |_| |\ V / ___ \ 
#/_/   \_\_| \_|\___/  \_/_/   \_\
#                                 

# Convert power from spectrogram to DataFrame
nd_idx = make_array_identifiers(mean_band_spectrograms)
mean_band_df = pd.DataFrame({\
        'band' : nd_idx[0],
        'chronological' : nd_idx[1],
        'time_bin' : nd_idx[2],
        'power' : mean_band_spectrograms.flatten()})
mean_band_df = mean_band_df.merge(delivery_times,'inner')

# Remove "bad trials"
mean_band_df = mean_band_df.loc[~mean_band_df.chronological.isin(bad_trials)]


# Find average power across ITI interval
#taste_band_avg_power = np.mean(zscore_taste_band_spectrograms,axis=-1)
#nd_idx_objs = make_array_identifiers(taste_band_avg_power)
#mean_band_df = pd.DataFrame({\
#                        'taste' : nd_idx_objs[0],
#                        'band' : nd_idx_objs[1],
#                        'trial' : nd_idx_objs[2],
#                        'power' : taste_band_avg_power.flatten()})


mean_trial_df = mean_band_df.groupby(['chronological','band']).aggregate('mean').reset_index()
mean_trial_df = mean_trial_df.merge(delivery_times[['chronological','trial']],'inner')

# Cluster trials into bins for anova
trial_bin_num = 5
mean_trial_df['trial_bin'] = pd.cut(mean_trial_df.trial,
        bins = trial_bin_num, include_lowest = True, labels = range(trial_bin_num))
mean_trial_df['trial_bin'] = mean_trial_df['trial_bin'].astype(np.dtype('int16')) 

mean_trial_df = mean_trial_df.astype('float')

# Plot dataframe to visualize
g = sns.FacetGrid(data = \
            mean_trial_df, col = 'band', hue = 'taste', sharey=False)
g.map(sns.pointplot, 'trial_bin','power',ci='sd').add_legend()
plt.show()


# Perform 2-way ANOVA to look at differences in taste and trial_bin
taste_trial_anova =\
    [mean_trial_df.loc[mean_trial_df.band == band_num].anova(dv = 'power', \
            between= ['trial_bin','taste'])[['Source','p-unc','np2']] \
            for band_num in np.sort(mean_trial_df.band.unique())]

#
