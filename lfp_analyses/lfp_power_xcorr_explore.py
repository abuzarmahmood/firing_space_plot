#import stuff
import os
import scipy
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt
import theano

%matplotlib inline
from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt
plt.set_cmap('viridis')
from matplotlib.lines import Line2D
from tqdm import tqdm
from pingouin import mwu,kruskal, read_dataset
import pandas as pd
import seaborn as sns

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

data_dir = '/media/bigdata/Abuzar_Data/AM12/AM12_4Tastes_191106_085215/'
dat = \
            ephys_data(data_dir)
dat.firing_rate_params = dat.default_firing_params
# dat.firing_rate_params['type'] = 'baks'
# dat.firing_rate_params['baks_resolution'] = 10e-3

dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()
dat.get_stft()

#visualize.firing_overview(dat.all_normalized_firing);
#print(dat.all_normalized_firing.shape)

median_amplitude = np.median(dat.amplitude_array,axis=(0,2))
visualize.firing_overview(stats.zscore(median_amplitude,axis=-1));

dat.get_lfp_electrodes()
gc_electrodes = dat.lfp_region_electrodes\
        [[num for num,x in enumerate(dat.region_names) if x=='gc'][0]]
visualize.firing_overview(stats.zscore(median_amplitude[gc_electrodes],axis=-1));


# if BLA present, plot BLA and GC spectra side-by-side
if 'bla' in dat.region_names:
        bla_electrodes = dat.lfp_region_electrodes[[num for num,x in enumerate(dat.region_names) if x=='bla'][0]]
            visualize.firing_overview(stats.zscore(median_amplitude[bla_electrodes],axis=-1));

# Remove trials with artifcats
## ** Instead of using just GC power, use power from ALL channels to 
## ** remove trials with artifacts

time_lims = [2000,4000]
channel = 0
this_amp_array = dat.amplitude_array[:,gc_electrodes[channel]]
# Cut by time_lims, smaller period means lower chance for artifacts
# This is dangerous in general though since pre and post stim time for stft is not properly accessible
this_amp_array = this_amp_array[...,time_lims[0]:time_lims[1]]
this_amp_array_long = np.reshape(this_amp_array,(-1,*this_amp_array.shape[2:]))

# Concatenate time trials along time axis for pca
this_amp_array_very_long = np.reshape(np.swapaxes(this_amp_array_long,0,1),(this_amp_array.shape[2],-1))
plt.figure(figsize=(15,5))
thresh = 3
plt.imshow(stats.zscore(this_amp_array_very_long,axis=-1),aspect='auto',vmin = -thresh, vmax = thresh, origin='lower', cmap='viridis');
plt.figure()
print(this_amp_array_very_long.shape)
plt.plot(this_amp_array_very_long[1])

# Find median absolute deivation for each frequency
# And remove trials with power higher than those
freq_med = np.median(this_amp_array_very_long,axis=-1)
MAD = np.median(np.abs((this_amp_array_very_long - freq_med[:,np.newaxis])),axis=-1)
print(MAD.shape)
print(freq_med)
print(MAD)

# Iterate through trials and remove ones outside thresholds
threshold = 50 # MADs from the median
lower_lim = freq_med - (threshold * MAD)
upper_lim = freq_med + (threshold * MAD)
freq = 4
plt.plot(this_amp_array_very_long[freq])
plt.ylim((0,upper_lim[freq]))

print(lower_lim)
outlier_trial_set = set()
for num, this_freq in enumerate(this_amp_array_long.swapaxes(0,1)):
    this_freq = this_amp_array_long[:,num]
    this_ll,this_ul = lower_lim[num],upper_lim[num]
    outlier_inds = np.where(this_freq > this_ul)
    outlier_trial_set = outlier_trial_set.union(set(outlier_inds[0]))

#clean_trial_inds = [x for x in np.arange(this_amp_array_long.shape[0]) if x not in outlier_trial_set]
#clean_this_amp_long = this_amp_array_long[np.array(clean_trial_inds)]
#print(clean_this_amp_long.shape)
#clean_amp_very_long = np.reshape(clean_this_amp_long.swapaxes(0,1),(len(dat.freq_vec),-1))
#print(clean_amp_very_long.shape)
#plt.figure(figsize=(15,5))
## thresh = 5
#plt.imshow(stats.zscore(clean_amp_very_long,axis=-1),aspect='auto',origin='lower',cmap='viridis');

this_bla_amp_array = dat.amplitude_array[:,bla_electrodes[channel]]
this_bla_amp_array = this_bla_amp_array[...,time_lims[0]:time_lims[1]]
print(this_bla_amp_array.shape)
bla_amp_long = np.reshape(this_bla_amp_array,(-1,*this_bla_amp_array.shape[2:]))
print(bla_amp_long.shape)
bla_amp_clean = bla_amp_long[clean_trial_inds]
print(bla_amp_clean.shape)
bla_zscore_amp_long = stats.zscore(bla_amp_clean,axis=-1)

# Plot GC and BLA activity together
trial_count = 30
trial_inds = np.sort(np.random.choice(np.arange(clean_dat_binned_long.shape[0]),trial_count, replace = False))

fig, ax = plt.subplots(trial_count*2, figsize = (5,3*trial_count))
for num,trial in enumerate(trial_inds):
    ax[num*2].imshow(clean_this_amp_long[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num*2+1].imshow(bla_amp_clean[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num*2].set_ylabel(num)
    ax[num*2+1].set_ylabel(num)
    # ax[0,0].set_title('GC')
    # ax[0,1].set_title('BLA')

fig,ax = plt.subplots(2,1,figsize = (10,10))
ax[0].imshow(stats.zscore(np.median(clean_this_amp_long,axis=0),axis=-1),aspect='auto',cmap='jet')
ax[1].imshow(stats.zscore(np.median(bla_amp_clean,axis=0),axis=-1),aspect='auto',cmap='jet')
ax[0].set_title('GC')
ax[1].set_title('BLA')

#__  ______                
#\ \/ / ___|___  _ __ _ __ 
# \  / |   / _ \| '__| '__|
# /  \ |__| (_) | |  | |   
#/_/\_\____\___/|_|  |_|   
#                          

# Perform zero-lag cross-correlation on single trials and shuffled trials
# Use normalized cross-correlation to remove amplitude effects

def norm_zero_lag_xcorr(vec1, vec2):
    """
    Calculates normalized zero-lag cross correlation
    Returns a single number
    """
    auto_v1 = np.sum(vec1**2,axis=-1)
    auto_v2 = np.sum(vec2**2,axis=-1)
    xcorr = np.sum(vec1 * vec2,axis=-1)
    denom = np.sqrt(np.multiply(auto_v1,auto_v2))
    return np.divide(xcorr, denom)

# XCorr TESTS
#================
# Testing function
x = np.linspace(0,10,1000)
y1 = np.array([np.sin(x*a) for a in range(1,10)])
y2 = y1/2
y3 = -y1
y4 = np.cos(x)
y5 = y2
y5[3] = np.ones(y5[3].shape)

plt.figure(figsize(15,5))
visualize.imshow(y1)

print(norm_zero_lag_xcorr(y1,y2))
print(norm_zero_lag_xcorr(y1,y3))
print(norm_zero_lag_xcorr(y1,y4))
print(norm_zero_lag_xcorr(y1,y5))

# Perform XCorr
# =============
xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(clean_this_amp_long,bla_amp_clean)])
resamples = clean_this_amp_long.shape[0]
print(resamples)
inds1 = np.random.choice(np.arange(clean_this_amp_long.shape[0]),size=resamples)
inds2 = np.random.choice(np.arange(clean_this_amp_long.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
shuffled_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) \
                                    for v1,v2 in zip(clean_this_amp_long[inds1],bla_amp_clean[inds2])])
mean_xcorrs = np.mean(xcorrs,axis=0)
std_xcorrs = np.std(xcorrs,axis=0)
mean_shuffle_xcorrs = np.mean(shuffled_xcorrs,axis=0)
std_shuffle_xcorrs = np.std(shuffled_xcorrs,axis=0)
plt.errorbar(dat.freq_vec,mean_xcorrs,std_xcorrs, label = 'actual',marker = 'o')
plt.errorbar(dat.freq_vec,mean_shuffle_xcorrs,std_shuffle_xcorrs, label = 'shuffle',marker = 'o')
plt.legend();

print(xcorrs.shape)
print(shuffled_xcorrs.shape)

# Perform xcorr between pairs of channels within BLA and GC for reference
channel2 = channel+1

this_gc_amp_array2 = dat.amplitude_array[:,gc_electrodes[channel2]]
this_gc_amp_array2 = this_gc_amp_array2[...,time_lims[0]:time_lims[1]]
print(this_gc_amp_array2.shape)
gc_amp_long2 = np.reshape(this_gc_amp_array2,(-1,*this_gc_amp_array2.shape[2:]))
print(gc_amp_long2.shape)
gc_amp_clean2 = gc_amp_long2[clean_trial_inds]
print(gc_amp_clean2.shape)

this_bla_amp_array2 = dat.amplitude_array[:,bla_electrodes[channel2]]
this_bla_amp_array2 = this_bla_amp_array2[...,time_lims[0]:time_lims[1]]
print(this_bla_amp_array2.shape)
bla_amp_long2 = np.reshape(this_bla_amp_array2,(-1,*this_bla_amp_array2.shape[2:]))
print(bla_amp_long2.shape)
bla_amp_clean2 = bla_amp_long2[clean_trial_inds]
print(bla_amp_clean2.shape)

# Plot pairs of electrodes side by side to VISUALIZE similarity
# Overlay raster with CDF of switchpoints
trial_count = 30
trial_inds = np.sort(np.random.choice(np.arange(clean_dat_binned_long.shape[0]),trial_count, replace = False))

fig, ax = plt.subplots(trial_count, 4, figsize = (15,3*trial_count))
for num,trial in enumerate(trial_inds):
    ax[num,0].imshow(clean_this_amp_long[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,1].imshow(gc_amp_clean2[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,2].imshow(bla_amp_clean[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,3].imshow(bla_amp_clean2[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,0].set_ylabel(num)

# Intra-GC xcorrs
gc_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(clean_this_amp_long,gc_amp_clean2)])
resamples = clean_this_amp_long.shape[0]
print(resamples)
inds1 = np.random.choice(np.arange(clean_this_amp_long.shape[0]),size=resamples)
inds2 = np.random.choice(np.arange(clean_this_amp_long.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
gc_shuffled_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) \
                                    for v1,v2 in zip(clean_this_amp_long[inds1],gc_amp_clean2[inds2])])
gc_mean_xcorrs = np.mean(gc_xcorrs,axis=0)
gc_std_xcorrs = np.std(gc_xcorrs,axis=0)
gc_mean_shuffle_xcorrs = np.mean(gc_shuffled_xcorrs,axis=0)
gc_std_shuffle_xcorrs = np.std(gc_shuffled_xcorrs,axis=0)

# Intra BLA xcorrs
bla_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(bla_amp_clean,bla_amp_clean2)])
resamples = bla_amp_clean.shape[0]
print(resamples)
inds1 = np.random.choice(np.arange(bla_amp_clean.shape[0]),size=resamples)
inds2 = np.random.choice(np.arange(bla_amp_clean.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
bla_shuffled_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) \
                                    for v1,v2 in zip(bla_amp_clean[inds1],bla_amp_clean2[inds2])])
bla_mean_xcorrs = np.mean(bla_xcorrs,axis=0)
bla_std_xcorrs = np.std(bla_xcorrs,axis=0)
bla_mean_shuffle_xcorrs = np.mean(bla_shuffled_xcorrs,axis=0)
bla_std_shuffle_xcorrs = np.std(bla_shuffled_xcorrs,axis=0)

# Plot cross-region with w/i region
fig,ax = plt.subplots(1,3,sharey=True, figsize = (15,5))
print(mean_xcorrs.shape)
ax[0].errorbar(dat.freq_vec,mean_xcorrs,std_xcorrs, label = 'actual',marker = 'o')
ax[0].errorbar(dat.freq_vec,mean_shuffle_xcorrs,std_shuffle_xcorrs, label = 'shuffle',marker = 'o')
ax[0].set_title('BLA-GC xcorr')
plt.legend();

ax[1].errorbar(dat.freq_vec,gc_mean_xcorrs,gc_std_xcorrs, label = 'actual',marker = 'o')
ax[1].errorbar(dat.freq_vec,gc_mean_shuffle_xcorrs,gc_std_shuffle_xcorrs, label = 'shuffle',marker = 'o')
ax[1].set_title('GC-GC xcorr')
plt.legend();

ax[2].errorbar(dat.freq_vec,bla_mean_xcorrs,bla_std_xcorrs, label = 'actual',marker = 'o')
ax[2].errorbar(dat.freq_vec,bla_mean_shuffle_xcorrs,bla_std_shuffle_xcorrs, label = 'shuffle',marker = 'o')
ax[2].set_title('BLA-BLA xcorr')
plt.legend();

print(gc_xcorrs.shape)
print(gc_shuffled_xcorrs.shape)

# Not that great
# It seems like we need to do rolling zscoring to capture local changes rather than over the entire trial

# ========================================
# ____       _ _ _               _____    ____                     
#|  _ \ ___ | | (_)_ __   __ _  |__  /   / ___|  ___ ___  _ __ ___ 
#| |_) / _ \| | | | '_ \ / _` |   / /____\___ \ / __/ _ \| '__/ _ \
#|  _ < (_) | | | | | | | (_| |  / /|_____|__) | (_| (_) | | |  __/
#|_| \_\___/|_|_|_|_| |_|\__, | /____|   |____/ \___\___/|_|  \___|
#                        |___/                                     
# ========================================

# Test signal
x_vec = np.linspace(0,50,1000)
x = np.array([x_vec*c for c in range(1,10)])
print(x.shape)
y = np.sin(x)
a = x/np.max(x)

y_a = a*y
fig,ax = plt.subplots(4,1,figsize=(10,10));
ax[0].imshow(x,aspect='auto')
ax[1].imshow(y,aspect='auto')
ax[2].imshow(y_a,aspect='auto')

def rolling_zscore(array, window_size):
    out = np.zeros(array.shape)
    starts = np.arange((array.shape[-1] - window_size))
    inds = list(zip(starts,starts+window_size))
    for this_ind in inds:
        out[...,this_ind[0]:this_ind[1]] += \
                stats.zscore(array[...,this_ind[0]:this_ind[1]],axis=-1)
    return out/window_size

# Good in theory but blow up memory for large arrays
#def rolling_zscore(array, window_size):
#    out = np.zeros(array.shape)
#    starts = np.arange((array.shape[-1] - window_size))
#    inds = list(zip(starts,starts+window_size))
#    zscore_temp_array = stats.zscore(\
#            [array[...,this_ind[0]:this_ind[1]] for this_ind in inds], axis=-1)
#    for this_dat, this_ind in zip(zscore_temp_array, inds):
#        out[..., this_ind[0]:this_ind[1]] += this_dat
#    return out/window_size

ax[3].imshow(rolling_zscore(y_a,100),aspect='auto');

# Plot rolling zscored spectrograms
trial_count = 20
trial_inds = np.sort(np.random.choice(np.arange(clean_dat_binned_long.shape[0]),trial_count, replace = False))
rolling_size = 500

fig, ax = plt.subplots(trial_count*2, figsize = (5,3*trial_count))
for num,trial in enumerate(trial_inds):
    ax[num*2].imshow(rolling_zscore(clean_this_amp_long[trial],rolling_size),aspect='auto',origin='lower')
    ax[num*2+1].imshow(rolling_zscore(bla_amp_clean[trial],rolling_size),aspect='auto',origin='lower')
    ax[num*2].set_ylabel(num)
    ax[num*2+1].set_ylabel(num)

# Perform rolling zscoring on spectrograms for both bla and gc
amp_array_long = np.swapaxes(dat.amplitude_array,1,2)
amp_array_long = np.reshape(amp_array_long,(-1,*amp_array_long.shape[2:]))
print(amp_array_long.shape)
channel = 0
gc_amp = amp_array_long[:,gc_electrodes[channel]]
gc_clean_amp = gc_amp[clean_trial_inds]
bla_amp = amp_array_long[:,bla_electrodes[channel]]
bla_clean_amp = bla_amp[clean_trial_inds]
print(gc_clean_amp.shape)

window_size = 500
rzscore_gc_clean = np.array([rolling_zscore(x,window_size) for x in tqdm(gc_clean_amp)])
rzscore_bla_clean = np.array([rolling_zscore(x,window_size) for x in tqdm(bla_clean_amp)])
print(rzscore_gc_clean.shape)

rzscore_gc_cut = rzscore_gc_clean[...,time_lims[0]:time_lims[1]]
rzscore_bla_cut = rzscore_bla_clean[...,time_lims[0]:time_lims[1]]

freq_vec = np.vectorize(np.int)(dat.freq_vec)[:-2]
freq_tick_labels = np.arange(freq_vec.min(),freq_vec.max(),3)
print(freq_tick_labels)
freq_ticks = np.linspace(0,len(freq_vec),len(freq_tick_labels))
print(freq_ticks)

# Plot rolling zscored spectrograms
trial_count = 10

trial_inds = np.sort(np.random.choice(np.arange(clean_dat_binned_long.shape[0]),trial_count, replace = False))
plt.set_cmap('viridis')
fig, ax = plt.subplots(trial_count*2, figsize = (5,3*trial_count))
for num,trial in enumerate(trial_inds):
    ax[num*2].imshow(rzscore_gc_cut[trial],aspect='auto',origin='lower')
    ax[num*2+1].imshow(rzscore_bla_cut[trial],aspect='auto',origin='lower')

    ax[num*2].text(1.02,0.5,f'GC Trial {num}',rotation=270,verticalalignment='center', transform=ax[num*2].transAxes)
    ax[num*2+1].text(1.02,0.5,f'BLA Trial {num}',rotation=270,verticalalignment='center', transform=ax[num*2+1].transAxes)

    ax[num*2].set_yticks(ticks=freq_ticks)
    ax[num*2+1].set_yticks(ticks=freq_ticks)
    ax[num*2].set_yticklabels(labels = freq_tick_labels)
    ax[num*2+1].set_yticklabels(labels = freq_tick_labels)

    ax[num*2].set_ylabel('Freq (Hz)')
    ax[num*2+1].set_ylabel('Freq (Hz)')
    ax[num*2].set_xlabel('Time post-stimulus delivery (ms)')
    ax[num*2+1].set_xlabel('Time post-stimulus delivery (ms)')
plt.tight_layout()

# Compare regular XCorrs with Rolling Z-scored ones
r_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(rzscore_gc_cut,rzscore_bla_cut)])
resamples = rzscore_gc_cut.shape[0]
r_inds1 = np.random.choice(np.arange(rzscore_gc_cut.shape[0]),size=resamples)
r_inds2 = np.random.choice(np.arange(rzscore_gc_cut.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
r_shuffled_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) \
                                    for v1,v2 in zip(rzscore_gc_cut[inds1],rzscore_bla_cut[inds2])])
r_mean_xcorrs = np.mean(r_xcorrs,axis=0)
r_std_xcorrs = np.std(r_xcorrs,axis=0)
r_sem_xcorrs = r_std_xcorrs/np.sqrt(r_xcorrs.shape[0])
r_mean_shuffle_xcorrs = np.mean(r_shuffled_xcorrs,axis=0)
r_std_shuffle_xcorrs = np.std(r_shuffled_xcorrs,axis=0)
r_sem_shuffle_xcorrs = r_std_shuffle_xcorrs/np.sqrt(r_shuffled_xcorrs.shape[0])

print(r_xcorrs.shape)
print(r_shuffled_xcorrs.shape)

freq_vec = np.vectorize(np.int)(dat.freq_vec)
freq_tick_labels = freq_vec
print(freq_tick_labels)
freq_ticks = np.linspace(0,len(freq_vec),len(freq_tick_labels))
print(freq_ticks)

fig,ax = plt.subplots(2,1,figsize = (10,10))
ax[0].errorbar(dat.freq_vec,mean_xcorrs,std_xcorrs, label = 'Matched',marker = 'o')
ax[0].errorbar(dat.freq_vec,mean_shuffle_xcorrs,std_shuffle_xcorrs, label = 'Shuffled',marker = 'o')
ax[0].set_title('Raw')
ax[0].set_ylabel('Average Cross-correlation +/- SEM (A.U.)')
ax[0].set_xlabel('Frequency (Hz)')
# ax[1].errorbar(dat.freq_vec,r_mean_xcorrs,r_sem_xcorrs, label = 'Matched',marker = 'o')
# ax[1].errorbar(dat.freq_vec,r_mean_shuffle_xcorrs,r_sem_shuffle_xcorrs, label = 'Shuffled',marker = 'o')
ax[1].fill_between(dat.freq_vec,r_mean_xcorrs - r_sem_xcorrs, r_mean_xcorrs + r_sem_xcorrs,alpha = 0.5)
ax[1].fill_between(dat.freq_vec,r_mean_shuffle_xcorrs - r_sem_shuffle_xcorrs,r_mean_shuffle_xcorrs + r_sem_shuffle_xcorrs,alpha = 0.5)
ax[1].plot(dat.freq_vec,r_mean_xcorrs,label = 'Matched')
ax[1].plot(dat.freq_vec,r_mean_shuffle_xcorrs,label = 'Shuffled')
ax[1].set_title('Rolling Z-Scored')
ax[1].set_ylabel('Average Cross-correlation +/- SEM (A.U.)')
ax[1].set_xlabel('Frequency (Hz)')

ax[0].set_xticks(ticks=freq_tick_labels)
ax[1].set_xticks(ticks=freq_tick_labels)
ax[0].set_xticklabels(labels = freq_tick_labels)
ax[1].set_xticklabels(labels = freq_tick_labels)
plt.legend();
plt.tight_layout()

# Trials ranked by xcorrs from plotting
xcorr_sort = np.argsort(np.mean(r_xcorrs,axis=1))[::-1]

# Plot top 5 trials
# Plot rolling zscored spectrograms
trial_count = 10

trial_inds = xcorr_sort[:trial_count]
plt.set_cmap('viridis')
fig, ax = plt.subplots(trial_count*2, figsize = (5,4*trial_count))
for num,trial in enumerate(trial_inds):
    ax[num*2].imshow(rzscore_gc_cut[trial],aspect='auto',origin='lower')
    ax[num*2+1].imshow(rzscore_bla_cut[trial],aspect='auto',origin='lower')

    ax[num*2].text(1.02,0.5,f'GC Trial {num}',rotation=270,verticalalignment='center', transform=ax[num*2].transAxes, size='large')
    ax[num*2+1].text(1.02,0.5,f'BLA Trial {num}',rotation=270,verticalalignment='center', transform=ax[num*2+1].transAxes, size='large')

    ax[num*2].set_yticks(ticks=freq_ticks)
    ax[num*2+1].set_yticks(ticks=freq_ticks)
    ax[num*2].set_yticklabels(labels = freq_tick_labels)
    ax[num*2+1].set_yticklabels(labels = freq_tick_labels)

    ax[num*2].set_ylabel('Freq (Hz)')
    ax[num*2+1].set_ylabel('Freq (Hz)')
    ax[num*2].set_xlabel('Time post-stimulus delivery (ms)')
    ax[num*2+1].set_xlabel('Time post-stimulus delivery (ms)')
plt.tight_layout()

# Perform xcorr between pairs of channels within BLA and GC for reference

gc_amp2 = amp_array_long[:,gc_electrodes[channel2]]
gc_clean_amp2 = gc_amp2[clean_trial_inds]
bla_amp2 = amp_array_long[:,bla_electrodes[channel2]]
bla_clean_amp2 = bla_amp2[clean_trial_inds]

rzscore_gc_clean2 = np.array([rolling_zscore(x,window_size) for x in tqdm(gc_clean_amp2)])
rzscore_bla_clean2 = np.array([rolling_zscore(x,window_size) for x in tqdm(bla_clean_amp2)])

rzscore_gc_cut2 = rzscore_gc_clean2[...,time_lims[0]:time_lims[1]]
rzscore_bla_cut2 = rzscore_bla_clean2[...,time_lims[0]:time_lims[1]]
print(rzscore_gc_clean2.shape)
print(rzscore_gc_cut2.shape)

# Plot pairs of electrodes side by side to VISUALIZE similarity
# Overlay raster with CDF of switchpoints
trial_count = 30
trial_inds = np.sort(np.random.choice(np.arange(rzscore_gc_clean2.shape[0]),trial_count, replace = False))

fig, ax = plt.subplots(trial_count, 4, figsize = (15,3*trial_count))
for num,trial in enumerate(trial_inds):
    ax[num,0].imshow(rzscore_gc_cut[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,1].imshow(rzscore_gc_cut2[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,2].imshow(rzscore_bla_cut[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,3].imshow(rzscore_bla_cut2[trial],interpolation='nearest',aspect='auto',origin='lower')
    ax[num,0].set_ylabel(num)

# Intra-GC xcorrs
r_gc_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(rzscore_gc_cut,rzscore_gc_cut2)])
resamples = rzscore_gc_cut.shape[0]
print(resamples)
inds1 = np.random.choice(np.arange(rzscore_gc_cut.shape[0]),size=resamples)
inds2 = np.random.choice(np.arange(rzscore_gc_cut.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
r_gc_shuffled_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) \
                                    for v1,v2 in zip(rzscore_gc_cut[inds1],rzscore_gc_cut2[inds2])])
r_gc_mean_xcorrs = np.mean(r_gc_xcorrs,axis=0)
r_gc_std_xcorrs = np.std(r_gc_xcorrs,axis=0)
r_gc_mean_shuffle_xcorrs = np.mean(r_gc_shuffled_xcorrs,axis=0)
r_gc_std_shuffle_xcorrs = np.std(r_gc_shuffled_xcorrs,axis=0)

# Intra-BLA xcorrs
r_bla_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(rzscore_bla_cut,rzscore_bla_cut2)])
resamples = rzscore_bla_cut.shape[0]
print(resamples)
inds1 = np.random.choice(np.arange(rzscore_bla_cut.shape[0]),size=resamples)
inds2 = np.random.choice(np.arange(rzscore_bla_cut.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
r_bla_shuffled_xcorrs = np.array([norm_zero_lag_xcorr(v1,v2) \
                                    for v1,v2 in zip(rzscore_bla_cut[inds1],rzscore_bla_cut2[inds2])])
r_bla_mean_xcorrs = np.mean(r_bla_xcorrs,axis=0)
r_bla_std_xcorrs = np.std(r_bla_xcorrs,axis=0)
r_bla_mean_shuffle_xcorrs = np.mean(r_bla_shuffled_xcorrs,axis=0)
r_bla_std_shuffle_xcorrs = np.std(r_bla_shuffled_xcorrs,axis=0)

# Plot cross-region with w/i region
fig,ax = plt.subplots(1,3,sharey=True, figsize = (15,5))
print(mean_xcorrs.shape)
ax[0].errorbar(dat.freq_vec,r_mean_xcorrs,r_std_xcorrs, label = 'actual',marker = 'o')
ax[0].errorbar(dat.freq_vec,r_mean_shuffle_xcorrs,r_std_shuffle_xcorrs, label = 'shuffle',marker = 'o')
ax[0].set_title('BLA-GC xcorr')
plt.legend();

ax[1].errorbar(dat.freq_vec,r_gc_mean_xcorrs,r_gc_std_xcorrs, label = 'actual',marker = 'o')
ax[1].errorbar(dat.freq_vec,r_gc_mean_shuffle_xcorrs,r_gc_std_shuffle_xcorrs, label = 'shuffle',marker = 'o')
ax[1].set_title('GC-GC xcorr')
plt.legend();


ax[2].errorbar(dat.freq_vec,r_bla_mean_xcorrs,r_bla_std_xcorrs, label = 'actual',marker = 'o')
ax[2].errorbar(dat.freq_vec,r_bla_mean_shuffle_xcorrs,r_bla_std_shuffle_xcorrs, label = 'shuffle',marker = 'o')
ax[2].set_title('BLA-BLA xcorr')
plt.legend();

print(gc_xcorrs.shape)
print(gc_shuffled_xcorrs.shape)

# Perform Kruskal-Wallis to check for difference b/w actual and shuffle (in both raw and r-zscored)

# Raw
actual_inds = np.array(list(np.ndindex(xcorrs.shape)))
print(f'actual_inds_shape : {actual_inds.shape}')
xcorr_frame = pd.DataFrame({
                    'trials':actual_inds[:,0],
                    'freqs' : actual_inds[:,1],
                    'corr' : xcorrs.flatten(),
                    'class': ['actual']*actual_inds.shape[0]
                    })
shuffle_inds = np.array(list(np.ndindex(shuffled_xcorrs.shape)))
xcorr_frame = xcorr_frame.append(pd.DataFrame({
                    'trials': shuffle_inds[:,0],
                    'freqs' : shuffle_inds[:,1],
                    'corr' : shuffled_xcorrs.flatten(),
                    'class': ['shuffle']*shuffle_inds.shape[0]
                    }))
# print(xcorr_frame)
print(kruskal(data=xcorr_frame, dv='corr', between='class'))

## Also perform pairwise Mann-Whitney Test
mwu_tests = pd.concat([mwu(x, y, tail='two-sided') for x,y in zip(xcorrs.T, shuffled_xcorrs.T)])
print(mwu_tests)

# ==============================================================#
# Rolling z-score
r_actual_inds = np.array(list(np.ndindex(r_xcorrs.shape)))
print(r_actual_inds.shape)
rz_xcorr_frame = pd.DataFrame({
                    'trials':r_actual_inds[:,0],
                    'freqs' : r_actual_inds[:,1],
                    'corr' : r_xcorrs.flatten(),
                    'class': ['actual']*r_actual_inds.shape[0]
                    })
r_shuffle_inds = np.array(list(np.ndindex(r_shuffled_xcorrs.shape)))
rz_xcorr_frame = rz_xcorr_frame.append(pd.DataFrame({
                    'trials': r_shuffle_inds[:,0],
                    'freqs' : r_shuffle_inds[:,1],
                    'corr' : r_shuffled_xcorrs.flatten(),
                    'class': ['shuffle']*r_shuffle_inds.shape[0]
                    }))
# print(rz_xcorr_frame)
print(kruskal(data=rz_xcorr_frame, dv='corr', between='class'))

## Also perform pairwise Mann-Whitney Test
mwu_tests = pd.concat([mwu(x, y, tail='two-sided') for x,y in zip(r_xcorrs.T, r_shuffled_xcorrs.T)])
print(mwu_tests)

# Calculate effect-size of difference for each frequency
# This will allow us to plot effect size distributions for all 
# electrode pairs

# Plot variance to visualize how different they are for Cohen's D SD pooling
fig,ax = plt.subplots(1,3,sharey=True, figsize = (15,5))
ax[0].plot(dat.freq_vec,r_std_xcorrs)
ax[0].plot(dat.freq_vec,r_std_shuffle_xcorrs)

ax[1].plot(dat.freq_vec,r_gc_std_xcorrs)
ax[1].plot(dat.freq_vec,r_gc_std_shuffle_xcorrs)

ax[2].plot(dat.freq_vec,r_bla_std_xcorrs)
ax[2].plot(dat.freq_vec,r_bla_std_shuffle_xcorrs)

# We'll ignore testing variance for now
pooled_sd_xcorr = np.sqrt(np.mean(r_std_xcorrs**2 + r_std_shuffle_xcorrs**2))
pooled_gc_sd_xcorr = np.sqrt(np.mean(r_gc_std_xcorrs**2 + r_gc_std_shuffle_xcorrs**2))
pooled_bla_sd_xcorr = np.sqrt(np.mean(r_bla_std_xcorrs**2 + r_bla_std_shuffle_xcorrs**2))

xcorr_effect = np.abs(r_mean_xcorrs - r_mean_shuffle_xcorrs)/pooled_sd_xcorr
gc_xcorr_effect = np.abs(r_gc_mean_xcorrs - r_gc_mean_shuffle_xcorrs)/pooled_sd_xcorr
bla_xcorr_effect = np.abs(r_bla_mean_xcorrs - r_bla_mean_shuffle_xcorrs)/pooled_sd_xcorr

plt.figure(figsize=(5,5))
plt.plot(dat.freq_vec,xcorr_effect,label='bla-gc')
plt.plot(dat.freq_vec,gc_xcorr_effect,label='gc-gc')
plt.plot(dat.freq_vec,bla_xcorr_effect,label='bla-bla')
plt.legend();

# Chop trial into 500ms bins and perform xcorr on each bin separately
splits = clean_this_amp_long.shape[-1]//500
gc_amp_split = np.array(np.split(clean_this_amp_long,splits,axis=-1))
bla_amp_split = np.array(np.split(bla_amp_clean,splits,axis=-1))
xcorrs = np.array([[norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(this_gc,this_bla)]\
                         for this_gc,this_bla in zip(gc_amp_split,bla_amp_split)])
print(xcorrs.shape)

resamples = 1000
inds1 = np.random.choice(np.arange(clean_this_amp_long.shape[0]),size=resamples)
inds2 = np.random.choice(np.arange(clean_this_amp_long.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
shuffled_xcorrs = np.array([[norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(this_gc,this_bla)]\
                         for this_gc,this_bla in zip(gc_amp_split[:,inds1],bla_amp_split[:,inds2])])

mean_xcorrs = np.mean(xcorrs,axis=1)
std_xcorrs = np.std(xcorrs,axis=1)
mean_shuffle_xcorrs = np.mean(shuffled_xcorrs,axis=1)
std_shuffle_xcorrs = np.std(shuffled_xcorrs,axis=1)

fig,ax = plt.subplots(1,mean_xcorrs.shape[0],figsize = (15,5),sharey=True)
for this_bin in np.arange(mean_xcorrs.shape[0]):
        ax[this_bin].errorbar(dat.freq_vec,mean_xcorrs[this_bin],
                std_xcorrs[this_bin], label = 'actual',marker = 'o')
        ax[this_bin].errorbar(dat.freq_vec,mean_shuffle_xcorrs[this_bin],
                std_shuffle_xcorrs[this_bin], label = 'shuffle',marker = 'o')
        plt.legend();

        # Plot mean difference for each bin
        mean_diff = mean_xcorrs - mean_shuffle_xcorrs
        inds = np.array(list(np.ndindex(mean_diff.shape)))
        mean_diff_frame = pd.DataFrame({
                                'bin' : inds[:,0],
                                'freq' : inds[:,1],
                                'diff' : mean_diff.flatten()
                                })
        plt.figure()
        g = sns.catplot(
                    data=mean_diff_frame, kind="bar",
                        x="bin", y="diff", hue="freq",
                        ci="sd", palette="dark", alpha=.6, height=6)

        # Anova on each bin
        inds = np.array(list(np.ndindex(xcorrs.shape)))
        xcorr_frame = pd.DataFrame({
                            'bin' : inds[:,0],
                            'trial' : inds[:,1],
                            'freq' : inds[:,2],
                            'corr' : xcorrs.flatten(),
                            'class' : ['actual']*inds.shape[0]
                            })
        shuffle_inds = np.array(list(np.ndindex(shuffled_xcorrs.shape)))
        xcorr_frame = xcorr_frame.append(pd.DataFrame({
                            'bin' : shuffle_inds[:,0],
                            'trial' : shuffle_inds[:,1],
                            'freq' : shuffle_inds[:,2],
                            'corr' : shuffled_xcorrs.flatten(),
                            'class' : ['shuffle']*shuffle_inds.shape[0]
                            }))

        print(pd.concat([\
                        kruskal(data=xcorr_frame.query(f'bin == {x}'), 
                        dv='corr', between='class')\
                        for x in range(xcorrs.shape[0])]))

# Do same for rolling-zscored data
# Chop trial into 500ms bins and perform xcorr on each bin separately
splits = rzscore_gc_cut.shape[-1]//500
r_gc_amp_split = np.array(np.split(rzscore_gc_cut,splits,axis=-1))
r_bla_amp_split = np.array(np.split(rzscore_bla_cut,splits,axis=-1))
r_xcorrs = np.array([[norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(this_gc,this_bla)]\
                         for this_gc,this_bla in zip(r_gc_amp_split,r_bla_amp_split)])
print(r_xcorrs.shape)

resamples = 1000
inds1 = np.random.choice(np.arange(rzscore_gc_cut.shape[0]),size=resamples)
inds2 = np.random.choice(np.arange(rzscore_gc_cut.shape[0]),size=resamples)
print(inds1[:10])
print(inds2[:10])
r_shuffled_xcorrs = np.array([[norm_zero_lag_xcorr(v1,v2) for v1,v2 in zip(this_gc,this_bla)]\
                         for this_gc,this_bla in zip(r_gc_amp_split[:,inds1],r_bla_amp_split[:,inds2])])

r_mean_xcorrs = np.mean(r_xcorrs,axis=1)
r_std_xcorrs = np.std(r_xcorrs,axis=1)
r_mean_shuffle_xcorrs = np.mean(r_shuffled_xcorrs,axis=1)
r_std_shuffle_xcorrs = np.std(r_shuffled_xcorrs,axis=1)

fig,ax = plt.subplots(1,r_mean_xcorrs.shape[0],figsize = (15,5),sharey=True)
for this_bin in np.arange(r_mean_xcorrs.shape[0]):
        ax[this_bin].errorbar(dat.freq_vec,r_mean_xcorrs[this_bin],
                r_std_xcorrs[this_bin], label = 'actual',marker = 'o')
        ax[this_bin].errorbar(dat.freq_vec,r_mean_shuffle_xcorrs[this_bin],
                r_std_shuffle_xcorrs[this_bin], label = 'shuffle',marker = 'o')
        plt.legend();

        # Plot mean difference for each bin
        r_mean_diff = r_mean_xcorrs - r_mean_shuffle_xcorrs
        r_inds = np.array(list(np.ndindex(r_mean_diff.shape)))
        r_mean_diff_frame = pd.DataFrame({
                                'bin' : r_inds[:,0],
                                'freq' : r_inds[:,1],
                                'diff' : r_mean_diff.flatten()
                                })
        plt.figure()
        g = sns.catplot(
                    data= r_mean_diff_frame, kind="bar",
                    x="bin", y="diff", hue="freq",
                    ci="sd", palette="dark", alpha=.6, height=6)

        # Anova on each bin
        r_inds = np.array(list(np.ndindex(r_xcorrs.shape)))
        r_xcorr_frame = pd.DataFrame({
                            'bin' : r_inds[:,0],
                            'trial' : r_inds[:,1],
                            'freq' : r_inds[:,2],
                            'corr' : r_xcorrs.flatten(),
                            'class' : ['actual']*r_inds.shape[0]
                            })
        r_shuffle_inds = np.array(list(np.ndindex(shuffled_xcorrs.shape)))
        r_xcorr_frame = r_xcorr_frame.append(pd.DataFrame({
                            'bin' : r_shuffle_inds[:,0],
                            'trial' : r_shuffle_inds[:,1],
                            'freq' : r_shuffle_inds[:,2],
                            'corr' : r_shuffled_xcorrs.flatten(),
                            'class' : ['shuffle']*r_shuffle_inds.shape[0]
                            }))

        print(pd.concat([\
                        kruskal(data=r_xcorr_frame.query(f'bin == {x}'), 
                        dv='corr', between='class')\
                        for x in range(r_xcorrs.shape[0])]))
