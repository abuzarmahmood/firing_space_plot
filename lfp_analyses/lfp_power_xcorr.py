"""
Cross correlations in LFP amplitudes between
brain regions


1) Load data
    - Any transformations should be done here
        so that future analyses are agnostic to
        what data is fed
    - e.g. rolling window z-scoring

** All analyses below will be served with corresponding
    shuffles

2) XCorr b/w representative electrodes from both regions
3) XCorr b/w all pairs of electrodes within the same
    region to contextualize what inter-region xcorr represents
4) Inter-region XCorr within time windows
"""

# import stuff

import os
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt
plt.set_cmap('viridis')
from matplotlib.lines import Line2D
from tqdm import tqdm
from pingouin import mwu,kruskal, read_dataset
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

data_dir = '/media/bigdata/Abuzar_Data/AM12/AM12_4Tastes_191106_085215/'
dat = ephys_data(data_dir)
dat.get_stft()
dat.get_lfp_electrodes()

median_amplitude = np.median(dat.amplitude_array,axis=(0,2))
#visualize.firing_overview(stats.zscore(median_amplitude,axis=-1));plt.show()

amplitude_array = dat.amplitude_array.swapaxes(0,1)

# To perform pairwise shuffle, determine when each TASTE trial
# was delivered chronologically`
# Grab dig-ins from hf5 file
taste_digins = dat.info_dict['taste_params']['dig_ins']
with tables.open_file(dat.hdf5_name,'r') as hf5:
    dig_in_array = np.array([dig_in[:] \
            for dig_in in hf5.list_nodes('/digital_in') \
            if int(dig_in.name[-1]) in taste_digins])

# Find out when trials occured
dig_in_diff = np.diff(dig_in_array,axis=-1)
dig_in_starts = np.where(dig_in_diff > 0)
# This order applies to all tastes CONCATENATED
chron_order = np.argsort(dig_in_starts[1])
chron_tastes = dig_in_starts[0][chron_order]

##################################################
#|_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___  
#  | || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ 
#  | || | | (_| | | | \__ \  _| (_) | |  | | | | | |
#  |_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|
##################################################

# If we take rolling z-zscore at this stage, then we can likely
# eliminate the impact of many artifacts
# A "transformation" can also be removing trials with artifacts
# from the raw data
#transformed_long = min_err_long # Trivial transformation

def rolling_zscore(array, window_size):
    """
    Performs rolling z-score on last dimension of given array
    """
    out = np.zeros(array.shape)
    starts = np.arange((array.shape[-1] - window_size))
    inds = list(zip(starts,starts+window_size))
    for this_ind in tqdm(inds):
        out[...,this_ind[0]:this_ind[1]] += \
                stats.zscore(array[...,this_ind[0]:this_ind[1]],axis=-1)
    return out/window_size

# Perform rolling window on WHOLE trial so that edges can
# be choppoed away later
window_size = 500
# This will need to be parallelilzed
transformed_array = np.array(\
        parallelize(lambda x: rolling_zscore(x, window_size), amplitude_array))

# Chop to relevant time period
time_lims = [2000,4000]
transformed_array = transformed_array[...,time_lims[0]:time_lims[1]]

del amplitude_array

# Compare rolling zscored to raw
#med_amps = np.concatenate([stats.zscore(np.median(min_err_long,axis=1),axis=-1), 
#                        np.median(rzscore_min_err_long,axis=1)])
#fig,ax = plt.subplots(4,1)
#for this_dat, this_ax in zip(med_amps, ax):
#    this_ax.imshow(this_dat,aspect='auto',cmap='jet', origin='lower')
#plt.show()

# For each channel, determine trials which go outside their respective
# MADs, then take union
#transformed_vlong = np.reshape(transformed_long.swapaxes(1,2),
#            (transformed_long.shape[0],transformed_long.shape[2],-1))

# Compare raw to rolling zscore long
#fig,ax = plt.subplots(2,1)
#ax[0].imshow(min_err_vlong[0],aspect='auto',cmap='viridis')
#ax[1].imshow(transformed_vlong[0],aspect='auto',cmap='viridis')
#plt.show()

# We can use ALL trials for rolling zscored power rather than
# arbitrarily removing trials due to artifacts

##############################
#__  ______                
#\ \/ / ___|___  _ __ _ __ 
# \  / |   / _ \| '__| '__|
# /  \ |__| (_) | |  | |   
#/_/\_\____\___/|_|  |_|   
#                          
##############################

# Find representative channel from each region
# That is, channel closest to the median of all channels
transformed_array_long = transformed_array.reshape(\
        (transformed_array.shape[0],-1,*transformed_array.shape[3:]))
split_amplitude_list = \
        [transformed_array_long[region] for region in dat.lfp_region_electrodes]
#split_median_amplitude_list = [np.median(x,axis=0) for x in split_amplitude_list]
#error_list = [np.sum(np.abs(this_region - median), \
#                    axis = tuple(np.arange(1,len(this_region.shape)))) \
#                for this_region, median in \
#                zip(split_amplitude_list, split_median_amplitude_list)] 
#min_err_channels_inds = [np.argmin(region) for region in error_list]
#min_err_channels = np.array([region[ind] for region, ind in \
#                        zip(split_amplitude_list, min_err_channels_inds)])
#
#min_err_long = np.reshape(min_err_channels,
#        (min_err_channels.shape[0],-1,*min_err_channels.shape[3:]))

# Chop to relevant time period
#time_lims = [2000,4000]
#min_err_channels = min_err_channels[...,time_lims[0]:time_lims[1]]
#min_err_vlong = np.reshape(min_err_long.swapaxes(1,2),
#            (min_err_long.shape[0],min_err_long.shape[2],-1))

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

########################################
## Inter-Region
########################################

#inter_region_xcorr = norm_zero_lag_xcorr(min_err_long[0],min_err_long[1])
#random_shuffle_inter_region_xcorr = norm_zero_lag_xcorr(\
#                    np.random.permutation(min_err_long[0]),min_err_long[1])

# Mean XCorr between all pairs of channels from both regions
chan_count = [len(x) for x in split_amplitude_list]
wanted_order = np.argsort(chan_count)[::-1]
temp_split_amp_list = [split_amplitude_list[x] for x in wanted_order]
chan_count = [len(x) for x in temp_split_amp_list]
all_pairs = np.arange(1,1+chan_count[0])[:,np.newaxis].\
        dot(np.arange(1,1+chan_count[1])[np.newaxis,:])
pair_inds = list(zip(*np.where(np.tril(all_pairs))))
inter_region_xcorr = np.array([\
        norm_zero_lag_xcorr(temp_split_amp_list[0][ind[0]],
                            temp_split_amp_list[1][ind[1]]) \
                    for ind in pair_inds])
shuffled_inter_region_xcorr = np.array([\
        norm_zero_lag_xcorr(np.random.permutation(temp_split_amp_list[0][ind[0]]),
                            temp_split_amp_list[1][ind[1]]) \
                    for ind in pair_inds])
mean_inter_region_xcorr = np.mean(inter_region_xcorr,axis=1)
mean_shuffled_inter_region_xcorr = np.mean(shuffled_inter_region_xcorr,axis=1)

# Perform "stringent" shuffle by permuting trials in chronological pairs
#permute_chron_order = chron_order.reshape((-1,2))[:,::-1].flatten()
#pair_shuffle_inter_region_xcorr = norm_zero_lag_xcorr(\
#                    min_err_long[0][chron_order],
#                    min_err_long[1][permute_chron_order])

########################################
## Within Region
########################################
# Perform xcorr between all pairs of channels within a region
pair_list = [list(it.combinations(np.arange(x.shape[0]),2)) \
                        for x in split_amplitude_list]
intra_region_xcorr_list = [np.array([\
        norm_zero_lag_xcorr(region[ind[0]],region[ind[1]]) \
        for ind in this_pair_list]) \
        for this_pair_list, region in zip(pair_list, split_amplitude_list)]
mean_intra_region_xcorr = [np.mean(x,axis=(1)) for x in intra_region_xcorr_list]

########################################
## Binned Inter-Region
########################################
window_size = 250
splits = np.abs(np.diff(time_lims)[0])//window_size
binned_amp = [np.split(x,splits,axis=-1) for x in temp_split_amp_list]

binned_inter_region_xcorr = np.array([[\
        norm_zero_lag_xcorr(this_bin[0][ind[0]],
                            this_bin[1][ind[1]]) \
                    for ind in pair_inds]\
                    for this_bin in zip(*binned_amp)])
shuffled_binned_inter_region_xcorr = np.array([[\
        norm_zero_lag_xcorr(np.random.permutation(this_bin[0][ind[0]]),
                            this_bin[1][ind[1]]) \
                    for ind in pair_inds]\
                    for this_bin in zip(*binned_amp)])
mean_binned_inter_xcorr = np.mean(binned_inter_region_xcorr,axis=2)
mean_shuffled_binned_inter_xcorr = np.mean(shuffled_binned_inter_region_xcorr,axis=2)

########################################
## Plotting
########################################

bins = None
alpha = 0.8
plthist = lambda x, bins, label, alpha : \
    plt.hist(x.flatten(), bins = bins, label = label, alpha = alpha, density = False)
plthist(mean_inter_region_xcorr,bins, 'inter', alpha)
plthist(mean_shuffled_inter_region_xcorr,bins, 'shuffle', alpha)
plthist(mean_intra_region_xcorr[1],None, dat.region_names[1], alpha)
plthist(mean_intra_region_xcorr[0],None, dat.region_names[0], alpha)
plt.legend()
plt.show()

# Convert arrays to dataframes to plot with seaborn
def gen_df(array, label):
    inds = np.array(list(np.ndindex(array.shape)))
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'freq' : dat.freq_vec[inds[:,1]],
            'xcorr' : array.flatten()})

plot_frame = pd.concat([\
            gen_df(mean_inter_region_xcorr, 'inter'),
            gen_df(mean_shuffled_inter_region_xcorr, 'shuffle'),
            gen_df(mean_intra_region_xcorr[0], dat.region_names[0]),
            gen_df(mean_intra_region_xcorr[1], dat.region_names[1])])

sns.boxplot(x='freq',y='xcorr',hue='label',data=plot_frame)
plt.show()

def gen_df_bin(array, label):
    inds = np.array(list(np.ndindex(array.shape)))
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'bin' : inds[:,0],
            'freq' : dat.freq_vec[inds[:,2]],
            'xcorr' : array.flatten()})

plot_frame = pd.concat([\
            gen_df_bin(mean_binned_inter_xcorr, 'inter'),
            gen_df_bin(mean_shuffled_binned_inter_xcorr, 'shuffle')])

sns.relplot(x='freq',y='xcorr',hue='label',col = 'bin', 
                    data=plot_frame, kind = 'line', ci='sd',col_wrap=4)
plt.show()
