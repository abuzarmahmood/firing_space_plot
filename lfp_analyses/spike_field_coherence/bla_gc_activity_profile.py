"""
Look at activity profile of each region to see whether
early part of BLA-->GC spike-field coherence can be
explained by BLA activity alone.
"""

# Import required modules
from scipy.stats import mannwhitneyu, linregress
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import tables
import numpy as np
from tqdm import tqdm, trange
import shutil
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
import pingouin as pg

def parallelize(func, iterator):
    return Parallel(n_jobs=cpu_count()-2)(delayed(func)(this_iter) for this_iter in tqdm(iterator))


##################################################
## Read in data
##################################################
plot_dir = '/media/bigdata/firing_space_plot/lfp_analyses/spike_field_coherence/plots'

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path, 'r').readlines()]

frame_save_path = '/stft/analyses/spike_phase_coherence'

spikes_list = []
for this_dir in tqdm(dir_list):
    dat = ephys_data(this_dir)
    dat.get_spikes()
    region_spikes = [dat.return_region_spikes(x) for x in ['bla','gc']]
    spikes_list.append(region_spikes)

spikes_list = [[np.concatenate(x,axis=0) for x in session] for session in spikes_list]
region_spikes_list = list(zip(*spikes_list))
region_spikes_list = [np.concatenate(x,axis=1) for x in region_spikes_list]

# Convert to dataframes
spike_inds_list = [np.where(x) for x in region_spikes_list]

regions = ['bla','gc']
spike_inds_frames_list = [pd.DataFrame(
    dict(
        trial_num = x[0],
        neuron_num = x[1],
        time = x[2],
        region = regions[i]
        )
    ) for i,x in enumerate(spike_inds_list)]

spike_inds_frame = pd.concat(spike_inds_frames_list)
stim_t = 2000
time_lims = [-500, 2500]
spike_inds_frame['time'] = spike_inds_frame['time'] - stim_t
spike_inds_frame = spike_inds_frame.loc[
        np.logical_and(spike_inds_frame['time'] >= time_lims[0],
                       spike_inds_frame['time'] <= time_lims[1])]

# Cut time into bins
bin_width = 100
bins = np.arange(time_lims[0], time_lims[1] + bin_width, bin_width)
spike_inds_frame['time_bin'] = pd.cut(spike_inds_frame['time'],
                                      bins = bins,
                                      )

# Calculate spike counts
binned_spikes_frame = spike_inds_frame.groupby(
        ['neuron_num','time_bin', 'region']).count()['trial_num'].reset_index()
binned_spikes_frame = binned_spikes_frame.rename(
        columns = {'trial_num':'spike_count'})
binned_spikes_frame['bin_start_time'] = binned_spikes_frame['time_bin'].apply(
        lambda x: x.left)

# Plot heatmaps for each region
fig, ax = plt.subplots(1,2, figsize = (10,5))
for i, region in enumerate(regions):
    this_frame = binned_spikes_frame.loc[binned_spikes_frame['region'] == region]
    this_frame.reset_index(inplace=True, drop=True)
    pivot_frame = pd.pivot_table(
            this_frame,                    
            columns = 'time_bin',
              index = 'spike_count',
              aggfunc = 'sum')
    sns.heatmap(pivot_frame, ax = ax[i], origin = 'lower')
    ax[i].set_title(region)
fig.suptitle('Spike counts per bin')
fig.savefig(os.path.join(plot_dir,'binned_spike_counts_heatmap.png'))
plt.close(fig)

sns.displot(data = binned_spikes_frame, x = 'bin_start_time', y = 'spike_count',
         col = 'region') 
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'binned_spike_counts.png'))
plt.close(fig)


# Repeat with hist2d 
fig, ax = plt.subplots(1,2, figsize = (10,5))
for i, region in enumerate(regions):
    this_frame = binned_spikes_frame.loc[binned_spikes_frame['region'] == region]
    this_frame.reset_index(inplace=True, drop=True)
    x_edges = this_frame['bin_start_time'].unique()
    y_edges = np.logspace(0, np.log10(this_frame['spike_count'].max()), 20)
    im = ax[i].hist2d(this_frame['bin_start_time'], this_frame['spike_count'], 
                 bins = [x_edges, y_edges], 
                  norm = plt.matplotlib.colors.LogNorm(), cmap = 'viridis')
    ax[i].set_title(region)
    ax[i].axvline(0, color = 'red', linestyle = '--')
    plt.colorbar(im[3], ax = ax[i])
fig.suptitle('Spike counts per bin')
fig.savefig(os.path.join(plot_dir,'binned_spike_counts_hist2d.png'))
plt.close(fig)



# ##################################################
# # Plot binned spike counts
# 
# sns.displot(data = spike_inds_frame, x = 'time', y = 'spike_count',
#             col = 'region', bins = 50) 
# fig = plt.gcf()
# fig.savefig(os.path.join(plot_dir,'binned_spike_counts.png'))
# plt.close(fig)
# 
# ##################################################
# # Calculate binned spike counts
# 
# time_vec = np.arange(region_spikes_list[0].shape[-1])
# time_vec = time_vec - stim_t
# 
# binned_time_vec = np.reshape(time_vec, (-1, bin_width)).mean(axis=1)
# #wanted_bin_inds = np.logical_and(binned_time_vec >= time_lims[0],
# #                                 binned_time_vec <= time_lims[1])
# 
# binned_spikes_list = [np.reshape(x, 
#                                  (*x.shape[:-1], -1, bin_width)).sum(axis=-1) \
#                                          for x in region_spikes_list] 
# #binned_spikes_list = [x[...,wanted_bin_inds] for x in binned_spikes_list]
# binned_spike_frame = []
# for i, region in enumerate(binned_spikes_list):
#     this_inds = np.array(list(np.ndindex(region.shape)))
#     region_frame = pd.DataFrame(
#             dict(
#                 spike_count = region.flatten(),
#                 trial_num = this_inds[:,0],
#                 neuron_num = this_inds[:,1],
#                 time_bin = this_inds[:,2]
#                 )
#             )
#     region_frame['region'] = regions[i] 
#     binned_spike_frame.append(region_frame)
# binned_spike_frame = pd.concat(binned_spike_frame)
# binned_spike_frame['time'] = binned_time_vec[binned_spike_frame['time_bin']]
# binned_spike_frame = binned_spike_frame.loc[
#         np.logical_and(binned_spike_frame['time'] >= time_lims[0],
#                        binned_spike_frame['time'] <= time_lims[1])]
# 
# ##################################################
# # Plot binned spike counts
# 
# sns.displot(data = binned_spike_frame, x = 'time', y = 'spike_count',
#             col = 'region', kind = 'kde', bins = 50)
# fig = plt.gcf()
# fig.savefig(os.path.join(plot_dir,'binned_spike_counts.png'))
# plt.close(fig)
# 
# # Use hist2d to plot spike counts
# fig, ax = plt.subplots(1,2,sharey=True, sharex=True)
# for i, region in enumerate(regions):
#     this_frame = binned_spike_frame.loc[binned_spike_frame['region'] == region]
#     im = ax[i].hist2d(this_frame['time'], this_frame['spike_count'], bins = 50,
#                  norm = plt.matplotlib.colors.LogNorm(), cmap = 'Greys')
#     plt.colorbar(im[3], ax = ax[i])
#     ax[i].set_title(region)
# fig.savefig(os.path.join(plot_dir,'binned_spike_counts_hist2d.png'))
# plt.close(fig)
