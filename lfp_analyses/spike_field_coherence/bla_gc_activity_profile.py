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
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

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

mean_taste_spikes = [[np.mean(x,axis=1) for x in session] for session in spikes_list]
mean_region_spikes = list(zip(*mean_taste_spikes))
mean_region_spikes = [np.concatenate(x,axis=1) for x in mean_region_spikes]
mean_region_spikes = [np.concatenate(x,axis=0) for x in mean_region_spikes]

spikes_list = [[np.concatenate(x,axis=0) for x in session] for session in spikes_list]
region_spikes_list = list(zip(*spikes_list))
region_spikes_list = [np.concatenate(x,axis=1) for x in region_spikes_list]

############################################################
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
bin_width = 50
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

# hist2d 
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



##################################################
# Calculate binned spike counts

time_vec = np.arange(region_spikes_list[0].shape[-1])
time_vec = time_vec - stim_t

binned_time_vec = np.reshape(time_vec, (-1, bin_width)).mean(axis=1)
wanted_time_inds = np.logical_and(binned_time_vec >= time_lims[0],
                                  binned_time_vec <= time_lims[1])

binned_spikes_list = [np.reshape(x, 
                                 (x.shape[0], -1, bin_width)).sum(axis=-1) \
                                         for x in mean_region_spikes]
binned_spikes_list = [x[:,wanted_time_inds] for x in binned_spikes_list]
binned_time_vec = binned_time_vec[wanted_time_inds]

zscored_binned_spikes_list = [zscore(x, axis = 1) for x in binned_spikes_list]
# Remove nans
zscored_binned_spikes_list = [x[~np.isnan(x).any(axis=1)] for x in zscored_binned_spikes_list]

# Get time_bins
time_bins = np.reshape(time_vec, (-1, bin_width))[wanted_time_inds,:]
time_bin_lims = np.array([time_bins[:,0], time_bins[:,-1]]).T

# Perform PCA on zscored binned spikes for each region
pca_list = [PCA(n_components = 3) for x in regions]
pca_zscored_binned_spikes = [x.fit_transform(y.T).T for x,y in zip(pca_list, zscored_binned_spikes_list)]
#explained_variance_list = [x.explained_variance_ratio_ for x in pca_list]

# Sort zscored_binned_spikes using agglomerative clustering
n_clusters = 5
cluster_list = [AgglomerativeClustering(n_clusters = n_clusters) for x in regions]
cluster_labels_list = [x.fit_predict(y) for x,y in zip(cluster_list, zscored_binned_spikes_list)]
sort_inds_list = [np.argsort(x) for x in cluster_labels_list]
sorted_zscored_binned_spikes_list = [x[y] for x,y in zip(zscored_binned_spikes_list, sort_inds_list)]
sorted_binned_spike_list = [x[y] for x,y in zip(binned_spikes_list, sort_inds_list)]

# Plot binned spikes
fig, ax = plt.subplots(2,2, figsize = (5,10), sharex = True, sharey='row')
for i, region in enumerate(regions):
    this_binned_spikes = sorted_binned_spike_list[i]
    this_zscored_binned_spikes = sorted_zscored_binned_spikes_list[i]
    ax[0,i].pcolormesh(binned_time_vec, np.arange(this_binned_spikes.shape[0]),
                          this_binned_spikes, cmap = 'viridis')
    ax[0,i].set_title(region)
    ax[0,i].axvline(np.where(binned_time_vec >= 0)[0][0], color = 'red', linestyle = '--',
                    linewidth = 0.5)
    ax[0,i].set_xlabel('Time (ms)')
    ax[0,i].set_ylabel('Neuron #')
    ax[1,i].pcolormesh(binned_time_vec, np.arange(this_binned_spikes.shape[0]),
                       this_zscored_binned_spikes, cmap = 'viridis')
    ax[1,i].axvline(np.where(binned_time_vec >= 0)[0][0], color = 'red', linestyle = '--',
                    linewidth = 0.5)
# Set xlim to [-500, 1000]
for i in range(2):
    for j in range(2):
        ax[i,j].set_xlim([-500, 1000])
fig.suptitle('Binned spike counts')
fig.savefig(os.path.join(plot_dir,'binned_spike_counts_all_neurons.png'),
            bbox_inches = 'tight', dpi = 300)
plt.close(fig)

# Separate figure for each region, plot heatmap of each cluster along
# with the mean of each cluster
for i, region in enumerate(regions):
    fig, ax = plt.subplots(n_clusters,2, figsize = (5,10), sharex = True)
    this_zscored_binned_spikes = zscored_binned_spikes_list[i]
    this_labels = cluster_labels_list[i]
    for j in range(n_clusters):
        this_cluster = this_zscored_binned_spikes[this_labels == j]
        ax[j,0].pcolormesh(binned_time_vec, np.arange(this_cluster.shape[0]),
                           this_cluster, cmap = 'viridis')
        ax[j,0].set_ylabel('Cluster {}'.format(j+1))
        ax[j,0].axvline(np.where(binned_time_vec >= 0)[0][0], color = 'red', linestyle = '--',
                    linewidth = 0.5)
        ax[j,1].plot(binned_time_vec, this_cluster.mean(axis=0), color = 'black',
                     linewidth = 0.5, marker = 'o', markersize = 1)
        ax[j,1].axvline(np.where(binned_time_vec >= 0)[0][0], color = 'red', linestyle = '--',
                    linewidth = 0.5)
    ax[0,0].set_title(region)
    ax[0,1].set_title('Mean')
    ax[-1,0].set_xlabel('Time (ms)')
    ax[-1,1].set_xlabel('Time (ms)')
    # Set xlim to [-500, 1000]
    for this_ax in ax.flatten():
        this_ax.set_xlim([-500, 1000])
    fig.suptitle('Binned spike counts')
    fig.savefig(os.path.join(plot_dir,f'{region}_clustered_binned_spike_counts.png'),
                bbox_inches = 'tight', dpi = 300)
    plt.close(fig)

# Plot principal components
fig, ax = plt.subplots(3,2, figsize = (5,10), sharex = True, sharey=True)
for i, region in enumerate(regions):
    for j, this_pca in enumerate(pca_zscored_binned_spikes[i]):
        ax[j,i].plot(binned_time_vec, this_pca, marker = 'o', 
                   label = 'PC {}'.format(j+1), alpha = 0.5)
        ax[j,i].axvline(np.where(binned_time_vec >= 0)[0][0], 
                      color = 'red', linestyle = '--')
        # Mark the first timepoint after stimulus onset
        ax[j,i].scatter(binned_time_vec[np.where(binned_time_vec >= 0)[0][0]],
                          this_pca[np.where(binned_time_vec >= 0)[0][0]],
                          color = 'red', marker = 'o', s = 50)
        # Mark time of first timepoint after stimulus onset
        first_time_bin = time_bin_lims[np.where(binned_time_vec >= 0)[0][0]]
        first_time_bin_str = [str(x) for x in first_time_bin]
        ax[j,i].text(binned_time_vec[np.where(binned_time_vec >= 0)[0][0]] * 1.1,
                     this_pca[np.where(binned_time_vec >= 0)[0][0]] * 1.1,
                     "->".join(first_time_bin_str),
                     )
        ax[j,i].set_ylabel('PC {}'.format(j+1))
        if j == 0:
            ax[j,i].set_title(region)
        if j == 2:
            ax[j,i].set_xlabel('Time (ms)')
# Set xlim to [-500, 1000]
for this_ax in ax.flatten():
    this_ax.set_xlim([-500, 1000])
fig.suptitle('PCA of binned spike counts')
fig.savefig(os.path.join(plot_dir,'pca_binned_spike_counts_all_neurons.png'))
plt.close(fig)


