"""
Given percentiles calculated for split fits, compile percentiles
for different splits of the same recording and also acros recordings
"""

import numpy as np
import json
from glob import glob
import os
import pandas as pd
import pickle 
import sys
from scipy import stats
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange 
import tables
import pylab as plt

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs')
from check_data import check_data 

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/split_region'
wanted_names = ['rho_percentiles','mse_percentiles']


# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/single_region_split_frame.pkl'
split_frame = pd.read_pickle(data_dir_pkl)

session_num_list = []
all_rho_percentiles = []
all_mse_percentiles = []

for num, data_dir in tqdm(enumerate(split_frame.path)):
    #data_dir = split_frame.path.iloc[0]
    hf5_path = glob(os.path.join(data_dir,'*.h5'))[0]
    #hf5 = tables.open_file(hf5_path,'r')

    with tables.open_file(hf5_path,'r') as hf5:
        if save_path in hf5:

            split_nodes = hf5.list_nodes(save_path)
            this_rho_percentiles = \
                    [this_node[wanted_names[0]][:] for this_node in split_nodes]
            this_mse_percentiles = \
                    [this_node[wanted_names[1]][:] for this_node in split_nodes]
            all_rho_percentiles.append(this_rho_percentiles)
            all_mse_percentiles.append(this_mse_percentiles)
            session_num_list.append(num)

all_rho_percentiles = np.array(all_rho_percentiles)
all_mse_percentiles = np.array(all_mse_percentiles)

########################################
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
########################################
                       
plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/plots/single_region_split'

#session_num = 0
for session_num in trange(len(all_rho_percentiles)):
    session_name = split_frame.name.iloc[session_num]
    this_plot_dir = os.path.join(plot_dir, session_name)
    if not os.path.exists(this_plot_dir):
        os.makedirs(this_plot_dir)
    session_dat = np.array([all_rho_percentiles[session_num].T,
                        all_mse_percentiles[session_num].T])

    # Intra session hists
    hist_bins = np.linspace(0,100,21)
    inds = list(np.ndindex(session_dat.shape[:-1]))
    plot_count = all_rho_percentiles.shape[-1]
    fig, ax = plt.subplots(plot_count,2)
    for this_inds in inds:
        plot_ind = tuple(reversed(this_inds))
        ax[plot_ind].hist(session_dat[this_inds], bins = hist_bins)
    ax[0,0].set_title('Corr percentiles')
    ax[0,1].set_title('MSE percentiles')
    for transition_num in range(ax.shape[0]):
        ax[transition_num,0].set_ylabel(f'Transition {transition_num}')
    plt.suptitle(split_frame.name.iloc[session_num_list[session_num]] + '\n' + \
            str(split_frame.regions.iloc[session_num_list[session_num]]))
    fig.savefig(os.path.join(this_plot_dir,'percentile_hists'))
    plt.close(fig)
    #plt.show()

    ## Intra session scatter
    fig,ax = plt.subplots(plot_count,2, sharex = True, sharey = True)
    for trans_num, this_dat in enumerate(session_dat.swapaxes(0,1)):
        ax[trans_num, 0].scatter(this_dat[0],this_dat[1], s = 10,
                facecolors = 'none', edgecolors = 'k')
        #ax[trans_num, 0].set_aspect('equal','box')
        ax[trans_num, 1].hist2d(this_dat[0],this_dat[1], bins = hist_bins)
        #ax[trans_num, 1].set_aspect('equal','box')
    ax[0, 0].set_xlim(0,100)
    ax[0, 0].set_ylim(0,100)
    plt.suptitle(session_name+ '\n' + \
            str(split_frame.regions.iloc[session_num_list[session_num]]))
    for transition_num in range(ax.shape[0]):
        ax[transition_num,0].set_ylabel(f'Transition {transition_num}')
    ax[-1,-1].set_xlabel('Corr percentiles')
    ax[-1,-1].set_ylabel('MSE percentiles')
    fig.savefig(os.path.join(this_plot_dir,'percentile_scatters'))
    plt.close(fig)
    #plt.show()

# Aggregate percentiles

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/plots/single_region_split/aggregate'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

median_rho_percentiles = np.median(all_rho_percentiles, axis=1).T
median_mse_percentiles = np.median(all_mse_percentiles, axis=1).T

hist_bins = np.linspace(0,100,21)
fig, ax = plt.subplots(2,median_rho_percentiles.shape[0], 
        sharex = True, sharey = True)
for transition_num in range(median_rho_percentiles.shape[0]):
        ax[0,transition_num].hist(median_rho_percentiles[transition_num],
                bins = hist_bins)
        ax[1,transition_num].hist(median_mse_percentiles[transition_num],
                bins = hist_bins)
        ax[0, transition_num].set_title(f'Transition {transition_num}')
ax[0,0].set_ylabel('Corr percentiles')
ax[1,0].set_ylabel('MSE percentiles')
plt.suptitle('Single Region Splits \n Median Percentiles \n '\
        'Count = {}'.format(median_mse_percentiles.shape[1]))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
fig.savefig(os.path.join(plot_dir,'aggregate_median_hists'))
plt.close(fig)
#plt.show()

fig, ax = plt.subplots(2,median_rho_percentiles.shape[0], 
        sharex = True, sharey = True)
for transition_num in range(median_rho_percentiles.shape[0]):
        ax[0, transition_num].scatter(median_rho_percentiles[transition_num],
                            median_mse_percentiles[transition_num])
        ax[0, transition_num].set_xlim(0,100)
        ax[0, transition_num].set_ylim(0,100)
        ax[1, transition_num].hist2d(median_rho_percentiles[transition_num],
                            median_mse_percentiles[transition_num],
                            bins = hist_bins)
        ax[0, transition_num].set_title(f'Transition {transition_num}')
ax[-1,-1].set_xlabel('Corr percentiles')
ax[-1,-1].set_ylabel('MSE percentiles')
plt.suptitle('Single Region Splits \n Median Percentiles \n '\
        'Count = {}'.format(median_mse_percentiles.shape[1]))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
fig.savefig(os.path.join(plot_dir,'aggregate_median_scatters'))
plt.close(fig)
#plt.show()

# Analysis to show that good and bad percentiles are dependent
# on each particular recording
# Use K-Means clustering on collective percentiles matrix
from sklearn.cluster import KMeans
all_percentiles = np.concatenate((median_rho_percentiles,
                    median_mse_percentiles), axis=0).T
kmeans = KMeans(n_clusters=3, random_state=0).fit(all_percentiles)
inds = np.argsort(kmeans.labels_)

plt.imshow(all_percentiles[inds],aspect='auto')
plt.xlabel('Percentiles')
plt.ylabel('Recording sessions')
plt.colorbar()
plt.suptitle('Aggregate percentile clustering')
plt.savefig(os.path.join(plot_dir,'aggregate_median_percentile_clustering'))
plt.close('all')
