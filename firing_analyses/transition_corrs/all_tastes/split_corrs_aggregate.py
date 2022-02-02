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
from scipy.stats import percentileofscore as p_of_s
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
wanted_names = ['rho_percentiles','rho_shuffles',
        'tau_corrs','tau_list', 'mse_percentiles'] 


# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/single_region_split_frame.pkl'
split_frame = pd.read_pickle(data_dir_pkl)

dat_list = []
session_num_list = []
#all_rho_percentiles = []
#all_mse_percentiles = []

for num, data_dir in tqdm(enumerate(split_frame.path)):
    #data_dir = split_frame.path.iloc[0]
    hf5_path = glob(os.path.join(data_dir,'*.h5'))[0]
    #hf5 = tables.open_file(hf5_path,'r')

    with tables.open_file(hf5_path,'r') as hf5:
        if save_path in hf5:

            #this_dat = [hf5.get_node(save_path, this_name)[:] \
            #        for this_name in wanted_names]

            split_nodes = hf5.list_nodes(save_path)
            this_dat = [[this_node[this_name][:] \
                    for this_name in wanted_names]\
                    for this_node in split_nodes]

            #this_rho_percentiles = \
            #        [this_node[wanted_names[0]][:] for this_node in split_nodes]
            #this_mse_percentiles = \
            #        [this_node[wanted_names[1]][:] for this_node in split_nodes]
            #all_rho_percentiles.append(this_rho_percentiles)
            #all_mse_percentiles.append(this_mse_percentiles)
            session_num_list.append(num)
            dat_list.append(this_dat)
        else:
            raise Exception('Saved percentiles not found')
dat_list_zip = [list(zip(*this_dat)) for this_dat in dat_list]
dat_list_zip = list(zip(*dat_list_zip)) 
dat_list_zip = [np.stack(x) for x in dat_list_zip]
for this_var, this_dat in zip(wanted_names, dat_list_zip):
    globals()[this_var] = this_dat 

# For reverse consistency
all_rho_percentiles = np.array(rho_percentiles)
all_mse_percentiles = np.array(mse_percentiles)

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

########################################
## Shuffled Rho histograms overlayed with actual data
########################################
wanted_perc= 90
perc_mark_val = stats.scoreatpercentile(rho_shuffles[0,0,0],wanted_perc)
bins = np.linspace(-0.5,0.5,50)
shuff_hist_list = np.stack([[np.histogram(x,bins = bins)[0] for x in y]\
        for y in rho_shuffles[:,0].T])
mean_counts = np.mean(shuff_hist_list,axis=0)
std_counts = np.std(shuff_hist_list,axis=0)
#std_counts = np.percentile(shuff_hist_list,[25,75],axis=0)

bins = np.linspace(-0.5,0.5,50)
fig,ax = plt.subplots(tau_corrs.shape[2],1, sharex = True, sharey = True,
        figsize = (7,10))
for trans_num in range(tau_corrs.shape[2]):
    ax[trans_num].errorbar(x = bins[:-1], y = mean_counts[trans_num],
                yerr = std_counts[trans_num], alpha = 0.5,
                label = 'Expected Shuffle Counts', linewidth = 2)
    ax[trans_num].hist(tau_corrs[:,0,trans_num],
            alpha = 0.5, bins = bins, label = 'Actual')
    ax[trans_num].axvline(perc_mark_val, color = 'red', linestyle = 'dashed',
                        linewidth = 2, 
                        label = f'{wanted_perc}th shuffle percentile')
    ax[trans_num].set_title(f'Transition {trans_num}')
ax[-1].set_xlabel('Rho Value')
ax[-1].legend()
plt.suptitle('Simlated Corr Hists')
fig.savefig(os.path.join(plot_dir, 'simulated_agg_rho_comparison2')) 
plt.close(fig)

########################################
## Shuffled Rho dist vs actual data 
########################################
bins = np.linspace(-0.5,0.5,50)
fig,ax = plt.subplots(tau_corrs.shape[2],1, sharex = True, sharey = True,
        figsize = (7,10))
for trans_num in range(tau_corrs.shape[2]):
    ax[trans_num].hist(rho_shuffles[:,0,trans_num,:1000].flatten(),
            alpha = 0.5, bins = bins, label = 'Shuffle', density = True)
    ax[trans_num].hist(tau_corrs[:,0,trans_num],
            alpha = 0.5, bins = bins, label = 'Actual')
    ax[trans_num].hist(rho_shuffles[:,0,trans_num,:1000].flatten(),
            alpha = 0.5, bins = bins, density = True, histtype = 'step')
    ax[trans_num].hist(tau_corrs[:,0,trans_num],
            alpha = 0.5, bins = bins, histtype = 'step')
    ax[trans_num].axvline(perc_mark_val, color = 'red', linestyle = 'dashed',
                        linewidth = 2, 
                        label = f'{wanted_perc}th shuffle percentile')
    ax[trans_num].set_title(f'Transition {trans_num}')
ax[-1].set_xlabel('Rho Value')
ax[-1].legend()
plt.suptitle('Simlated Corr Hists')
fig.savefig(os.path.join(plot_dir, 'simulated_agg_rho_comparison')) 
plt.close(fig)

########################################
# Plot count of recordings with percentiles above a threshold
# Compared with shuffles
########################################
rho_percs_cut = rho_percentiles[:,0]
rho_percs_shuff_cut = rho_shuffles[:,0]
rho_shuffle_percs = np.array([[[p_of_s(x,z) for z in x] for x in y] \
                        for y in tqdm(rho_percs_shuff_cut)])

# Hist counts you will get from random data
rho_shuff_perc_cut = rho_shuffle_percs[...,:1000]
sig_hist_bins = [90,100]
random_hists_counts = np.array(
    [[np.histogram(x, sig_hist_bins)[0] for x in y] 
        for y in rho_shuff_perc_cut.T]) 

mean_hist_counts = np.mean(random_hists_counts, axis = 0)
std_hist_counts = np.std(random_hists_counts, axis = 0)
mean_hist_frac = mean_hist_counts/tau_corrs.shape[0]
std_hist_frac = std_hist_counts/tau_corrs.shape[0]

rho_perc_counts = np.array(
        [np.histogram(x,sig_hist_bins)[0] for x in rho_percentiles[:,0].T])
rho_perc_frac = rho_perc_counts/tau_corrs.shape[0]
all_bin_percs = np.array([[p_of_s(
                    random_hists_counts[:,ch_ind, bin_ind], 
                    rho_perc_counts[ch_ind, bin_ind])\
        for ch_ind in range(rho_perc_counts.shape[0])]\
        for bin_ind in range(rho_perc_counts.shape[1])] ).T

# 2 tailed, Bonferroni corrected p-value
comparisons = all_bin_percs.shape[0]
alpha = 0.05
abs_diff_perc = np.min(np.stack([100 - all_bin_percs, all_bin_percs]),axis=0)
abs_diff_perc = ((abs_diff_perc*2)/100)*comparisons
bonf_sig = abs_diff_perc <= alpha 

x = np.arange(len(bonf_sig))
fig,ax = plt.subplots(figsize=(5,7))
cmap = plt.get_cmap('tab10')
#ax.bar(x, rho_perc_counts.flatten(), label = 'Actual', 
ax.bar(x, rho_perc_frac.flatten(), label = 'Actual', 
        color = cmap(0), edgecolor = None, alpha = 0.7, linewidth = 2)
        #color = cmap(0), edgecolor = cmap(0), alpha = 0.7, linewidth = 2)
#ax.errorbar(x, mean_hist_counts, std_hist_counts, 
ax.errorbar(x, mean_hist_frac.flatten(), std_hist_frac.flatten(), 
        label = 'Shuffle', color = 'k', linewidth = 5, alpha = 0.7)
        #label = 'Shuffle', color = cmap(1), linewidth = 5, alpha = 0.7)
for num,(perc,sig) in enumerate(zip(abs_diff_perc,bonf_sig)):
    #ax.text(x[num], np.max(rho_perc_counts) - 1, 
    ax.text(x[num], np.max(rho_perc_frac)*0.9, 
            np.round(perc[0],3), ha = 'center', rotation = 45)
    if sig:
        #ax.text(x[num], np.max(rho_perc_counts) - 2, 
        ax.text(x[num], np.max(rho_perc_frac)*0.8, 
            '*', ha = 'center', fontweight = 'bold',
            fontsize = 'xx-large')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
ax.set_xlabel('Transition Number')
ax.set_ylabel('Frequency Number')
#ax[0,1].set_title('MSE Percentiles')
plt.suptitle(f'Aggregate transition hist : Bin = {sig_hist_bins}' + \
        "\n" + "Numbers = Bonf Corrected 2-tailed p-vals" + "\n" +\
        f'total count : {rho_percentiles.shape[0]}' +\
        f' ::: alpha = {alpha}')
fig.savefig(os.path.join(plot_dir, 
    f'sig_hist_bins_transition_max{np.diff(sig_hist_bins)[0]}'), 
    bbox_inches = 'tight')
plt.close(fig)
#plt.show()


########################################
## Plot median rho percentiles hist
########################################
median_rho_percentiles = np.median(all_rho_percentiles, axis=1).T
median_mse_percentiles = np.median(all_mse_percentiles, axis=1).T

hist_bins = np.linspace(0,100,11)
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
