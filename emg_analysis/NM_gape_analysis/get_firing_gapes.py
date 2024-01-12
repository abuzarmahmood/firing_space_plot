"""
# Run PCA on whole trials
Analyze differences in neural and emg activity conditioned on 
strength of gaping activity per trial
"""
########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

########################################
# Import modules
########################################

import os
import sys
import numpy as np
from tqdm import tqdm
import tables
import itertools as it
import pylab as plt
from glob import glob
import pandas as pd
import seaborn as sns
import re
import pingouin as pg
from scipy.stats import zscore
import tensortools as tt
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import xarray as xr
from sklearn.cluster import KMeans
from collections import Counter

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

plot_dir = '/media/bigdata/firing_space_plot/NM_gape_analysis/plots'

file_list_path = '/media/fastdata/NM_sorted_data/h5_file_list.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
dir_names = [os.path.dirname(x) for x in file_list]

def time_box_conv(array, kern_width):
    """
    Convolution with 1D box kern along last dim
    """
    box_kern = np.ones((kern_width))/kern_width
    inds = list(np.ndindex(array.shape[:-1]))
    conv_array = np.empty(array.shape)
    for this_ind in tqdm(inds):
        conv_array[this_ind] = np.convolve(array[this_ind], box_kern, mode='same') 
    return conv_array

############################################################
# / ___| __ _ _ __   ___  |  _ \  __ _| |_ __ _ 
#| |  _ / _` | '_ \ / _ \ | | | |/ _` | __/ _` |
#| |_| | (_| | |_) |  __/ | |_| | (_| | || (_| |
# \____|\__,_| .__/ \___| |____/ \__,_|\__\__,_|
#            |_|                                
############################################################

gape_path = '/ancillary_analysis/gapes'
# laser_conds x taste x trials x time

emg_bsa_path = '/ancillary_analysis/emg_BSA_results'
# laser_conds x taste x trials x time x freq 

laser_dl_path = '/ancillary_analysis/laser_combination_d_l'
# condition_num x (duration + onset) 

# /emg_BSA_results
# taste_0_p : something? x time x something?

#ind = 1
#this_file_path = file_list[ind]
#this_dir = dir_names[ind]
#

condition_list = []
gapes_list = []
for this_path in file_list:
    # Gape related info
    #with tables.open_file(dat.hdf5_path,'r') as h5:
    with tables.open_file(this_path,'r') as h5:
        gape_laser_conditions = h5.get_node(laser_dl_path)[:] 
        #emg_bsa_results = h5.get_node(emg_bsa_path)[:] 
        gapes_array = h5.get_node(gape_path)[:] 
    condition_list.append(gape_laser_conditions)
    gapes_list.append(gapes_array)

flat_gapes = [x.flatten() for x in gapes_list]
flat_gapes = [x for y in flat_gapes for x in y]
#plt.hist(flat_gapes[::10]);plt.show()

# Some values are really low (e.g. e-88), clean those out
gapes_list = [(x>0.5)*1 for x in gapes_list]

wanted_condition = [np.where(x.sum(axis=-1)==0)[0][0] for x in condition_list]
wanted_gape_array = [x[i] for x,i in zip(gapes_list, wanted_condition)]

########################################
## Quin vs Suc Gape reponse
########################################

time_lims = [1000,5000]
real_time = np.arange(-2000, 5000)
cut_real_time = real_time[time_lims[0]:time_lims[1]]
stim_t = 2000 - time_lims[0]
taste_inds = np.array([0,3]) # 0:Sucrose, 3:quinine
taste_names = ['suc','quin']
quin_ind = 3
wanted_gape_array = [x[taste_inds] for x in wanted_gape_array]
wanted_gape_array = [x[...,time_lims[0]:time_lims[1]] for x in wanted_gape_array]

# Check for which recordings distances are sufficiently separate
inds = [np.array(list(np.ndindex(x.shape))) for x in wanted_gape_array]
gape_frames = [pd.DataFrame(dict(
                session = num,
                taste = this_inds[:,0],
                trials = this_inds[:,1],
                time = this_inds[:,2],
                vals = this_dat.flatten())) \
        for num, (this_inds,this_dat) in enumerate(zip(inds, wanted_gape_array))]
fin_gape_frame = pd.concat(gape_frames)
fin_gape_frame['real_time'] = cut_real_time[fin_gape_frame['time']]

# Downsample for ANOVA
binsize = 500
bincount = int(np.diff(time_lims)[0]/binsize)
fin_gape_frame['time_bins'] = pd.cut(fin_gape_frame['time'], bincount, 
       labels = np.arange(bincount))
fin_gape_frame['vals'] += np.random.random(fin_gape_frame['vals'].shape)*0.01

group_cols = ['session','taste','trials','time_bins']
bin_gape_frame = fin_gape_frame.groupby(group_cols).mean().reset_index()
bin_gape_frame.dropna(inplace=True)

# Perform ANOVA
# Perform separately for each session
group_bin_gape = [x[1] for x in list(bin_gape_frame.groupby('session'))]
anova_list = [pg.anova(data = this_dat,
            dv = 'vals', between = ['taste','time_bins']) \
                    for this_dat in group_bin_gape]
pval_list = [x['p-unc'] for x in anova_list]
taste_bool = np.stack(pval_list)[:,0]<0.05

# Also check that quinine is HIGHER than sucrose
quin_bool = [x.groupby('taste')['vals'].mean().diff()[1]>0 for x in group_bin_gape]

#g = sns.relplot(data = bin_gape_frame,
#        x = 'real_time', y = 'vals',
#        hue = 'taste', col = 'session', col_wrap = 5,
#        kind = 'line')
#for this_ax, this_taste_bool, this_quin_bool  in zip(g.axes, taste_bool, quin_bool):
#    this_ax.set_title((this_taste_bool, this_quin_bool))
#plt.show()

# Only take session where both are true
fin_bool = np.logical_and(taste_bool, quin_bool)
fin_bool_inds = np.where(fin_bool)[0]
fin_bin_gape = bin_gape_frame[bin_gape_frame['session'].isin(fin_bool_inds)] 

#g = sns.relplot(data = fin_bin_gape,
#        x = 'real_time', y = 'vals',
#        hue = 'taste', col = 'session', col_wrap = 5,
#        kind = 'line')
#plt.show()

########################################
## Subtract AVERAGE sucrose response as non-specific EMG response 
########################################
taste_frame = fin_gape_frame.copy()

corr_frame = taste_frame[taste_frame.real_time > 0].groupby(['session','taste']).mean()
x = np.linspace(corr_frame.vals.min(), corr_frame.vals.max())
plt.scatter(*[x[1].vals for x in list(corr_frame.groupby('taste'))])
plt.xlabel('Average sucrose response')
plt.ylabel('Average quinine response')
plt.plot(x,x, color = 'red', alpha = 0.3, linestyle = '--')
fig = plt.gcf()
fig.suptitle('Correlated Taste Responses')
fig.savefig(os.path.join(plot_dir, 'correlated_taste_responses.png'))
plt.show()

mean_taste_frame = taste_frame.groupby(['session','taste','time']).mean()
mean_taste_frame = mean_taste_frame.drop(columns = 'trials')
mean_taste_array = mean_taste_frame.to_xarray()['vals']
mean_taste_array = mean_taste_array[fin_bool].reset_index('session')

taste_diff_array = mean_taste_array.diff(dim = 'taste').squeeze()

g = mean_taste_array.plot(
        x = 'time',
        y = 'session',
        col = 'taste',
        aspect = 2,
        size = 3
        );
for num, ax in enumerate(g.axes[0]):
    ax.axvline(stim_t, linestyle = '--', color = 'red', linewidth = 2,
            label = 'Stim Delivery')
    ax.set_title(taste_names[num])
    #ax.legend()
fig = plt.gcf()
fig.suptitle('Average EMG Resopnses')
#plt.subplots_adjust(top = 0.8)
fig.savefig(os.path.join(plot_dir, 'average_emg_responses.png'))
#plt.show()

taste_diff_array.plot(cmap = 'viridis', aspect = 2, size = 3);
ax = plt.gca()
ax.axvline(stim_t, linestyle = '--', color = 'k')
fig = plt.gcf()
fig.suptitle('Quin - Suc Responses')
fig.savefig(os.path.join(plot_dir, 'average_subtracted_emg_responses.png'))
#plt.show()

########################################
## Clustering in gape responses to quinine 
########################################
quin_gape_array = [x[1] for x in wanted_gape_array]
quin_gape_array = [quin_gape_array[i] for i in fin_bool_inds]

# Subtract mean sucrose response from respective quinine responses
suc_gape_array = [x[0] for x in wanted_gape_array]
suc_gape_array = [suc_gape_array[i] for i in fin_bool_inds]
mean_suc_gape = np.stack([np.mean(x,axis=0) for x in suc_gape_array])
quin_gape_array = [x-y for x,y in zip(quin_gape_array, mean_suc_gape)]

#vz.imshow(mean_suc_gape);plt.colorbar();plt.show()

gape_t_lims = [750,2500]
gape_t_lims = [x+time_lims[0] for x in gape_t_lims]

process_gape_array = [x[...,gape_t_lims[0]:gape_t_lims[1]] for x in quin_gape_array]
mean_gape_val = [x.mean(axis=-1) for x in process_gape_array]

# Divide into groups by mean_vals
sort_inds = [np.argsort(x) for x in mean_gape_val]

#fig,ax = vz.gen_square_subplots(len(quin_gape_array), sharex=True)
#for this_dat, this_ax in zip(mean_gape_val, ax.flatten()):
#    this_ax.hist(this_dat, bins = 15) 
#plt.show()

min_val = np.min([x.min(axis=None) for x in quin_gape_array])
max_val = np.max([x.max(axis=None) for x in quin_gape_array])

fig,ax = vz.gen_square_subplots(len(quin_gape_array))
for this_dat, this_ax, this_inds in zip(quin_gape_array, ax.flatten(), sort_inds):
    this_ax.imshow(this_dat[this_inds], 
            aspect='auto', interpolation = 'nearest', cmap = 'viridis',
            vmin = min_val, vmax = max_val)
fig.savefig(os.path.join(plot_dir, 'sucrose_sub_quin_emg.png'))
#plt.show()

fig,ax = vz.gen_square_subplots(len(quin_gape_array))
for this_dat, this_ax, this_inds in zip(process_gape_array, ax.flatten(), sort_inds):
    this_ax.imshow(this_dat[this_inds], 
            aspect='auto', interpolation = 'nearest', cmap = 'viridis',
            vmin = min_val, vmax = max_val)
fig.savefig(os.path.join(plot_dir, 'sucrose_sub_quin_emg_cut.png'))
#plt.show()

# Simply dividing into equally sized groups doesn't make sense
groups = 2
group_labels = np.arange(groups)
cluster_inds = [KMeans(n_clusters=groups).fit(dat.reshape(-1,1)).labels_ \
                    for dat in mean_gape_val]
cluster_sort_inds = [np.argsort(x) for x in cluster_inds]
cluster_counts = [np.cumsum(list(Counter(sorted(x)).values())) for x in cluster_inds]

# Cherry picking sessions
wanted_inds = [0,2,3,4,6,7,8,9]

#fig,ax = vz.gen_square_subplots(len(quin_gape_array))
#for num, this_ax in enumerate(ax.flatten()):
#    if num not in wanted_inds:
#        continue
#    this_dat = process_gape_array[num]
#    this_inds = cluster_sort_inds[num]
#    this_lines = cluster_counts[num]
#    this_ax.imshow(this_dat[this_inds], 
#            aspect='auto', interpolation = 'nearest', cmap = 'viridis')
#    for x in this_lines:
#        this_ax.axhline(x - 0.5, color = 'red', linewidth = 2)
#plt.show()

## Pick out above 'wanted_inds' session
## Sort clusters by low to high emg
## Get mean emg per cluster
## Might not see big changes for all sessions, but maybe can make argument
## about DIFFERENCE in emg related to changes in neural activity

fin_session_inds = fin_bool_inds[np.array(wanted_inds)]

# Instead of indexing from previous quin_gape_array,
# index from wanted_gape_array to make sure fin_session_inds are correct

wanted_quin_array = [wanted_gape_array[i][1] for i in fin_session_inds]
#wanted_quin_array = [quin_gape_array[i] for i in wanted_inds]

process_gape_array = [x[...,gape_t_lims[0]:gape_t_lims[1]] for x in wanted_quin_array]
mean_gape_val = [x.mean(axis=-1) for x in process_gape_array]

#groups = 2
#group_labels = np.arange(groups)
cluster_inds = [KMeans(n_clusters=groups).fit(dat.reshape(-1,1)).labels_ \
                    for dat in mean_gape_val]
cluster_sort_inds = [np.argsort(x) for x in cluster_inds]
cluster_counts = [np.cumsum(list(Counter(sorted(x)).values())) for x in cluster_inds]

mean_clustered_vals = []
for ind in range(len(process_gape_array)):
    this_dat = process_gape_array[ind]
    this_inds = cluster_inds[ind]
    clustered_trials = [this_dat[this_inds==x] for x in group_labels]
    mean_clustered_vals.append([np.round(x.mean(axis=None),2) for x in clustered_trials])

sorted_cluster_order = [np.argsort(x) for x in mean_clustered_vals]
cluster_map = [dict(zip(group_labels, x)) for x in sorted_cluster_order]
fin_cluster_inds = [np.array([this_map[x] for x in this_inds]) \
        for this_map, this_inds in zip(cluster_map, cluster_inds)]
fin_cluster_sort_inds = [np.argsort(x) for x in fin_cluster_inds]
fin_cluster_cum_counts = [np.cumsum(list(Counter(sorted(x)).values())) for x in fin_cluster_inds]
fin_cluster_counts = [list(Counter(sorted(x)).values()) for x in fin_cluster_inds]
fin_mean_clustered_vals = [np.array(x)[y] \
        for x,y in zip(mean_clustered_vals, sorted_cluster_order)]

# Plot clustered emg from uncut data for trials
fig,ax = vz.gen_square_subplots(len(wanted_quin_array), sharex=True)
for num, this_dat in enumerate(wanted_quin_array):
    this_ax = ax.flatten()[num]
    this_inds = fin_cluster_sort_inds[num]
    this_lines = fin_cluster_cum_counts[num]
    this_ax.pcolormesh(cut_real_time, np.arange(this_dat.shape[0] + 1),
            this_dat[this_inds], cmap = 'viridis') 
    #this_ax.imshow(this_dat[this_inds], 
    #        aspect='auto', interpolation = 'nearest', cmap = 'viridis')
    for x in this_lines:
        this_ax.axhline(x, color = 'red', linewidth = 2)
    #this_ax.set_title(fin_mean_clustered_vals[num])
    this_ax.axvline(0, color = 'red', linestyle = '--')
plt.suptitle(f'Plot time = ({cut_real_time.min()}, {cut_real_time.max()})')
plt.show()

fig.savefig(os.path.join(plot_dir, 'sorted_whole_emg_reponses.png'))
plt.close(fig)

fin_gape_lims = [x+time_lims[0] for x in gape_t_lims]
# Plot clustered emg from cut data for trials
fig,ax = vz.gen_square_subplots(len(wanted_quin_array), sharex=True)
for num, this_dat in enumerate(process_gape_array):
    this_ax = ax.flatten()[num]
    this_inds = fin_cluster_sort_inds[num]
    this_lines = fin_cluster_cum_counts[num]
    this_ax.pcolormesh(np.arange(fin_gape_lims[0], fin_gape_lims[1]) - 2000, 
            np.arange(this_dat.shape[0] + 1),
            this_dat[this_inds], cmap = 'viridis') 
    #this_ax.imshow(this_dat[this_inds], 
    #        aspect='auto', interpolation = 'nearest', cmap = 'viridis')
    for x in this_lines:
        this_ax.axhline(x, color = 'red', linewidth = 2)
    #this_ax.set_title(fin_mean_clustered_vals[num])
plt.suptitle(f'Plot time = {np.array(gape_t_lims) - stim_t}')
plt.show()

fig.savefig(os.path.join(plot_dir, 'sorted_cut_emg_reponses.png'))
plt.close(fig)

# Plot change in prob across sessions
fig,ax = plt.subplots()
s_mult = 50
for ind in group_labels:
    ax.scatter(np.ones(len(fin_mean_clustered_vals))*ind, 
                [x[ind] for x in fin_mean_clustered_vals],
                s = [x[ind]*s_mult for x in fin_cluster_counts],
                alpha = 0.7)
    #ax.scatter(np.ones(len(fin_mean_clustered_vals))*1, 
    #        [x[1] for x in fin_mean_clustered_vals],
    #            s = [x[1]*s_mult for x in fin_cluster_counts],
    #            alpha = 0.7)
for vals in fin_mean_clustered_vals:
    ax.plot(group_labels, vals, color = 'grey', alpha = 0.7)
#plt.show()
fig.savefig(os.path.join(plot_dir, 'mean_clustered_emg_prob.png'))
plt.close(fig)

############################################################
#| \ | | ___ _   _ _ __ __ _| | |  _ \  __ _| |_ __ _ 
#|  \| |/ _ \ | | | '__/ _` | | | | | |/ _` | __/ _` |
#| |\  |  __/ |_| | | | (_| | | | |_| | (_| | || (_| |
#|_| \_|\___|\__,_|_|  \__,_|_| |____/ \__,_|\__\__,_|
############################################################

# Given the above clusters, extract neural data and cluster similarly
fin_file_list = [file_list[i] for i in fin_session_inds]
fin_basenames = [os.path.basename(x) for x in fin_file_list]
fin_dirnames = [os.path.dirname(x) for x in fin_file_list]

neuron_pval_list = []

#this_session_ind = 0
for this_session_ind in range(len(fin_file_list)):
    this_dir = fin_dirnames[this_session_ind]
    this_basename = fin_basenames[this_session_ind]
    dat = ephys_data(this_dir)
    dat.get_spikes()
    dat.check_laser()
    #dat.laser_durations

    quin_laser = dat.laser_durations[quin_ind]
    quin_spikes = dat.spikes[quin_ind] 
    quin_off_spikes = quin_spikes[quin_laser == 0]

    this_trial_clusters = fin_cluster_inds[this_session_ind]
    cluster_inds = [np.where(this_trial_clusters==i)[0] for i in group_labels]
    quin_spikes_clustered = [quin_off_spikes[i] for i in cluster_inds]

    # Cut to same time duration as gapes (for consistency)

    kern_width = 250

    quin_firing = time_box_conv(quin_off_spikes, kern_width)
    # Zscore activity of each neuron
    zscore_quin_firing = np.stack([zscore(quin_firing[:,i],axis=None) \
                                        for i in range(quin_firing.shape[1])]).swapaxes(0,1) 

    #quin_firing_clustered = [time_box_conv(x, kern_width) for x in quin_spikes_clustered] 
    quin_firing_clustered = [zscore_quin_firing[i] for i in cluster_inds]
    quin_firing_clustered = [x[...,fin_gape_lims[0]:fin_gape_lims[1]] for x in quin_firing_clustered]

    mean_quin_firing = [x.mean(axis=0) for x in quin_firing_clustered]
    std_quin_firing = [x.std(axis=0) for x in quin_firing_clustered]

    x = np.arange(*(np.array(fin_gape_lims) - 2000))
    fig,ax = plt.subplots(len(mean_quin_firing))
    for num, (this_dat, this_ax) in enumerate(zip(mean_quin_firing, ax.flatten())):
        #im = this_ax.imshow(this_dat, interpolation='nearest', aspect='auto',
        im = this_ax.pcolormesh(x, np.arange(this_dat.shape[0]),
                this_dat, vmin = -2, vmax = 2, cmap = 'jet')
        this_ax.set_ylabel(f'Cluster {num}')
    ax[1].set_xlabel('Time post-stim (ms)')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle('Mean population firing per cluster')
    fig.savefig(os.path.join(plot_dir, 
        f'session_{this_session_ind}' + "_" + 'mean_firing_per_cluster.png'))
    plt.close(fig)
    #plt.show()

    #x = np.arange(*(np.array(fin_gape_lims) - 2000))
    #fig,ax = plt.subplots(len(mean_quin_firing[0]), sharex=True,
    #        figsize = (5,len(mean_quin_firing[0])))
    #for num, this_ax in enumerate(ax.flatten()):
    #    this_ax.plot(x, mean_quin_firing[0][num])
    #    this_ax.plot(x, mean_quin_firing[1][num])
    #    this_ax.fill_between(x = x,
    #            y1 = mean_quin_firing[0][num] + std_quin_firing[0][num],
    #            y2 = mean_quin_firing[0][num] - std_quin_firing[0][num],
    #            alpha = 0.5)
    #    this_ax.fill_between(x = x,
    #            y1 = mean_quin_firing[1][num] + std_quin_firing[1][num],
    #            y2 = mean_quin_firing[1][num] - std_quin_firing[1][num],
    #            alpha = 0.5)
    #ax[-1].set_xlabel('Time post-stim (ms)')
    #plt.show()

    # Calculate number of neurons with significant differences
    # Convert to dataframe
    inds = [np.array(list(np.ndindex(x.shape))) for x in quin_firing_clustered] 
    quin_frames = [pd.DataFrame(dict(
                        cluster = num,
                        trials = this_inds[:,0],
                        neurons = this_inds[:,1],
                        time = this_inds[:,2],
                        vals = this_dat.flatten()))
                    for num, (this_inds, this_dat) in enumerate(zip(inds, quin_firing_clustered))]
    fin_quin_frame = pd.concat(quin_frames)
    fin_quin_frame['real_time'] = x[fin_quin_frame['time']]

    # Downsample for ANOVA
    binsize = 250
    bincount = int(np.diff(fin_gape_lims)[0]/binsize)
    fin_quin_frame['time_bins'] = pd.cut(fin_quin_frame['time'], bincount, 
           labels = np.arange(bincount))

    group_cols = ['cluster','trials','neurons','time_bins']
    bin_quin_frame = fin_quin_frame.groupby(group_cols).mean().reset_index()
    bin_quin_frame.dropna(inplace=True)

    group_bin_quin_frame = [x[1] for x in list(bin_quin_frame.groupby('neurons'))]
    anova_list = [pg.anova(data = this_dat,
                dv = 'vals', between = ['cluster','time_bins']) \
                        for this_dat in group_bin_quin_frame]
    alpha = 0.05
    pval_list = [np.round(x['p-unc'][0],3) for x in anova_list]
    pval_sig = np.array(pval_list)<=alpha
    pval_sort_inds = np.argsort(pval_list)
    neuron_pval_list.append(pval_list)

    x = np.arange(*(np.array(fin_gape_lims) - 2000))
    fig,ax = plt.subplots(len(mean_quin_firing[0]), sharex=True, sharey=True,
            figsize = (7,len(mean_quin_firing[0])))
    for ind, this_ax in enumerate(ax.flatten()):
        num = pval_sort_inds[ind]
        this_ax.plot(x, mean_quin_firing[0][num])
        this_ax.plot(x, mean_quin_firing[1][num])
        this_ax.fill_between(x = x,
                y1 = mean_quin_firing[0][num] + std_quin_firing[0][num],
                y2 = mean_quin_firing[0][num] - std_quin_firing[0][num],
                alpha = 0.5)
        this_ax.fill_between(x = x,
                y1 = mean_quin_firing[1][num] + std_quin_firing[1][num],
                y2 = mean_quin_firing[1][num] - std_quin_firing[1][num],
                alpha = 0.5)
        this_ax.set_ylabel(pval_list[num])
        if pval_sig[num]:
            this_ax.axvline(x[0], linewidth = 2, color = 'red')
    ax[-1].set_xlabel('Time post-stim (ms)')
    plt.suptitle(f'Session {this_session_ind} : Firing per cluster')
    fig.savefig(os.path.join(plot_dir, f'session_{this_session_ind}' + \
            "_" + 'firing_per_cluster.png'))
    plt.close(fig)
    #plt.show()

    ############################################################
    ## Trial Decomposition 
    ############################################################
    # Calculate "Euclidean distance" between trials
    # Perform tensor decomposition on zscored trails
    quin_firing_cut = zscore_quin_firing[...,fin_gape_lims[0]:fin_gape_lims[1]]

    data = quin_firing_cut# ... specify a numpy array holding the tensor you wish to fit

    # Fit an ensemble of models, 4 random replicates / optimization runs per model rank
    #ensemble = tt.Ensemble(fit_method="ncp_hals")
    #ensemble.fit(data, ranks=range(1, 5), replicates=4)

    #fig, axes = plt.subplots(1, 2)
    ## plot reconstruction error as a function of num components.
    #tt.plot_objective(ensemble, ax=axes[0])   
    ## plot model similarity as a function of num components.
    #tt.plot_similarity(ensemble, ax=axes[1])  
    #fig.tight_layout()
    #plt.show()

    # Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
    num_components = 3
    replicates= 5
    fits = [tt.ncp_hals(quin_firing_cut, rank = 3, verbose=True) for i in range(replicates)]
    objs = [x.obj for x in fits]
    fin_model = fits[np.argmin(objs)]

    ## plot the low-d factors
    #tt.plot_factors(ensemble.factors(num_components)[replicate])  
    #plt.show()

    #trial_factors = ensemble.factors(num_components)[replicate].factors[0]
    #trial_factors = ensemble.factors(num_components)[replicate].factors[0]
    trial_factors = fin_model.factors.factors[0]
    zscore_trial_factors = zscore(trial_factors,axis=0)
    trial_factors_plot = trial_factors + (np.random.random(trial_factors.shape)-0.5)*0.3

    sorted_trial_factors = zscore_trial_factors[np.concatenate(cluster_inds)]
    trial_factor_dists = squareform(pdist(sorted_trial_factors))

    fig,ax = plt.subplots(1,2)
    ax[0].matshow(zscore_trial_factors[np.concatenate(cluster_inds)], cmap = 'viridis')
    trial_markers = np.cumsum([len(x) for x in cluster_inds])
    for x in trial_markers:
        ax[0].axhline(x - 0.5, color = 'red', linewidth = 2)
    ax[0].set_ylabel('Trials')
    ax[0].set_xlabel('Trial Component')
    ax[0].set_title('Sorted Trial components')
    ax[1].matshow(trial_factor_dists, cmap = 'viridis')
    for x in trial_markers:
        ax[1].axhline(x - 0.5, color = 'red', linewidth = 4, linestyle = '--')
        ax[1].axvline(x - 0.5, color = 'red', linewidth = 4, linestyle = '--')
    ax[1].set_title('Distance between Sorted Trial components')
    #plt.suptitle(this_basename)
    plt.suptitle(f'Session {this_session_ind}' + \
            " ::: " + f'{quin_firing_cut.shape[1]} neurons')
    plt.subplots_adjust(top = 0.8)
    #fig.savefig(os.path.join(plot_dir, this_basename + "_" + 'trial_comp_dists.png'))
    fig.savefig(os.path.join(plot_dir, 
        f'session_{this_session_ind}' + "_" + 'trial_comp_dists.png'))
    plt.close(fig)
    #plt.show()

    ############################################################
    ## Cluster trials by single PC across neurons 
    ############################################################
    quin_firing_long = np.reshape(quin_firing_cut.swapaxes(0,1), (quin_firing_cut.shape[1],-1))
    pca_obj = PCA(n_components = 1).fit(quin_firing_long.T)
    pca_quin_firing = np.squeeze(np.stack([pca_obj.transform(x.T) for x in quin_firing_cut]))
    sorted_quin_pca = pca_quin_firing[np.concatenate(cluster_inds)]

    pca_trial_dists = squareform(pdist(sorted_quin_pca))

    fig,ax = plt.subplots()
    ax.matshow(pca_trial_dists, cmap = 'viridis')
    for x in trial_markers:
        ax.axhline(x - 0.5, color = 'red', linewidth = 4, linestyle = '--')
        ax.axvline(x - 0.5, color = 'red', linewidth = 4, linestyle = '--')
    ax.set_title('Distance between Sorted Trial components')
    plt.suptitle(f'PCA Trial Factors - Session {this_session_ind}')
    plt.subplots_adjust(top = 0.8)
    #plt.show()
    fig.savefig(os.path.join(plot_dir, 
        f'session_{this_session_ind}' + "_" + 'trial_comp_dists_pca.png'))
    plt.close(fig)


    ############################################################
    ## Trial Decomposition with ONLY significant neurons 
    ############################################################
    if any(pval_sig):
        sig_quin_cut = quin_firing_cut[:,pval_sig]

        num_components = 3
        replicates= 5
        fits = [tt.ncp_hals(sig_quin_cut, rank = 3, verbose=True) for i in range(replicates)]
        objs = [x.obj for x in fits]
        fin_model = fits[np.argmin(objs)]

        trial_factors = fin_model.factors.factors[0]
        zscore_trial_factors = zscore(trial_factors,axis=0)
        trial_factors_plot = trial_factors + (np.random.random(trial_factors.shape)-0.5)*0.3

        sorted_trial_factors = zscore_trial_factors[np.concatenate(cluster_inds)]
        trial_factor_dists = squareform(pdist(sorted_trial_factors))

        fig,ax = plt.subplots(1,2)
        ax[0].matshow(zscore_trial_factors[np.concatenate(cluster_inds)], cmap = 'viridis')
        trial_markers = np.cumsum([len(x) for x in cluster_inds])
        for x in trial_markers:
            ax[0].axhline(x - 0.5, color = 'red', linewidth = 2)
        ax[0].set_ylabel('Trials')
        ax[0].set_xlabel('Trial Component')
        ax[0].set_title('Sorted Trial components')
        ax[1].matshow(trial_factor_dists, cmap = 'viridis')
        for x in trial_markers:
            ax[1].axhline(x - 0.5, color = 'red', linewidth = 4, linestyle = '--')
            ax[1].axvline(x - 0.5, color = 'red', linewidth = 4, linestyle = '--')
        ax[1].set_title('Distance between Sorted Trial components')
        plt.suptitle(f'Session {this_session_ind}' + " ::: " + \
                f'Sig nrns : {sum(pval_sig)}/{len(pval_sig)}')
        plt.subplots_adjust(top = 0.8)
        #plt.show()
        fig.savefig(os.path.join(plot_dir, 
            f'session_{this_session_ind}' + "_" + 'sig_nrn_trial_comp_dists.png'))
        plt.close(fig)
    else:
        fig,ax = plt.subplots()
        plt.suptitle('No sig neurons')
        fig.savefig(os.path.join(plot_dir, 
            f'session_{this_session_ind}' + "_" + 'sig_nrn_trial_comp_dists.png'))
        plt.close(fig)


############################################################
## Trials per cluster 
############################################################
fin_trial_fracs = [x/max(x) for x in cluster_counts]
fig,ax = plt.subplots()
for ind in group_labels:
    ax.bar(np.arange(len(fin_trial_fracs)), 
        np.array([x[ind] for x in fin_trial_fracs]), alpha = 0.7,
        zorder = groups - ind, label = f'Clust {ind}')
plt.legend()
plt.suptitle('Fraction of trials per cluster')
plt.xlabel('Session Number')
plt.ylabel('Cumulative Fraction')
fig.savefig(os.path.join(plot_dir, 'fraction_per_cluster.png')) 
plt.close(fig)
#plt.show()

############################################################
## Significant neurons per session 
############################################################
alpha = 0.05
sig_counts = [np.mean(np.array(x)<=alpha) for x in neuron_pval_list]
plt.hist(sig_counts)
fig = plt.gcf()
plt.xlabel('Fraction of Neurons with significant change')
plt.ylabel('Frequency')
plt.suptitle('Fraction of signficant neurons per session')
fig.savefig(os.path.join(plot_dir, 'sig_nrn_count.png'))
plt.close(fig)
