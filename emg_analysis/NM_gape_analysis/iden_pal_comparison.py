"""
Compare Identity and palatability information between gape trial clusters
But since we only have a single taste, I guess we can compare distance
from 1) Mean sucrose response, and 2) Mean quinine response (regardless of clusters)
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
import pylab as plt
from glob import glob
import pandas as pd
import seaborn as sns
import pingouin as pg
from scipy.stats import (
        zscore, 
        spearmanr, 
        linregress,
        entropy)
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import itertools as it

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

sys.path.append('/media/bigdata/firing_space_plot/NM_gape_analysis')
from return_gape_neural_data import *

plot_dir = '/media/bigdata/firing_space_plot/NM_gape_analysis/plots'

file_list_path = '/media/fastdata/NM_sorted_data/h5_file_list.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
dir_names = [os.path.dirname(x) for x in file_list]

gape_t_lims = [2750,4500]
off_gape_array, cluster_inds, bool_inds = return_gape_data()
off_spikes_list, off_firing_list, off_firing_time = return_neural_data()
file_list, basename_list = return_names()

# Make plots for off-firing to make sure all looks good
long_off_firing = [x.reshape((-1, *x.shape[2:])).swapaxes(1,0) for x in off_firing_list]
# Downsample for easier plotting
long_off_firing = [x[...,::10] for x in long_off_firing]

#for x in long_off_firing:
#    vz.firing_overview(x);
#plt.show()

quin_gape_array = [x[1] for x in off_gape_array]
cluster_sort_inds = [np.argsort(x) for x in cluster_inds]
cluster_div = [np.sum(x==0) for x in cluster_inds]
sorted_quin_array = [x[y] for x,y in zip(quin_gape_array, cluster_sort_inds)]
cut_sorted_quin = [x[..., gape_t_lims[0]:gape_t_lims[1]] for x in sorted_quin_array]

clustered_cut_quin = np.stack(
        [[x[:y].mean(axis=None), x[y:].mean(axis=None)] \
                for x,y in zip(cut_sorted_quin, cluster_div)]
        ) 
clustered_quin_diff = np.diff(clustered_cut_quin,axis=-1)

quin_firing_list = [x[taste_dict['quin']] for x in off_firing_list]
# Sort quinine to have same order of trials as gape array
quin_firing_list = [x[y] for x,y in zip(quin_firing_list, cluster_sort_inds)]
suc_firing_list = [x[taste_dict['suc']] for x in off_firing_list]

# Do lower emg trials lie closer to SUC on the quin-suc axis
# Or are they just different
iden_lims = [300,750]
pal_lims = [750,1250]
epoch_names = ['iden','pal']
stim_t = 2000
epoch_lims = [iden_lims, pal_lims]
epoch_lims = [[x+stim_t for x in y] for y in epoch_lims]

# Use spikes rather than firing to avoid temporal bleeding
# We're also only looking at mean firing, so mean spikes should be fine
firing_epochs = [[x[...,lims[0]:lims[1]].mean(axis=-1) for lims in epoch_lims]
    for x in off_spikes_list] 

firing_epochs = [[x+(np.random.random(x.shape)*1e-3) for x in y]\
        for y in firing_epochs]

quin_epochs = [np.stack([this_epoch[taste_dict['quin']] for this_epoch in this_session])
        for this_session in firing_epochs]
suc_epochs = [np.stack([this_epoch[taste_dict['suc']] for this_epoch in this_session])
        for this_session in firing_epochs]

############################################################
## Number of neurons which show significant differences between quinine
## clusters in each epoch
firing_frame_list=  []
for epoch_num in range(len(epoch_lims)):
    for session_num in range(len(quin_epochs)):
        quin_dat = quin_epochs[session_num][epoch_num]
        this_cluster_div = cluster_div[session_num]
        cluster_labels = np.zeros(quin_dat.shape[0])
        cluster_labels[this_cluster_div:] = 1
        inds = np.array(list(np.ndindex(quin_dat.shape)))
        this_frame = pd.DataFrame(
                dict(
                    trial = inds[:,0],
                    neuron = inds[:,1],
                    cluster = cluster_labels[inds[:,0]],
                    value = quin_dat.flatten(),
                    epoch = epoch_num,
                    session = session_num
                    )
                )
        firing_frame_list.append(this_frame)
firing_frame = pd.concat(firing_frame_list)

############################################################
# Significance of difference in population activity
def get_dists(x,y = None):
    if y is None:
        y = x.copy()
    full_dists = distance_matrix(x,y)
    inds = np.indices(full_dists.shape)
    wanted_inds = inds[0] > inds[1]
    fin_dists = full_dists[wanted_inds]
    return fin_dists

dist_frame_list = []
iters = list(it.product(firing_frame.epoch.unique(), firing_frame.session.unique()))
for epoch, session in tqdm(iters):
    this_dat = quin_epochs[session][epoch]
    norm_dat = zscore(this_dat, axis = 0)
    this_cluster_div = cluster_div[session]
    clust_dat = [norm_dat[:this_cluster_div], norm_dat[this_cluster_div:]]
    ############################################################ 
    # Look at distances
    intra_dists = [get_dists(x) for x in clust_dat] 
    inter_dists = get_dists(clust_dat[0], clust_dat[1])
    intra_frames = [pd.DataFrame(dict(clust = num, type = 'intra', dist = x)) \
            for num, x in enumerate(intra_dists)]
    inter_frame = pd.DataFrame(
            dict(
                clust = 'both',
                type = 'inter',
                dist = inter_dists
                )
            )
    fin_frame = pd.concat([*intra_frames, inter_frame])
    fin_frame['epoch'] = epoch
    fin_frame['session'] = session
    dist_frame_list.append(fin_frame)
fin_dist_frame = pd.concat(dist_frame_list)

# Replace cluster numbers with labels
label_dict = {0 : 'low', 1 : 'high', 'both' : 'both'}
fin_dist_frame['clust'] = [label_dict[x] for x in fin_dist_frame['clust']]

group_frame = [x[1] for x in list(fin_dist_frame.groupby(['session','epoch']))]
for x in group_frame:
    x['norm_dist'] = x['dist'] / x.loc[x.clust == 'low'].dist.mean()
fin_dist_frame = pd.concat(group_frame)
fin_dist_frame['dist_type'] = fin_dist_frame['type'].astype('str') \
        + '_' + fin_dist_frame['clust'].astype('str')
fin_dist_frame.drop(columns = ['clust','type'], inplace=True)

sns.catplot(
        data = fin_dist_frame,
        x = 'dist_type',
        y = 'norm_dist',
        col = 'session',
        row = 'epoch',
        kind = 'box'
        )
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, f'cluster_distance_comparison.png'))
plt.close(fig)
#plt.show()

#mean_dist_frame = fin_dist_frame.groupby(['session','epoch','dist_type'])\
#        .mean().reset_index()
#
#sns.boxplot(
#       data = mean_dist_frame,
#       x = 'dist_type',
#       y = 'dist'
#        )
#plt.show()


############################################################
# Look at individual epochs first, and then sum of epochs
alpha = 0.05
grouped_list = list(firing_frame.groupby(['epoch','session','neuron']))
group_labels = [x[0] for x in grouped_list]
group_data = [x[1] for x in grouped_list]
p_val_list = [pg.kruskal(
                data = x,
                dv = 'value',
                between = 'cluster'
                )['p-unc'].values[0] for x in group_data]
pval_frame = pd.DataFrame(
        group_labels,
        columns = ['epoch','session','neuron']
        )
pval_frame['p_val'] = p_val_list
pval_frame['sig_bool'] = pval_frame['p_val'] < alpha

epoch_sig_comparisons = pval_frame.groupby(['epoch','session'])\
        .mean().reset_index()[['epoch','session','sig_bool']]
epoch_sig_comparisons['emg_diff'] = clustered_quin_diff[epoch_sig_comparisons['session']]

g = sns.lmplot(
        data = epoch_sig_comparisons,
        x = 'emg_diff',
        y = 'sig_bool',
        col = 'epoch',
        height = 4,
        aspect = 1,
        robust = True,
        n_boot = 100
        )
for this_ax in g.axes[0]: 
    this_ax.set_ylabel('# of Sig Neurons')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, f'sig_neurons_per_epoch_vs_emg_diff.png'))
plt.close(fig)
#plt.show()

epochwise_or_sig = [bool(x[1]['sig_bool'].sum()) \
        for x in list(pval_frame.groupby(['session', 'neuron']))]
all_sig = pval_frame.groupby(['session', 'neuron']).mean().reset_index()
all_sig['sig_bool'] = epochwise_or_sig
all_sig = pval_frame.groupby(['session'])\
        .mean().reset_index()[['session','sig_bool']]
all_sig['emg_diff'] = clustered_quin_diff[all_sig['session']]

sns.lmplot(
        data = all_sig,
        x = 'emg_diff',
        y = 'sig_bool',
        robust = True,
        )
plt.tight_layout()
plt.title('Iden or Pal Sig')
plt.ylabel('# of Sig Neurons')
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, f'sig_neurons_both_epochs_vs_emg_diff.png'))
plt.close(fig)
#plt.show()

########################################
# Run anova over full response (0-2500 ms post-stim)


############################################################

# LDA plots for suc and quin for each session
# To confirm there is separability from the get-go
quin_cluster_labels = ['low_emg','high_emg']
mean_diffs = np.zeros((len(quin_epochs), len(epoch_lims)))
quin_entropy_list = [[],[]]
#cmap = plt.get_cmap('tab10')
vline_colors = ['r','k']
for epoch_num in range(len(epoch_lims)):
    fig,ax = vz.gen_square_subplots(len(quin_epochs), figsize = (10,7))
    for session_num in range(len(quin_epochs)):
        quin_dat = quin_epochs[session_num][epoch_num]
        suc_dat = suc_epochs[session_num][epoch_num]
        concat_dat = np.concatenate([suc_dat, quin_dat], axis=0)
        zscore_concat_dat = zscore(concat_dat, axis=0)
        labels = np.squeeze([[0]*len(quin_dat) + [1]*len(suc_dat)])
        #pca_dat = PCA(n_components = 2).fit_transform(concat_dat)
        lda_obj = LDA().fit(zscore_concat_dat, labels)
        lda_dat = lda_obj.transform(zscore_concat_dat)
        #bins = np.linspace(lda_dat.min(), lda_dat.max(), 15)
        # Fix bins so entropy is comparable across sessions
        bins = np.linspace(-3, 3, 15)
        lda_groups = [lda_dat[labels==x] for x in np.unique(labels)]
        group_dist = []
        for num, x in enumerate(lda_groups):
            freq, bins, _ = ax.flatten()[session_num].hist(x, alpha = 0.7, bins = bins,
                        label = taste_names[num])
            group_dist.append(freq)
        # Calculate mean projection for quin clusters and plot
        this_cluster_div = cluster_div[session_num]
        cluster_quin_dat = [lda_groups[1][:this_cluster_div], lda_groups[1][this_cluster_div:]]
        cluster_quin_mean = [x.mean() for x in cluster_quin_dat]
        #quin_entropy = entropy(group_dist[1]) 
        quin_entropy = np.var(group_dist[1]) 
        quin_entropy_list[epoch_num].append(quin_entropy)
        mean_diffs[session_num, epoch_num] = np.diff(cluster_quin_mean)
        for num,x in enumerate(cluster_quin_mean):
            ax.flatten()[session_num].axvline(x, 
                    c=vline_colors[num], linewidth = 5, alpha = 0.7,
                    label = quin_cluster_labels[num])
        if session_num == 0:
            ax.flatten()[session_num].legend()
        #ax.flatten()[session_num].scatter(pca_dat[:,0], pca_dat[:,1], c = labels)
    fig.suptitle(epoch_names[epoch_num])
    fig.savefig(os.path.join(plot_dir, f'quin_suc_lda_{epoch_names[epoch_num]}.png'))
    plt.close(fig)
#plt.show()

plt.figure()
plt.hist(mean_diffs[:,0], alpha = 0.7, bins = 20, label = 'Identity')
plt.hist(mean_diffs[:,1], alpha = 0.7, bins = 20, label = 'Palatability')
plt.axvline(0, color = 'red')
plt.legend()
plt.xlabel('LDA Diff' + '\n -ve = low_emg less like Suc, +ve = low_emg more like suc')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'quin_suc_lda_diff.png'))
plt.close()
#plt.show()

# Is magnitude of difference between emg correlated to difference in lda
clustered_quin_diff = np.diff(clustered_cut_quin,axis=-1)
fig,ax = plt.subplots(2,1, figsize = (5,10))
ax[0].scatter(clustered_quin_diff, mean_diffs[:,0])
reg = linregress(clustered_quin_diff.flatten(), mean_diffs[:,0].flatten())
ax[0].plot(clustered_quin_diff, reg.intercept + reg.slope*clustered_quin_diff, 'r') 
corr_out = spearmanr(clustered_quin_diff, mean_diffs[:,0])
ax[0].set_title('Identity' + '\n' + f'Corr {corr_out[0] : .3}, p-val {corr_out[1] : .3}')
ax[1].scatter(clustered_quin_diff, mean_diffs[:,1])
reg = linregress(clustered_quin_diff.flatten(), mean_diffs[:,1].flatten())
ax[1].plot(clustered_quin_diff, reg.intercept + reg.slope*clustered_quin_diff, 'r') 
corr_out = spearmanr(clustered_quin_diff, mean_diffs[:,1])
ax[1].set_title('Palatability' + '\n' + f'Corr {corr_out[0] : .3}, p-val {corr_out[1] : .3}')
fig.suptitle('Difference in EMG vs Difference in Firing LDA')
for this_ax in ax:
    this_ax.set_xlabel('LDA Difference')
    this_ax.set_ylabel('EMG Difference')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'emgdiff_vs_ldadiff.png'))
plt.close()

# Same analysis as above but lda differences are scaled by entropy
# of quinine distribution
# Is magnitude of difference between emg correlated to difference in lda
quin_entropy_array = np.stack(quin_entropy_list).T
scaled_mean_diffs = mean_diffs / quin_entropy_array
#keep_inds = np.isinf(scaled_mean_diffs).sum(axis=-1) == 0
keep_inds = (scaled_mean_diffs > 2).sum(axis=-1) == 0
scaled_mean_diffs = scaled_mean_diffs[keep_inds]

plt.figure()
plt.hist(scaled_mean_diffs[:,0], alpha = 0.7, bins = 20, label = 'Identity')
plt.hist(scaled_mean_diffs[:,1], alpha = 0.7, bins = 20, label = 'Palatability')
plt.axvline(0, color = 'red')
plt.legend()
plt.xlabel('LDA Diff')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plot_dir, 'quin_suc_scaled_lda_diff.png'))
plt.close()
#plt.show()

clustered_quin_diff = np.diff(clustered_cut_quin,axis=-1)[keep_inds]
fig,ax = plt.subplots(2,1, figsize = (5,10))
ax[0].scatter(clustered_quin_diff, scaled_mean_diffs[:,0])
reg = linregress(clustered_quin_diff.flatten(), scaled_mean_diffs[:,0].flatten())
ax[0].plot(clustered_quin_diff, reg.intercept + reg.slope*clustered_quin_diff, 'r') 
corr_out = spearmanr(clustered_quin_diff, scaled_mean_diffs[:,0])
ax[0].set_title('Identity' + '\n' + f'Corr {corr_out[0] : .3}, p-val {corr_out[1] : .3}')
ax[1].scatter(clustered_quin_diff, scaled_mean_diffs[:,1])
reg = linregress(clustered_quin_diff.flatten(), scaled_mean_diffs[:,1].flatten())
ax[1].plot(clustered_quin_diff, reg.intercept + reg.slope*clustered_quin_diff, 'r') 
corr_out = spearmanr(clustered_quin_diff, scaled_mean_diffs[:,1])
ax[1].set_title('Palatability' + '\n' + f'Corr {corr_out[0] : .3}, p-val {corr_out[1] : .3}')
fig.suptitle('Difference in EMG vs Difference in Firing LDA')
for this_ax in ax:
    this_ax.set_xlabel('LDA Difference')
    this_ax.set_ylabel('EMG Difference')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'emgdiff_vs_scaled_ldadiff.png'))
plt.close()

#import statsmodels.api as sm
#y = mean_diffs[:,1] 
#x = clustered_quin_diff 
#x = sm.add_constant(x)
#model = sm.OLS(y, x).fit()
#influence = model.get_influence()
#cooks = influence.cooks_distance
#
#plt.scatter(clustered_quin_diff, cooks[0])
#plt.xlabel('x')
#plt.ylabel('Cooks Distance')
#plt.show()

#############################################################
## PCA plot with Suc, Quin Low EMG, Quin High EMG 
# PCA plots for suc and quin for each session
# To confirm there is separability from the get-go
cmap = plt.get_cmap('tab10')
taste_order = ['low_quin','high_quin','suc']
for epoch_num in range(len(epoch_lims)):
    fig,ax = vz.gen_square_subplots(len(quin_epochs), figsize = (10,7))
    for session_num in range(len(quin_epochs)):
        quin_dat = quin_epochs[session_num][epoch_num]
        this_cluster_div = cluster_div[session_num]
        cluster_quin_dat = [quin_dat[:this_cluster_div], quin_dat[this_cluster_div:]]
        suc_dat = suc_epochs[session_num][epoch_num]
        all_dat = [*cluster_quin_dat, suc_dat]
        labels = [[num]*len(x) for num,x in enumerate(all_dat)] 
        labels = [x for y in labels for x in y]
        concat_dat = np.concatenate(all_dat, axis=0)
        zscore_concat_dat = zscore(concat_dat, axis=0)
        pca_dat = PCA(n_components = 2).fit_transform(concat_dat)
        pca_dat_group = np.stack([pca_dat[labels==x].mean(axis=0) for x in np.unique(labels)])
        ax.flatten()[session_num].scatter(pca_dat[:,0], pca_dat[:,1], 
                c = cmap(labels), zorder = 1)
        for num, this_mean in enumerate(pca_dat_group):
            ax.flatten()[session_num].scatter(this_mean[0], this_mean[1], 
                    c = cmap(np.unique(labels)[num]), s = 200, zorder = 2, 
                    label = taste_order[num])
        if session_num == len(quin_epochs)-1:
            ax.flatten()[session_num].legend()
    fig.suptitle(epoch_names[epoch_num])
    fig.savefig(os.path.join(plot_dir, f'quin_suc_pca_{epoch_names[epoch_num]}.png'))
    plt.close(fig)

#############################################################
## Mean (Isotropic) variance of clusters

taste_order = ['low_quin','high_quin','total_quin','suc']
var_list = [[],[]] 
for epoch_num in range(len(epoch_lims)):
    for session_num in range(len(quin_epochs)):
        quin_dat = quin_epochs[session_num][epoch_num]
        this_cluster_div = cluster_div[session_num]
        cluster_quin_dat = [quin_dat[:this_cluster_div], quin_dat[this_cluster_div:]]
        suc_dat = suc_epochs[session_num][epoch_num]
        all_dat = [*cluster_quin_dat, quin_dat, suc_dat]
        concat_dat = np.concatenate(all_dat, axis=0)
        #zscore_concat_dat = zscore(concat_dat, axis=0)
        zscore_obj = StandardScaler().fit(concat_dat)
        zscore_all_dat = [zscore_obj.transform(x) for x in all_dat]
        nrn_vars = [np.var(x,axis=0) for x in zscore_all_dat]
        mean_vars = [np.mean(x) for x in nrn_vars]
        var_list[epoch_num].append(mean_vars)
var_array = np.stack(var_list)

inds = np.array(list(np.ndindex(var_array.shape)))
var_frame = pd.DataFrame(
        dict(
            epoch = inds[:,0],
            session = inds[:,1],
            group = inds[:,2],
            variance = var_array.flatten()
            )
        )

var_frame['taste'] = [taste_order[i] for i in var_frame['group']]
#var_frame['session'] = np.arange(len(var_frame))
#var_frame = var_frame.melt(id_vars = 'session', value_vars = taste_order,
#        var_name = 'taste', value_name = 'variance')

g = sns.catplot(
        data = var_frame,
        x = 'taste',
        y = 'variance',
        col = 'epoch',
        kind = 'box'
        )
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, f'quin_suc_variance.png'))
plt.close(fig)
#plt.show()

############################################################
# Calculate distances b/w all groups

im_kwargs = dict(interpolation = 'nearest', aspect = 'auto', cmap = 'viridis')
line_kwargs = dict(color = 'red', linewidth = 2, linestyle = '--')
taste_order = ['low_quin','high_quin','suc']
for epoch_num in range(len(epoch_lims)):
    fig,ax = vz.gen_square_subplots(len(quin_epochs), figsize = (10,7))
    for session_num in range(len(quin_epochs)):
        this_ax = ax.flatten()[session_num]
        quin_dat = quin_epochs[session_num][epoch_num]
        this_cluster_div = cluster_div[session_num]
        cluster_quin_dat = [quin_dat[:this_cluster_div], quin_dat[this_cluster_div:]]
        suc_dat = suc_epochs[session_num][epoch_num]
        all_dat = [*cluster_quin_dat, suc_dat]
        labels = [[num]*len(x) for num,x in enumerate(all_dat)] 
        labels = [x for y in labels for x in y]
        concat_dat = np.concatenate(all_dat, axis=0)
        zscore_concat_dat = zscore(concat_dat, axis=0)
        lens = np.cumsum([len(x) for x in all_dat])
        dist_mat = distance_matrix(zscore_concat_dat, zscore_concat_dat)
        this_ax.imshow(dist_mat, **im_kwargs)
        for this_len in lens:
            this_ax.axhline(this_len - 0.5, **line_kwargs)
            this_ax.axvline(this_len - 0.5, **line_kwargs)
    fig.suptitle(epoch_names[epoch_num])
    fig.savefig(os.path.join(plot_dir, f'quin_suc_dists_{epoch_names[epoch_num]}.png'))
    plt.close(fig)
#plt.show()

# Distances between means of each group
dist_mat_list = [[],[]]
for epoch_num in range(len(epoch_lims)):
    fig,ax = vz.gen_square_subplots(len(quin_epochs), figsize = (10,7))
    for session_num in range(len(quin_epochs)):
        this_ax = ax.flatten()[session_num]
        quin_dat = quin_epochs[session_num][epoch_num]
        this_cluster_div = cluster_div[session_num]
        cluster_quin_dat = [quin_dat[:this_cluster_div], quin_dat[this_cluster_div:]]
        suc_dat = suc_epochs[session_num][epoch_num]
        all_dat = [*cluster_quin_dat, suc_dat]
        mean_dat = np.stack([x.mean(axis=0) for x in all_dat])
        zscore_mean = zscore(mean_dat, axis=0)
        dist_mat = distance_matrix(zscore_mean, zscore_mean)
        dist_mat_list[epoch_num].append(dist_mat)
        this_ax.imshow(dist_mat, **im_kwargs)
    fig.suptitle(epoch_names[epoch_num])
    fig.savefig(os.path.join(plot_dir, f'quin_suc_mean_dists_{epoch_names[epoch_num]}.png'))
    plt.close(fig)
#plt.show()

dist_mat_array = np.stack(dist_mat_list)
mean_dist_mat = dist_mat_array.mean(axis=1)
fig,ax = plt.subplots(1, len(mean_dist_mat))
for num, (this_ax, this_dat) in enumerate(zip(ax, mean_dist_mat)):
    this_ax.imshow(this_dat, interpolation = 'nearest', aspect = 'equal', cmap = 'viridis')
    this_ax.set_title(epoch_names[num])
    fig.savefig(os.path.join(plot_dir, f'quin_suc_mean_mean_dists_.png'))
    plt.close(fig)
#plt.show()

# Plot with error bars
inds = np.array(list(np.ndindex(dist_mat_array.shape)))
dist_frame = pd.DataFrame(
        dict(
            epoch = inds[:,0],
            session = inds[:,1],
            group1 = inds[:,2],
            group2 = inds[:,3],
            dists = dist_mat_array.flatten()
            )
        )
dist_frame['group1_name'] = np.array(taste_order)[dist_frame['group1']]
dist_frame['group2_name'] = np.array(taste_order)[dist_frame['group2']]

# Remove equal and duplicates
dist_frame['comp_sets'] = [set([x[1]['group1'], x[1]['group2']]) for x in dist_frame.iterrows()]
dist_frame = dist_frame.loc[[len(x) > 1 for x in dist_frame['comp_sets']]]
dist_frame['set_str'] = [str(x) for x in dist_frame['comp_sets']]
dist_frame['comp_name'] = [np.array(taste_order)[np.array(list(x))] \
        for x in dist_frame['comp_sets']]
dist_frame['comp_str'] = [str(x) for x in dist_frame['comp_name']]
dist_frame = dist_frame.loc[dist_frame[['epoch','session','set_str']].duplicated()]

g = sns.catplot(
        data = dist_frame,
        col = 'epoch',
        x = 'comp_str',
        y = 'dists',
        kind = 'bar'
        )
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, f'quin_suc_mean_mean_dists_bar.png'))
plt.close(fig)
#plt.show()

#############################################################
## Replot emg activity to make sure output is correct
#
#fig, ax = vz.gen_square_subplots(len(off_gape_array))
#for num, (this_ax, this_dat) in enumerate(zip(ax.flatten(), sorted_quin_array)):
#    line_pos = cluster_div[num] 
#    this_ax.imshow(this_dat, 
#            aspect='auto', interpolation = 'nearest', cmap = 'viridis')
#    this_ax.axhline(line_pos - 0.5, color = 'red')
#fig.savefig(os.path.join(plot_dir, 'test_emg_plots.png'))
#
fig, ax = vz.gen_square_subplots(len(off_gape_array))
for num, (this_ax, this_dat) in enumerate(zip(ax.flatten(), cut_sorted_quin)):
    line_pos = cluster_div[num] 
    this_ax.imshow(this_dat, 
            aspect='auto', interpolation = 'nearest', cmap = 'viridis')
    this_ax.axhline(line_pos - 0.5, color = 'red')
plt.show()
#fig.savefig(os.path.join(plot_dir, 'test_emg_plots_cut.png'))
