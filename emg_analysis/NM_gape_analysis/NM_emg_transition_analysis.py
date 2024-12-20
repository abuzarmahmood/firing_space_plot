"""
Fit models to given spike array
"""

import json
import os
import pickle
import shutil
import uuid
from datetime import date, datetime
from glob import glob
from tqdm import tqdm, trange
import seaborn as sns
import pingouin as pg
import sys

import numpy as np
import pandas as pd

import sys
base_dir = '/media/bigdata/projects/pytau'
sys.path.append(base_dir)

from pytau import changepoint_model
from pytau import changepoint_preprocess
from pytau.utils import EphysData

import pymc3
import theano

sys.path.append('/media/bigdata/firing_space_plot/NM_gape_analysis')
from return_gape_neural_data import *

def convert_to_hist_array(x, bins):
    hist_array = np.zeros((len(bins)-1, *x.shape[1:]))
    inds_list = list(np.ndindex(x.shape[1:]))
    for this_ind in inds_list:
        hist_array[:, this_ind[0], this_ind[1]] = \
                np.histogram(
                        x[:,this_ind[0], this_ind[1]],
                        bins = bins
                        )[0]
    return hist_array

save_path = '/media/bigdata/firing_space_plot/NM_gape_analysis/inference_output'
plot_dir = '/media/bigdata/firing_space_plot/NM_gape_analysis/plots'

off_gape_array, cluster_inds, bool_inds = return_gape_data()
off_spikes_list, off_firing_list, off_firing_time = return_neural_data()
file_list, basename_list = return_names()

temporal_wilcoxon = pg.wilcoxon(*list(zip(*mean_cluster_inds)))

# Is there a temporal component to the clustering?
mean_cluster_inds = []
fig,ax = plt.subplots(len(cluster_inds) + 1, 1, 
        sharex=True, sharey=True, figsize = (4,15))
for x, this_ax, in zip(cluster_inds,ax[:-1]):
    x_vals = np.linspace(0,1,len(x))
    this_ax.plot(x_vals,x)
    mean_cluster_inds.append((np.mean(x_vals[x==0]),np.mean(x_vals[x>0])))
    this_ax.set_ylabel('Cluster')
ax[-1].set_xlabel('Trial location')
ax[-1].scatter(*list(zip(*mean_cluster_inds)), alpha = 0.5)
ax[-1].set_xlabel('Zero cluster mean')
ax[-1].set_ylabel('One cluster mean')
ax[-1].set_aspect('equal')
x_lin = np.linspace(0,1)
ax[-1].plot(x_lin, x_lin, color = 'red', alpha = 0.7, linestyle = '--')
plt.suptitle('Temporal effect on cluster membership' + '\n' + \
    f'Wilcoxon paired : p_val = {np.round(temporal_wilcoxon["p-val"][0],3)}')
#ax[-1].hist(mean_cluster_inds, alpha = 0.7)
#ax[-1].axvline(0.5, color = 'red', linestyle = '--', linewidth = 2)
fig.savefig(os.path.join(plot_dir, 'temporal_cluster_membership_test.png'))
plt.close(fig)
#plt.show()


############################################################
quin_gape_array = [x[1] for x in off_gape_array]
cluster_sort_inds = [np.argsort(x) for x in cluster_inds]
cluster_div = [np.sum(x==0) for x in cluster_inds]
sorted_quin_array = [x[y] for x,y in zip(quin_gape_array, cluster_sort_inds)]

quin_spikes_list = [x[taste_dict['quin']] for x in off_spikes_list]
# Sort quinine to have same order of trials as gape array
quin_spikes_list = [x[y] for x,y in zip(quin_spikes_list, cluster_sort_inds)]

quin_spikes_clusters = [
        [x[:num], x[num:]] for x,num in zip(quin_spikes_list, cluster_div)
        ]

#############################################################
## Changes in population firing rate across clusters and epochs
#stim_t = 2000
#epoch_lims = np.stack([
#        [0,250],
#        [250,750],
#        [750,1250],
#        [1250,1750]
#        ]) + stim_t
#
#
## Compare similarity between normalize population vectors for each cluster
#
############################################################
quin_clust_flat = [x for y in quin_spikes_clusters for x in y]

id_frame = pd.DataFrame(
        dict (
            cluster = np.array(['low','high'] * len(file_list)).flatten(),
            name = np.repeat(basename_list, (2))
            )
        )
id_frame['basename'] = [x.split('_repacked')[0] for x in id_frame['name']]
id_frame['fin_name'] = id_frame['basename'] + '_' + id_frame['cluster']

bins = np.linspace(0, 40, 101)
#for ind in trange(len(quin_clust_flat)):
#    data = quin_clust_flat[ind] 
#    model_parameters_keys = ['states','fit','samples', 'model_kwargs']
#    model_parameters_values = [4, 80000, 20000, {'None':None}]
#    preprocess_parameters_keys = ['time_lims','bin_width','data_transform']
#    preprocess_parameters_values = [[2000,4000], 50, None]
#
#    model_params = dict(zip(model_parameters_keys, model_parameters_values))
#    preprocess_params = dict(zip(preprocess_parameters_keys, 
#        preprocess_parameters_values))
#
#    preprocessor = changepoint_preprocess.preprocess_single_taste
#    model_template= changepoint_model.single_taste_poisson
#    inference_func = changepoint_model.advi_fit
#
#    preprocessed_data = preprocessor(data, **preprocess_params)
#
#    model = model_template(preprocessed_data,
#                                 model_params['states'],
#                                 **model_params['model_kwargs'])
#
#    temp_outs = inference_func(model,
#                        model_params['fit'],
#                        model_params['samples'])
#    varnames = ['model', 'approx', 'lambda', 'tau', 'data']
#    inference_outs = dict(zip(varnames, temp_outs))
#
#    tau_array = inference_outs['tau']
#    tau_hist_array = convert_to_hist_array(tau_array, bins)
#    np.save(os.path.join(save_path, id_frame['fin_name'][ind]), tau_hist_array)

# Load inferred tau, calculate mean latency, and variance
tau_list = []
for name in id_frame['fin_name']:
    this_path = os.path.join(save_path, name + '.npy')
    this_array = np.load(this_path)
    tau_list.append(this_array)

scaled_bins = (bins / bins.max()) * 2000
fin_bins = scaled_bins[:-1]
mode_tau = [scaled_bins[np.argmax(x, axis=0)] for x in tau_list]

#flat_tau = np.reshape(np.concatenate(tau_list, axis=1),(100,-1))
#plt.plot(flat_tau[:,:10]);plt.show()

def hist_mean(bins, dist):
    pdf = dist/dist.sum()
    return np.sum(bins*pdf)

def hist_std(bins, dist):
    pdf = dist/dist.sum()
    mean = hist_mean(bins, dist)
    std = np.sqrt(hist_mean((bins - mean)**2, pdf))
    return std

std_tau = [
        np.stack(
        [
            [hist_std(fin_bins, x) for x in y]
            for y in this_session.T
            ]
        ).T
        for this_session in tau_list
        ] 

frame_list = []
for num, (this_mode, this_std) in enumerate(zip(mode_tau, std_tau)):
    inds = np.array(list(np.ndindex(this_mode.shape)))
    this_frame = pd.DataFrame(
            dict(
                dat_num = num,
                trial = inds[:,0],
                transition = inds[:,1],
                mode = this_mode.flatten(),
                std = this_std.flatten()
                )
            )
    frame_list.append(this_frame)
fin_frame = pd.concat(frame_list)
fin_frame['cluster'] = id_frame['cluster'][fin_frame['dat_num']].values
fin_frame['basename'] = id_frame['basename'][fin_frame['dat_num']].values

sns.boxplot(
        data = fin_frame,
        x = 'transition',
        y = 'mode')
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'mean_transition_latency.png'))
plt.close(fig)
#plt.show()

sns.catplot(
        data = fin_frame,
        col = 'basename',
        hue = 'cluster',
        x = 'transition',
        y = 'mode',
        kind = 'bar',
        col_wrap = 4,
        )
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'transition_latency_comp.png'))
plt.close(fig)
#plt.show()

fin_frame['basename'] = fin_frame['basename'].astype('category')
fin_frame['dat_num'] = fin_frame.basename.cat.codes
fin_frame['cluster'] = pd.Categorical(fin_frame['cluster'], categories = ['low','high'], ordered = True)

mean_fin_frame = fin_frame.groupby(['dat_num','transition','cluster']).mean().reset_index(drop=False)

fig,ax = plt.subplots(2, len(mean_fin_frame.transition.unique()), sharey = 'row', sharex='row')
for num, this_ax in enumerate(ax.T):
    this_dat = mean_fin_frame.loc[mean_fin_frame.transition == num]
    this_ax[0].hist(this_dat.groupby('dat_num')['mode'].diff().dropna())
    this_ax[0].axvline(0, color = 'red', linestyle = '--')
    this_ax[1].scatter(this_dat.cluster, this_dat['mode'])
    this_ax[1].set_xlabel("EMG cluster")
    this_ax[0].set_xlabel("Diff bw clusters")
    for x,y in this_dat.groupby('dat_num'):
        this_ax[1].plot(y['cluster'],y['mode'], color = 'grey')
ax[0,0].set_ylabel('Frequency')
ax[1,0].set_ylabel('Mean Transition')
plt.suptitle('Mean transition location per session')
ax[0,1].set_title('Difference in mean location per cluster')
ax[1,1].set_title('Paired mean locations per cluster')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'agg_transition_latency_comp.png'))
plt.close(fig)
#plt.show()

fig,ax = plt.subplots(2, len(mean_fin_frame.transition.unique()), sharey = 'row', sharex='row')
for num, this_ax in enumerate(ax.T):
    this_dat = mean_fin_frame.loc[mean_fin_frame.transition == num]
    this_ax[0].hist(this_dat.groupby('dat_num')['std'].diff().dropna())
    this_ax[0].axvline(0, color = 'red', linestyle = '--')
    this_ax[1].scatter(this_dat.cluster, this_dat['std'])
    this_ax[1].set_xlabel("EMG cluster")
    this_ax[0].set_xlabel("Diff bw clusters")
    for x,y in this_dat.groupby('dat_num'):
        this_ax[1].plot(y['cluster'],y['std'], color = 'grey')
ax[0,0].set_ylabel('Frequency')
ax[1,0].set_ylabel('Mean Transition Variance')
plt.suptitle('Mean transition variance location per session')
ax[0,1].set_title('Difference in mean variance per cluster')
ax[1,1].set_title('Paired mean variances per cluster')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'agg_transition_std_comp.png'))
plt.close(fig)
#plt.show()

sns.catplot(
        data = fin_frame,
        col = 'basename',
        hue = 'cluster',
        x = 'transition',
        y = 'std',
        kind = 'bar',
        col_wrap = 4,
        )
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'transition_std_comp.png'))
plt.close(fig)
#plt.show()

latency_anova = pg.rm_anova(
            data = fin_frame,
            dv = 'mode',
            within = ['transition', 'cluster'],
            subject = 'basename',
            )
latency_anova.to_csv(
        os.path.join(plot_dir, 'transition_latency_anova.csv')
        )

std_anova = pg.rm_anova(
            data = fin_frame,
            dv = 'std',
            within = ['transition', 'cluster'],
            subject = 'basename',
            )
std_anova.to_csv(
        os.path.join(plot_dir, 'transition_std_anova.csv')
        )

std_pairwise = pg.pairwise_tests(
        data = fin_frame,
        dv = 'std',
        within = ['transition', 'cluster'],
        subject = 'basename'
        )
std_pairwise.to_csv(
        os.path.join(plot_dir, 'transition_std_pairwise.csv')
        )

#x = tau_list[1][:,0,2]
#mean_x = hist_mean(fin_bins, x)
#std_x = hist_std(fin_bins, x) 
#
#plt.plot(fin_bins, x)
#plt.axvline(mean_x - std, color = 'red')
#plt.axvline(mean_x + std, color = 'red')
#plt.show()
