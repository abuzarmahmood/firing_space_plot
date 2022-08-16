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
