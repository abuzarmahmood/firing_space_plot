#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:08:29 2019

@author: abuzarmahmood
"""

######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data

from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# HMM imports

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

## Load data
#dir_list = ['/media/bigdata/jian_you_data/des_ic']#['/media/bigdata/brads_data/BS28_4Tastes_180801_112138']#
dir_list = ['/media/bigdata/Abuzar_Data/']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


file  = -1

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'baks',700]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()
data.firing_overview('off')

dat = np.asarray(data.all_normal_off_firing)
#dat += np.random.random(dat.shape)*1e-9

num_tastes = 4
taste_labels = np.sort(np.asarray(list(range(num_tastes))*(dat.shape[1]//num_tastes)))
trial_labels = np.arange(dat.shape[1])

# =============================================================================
# LDA classification per neuron
# =============================================================================
repeats = 10
train_fraction = 0.5
classification_accuracies = np.zeros((dat.shape[0],dat.shape[-1],repeats))

for repeat in tqdm(range(repeats)):
    for nrn in range(dat.shape[0]):
        
        train_trials = np.random.permutation(trial_labels)[:np.int(len(trial_labels)*train_fraction)]
        test_trials = np.asarray([x for x in trial_labels if x not in train_trials])
        
        train_labels = taste_labels[train_trials]
        test_labels = taste_labels[test_trials]
        
        train_dat = dat[nrn,train_trials,:]
        test_dat = dat[nrn,test_trials,:]
        
        for time in range(dat.shape[-1]):
            this_train_dat = train_dat[:,time].reshape(-1, 1)
            this_test_dat = test_dat[:,time].reshape(-1, 1)
            
            clf = LDA()
            clf.fit(this_train_dat, train_labels)
            classification_accuracies[nrn,time,repeat] = clf.score(this_test_dat, test_labels)
            
# =============================================================================
# mean_accuracy = np.mean(classification_accuracies,axis=(-1)).T
# std_accuracy = np.std(classification_accuracies,axis=(-1)).T
# x_range = np.arange(len(mean_accuracy))
# plt.fill_between(x_range,y1 = mean_accuracy-std_accuracy, y2 = mean_accuracy+std_accuracy,alpha = 0.5)
# plt.plot(x_range,mean_accuracy)
# =============================================================================
mean_accuracy = np.mean(classification_accuracies,axis=(-1)).T
std_accuracy = np.std(classification_accuracies,axis=(-1)).T
x_range = np.linspace(-2000,5000,classification_accuracies.shape[1])
square_len = np.int(np.ceil(np.sqrt(dat.shape[0])))

fig, ax = plt.subplots(square_len,square_len,sharey=True)
nd_idx_objs = []
for dim in range(ax.ndim):
    this_shape = np.ones(len(ax.shape))
    this_shape[dim] = ax.shape[dim]
    nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(ax.shape[dim]),this_shape.astype('int')), ax.shape).flatten())

for nrn in range(dat.shape[0]):
    plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
    plt.fill_between(x_range,y1 = mean_accuracy[:,nrn]-std_accuracy[:,nrn], 
                     y2 = mean_accuracy[:,nrn]+std_accuracy[:,nrn],alpha = 0.5)
    plt.plot(x_range,mean_accuracy[:,nrn])
    plt.title(nrn)


# =============================================================================
# bs28_pretty_nrns = np.asarray([2,5,8,9,12,14])
# pretty_accuracies = classification_accuracies[bs28_pretty_nrns,:,:]
# pretty_mean = np.mean(pretty_accuracies,axis=(0,-1)).T
# pretty_std = np.std(pretty_accuracies,axis=(0,-1))/np.sqrt(pretty_accuracies.shape[0]).T
# 
# plt.fill_between(x_range,y1 = pretty_mean-pretty_std, 
#                  y2 = pretty_mean+pretty_std,alpha = 0.5)
# plt.plot(x_range,pretty_mean)
# =============================================================================
