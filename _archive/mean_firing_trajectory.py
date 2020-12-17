#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 10:36:25 2019

@author: abuzarmahmood
"""

######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt

import tensortools as tt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat
from sklearn.manifold import TSNE as tsne
from sklearn.cluster import KMeans as kmeans

from scipy.stats import mannwhitneyu as mnu
import scipy

from skimage import exposure

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# HMM imports
import os
import tables
os.chdir('/media/bigdata/PyHMM/PyHMM/')
import numpy as np
from hinton import hinton
from hmm_fit_funcs import *
from fake_firing import *

# =============================================================================
# =============================================================================
dir_list = ['/media/bigdata/veronica_data/VF313_prex1_170821_102612','/media/bigdata/veronica_data/VF313_prex3_170823_100845']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

day1_nrns = [3,7,9,10,18,19,21]
day3_nrns = [3,6,8,11,16,17,18]

# =============================================================================
#  First Day
# =============================================================================
file  = 0

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data1 = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data1.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'conv',269]))
data1.get_data()

# Since all taste don't have the same number of trials, find the lowest number of trials
# and cut the rest of them to the same size
min_trials = np.min([x.shape[1] for x in data1.off_spikes])
data1.off_spikes = [x[:,range(min_trials),:] for x in data1.off_spikes]

data1.get_firing_rates()
data1.get_normalized_firing()

# =============================================================================
# Third Day
# =============================================================================
file  = 1

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data3 = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data3.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'conv',269]))
data3.get_data()

# Since all taste don't have the same number of trials, find the lowest number of trials
# and cut the rest of them to the same size
min_trials = np.min([x.shape[1] for x in data3.off_spikes])
data3.off_spikes = [x[:,range(min_trials),:] for x in data3.off_spikes]

data3.get_firing_rates()
data3.get_normalized_firing()

# =============================================================================
# Convert data into arrays
# =============================================================================
firing_inds = np.arange(0,269) #np.arange(60:140)
day1_data = np.asarray(data1.normal_off_firing)[:,day1_nrns,:,0:269].swapaxes(-1,-2)
day3_data = np.asarray(data3.normal_off_firing)[:,day3_nrns,:,0:269].swapaxes(-1,-2)

all_data = np.concatenate((day1_data,day3_data[:,:,:,:32]),axis=0)
# =============================================================================
#  Take means of data and generate coordinate transformation based on that
# =============================================================================
day1_mean = np.mean(day1_data,axis=-1)
day3_mean = np.mean(day3_data,axis=-1)

all_data_mean = np.concatenate((day1_mean,day3_mean),axis=0)

all_data_long = all_data_mean[0,:,:]
for taste in range(1,all_data_mean.shape[0]):
    all_data_long = np.concatenate((all_data_long,all_data_mean[taste,:,:]),axis=-1)

all_mean_red_pca = pca(n_components = 3).fit(all_data_long.T)
all_mean_red = all_mean_red_pca.transform(all_data_long.T)

# Convert mean data back to array
all_mean_red_array = np.zeros((all_data_mean.shape[0],all_mean_red.shape[1],all_data_mean.shape[2]))
all_mean_red_list = np.split(all_mean_red,6)
for taste in range(len(all_mean_red_list)):
    all_mean_red_array[taste,:,:] = all_mean_red_list[taste].T
    
# Smooth mean data
smooth_mean_dat = np.zeros(all_mean_red_array.shape)
for taste in range(smooth_mean_dat.shape[0]):
    for dim in range(smooth_mean_dat.shape[1]):
        smooth_mean_dat[taste,dim,:] = scipy.ndimage.filters.gaussian_filter(all_mean_red_array[taste,dim,:],1)

# Use same transformation to reduce single trials
all_data_red = np.zeros((all_data.shape[0],3,all_data.shape[2],all_data.shape[3]))
for taste in range(all_data.shape[0]):
    for trial in range(all_data.shape[-1]):
        all_data_red[taste,:,:,trial] = all_mean_red_pca.transform(all_data[taste,:,:,trial].T).T

# Smooth individual trials
smooth_all_dat = np.zeros(all_data_red.shape)
for taste in range(smooth_all_dat.shape[0]):
    for dim in range(smooth_all_dat.shape[1]):
        for trial in range(smooth_all_dat.shape[-1]):
            smooth_all_dat[taste,dim,:,trial] = scipy.ndimage.filters.gaussian_filter(all_data_red[taste,dim,:,trial],1)

### Plots plots plots ###
    
c = [(1,0,0),
     (0,1,0),
     (0,0,1),
     (0.5,0,0),
     (0,0.5,0),
     (0,0,0.5)]

# =============================================================================
# Single trials without smoothing
# =============================================================================
fig = plt.figure()
ax = Axes3D(fig)
for taste in range(all_data_red.shape[0]//2):
    for trial in range(all_data_red.shape[-1]):
        l = ax.plot(all_data_red[taste,0,:,trial],
                    all_data_red[taste,1,:,trial],
                    all_data_red[taste,2,:,trial], 
                    color = c[taste],
                    alpha = 0.2)
ax.axes.set_xlabel('blalalalala')
ax.axes.set_ylabel('bwarararara')
ax.axes.set_zlabel('pca3 neuronal firing')

# =============================================================================
#  Single trials with Smoothing
# =============================================================================
fig = plt.figure()
ax = Axes3D(fig)
for taste in range(smooth_all_dat.shape[0]//2):
    for trial in range(smooth_all_dat.shape[-1]):
        l = ax.plot(smooth_all_dat[taste,0,:,trial],
                    smooth_all_dat[taste,1,:,trial],
                    smooth_all_dat[taste,2,:,trial], 
                    color = c[taste],
                    alpha = 0.2)
ax.axes.set_xlabel('blalalalala')
ax.axes.set_ylabel('bwarararara')
ax.axes.set_zlabel('pca3 neuronal firing')

# =============================================================================
#  Mean taste Without smoothing
# =============================================================================

fig = plt.figure()
ax = Axes3D(fig)
for taste in range(all_mean_red_array.shape[0]//2):
    l = ax.plot(all_mean_red_array[taste,0,:],all_mean_red_array[taste,1,:],all_mean_red_array[taste,2,:], color = c[taste])
ax.axes.set_xlabel('blalalalala')
ax.axes.set_ylabel('bwarararara')
ax.axes.set_zlabel('pca3 neuronal firing')

fig = plt.figure()
ax = Axes3D(fig)
for taste in range(all_mean_red_array.shape[0]//2,all_mean_red_array.shape[0]):
    l = ax.plot(all_mean_red_array[taste,0,:],all_mean_red_array[taste,1,:],all_mean_red_array[taste,2,:], color = c[taste])
ax.axes.set_xlabel('blalalalala')
ax.axes.set_ylabel('bwarararara')
ax.axes.set_zlabel('pca3 neuronal firing')

# =============================================================================
# Mean taste With smoothing
# =============================================================================

fig = plt.figure()
ax = Axes3D(fig)
for taste in range(smooth_mean_dat.shape[0]//2):
    l = ax.plot(smooth_mean_dat[taste,0,:],smooth_mean_dat[taste,1,:],smooth_mean_dat[taste,2,:], color = c[taste])
ax.axes.set_xlabel('blalalalala')
ax.axes.set_ylabel('bwarararara')
ax.axes.set_zlabel('pca3 neuronal firing')

fig = plt.figure()
ax = Axes3D(fig)
for taste in range(smooth_mean_dat.shape[0]//2,smooth_mean_dat.shape[0]):
    l = ax.plot(smooth_mean_dat[taste,0,:],smooth_mean_dat[taste,1,:],smooth_mean_dat[taste,2,:], color = c[taste])
ax.axes.set_xlabel('blalalalala')
ax.axes.set_ylabel('bwarararara')
ax.axes.set_zlabel('pca3 neuronal firing')