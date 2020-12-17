#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:04:09 2019

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
from sklearn.cluster import KMeans as kmeans

# HMM imports

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

from skimage import exposure

# =============================================================================
# Load Data
# =============================================================================

dir_list = ['/media/bigdata/brads_data/BS45']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


file  = -1

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [50,500,7000,'baks',700]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()
data.firing_overview('off')
plt.title('Off')
data.firing_overview('on')
plt.title('On')

bla_nrn_inds = range(3,12)
off_dat = np.asarray(data.off_firing)[:,bla_nrn_inds,:,:]
on_dat = np.asarray(data.on_firing)[:,bla_nrn_inds,:,:]

mean_off_dat = np.mean(off_dat,axis=2)
mean_on_dat = np.mean(on_dat,axis=2)
time_inds = range(80,200)
for nrn in range(mean_off_dat.shape[1]):
    plt.figure()
    for taste in range(mean_off_dat.shape[0]):
        plt.subplot(4,1,taste+1)
        plt.plot(np.linspace(0,2.5,len(time_inds)),off_dat[taste,nrn,:,time_inds],c = 'r')
        plt.plot(np.linspace(0,2.5,len(time_inds)),on_dat[taste,nrn,:,time_inds],c='b')

# =============================================================================
# Split trials for a taste into 2 groups to remove dead trials, then do average
# =============================================================================
nrn = 4
for taste in range(4):
    this_nrn_off = np.asarray(data.normal_off_firing)[taste,nrn,:,:]
    this_nrn_on = np.asarray(data.normal_on_firing)[taste,nrn,:,:]
    
    off_dist = exposure.equalize_hist(dist_mat(this_nrn_off,this_nrn_off))
    on_dist = exposure.equalize_hist(dist_mat(this_nrn_on,this_nrn_on))
    
    n_components = 2
    clf = kmeans(n_clusters = n_components, n_init = 500)
    off_groups = clf.fit_predict(off_dist)
    on_groups = clf.fit_predict(on_dist)
    
    
    off_order = np.argsort(off_groups)
    on_order = np.argsort(on_groups)
    
    sorted_off = this_nrn_off[off_order]
    sorted_on = this_nrn_on[on_order]
    
    off_clust_list = []
    on_clust_list = []
    for cluster in range(n_components):
        off_clust_list.append(this_nrn_off[off_groups == cluster,:])
        on_clust_list.append(this_nrn_on[on_groups == cluster,:])
        
    plt.figure()
    plt.subplot(121)
    data.imshow(sorted_off)
    plt.subplot(122)
    data.imshow(sorted_on)




# =============================================================================
# Analysis
# =============================================================================

# =============================================================================
# time_inds = range(0,700)
# off_dat = off_dat[:,:,:,time_inds]
# on_dat = on_dat[:,:,:,time_inds]
# =============================================================================

# Use all data on a per-taste + per_neuron basis to find PCA axes
all_firing = data.all_firing_array[:,bla_nrn_inds,:,:]

taste = 0
trial = 0
this_off_dat = off_dat[taste,:,trial,:]
this_on_dat = on_dat[taste,:,trial,:]
this_dist = dist_mat(this_off_dat.T,this_on_dat.T)[np.diag_indices(this_off_dat.shape[-1])]
plt.plot(this_dist)