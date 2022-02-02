#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:37:54 2018

@author: abuzarmahmood

Discovering Spike Patterns in Neuronal Responses
Jean-Marc Fellous, Paul H. E. Tiesinga, Peter J. Thomas and Terrence J. Sejnowski
Journal of Neuroscience 24 March 2004, 24 (12) 2989-3001; DOI: https://doi.org/10.1523/JNEUROSCI.4649-03.2004 

Using GMM on distance matrix to find patterns in neuronal firing
Using surrgoat dataset from artricle above to benchmark analysis
"""

# =============================================================================
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *

import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure

import glob

import scipy.io as sio

def gauss_filt(data,window_size):
    """
    data : 1D array
    """
    std = int(window_size/2/3)
    window = signal.gaussian(window_size, std=std)
    window = window/window.sum()
    filt_data = np.convolve(data,window,mode='same')
    
    return filt_data
# =============================================================================
# =============================================================================
dir_list = ['/media/bigdata/firing_space_plot/fellous_surrogate_data']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.mat',recursive=True)

str_template = '5-3-5.'
this_file = np.where([x.find(str_template)>0 for x in file_list])[0]
file = this_file[0]
data = sio.loadmat(file_list[file])

base_file_name = os.path.basename(file_list[file])
base_file_name = base_file_name.split('-')

clusters = int(base_file_name[1])
extra_spike_level = int(base_file_name[2])
jitter_level = int(base_file_name[3][0])

# rspiketimes(Ntrials, N, MaxNspikes) UNORDERED
# spiketimes(Ntrials, N, MaxNspikes) ORDERED
time_steps = 1000
rspiketimes = np.floor(data['rspiketimes']*time_steps)
rspikes = np.zeros((rspiketimes.shape[0],rspiketimes.shape[1],time_steps))
for i in range(rspikes.shape[0]):
    for j in range(rspikes.shape[1]):
        this_spiketimes = rspiketimes[i,j,:]
        this_spiketimes = this_spiketimes[this_spiketimes > 0]
        this_spiketimes = np.asarray([int(x) for x in this_spiketimes])
        rspikes[i,j,this_spiketimes - 1] = 1

# =============================================================================
# =============================================================================
nrn = 1
n_components = clusters
this_spikes = rspikes[:,nrn,:]

this_off = np.zeros(this_spikes.shape)
for trial in range(this_off.shape[0]):
    this_off[trial,:] = gauss_filt(this_spikes[trial,:],100)

nrn_dist = exposure.equalize_hist(dist_mat(this_off,this_off))

gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                      n_init = n_components*200).fit(nrn_dist)
print(gmm.predict(nrn_dist))
        

this_groups = gmm.predict(nrn_dist)
trial_order = np.argsort(this_groups)

# Pull out and cluster distance matrices
clust_post_dist = nrn_dist[trial_order,:]
clust_post_dist = clust_post_dist[:,trial_order]

## Distance matrix cluster plots
plt.figure()
plt.subplot(221);plt.imshow(exposure.equalize_hist(nrn_dist));plt.title('Un Stim')
plt.subplot(222);plt.imshow(exposure.equalize_hist(clust_post_dist));plt.title('Clust Stim')
line_num = np.where(np.diff(np.sort(this_groups)))[0]
for point in line_num:
    plt.axhline(point+0.5,color = 'red')
    plt.axvline(point+0.5,color = 'red')
    
## Clustered raster plots
plt.figure();dot_raster(this_spikes,markersize=2.5)
plt.figure();dot_raster(this_spikes[trial_order],markersize=2.5)
line_num = np.append(-1,np.where(np.diff(np.sort(this_groups)))[0])
for point in range(len(line_num)):
    plt.axhline(line_num[point]+0.5,color = 'red')
    plt.text(0,line_num[point]+0.5,point,fontsize = 20,color = 'r')
    
    
## Cluster pre- and post-stimulus firing
post_clust_list = []
for cluster in range(n_components):
    this_cluster_post = this_off[this_groups == cluster,:]
    post_clust_list.append(this_cluster_post)

post_max_vals = []
post_clust_means = []   
for cluster in range(len(post_clust_list)):
    dat = np.mean(post_clust_list[cluster],axis=0)
    post_clust_means.append(dat)
    post_max_vals.append(np.max(dat))
    
## Firing rate Plots
plt.figure()
count = 1
for cluster in range(n_components):
    plt.errorbar(x = np.arange(len(post_clust_means[cluster])),y = post_clust_means[cluster],
                 yerr = np.std(post_clust_list[cluster],axis=0)/np.sqrt(post_clust_list[cluster].shape[0]),
                 label = cluster)
    #plt.ylim((0,np.max(post_max_vals)))
    plt.title('n = %i' %post_clust_list[cluster].shape[0])
plt.legend()