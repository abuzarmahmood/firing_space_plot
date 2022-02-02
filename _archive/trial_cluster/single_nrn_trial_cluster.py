#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:29:34 2018

@author: abuzarmahmood
"""

######################### Import dat ish #########################
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
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans as kmeans

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure

import glob

def gauss_filt(data,window_size):
    """
    data : 1D array
    """
    std = int(window_size/2/3)
    window = signal.gaussian(window_size, std=std)
    window = window/window.sum()
    filt_data = np.convolve(data,window,mode='same')
    
    return filt_data

def calc_MI(mat):
    """
    Given a matrix
    1) Normalizes if unnormalized
    2) Calculate mutual information between variables
    """
    norm_mat = mat/np.sum(mat, axis = None)
    px = np.sum(norm_mat,axis=0)/np.sum(np.sum(norm_mat,axis=0),axis=None)
    py = np.sum(norm_mat,axis=1)/np.sum(np.sum(norm_mat,axis=1),axis=None)
    hx = -np.sum(px*np.log2(px))
    all_cond_entropy = np.zeros(norm_mat.shape[0])
    for yval in range(norm_mat.shape[0]):
        this_hx_given_y = norm_mat[yval,:]/py[yval]
        this_cond_entropy = -np.sum(this_hx_given_y*np.log2(this_hx_given_y))
        all_cond_entropy[yval] = this_cond_entropy
    cond_entropy = np.sum(py*all_cond_entropy)
    
    mi = hx - cond_entropy
    
    return mi
# =============================================================================
# =============================================================================
"""
Cluster single neurons to investigate patterns in taste response
"""

#dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
dir_list = ['/media/bigdata/brads_data/BS28_4Tastes_180801_112138'] #['/media/bigdata/Jenn_Data/']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

for file in range(len(dir_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
# =============================================================================
#     data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
#                                    [25,250,7000]))
# =============================================================================
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'baks',269]))
    data.get_data()
    data.get_firing_rates()
    data.get_normalized_firing()


    n_components = 3
    
    taste = 0
    nrn = 11
    
    this_off = data.normal_off_firing[taste]
    this_off = this_off[nrn,:,80:160]
    nrn_dist = dist_mat(this_off,this_off)
    
    this_spikes = data.off_spikes[taste]
    this_spikes = this_spikes[nrn,:,2000:4000]
    
    nrn_red_pca = pca(n_components = 5).fit(this_off)
    nrn_off_red = nrn_red_pca.transform(this_off)
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                      n_init = n_components*200).fit(nrn_off_red)
    print(gmm.predict(nrn_off_red))
    
    this_groups = gmm.predict(nrn_off_red)
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
    plt.figure();raster(this_spikes[trial_order])
    line_num = np.append(-0.5,np.where(np.diff(np.sort(this_groups)))[0])
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

# =============================================================================
# =============================================================================
"""
Cluster trials using distance matrix
"""
#dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
dir_list = ['/media/bigdata/brads_data/BS28_4Tastes_180801_112138'] #['/media/bigdata/Jenn_Data/']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

file = 0
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
# =============================================================================
#     data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
#                                    [25,250,7000]))
# =============================================================================
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                           [25,250,7000,'baks',269]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()

n_components = 4

taste = 1
nrn = 6

this_spikes = data.off_spikes[taste]
this_spikes = this_spikes[nrn,:,2000:5000]

this_off = data.normal_off_firing[taste]
this_off = this_off[nrn,:,80:200]

nrn_dist = exposure.equalize_hist(dist_mat(this_off,this_off))

# =============================================================================
# nrn_red_pca = pca(n_components = 15).fit(this_off)
# nrn_off_red = nrn_red_pca.transform(this_off)
# nrn_dist = exposure.equalize_hist(dist_mat(nrn_off_red,nrn_off_red))
# =============================================================================

# =============================================================================
# nrn_red_pca = pca(n_components = 15).fit(this_off)
# nrn_off_red = nrn_red_pca.transform(this_off)
# nrn_dist = squareform(pdist(nrn_off_red,metric='mahalanobis'))
# =============================================================================

# =============================================================================
# gmm = GaussianMixture(n_components=n_components, covariance_type='full',
#                       n_init = n_components*500).fit(nrn_dist)
# print(gmm.predict(nrn_dist))
# this_groups = gmm.predict(nrn_dist)
# =============================================================================

clf = kmeans(n_clusters = n_components, n_init = 500)
this_groups = clf.fit_predict(nrn_dist)


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
plt.figure();dot_raster(this_spikes[trial_order])
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
    plt.ylim((0,np.max(post_max_vals)*1.3))
    plt.title('Mean Cluster Firing +/- SEM')
plt.legend()

# =============================================================================
# =============================================================================
"""
Cluster trials using correlational distance of firing rate
"""
dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

file = 0
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                               [25,250,7000]))
data.get_data()
data.get_firing_rates()

n_components = 3

taste = 0
nrn = 11

this_spikes = data.off_spikes[taste]
this_spikes = this_spikes[nrn,:,2000:4000]

this_off = data.normal_off_firing[taste]
this_off = this_off[nrn,:,80:160]

all_corrs = np.empty((this_off.shape[0]*this_off.shape[0],this_off.shape[1]))

count = 0
for i in range(this_off.shape[0]):
    for j in range(this_off.shape[0]):
        this_corr = scipy.signal.correlate(this_off[i,:],this_off[j,:],mode='same')
        all_corrs[count,:] = this_corr/np.max(this_corr)
        count+= 1

all_corrs_pca = pca(n_components = 1).fit(all_corrs)

corr_dists = np.empty((this_off.shape[0],this_off.shape[0]))
for i in range(corr_dists.shape[0]):
    for j in range(corr_dists.shape[0]):
        this_corr = scipy.signal.correlate(this_off[i,:],this_off[j,:],mode='same')
        this_corr_red = all_corrs_pca.transform(this_corr[np.newaxis,:])
        corr_dists[i,j] = this_corr_red
        
for i in range(corr_dists.shape[0]):
    for j in range(i, corr_dists.shape[1]):
        corr_dists[j][i] = corr_dists[i][j]

nrn_dist =  exposure.equalize_hist(corr_dists)
#nrn_dist = exposure.equalize_hist(dist_mat(this_off,this_off))

clf = kmeans(n_clusters = n_components, n_init = 500)
this_groups = clf.fit_predict(nrn_dist)


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
plt.figure();dot_raster(this_spikes[trial_order])
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

# =============================================================================    
# =============================================================================
"""
Cluster single trials for every taste and see if the taste-responsiveness of
neurons is changed by breaking trials into clusters
"""

dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    
for file in range(len(dir_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    baseline_inds = np.arange(0,80)
    stimulus_inds = np.arange(80,160)
    
    # Smooth firing rates to have smooth PDF
    all_firing_array = np.asarray(data.normal_off_firing)
    smooth_array = np.zeros(all_firing_array.shape)
    for taste in range(smooth_array.shape[0]):
        for neuron in range(smooth_array.shape[1]):
            for trial in range(smooth_array.shape[2]):
                smooth_array[taste,neuron,trial,:] = gauss_filt(all_firing_array[taste,neuron,trial,:],100)
    
    smooth_array += np.random.random(smooth_array.shape)*1e-6
    
    mean_smooth = np.mean(smooth_array,axis=2)
        
    all_mi = []
    all_joint_p = []
    symbols = 8
    for neuron in range(smooth_array.shape[1]):        
        this_neuron = smooth_array[:,neuron,:,80:]
        quartiles = np.linspace(0,100,symbols+1)
        #quart_vals = np.percentile(this_neuron.flatten()[np.nonzero(this_neuron.flatten())],quartiles)
        quart_vals = np.percentile(this_neuron.flatten(),quartiles)
        
        # Create joint probability table
        joint_p = np.zeros((this_neuron.shape[0],symbols))
        for taste in range(this_neuron.shape[0]):
            for val in range(symbols):
                this_dat = this_neuron[taste,:,:].flatten()
                joint_p[taste,val] = np.sum((this_dat < quart_vals[val+1]) &
                              (this_dat >= quart_vals[val]))
            
        joint_p = joint_p/np.sum(joint_p,axis=None)
        joint_p += 1e-9
        all_joint_p.append(joint_p)
        taste_mi = calc_MI(joint_p)
        all_mi.append(taste_mi)
        
    plt.figure();plt.imshow(all_joint_p[np.nanargmax(all_mi)],vmin=0,vmax=np.max(all_joint_p));plt.colorbar()
    plt.figure();plt.imshow(all_joint_p[np.nanargmin(all_mi)],vmin=0,vmax=np.max(all_joint_p));plt.colorbar()
    plt.figure();plt.plot(mean_smooth[:,np.nanargmax(all_mi),:].T)
    plt.title(np.nanmax(all_mi))
    plt.figure();plt.plot(mean_smooth[:,np.nanargmin(all_mi),:].T)
    plt.title(np.nanmin(all_mi))

    n_components = 3
    nrn = np.nanargmin(all_mi)
    
    fig,ax = plt.subplots(4,1,sharey=True)
    
    all_firing_array = np.asarray(data.normal_off_firing)[:,nrn,:,:]
    
    for taste in range(4):
        
        this_off = data.normal_off_firing[taste]
        #this_off = smooth_array[taste,:,:,:]
        this_off = this_off[nrn,:,80:160]
        nrn_dist = dist_mat(this_off,this_off)
        
        this_spikes = data.off_spikes[taste]
        this_spikes = this_spikes[nrn,:,2000:4000]
        
        nrn_red_pca = pca(n_components = 5).fit(this_off)
        nrn_off_red = nrn_red_pca.transform(this_off)
        
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          n_init = n_components*200).fit(nrn_off_red)
        print(gmm.predict(nrn_off_red))
        
        this_groups = gmm.predict(nrn_off_red)
        trial_order = np.argsort(this_groups)
        
        # Pull out and cluster distance matrices
        clust_post_dist = nrn_dist[trial_order,:]
        clust_post_dist = clust_post_dist[:,trial_order]
        
        ## Distance matrix cluster plots
# =============================================================================
#         plt.figure()
#         plt.subplot(221);plt.imshow(exposure.equalize_hist(nrn_dist));plt.title('Un Stim')
#         plt.subplot(222);plt.imshow(exposure.equalize_hist(clust_post_dist));plt.title('Clust Stim')
#         line_num = np.where(np.diff(np.sort(this_groups)))[0]
#         for point in line_num:
#             plt.axhline(point+0.5,color = 'red')
#             plt.axvline(point+0.5,color = 'red')
# =============================================================================
            
        ## Clustered raster plots
        plt.figure();raster(this_spikes[trial_order]);plt.title('taste %i' % taste)
        line_num = np.append(-0.5,np.where(np.diff(np.sort(this_groups)))[0])
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
        for cluster in range(n_components):
            ax[taste].plot(post_clust_means[cluster], label = cluster)
            ax[taste].set_title('taste %i' % taste)
            ax[taste].legend()
            
        ## Mean taste firing rate
        plt.figure(100)
        plt.plot(np.mean(all_firing_array[taste,:,stimulus_inds],axis=1).T,label = taste)
        plt.legend()
    
    plt.figure();data.imshow(data.all_normal_off_firing[nrn,:,stimulus_inds].T)