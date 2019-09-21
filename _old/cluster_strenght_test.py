#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:59:15 2018

@author: abuzarmahmood

Generate data to test efficacy of cluster strength
"""
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
from sklearn import svm
from sklearn.linear_model import LogisticRegression as log_reg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from skimage import exposure

def clust_strength(mat,groups):
    """
    Given a matrix and groups within the matrix, calculates a measure of
    clustering strength
    PARAMS
    mat : trials x features
    groups : vector of groups
    """
    cluster_strengths = []
    for group in range(len(np.unique(groups))):
        this_cluster = mat[groups==group,:]
        this_cluster_mean = np.mean(this_cluster,axis=0)
        all_dists = mat - this_cluster_mean
        out_dists = np.linalg.norm(all_dists[groups!=group],axis=1)
        in_dists = np.linalg.norm(all_dists[groups==group],axis=1)
        this_strength = np.mean(out_dists)/np.mean(in_dists)
        cluster_strengths.append(this_strength)
        
    return np.mean(cluster_strengths)

# =============================================================================
# =============================================================================
all_strenghts = []
all_deviations = []
repeats = 50
points = 15
dims = 120

for repeat in range(repeats):
    fake_dat = np.random.normal(size = (points,dims))
    #plt.scatter(dat[:,0],dat[:,1])
    
    n_components = 2
    dist = exposure.equalize_hist(dist_mat(fake_dat,fake_dat))
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          n_init = n_components*100).fit(dist)
    print(gmm.predict(dist))
    
    this_groups = gmm.predict(dist)
    trial_order = np.argsort(this_groups)
    
    # Pull out and cluster distance matrices
    clust_post_dist = dist[trial_order,:]
    clust_post_dist = clust_post_dist[:,trial_order]
    all_strenghts.append(clust_strength(fake_dat,this_groups))
    
plt.hist(all_strenghts,20)
# =============================================================================
# ## Distance matrix cluster plots
# plt.figure()
# plt.subplot(221);plt.imshow(exposure.equalize_hist(dist));plt.title('Un Stim')
# plt.subplot(222);plt.imshow(exposure.equalize_hist(clust_post_dist));plt.title('Clust Stim')
# line_num = np.where(np.diff(np.sort(this_groups)))[0]
# for point in line_num:
#     plt.axhline(point+0.5,color = 'red')
#     plt.axvline(point+0.5,color = 'red')
# 
# red_dat = pca(n_components = 2).fit_transform(fake_dat)
# 
# ## LDA
# clf = lda()
# clf.fit(red_dat, this_groups)
# fit_coefs = clf.coef_[0]
# best_sep = np.argsort(np.abs(fit_coefs))
# plt.figure()
# plt.scatter(red_dat[:,best_sep[0]],red_dat[:,best_sep[1]],c=this_groups)
# plt.colorbar()
# 
# ## Cluster pre- and post-stimulus firing
# post_clust_list = []
# for cluster in range(n_components):
#     this_cluster_post = fake_dat[this_groups == cluster,:]
#     post_clust_list.append(this_cluster_post)
# 
# post_max_vals = []
# post_clust_means = []   
# for cluster in range(len(post_clust_list)):
#     dat = np.mean(post_clust_list[cluster],axis=0)
#     post_clust_means.append(dat)
#     post_max_vals.append(np.max(dat))
# 
# ## Firing rate Plots
# plt.figure()
# count = 1
# for cluster in range(n_components):
#     plt.errorbar(x = np.arange(len(post_clust_means[cluster])),y = post_clust_means[cluster],
#                  yerr = np.std(post_clust_list[cluster],axis=0),
#                  label = cluster)
#     #plt.ylim((0,np.max(post_max_vals)))
#     plt.title('n = %i' %post_clust_list[cluster].shape[0])
# plt.legend()
# plt.suptitle(clust_strength(fake_dat,this_groups))
# =============================================================================
