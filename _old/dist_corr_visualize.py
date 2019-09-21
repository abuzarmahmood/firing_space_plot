#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:33:31 2018

@author: abuzarmahmood

Visually confirm that populations with high or low correlation APPEAR that way
when you look at raw firing
"""

######################### Import dat ish #########################
import tables
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from scipy.spatial import distance_matrix as dist_mat
from scipy.stats.mstats import zscore
from scipy.stats import pearsonr
from scipy import signal

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import seaborn as sns
import glob

from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp

from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu
import scipy

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from skimage import exposure



#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#
dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    
corr_dat = pd.DataFrame()

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    #for start, end in zip(range(2000,5500,500), range(2500,5500,500)):
    data.correlation_params = dict(zip(['stimulus_start_time', 'stimulus_end_time',
                                        'baseline_start_time', 'baseline_end_time',
                                        'shuffle_repeats', 'accumulated'],
                                       [2000, 4000, 0, 2000, 100, True]))
    data.get_correlations()
    data.get_dataframe()
        
    corr_dat = pd.concat([corr_dat, data.data_frame])
    print('file %i' % file)
    
#############################
sns.swarmplot(x='file',y='rho',hue='shuffle',data= corr_dat.query('laser == False and shuffle == False'))

############################
file = 0

data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                               [25,250,7000]))
data.get_data()
data.get_firing_rates()

#for start, end in zip(range(2000,5500,500), range(2500,5500,500)):
data.correlation_params = dict(zip(['stimulus_start_time', 'stimulus_end_time',
                                    'baseline_start_time', 'baseline_end_time',
                                    'shuffle_repeats', 'accumulated'],
                                   [2000, 4000, 0, 2000, 100, True]))
data.get_correlations()

off_pre_dists = data.off_corr['pre_dists']
off_stim_dists = data.off_corr['stim_dists']
on_pre_dists = data.on_corr['pre_dists']
on_stim_dists = data.on_corr['stim_dists']

##
all_off_firing = data.all_normal_off_firing
all_off_firing_long = all_off_firing[0,:,:]
for nrn in range(1,all_off_firing.shape[0]):
    all_off_firing_long = np.concatenate((all_off_firing_long,all_off_firing[int(nrn),:,:]),axis=1)

all_off_red_pca = pca(n_components = 20).fit(all_off_firing_long)
all_off_red = all_off_red_pca.transform(all_off_firing_long)

plt.imshow(exposure.equalize_hist(all_off_red))
groups = np.sort(np.asarray([0,1,2,3]*15))
plt.figure();plt.scatter(all_off_red[:,0],all_off_red[:,1],c=groups)
plt.colorbar()

taste_lda = lda().fit(all_off_red,groups)
print(np.mean(taste_lda.predict(all_off_red) == groups))

trial_dist = dist_mat(all_off_firing_long,all_off_firing_long)
plt.figure();plt.imshow(exposure.equalize_hist(trial_dist))

##

n_components = 3
taste = 1
pre_inds = np.arange(0,80)
post_inds = np.arange(80,160)

this_off = data.normal_off_firing[taste]
this_off_pre = this_off[:,:,pre_inds]
this_off_post = this_off[:,:,post_inds]

total_off_post = this_off_post[0,:,:]
for nrn in range(1,this_off_post.shape[0]):
    total_off_post = np.concatenate((total_off_post,this_off_post[int(nrn),:,:]),axis=1)

reduced_off_post_pca = pca(n_components = 15).fit(total_off_post)
print(sum(reduced_off_post_pca.explained_variance_ratio_))
reduced_off_post = reduced_off_post_pca.transform(total_off_post)
gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                      n_init = 500).fit(reduced_off_post)
print(gmm.predict(reduced_off_post))

this_groups = gmm.predict(reduced_off_post)
trial_order = np.argsort(this_groups)

# Pull out and cluster distance matrices
this_dist_post = off_stim_dists[taste]   
clust_post_dist = this_dist_post[trial_order,:]
clust_post_dist = clust_post_dist[:,trial_order]

this_pre_dist = off_pre_dists[taste]
clust_pre_dist = this_pre_dist[trial_order,:]
clust_pre_dist = clust_pre_dist[:,trial_order]

    
## Distance matrix cluster plots
plt.figure()
plt.subplot(221);plt.imshow(exposure.equalize_hist(this_dist_post));plt.title('Un Stim')
plt.subplot(222);plt.imshow(exposure.equalize_hist(clust_post_dist));plt.title('Clust Stim')
line_num = np.where(np.diff(np.sort(this_groups)))[0]
for point in line_num:
    plt.axhline(point+0.5,color = 'red')
    plt.axvline(point+0.5,color = 'red')
    
plt.subplot(223);plt.imshow(exposure.equalize_hist(this_pre_dist));plt.title('Un Pre')
plt.subplot(224);plt.imshow(exposure.equalize_hist(clust_pre_dist));plt.title('Clust Pre')
line_num = np.where(np.diff(np.sort(this_groups)))[0]
for point in line_num:
    plt.axhline(point+0.5,color = 'red')
    plt.axvline(point+0.5,color = 'red')

## Cluster pre- and post-stimulus firing
post_clust_list = []
for cluster in range(n_components):
    this_cluster_post = this_off_post[:,this_groups == cluster,:]
    post_clust_list.append(this_cluster_post)

post_max_vals = []
post_clust_means = []   
for cluster in range(len(post_clust_list)):
    dat = np.mean(post_clust_list[cluster],axis=1)
    post_clust_means.append(dat)
    post_max_vals.append(np.max(dat))
    
pre_clust_list = []
for cluster in range(n_components):
    this_cluster_pre = this_off_pre[:,this_groups == cluster,:]
    pre_clust_list.append(this_cluster_pre)

pre_max_vals = []
pre_clust_means = []   
for cluster in range(len(pre_clust_list)):
    dat = np.mean(pre_clust_list[cluster],axis=1)
    pre_clust_means.append(dat)
    pre_max_vals.append(np.max(dat))

## Firing rate Plots
plt.figure()
count = 1
for cluster in range(n_components):
    plt.subplot(n_components,2,count)
    pre_dat = np.mean(pre_clust_list[cluster],axis=1)
    plt.imshow(pre_dat,
               interpolation='nearest',aspect='auto',vmin=0,vmax=max(pre_max_vals))
    count += 1
    plt.subplot(n_components,2,count)
    post_dat = np.mean(post_clust_list[cluster],axis=1)
    plt.imshow(post_dat,
               interpolation='nearest',aspect='auto',vmin=0,vmax=max(post_max_vals))
    plt.colorbar()
    plt.title('n = %i' %pre_clust_list[cluster].shape[1])
    count += 1

plt.figure()
count = 1
for nrn in range(this_off.shape[0]):
    plt.subplot(this_off.shape[0],2,count)
    for cluster in range(len(pre_clust_list)):
        dat = pre_clust_list[cluster][nrn,:,:]
        y = np.mean(dat,axis=0)
        x = range(dat.shape[1])
        yerr = np.std(dat,axis=0)/np.sqrt(dat.shape[0])
        plt.errorbar(x = x,y = y,yerr = yerr)
    count += 1
    
    plt.subplot(this_off.shape[0],2,count)
    for cluster in range(len(post_clust_list)):
        dat = post_clust_list[cluster][nrn,:,:]
        y = np.mean(dat,axis=0)
        x = range(dat.shape[1])
        yerr = np.std(dat,axis=0)/np.sqrt(dat.shape[0])
        plt.errorbar(x = x,y = y,yerr = yerr)
    count += 1