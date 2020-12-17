#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 18:00:47 2018

@author: abuzarmahmood

Investigate changes in taste representation when BLA is optogenetically inhibited (BLAx)
1) Visualize changes in taste representation under BLAx
    a) Intead of using PCA, find hyperplane which maximally separates on and off (LDA)
2) Investigate the neuronal makeup of this change any change in representations
    a)  Determine which neurons are dynamic in both on and off conditions and if they
        match up across conditions
        - Are neurons equally dynamic for all tastes?
    b)  Is any change in representation differentially carried by neuros with more
        dynamic responses (i.e. are stable neurons showing the same representation
        and is that of unstable neurons more different?)
        - Try both euclidean distance and cosine distance
3) Look at 
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
import pandas as pd
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat
from sklearn import svm
from sklearn.linear_model import LogisticRegression as log_reg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.cluster import KMeans as kmeans

import scipy
from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu as mwu

from skimage import exposure

from mpl_toolkits.mplot3d import Axes3D

import glob

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
        this_cluster = mat[groups==groups[group],:]
        this_cluster_mean = np.mean(this_cluster,axis=0)
        all_dists = mat - this_cluster_mean
        out_dists = np.linalg.norm(all_dists[groups!=groups[group]],axis=1)
        in_dists = np.linalg.norm(all_dists[groups==groups[group]],axis=1)
        this_strength = np.mean(out_dists)/np.mean(in_dists)
        cluster_strengths.append(this_strength)
        
    return np.mean(cluster_strengths)
        
"""
   ____        _          _____   _____          
  / __ \      | |        |  __ \ / ____|   /\    
 | |  | |_ __ | |_ ___   | |__) | |       /  \   
 | |  | | '_ \| __/ _ \  |  ___/| |      / /\ \  
 | |__| | |_) | || (_) | | |    | |____ / ____ \ 
  \____/| .__/ \__\___/  |_|     \_____/_/    \_\
        | |                                      
        |_|  
"""
# =============================================================================
# =============================================================================

dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

file = 0
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                               [25,250,7000]))
data.get_data()
data.get_firing_rates()


# =============================================================================
# =============================================================================
## Visualize change in representation because of Opto
opto = np.asarray(np.sort([1,2]*60))

all_firing = np.concatenate([data.all_normal_off_firing,data.all_normal_on_firing], axis=1)
all_firing = all_firing[:,:,80:200]
all_firing_long = all_firing[0,:,:]
for nrn in range(1,all_firing.shape[0]):
    all_firing_long = np.concatenate((all_firing_long,all_firing[int(nrn),:,:]),axis=1)
    
all_reduced_pca = pca(n_components = 15).fit(all_firing_long)
all_reduced = all_reduced_pca.transform(all_firing_long)

# =============================================================================
# plt.plot(all_reduced_pca.explained_variance_ratio_/np.max(all_reduced_pca.explained_variance_ratio_),'-o')
# plt.xlabel('PCA number');plt.ylabel('Variance Explained Ratio')
# 
# plt.set_cmap('viridis')
# 
# plt.figure()
# plt.scatter(all_reduced[:,0],all_reduced[:,1],
#                c =opto, s=20)
# plt.colorbar()
# plt.xlabel('PCA1');plt.ylabel('PCA2')
# 
# 
# fig = plt.figure()
# ax = Axes3D(fig)
# p = ax.scatter(all_reduced[:,0],all_reduced[:,1],all_reduced[:,2],
#                c =opto ,s=20)
# fig.colorbar(p)
# =============================================================================

## LDA
clf = lda()
clf.fit(all_reduced, opto)
fit_coefs = clf.coef_[0]
best_sep = np.argsort(np.abs(fit_coefs))[-3:]
plt.figure()
plt.scatter(all_reduced[:,best_sep[2]],all_reduced[:,best_sep[1]],c=opto)
plt.colorbar()

# =============================================================================
# fig = plt.figure()
# ax = Axes3D(fig)
# p = ax.scatter(all_reduced[:,best_sep[0]],all_reduced[:,best_sep[1]],all_reduced[:,best_sep[2]],
#                c =opto,s=20)
# fig.colorbar(p)
# =============================================================================

## Visualize changes in on and off conditions individually
taste = np.asarray(np.sort([1,2,3,4]*15))

# Off firing
off_firing = data.all_normal_off_firing
off_firing = off_firing[:,:,80:200]
all_off_long = off_firing[0,:,:]
for nrn in range(1,off_firing.shape[0]):
    all_off_long = np.concatenate((all_off_long,off_firing[int(nrn),:,:]),axis=1)
    
all_off_pca = pca(n_components = 5).fit(all_off_long)
all_off_red = all_off_pca.transform(all_off_long)
plt.plot(all_off_pca.explained_variance_ratio_/np.max(all_off_pca.explained_variance_ratio_),'-o')
plt.xlabel('PCA number');plt.ylabel('Variance Explained Ratio')

plt.figure();data.imshow(all_off_red)


plt.figure()
plt.scatter(all_off_red[:,0],all_off_red[:,1],
               c =taste, s=20)
plt.colorbar()
plt.xlabel('PCA1');plt.ylabel('PCA2')


fig = plt.figure()
ax = Axes3D(fig)
p = ax.scatter(all_off_red[:,0],all_off_red[:,1],all_off_red[:,2],
               c =taste,cmap='Set1',s=20)
fig.colorbar(p)

# On firing
on_firing = data.all_normal_on_firing
on_firing = on_firing[:,:,80:200]
all_on_long = on_firing[0,:,:]
for nrn in range(1,on_firing.shape[0]):
    all_on_long = np.concatenate((all_on_long,on_firing[int(nrn),:,:]),axis=1)
    
all_on_pca = pca(n_components = 5).fit(all_on_long)
all_on_red = all_on_pca.transform(all_on_long)
plt.plot(all_on_pca.explained_variance_ratio_/np.max(all_on_pca.explained_variance_ratio_),'-o')
plt.xlabel('PCA number');plt.ylabel('Variance Explained Ratio')

plt.figure();data.imshow(all_on_red)


plt.figure()
plt.scatter(all_on_red[:,0],all_on_red[:,1],
               c =taste, s=20)
plt.colorbar()
plt.xlabel('PCA1');plt.ylabel('PCA2')


fig = plt.figure()
ax = Axes3D(fig)
p = ax.scatter(all_on_red[:,0],all_on_red[:,1],all_on_red[:,2],
               c =taste,cmap='Set1',s=20)
fig.colorbar(p)

"""
           _ _   _            _            
     /\   | | | | |          | |           
    /  \  | | | | |_ __ _ ___| |_ ___  ___ 
   / /\ \ | | | | __/ _` / __| __/ _ \/ __|
  / ____ \| | | | || (_| \__ \ ||  __/\__ \
 /_/    \_\_|_|  \__\__,_|___/\__\___||___/
 """
# =============================================================================
# =============================================================================
# Determine stable vs dynamic neurons in off condition

dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

off_clust_dat = pd.DataFrame()
n_components = 3

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    for nrn in range(data.off_spikes[0].shape[0]):
        for taste in range(4):
            
            # Only take neurons which fire in every trial
            this_spikes = data.off_spikes[taste]
            this_spikes = this_spikes[nrn,:,2000:4000]
            if not (np.sum(np.sum(this_spikes,axis=1) == 0) > 0):
                
                this_off = data.normal_off_firing[taste]
                this_off = this_off[nrn,:,80:160]
                
                mean_coeff_var = np.mean(np.std(this_off,axis=0)/np.mean(this_off,axis=0))
                
                #this_off_red = pca(n_components = 5).fit_transform(this_off)
                
                nrn_dist = exposure.equalize_hist(dist_mat(this_off,this_off))
                
            # =============================================================================
            #     gmm = GaussianMixture(n_components=n_components, covariance_type='full',
            #                           n_init = 100)
            #     this_groups = gmm.fit(nrn_dist).predict(nrn_dist)
            # =============================================================================
                
                clf = kmeans(n_clusters = n_components, n_init = 100)
                this_groups = clf.fit_predict(nrn_dist)
                

                group_sizes  = np.asarray([sum(this_groups == x) for x in np.unique(this_groups)])
                min_group_size = len(this_groups)/3
                #max_group_size = len(this_groups)*2/3
                dynamic_criterion = (np.sum(group_sizes >= min_group_size) >= 2) # Atleast 2 groups are greater than 1/3 of total number
                #stable_criterion = (np.sum(group_sizes >= max_group_size) >= 1) # Atleast 1 group is greater than 2/3 of total number     
                # If only one trial in a group, take that out
                if sum(group_sizes == 1):
                    outlier_trial = np.where(this_groups == range(n_components)[np.where(group_sizes == 1)[0][0]])[0][0]
                    this_off = this_off[np.arange(this_off.shape[0])!=outlier_trial,:]
                    this_groups = this_groups[np.arange(len(this_groups))!=outlier_trial]
                this_cluster_strength = clust_strength(this_off,this_groups)
                
                trial_order = np.argsort(this_groups)
        
                this_clust_dat = pd.DataFrame(dict(
                            file = file, 
                            taste = taste, 
                            neuron = nrn,
                            clust_strength = this_cluster_strength,
                            dynamic_crit = dynamic_criterion,
                            #stable_crit = stable_criterion,
                            index = [0]))
                
                off_clust_dat = pd.concat([off_clust_dat,this_clust_dat])
            
            print([file,taste,nrn])

# Make sure all 4 tastes are present
# Some neurons may have fired in all trials in only a few tastes
all_taste_counts = off_clust_dat.groupby(by = ['file','neuron'])['taste'].count()
acceptable_neurons = all_taste_counts[all_taste_counts == 4]
off_clust_dat.set_index(['file','neuron'],inplace=True)
acceptable_dat = off_clust_dat.loc[acceptable_neurons.to_frame().index]

all_nrn_cov = acceptable_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(scipy.stats.variation)
all_nrn_mean = acceptable_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(np.mean)
all_nrn_stats = pd.DataFrame(dict(
        coeff_var = all_nrn_cov,
        mean = all_nrn_mean))
#sns.lmplot(x='mean',y='coeff_var',data=all_nrn_stats).set(xlim=(0,5),ylim=(0,1))



# Pick out stable and dynamic neurons with atleast 3 tastes
#stable_dat = acceptable_dat.query('stable_crit == True')          
dynamic_dat = acceptable_dat.query('dynamic_crit == True')

#plt.hist([dynamic_dat.clust_strength,stable_dat.clust_strength],50,range=[0,5])
#plt.legend(['Even clusters','Uneven clusters'])

# Find neurons with altleast 3 tastes after cutting for stable and dynamic criteria
dynamic_taste_counts = dynamic_dat.groupby(by = ['file','neuron'])['taste'].count()
acceptable_dynamic_neurons = dynamic_taste_counts[dynamic_taste_counts >= 3]

# =============================================================================
# stable_taste_counts = stable_dat.groupby(by = ['file','neuron'])['taste'].count()
# acceptable_stable_neurons = stable_taste_counts[stable_taste_counts >= 3]
# =============================================================================

#dynamic_dat = dynamic_dat.set_index(['file','neuron'])
acceptable_dynamic_dat = dynamic_dat.loc[acceptable_dynamic_neurons.to_frame().index]
dynamic_nrn_cov = acceptable_dynamic_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(scipy.stats.variation)
dynamic_nrn_mean = acceptable_dynamic_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(np.mean)
dynamic_nrn_stats = pd.DataFrame(dict(
        coeff_var = dynamic_nrn_cov,
        mean = dynamic_nrn_mean))

# =============================================================================
# stable_dat = stable_dat.set_index(['file','neuron'])
# acceptable_stable_dat = stable_dat.loc[acceptable_stable_neurons.to_frame().index]
# stable_nrn_cov = acceptable_stable_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(scipy.stats.variation)
# stable_nrn_mean = acceptable_stable_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(np.mean)
# stable_nrn_stats = pd.DataFrame(dict(
#         coeff_var = stable_nrn_cov,
#         mean = stable_nrn_mean))
# =============================================================================

sns.lmplot(x='mean',y='coeff_var',data=dynamic_nrn_stats)
pearsonr(dynamic_nrn_cov,dynamic_nrn_mean)

# Restrict to neurons with low coefficient of variation
med_coeff_var = np.median(dynamic_nrn_cov)
low_cov_nrns = dynamic_nrn_stats[dynamic_nrn_stats.coeff_var < med_coeff_var]
#low_cov_nrns = dynamic_nrn_stats.copy()

med_mean = np.median(low_cov_nrns['mean'])
low_cov_nrns['stable'] = np.asarray(low_cov_nrns['mean'] < med_mean)
#stable_nrns = low_cov_nrns[low_cov_nrns['mean'] < med_mean]
#unstable_nrns = low_cov_nrns[low_cov_nrns['mean'] > med_mean]

most_stable = low_cov_nrns[low_cov_nrns['mean'] == np.min(low_cov_nrns['mean'])]
most_dynamic = low_cov_nrns[low_cov_nrns['mean'] == np.max(low_cov_nrns['mean'])]

# =============================================================================
# =============================================================================
# Check if the stable vs dynamic neurons are differentially modulated between
# on and off conditions
dist_frame = pd.DataFrame()

last_file = low_cov_nrns.index[0][0]
for i in range(len(low_cov_nrns.index)):
    this_index = low_cov_nrns.index[i]
    file = this_index[0]
    nrn = this_index[1]
    
    if (i == 0) or (file != last_file):
        data_dir = os.path.dirname(file_list[file])
        data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
        data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                       [25,250,7000]))
        data.get_data()
        data.get_firing_rates()
    
    last_file = file
    
    this_off = np.asarray(data.normal_off_firing)
    this_off = this_off[:,nrn,:,80:160]
    this_off_mean = np.mean(this_off,axis=1)
    
    this_on = np.asarray(data.normal_on_firing)
    this_on = this_on[:,nrn,:,80:160]
    this_on_mean = np.mean(this_on,axis=1)
    
    plt.plot(this_off_mean.T)
    plt.figure();plt.plot(this_on_mean.T)
    
    # Calculate euclidean distance between on and off means
    taste_dists = dist_mat(this_off_mean,this_on_mean)
    this_mean_dist = np.mean(taste_dists[np.diag_indices(taste_dists.shape[0])])
    
    all_cos_dists = []
    for taste in range(4):
        all_cos_dists.append(1 - np.dot(this_off_mean[taste,:],this_on_mean[taste,:])/\
        (np.linalg.norm(this_off_mean[taste,:])*np.linalg.norm(this_on_mean[taste,:])))
        
    this_mean_cos_dist = (np.mean(all_cos_dists))
    
    this_dist_frame = pd.DataFrame(dict(
            file = file,
            neuron = nrn,
            euc_dist = this_mean_dist,
            cos_dist = this_mean_cos_dist,
            index = [0]))
    
    dist_frame = pd.concat([dist_frame, this_dist_frame])
    
    print(this_index)

low_cov_nrns['file'] = low_cov_nrns.index.get_level_values('file')
low_cov_nrns['neuron'] = low_cov_nrns.index.get_level_values('neuron')
low_cov_nrns = pd.merge(low_cov_nrns,dist_frame)

plt.figure();sns.swarmplot(x = 'stable',y='euc_dist',data=low_cov_nrns)
plt.figure();sns.swarmplot(x = 'stable',y='cos_dist',data=low_cov_nrns)

mwu(low_cov_nrns.query('stable == True').euc_dist, low_cov_nrns.query('stable == False').euc_dist)
mwu(low_cov_nrns.query('stable == True').cos_dist, low_cov_nrns.query('stable == False').cos_dist)
# =============================================================================
# =============================================================================
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
            
        ## Cluster pre- and post-stimulus firing
        post_clust_list = []
        unique_groups = np.unique(this_groups)
        for cluster in range(len(unique_groups)):
            this_cluster_post = this_off[this_groups == unique_groups[cluster],:]
            post_clust_list.append(this_cluster_post)
        
        post_max_vals = []
        post_clust_means = []   
        for cluster in range(len(post_clust_list)):
            dat = np.mean(post_clust_list[cluster],axis=0)
            post_clust_means.append(dat)
            post_max_vals.append(np.max(dat))
            
        
    # =============================================================================
    #     #################################
    #     ## Shuffle control for clustering
    #     all_shuffle_strenghts = []
    #     shuffle_repeats = 20
    #     for repeat in range(shuffle_repeats):
    #         np.random.seed(repeat)
    #         this_shuffle_off_spikes = np.zeros(this_spikes.shape)
    #         for time in range(this_shuffle_off_spikes.shape[1]):
    #             this_shuffle_off_spikes[:,time] = np.random.permutation(this_spikes[:,time])
    #         this_shuffle_off = firing_rates(this_shuffle_off_spikes[np.newaxis,:,:],25,250)[1][0,:,:]
    #         shuffle_nrn_dist = exposure.equalize_hist(dist_mat(this_shuffle_off,this_shuffle_off))
    # # =============================================================================
    # #         gmm = GaussianMixture(n_components=n_components, covariance_type='full',
    # #                               n_init = 100).fit(nrn_dist)
    # #         this_shuffle_groups = gmm.predict(shuffle_nrn_dist)
    # # =============================================================================
    #         clf = kmeans(n_clusters = n_components, n_init = 100)
    #         this_shuffle_groups = clf.fit_predict(shuffle_nrn_dist)
    #         all_shuffle_strenghts.append(clust_strength(this_shuffle_off,this_shuffle_groups))
    #         
    #     all_shuffle_strenghts = np.asarray(all_shuffle_strenghts)
    #     mean_shuffle_strength = np.mean(all_shuffle_strenghts[np.where(np.isfinite(all_shuffle_strenghts))[0]])
    # =============================================================================
        plt.figure();data.imshow(this_off[trial_order])
        line_num = np.append(-1,np.where(np.diff(np.sort(this_groups)))[0])
        for point in range(len(line_num)):
            plt.axhline(line_num[point]+0.5,color = 'red')
            plt.text(0,line_num[point]+0.5,point,fontsize = 20,color = 'r')
        plt.suptitle('actual = %.3f' % (this_cluster_strength))
        
    # =============================================================================
    #     ## Clustered raster plots
    #     plt.figure();dot_raster(this_spikes[trial_order])
    #     line_num = np.append(-1,np.where(np.diff(np.sort(this_groups)))[0])
    #     for point in range(len(line_num)):
    #         plt.axhline(line_num[point]+0.5,color = 'red')
    #         plt.text(0,line_num[point]+0.5,point,fontsize = 20,color = 'r')
    #     plt.suptitle(clust_strength(this_off,this_groups)/mean_shuffle_strength)
    # =============================================================================
        
        ## Firing rate Plots
        plt.figure()
        count = 1
        for cluster in range(len(unique_groups)):
            plt.errorbar(x = np.arange(len(post_clust_means[cluster])),y = post_clust_means[cluster],
                         yerr = np.std(post_clust_list[cluster],axis=0),
                         label = cluster)
            #plt.ylim((0,np.max(post_max_vals)))
        plt.legend()

"""
  _______        _         _             _______        _       
 |__   __|      | |       | |           |__   __|      | |      
    | | __ _ ___| |_ ___  | |__  _   _     | | __ _ ___| |_ ___ 
    | |/ _` / __| __/ _ \ | '_ \| | | |    | |/ _` / __| __/ _ \
    | | (_| \__ \ ||  __/ | |_) | |_| |    | | (_| \__ \ ||  __/
    |_|\__,_|___/\__\___| |_.__/ \__, |    |_|\__,_|___/\__\___|
                                  __/ |                         
                                 |___/     

Look at whether neurons that changed under opto have differential cluster strenghts
Also look at whether degree of modulation (mutual information?) is linked to cluster strength
"""
# =============================================================================
# =============================================================================
dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

off_clust_dat = pd.DataFrame()
n_components = 3

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    for nrn in range(data.off_spikes[0].shape[0]):
        for taste in range(4):
            
            # Only take neurons which fire in every trial
            this_spikes = data.off_spikes[taste]
            this_spikes = this_spikes[nrn,:,2000:4000]
            if not (np.sum(np.sum(this_spikes,axis=1) == 0) > 0):
                
                this_off = data.normal_off_firing[taste]
                this_off = this_off[nrn,:,80:160]                
                
                nrn_dist = exposure.equalize_hist(dist_mat(this_off,this_off))
                
                clf = kmeans(n_clusters = n_components, n_init = 100)
                this_groups = clf.fit_predict(nrn_dist)
                

                group_sizes  = np.asarray([sum(this_groups == x) for x in np.unique(this_groups)])
                min_group_size = len(this_groups)/3
                dynamic_criterion = (np.sum(group_sizes >= min_group_size) >= 2) # Atleast 2 groups are greater than 1/3 of total number
                
                # If only one trial in a group, take that out
                if sum(group_sizes == 1):
                    outlier_trial = np.where(this_groups == range(n_components)[np.where(group_sizes == 1)[0][0]])[0][0]
                    this_off = this_off[np.arange(this_off.shape[0])!=outlier_trial,:]
                    this_groups = this_groups[np.arange(len(this_groups))!=outlier_trial]
                this_cluster_strength = clust_strength(this_off,this_groups)
                
                trial_order = np.argsort(this_groups)
        
                this_clust_dat = pd.DataFrame(dict(
                            file = file, 
                            taste = taste, 
                            neuron = nrn,
                            clust_strength = this_cluster_strength,
                            dynamic_crit = dynamic_criterion,
                            #stable_crit = stable_criterion,
                            index = [0]))
                
                off_clust_dat = pd.concat([off_clust_dat,this_clust_dat])
            
            print([file,taste,nrn])

# Make sure all 4 tastes are present
# Some neurons may have fired in all trials in only a few tastes
all_taste_counts = off_clust_dat.groupby(by = ['file','neuron'])['taste'].count()
acceptable_neurons = all_taste_counts[all_taste_counts == 4]
off_clust_dat.set_index(['file','neuron'],inplace=True)
acceptable_dat = off_clust_dat.loc[acceptable_neurons.to_frame().index]

all_nrn_cov = acceptable_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(scipy.stats.variation)
all_nrn_mean = acceptable_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(np.mean)
all_nrn_stats = pd.DataFrame(dict(
        coeff_var = all_nrn_cov,
        mean = all_nrn_mean))

# Pick out dynamic neurons with atleast 3 tastes         
dynamic_dat = acceptable_dat.query('dynamic_crit == True')

# Find neurons with altleast 3 tastes after cutting for stable and dynamic criteria
dynamic_taste_counts = dynamic_dat.groupby(by = ['file','neuron'])['taste'].count()

acceptable_dynamic_neurons = dynamic_taste_counts.copy() #dynamic_taste_counts[dynamic_taste_counts >= 3]

acceptable_dynamic_dat = dynamic_dat.loc[acceptable_dynamic_neurons.to_frame().index]
dynamic_nrn_cov = acceptable_dynamic_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(scipy.stats.variation)
dynamic_nrn_mean = acceptable_dynamic_dat.groupby(by = ['file','neuron'])['clust_strength'].aggregate(np.mean)
dynamic_nrn_stats = pd.DataFrame(dict(
        coeff_var = dynamic_nrn_cov,
        mean = dynamic_nrn_mean))

# Find neurons among these with significant differences for any taste
# between laser on and off conditions
delta_frame = pd.DataFrame()

last_file = dynamic_nrn_stats.index[0][0]
for i in range(len(dynamic_nrn_stats.index)):
    this_index = dynamic_nrn_stats.index[i]
    file = this_index[0]
    nrn = this_index[1]
    
    if (i == 0) or (file != last_file):
        data_dir = os.path.dirname(file_list[file])
        data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
        data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                       [25,250,7000]))
        data.get_data()
        data.get_firing_rates()
    
    last_file = file
    
    this_off = np.asarray(data.normal_off_firing)
    this_off = this_off[:,nrn,:,80:160]
    this_off_mean = np.mean(this_off,axis=1)
    
    this_on = np.asarray(data.normal_on_firing)
    this_on = this_on[:,nrn,:,80:160]
    this_on_mean = np.mean(this_on,axis=1)
    
    # Perform Mann-Whitney U-test on every timepoint
    alpha = 0.05/this_off.shape[2]
    p_vals = np.empty((this_off.shape[0],this_off.shape[2]))
    for taste in range(4):
        for time in range(this_off.shape[2]):
            p_vals[taste,time] = mwu(this_off[taste,:,time],this_on[taste,:,time])[1]
    significant = np.sum(p_vals<alpha,axis=1) > 100/25
    min_p_vals = np.min(p_vals,axis=1)
    
    if len(significant) >0:
        this_delta_frame = pd.DataFrame(dict(
                file = file,
                neuron = nrn,
                #taste = significant,
                taste = range(4),
                min_p_vals = np.log(min_p_vals)
                #laser_effect = True
                ))
        
        delta_frame = pd.concat([delta_frame,this_delta_frame])
    
    print([file,nrn])

acceptable_dynamic_dat['file'] = acceptable_dynamic_dat.index.get_level_values('file')
acceptable_dynamic_dat['neuron'] = acceptable_dynamic_dat.index.get_level_values('neuron')
acceptable_dynamic_dat['taste'] = acceptable_dynamic_dat.index.get_level_values('taste')

sns.lmplot('clust_strength','min_p_vals',data = pd.merge(acceptable_dynamic_dat,delta_frame))

delta_frame.set_index(['file','neuron','taste'],inplace=True)
acceptable_dynamic_dat.set_index(['taste'],inplace=True,append=True)
acceptable_dynamic_dat['laser_effect'] = False

unchanged_clusts = acceptable_dynamic_dat.loc[acceptable_dynamic_dat.index.difference(delta_frame.index),'clust_strength'].dropna()
changed_clusts = acceptable_dynamic_dat.loc[delta_frame.index,'clust_strength'].dropna()
plt.hist([changed_clusts,unchanged_clusts])
mwu(changed_clusts,unchanged_clusts)

# =============================================================================
#         if len(significant) >1:
#             fig, ax = plt.subplots(len(significant),1)
#             for i in range(len(significant)):
#                 taste = significant[i]
#                 ax[i].errorbar(x = np.arange(len(this_off_mean[taste,:])), y= this_off_mean[taste,:],
#                              yerr = np.std(this_off[taste,:,:],axis=0))
#                 ax[i].errorbar(x = np.arange(len(this_on_mean[taste,:])), y= this_on_mean[taste,:],
#                              yerr = np.std(this_on[taste,:,:],axis=0))
#         elif len(significant) >0:
#             taste = significant[0]
#             plt.figure()
#             plt.errorbar(x = np.arange(len(this_off_mean[taste,:])), y= this_off_mean[taste,:],
#                          yerr = np.std(this_off[taste,:,:],axis=0))
#             plt.errorbar(x = np.arange(len(this_on_mean[taste,:])), y= this_on_mean[taste,:],
#                          yerr = np.std(this_on[taste,:,:],axis=0))
# =============================================================================
    