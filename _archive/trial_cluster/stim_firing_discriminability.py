"""
**  Use a Linear Discriminant Classifier using leave-one-out to determine
    how different trials from post stimulus firing clustered using GMMs is
**  2 Component GMM
**  Ummm...actually don't need an LDA classifier, can just use a GMM
"""

#  _____                            _   
# |_   _|                          | |  
#   | |  _ __ ___  _ __   ___  _ __| |_ 
#   | | | '_ ` _ \| '_ \ / _ \| '__| __|
#  _| |_| | | | | | |_) | (_) | |  | |_ 
# |_____|_| |_| |_| .__/ \___/|_|   \__|
#                 | |                   
#                 |_|
#
import tables
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from skimage import exposure

import glob

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

baseline_inds = range(80)
stimulus_inds = range(80,160)

class_acc = []
all_groups = []

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    data.correlation_params = dict(zip(['stimulus_start_time', 'stimulus_end_time',
                                        'baseline_start_time', 'baseline_end_time',
                                        'shuffle_repeats', 'accumulated'],
                                       [2000, 4000, 0, 2000, 100, True]))
    data.get_correlations()
    
    #                       _           _     
    #     /\               | |         (_)    
    #    /  \   _ __   __ _| |_   _ ___ _ ___ 
    #   / /\ \ | '_ \ / _` | | | | / __| / __|
    #  / ____ \| | | | (_| | | |_| \__ \ \__ \
    # /_/    \_\_| |_|\__,_|_|\__, |___/_|___/
    #                          __/ |          
    #                         |___/   
                         
    off_pre_dists = data.off_corr['pre_dists']
    off_stim_dists = data.off_corr['stim_dists']
    on_pre_dists = data.on_corr['pre_dists']
    on_stim_dists = data.on_corr['stim_dists']
    
    off_firing = data.normal_off_firing
    
    print('file %i' % file)

    n_components = 2
    
    for taste in range(4):
        stim_off = data.normal_off_firing[taste]
        stim_off = stim_off[:,:,stimulus_inds]
        total_stim_off = stim_off[0,:,:]
        for nrn in range(1,stim_off.shape[0]):
            total_stim_off = np.concatenate((total_stim_off,stim_off[int(nrn),:,:]),axis=1)
        
        reduced_stim_pca = pca(n_components = 10).fit(total_stim_off)
        #print(sum(reduced_stim_pca.explained_variance_ratio_))
        reduced_stim = reduced_stim_pca.transform(total_stim_off)
        
            
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              n_init = 500).fit(reduced_stim)
        #print(gmm.predict(reduced_stim))
        
        groups = gmm.predict(reduced_stim)
        all_groups.append(sum(groups))
        trial_order = np.argsort(groups)
        
        # Train LDA classifier on firing from both clusters            
        
        repeats = 500
        stim_acc = []
        
        for i in range(repeats):
            test_stim = np.random.choice(np.arange(15),size=1,replace=False)[0]
            train_stim = np.arange(15)
            train_stim = np.delete(train_stim,test_stim)
            
            stim_lda = lda()
            stim_lda.fit(reduced_stim[train_stim,:], groups[train_stim])
            stim_acc.append(sum(stim_lda.predict(reduced_stim[test_stim,:][np.newaxis,:]) == groups[test_stim]))
            #print('explained_var = %.3f, accuracy = %.3f' % (explained_var_stim,accuracy))
            
        class_acc.append(np.mean(stim_acc))
# =============================================================================
# =============================================================================
        
        # Pull out and cluster distance matrices
        this_dist = off_stim_dists[taste]   
        clust_dist = this_dist[trial_order,:]
        clust_dist = clust_dist[:,trial_order]
        
        this_pre_dist = off_pre_dists[taste]
        clust_pre_dist = this_pre_dist[trial_order,:]
        clust_pre_dist = clust_pre_dist[:,trial_order]
            
        ## Distance matrix cluster plots
        plt.figure()
        plt.subplot(221);plt.imshow(exposure.equalize_hist(this_dist));plt.title('Un Stim')
        plt.subplot(222);plt.imshow(exposure.equalize_hist(clust_dist));plt.title('Clust Stim')
        line_num = np.where(np.diff(np.sort(groups)))[0]
        for point in line_num:
            plt.axhline(point+0.5,color = 'red')
            plt.axvline(point+0.5,color = 'red')
            
        plt.subplot(223);plt.imshow(exposure.equalize_hist(this_pre_dist));plt.title('Un Pre')
        plt.subplot(224);plt.imshow(exposure.equalize_hist(clust_pre_dist));plt.title('Clust Pre')
        line_num = np.where(np.diff(np.sort(groups)))[0]
        for point in line_num:
            plt.axhline(point+0.5,color = 'red')
            plt.axvline(point+0.5,color = 'red')
        
# =============================================================================
#         clust_list = []
#         for cluster in range(n_components):
#             this_cluster = stim_off[:,groups == cluster,:]
#             clust_list.append(this_cluster)
#         
#         max_vals = []
#         clust_means = []   
#         for cluster in range(len(clust_list)):
#             dat = np.mean(clust_list[cluster],axis=1)
#             clust_means.append(dat)
#             max_vals.append(np.max(dat))
#             
#         ## Firing rate Plots
#         plt.figure()
#         for cluster in range(len(clust_list)):
#             plt.subplot(n_components,1,cluster+1)
#             dat = np.mean(clust_list[cluster],axis=1)
#             plt.imshow(dat,
#                        interpolation='nearest',aspect='auto',vmin=0,vmax=max(max_vals))
#             plt.title('n = %i' % clust_list[cluster].shape[1])
#             plt.colorbar()
#         
#         plt.figure()
#         for nrn in range(stim_off.shape[0]):
#             plt.subplot(stim_off.shape[0],1,nrn+1)
#             for cluster in range(len(clust_list)):
#                 dat = clust_list[cluster][nrn,:,:]
#                 y = np.mean(dat,axis=0)
#                 x = range(dat.shape[1])
#                 yerr = np.std(dat,axis=0)
#                 plt.errorbar(x = x,y = y,yerr = yerr)
# =============================================================================
        
