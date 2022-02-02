#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:46:31 2018

@author: abuzarmahmood

Run VI HMM on sets of clustered trials from a population
Compare between HMMs fit on entire taste data vs subsets of clustered trials
"""
# Use Jen's data since more trials
# Remove neurons which don't fire

######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat
from sklearn.cluster import KMeans as kmeans

from skimage import exposure
import glob
import tensortools as tt

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
#dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
#dir_list = ['/media/bigdata/Jenn_Data/']
dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


for file in range(len(file_list)):

    this_dir = file_list[file].split(sep='/')[-2]
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    all_spikes_array = np.asarray(data.off_spikes)
    all_firing_array = np.asarray(data.normal_off_firing)
    
    # Remove neurons that less than threshold spikes
    threshold = 1000
    sum_firing = np.sum(all_firing_array,axis=(0,2,3))
    all_firing_array = all_firing_array[:,sum_firing>threshold,:,:]
    n_components = 3
    
    for taste in range(4):
        
        plot_dir = '/media/bigdata/firing_space_plot/trial_cluster/cluster_hmm_plots/' + \
                    this_dir + '/' + 'taste_' + str(taste)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        this_off = all_firing_array[taste,:,:,:]
        this_spikes = all_spikes_array[taste,:,:,:]
        
# =============================================================================
#         total_off = this_off[0,:,:]
#         for nrn in range(1,this_off.shape[0]):
#             total_off = np.concatenate((total_off,this_off[int(nrn),:,:]),axis=1)
# =============================================================================
        
        # Tensor decomposition for trial clustering
        rank = 3 #X.shape[0]
        U = tt.cp_als(np.swapaxes(this_off,1,2), rank=rank, verbose=True)
        fig, ax, po = tt.plot_factors(U.factors)
        [nrn_f,time_f,trial_f] = U.factors.factors
        trial_dists = exposure.equalize_hist(dist_mat(trial_f,trial_f))
        
        #trial_dists = exposure.equalize_hist(dist_mat(total_off,total_off))
        
        # =============================================================================
        # reduced_off_pca = pca(n_components = 15).fit(total_off)
        # reduced_off = reduced_off_pca.transform(total_off)
        # trial_dists_red = dist_mat(reduced_off,reduced_off)
        # =============================================================================
        
        clf = kmeans(n_clusters = n_components, n_init = 200)
        this_groups = clf.fit_predict(trial_dists)
        
# =============================================================================
#         gmm = GaussianMixture(n_components=n_components, covariance_type='full',
#                               n_init = 500).fit(trial_dists)
#         this_groups = gmm.predict(trial_dists)
# =============================================================================

        trial_order = np.argsort(this_groups)
        
        # Pull out and cluster distance matrices
        clust_dist = trial_dists[trial_order,:]
        clust_dist = clust_dist[:,trial_order]
        
        ## Distance matrix cluster plots
        plt.figure()
        plt.subplot(121);plt.imshow(exposure.equalize_hist(trial_dists));plt.title('Un Stim')
        plt.subplot(122);plt.imshow(exposure.equalize_hist(clust_dist));plt.title('Clust Stim')
        line_num = np.where(np.diff(np.sort(this_groups)))[0]
        for point in line_num:
            plt.axhline(point+0.5,color = 'red')
            plt.axvline(point+0.5,color = 'red')
        plt.suptitle('Trial Clustering')
        plt.tight_layout()
        plt.savefig(plot_dir + '/' + 'trial_clustering.png')
        plt.close()
        
        clust_list = []
        for cluster in range(n_components):
            this_cluster = this_off[:,this_groups == cluster,:]
            clust_list.append(this_cluster)
        
        max_vals = []
        clust_means = []   
        for cluster in range(len(clust_list)):
            dat = np.mean(clust_list[cluster],axis=1)
            clust_means.append(dat)
            max_vals.append(np.max(dat))
        
        ## Firing rate Plots
        plt.figure()
        for cluster in range(len(clust_list)):
            plt.subplot(n_components,1,cluster+1)
            dat = np.mean(clust_list[cluster],axis=1)
            plt.imshow(dat,
                       interpolation='nearest',aspect='auto',vmin=0,vmax=max(max_vals))
            plt.title('n = %i' % clust_list[cluster].shape[1])
            plt.colorbar()
        plt.suptitle('Mean cluster firing')
        plt.savefig(plot_dir + '/' + 'mean_cluster_firing.png')
        plt.close()
        
        fig = plt.figure(figsize = (8,10))
        for nrn in range(this_off.shape[0]):
            plt.subplot(this_off.shape[0],1,nrn+1)
            for cluster in range(0,len(clust_list)-1):
                dat = clust_list[cluster][nrn,:,:]
                y = np.mean(dat,axis=0)
                x = range(dat.shape[1])
                yerr = np.std(dat,axis=0)/np.sqrt(dat.shape[0])
                plt.errorbar(x = x,y = y,yerr = yerr)
        plt.suptitle('Individual neuron cluster firing')
        plt.savefig(plot_dir + '/' + 'nrn_cluster_firing.png')
        plt.close()
                
        # =============================================================================
        # =============================================================================
        # Convert data to categorical
        
        bin_size = 10
        start_t = 2000
        end_t = 5000
            
        temp_spikes = this_spikes[:,:,start_t:end_t]
        temp_spikes = np.swapaxes(temp_spikes,0,1) 
        # Bin spikes (might decrease info for fast spiking neurons)
        binned_spikes = np.zeros((temp_spikes.shape[0],temp_spikes.shape[1], int((end_t - start_t)/bin_size)))
        for i in range(temp_spikes.shape[0]): # Loop over trials
            for j in range(temp_spikes.shape[1]): # Loop over neurons
                for k in range(binned_spikes.shape[2]): # Loop over time
                    if (np.sum(temp_spikes[i, j, k*bin_size:(k+1)*bin_size]) > 0):
                        binned_spikes[i,j,k] = 1
                        
        # Convert binned spikes to clusters
        cat_clust_list = []
        for cluster in range(n_components):
            this_cluster = binned_spikes[this_groups == cluster,:]
            cat_clust_list.append(this_cluster)
        
        # Equalize the size of the clusters -> Make more trials by sampling spikes
        # randomly from trials within the clusters
        new_cat_clust_list = []
        for cluster in cat_clust_list:
            sample_inds = np.random.randint(0,cluster.shape[0],size=binned_spikes.shape)
            new_clust = np.zeros(binned_spikes.shape)
            for trial in range(new_clust.shape[0]):
                for neuron in range(new_clust.shape[1]):
                    for time in range(new_clust.shape[2]):
                        new_clust[trial,neuron,time] = cluster[sample_inds[trial,neuron,time],neuron,time]
            new_cat_clust_list.append(new_clust)
        
        # Add mixed trials for comparison
        new_cat_clust_list.append(binned_spikes)
        
        # Plot firing rates and confirm separation of clusters
        cluster = 0
        nrn = 1
        plt.plot(np.sum(new_cat_clust_list[cluster][:,nrn,:],0)/new_cat_clust_list[cluster][:,nrn,:].shape[0])
        plt.plot(np.sum(cat_clust_list[cluster][:,nrn,:],0)/cat_clust_list[cluster][:,nrn,:].shape[0]) 
        
        plt.subplot(211)
        raster(new_cat_clust_list[cluster][:,nrn,:])
        plt.subplot(212)
        raster(cat_clust_list[cluster][:,nrn,:])
         
        ######### For categorical HMM ########  
        # Remove multiple spikes in same time bin (for categorical HMM)
        for cluster in range(len(new_cat_clust_list)):
            for i in range(new_cat_clust_list[cluster].shape[0]): # Loop over trials
                for k in range(new_cat_clust_list[cluster].shape[2]): # Loop over time
                    n_firing_units = np.where(new_cat_clust_list[cluster][i,:,k] > 0)[0]
                    if len(n_firing_units)>0:
                        new_cat_clust_list[cluster][i,:,k] = 0
                        new_cat_clust_list[cluster][i,np.random.choice(n_firing_units),k] = 1
        
        # Convert bernoulli trials to categorical data
        cat_binned_spikes_list = []
        for cluster in range(len(new_cat_clust_list)):      
            cat_binned_spikes = np.zeros((new_cat_clust_list[cluster].shape[0],new_cat_clust_list[cluster].shape[2]))
            for i in range(cat_binned_spikes.shape[0]):
                for j in range(cat_binned_spikes.shape[1]):
                    firing_unit = np.where(new_cat_clust_list[cluster][i,:,j] > 0)[0]
                    if firing_unit.size > 0:
                        cat_binned_spikes[i,j] = firing_unit + 1
            cat_binned_spikes_list.append(cat_binned_spikes)
            
            
# =============================================================================
#         # Make plots to confirm code working
#         # Check firing rates of neurons in both clusters before and after processing
#         # Just do convolution
#         window_size = 250
#         box_kern = np.ones((1,window_size))
#         for cluster in range(len(cat_clust_list)):
#             for nrn in cat_clust_list[cluster].shape[0]:
# =============================================================================
        cluster = 0
        trial = 0
        plt.subplot(211)
        raster(cat_binned_spikes_list[cluster][trial,:])
        plt.subplot(212)
        raster(cat_clust_list[cluster][trial,:,:])
                
        
        

                    
# =============================================================================
#         cat_clust_list = []
#         for cluster in range(n_components):
#             this_cluster = cat_binned_spikes[this_groups == cluster,:]
#             cat_clust_list.append(this_cluster)
#             
#         # Equalize the size of the clusters -> Make more trials by sampling spikes
#         # randomly from trials within the clusters
#         new_cat_clust_list = []
#         for cluster in cat_clust_list:
#             sample_inds = np.random.randint(0,cluster.shape[0]-1,size=cat_binned_spikes.shape)
#             new_clust = np.zeros(cat_binned_spikes.shape)
#             for trial in range(new_clust.shape[0]):
#                 for time in range(new_clust.shape[1]):
#                     new_clust[trial,time] = cluster[sample_inds[trial,time],time]
#             new_cat_clust_list.append(new_clust)
#             
#         # Plot firing rates and confirm separation of clusters
#         nrn = 1
#         plt.plot(np.sum(new_cat_clust_list[0] == nrn,axis=0)/new_cat_clust_list[0].shape[0])
#         plt.plot(np.sum(cat_clust_list[0] == nrn,axis=0)/cat_clust_list[0].shape[0])
# =============================================================================
        
        
        # ============================================================================= 
        # =============================================================================
        # Fit data to HMM
        model_num_states = 3
        
        for cluster in range(len(cat_binned_spikes_list)):
            data = cat_binned_spikes_list[cluster]
            model_VI, model_MAP = hmm_cat_var_multi(
                                        binned_spikes = data, 
                                        num_seeds = 100,
                                        num_states = model_num_states, 
                                        initial_conds_type = 'des', 
                                        max_iter = 1500, 
                                        threshold = 1e-4, 
                                        n_cpu = mp.cpu_count())
            ### MAP Outputs ###
            alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_MAP.E_step()
            
            for i in range(data.shape[0]):
                fig = plt.figure()
                raster(data[i,:],expected_latent_state=expected_latent_state[:,i,:])
                plt.savefig(plot_dir + '/' + 'clust%i_%i_map_%ist.png' % (cluster,i,model_num_states))
                plt.close(fig)
        
            plt.figure()
            hinton(model_MAP.p_transitions)
            plt.title('log_post = %f' %model_MAP.log_posterior[-1])
            plt.suptitle('Model converged = ' + str(model_MAP.converged))
            plt.savefig(plot_dir + '/' + 'transitions_map_clust%i_%ist.png' % (cluster,model_num_states))
            plt.close()
            
            plt.figure()
            hinton(model_MAP.p_emissions)
            plt.title('log_post = %f' %model_MAP.log_posterior[-1])
            plt.suptitle('Model converged = ' + str(model_MAP.converged))
            plt.savefig(plot_dir + '/' + 'emissions_map_clust%i_%ist.png' % (cluster,model_num_states))
            plt.close()
            
            plt.figure()
            plt.plot(np.mean(expected_latent_state,axis=1).T)
            plt.title('Mean state probabilities')
            plt.savefig(plot_dir + '/' + 'mean_map_clust%i_p_%ist.png' % (cluster,model_num_states))
            plt.close()
            
            
            ### VI Outputs ###
            alpha, beta, scaling, expected_latent_state, expected_latent_state_pair = model_VI.E_step()
            
            # Save figures in appropriate directories
            for i in range(data.shape[0]):
                fig = plt.figure()
                raster(data[i,:],expected_latent_state=expected_latent_state[:,i,:])
                fig.savefig(plot_dir + '/' + 'clust%i_%i_vi_%ist.png' % (cluster,i,model_num_states))
                plt.close(fig)
        
            fig = plt.figure()
            hinton(model_VI.transition_counts)
            plt.title('ELBO = %f' %model_VI.ELBO[-1])
            plt.suptitle('Model converged = ' + str(model_VI.converged))
            fig.savefig(plot_dir + '/' + 'transitions_vi_clust%i_%ist.png' % (cluster,model_num_states))
            plt.close(fig)
            
            plt.figure()
            hinton(model_VI.emission_counts)
            plt.title('ELBO = %f' %model_VI.ELBO[-1])
            plt.suptitle('Model converged = ' + str(model_VI.converged))
            plt.savefig(plot_dir + '/' + 'emissions_vi_clust%i_%ist.png' % (cluster,model_num_states))
            plt.close()
            
            plt.figure()
            plt.plot(np.mean(expected_latent_state,axis=1).T)
            plt.title('Mean state probabilities')
            plt.savefig(plot_dir + '/' + 'mean_vi_clust%i_p_%ist.png' % (cluster,model_num_states))
            plt.close()