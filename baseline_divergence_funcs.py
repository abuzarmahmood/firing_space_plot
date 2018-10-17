import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from scipy.spatial import distance_matrix as dist_mat
from scipy.stats.mstats import zscore
from scipy.stats import pearsonr
from scipy import signal

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal
import copy

import multiprocessing as mp

def firing_rates(spike_array, step_size, window_size):
    """
    Takes in a spike array and converts to firing rates by moving window smoothing
    **Note: The normalization only considers data in the given array,
            not recommended if there are multiple arrays for the same set of neurons.
            E.g. when there are multiple tastes as separate arrays
            
    PARAMS
    : spike_array: Array with dims neurons x trials x time
    : step_size: Step size for moving window
    : window_size: Window size for moving window
    """
    tot_time = spike_array.shape[-1]
    firing_len = int((tot_time-window_size)/step_size)-1
    
    firing = np.zeros((spike_array.shape[0],spike_array.shape[1],firing_len))
    for i in range(firing.shape[0]): # nrns
        for j in range(firing.shape[1]): # trials
            for k in range(firing.shape[2]): # time
                firing[i, j, k] = np.mean(spike_array[i, j, step_size*k:step_size*k + window_size])
                if np.isnan(firing[i, j, k]):
                    print('found nan')
                    break

    normal_firing = copy.deepcopy(firing)
    
    # Normalized firing of every neuron over entire dataset
    firing_array = np.asarray(normal_firing) #(nrn x trial x time)
    for m in range(firing_array.shape[0]): # nrn
        min_val = np.min(firing_array[m,:,:]) # Find min and max vals in entire dataset
        max_val = np.max(firing_array[m,:,:])
        for n in range(normal_firing.shape[1]): # trial
            normal_firing[m,n,:] = (normal_firing[m,n,:] - min_val)/(max_val-min_val)

    return firing, normal_firing

def firing_correlation(firing_array, baseline_window, stimulus_window, 
                    data_step_size = 25, shuffle_repeats = 100, accumulated = False):

    """
    General function, not bound by object parameters
    Calculates correlations in 2 windows of a firin_array (defined below) 
        according to either accumulated distance or distance of mean points
    PARAMS
    :firing_array: (nrn x trial x time) array of firing rates
    :baseline_window: Tuple of time in ms of what window to take for BASELINE firing
    :stimulus_window: Tuple of time in ms of what window to take for STIMULUS firing
    :data_step_size: Resolution at which the data was binned (if at all)
    :shuffle repeats: How many shuffle repeats to perform for analysis control
    :accumulated:   If True -> will calculate temporally integrated pair-wise distances between all points
                    If False -> will calculate distance between mean of all points  
    """
    # Calculate indices for slicing data
    baseline_start_ind = int(baseline_window[0]/data_step_size)
    baseline_end_ind = int(baseline_window[1]/data_step_size)
    stim_start_ind = int(stimulus_window[0]/data_step_size)
    stim_end_ind = int(stimulus_window[1]/data_step_size)
    
    pre_dat = firing_array[:,:,baseline_start_ind:baseline_end_ind]
    stim_dat = firing_array[:,:,stim_start_ind:stim_end_ind]
    
    if accumulated:
        # Calculate accumulated pair-wise distances for baseline data
        pre_dists = np.zeros((pre_dat.shape[1],pre_dat.shape[1],pre_dat.shape[2]))
        for time_bin in range(pre_dists.shape[2]):
            pre_dists[:,:,time_bin] = dist_mat(pre_dat[:,:,time_bin].T,pre_dat[:,:,time_bin].T)
        sum_pre_dist = np.sum(pre_dists,axis = 2)
        
        # Calculate accumulated pair-wise distances for post-stimulus data
        stim_dists = np.zeros((stim_dat.shape[1],stim_dat.shape[1],stim_dat.shape[2]))
        for time_bin in range(stim_dists.shape[2]):
            stim_dists[:,:,time_bin] = dist_mat(stim_dat[:,:,time_bin].T,stim_dat[:,:,time_bin].T)
        sum_stim_dist = np.sum(stim_dists,axis = 2)
        
        # Remove lower triangle in correlation to not double count points
        indices = np.mask_indices(stim_dat.shape[1], np.triu, 1)
        rho, p = pearsonr(sum_pre_dist[indices], sum_stim_dist[indices])
        
        pre_mat, stim_mat = sum_pre_dist, sum_stim_dist

    else:
        # Calculate accumulate pair-wise distances for baseline data
        mean_pre = np.mean(pre_dat,axis = 2)
        mean_pre_dist = dist_mat(mean_pre.T, mean_pre.T)
        
        # Calculate accumulate pair-wise distances for post-stimulus data
        mean_stim = np.mean(stim_dat, axis = 2)
        mean_stim_dist = dist_mat(mean_stim.T, mean_stim.T)
        
        indices = np.mask_indices(stim_dat.shape[1], np.triu, 1)
        rho, p = pearsonr(mean_pre_dist[indices], mean_stim_dist[indices])
        
        pre_mat, stim_mat = mean_pre_dist, mean_stim_dist
    
    
    rho_sh_vec = np.empty(shuffle_repeats)
    p_sh_vec = np.empty(shuffle_repeats)
    for repeat in range(shuffle_repeats):
        rho_sh_vec[repeat], p_sh_vec[repeat] = pearsonr(np.random.permutation(pre_mat[indices]), stim_mat[indices])
    
    return rho, p, rho_sh_vec, p_sh_vec, pre_mat, stim_mat

def blockify(mat,thresh,mat_type = 'dist'):
    """
    Takes in a symmetric distance matrix and attempts to convert it into 
    block-diagonal form.
    Returns reordered matrix, predicted groups, and reordering vector
    """
    pred_group = np.zeros((1,mat.shape[0]))[0]
    if mat_type in 'correlation':
        for item in range(mat.shape[0]):
            if pred_group[item] == 0:
                this_group = np.max(np.unique(pred_group)) + 1
                pred_group[item] = this_group
                pred_group[np.abs(mat[item,:]) > thresh*np.max(mat)] = this_group
         
    elif mat_type in 'distance':
        for item in range(mat.shape[0]):
            if pred_group[item] == 0:
                this_group = np.max(np.unique(pred_group)) + 1
                pred_group[item] = this_group
                pred_group[np.abs(mat[item,:]) < thresh*np.max(mat)] = this_group
         
    
    order_vec = np.argsort(pred_group)
    new_mat = mat[order_vec,:]
    new_mat = new_mat[:,order_vec]
    
    return new_mat, pred_group, order_vec

def entropy_proxy(mat):
    """
    Calculates a "dreamed-up" measure of disorder in
    a matrix
    """
    mat_conv = signal.convolve2d(mat,mat)
    mat_conv = mat_conv/np.max(mat_conv)
    ent_val = np.sum(mat_conv)
    return ent_val


def cluster_GMM():
    reduced_off_pca = pca(n_components = 15).fit(total_off)
    print(sum(reduced_off_pca.explained_variance_ratio_))
    reduced_off = reduced_off_pca.transform(total_off)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                          n_init = 100).fit(reduced_off)
    print(gmm.predict(reduced_off))

def cluster_colocalization(groups_a, groups_B):
    """
    Takes in 2 distance matrices
    Calculates the probability that elements in a particular
    cluster in matrix_a fall into the same cluster in matrix_b
    """
    
    