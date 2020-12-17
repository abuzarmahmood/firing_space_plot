#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:54:58 2018

@author: abuzarmahmood

  _______        _         _____        __      
 |__   __|      | |       |_   _|      / _|     
    | | __ _ ___| |_ ___    | |  _ __ | |_ ___  
    | |/ _` / __| __/ _ \   | | | '_ \|  _/ _ \ 
    | | (_| \__ \ ||  __/  _| |_| | | | || (_) |
    |_|\__,_|___/\__\___| |_____|_| |_|_| \___/

Attempting to determine which neurons are MOST taste responsive
by using mutual information
Will use traditional analysis to determine significance of taste
responsiveness
"""

######################### Import dat ish #########################
import tables
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp
import pandas as pd
import glob
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca

from scipy.spatial import distance_matrix as dist_mat
from scipy.stats import pearsonr

from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from scipy.stats import ranksums

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from matplotlib import mlab

from scipy.signal import gaussian

from skimage import exposure

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

    

#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#

dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

# Iterate over files and calculate correlations with firing rates
#for file in range(len(file_list)):
    
baseline_inds = np.arange(0,80)
stimulus_inds = np.arange(80,160)
symbols = 4

#all_mi = np.zeros(smooth_array.shape[:2])
all_mi = []
shuffle_mi = []
all_joint_p = []
all_dist_corrs = []

for file in range(len(file_list)):
    ################ Indent everything after this
    
    # Load file and find corresponding directory
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type'],
                                   [25,250,7000,'conv']))
    data.get_data()
    data.get_firing_rates()    
    data.get_normalized_firing()
    
    # Smooth firing rates to have smooth PDF
    all_firing_array = np.asarray(data.normal_off_firing)
    smooth_array = np.zeros(all_firing_array.shape)
    for taste in range(smooth_array.shape[0]):
        for neuron in range(smooth_array.shape[1]):
            for trial in range(smooth_array.shape[2]):
                smooth_array[taste,neuron,trial,:] = gauss_filt(all_firing_array[taste,neuron,trial,:],100)
    
    smooth_array += np.random.random(smooth_array.shape)*1e-6
    # Instead of taking out zero values when generating quartiles, add a very
    # small amount of positive noise to all the data.
    # That way an entire bin can't be all zero
    # =============================================================================
    # smooth_long = smooth_array[0,:,:,:]
    # for taste in range(1,smooth_array.shape[0]):
    #     smooth_long = np.concatenate((smooth_long, smooth_array[taste,:,:,:]),axis=1)
    # =============================================================================
        
    # Make joint distribution over baseline and post-stimulus firing
                
    for taste in range(smooth_array.shape[0]):
        for neuron in range(smooth_array.shape[1]):
            this_neuron = smooth_array[taste,neuron,:,:]
            this_pre = this_neuron[:,baseline_inds]
            this_post = this_neuron[:,stimulus_inds]
            
            # Simultaneously do distance correlation to see if methods corroborate eachother
            pre_dists = dist_mat(this_pre,this_pre)
            post_dists = dist_mat(this_post,this_post)
            dist_corr = pearsonr(pre_dists[np.tril_indices(pre_dists.shape[0])],post_dists[np.tril_indices(post_dists.shape[0])])[0]
            all_dist_corrs.append(dist_corr)
            
            # Determine quartiles over all trials, but for post and pre separately
            quartiles = np.linspace(0,100,symbols+1)
            pre_vals = np.percentile(this_pre.flatten(),quartiles)
            post_vals = np.percentile(this_post.flatten(),quartiles)
            
            pre_bins = np.zeros((this_neuron.shape[0],symbols))
            post_bins = np.zeros((this_neuron.shape[0],symbols))
            for trial in range(this_neuron.shape[0]):
                for val in range(symbols):
                    pre_bins[trial,val] = np.sum((this_pre[trial,:] < pre_vals[val+1]) &
                                  (this_pre[trial,:] >= pre_vals[val])) / len(this_pre[trial,:])
                    post_bins[trial,val] = np.sum((this_post[trial,:] < post_vals[val+1]) &
                                  (this_post[trial,:] >= post_vals[val])) / len(this_post[trial,:])
            
            # Binarize quartiles so their frequency can be counted
            binary_pre = np.ones((this_neuron.shape[0],symbols))
            binary_post = np.ones((this_neuron.shape[0],symbols))
            pre_meds = np.median(pre_bins,axis=0)
            post_meds = np.median(post_bins,axis=0)
            
            for trial in range(this_neuron.shape[0]):
                for val in range(symbols):
                    if (pre_bins[trial,val] <= pre_meds[val]):
                        binary_pre[trial,val] = 0
                    if (post_bins[trial,val] <= post_meds[val]):
                        binary_post[trial,val] = 0
            
            # Create joint distribution and normalize it            
            joint_p = np.zeros((symbols,symbols))
            for pre_symbol in range(symbols):
                for post_symbol in range(symbols):
                    joint_p[pre_symbol,post_symbol] = np.sum(binary_pre[:,pre_symbol] * binary_post[:,post_symbol])
                    
            joint_p = joint_p/np.sum(joint_p,axis=None)
            all_joint_p.append(joint_p)
            
            mutual_info = calc_MI(joint_p)
            #all_mi[taste,neuron] = mutual_info
            all_mi.append(mutual_info)
            
            # Now do shuffles of trials and look at mutual information
            shuffle_repeats = 500
            
            for repeat in range(shuffle_repeats):
                binary_pre_sh = binary_pre[np.random.permutation(range(pre_bins.shape[0])),:]
                
                joint_p_sh = np.zeros((symbols,symbols))
                for pre_symbol in range(symbols):
                    for post_symbol in range(symbols):
                        joint_p_sh[pre_symbol,post_symbol] = np.sum(binary_pre_sh[:,pre_symbol] * binary_post[:,post_symbol])
                        
                joint_p_sh = joint_p_sh/np.sum(joint_p_sh,axis=None)
                
                shuffle_mi.append(calc_MI(joint_p_sh))
            
            print([file,taste,neuron])
        
plt.violinplot(all_mi[~np.isnan(all_mi)],shuffle_mi[~np.isnan(shuffle_mi)])
ks_2samp(all_mi,shuffle_mi)
mannwhitneyu(all_mi,shuffle_mi)

## MI Plots
min_inds = np.argsort(np.mean(smooth_array[taste,np.nanargmin(all_mi),:,baseline_inds],axis=0))
max_inds = np.argsort(np.mean(smooth_array[taste,np.nanargmax(all_mi),:,baseline_inds],axis=0))
plt.figure();data.imshow(smooth_array[taste,np.nanargmin(all_mi),min_inds,:160]);plt.colorbar();plt.vlines(80,0,14,colors='r')
plt.figure();plt.imshow(all_joint_p[np.nanargmin(all_mi)],vmin=0,vmax=np.max(all_joint_p,axis=None));plt.colorbar()
plt.figure();data.imshow(smooth_array[taste,np.nanargmax(all_mi),max_inds,:160]);plt.colorbar();plt.vlines(80,0,14,colors='r')
plt.figure();plt.imshow(all_joint_p[np.nanargmax(all_mi)],vmin=0,vmax=np.max(all_joint_p,axis=None));plt.colorbar()
    
## Dist Plots
min_inds = np.argsort(np.mean(smooth_array[taste,np.nanargmin(all_dist_corrs),:,baseline_inds],axis=0))
max_inds = np.argsort(np.mean(smooth_array[taste,np.nanargmax(all_dist_corrs),:,baseline_inds],axis=0))
plt.figure();data.imshow(smooth_array[taste,np.nanargmin(all_dist_corrs),min_inds,:160]);plt.colorbar();plt.vlines(80,0,14,colors='r')
plt.figure();data.imshow(smooth_array[taste,np.nanargmax(all_dist_corrs),max_inds,:160]);plt.colorbar();plt.vlines(80,0,14,colors='r')
# =============================================================================
# =============================================================================
"""
Use MI to detect most taste responsive neurons
"""    

smooth_long = smooth_array[0,:,:,:]
for taste in range(1,smooth_array.shape[0]):
     smooth_long = np.concatenate((smooth_long, smooth_array[taste,:,:,:]),axis=1)   
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


mi_order = np.argsort(all_mi)
for neuron in range(smooth_array.shape[1]):  
    plt.figure();plt.subplot(121);plt.plot(mean_smooth[:,mi_order[neuron],:].T)
    plt.title(np.asarray(all_mi)[mi_order[neuron]])
    plt.subplot(122);plt.imshow(all_joint_p[neuron],vmin=0,vmax=np.max(all_joint_p));plt.colorbar()

    
plt.figure();plt.imshow(all_joint_p[np.nanargmax(all_mi)],vmin=0,vmax=np.max(all_joint_p));plt.colorbar()
plt.figure();plt.imshow(all_joint_p[np.nanargmin(all_mi)],vmin=0,vmax=np.max(all_joint_p));plt.colorbar()
plt.figure();plt.plot(mean_smooth[:,np.nanargmax(all_mi),:].T)
plt.title(np.nanmax(all_mi))
plt.figure();plt.plot(mean_smooth[:,np.nanargmin(all_mi),:].T)
plt.title(np.nanmin(all_mi))
        
        
# To determine even binning
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(this_neuron.flatten(), 50, density=1, histtype='step',
                           cumulative=True)
    
    for val in range(len(quart_vals)-1):
        print(np.sum((this_neuron.flatten() < quart_vals[val+1]) &
                         (this_neuron.flatten() > quart_vals[val])) / len(this_neuron.flatten()))
        
        

    

