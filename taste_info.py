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
file = 0    
    ################ Indent everything after this
    
# Load file and find corresponding directory
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                               [10,500,7000]))
data.get_data()
data.get_firing_rates()    

### Find most discriminative neurons ###

all_firing_array = np.asarray(data.normal_off_firing)
smooth_array = np.zeros(all_firing_array.shape)
for taste in range(smooth_array.shape[0]):
    for neuron in range(smooth_array.shape[1]):
        for trial in range(smooth_array.shape[2]):
            smooth_array[taste,neuron,trial,:] = gauss_filt(all_firing_array[taste,neuron,trial,:],100)

symbols = 5
# Determine equally popoulated quartiles of firing for each neuron individually
# but over all tastes and trials
for neuron in range(smooth_array.shape[1]):
    this_neuron = smooth_array[:,neuron,:,:]
    quartiles = np.linspace(0,100,symbols+1)
    quart_vals = []
    for quart in quartiles:
        quart_vals.append(np.percentile(this_neuron.flatten(),quart))
    
    # Create joint probability table
    joint_p = np.zeros((this_neuron.shape[0],symbols))
    for taste in range(this_neuron.shape[0]):
        for val in range(len(quart_vals)-1):
            this_dat = this_neuron[taste,:,:].flatten()
            joint_p[taste,val] = np.sum((this_dat < quart_vals[val+1]) &
                          (this_dat > quart_vals[val])) / len(this_dat)
    
    # Normalize joint_p -> Needs correcting
    taste_sum = np.sum(np.sum(joint_p,axis=0))
    val_sum = np.sum(np.sum(joint_p,axis=1))
    for taste in range(joint_p.shape[0]):
        joint_p[taste,:] = joint_p[taste,:]/taste_sum
    for val in range(joint_p.shape[1]):
        joint_p[:,val] = joint_p[:,val]/val_sum
# To determine even binning
# =============================================================================
#     fig, ax = plt.subplots()
#     n, bins, patches = ax.hist(this_neuron.flatten(), 50, density=1, histtype='step',
#                            cumulative=True)
#     
#     for val in range(len(quart_vals)-1):
#         print(np.sum((this_neuron.flatten() < quart_vals[val+1]) &
#                          (this_neuron.flatten() > quart_vals[val])) / len(this_neuron.flatten()))
# =============================================================================
        
        

    

