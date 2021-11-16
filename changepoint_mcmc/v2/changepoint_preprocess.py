"""
Code to preprocess spike trains before feeding into model
"""

########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import os
import sys
import json

import numpy as np
import pickle
import argparse

def preprocess_single_taste(spike_array, time_lims, bin_width, data_transform):
    """
    spike_array : trials x neurons x time
    time_lims : Limits to cut spike_array according to
    bin_width: Width of bins to create spike counts
    data_transform : Data-type to return {actual, shuffled, simulated}

    ** Note, it may be useful to use x-arrays here to keep track of coordinates
    """

    ##################################################
    ## Create shuffled data
    ##################################################
    # Shuffle neurons across trials FOR SAME TASTE
    shuffled_dat = np.array([np.random.permutation(neuron) \
                for neuron in np.swapaxes(taste_dat,2,0)])
    shuffled_dat = np.swapaxes(shuffled_dat,0,2)

    ##################################################
    ## Create simulated data 
    ##################################################
    # Inhomogeneous poisson process using mean firing rates

    mean_firing = np.mean(taste_dat,axis=1)

    # Simulate spikes
    simulated_spike_array = np.array(\
            [np.random.random(mean_firing.shape) < mean_firing \
            for trial in range(shuffled_dat_binned.shape[1])])*1
    simulated_spike_array = simulated_spike_array.swapaxes(0,1)
    # taste_dat : 

    ##################################################
    ## Bin Data 
    ##################################################
    this_dat_binned = \
            np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
            reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
    this_dat_binned = np.vectorize(np.int)(this_dat_binned)

def preprocess_all_taste():
    pass
