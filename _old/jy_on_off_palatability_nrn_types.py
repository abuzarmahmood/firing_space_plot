#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:25:21 2018

@author: abuzarmahmood

Are palatability representing neurons in GC differentially modulated by BLAx
than neurons which only represent identity?
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
import tables

# =============================================================================
# =============================================================================
dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

off_clust_dat = pd.DataFrame()
n_components = 3

for file in range(len(file_list)):
    data = tables.open_file(file_list[file])
    palatability_p = data.root.ancillary_analysis.p_pearson[:][0,:,:]
    palatability_r = data.root.ancillary_analysis.p_pearson[:][0,:,:]
    identity_p = data.root.ancillary_analysis.p_identity[:][0,:,:]
    laser_conditions = data.root.ancillary_analysis.laser_combination_d_l[:]
    
    # Perform Mann-Whitney U-test on every timepoint
    required_bins = 100/25
    alpha = 0.05
    significant_pal = np.sum(palatability_p<alpha,axis=0) > required_bins
    significant_iden = np.sum(palatability_p<alpha,axis=0) > required_bins
    
    
# ============================================================================= 
# =============================================================================
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    for nrn in range(data.off_spikes[0].shape[0]):
        for taste in range(4):
            
            # Only take neurons which fire in every trial
            all_spikes = np.concatenate((np.asarray(data.off_spikes),np.asarray(data.on_spikes)),axis=2)
            all_spikes = all_spikes[:,nrn,:,2000:4000]
            if not (np.sum((np.sum(all_spikes,axis=2) == 0).flatten()) > 0):
                
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