#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:26:26 2019

@author: abuzarmahmood
"""


######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt

import tensortools as tt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat
from sklearn.cluster import KMeans as kmeans

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure

import glob

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

file  = 4

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'conv',700]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()
data.firing_overview('off')

all_firing_array = np.asarray(data.all_normal_off_firing)

#taste = 2
X = all_firing_array.swapaxes(1,2) #data.all_normal_off_firing[:,:,100:200].swapaxes(1,2) #
rank = 7

# Fit CP tensor decomposition (two times).
U = tt.cp_als(X, rank=rank, verbose=True)

# Compare the low-dimensional factors from the two fits.
fig, _, _ = tt.plot_factors(U.factors)

## We should be able to see differences in tastes by using distance matrices on trial factors
trial_factors = U.factors.factors[-1]
trial_distances = dist_mat(trial_factors,trial_factors)
plt.figure();plt.imshow(trial_distances)