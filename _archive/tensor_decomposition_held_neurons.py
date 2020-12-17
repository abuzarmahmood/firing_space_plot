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
from sklearn.manifold import TSNE as tsne
from sklearn.cluster import KMeans as kmeans

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure

import glob
from tqdm import tqdm

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
dir_list = ['/media/bigdata/veronica_data/VF313_prex1_170821_102612','/media/bigdata/veronica_data/VF313_prex3_170823_100845']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

day1_nrns = [3,7,9,10,18,19,21]
day3_nrns = [3,6,8,11,16,17,18]

# =============================================================================
#  First Day
# =============================================================================
file  = 0

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data1 = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data1.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type'],
                               [25,250,7000,'conv']))
data1.get_data()

# Since all taste don't have the same number of trials, find the lowest number of trials
# and cut the rest of them to the same size
min_trials = np.min([x.shape[1] for x in data1.off_spikes])
data1.off_spikes = [x[:,range(min_trials),:] for x in data1.off_spikes]

data1.get_firing_rates()
data1.get_normalized_firing()

day1_data = np.asarray(data1.normal_off_firing)[:,day1_nrns,:,80:160].swapaxes(-1,-2)
#day1_data = data1.all_normal_off_firing[day1_nrns,:,80:160].swapaxes(1,2) #all_firing_array[taste,:,:,:].swapaxes(1,2)

# =============================================================================
# Third Day
# =============================================================================
file  = 1

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data3 = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data3.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type'],
                               [25,250,7000,'conv']))
data3.get_data()

# Since all taste don't have the same number of trials, find the lowest number of trials
# and cut the rest of them to the same size
min_trials = np.min([x.shape[1] for x in data3.off_spikes])
data3.off_spikes = [x[:,range(min_trials),:] for x in data3.off_spikes]

data3.get_firing_rates()
data3.get_normalized_firing()
#day3_data = data3.all_normal_off_firing[day3_nrns,:,80:160].swapaxes(1,2) #all_firing_array[taste,:,:,:].swapaxes(1,2)
day3_data = np.asarray(data3.normal_off_firing)[:,day3_nrns,:,80:160].swapaxes(-1,-2)
# =============================================================================
# Tensor Decomposition
# =============================================================================
taste = 0
day3_data = day3_data[:,:,:,range(32)]
all_data = np.concatenate((day1_data,day3_data),axis=-1)
all_data_long = all_data[0,:,:,:]
for taste in range(1,all_data.shape[0]):
    all_data_long = np.concatenate((all_data_long, all_data[taste,:,:,:]),axis=-1)


# Fit CP tensor decomposition (two times).
rank = 10
repeats = 10
all_models = []
all_obj = []
for repeat in tqdm(range(repeats)):
    U = tt.cp_als(all_data_long, rank=rank, verbose=False)
    all_models.append(U)
    all_obj.append(U.obj)

U = all_models[np.argmin(all_obj)]


## We should be able to see differences in tastes by using distance matrices on trial factors
trial_factors = U.factors.factors[-1]
trial_distances = dist_mat(trial_factors,trial_factors)
plt.figure();plt.imshow(exposure.equalize_hist(trial_distances))

# pca on trial factors and plot by taste
trial_labels = np.sort([0,1,2,3,4,5]*32)
reduced_trials_pca = pca(n_components = 2).fit_transform(trial_factors)
plt.scatter(reduced_trials_pca[:,0],reduced_trials_pca[:,1],c=trial_labels)

# tsne on trial factors and plot by taste
X_embedded = tsne(n_components = 2,perplexity = 40).fit_transform(trial_factors)
plt.figure();plt.scatter(X_embedded[:,0],X_embedded[:,1],c=trial_labels)

# Compare the low-dimensional factors from the two fits.
fig, _, _ = tt.plot_factors(U.factors)
