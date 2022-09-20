"""
Investigate whether explained variance from PCA is a good tool
to judge "dynamicity" of a neural population
Idea being, that if we want to fit a 4 state model to a time-period of population
activity, then it's firing rate should be decomposable into multiple components
such that EACH COMPONENT has a high explained variance.

A metric for this is the L2 norm of the explained_variance vector
"""

############################################################
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
############################################################

import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/projects/pytau')
from ephys_data import ephys_data
import visualize as vz
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
#from joblib import Parallel, cpu_count, delayed
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import mode
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from  matplotlib import colors
import itertools as it
import pymc3 as pm
from scipy.ndimage import gaussian_filter1d as gauss_filt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in file_list]

############################################################
# Load normalized firing for all data
normal_firing_list = []
#ind = 0
for ind in trange(len(file_list)):
    dat = ephys_data(file_list[ind])
    dat.get_spikes()
    dat.firing_rate_params = dat.default_firing_params
    dat.get_firing_rates()
    this_firing = dat.firing_array
    this_normal_firing = dat.firing_array.copy()
    # Zscore
    mean_vals = this_normal_firing.mean(axis=(2,3))
    std_vals = this_normal_firing.std(axis=(2,3))
    this_normal_firing = this_normal_firing - np.expand_dims(mean_vals, (2,3)) 
    this_normal_firing = this_normal_firing / np.expand_dims(std_vals, (2,3)) 
    normal_firing_list.append(this_normal_firing)

#test = np.moveaxis(this_normal_firing,0,1)
#test = np.reshape(test, (len(test), -1))
#np.array([x.shape for x in normal_firing_list])

# Unstack by taste
unstack_firing = [np.split(x, len(x), axis=0) for x in normal_firing_list]
unstack_names = [[x + f'_{i}' for i in range(len(y))] for x,y in zip(basenames, unstack_firing)]
unstack_names = [x for y in unstack_names for x in y]
unstack_firing = [np.squeeze(x) for y in unstack_firing for x in y]

#vz.firing_overview(unstack_firing[0].swapaxes(0,1));plt.show()

# Chop by time
time_lims = [2000,3500]
wanted_inds = np.arange(time_lims[0], time_lims[1]) / dat.firing_rate_params['step_size']
wanted_inds = np.unique(np.vectorize(np.int)(wanted_inds))

unstack_firing = [x[...,wanted_inds] for x in unstack_firing]
nrn_counts = [x.shape[0] for x in unstack_firing]

def calc_variance_explained(array, n_components = 4):
    long_array = np.reshape(array, (len(array),-1))
    pca_obj = PCA(n_components = n_components).fit(long_array.T)
    return pca_obj.explained_variance_ratio_

def return_pca_trials(array, n_components = 4):
    long_array = np.reshape(array, (len(array),-1))
    pca_obj = PCA(n_components = n_components).fit(long_array.T)
    temp = np.stack([pca_obj.transform(x.T).T for x in array.swapaxes(0,1)])
    return temp.swapaxes(0,1) 

pca_trial_list = []
for x in tqdm(unstack_firing):
    pca_trial_list.append(return_pca_trials(x, n_components = 5))

mean_pca = np.stack([x.mean(axis=1) for x in pca_trial_list])

vz.firing_overview(mean_pca.swapaxes(0,1));plt.show()

exp_var_list = []
for x in tqdm(unstack_firing):
    exp_var_list.append(calc_variance_explained(x))
exp_var_array = np.stack(exp_var_list)
exp_var_norm = np.linalg.norm(exp_var_array, axis=-1)

plt.scatter(nrn_counts, exp_var_norm)
plt.xlabel('Neuron Counts')
plt.ylabel('Variance Explained')
plt.show()

sorted_exp_var = exp_var_array[np.argsort(exp_var_norm)]
sorted_names = [unstack_names[i] for i in np.argsort(exp_var_norm)]

vz.imshow(sorted_exp_var);plt.show()
