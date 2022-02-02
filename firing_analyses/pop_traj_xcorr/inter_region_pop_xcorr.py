"""
Calculate XCorr's between SPEED of population firing trajectories
on matched vs shuffled trials

Will SPECIFICALLY test whether CHANGES in population activity are coordinated
"""

########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

########################################
# Import modules
########################################

import os
import sys
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
import pandas as pd
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare
import pylab as plt
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA as pca
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

#data_dir = sys.argv[1]
#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201229_150307'
data_dir = '/media/bigdata/Abuzar_Data/AM34/AM34_4Tastes_201219_130532'
name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()

dat.firing_rate_params = dat.default_firing_params
dat.firing_rate_params['type'] = 'baks'
dat.firing_rate_params['baks_resolution'] = 1e-2
dat.firing_rate_params['baks_dt'] = 1e-3
dat.get_firing_rates()

visualize.firing_overview(dat.all_normalized_firing)
plt.show()

## Path to save noise corrs in HDF5
#save_path = '/ancillary_analysis/firing_speed_corr'
#
#with tables.open_file(dat.hdf5_path,'r+') as hf5:
#    if save_path not in hf5:
#        hf5.create_group(os.path.dirname(save_path),os.path.basename(save_path),
#                createparents = True)

########################################
#|  _ \ ___  _ __   |_   _| __ __ _ (_)
#| |_) / _ \| '_ \    | || '__/ _` || |
#|  __/ (_) | |_) |   | || | | (_| || |
#|_|   \___/| .__/    |_||_|  \__,_|/ |
#           |_|                   |__/ 
########################################
# Calculation population trajectories for single trials
#all_baks_firing = dat.all_normalized_firing
time_range = (0,700)

this_taste = dat.firing_array[0][...,time_range[0]:time_range[1]]
this_taste_long = np.reshape(this_taste, (this_taste.shape[0],-1))
scaler_object = scaler().fit(this_taste_long.T)
scaled_long_data = scaler_object.transform(this_taste_long.T).T
visualize.imshow(scaled_long_data);plt.show()

n_components = 10
pca_object = pca(n_components = n_components)\
        .fit(scaled_long_data.T)
plt.plot(np.cumsum(pca_object.explained_variance_ratio_),'-x');
#plt.ylim([0,1])
plt.show()
#plt.plot(pca_object.explained_variance_ratio_,'x');plt.show()
pca_long_data = pca_object.transform(scaled_long_data.T).T
visualize.imshow(pca_long_data);plt.show()

# Use pca to convert data trial by trial
this_taste_swap = this_taste.swapaxes(0,1)
scaled_taste = np.array([scaler_object.transform(trial.T).T \
        for trial in this_taste_swap])
pca_taste = np.array([pca_object.transform(trial.T).T \
        for trial in scaled_taste])

pca_median_taste = np.median(pca_taste,axis=0)
pca_mean_taste = np.mean(pca_taste,axis=0)

# This plot well demonstrates that the mean population
# firing across trials is not very representative of all trials

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(pca_long_data[0],pca_long_data[1],pca_long_data[2], alpha = 0.5)
ax.scatter(pca_median_taste[0],pca_median_taste[1],pca_median_taste[2])
ax.plot(pca_mean_taste[0],pca_mean_taste[1],pca_mean_taste[2])
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pca_median_taste[0],pca_median_taste[1],pca_median_taste[2],
        c = 'red', alpha = 0.5)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(pca_long_data[0],pca_long_data[1], alpha = 0.5)
ax.scatter(pca_median_taste[0],pca_median_taste[1])
ax2 = fig.add_subplot(122)
ax2.scatter(pca_median_taste[0],pca_median_taste[1],
        c = 'red', alpha = 0.5)
plt.show()


########################
# / ___|___  _ __ _ __ 
#| |   / _ \| '__| '__|
#| |__| (_) | |  | |   
# \____\___/|_|  |_|   
########################
# For testing, split population into random halves
# Calculate population trajectory speed correlation for each half
# and bootstreap error bars

time_range = (0,700)
this_taste = dat.firing_array[0][...,time_range[0]:time_range[1]]
this_taste_long = np.reshape(this_taste, (this_taste.shape[0],-1))
scaler_object = scaler().fit(this_taste_long.T)
scaled_long_data = scaler_object.transform(this_taste_long.T).T
visualize.imshow(scaled_long_data);plt.show()

nrn_inds = np.array_split(\
        np.random.permutation(np.arange(scaled_long_data.shape[0])),2)
split_data = [this_taste[x] for x in nrn_inds]  
split_data_long = [np.reshape(x, (x.shape[0],-1)) for x in split_data]
scaler_list = [scaler().fit(x.T) for x in split_data_long]
scaled_long_list = [this_object.transform(this_data.T).T for \
        this_object, this_data in zip(scaler_list, split_data_long)]

#for x in split_data:
#    visualize.firing_overview(x)
#plt.show()

#for x in scaled_long_list:
#    plt.figure()
#    plt.imshow(x,aspect='auto')
#plt.show()

n_components = 25
pca_obj_list = [pca(n_components = n_components)\
        .fit(x.T) for x in scaled_long_list]
#plt.plot(pca_obj_list[0].explained_variance_ratio_,'x')
#plt.plot(pca_obj_list[1].explained_variance_ratio_,'x');plt.show()
pca_long_list = [this_object.transform(this_data.T).T \
        for this_object,this_data in zip(pca_obj_list, scaled_long_list)]

#for x in pca_long_list:
#    plt.figure()
#    plt.imshow(x,aspect='auto')
#plt.show()

# Use pca to convert data trial by trial
taste_swap_list = [x.swapaxes(0,1) for x in split_data]
scaled_taste_list = [np.array([scaler_object.transform(trial.T).T \
        for trial in this_taste_swap]) \
        for scaler_object, this_taste_swap in zip(scaler_list, taste_swap_list)]

pca_taste_list = [np.array([pca_object.transform(trial.T).T \
        for trial in this_taste_swap]) \
        for pca_object, this_taste_swap in zip(pca_obj_list, taste_swap_list)]

# Plot trials from each half for comparison
trial_ind = 2
plt.plot(*pca_taste_list[0][trial_ind,:2])
plt.plot(*pca_taste_list[1][trial_ind,:2])
plt.show()

# Calculate speed for each trajectory
velocity_list = [np.diff(x,axis=-1) for x in pca_taste_list]
speed_list = [np.linalg.norm(x,axis=1) for x in velocity_list]

# Plot trials from each half for comparison
trial_ind = 15
plt.plot(speed_list[0][trial_ind])
plt.plot(speed_list[1][trial_ind])
plt.show()

# Zero-lag xcorr
def norm_zero_lag_xcorr(vec1, vec2):
    """
    Calculates normalized zero-lag cross correlation
    Returns a single number
    """
    auto_v1 = np.sum(vec1**2,axis=-1)
    auto_v2 = np.sum(vec2**2,axis=-1)
    xcorr = np.sum(vec1 * vec2,axis=-1)
    denom = np.sqrt(np.multiply(auto_v1,auto_v2))
    return np.divide(xcorr, denom)

xcorr_vals = norm_zero_lag_xcorr(*speed_list)

# Generate trial-shuffled xcorrs for comparison
shuffle_num = 1000
shuffled_xcorr_list = np.array([\
        norm_zero_lag_xcorr(np.random.permutation(speed_list[0]),
                            speed_list[1]) \
                                    for repeat in np.arange(shuffle_num)])

plt.hist(xcorr_vals.flatten(), label = 'Actual',alpha = 0.5,density=True)
plt.hist(shuffled_xcorr_list.flatten(), label = 'Shuffled',alpha = 0.5,density=True)
plt.legend()
plt.show()
