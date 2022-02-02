"""
Calculate consistency of difference vectors between PCA'd population
firing
If vectors are more "consistent" compared to trial-shuffled controls,
then there is coordination between the 2 populations
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
data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM35/AM35_4Tastes_201229_150307'
#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM34/AM34_4Tastes_201219_130532'
name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()

dat.firing_rate_params = dat.default_firing_params
#dat.firing_rate_params['type'] = 'baks'
#dat.firing_rate_params['baks_resolution'] = 1e-2
#dat.firing_rate_params['baks_dt'] = 1e-3
dat.get_firing_rates()

#visualize.firing_overview(dat.all_normalized_firing)
#plt.show()

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
#time_range = (0,700)

this_taste = dat.firing_array[0]#[...,time_range[0]:time_range[1]]
# For testing, split same population into 2
this_taste = np.random.permutation(this_taste)

this_taste_long = np.reshape(this_taste, (this_taste.shape[0],-1))
scaler_object = scaler().fit(this_taste_long.T)
scaled_long_data = scaler_object.transform(this_taste_long.T).T
visualize.imshow(scaled_long_data);plt.show()

pop_list = np.array_split(scaled_long_data,2)

n_components = 3
pca_objects = [pca(n_components = n_components).fit(x.T) for x in pop_list]
pca_long_data = [this_obj.transform(x.T).T \
                        for this_obj,x in zip(pca_objects, pop_list)]

visualize.imshow(pca_long_data[0]);plt.show()

pca_trial_data = np.array([np.reshape(x, (x.shape[0], *this_taste.shape[1:]))\
                    for x in pca_long_data])
pca_trial_data -= np.mean(pca_trial_data,axis=2)[:,:,np.newaxis]

diff_pca_trial = np.squeeze(np.diff(pca_trial_data,axis=0))

random_inds = np.random.permutation(np.arange(pca_trial_data.shape[2]))
shuffled_pca_trial_data = np.stack(
        [pca_trial_data[0,:,random_inds].swapaxes(0,1), pca_trial_data[1]])
shuffled_diff_pca_trial = np.squeeze(np.diff(shuffled_pca_trial_data,axis=0))

mean_diff = np.mean(diff_pca_trial,axis=1)
shuffled_mean_diff = np.mean(shuffled_diff_pca_trial,axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(*diff_pca_trial[:,:,80], alpha = 0.5)
ax.plot(*shuffled_diff_pca_trial[:,:,80], c = 'red', alpha = 0.5)
plt.show()
