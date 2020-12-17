"""
Clark et al 2014
"The characteristic direction: a geometrical approach to identify differentially
expressed genes"
Multivariate analysis of different conditions provides greater power in
statistical analysis than element-wise univariate analysis
By this token, it is conceivable that neurons which don't pass the cut
to be classified as taste discriminative using a RM ANOVA collectively
form a population vector that can still discriminate tastes
This can be assessed using a multivariate classifier
For simplicity, LDA is used here
"""

# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   

import numpy as np
import tables
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import sys 
from tqdm import trange
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from itertools import groupby
import pingouin as pg

sys.path.append('/media/bigdata/firing_space_plot/_old')
from ephys_data import ephys_data

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, ShuffleSplit, \
                                                    cross_val_score

# ____        __   _____                     
#|  _ \  ___ / _| |  ___|   _ _ __   ___ ___ 
#| | | |/ _ \ |_  | |_ | | | | '_ \ / __/ __|
#| |_| |  __/  _| |  _|| |_| | | | | (__\__ \
#|____/ \___|_|   |_|   \__,_|_| |_|\___|___/
#                                            

def firing_overview(data, time_step = 25, cmap = 'jet'):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron
    """
    num_nrns = data.shape[0]
    t_vec = np.arange(data.shape[-1])*time_step 

    # Plot firing rates
    square_len = np.int(np.ceil(np.sqrt(num_nrns)))
    fig, ax = plt.subplots(square_len,square_len)
    
    nd_idx_objs = []
    for dim in range(ax.ndim):
        this_shape = np.ones(len(ax.shape))
        this_shape[dim] = ax.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to( 
                    np.reshape(
                        np.arange(ax.shape[dim]),
                        this_shape.astype('int')), ax.shape).flatten())
    
    for nrn in range(num_nrns):
        plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
        plt.gca().set_title(nrn)
        plt.gca().pcolormesh(t_vec, np.arange(data.shape[1]),
                data[nrn,:,:],cmap=cmap)
    return ax

def dat_imshow(x):
    plt.imshow(x,interpolation='nearest',aspect='auto')

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

## Load data
#dir_list = ['/media/bigdata/Abuzar_Data/run_this_file']
dir_list = ['/media/bigdata/jian_you_data/des_ic',
                        '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


file =  0 
this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'conv',269]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()

firing = np.asarray(data.normal_off_firing)
# Index firing by post-stimulus response
time_bounds = (80,200)
firing = firing[:,:,:,time_bounds[0]:time_bounds[1]]
data.firing_overview('off', cmap = 'viridis', zscore_bool=1)

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

idx = make_array_identifiers(firing)
neuron_frame = pd.DataFrame(\
        data = { 'taste' : idx[0].flatten(),
                'neuron' : idx[1].flatten(),
                'trial' : idx[2].flatten(),
                'time' : idx[3].flatten(),
                'firing_rate' : firing.flatten() })

# Convert time into discrete 500ms bins
n_bins = 12
time_bin_frame = neuron_frame.copy()
time_bin_frame['time_bin'] = pd.cut(time_bin_frame.time,
        bins = n_bins ,include_lowest = True, labels = np.arange(n_bins))
# Drop time axis (no longer needed)
# Take mean of firing rate 
time_bin_frame.drop('time',inplace=True,axis=1)
time_bin_frame =\
time_bin_frame.groupby(['neuron','taste','trial','time_bin'])\
                .mean().reset_index()

# Plot all neurons
g = sns.FacetGrid(data = \
            time_bin_frame, 
            col = 'neuron', hue = 'taste')
g.map(sns.pointplot, 'time_bin', 'firing_rate', ci = 68)
plt.show()


# _____         _       
#|_   _|_ _ ___| |_ ___ 
#  | |/ _` / __| __/ _ \
#  | | (_| \__ \ ||  __/
#  |_|\__,_|___/\__\___|
#                       
# ____  _               _           _             _   _           
#|  _ \(_)___  ___ _ __(_)_ __ ___ (_)_ __   __ _| |_(_)_   _____ 
#| | | | / __|/ __| '__| | '_ ` _ \| | '_ \ / _` | __| \ \ / / _ \
#| |_| | \__ \ (__| |  | | | | | | | | | | | (_| | |_| |\ V /  __/
#|____/|_|___/\___|_|  |_|_| |_| |_|_|_| |_|\__,_|\__|_| \_/ \___|
                                                                 
# Mark which neurons are taste discriminative 

anova_list = [
    time_bin_frame.loc[time_bin_frame.neuron == nrn,:]\
            .rm_anova(dv = 'firing_rate', \
            within = ['time_bin','taste'], subject = 'trial') \
            for nrn in tqdm(time_bin_frame.neuron.unique())]

# Extract number of taste discriminative units
taste_p_vec = np.asarray([anova_result['p-unc'][1] \
        for anova_result in anova_list])

# Plot all discriminative neurons
g = sns.FacetGrid(data = \
        time_bin_frame[time_bin_frame.neuron.isin(np.where(taste_p_vec < 0.05)[0])],
            col = 'neuron', hue = 'taste',\
        col_wrap = 8)
g.map(sns.pointplot, 'time_bin', 'firing_rate')
plt.show()

# Plot all non-discriminative neurons
g = sns.FacetGrid(data = \
        time_bin_frame[time_bin_frame.neuron.isin(np.where(taste_p_vec > 0.05)[0])],
            col = 'neuron', hue = 'taste',\
        col_wrap = 8)
g.map(sns.pointplot, 'time_bin', 'firing_rate')
#plt.show()

# Overview plot of non-discriminative neurons
firing_overview(data.all_normal_off_firing[taste_p_vec>0.05]);plt.show()

# __  __       _ _                     ____  _          
#|  \/  |_   _| | |___   ____ _ _ __  |  _ \(_)___  ___ 
#| |\/| | | | | | __\ \ / / _` | '__| | | | | / __|/ __|
#| |  | | |_| | | |_ \ V / (_| | |    | |_| | \__ \ (__ 
#|_|  |_|\__,_|_|\__| \_/ \__,_|_|    |____/|_|___/\___|
#                                                       

# Extract the non-responsive population and check whether we can
# use those neurons to discriminate tastes using a multivariate analysis

non_taste_firing = data.all_normal_off_firing[taste_p_vec>0.05,:,\
        time_bounds[0]:time_bounds[1]] 

# At every time-point, check accuracy of classification using LDA
labels = np.sort(list(range(4))*15)

# Use shuffle splits to estimate accuracy predictions for each component
bootstrap_iters = 10
score_array = np.zeros((non_taste_firing.shape[-1], bootstrap_iters))
cv = ShuffleSplit(n_splits = bootstrap_iters, test_size = 0.25, random_state = 0)
clf = lda(solver = 'eigen', shrinkage='auto')
for t_bin in trange(non_taste_firing.shape[-1]):
    score_array[t_bin] = cross_val_score(clf, non_taste_firing[:,:,t_bin].T,labels, cv = cv) 
dat_imshow(score_array);plt.show()

plt.errorbar( x = np.arange(score_array.shape[0]),
                y = np.mean(score_array,axis=-1),
                yerr = np.std(score_array,axis=-1))
plt.show()
