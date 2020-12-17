"""
Kaufman et al 2016
"The Largest Response Component in the Motor Cortex Reflects 
Movement Timing but Not Movement Type"
The largest component of neural population firing is condition agnostic
From inspection, this is likely true for GC firing too
Use LDA to find which components are taste discriminative


Update:
Seems like the stimulus responsive (but not discriminative component)
is either not as prevalent in GC as I thought or is not easily extracted
from the number of neurons we usually get.
Look into dPCA to extract condition-invariant components in a more
principled manner
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


file =  1 
this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'conv',269]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()

firing = data.all_normal_off_firing
data.firing_overview('off',subtract=True, cmap = 'viridis', zscore_bool=1)

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

# Generate firing frame for single trial reduction
# zscore firing for every neuron
# Subtract the mean of all trials from a neuron to subtract 
# the condition invariant component from single trial firing
#mean_sub_firing = firing - np.mean(firing,axis=1)[:,np.newaxis,:] 
#zscore_firing = np.asarray([zscore(x,axis=None) for x in mean_sub_firing])
zscore_firing = np.asarray([zscore(x,axis=None) for x in firing])
# Segment out appropriate sections of firing
relevant_time = (80,200)
zscore_firing = zscore_firing[:,:,relevant_time[0]:relevant_time[1]]
zscore_firing_long = zscore_firing.reshape(\
        (zscore_firing.shape[0],np.prod(zscore_firing.shape[1:])), order = 'C')

#firing_long = firing.reshape(\
#        (firing.shape[0],np.prod(firing.shape[1:])), order = 'C')

reduction_frame = pca(n_components = 5).fit(zscore_firing_long.T)

# How manycomponents explain up to 95% variance
#dims_to_keep = np.sum(reduction_frame.explained_variance_ratio_.cumsum()<0.95)

# Reduce each trial individually
reduced_data = np.asarray([\
        reduction_frame.transform(zscore_firing[:,trial].T).T \
        for trial in range(firing.shape[1])]).swapaxes(0,1)
#reduced_data = reduced_data[:dims_to_keep]

# Check if distribtution of weights is spread out across neurons
# Otherwise the dimensionality reduction is basically picking out neurons
plt.imshow(reduction_frame.components_);plt.show()
firing_overview(reduced_data)
firing_overview(zscore_firing);plt.show()

# Perform LDA on each component
labels = np.sort(list(range(4))*15)

# Use shuffle splits to estimate accuracy predictions for each component
bootstrap_iters = 100
score_array = np.zeros((reduced_data.shape[0],bootstrap_iters))

cv = ShuffleSplit(n_splits = bootstrap_iters, test_size = 0.5, random_state = 0)
clf = lda(solver = 'eigen', shrinkage='auto')
for comp in trange(reduced_data.shape[0]):
    score_array[comp] = cross_val_score(clf, reduced_data[comp],labels, cv = cv) 

# Plot output
# Plot as swarms
score_frame = pd.melt(
            pd.DataFrame(score_array.T, 
            columns = np.arange(score_array.shape[0])),
        )
sns.boxplot(x = 'variable', y = 'value', data = score_frame)
sns.swarmplot(x = 'variable', y = 'value', data = score_frame)
plt.show()

# Visualize distributions as histograms
plt.hist(score_array[:3].T, histtype = 'step',cumulative=True);plt.legend([0,1,2]);plt.show()
