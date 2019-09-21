#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:02:42 2018

@author: abuzarmahmood

Visualize localization of all tastes and baseline firing in a density plot
"""

# =============================================================================
# =============================================================================
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
from scipy.stats import mannwhitneyu as mwu

from skimage import exposure

from mpl_toolkits.mplot3d import Axes3D

import glob

# =============================================================================
# =============================================================================

dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    # =============================================================================
    # =============================================================================
    baseline_inds = np.arange(0,80)
    stimulus_inds = np.arange(80,160)
    
    off_firing = data.all_normal_off_firing
    sep_firing = np.concatenate([off_firing[:,:,baseline_inds],off_firing[:,:,stimulus_inds]],axis=1)
    
    sep_firing_long = sep_firing[0,:,:]
    for nrn in range(1,sep_firing.shape[0]):
        sep_firing_long = np.concatenate((sep_firing_long,sep_firing[int(nrn),:,:]),axis=1)
        
    this_pca = pca(n_components = 3).fit(sep_firing_long)
    sep_reduced = this_pca.transform(sep_firing_long)
    
    groups = [0]*60 + [1,2,3,4]*15
    groups.sort()
    
    this_frame = pd.DataFrame(dict(
            x = sep_reduced[:,0],
            y = sep_reduced[:,1],
            groups = groups))
    
    import matplotlib.cm as cm
    #colors = cm.jet(np.unique(groups)/np.max(groups))
    colors = ['r','g','b','y','k']
    
    #plt.figure();plt.scatter(x = sep_reduced[:,0], y = sep_reduced[:,1], c = groups); plt.colorbar()
    
    color_maps = ['Reds','prism','Blues','Spectral','Greys']
    plt.figure();
    for group in range(len(np.unique(groups))):
        ax = sns.kdeplot(data=this_frame[this_frame.groups == np.unique(groups)[group]].x,
                                data2 = this_frame[this_frame.groups == np.unique(groups)[group]].y,
                                shade = True, cmap = color_maps[group],shade_lowest=False,
                                alpha = 0.5)
