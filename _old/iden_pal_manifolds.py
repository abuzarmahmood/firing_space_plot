#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 15:05:55 2018

@author: abuzarmahmood

1) Visualize identity and palatability discrimination manifolds (linear)
2) Investigate degree of overlap / angle betweent the two manifolds to determine
    degree of differentiability between the two
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
from sklearn.manifold import TSNE as tsne

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

file = 0
 
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                               [25,250,7000]))
data.get_data()
data.get_firing_rates()

tastes = np.sort(np.asarray([0,1,2,3]*15))
pals = np.sort(np.asarray([0,1]*30))
#for nrn in range(data.off_spikes[0].shape[0]):
nrn = 0
        
plt.plot(np.mean(np.asarray(data.normal_off_firing),axis=2)[:,nrn,:].T)
plt.figure();data.imshow(data.all_normal_off_firing[nrn,:,:])

# Only take neurons which fire in every trial
# =============================================================================
# all_spikes = np.concatenate((np.asarray(data.off_spikes),np.asarray(data.on_spikes)),axis=2)
# all_spikes = all_spikes[:,nrn,:,2000:4000]
# if not (np.sum((np.sum(all_spikes,axis=2) == 0).flatten()) > 0):
# =============================================================================
    
this_off = np.asarray(data.all_normal_off_firing)
this_off = this_off[:,:,80:160]
total_this_off = this_off[0,:,:]
for nrn in range(1,this_off.shape[0]):
    total_this_off = np.concatenate((total_this_off,this_off[int(nrn),:,:]),axis=1)

reduced_stim_pca = pca(n_components = 45).fit(total_this_off)
reduced_stim = reduced_stim_pca.transform(total_this_off)
plt.plot(np.cumsum(reduced_stim_pca.explained_variance_ratio_))

## Identity
clf = lda()
clf.fit(reduced_stim, tastes)
fit_coefs = clf.coef_[0]
best_sep = np.argsort(np.abs(fit_coefs))[-3:]

plt.figure()
plt.scatter(reduced_stim[:,best_sep[2]],reduced_stim[:,best_sep[1]],c=tastes)
plt.colorbar()

clf.score(reduced_stim, tastes)

fig = plt.figure()
ax = Axes3D(fig)
p = ax.scatter(reduced_stim[:,best_sep[0]],reduced_stim[:,best_sep[1]],reduced_stim[:,best_sep[2]],
               c =tastes,s=20)
fig.colorbar(p)

#
for perp in np.arange(10,30,5):
    reduced_stim_tsne = tsne(perplexity = perp).fit_transform(total_this_off)
    plt.figure();plt.scatter(reduced_stim_tsne[:,0],reduced_stim_tsne[:,1],c=tastes)
    plt.title(perp)

## Palatability
clf = lda()
clf.fit(reduced_stim, pals)
fit_coefs = clf.coef_[0]
best_sep = np.argsort(np.abs(fit_coefs))[-3:]
plt.figure()
plt.scatter(reduced_stim[:,best_sep[2]],reduced_stim[:,best_sep[1]],c=pals)
plt.colorbar()

clf.score(reduced_stim, pals)

fig = plt.figure()
ax = Axes3D(fig)
p = ax.scatter(reduced_stim[:,best_sep[0]],reduced_stim[:,best_sep[1]],reduced_stim[:,best_sep[2]],
               c =pals,s=20)
fig.colorbar(p)