#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:21:13 2019

@author: abuzarmahmood

Train classifier using one taste vs all rest to see rise in classification accuracy
on a taste by taste basis
"""

######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data

from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# HMM imports

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

## Load data
dir_list = dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


file  = 4

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'baks',269]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()
plt.figure()
data.firing_overview('off')



nrn = 12
this_dat = np.asarray(data.all_normal_off_firing)[nrn,:,:]

time_inds = np.arange(80,260)
classifier_dat = this_dat[:,time_inds]

train_fraction = 0.5
total_features = classifier_dat.shape[1]
train_trial_num = classifier_dat.shape[0]*train_fraction
downsample_ratio_vec = np.linspace(0.1,10,100)
down_sample_ratio = np.ceil(downsample_ratio_vec[np.nanargmin(((total_features/downsample_ratio_vec) - (train_trial_num * downsample_ratio_vec))**2)])
down_sample_ratio = down_sample_ratio.astype('int') 

new_time_points = (classifier_dat.shape[-1] // down_sample_ratio) 
downsample_inds = np.asarray([x for x in range(down_sample_ratio)])

batched_classifier_dat = np.zeros((classifier_dat.shape[0]*down_sample_ratio,new_time_points))

for time in range(new_time_points):
    for trial in range(classifier_dat.shape[0]):
            batched_classifier_dat[trial*down_sample_ratio : (trial*down_sample_ratio)+down_sample_ratio,time] = \
            classifier_dat[trial,(time*down_sample_ratio) : (time*down_sample_ratio)+down_sample_ratio]
            
plt.figure();plt.subplot(121);data.imshow(batched_classifier_dat);plt.subplot(122);data.imshow(classifier_dat)

trial_labels = np.sort(np.asarray(list(range(classifier_dat.shape[0]))*down_sample_ratio))
num_tastes = 4
taste_labels = np.sort([x for x in range(num_tastes)]*np.int((len(trial_labels)/num_tastes)))


# =============================================================================
# Classification on all classes
# =============================================================================

repeats = 1000
classification_accuracies = np.zeros(repeats)
for repeat in tqdm(range(repeats)):
        
    train_trials = np.sort(np.random.permutation(np.arange(len(np.unique(trial_labels))))[:np.int(np.floor(len(np.unique(trial_labels))*train_fraction))])
    test_trials = np.asarray([x for x in np.arange(len(np.unique(trial_labels))) if x not in train_trials])
    
    train_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in train_trials])
    test_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in test_trials])
    
    train_labels = taste_labels[train_inds]
    test_labels = taste_labels[test_inds]
    
    train_dat = batched_classifier_dat[train_inds,:]
    test_dat = batched_classifier_dat[test_inds,:]
       
    clf = LDA()
    clf.fit(train_dat, train_labels)
    classification_accuracies[repeat] = clf.score(test_dat, test_labels)

plt.figure()
plt.hist(classification_accuracies,30)

# =============================================================================
# 1 vs rest classification of tastes
# =============================================================================

repeats = 1000
classification_accuracies = np.zeros((num_tastes,repeats))
for taste in range(num_tastes):
    for repeat in tqdm(range(repeats)):
            
        train_trials = np.sort(np.random.permutation(np.arange(len(np.unique(trial_labels))))[:np.int(np.floor(len(np.unique(trial_labels))*train_fraction))])
        test_trials = np.asarray([x for x in np.arange(len(np.unique(trial_labels))) if x not in train_trials])
        
        train_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in train_trials])
        test_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in test_trials])
        
        train_labels = taste_labels[train_inds]
        test_labels = taste_labels[test_inds]
        
        this_train_labels = train_labels == taste
        this_test_labels = test_labels == taste
        
        train_dat = batched_classifier_dat[train_inds,:]
        test_dat = batched_classifier_dat[test_inds,:]
           
        clf = LDA()
        clf.fit(train_dat, this_train_labels)
        classification_accuracies[taste,repeat] = clf.score(test_dat, this_test_labels)

plt.hist(classification_accuracies.T)
plt.legend(range(num_tastes))

for taste in range(num_tastes):
    sns.distplot(classification_accuracies[taste,:])
    
# =============================================================================
# Classification with increasing time windows
# =============================================================================

repeats = 10
classification_accuracies = np.zeros((num_tastes,new_time_points-1,repeats))
for time in tqdm(range(1,new_time_points)):
    for taste in range(num_tastes):
        for repeat in range(repeats):
                
            train_trials = np.sort(np.random.permutation(np.arange(len(np.unique(trial_labels))))[:np.int(np.floor(len(np.unique(trial_labels))*train_fraction))])
            test_trials = np.asarray([x for x in np.arange(len(np.unique(trial_labels))) if x not in train_trials])
            
            train_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in train_trials])
            test_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in test_trials])
            
            train_labels = taste_labels[train_inds]
            test_labels = taste_labels[test_inds]
            
            this_train_labels = train_labels == taste
            this_test_labels = test_labels == taste
            
            train_dat = batched_classifier_dat[train_inds,:time]
            test_dat = batched_classifier_dat[test_inds,:time]
               
            clf = LDA()
            clf.fit(train_dat, this_train_labels)
            classification_accuracies[taste,time-1,repeat] = clf.score(test_dat, this_test_labels)

mean_accuracies = np.mean(classification_accuracies,axis=-1)
std_accuracies = np.std(classification_accuracies,axis=-1)

x = np.arange(classification_accuracies.shape[1])
for taste in range(num_tastes):
    plt.plot(x ,mean_accuracies[taste,:])
    plt.fill_between(x ,mean_accuracies[taste,:]-std_accuracies[taste,:],
                     mean_accuracies[taste,:]+std_accuracies[taste,:],alpha=0.5)
    