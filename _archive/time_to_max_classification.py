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

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data

from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

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
data.firing_overview('off')

dat = np.asarray(data.normal_off_firing)
dat_mean = np.mean(np.asarray(dat),axis=-2)

dat_mean_long = dat_mean[0,:,:]
for taste in range(1,dat_mean.shape[0]):
    dat_mean_long = np.concatenate((dat_mean_long,dat_mean[taste,:,:]),axis=-1)

# Use all data from single neurons to define projection axes
n_components = 5
mean_red_pca = pca(n_components = n_components).fit(dat_mean_long.T)

# Reduce mean data
mean_red = np.asarray([mean_red_pca.transform(dat_mean[taste,:,:].T) for taste in range(dat_mean.shape[0])])

# Use same transformation to reduce single trials
all_data_red = np.zeros((dat.shape[0],n_components,dat.shape[2],dat.shape[3]))
for taste in range(dat.shape[0]):
    for trial in range(dat.shape[2]):
        all_data_red[taste,:,trial,:] = mean_red_pca.transform(dat[taste,:,trial,:].T).T
        
all_data_red_long = all_data_red[0,:,:,:]
for taste in range(1,all_data_red.shape[0]):
    all_data_red_long = np.concatenate((all_data_red_long,all_data_red[taste,:,:,:]),axis=1)

plt.figure()
for component in range(all_data_red_long.shape[0]):
    plt.subplot(1,all_data_red_long.shape[0],component+1)
    data.imshow(all_data_red_long[component,:,:])
# =============================================================================
# LDA
# =============================================================================
# Pick one taste as one class and all other tastes as other class
# Classifiy with increasing time-windows following stimulus delivery and track classification accuracy
# Use 50%-50% training-testing sets
        
repeats = 500
time_inds = np.arange(80,260)
classifier_dat = all_data_red_long[:,:,time_inds]

plt.figure()
for component in range(classifier_dat.shape[0]):
    plt.subplot(1,classifier_dat.shape[0],component+1)
    data.imshow(classifier_dat[component,:,:])

# Downsample data to increase number of trials
## Don't split downsampled trials from same batch across training and testing data

train_fraction = 0.5
total_features = classifier_dat.shape[0]*classifier_dat.shape[2]
train_trial_num = classifier_dat.shape[1]*train_fraction
downsample_ratio_vec = np.linspace(0.1,10,100)
down_sample_ratio = np.ceil(downsample_ratio_vec[np.nanargmin(((total_features/downsample_ratio_vec) - (train_trial_num * downsample_ratio_vec))**2)])
down_sample_ratio = down_sample_ratio.astype('int') 

new_time_points = (classifier_dat.shape[-1] // down_sample_ratio) 
downsample_inds = np.asarray([x for x in range(down_sample_ratio)])

batched_classifier_dat = np.zeros((classifier_dat.shape[0],classifier_dat.shape[1]*down_sample_ratio,new_time_points))

for time in range(new_time_points):
    for trial in range(classifier_dat.shape[1]):
            batched_classifier_dat[:,trial*down_sample_ratio : (trial*down_sample_ratio)+down_sample_ratio,time] = \
            classifier_dat[:,trial,(time*down_sample_ratio) : (time*down_sample_ratio)+down_sample_ratio]
            
plt.subplot(121);data.imshow(batched_classifier_dat[0,:,:]);plt.subplot(122);data.imshow(classifier_dat[0,:,:])


trial_labels = np.sort(np.asarray(list(range(classifier_dat.shape[1]))*down_sample_ratio))
taste_labels = np.sort([x for x in range(all_data_red.shape[0])]*all_data_red.shape[2]*down_sample_ratio)

classification_accuracies = np.zeros((all_data_red.shape[0],repeats,new_time_points))

for repeat in range(repeats):
    for taste in range(all_data_red.shape[0]):
        
        train_trials = np.sort(np.random.permutation(np.arange(len(np.unique(trial_labels))))[:len(np.unique(trial_labels))//2])
        test_trials = np.asarray([x for x in np.arange(len(np.unique(trial_labels))) if x not in train_trials])
        
        this_labels = (taste_labels==taste)*1
        
        train_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in train_trials])
        test_inds = np.asarray([x for x in range(len(trial_labels)) if trial_labels[x] in test_trials])
        
        train_labels = this_labels[train_inds]
        test_labels = this_labels[test_inds]
        
        train_dat = batched_classifier_dat[:,train_inds,:]
        test_dat = batched_classifier_dat[:,test_inds,:]
        
        # Flattened array such that dims are trials x (bins of pcas x time) i.e. every time point is 5 PCAs
        train_dat_long = np.reshape(np.swapaxes(np.swapaxes(train_dat,0,1),1,2),(train_dat.shape[1],train_dat.shape[0]*train_dat.shape[2]))
        test_dat_long = np.reshape(np.swapaxes(np.swapaxes(test_dat,0,1),1,2),(test_dat.shape[1],test_dat.shape[0]*test_dat.shape[2]))
        
        for time in range(1,new_time_points):
            this_train_dat = train_dat_long[:,:time*n_components]
            this_test_dat = train_dat_long[:,:time*n_components]
            
            clf = LDA()
            clf.fit(this_train_dat, train_labels)
            classification_accuracies[taste,repeat,time] = clf.score(this_test_dat, test_labels)

# Plot results from classification
# Plot means of distances
mean_accuracies = np.mean(classification_accuracies,axis=1)
std_accuracies = np.std(classification_accuracies,axis=1)
plt.figure()
for taste in range(mean_accuracies.shape[0]):
    plt.subplot(mean_accuracies.shape[0],1,taste+1)
    plt.plot(np.arange(mean_accuracies.shape[-1]),mean_accuracies[taste,:])
    plt.fill_between(np.arange(mean_accuracies.shape[-1]),mean_accuracies[taste,:]+std_accuracies[taste,:],
                     mean_accuracies[taste,:]-std_accuracies[taste,:],alpha=0.5)
    plt.title(taste)
    
print(mean_accuracies.T)
    