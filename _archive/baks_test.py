#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:09:31 2019

@author: abuzarmahmood
"""

######################### Import dat ish #########################
import tables
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp
import tensortools as tt
from sklearn.cluster import KMeans as kmeans


from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure
from scipy import signal
from scipy.signal import butter
from scipy.signal import filtfilt
import glob
from scipy.special import gamma

def BAKS(SpikeTimes, Time):
    
    
    N = len(SpikeTimes)
    a = 4
    b = N**0.8
    sumnum = 0; sumdenum = 0
    
    for i in range(N):
        numerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a)
        denumerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a-0.5)
        sumnum = sumnum + numerator
        sumdenum = sumdenum + denumerator
    h = (gamma(a)/gamma(a+0.5))*(sumnum/sumdenum)
    
    FiringRate = np.zeros((len(Time)))
    for j in range(N):
        K = (1/(np.sqrt(2*np.pi)*h))*np.exp(-((Time-SpikeTimes[j])**2)/((2*h)**2))
        FiringRate = FiringRate + K
        
    return FiringRate


#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#

dir_list = ['/media/bigdata/brads_data/BS28_4Tastes_180801_112138']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    
file = 0

# Get firing rate data

data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type'],
                               [25,250,7000,'conv']))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()

firing_array = np.asarray(data.normal_off_firing)
spikes_array = np.asarray(data.off_spikes)

# =============================================================================
#  Single trial comparison
# =============================================================================
# taste x nrn x trial x time
nrn = 1
test_firing = firing_array[0,nrn,0,:]
test_spikes = spikes_array[0,nrn,0,:]

baks_firing_rate = BAKS(np.where(test_spikes)[0]/1000,np.arange(len(test_spikes))/1000)

plt.figure()
raster(test_spikes)
plt.plot(np.linspace(0,len(test_spikes),len(test_firing)),test_firing/np.max(test_firing))
plt.plot(baks_firing_rate/np.max(baks_firing_rate))
plt.show()

# =============================================================================
# Multitrial comparison
# =============================================================================
# Take out nrns with trials having no firing
no_firing_nrns = np.unique(np.where(np.sum(spikes_array,axis=-1)==0)[1])
nrn_inds = np.asarray([x for x in range(spikes_array.shape[1]) if x not in no_firing_nrns])

firing_array = firing_array[:,nrn_inds,:,:]
spikes_array= spikes_array[:,nrn_inds,:,:]

baks_firing_array = np.zeros(firing_array.shape)
for taste in range(baks_firing_array.shape[0]):
    for nrn in range(baks_firing_array.shape[1]):
        for trial in range(baks_firing_array.shape[2]):
            baks_firing_array[taste,nrn,trial,:] = BAKS(np.where(spikes_array[taste,nrn,trial,:])[0]/1000, np.linspace(0,len(test_spikes),firing_array.shape[-1])/1000)

# Elongate arrays to have all tastes in a matrix
baks_firing_long = baks_firing_array[0,:,:,:]
for taste in range(1,baks_firing_array.shape[0]):
    baks_firing_long = np.concatenate((baks_firing_long,baks_firing_array[taste,:,:,:]),axis=1)

spikes_long = spikes_array[0,:,:,:]
for taste in range(1,spikes_array.shape[0]):
    spikes_long = np.concatenate((spikes_long,spikes_array[taste,:,:,:]),axis=1)

firing_array_long = data.all_normal_off_firing[nrn_inds,:,:]

nrn = 1

plt.figure()
plt.subplot(131)
plt.imshow(baks_firing_long[nrn,:,:],origin='lower',interpolation='nearest',aspect='auto')
plt.subplot(132)
plt.imshow(firing_array_long[nrn,:,:],origin='lower',interpolation='nearest',aspect='auto')
plt.show()
#plt.subplot(133)
#raster(spikes_long[nrn,:,:])
