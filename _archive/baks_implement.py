#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:09:31 2019

@author: abuzarmahmood
"""

######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *

import glob
from scipy.stats import sem


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
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'baks',700]))
data.get_data()
data.get_firing_rates()
data.firing_rate_comparison(16)

# Get spikes
all_spikes = np.asarray(data.spikes)
cropped_spikes = all_spikes[:,:,:,2000:5000]

# Heatmap of all firing for single neuron
data.get_normalized_firing()
data.imshow(data.all_normal_off_firing[3,:,:])

# Mean firing for single taste
normal_firing = np.asarray(data.normal_off_firing)
mean_normal_firing = np.mean(normal_firing,axis=2)
sem_normal_firing = sem(normal_firing,axis=2)

cropped_firing = normal_firing[:,:,:,200:500]

nrn = 0
for taste in range(mean_normal_firing.shape[0]):
    plt.plot(np.arange(mean_normal_firing.shape[-1]),mean_normal_firing[taste,nrn,:])
    plt.fill_between(x=np.arange(mean_normal_firing.shape[-1]),
                     y1=mean_normal_firing[taste,nrn,:] - sem_normal_firing[taste,nrn,:],
                     y2 = mean_normal_firing[taste,nrn,:] + sem_normal_firing[taste,nrn,:],
                     alpha = 0.5)


os.chdir('/media/bigdata/firing_space_plot/_old')
from baks import BAKS
from baks_new import BAKS_b
from timeit import timeit

t = np.linspace(0,10,1000)
spike_count = 100
spike_inds= np.sort(np.random.choice(np.arange(len(t)),spike_count))
spike_times = t[spike_inds]
spike_array = np.zeros(t.shape)
spike_array[spike_inds] = 1
baks_rate = BAKS(spike_times, t)
baks_b_rate = BAKS_b(spike_times,t)
plt.plot(spike_array,alpha=0.5)
plt.plot(2*baks_rate)
plt.plot(2.5*baks_b_rate);plt.show()

# speed test
def BAKS_time():
    return BAKS(spike_times,t)
def BAKS_b_time():
    return BAKS_b(spike_times,t)
timeit(BAKS_time, number =100)
timeit(BAKS_b_time, number = 100)
    
