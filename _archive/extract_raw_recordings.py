#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:38:53 2019

@author: abuzarmahmood
"""

# Import stuff!
import tables
import os
import numpy as np
import pylab as plt
import sklearn.preprocessing


# =============================================================================
# Load from ADC
# =============================================================================
# =============================================================================
# os.chdir('/media/bigdata/testing tank gods_190410_185919')
# dat = np.fromfile('board-ADC-00.dat', dtype = np.dtype('uint16')).astype('float64')
# dat = sklearn.preprocessing.scale(dat)
# =============================================================================

os.chdir('/media/bigdata/Abuzar_Data/js_drive_test_190307_142334')

ports = 'B'

all_data = []
for port in ports:
		for channel in range(32):
			all_data.append(np.fromfile('amp-' + port + '-%03d'%channel + '.dat', dtype = np.dtype('int16')))


data_array = np.asarray(all_data)

sampling_f = 30e3
time_vec = np.arange(data_array.shape[1])/sampling_f

# Downsample data to make it more manageable
from scipy.signal import resample

resampled_data = []
for channel in range(data_array.shape[0]):
    resampled_data.append(resample(data_array[channel,:],data_array.shape[1]//100))
resampled_array = np.asarray(resampled_data)
resampled_time = resample(time_vec,len(time_vec)//100)

# Calculate correlations between data
coeff_mat = np.corrcoef(resampled_array)

plt.imshow(coeff_mat)

square_len = np.int(np.ceil(np.sqrt(data_array.shape[0])))

fig, ax = plt.subplots(square_len,square_len,sharey=True)

nd_idx_objs = []
for dim in range(ax.ndim):
    this_shape = np.ones(len(ax.shape))
    this_shape[dim] = ax.shape[dim]
    nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(ax.shape[dim]),this_shape.astype('int')), ax.shape).flatten())

for channel in range(data_array.shape[0]):
    plt.sca(ax[nd_idx_objs[0][channel],nd_idx_objs[1][channel]])
    plt.plot(time_vec[100:200],resampled_array[channel,100:200])
    plt.title(channel)
