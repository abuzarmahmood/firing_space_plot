#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:26:16 2019

@author: bradly
"""
import numpy as np
import matplotlib.pyplot as plt

min_freq = 7
max_freq = 11
freq_step = 1
freq_space = np.linspace(min_freq,max_freq,np.int(1/freq_step))

dt = 0.001
max_t = 50
t = np.arange(0,max_t,dt)

#amplitudes = np.random.rand(len(freq_space),len(t))
amplitudes = np.ones((len(freq_space),len(t)))*0.5
lfp = np.zeros(len(t))
for freq in range(len(freq_space)):
    lfp += np.multiply(np.sin(freq_space[freq]*2*np.pi*t.flatten()),amplitudes[freq,:])

lfp  += np.abs(np.min(lfp))
lfp = (lfp/np.max(np.abs(lfp))) * 0.3

num_trials = 30
spikes = np.zeros((num_trials,len(t)))
for time in range(len(t)):
    #spikes[:,time] = np.random.random(spikes.shape[0]) < lfp[time]
	spikes[:,time] = np.random.poisson(lfp[time],num_trials)

spikes[spikes>=1] = 1

plt.imshow(spikes[:,0:400],interpolation='nearest',aspect='auto')
 
plt.plot(t[0:400],lfp[0:400])
  
# =============================================================================
# f, t, Sxx = signal.spectrogram(lfp, fs=1000)
# plt.pcolormesh(t, f[0:10], Sxx[0:10])
# =============================================================================

lfp_signal_detrend = lfp[:,np.newaxis].T
trial_avg_spike_array = spikes
# =============================================================================
# plt.scatter(t,spikes)
# plt.plot(t,lfp/np.max(lfp))
# =============================================================================