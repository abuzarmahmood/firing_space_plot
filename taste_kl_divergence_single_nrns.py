#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:16:12 2018

@author: abuzarmahmood

Look at KL Divergence between P(firing-rate|time) distributions for different
tastes in signle neurons
Use P(firing-rate|time) to create a taste classifier
Compare Bayesian Classifier with PSTH classifier
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

import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans as kmeans

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure

import glob

import statsmodels.stats.multicomp as multi

def gauss_filt(data,window_size):
    """
    data : 1D array
    """
    std = int(window_size/2/3)
    window = signal.gaussian(window_size, std=std)
    window = window/window.sum()
    filt_data = np.convolve(data,window,mode='same')
    
    return filt_data

def kl_divergence(vec_a,vec_b):
    """
    Both vectors are 1D arrays
    Vectors will be renormalized
    Order of divergence is D(A||B)
    """
    dat1 = vec_a/np.sum(vec_a)
    dat2 = vec_b/np.sum(vec_b)
    kl_div = np.sum(dat1*np.log(dat1/dat2))
    return kl_div

def percentile_probability(vec,percentile_bounds):
    """
    Given a sequence of values (VEC) and percentiles for a distribution,
    calculates an empirical estimate of the sample falling into each
    percentile
    Output is of len(percentile_bounds)-1
    """
    probs = np.empty(len(percentile_bounds)-1)
    for i in range(len(percentile_bounds)-1):
        probs[i] = np.sum(np.logical_and(vec>=percentile_bounds[i],vec<percentile_bounds[i+1]))/len(vec)
    return probs
# =============================================================================
# =============================================================================
"""
Firing rate / Time distribution for poorly responding neuron
"""
dir_list = ['/media/bigdata/Jenn_Data/']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

file = 0
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                               [20,200,7000]))
data.get_data()
data.get_firing_rates()

# Smooth firing rates to have smooth PDF
#all_firing_array = np.asarray(data.normal_off_firing)
all_firing_array = np.asarray(data.normal_off_firing)
smooth_array = np.zeros(all_firing_array.shape)
for taste in range(smooth_array.shape[0]):
    for neuron in range(smooth_array.shape[1]):
        for trial in range(smooth_array.shape[2]):
            smooth_array[taste,neuron,trial,:] = gauss_filt(all_firing_array[taste,neuron,trial,:],100)

smooth_array += np.random.random(smooth_array.shape)*1e-6

taste = 0
nrn = 13
stimulus_inds = range(100,250)

this_nrn_firing = np.swapaxes(all_firing_array[:,nrn,:,stimulus_inds].T,0,1)

# Use anova to determine taste responsiveness of this neuron by classical methods
all_pairs_comparisons = ['0,1','0,2','0,3','1,2','1,3','2,3']
sig_times = np.empty((len(all_pairs_comparisons),this_nrn_firing.shape[2]))
for time in range(sig_times.shape[1]):
    mc1 = multi.MultiComparison(this_nrn_firing[:,:,time].flatten(),np.sort([0,1,2,3]*this_nrn_firing.shape[1]))
    res1 = mc1.tukeyhsd(alpha=0.05)
    sig_times[:,time] = res1.reject


# Subplots of firing rate and significant difference
fig,ax = plt.subplots(3,1,sharex=True)
im = ax[0].imshow(sig_times,aspect='auto',interpolation = 'nearest')
#fig.colorbar(im,ax=ax[0])
for y in range(len(all_pairs_comparisons)): 
    ax[0].text(x = 0,y=y,s=all_pairs_comparisons[y],color = 'y',size=20)
ax[0].set_title('Multiple Comparisons ANOVA, a = 0.05')

for taste in range(4):
    ax[1].plot(range(this_nrn_firing.shape[2]),np.mean(this_nrn_firing[taste,:,:],axis=0),
      label=taste)
ax[1].legend()
ax[1].set_title('Mean Firing Rates')

for taste in range(4):
    ax[2].errorbar(x=range(this_nrn_firing.shape[2]),y=np.mean(this_nrn_firing[taste,:,:],axis=0),
                 yerr = np.std(this_nrn_firing[taste,:,:],axis=0),label=taste)
ax[2].legend()
ax[2].set_title('Firing rates with SD')
plt.tight_layout()

## Firing rate density plots
f, axes = plt.subplots(1, 4, figsize=(7, 7), sharex=True)
for taste in range(4):   
    this_off = this_nrn_firing[taste,:,:]
    
    this_off_long = this_off[0,:]
    for trial in range(1,this_off.shape[0]):
        this_off_long = np.concatenate((this_off_long,this_off[trial,:]))
        
    time_long = np.matlib.repmat(np.arange(0,this_off.shape[1]),1,this_off.shape[0]).flatten()
    
    inds = this_off_long<0.5
    this_off_long = this_off_long[inds]
    time_long = time_long[inds]
    
    g = sns.jointplot(y=this_off_long,x=time_long,kind='kde',ax=axes[taste],ylim=(0,0.5))
    axes[taste].set_ylim([-0.1,0.5])
    axes[taste].set_title('Taste %i' % taste)
    axes[taste].set_xlabel('Time')
    axes[taste].set_ylabel('Normalized Firing')

## Calculate conditional distribution of firing rate for time p(f|t)
# Divide firing into quartiles as previously and divide time into ~10 bins
symbols = 7
quartiles = np.linspace(0,100,symbols+1)
quart_vals = np.percentile(this_nrn_firing.flatten(),quartiles)

time_bin_count = 20
time_bins = np.floor(np.linspace(0,this_nrn_firing.shape[2],time_bin_count+1))

f_given_t = np.zeros((this_nrn_firing.shape[0],symbols,time_bin_count))
for taste in range(this_nrn_firing.shape[0]):
    for val in range(symbols):
        for this_bin in range(time_bin_count):
            time_inds = range(int(time_bins[this_bin]),int(time_bins[this_bin+1]))
            this_dat = this_nrn_firing[taste,:,time_inds].flatten()
            f_given_t[taste,val,this_bin] = np.sum((this_dat < quart_vals[val+1]) &
                          (this_dat >= quart_vals[val]))

f_given_t = f_given_t/np.sum(f_given_t,axis=0)
f_given_t += 1e-9

fig,ax = plt.subplots(4,1)
for taste in range(this_nrn_firing.shape[0]):
    im = ax[taste].imshow(f_given_t[taste,:,:],vmin = np.min(f_given_t,axis=None),
      vmax=np.max(f_given_t,axis=None),aspect='auto',interpolation='nearest')
    ax[taste].set_title(taste)
    ax[taste].set_xlabel('Time')
    ax[taste].set_ylabel('Firing Rate Quartile')
    fig.colorbar(im,ax=ax[taste])
    
# Calculate pairwise KL divergences for each timepoint
all_kl_div = np.zeros((this_nrn_firing.shape[0],this_nrn_firing.shape[0],time_bin_count))
for taste in range(all_kl_div.shape[0]):
    #all_tastes = range(all_kl_div.shape[0])
    #all_other_tastes = [x for x in range(all_kl_div.shape[0]) if taste != all_tastes[x]]
    for other_taste in range(all_kl_div.shape[0]):#range(len(all_other_tastes)):
        for this_bin in range(time_bin_count):
            dat1 = f_given_t[taste,:,this_bin]
            dat2 = f_given_t[other_taste,:,this_bin]
            all_kl_div[taste,other_taste,this_bin] = kl_divergence(dat1,dat2)
            

fig,ax = plt.subplots(4,1)
for taste in range(this_nrn_firing.shape[0]):
    for other_taste in range(this_nrn_firing.shape[0]):
        ax[taste].plot(np.cumsum(all_kl_div[taste,other_taste,:]),label = other_taste)
    ax[taste].legend()
    ax[taste].set_title(taste)
    ax[taste].set_xlabel('Time')
    ax[taste].set_ylabel('Cumulative KLD')

    
# Using MLE to classify taste
trial_labels = np.sort([0,1,2,3]*this_nrn_firing.shape[1])
this_nrn_firing_long = this_nrn_firing[0,:,:]
for i in range(1,this_nrn_firing.shape[0]):
    this_nrn_firing_long = np.concatenate((this_nrn_firing_long,this_nrn_firing[i,:,:]))

pred_taste = np.zeros((this_nrn_firing.shape[0],this_nrn_firing_long.shape[0]))
for trial in range(this_nrn_firing_long.shape[0]):
    taste_probs = np.zeros((this_nrn_firing.shape[0]))
    this_trial = this_nrn_firing_long[trial,:]
    binned_dat = np.empty((symbols,time_bin_count))
    for this_bin in range(binned_dat.shape[1]):
        time_inds = range(int(time_bins[this_bin]),int(time_bins[this_bin+1]))
        this_dat = this_trial[time_inds]
        binned_dat[:,this_bin] = percentile_probability(this_dat,quart_vals)
    for taste in range(this_nrn_firing.shape[0]):
        taste_probs[taste] = np.sum(np.multiply(f_given_t[taste,:,:],binned_dat),axis=None)
    pred_taste[:,trial] = taste_probs/np.sum(taste_probs)

fig = plt.imshow(pred_taste,aspect='auto',interpolation='nearest')
plt.scatter(range(this_nrn_firing_long.shape[0]),np.argmax(pred_taste,axis=0),color='r')
prob_class_accuracy = np.mean(np.argmax(pred_taste,axis=0)==trial_labels)*100
plt.title('Probabailistic Accuracy = %.1f%%' % prob_class_accuracy)
cbar = plt.colorbar(fig)
cbar.set_label('P(taste)')

# Taste break-down of accuracy
np.mean(np.argmax(pred_taste,axis=0)==trial_labels)*100
 
# To determine even binning
fig, ax = plt.subplots()
n, bins, patches = ax.hist(this_nrn_firing.flatten(), 50, density=1, histtype='step',
                       cumulative=True)

for val in range(len(quart_vals)-1):
    print(np.sum((this_nrn_firing.flatten() < quart_vals[val+1]) &
                     (this_nrn_firing.flatten() > quart_vals[val])) / len(this_nrn_firing.flatten()))

# =============================================================================
# =============================================================================
"""
PSTH Classifier for same neuron matched by parameter count
"""
psth_time_bin_count = f_given_t.shape[1]*f_given_t.shape[2]
psth_time_bins = np.floor(np.linspace(0,this_nrn_firing.shape[2],psth_time_bin_count+1))
binned_firing = np.zeros((this_nrn_firing.shape[0],this_nrn_firing.shape[1],psth_time_bin_count))
for taste in range(this_nrn_firing.shape[0]):
    for trial in range(this_nrn_firing.shape[1]):
        for this_bin in range(psth_time_bin_count):
            time_inds = range(int(psth_time_bins[this_bin]),int(psth_time_bins[this_bin+1]))
            this_dat = this_nrn_firing[taste,trial,time_inds].flatten()
            binned_firing[taste,trial,this_bin] = np.median(this_dat)

binned_firing_long = binned_firing[0,:,:]
for taste in range(1,binned_firing.shape[0]):
    binned_firing_long = np.concatenate((binned_firing_long,binned_firing[taste,:,:]))
    
pred_taste = np.zeros((binned_firing.shape[0],binned_firing_long.shape[0]))
mean_taste_firing = np.mean(binned_firing,axis=1)
for trial in range(binned_firing_long.shape[0]):
    this_dists = dist_mat(mean_taste_firing,binned_firing_long[trial,:][np.newaxis,:]).flatten()
    pred_taste[:,trial] = this_dists/np.sum(this_dists)

fig = plt.imshow(pred_taste,aspect='auto',interpolation='nearest')
plt.scatter(range(this_nrn_firing_long.shape[0]),np.argmin(pred_taste,axis=0),color='r')
psth_class_accuracy = np.mean(np.argmin(pred_taste,axis=0)==trial_labels)*100
plt.title('PSTH Accuracy = %.1f%%' % psth_class_accuracy)
cbar = plt.colorbar(fig)
cbar.set_label('Taste distance')