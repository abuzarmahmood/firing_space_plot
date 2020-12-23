"""
Noise correlations between:
    1) Whole trial BLA and GC neurons (and shuffle)
    2) Intra region BLA and GC neurons
    3) Binned for 1) and 2)
    4*) Repeat above analyses specifically for TASTE RESPONSIVE/DISCRIMINATIVE
        neurons

Shuffles:
    1) Trial shuffle to show trial-to-trial variability
    - Do STRINGENT trial shuffle
        - Shuffle pairs of trials for same taste in CHRONOLOGICAL ORDER


    ** Actually this second control should not be necessary
    ** If the first shuffle shows lower R than the actual data,
    ** then there should be no need for this
    2) Random chance of 2 neurons with same firing characteristics 
        generating that R value
        - Take Mean + Variance of both neurons to GENERATE trials
            with similar firing rate characteristics and calculate R
"""

########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

########################################
# Import modules
########################################

import os
import sys
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare
import pylab as plt

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

data_dir = '/media/bigdata/Abuzar_Data/AM12/AM12_4Tastes_191106_085215/'
#data_dir = sys.argv[1]
dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()
#dat.get_firing_rates()
#dat.firing_rate_params = dat.default_firing_params 
spikes = np.array(dat.spikes)

########################
# / ___|___  _ __ _ __ 
#| |   / _ \| '__| '__|
#| |__| (_) | |  | |   
# \____\___/|_|  |_|   
########################

##################################################
## Whole Trial
##################################################
time_lims = [2000,4000]
temp_spikes = spikes[...,time_lims[0]:time_lims[1]]
region_spikes = [temp_spikes.swapaxes(0,2)[region_inds]\
        for region_inds in dat.region_units]

unit_count = [len(x) for x in region_spikes]
wanted_order = np.argsort(unit_count)[::-1]
temp_region_spikes = [region_spikes[x] for x in wanted_order]
sorted_unit_count = [len(x) for x in temp_region_spikes]
all_pairs = np.arange(1,1+sorted_unit_count[0])[:,np.newaxis].\
        dot(np.arange(1,1+sorted_unit_count[1])[np.newaxis,:])
pair_inds = list(zip(*np.where(np.tril(all_pairs))))

sum_spikes = [np.sum(x,axis=-1) for x in temp_region_spikes]

# Perform correlation over all pairs for each taste separately
# Compare values to corresponding shuffles
repeats = 10000

#percentile_array = np.zeros((len(pair_inds), sum_spikes[0].shape[-1]))
#for num,this_ind in tqdm(enumerate(pair_inds)):
#    this_pair = np.array([sum_spikes[0][this_ind[0]], 
#                            sum_spikes[1][this_ind[1]]])
#    #corrs = np.abs([spearmanr(this_taste)[0] for this_taste in this_pair.T])
#    corrs = [spearmanr(this_taste)[0] for this_taste in this_pair.T]
#    # Random ok for now but replace with STRINGENT shuffle
#    #shuffled_corrs = np.abs(np.array(\
#    shuffled_corrs = np.array(\
#            [[spearmanr(np.random.permutation(this_taste[:,0]), 
#                            this_taste[:,1])[0] \
#                        for this_taste in this_pair.T]\
#                        for x in np.arange(repeats)])
#
#    percentile_array[num] = [percentileofscore(shuffle,actual) \
#                        for shuffle, actual in zip(shuffled_corrs.T,corrs)]        
#

def calc_corrs(this_ind):
    this_pair = np.array([sum_spikes[0][this_ind[0]], 
                            sum_spikes[1][this_ind[1]]])
    #corrs = np.abs([spearmanr(this_taste)[0] for this_taste in this_pair.T])
    out = \
            list(zip(*[spearmanr(this_taste) for this_taste in this_pair.T]))
    corrs, p_vals = [np.array(x) for x in out] 
    # Random ok for now but replace with STRINGENT shuffle
    #shuffled_corrs = np.abs(np.array(\
    out =  \
                    [[spearmanr(np.random.permutation(this_taste[:,0]), 
                            this_taste[:,1]) \
                        for this_taste in this_pair.T]\
                        for x in np.arange(repeats)]
    trans_out = [list(zip(*x)) for x in out]
    shuffled_corrs, shuffled_p_vals = [np.array(x) for x in list(zip(*trans_out))]
    percentile_list = [percentileofscore(shuffle,actual) \
                            for shuffle, actual in \
                            zip(shuffled_corrs.T,corrs)]        
    return corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentile_list

# Check corrs are significant individually aswell
out = parallelize(calc_corrs,pair_inds)
corr_array, p_val_array, shuffled_corrs, shuffled_p_vals, percentile_array = \
        [np.array(x) for x in list(zip(*out))]
mean_shuffled_corrs = np.mean(shuffled_corrs,axis=1)
mean_shuffled_p_vals = np.mean(shuffled_p_vals,axis=1)

#========================================
# Paired plots for every comparison
#for this_corr, this_shuffled_corr in \
#        zip(corr_array.flatten(), mean_shuffled_corrs.flatten()):
#    plt.plot([1,0],[this_corr,this_shuffled_corr])
#plt.show()

#========================================
# Side-by-side significance matrices
alpha = 0.05
#fig,ax = plt.subplots(1,2)
#for this_ax, this_dat in zip(ax,[p_val_array, mean_shuffled_p_vals]):
plt.imshow(p_val_array <= alpha,aspect='auto', vmin = 0, vmax = 1)
plt.xlabel('Taste')
plt.ylabel('All Neuron Pair Combinations')
plt.suptitle('Noise Correlation Significance')
plt.show()

# Side-by-side corrleation matrices
#fig,ax = plt.subplots(1,2)
#for this_ax, this_dat in zip(ax,[corr_array, mean_shuffled_corrs]):
#    this_ax.imshow(this_dat,aspect='auto', vmin = 0, vmax = 1)
#plt.show()

#========================================
# Same plot as above but histogram with sides that are nrns
# and bins counting how many significant correlations
mat_inds = np.array(list(np.ndindex(p_val_array.shape)))
inds = np.array(pair_inds)
sig_hist_array = np.zeros(inds[-1]+1)
for this_mat_ind,this_val in zip(mat_inds[:,0],(p_val_array<alpha).flatten()):
    sig_hist_array[inds[this_mat_ind,0], inds[this_mat_ind,1]] += \
                                    this_val
sorted_region_names = [dat.region_names[x] for x in wanted_order]

plt.imshow(sig_hist_array,aspect='auto',cmap='viridis');
# ROWS are region0, COLS are region1
plt.xlabel(str(sorted_region_names[1])+' Neuron #');
plt.ylabel(str(sorted_region_names[0])+' Neuron #')
plt.suptitle('Count of Significant\nNoise Correlations across all comparisons')
plt.colorbar();plt.show()

#========================================
# histogram of corr percentile relative to respective shuffle
freq_hist = np.histogram(percentile_array.flatten(),percentile_array.size//20)
chi_test = chisquare(freq_hist[0])
plt.hist(percentile_array.flatten(),percentile_array.size//20)
plt.suptitle('Percentile relative to respective shuffles\n' +\
        'Chi_sq vs. Uniform Discrete Dist \n p_val :' \
        + str(np.format_float_scientific(chi_test[1],3)))
plt.xlabel('Percentile of Corr Relative to respective shuffle distribution')
plt.ylabel('Frequency')
plt.show()

#========================================
# heatmap of corr percentile relative to respective shuffle
#plt.imshow(percentile_array,aspect='auto',cmap='viridis');plt.show()

#========================================
# histogram of corr percentile relative to respective shuffle
# Plot scatter plots to show correlation of actual data and a shuffle
# Find MAX corr
corr_mat_inds = np.where(corr_array == np.max(corr_array,axis=None))
nrn_inds = pair_inds[corr_mat_inds[0][0]]
this_pair = np.array([sum_spikes[0][nrn_inds[0],...,corr_mat_inds[1][0]], 
                        sum_spikes[1][nrn_inds[1],...,corr_mat_inds[1][0]]])

shuffled_pair = np.array([[np.random.permutation(this_pair[0]),this_pair[1]]\
            for x in np.arange(repeats)])
shuffled_pair = np.reshape(shuffled_pair,(shuffled_pair.shape[1],-1))

fig, ax = plt.subplots(2,2)
fig.suptitle('Firing Rate Scatterplots')
ax[0,0].set_title('Actual Data - Pair : ' + str(nrn_inds) +\
                        '\nTaste :' + str(corr_mat_inds[1][0]))
ax[1,0].set_title('Shuffle') 
ax[0,0].scatter(this_pair[0],this_pair[1])
ax[1,0].scatter(np.random.permutation(shuffled_pair[0]),shuffled_pair[1]);

# Find MIN corr
corr_mat_inds = np.where(corr_array == np.min(corr_array,axis=None))
nrn_inds = pair_inds[corr_mat_inds[0][0]]
this_pair = np.array([sum_spikes[0][nrn_inds[0],...,corr_mat_inds[1][0]], 
                        sum_spikes[1][nrn_inds[1],...,corr_mat_inds[1][0]]])

shuffled_pair = np.array([[np.random.permutation(this_pair[0]),this_pair[1]]\
            for x in np.arange(repeats)])
shuffled_pair = np.reshape(shuffled_pair,(shuffled_pair.shape[1],-1))

ax[0,1].set_title('Actual Data - Pair : ' + str(nrn_inds) +\
                        '\nTaste :' + str(corr_mat_inds[1][0]))
ax[1,1].set_title('Shuffle') 
ax[0,1].scatter(this_pair[0],this_pair[1])
ax[1,1].scatter(np.random.permutation(shuffled_pair[0]),shuffled_pair[1]);

for this_ax in ax.flatten():
    this_ax.set_xlabel('Nrn 1 Firing')
    this_ax.set_ylabel('Nrn 0 Firing')
plt.tight_layout()
plt.show()
