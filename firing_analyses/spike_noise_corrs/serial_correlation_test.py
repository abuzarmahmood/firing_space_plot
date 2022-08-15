"""
Confirm that spike_count trial-series are not serially correlated
using Ljung-Box Test
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
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare
import pylab as plt
from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox
from statsmodels.stats.stattools import durbin_watson as dbtest

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def parallelize(func, fixed_args, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(*fixed_args,this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

#========================================#
## Corr calculation functions
#========================================#

def calc_corrs(array1,array2,ind_tuple, repeats = 10000):
    """
    Calculate correlations and shuffled correlations between given arrays
    inputs ::
        array1,array2 : nrns x trials
        ind_tuple : tuple for indexing neurons from array1 and array2
        repeats : how many shuffles to generate 
            **Note** : 1000 repeats is anecdotally unstable
    """
    # Setting the seed SHOULD force each run of the function
    # to have the same order of shuffled trials
    # This will allow us to pool the different shuffles LATER
    np.random.seed(0)
    corr, p_val  = spearmanr(array1[ind_tuple[0]],array2[ind_tuple[1]])
    out = [spearmanr(\
                        np.random.permutation(array1[ind_tuple[0]]),
                        array2[ind_tuple[1]]) \
                for x in np.arange(repeats)]
    shuffled_corrs, shuffled_p_vals = np.array(out).T
    percentile = percentileofscore(shuffled_corrs,corr)

    return corr, p_val, shuffled_corrs, shuffled_p_vals, percentile

def taste_calc_corrs(array1,array2,ind_tuple):
    """
    Convenience function wrapping calc_corrs to extend to arrays with
    a taste dimension
    inputs ::
        array1,array2 : taste x nrns x trials
        ind_tuple : tuple for indexing neurons from array1 and array2
    """
    outs = list(zip(*[calc_corrs(this1,this2,ind_tuple) \
                    for this1,this2 in zip(array1,array2)]))
    outs = [np.array(x) for x in outs]
    corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentiles = outs
    return  corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentiles

def gen_df(corr_array, p_val_array, percentiles, pair_list, label):
    """
    Helper function to generate dataframes for analyses
    """
    inds = np.array(list(np.ndindex(corr_array.shape)))
    if percentiles is None:
        percentiles = np.array([np.nan]*inds.shape[0])
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'pair_ind' : inds[:,0],
            'pair' : [pair_list[x] for x in inds[:,0]],
            'taste' : inds[:,1],
            'corr' : corr_array.flatten(),
            'p_vals' : p_val_array.flatten(),
            'percentiles' : percentiles.flatten()})

def gen_bin_df(corr_array, p_val_array, percentiles, pair_list, label):
    """
    Helper function to generate dataframes for analyses
    """
    inds = np.array(list(np.ndindex(corr_array.shape)))
    if percentiles is None:
        percentiles = np.array([np.nan]*inds.shape[0])
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'bin_num' : inds[:,0],
            'pair_ind' : inds[:,1],
            'pair' : [pair_list[x] for x in inds[:,1]],
            'taste' : inds[:,2],
            'corr' : corr_array.flatten(),
            'p_vals' : p_val_array.flatten(),
            'percentiles' : percentiles.flatten()})


################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

dir_path_list = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_path_list,'r').readlines()]

diff_spike_list = []

for data_dir in tqdm(dir_list):
    dat = ephys_data(data_dir)
    dat.get_spikes()
    dat.get_region_units()
    spikes = np.array(dat.spikes)

    time_lims = [2000,4000]
    temp_spikes = spikes[...,time_lims[0]:time_lims[1]]
    region_spikes = [temp_spikes.swapaxes(0,2)[region_inds]\
            for region_inds in dat.region_units]

    unit_count = [len(x) for x in region_spikes]
    wanted_order = np.argsort(unit_count)[::-1]
    sorted_region_names = [dat.region_names[x] for x in wanted_order]
    temp_region_spikes = [region_spikes[x] for x in wanted_order]
    sorted_unit_count = [len(x) for x in temp_region_spikes]
    all_pairs = np.arange(1,1+sorted_unit_count[0])[:,np.newaxis].\
            dot(np.arange(1,1+sorted_unit_count[1])[np.newaxis,:])
    #pair_inds = list(zip(*np.where(np.tril(all_pairs))))
    pair_inds = list(zip(*np.where(all_pairs)))

    sum_spikes = [np.sum(x,axis=-1) for x in temp_region_spikes]
    # Try detrending with 1st order difference before corr
    diff_sum_spikes = [np.diff(region,axis=1) for region in sum_spikes]
    # Zscore along trial axis to normalize values across neurons
    diff_sum_spikes = [stats.zscore(region,axis=1) for region in diff_sum_spikes]
    diff_sum_spikes = [np.moveaxis(x,-1,0) for x in diff_sum_spikes]
    diff_spike_list.append(diff_sum_spikes)

db_stats = [[dbtest(x,axis=-1) for x in y] for y in diff_spike_list]
db_stats_flat = [x for y in db_stats for x in y]
db_stats_array = np.concatenate(db_stats_flat, axis=1).flatten()
#norm_range = [1.5,2.5]
norm_range = [1,3]
np.mean(
        np.logical_and(
            db_stats_array >= norm_range[0], db_stats_array<= norm_range[1]
            )
        )

