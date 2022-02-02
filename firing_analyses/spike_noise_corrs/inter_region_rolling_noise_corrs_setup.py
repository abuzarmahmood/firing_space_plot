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

    3) Calculation of correlation for baseline
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
from tqdm import tqdm,trange
import pandas as pd
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare
import pylab as plt
import xarray as xr
import xskillscore as xs

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def parallelize(func, iterator, fixed_args, kwargs):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(*fixed_args, **kwargs) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

#========================================#
## Corr calculation functions
#========================================#

#def calc_corrs(array1,array2,ind_tuple, repeats = 10):
#    """
#    Calculate correlations and shuffled correlations between given arrays
#    inputs ::
#        array1,array2 : nrns x trials
#        ind_tuple : tuple for indexing neurons from array1 and array2
#        repeats : how many shuffles to generate 
#            **Note** : 1000 repeats is anecdotally unstable
#    """
#    # Setting the seed SHOULD force each run of the function
#    # to have the same order of shuffled trials
#    # This will allow us to pool the different shuffles LATER
#    np.random.seed(0)
#    corr, p_val  = spearmanr(array1[ind_tuple[0]],array2[ind_tuple[1]])
#    out = [spearmanr(\
#                        np.random.permutation(array1[ind_tuple[0]]),
#                        array2[ind_tuple[1]]) \
#                for x in np.arange(repeats)]
#    shuffled_corrs, shuffled_p_vals = np.array(out).T
#    percentile = percentileofscore(shuffled_corrs,corr)
#
#    return corr, p_val, shuffled_corrs, shuffled_p_vals, percentile
#
#def taste_calc_corrs(array1,array2,ind_tuple):
#    """
#    Convenience function wrapping calc_corrs to extend to arrays with
#    a taste and time dimension
#    inputs ::
#        array1,array2 : taste x time x nrns x trials
#        ind_tuple : tuple for indexing neurons from array1 and array2
#    """
#    iter_inds = np.array(list(np.ndindex(array1.shape[:2])))
#
#    outs = list(zip(*[calc_corrs(array1[tuple(this_ind)],
#                            array2[tuple(this_ind)],ind_tuple) \
#                        for this_ind in tqdm(iter_inds)]))
#
#    outs = [np.reshape(np.array(x),(*array1.shape[:2],-1)) for x in outs]
#    outs = [np.squeeze(x) for x in outs]
#    corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentiles = outs
#    return  corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentiles

#def gen_df(corr_array, p_val_array, percentiles, pair_list, label):
#    """
#    Helper function to generate dataframes for analyses
#    """
#    inds = np.array(list(np.ndindex(corr_array.shape)))
#    if percentiles is None:
#        percentiles = np.array([np.nan]*inds.shape[0])
#    return pd.DataFrame({
#            'label' : [label] * inds.shape[0],
#            'pair_ind' : inds[:,0],
#            'pair' : [pair_list[x] for x in inds[:,0]],
#            'taste' : inds[:,1],
#            'corr' : corr_array.flatten(),
#            'p_vals' : p_val_array.flatten(),
#            'percentiles' : percentiles.flatten()})
#
#def gen_bin_df(corr_array, p_val_array, percentiles, pair_list, label):
#    """
#    Helper function to generate dataframes for analyses
#    """
#    inds = np.array(list(np.ndindex(corr_array.shape)))
#    if percentiles is None:
#        percentiles = np.array([np.nan]*inds.shape[0])
#    return pd.DataFrame({
#            'label' : [label] * inds.shape[0],
#            'bin_num' : inds[:,0],
#            'pair_ind' : inds[:,1],
#            'pair' : [pair_list[x] for x in inds[:,1]],
#            'taste' : inds[:,2],
#            'corr' : corr_array.flatten(),
#            'p_vals' : p_val_array.flatten(),
#            'percentiles' : percentiles.flatten()})

#def to_xarray(out, region_type):
#    actual_array = xr.Dataset(
#            data_vars = dict(
#                corrs = (["pairs","tastes","time"],out[0]),
#                pvals = (["pairs","tastes","time"],out[1]),
#                percentiles = (["pairs","tastes","time"],out[4])),
#            coords = dict(
#                time = starts),
#            attrs = dict(
#                window_size = window_size,
#                step_size = step_size,
#                time_lims = time_lims,
#                comparison_type = 'actual',
#                region_type = region_type)
#            )
#    shuffle_array = xr.Dataset(
#            data_vars = dict(
#                corrs = (["pairs","tastes","time","shuffle"],out[2]),
#                pvals = (["pairs","tastes","time","shuffle"],out[3])),
#            coords = dict(
#                time = starts),
#            attrs = dict(
#                window_size = window_size,
#                step_size = step_size,
#                time_lims = time_lims,
#                comparison_type = 'shuffle',
#                region_type = region_type)
#            )
#    return actual_array, shuffle_array


################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

data_dir = sys.argv[1]
#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM26/AM26_4Tastes_200829_100535'

if data_dir[-1] != '/':
    data_dir += '/'

#name_splits = os.path.basename(data_dir[:-1]).split('_')
#fin_name = name_splits[0]+'_'+name_splits[2]

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()

# Path to save noise corrs in HDF5
#save_path = '/ancillary_analysis/spike_noise_corrs'
save_path = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'spike_noise_corrs/data/rolling'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#with tables.open_file(dat.hdf5_path,'r+') as hf5:
#    if save_path not in hf5:
#        hf5.create_group(os.path.dirname(save_path),os.path.basename(save_path),
#                createparents = True)

########################
# / ___|___  _ __ _ __ 
#| |   / _ \| '__| '__|
#| |__| (_) | |  | |   
# \____\___/|_|  |_|   
########################

## If array not present, then perform calculation
#with tables.open_file(dat.hdf5_path,'r+') as hf5:
#    if os.path.join(save_path,'inter_region_frame') not in hf5:
#        present_bool = False 
#    else:
#        present_bool = True
#
##present_bool = False
#if not present_bool: 

##################################################
# Pre-processing
##################################################

time_lims = [0,4000]
spikes = np.array(dat.spikes)
temp_spikes = spikes[...,time_lims[0]:time_lims[1]]
region_spikes = [temp_spikes.swapaxes(0,2)[region_inds]\
        for region_inds in dat.region_units]

unit_count = [len(x) for x in region_spikes]
wanted_order = np.argsort(unit_count)[::-1]
sorted_region_names = [dat.region_names[x] for x in wanted_order]
temp_region_spikes = [region_spikes[x] for x in wanted_order]
sorted_unit_count = [len(x) for x in temp_region_spikes]
sorted_region_names = [dat.region_names[i] for i in wanted_order]
all_pairs = np.arange(1,1+sorted_unit_count[0])[:,np.newaxis].\
        dot(np.arange(1,1+sorted_unit_count[1])[np.newaxis,:])
#pair_inds = list(zip(*np.where(np.tril(all_pairs))))
pair_inds = list(zip(*np.where(all_pairs)))

window_size = 250
step_size = 25
starts = np.arange(0,(temp_spikes.shape[-1] - window_size) + step_size,step_size)
inds = list(zip(starts,starts+window_size))

sum_spikes = [np.array([np.sum(x[...,this_inds[0]:this_inds[1]],axis=-1) \
                    for this_inds in inds]).T \
                        for x in temp_region_spikes]

# Try detrending with 1st order difference before corr
diff_sum_spikes = [np.diff(region,axis=1) for region in sum_spikes]
# Zscore along trial axis to normalize values across neurons
diff_sum_spikes = [stats.zscore(region,axis=1) for region in diff_sum_spikes]
#diff_sum_spikes = [np.moveaxis(x,1,-1) for x in diff_sum_spikes]
diff_sum_spikes = [np.moveaxis(x,2,0) for x in diff_sum_spikes]

##################################################
## Inter-Region Whole Trial
##################################################
analysis_attrs = dict(time_lims = time_lims, window_size = window_size,
        step_size = step_size)

# Perform correlation over all pairs for each taste separately
# Compare values to corresponding shuffles
pair_inds_array = np.array(pair_inds)
diff_spikes_paired = np.array([this_array[this_inds] \
        for this_array,this_inds in zip(diff_sum_spikes, pair_inds_array.T)])

diff_spikes_x  = xr.DataArray(data = diff_spikes_paired,
        dims = ['region','pair','taste','trial','time'],
        attrs = dict(region_type = 'inter',comparison_type = 'actual',
                    **analysis_attrs))

def array_spearmanr(array1, array2, dim='trial', shuffle = False):
    if shuffle:
        shuff_inds = np.random.permutation(array1[dim])
        return xs.spearman_r(array1.isel({dim : shuff_inds}),
                        array2,dim=dim, keep_attrs = True), \
                    xs.spearman_r_p_value(array1.isel({dim : shuff_inds}),
                            array2,dim=dim, keep_attrs = True)
    else:
        return xs.spearman_r(array1, array2,dim=dim, keep_attrs = True), \
                xs.spearman_r_p_value(array1, array2,dim=dim, keep_attrs = True)

shuffle_num = 100
rho,p = array_spearmanr(diff_spikes_x[0], diff_spikes_x[1])
inter_x = xr.Dataset(data_vars = dict(rho = rho, p = p), attrs = rho.attrs)
#shuff_outs = [array_spearmanr(diff_spikes_x[0], diff_spikes_x[1], 
#                    shuffle = True) for i in trange(shuffle_num)]
shuff_outs = parallelize(array_spearmanr, np.arange(shuffle_num),
        diff_spikes_x, dict(shuffle = True))
shuff_rho, shuff_p = list(zip(*shuff_outs))
shuff_rho = np.stack(shuff_rho)
shuff_p = np.stack(shuff_p)
inter_shuff_x = xr.Dataset(data_vars = dict(
                rho = (['shuff_ind',*inter_x.p.dims], shuff_rho),
                p = (['shuff_ind',*inter_x.p.dims], shuff_p),),
        attrs = dict(region_type = 'inter',comparison_type = 'shuffle',
                            **analysis_attrs))

inter_outs = [inter_x, inter_shuff_x]
#inter_outs = list(zip(*parallelize(taste_calc_corrs,diff_sum_spikes,pair_inds)))
#inter_outs = [np.array(x) for x in inter_outs]
#inter_outs = to_xarray(inter_outs, 'inter_region')

##################################################
## INTRA-Region Whole Trial
##################################################

#this_save_path = os.path.join(save_path,'intra_region')

pair_list = [np.array(list(it.combinations(np.arange(x.shape[0]),2))) \
                        for x in diff_sum_spikes]

diff_paired_intra = [\
        np.stack([this_array[this_inds] for this_inds in  this_region_inds.T])\
        for this_array,this_region_inds in zip(diff_sum_spikes, pair_list)]

diff_intra_x  = [xr.DataArray(data = array,
        dims = ['set','pair','taste','trial','time'],) \
                for array in diff_paired_intra]

intra_outs = [array_spearmanr(x[0], x[1]) for x in diff_intra_x]
intra_x = [xr.Dataset(data_vars = dict(rho = x[0], p = x[1]),
     attrs = dict(region_type = region, comparison_type = 'actual',
                                        **analysis_attrs)) \
                        for region,x in zip(sorted_region_names, intra_outs)]

#shuff_outs = [array_spearmanr(diff_spikes_x[0], diff_spikes_x[1], 
#                    shuffle = True) for i in trange(shuffle_num)]
intra_shuff_outs = [parallelize(array_spearmanr, np.arange(shuffle_num),
        region, dict(shuffle = True)) for region in diff_intra_x]
intra_shuff_outs = [list(zip(*x)) for x in intra_shuff_outs]
intra_shuff_outs = [[np.stack(x) for x in y] for y in intra_shuff_outs]
intra_shuff_x = [xr.Dataset(data_vars = dict(
                rho = (['shuff_ind',*intra_x[num].p.dims], x[0]),
                p = (['shuff_ind',*intra_x[num].p.dims], x[1]),),
    attrs = dict(region_type = sorted_region_names[num], 
                        comparison_type = 'shuffle', **analysis_attrs)) \
                for num,x in enumerate(intra_shuff_outs)]

intra_outs = [*intra_x, *intra_shuff_x]
#intra_outs = [list(zip(*parallelize(taste_calc_corrs,[x,x],this_inds))) \
#        for x,this_inds in zip(diff_sum_spikes,pair_list)]
#intra_outs = [[np.array(x) for x in region] for region in intra_outs]
#intra_outs = [to_xarray(region, region_name) \
#        for region, region_name in zip(intra_outs,dat.region_names)]
#intra_outs = [x for region in intra_outs for x in region]

#out0 = list(zip(*parallelize(taste_calc_corrs,\
#                    [diff_sum_spikes[0],diff_sum_spikes[0]],pair_list[0])))
#out1 = list(zip(*parallelize(taste_calc_corrs,\
#                    [diff_sum_spikes[1],diff_sum_spikes[1]],pair_list[1])))

#intra_region_frame = pd.concat([\
#    gen_df(this_dat[0],this_dat[1],this_dat[-1],this_inds,region_name) \
#    for this_dat,this_inds,region_name in \
#    zip([out0,out1],pair_list,sorted_region_names)])

#shuffle_intra_region_frame = pd.concat([\
#    gen_df(this_dat[2],this_dat[3],None,this_inds,'shuffle_'+region_name) \
#    for this_dat,this_inds,region_name in \
#    zip([out0,out1],pair_list,sorted_region_names)])

########################################
## Save arrays 
########################################

def save_xarray(array, save_path, data_dir):
    tag = "_".join([array.attrs['region_type'], 
                        array.attrs['comparison_type']])
    basename = os.path.basename(data_dir[:-1])
    fin_save_path = os.path.join(save_path, "_".join([basename,tag]))
    array.to_netcdf(
        path = fin_save_path, 
        mode = 'w', engine="h5netcdf")

for this_array in [*inter_outs, *intra_outs]:
    save_xarray(this_array, save_path, dat.data_dir)


#frame_name_list = ['inter_region_frame',
#                        'shuffle_inter_region_frame',
#                        'intra_region_frame',
#                        'shuffle_intra_region_frame',
#                        'bin_inter_region_frame',
#                        'shuffle_bin_inter_region_frame',
#                        'baseline_inter_region_frame',
#                        'baseline_shuffle_inter_region_frame',
#                        'baseline_intra_region_frame',
#                        'baseline_shuffle_intra_region_frame']

##region_names_dict = {'note' : 'Dict contains region order after sorting'\
##                             'for region with 

#with tables.open_file(dat.hdf5_path,'r+') as hf5:
#    for frame_name in frame_name_list:
#        # Will only remove if array already there
#        remove_node(os.path.join(save_path, frame_name),hf5, recursive=True)
#        remove_node(os.path.join(save_path, 'region_names'),hf5)
#        hf5.create_array(save_path,'region_names', 
#                [str(region_names_dict)])

#for frame_name in frame_name_list:
#    # Save transformed array to HDF5
#    eval(frame_name).to_hdf(dat.hdf5_path,  
#            os.path.join(save_path, frame_name))

