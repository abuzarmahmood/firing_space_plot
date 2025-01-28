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
# Setup
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
# Load Data
################################################### 

data_dir = sys.argv[1]
print(data_dir)
#data_dir = '/media/bigdata/Abuzar_Data/AM26/AM26_4Tastes_200829_100535'
# name_splits = os.path.basename(data_dir[:-1]).split('_')
# fin_name = name_splits[0]+'_'+name_splits[2]

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()
spikes = np.array(dat.spikes)

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/spike_noise_corrs'
frame_name_list = [     
                'inter_region_frame',
                'shuffle_inter_region_frame',
                'intra_region_frame',
                'shuffle_intra_region_frame'
                   ]#,

with tables.open_file(dat.hdf5_path,'r+') as hf5:
    if save_path not in hf5:
        hf5.create_group(os.path.dirname(save_path),os.path.basename(save_path),
                createparents = True)
    else:
        present_list = []
        for name in frame_name_list:
            if os.path.join(save_path,name) in hf5:
                present_list.append(name)
        if len(present_list) > 0:
            print('Data already present in HDF5')
            print("\n".join(present_list))
            cont = input('Do you want to continue? (y/n)')
            while cont not in ['y','n']:
                cont = input('Please enter y or n')
            if cont == 'n':
                sys.exit()


########################
# Corr
########################

# If array not present, then perform calculation
with tables.open_file(dat.hdf5_path,'r+') as hf5:
    if os.path.join(save_path,'inter_region_frame') not in hf5:
        present_bool = False 
    else:
        present_bool = True

#present_bool = False
if not present_bool: 

    ##################################################
    # Pre-processing
    ##################################################

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

    ##################################################
    ## Inter-Region Whole Trial
    ##################################################

    # Perform correlation over all pairs for each taste separately
    # Compare values to corresponding shuffles

    print('== Processing Inter-Region Whole Trial ==')
    out = list(zip(*parallelize(taste_calc_corrs,diff_sum_spikes,pair_inds)))
    out = [np.array(x) for x in out]

    inter_region_frame = gen_df(out[0], out[1], out[-1],
                                pair_inds,'inter_region')
    print(inter_region_frame)

    shuffle_inter_region_frame = gen_df(out[2], out[3], None,
                                pair_inds,'shuffle_inter_region')

    ###################################################
    ### BINNED Inter-Region 
    ###################################################
    ## Sama pair_inds as above, add additional loop for bins
    #bin_width = 500
    #bin_count = np.diff(time_lims)[0]//bin_width 
    #binned_spikes = [np.array(np.split(x,bin_count,axis=-1)) for x in temp_region_spikes]
    #binned_sum_spikes = [np.sum(x,axis=-1) for x in binned_spikes]
    ## Try detrending with 1st order difference before corr
    #diff_bin_sum_spikes = [np.diff(region,axis=2) for region in binned_sum_spikes]
    ## Zscore along trial axis to normalize values across neurons
    #diff_bin_sum_spikes = [stats.zscore(region,axis=2) for region in diff_bin_sum_spikes]
    #diff_bin_sum_spikes = [np.moveaxis(x,-1,1) for x in diff_bin_sum_spikes]

    ## Better way to handle this list-nesting???
    #out = [parallelize(taste_calc_corrs,this_bin,pair_inds) \
    #        for this_bin in zip(*diff_bin_sum_spikes)]
    #out= [list(zip(*x)) for x in out]
    #out= [[np.array(array) for array in this_bin] for this_bin in out]
    #out= list(zip(*out))
    #out= [np.array(x) for x in out]

    #bin_inter_region_frame = gen_bin_df(out[0],out[1],out[-1],
    #                                    pair_inds, 'inter_region')
    #shuffle_bin_inter_region_frame = gen_bin_df(out[2],out[3],None,
    #                                    pair_inds, 'shuffle_inter_region')

    ##################################################
    ## INTRA-Region Whole Trial
    ##################################################

    #this_save_path = os.path.join(save_path,'intra_region')

    print('== Processing Intra-Region Whole Trial ==')
    pair_list = [list(it.combinations(np.arange(x.shape[1]),2)) \
                            for x in diff_sum_spikes]

    out0 = list(zip(*parallelize(taste_calc_corrs,\
                        [diff_sum_spikes[0],diff_sum_spikes[0]],pair_list[0])))
    out1 = list(zip(*parallelize(taste_calc_corrs,\
                        [diff_sum_spikes[1],diff_sum_spikes[1]],pair_list[1])))

    out0 = [np.array(x) for x in out0]
    out1 = [np.array(x) for x in out1]

    intra_region_frame = pd.concat([\
        gen_df(this_dat[0],this_dat[1],this_dat[-1],this_inds,region_name) \
        for this_dat,this_inds,region_name in \
        zip([out0,out1],pair_list,sorted_region_names)])

    shuffle_intra_region_frame = pd.concat([\
        gen_df(this_dat[2],this_dat[3],None,this_inds,'shuffle_'+region_name) \
        for this_dat,this_inds,region_name in \
        zip([out0,out1],pair_list,sorted_region_names)])


    ###################################################
    ### BASELINE Whole Trial Inter-Region 
    ###################################################
    #time_lims = [0,2000]

    #base_temp_spikes = spikes[...,time_lims[0]:time_lims[1]]
    #base_region_spikes = [base_temp_spikes.swapaxes(0,2)[region_inds]\
    #        for region_inds in dat.region_units]
    #base_temp_region_spikes = [base_region_spikes[x] for x in wanted_order]
    #base_sum_spikes = [np.sum(x,axis=-1) for x in base_temp_region_spikes]
    ## Try detrending with 1st order difference before corr
    #base_diff_sum_spikes = [np.diff(region,axis=1) for region in base_sum_spikes]
    ## Zscore along trial axis to normalize values across neurons
    #base_diff_sum_spikes = [stats.zscore(region,axis=1) for region in base_diff_sum_spikes]
    #base_diff_sum_spikes = [np.moveaxis(x,-1,0) for x in base_diff_sum_spikes]

    #base_out = list(zip(*parallelize(taste_calc_corrs,base_diff_sum_spikes,pair_inds)))
    #base_out = [np.array(x) for x in base_out]

    #baseline_inter_region_frame = gen_df(base_out[0], base_out[1], base_out[-1],
    #                            pair_inds,'inter_region')

    #baseline_shuffle_inter_region_frame = gen_df(base_out[2], base_out[3], None,
    #                            pair_inds,'shuffle_inter_region')

    ###################################################
    ### BASELINE INTRA-Region 
    ###################################################

    ##this_save_path = os.path.join(save_path,'intra_region')

    #base_out0 = list(zip(*parallelize(taste_calc_corrs,\
    #                    [base_diff_sum_spikes[0],base_diff_sum_spikes[0]],pair_list[0])))
    #base_out1 = list(zip(*parallelize(taste_calc_corrs,\
    #                    [base_diff_sum_spikes[1],base_diff_sum_spikes[1]],pair_list[1])))

    #base_out0 = [np.array(x) for x in base_out0]
    #base_out1 = [np.array(x) for x in base_out1]

    #baseline_intra_region_frame = pd.concat([\
    #    gen_df(this_dat[0],this_dat[1],this_dat[-1],this_inds,region_name) \
    #    for this_dat,this_inds,region_name in \
    #    zip([base_out0,base_out1],pair_list,sorted_region_names)])

    #baseline_shuffle_intra_region_frame = pd.concat([\
    #    gen_df(this_dat[2],this_dat[3],None,this_inds,'shuffle_'+region_name) \
    #    for this_dat,this_inds,region_name in \
    #    zip([base_out0,base_out1],pair_list,sorted_region_names)])

    ########################################
    ## Save arrays 
    ########################################
                            #'bin_inter_region_frame',
                            #'shuffle_bin_inter_region_frame']#,
                            #'baseline_inter_region_frame',
                            #'baseline_shuffle_inter_region_frame',
                            #'baseline_intra_region_frame',
                            #'baseline_shuffle_intra_region_frame']

   #region_names_dict = {'note' : 'Dict contains region order after sorting'\
   #                             'for region with 

    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        for frame_name in frame_name_list:
            # Will only remove if array already there
            remove_node(os.path.join(save_path, frame_name),hf5, recursive=True)
            remove_node(os.path.join(save_path, 'region_names'),hf5)
            #hf5.create_array(save_path,'region_names', 
            #        [str(region_names_dict)])

    for frame_name in frame_name_list:
        # Save transformed array to HDF5
        eval(frame_name).to_hdf(dat.hdf5_path,  
                os.path.join(save_path, frame_name))

