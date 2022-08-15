"""
Cross correlations in LFP amplitudes between
brain regions


1) Load data
    - Any transformations should be done here
        so that future analyses are agnostic to
        what data is fed
    - e.g. rolling window z-scoring

** All analyses below will be served with corresponding
    shuffles

2) XCorr b/w representative electrodes from both regions
3) XCorr b/w all pairs of electrodes within the same
    region to contextualize what inter-region xcorr represents
4) Inter-region XCorr within time windows
5) Taste-specific inter-region XCorr
    - Both binned and whole trial
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
#import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
from numba import jit
import ast

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
#import visualize

########################################
## Define Functions
########################################


def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def rolling_zscore(array, window_size):
    """
    Performs rolling z-score on last dimension of given array
    """
    out = np.zeros(array.shape)
    starts = np.arange((array.shape[-1] - window_size))
    inds = list(zip(starts,starts+window_size))
    for this_ind in tqdm(inds):
        out[...,this_ind[0]:this_ind[1]] += \
                stats.zscore(array[...,this_ind[0]:this_ind[1]],axis=-1)
    return out/window_size

@jit(nopython=True)
def norm_zero_lag_xcorr(vec1, vec2):
    """
    Calculates normalized zero-lag cross correlation
    Returns a single number
    """
    auto_v1 = np.sum(vec1**2,axis=-1)
    auto_v2 = np.sum(vec2**2,axis=-1)
    xcorr = np.sum(vec1 * vec2,axis=-1)
    denom = np.sqrt(np.multiply(auto_v1,auto_v2))
    return np.divide(xcorr, denom)

##############################
## HDF5 I/O
##############################

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

def gen_df(array, label):
    inds = np.array(list(np.ndindex(array.shape)))
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'pair' : inds[:,0],
            'taste' : inds[:,1],
            'trial' : inds[:,2],
            'freq' : dat.freq_vec[inds[:,-1]],
            'xcorr' : array.flatten()})

def gen_df_bin(array, label):
    inds = np.array(list(np.ndindex(array.shape)))
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'bin' : inds[:,0],
            'pair' : inds[:,1],
            'taste' : inds[:,2],
            'trial' : inds[:,3],
            'freq' : dat.freq_vec[inds[:,-1]],
            'xcorr' : array.flatten()})

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
data_dir = sys.argv[1]
dat = ephys_data(data_dir)
dat.get_lfp_electrodes()

#median_amplitude = np.median(dat.amplitude_array,axis=(0,2))
#visualize.firing_overview(stats.zscore(median_amplitude,axis=-1));plt.show()


##################################################
#|_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___  
#  | || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ 
#  | || | | (_| | | | \__ \  _| (_) | |  | | | | | |
#  |_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|
##################################################

# If we take rolling z-zscore at this stage, then we can likely
# eliminate the impact of many artifacts
# A "transformation" can also be removing trials with artifacts
# from the raw data
#transformed_long = min_err_long # Trivial transformation

# Perform rolling window on WHOLE trial so that edges can
# be choppoed away later
recalculate_transform = True
save_path = '/stft/analyses/amplitude_xcorr'
# This will need to be parallelilzed
with tables.open_file(dat.hdf5_path,'r+') as hf5:
    #============================== 
    if save_path not in hf5:
        hf5.create_group(os.path.dirname(save_path),os.path.basename(save_path),
                createparents = True)

    #============================== 
    if os.path.join(save_path, 'transformed_amplitude_array') not in hf5 \
                            or recalculate_transform:
        perform_transormation_bool = True

    else:
        perform_transormation_bool = False
        transformed_array = hf5.get_node(save_path,'transformed_amplitude_array')[:] 
        transformed_array_params = ast.literal_eval(\
                    hf5.get_node(save_path,'transformed_amplitude_info')[:][0]\
                    .decode('utf-8'))

# This decision branch is made to avoid unnecessarily loading STFT data
# which are usually quite large
if perform_transormation_bool:
    dat.get_stft()
    window_size_ms = 500
    window_size = int(500/(np.mean(np.diff(dat.time_vec))*1000))

    # Only pull STFT if transformation needs to be calculated
    amplitude_array = dat.amplitude_array.swapaxes(0,1)
    transformed_array = np.array(\
            parallelize(lambda x: rolling_zscore(x, window_size), amplitude_array))
    #transformed_array = amplitude_array

    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        #============================== 
        # Will only remove if array already there
        remove_node('/stft/analyses/amplitude_xcorr/transformed_amplitude_array',hf5)
        remove_node('/stft/analyses/amplitude_xcorr/transformed_amplitude_info',hf5)
        # Save transformed array to HDF5
        hf5.create_array(save_path,'transformed_amplitude_array',transformed_array)
        transform_array_params = {
                'transformation_type' : 'rolling_zscore',
                'transformation_params' : {'window_size' : 500},
                'notes' : ''}
        hf5.create_array(save_path,'transformed_amplitude_info', 
                [str(transform_array_params)])

        #============================== 
        del amplitude_array

# Chop to relevant time period
baseline_time_lims = [0,2000]
t_vec = dat.time_vec.copy()*1000
baseline_lims = [np.argmin(np.abs(t_vec-x)) for x in baseline_time_lims] 
base_transformed_array = transformed_array[...,baseline_lims[0]:baseline_lims[1]]

# Chop to relevant time period
time_time_lims = [2000,4000]
time_lims = [np.argmin(np.abs(t_vec-x)) for x in time_time_lims] 
transformed_array = transformed_array[...,time_lims[0]:time_lims[1]]

# We can use ALL trials for rolling zscored power rather than
# arbitrarily removing trials due to artifacts

##############################
#__  ______                
#\ \/ / ___|___  _ __ _ __ 
# \  / |   / _ \| '__| '__|
# /  \ |__| (_) | |  | |   
#/_/\_\____\___/|_|  |_|   
#                          
##############################
# Perform zero-lag cross-correlation on single trials and shuffled trials
# Use normalized cross-correlation to remove amplitude effects

split_amplitude_list = \
        [transformed_array[region] for region in dat.lfp_region_electrodes]
base_split_amplitude_list = \
        [base_transformed_array[region] for region in dat.lfp_region_electrodes]

recalculate_xcorr = True
# If not there, or recalculate flag True, then calculate

with tables.open_file(dat.hdf5_path,'r+') as hf5:
    if os.path.join(save_path, 'inter_region_frame') not in hf5:
        present_bool = False 
    else:
        present_bool = True

if (not present_bool) or recalculate_xcorr:
    
    # To dissociate from calculation of transformed_array above
    if 'freq_vec' not in dir(dat):
        dat.get_stft()

    ########################################
    ## Inter-Region
    ########################################

    # Mean XCorr between all pairs of channels from both regions
    chan_count = [len(x) for x in split_amplitude_list]
    wanted_order = np.argsort(chan_count)[::-1]
    temp_split_amp_list = [split_amplitude_list[x] for x in wanted_order]
    chan_count = [len(x) for x in temp_split_amp_list]
    all_pairs = np.arange(1,1+chan_count[0])[:,np.newaxis].\
            dot(np.arange(1,1+chan_count[1])[np.newaxis,:])
    pair_inds = list(zip(*np.where(all_pairs)))

    inter_region_xcorr = np.array([\
            norm_zero_lag_xcorr(temp_split_amp_list[0][ind[0]],
                                temp_split_amp_list[1][ind[1]]) \
                        for ind in pair_inds])
    shuffled_inter_region_xcorr = np.array([\
                    norm_zero_lag_xcorr(\
                        temp_split_amp_list[0][ind[0]]\
                            [:,np.random.permutation(\
                                        np.arange(temp_split_amp_list[0].shape[2]))],
                        temp_split_amp_list[1][ind[1]]) \
                    for ind in pair_inds])

    #########################################
    ### Binned Inter-Region
    #########################################
    #window_size = 250
    #splits = np.abs(np.diff(time_lims)[0])//window_size
    #binned_amp = [np.split(x,splits,axis=-1) for x in temp_split_amp_list]

    #binned_inter_region_xcorr = np.array([[\
    #        norm_zero_lag_xcorr(this_bin[0][ind[0]],
    #                            this_bin[1][ind[1]]) \
    #                    for ind in pair_inds]\
    #                    for this_bin in zip(*binned_amp)])
    #shuffled_binned_inter_region_xcorr = np.array([[\
    #        norm_zero_lag_xcorr(this_bin[0][ind[0]]\
    #                        [:,np.random.permutation(\
    #                                    np.arange(temp_split_amp_list[0].shape[2]))],
    #                            this_bin[1][ind[1]]) \
    #                    for ind in pair_inds]\
    #                    for this_bin in zip(*binned_amp)])


    ########################################
    ## Within Region
    ########################################
    # Perform xcorr between all pairs of channels within a region
    pair_list = [list(it.combinations(np.arange(x.shape[0]),2)) \
                            for x in split_amplitude_list]

    intra_region_xcorr_list = [np.array([\
            norm_zero_lag_xcorr(region[ind[0]],region[ind[1]]) \
            for ind in this_pair_list]) \
            for this_pair_list, region in zip(pair_list, split_amplitude_list)]
    shuffled_intra_region_xcorr_list = [np.array([\
            norm_zero_lag_xcorr(\
                   region[ind[0]]
                        [:,np.random.permutation(\
                                    np.arange(temp_split_amp_list[0].shape[2]))],
                    region[ind[1]]) \
            for ind in this_pair_list]) \
            for this_pair_list, region in zip(pair_list, split_amplitude_list)]

    #########################################
    ### Binned Within Region
    #########################################
    ## I wonder if these would be more readable as for-loops :p
    ## Remake binned_amp so it's not using the region sorted temp_split_amp_list
    #binned_amp = [np.split(x,splits,axis=-1) for x in split_amplitude_list]
    #binned_intra_region_xcorr_list = [ np.array([[\
    #        norm_zero_lag_xcorr(this_bin[ind[0]],
    #                            this_bin[ind[1]]) \
    #                    for ind in this_pair_list]\
    #                    for this_bin in this_region]) \
    #                    for this_pair_list, this_region in zip(pair_list,binned_amp)]

    #shuffled_binned_intra_region_xcorr_list = [np.array([[\
    #        norm_zero_lag_xcorr(this_bin[ind[0]]\
    #                        [:,np.random.permutation(\
    #                                    np.arange(split_amplitude_list[0].shape[2]))],
    #                            this_bin[ind[1]]) \
    #                    for ind in this_pair_list]\
    #                    for this_bin in this_region]) \
    #                    for this_pair_list, this_region in zip(pair_list,binned_amp)]

    #########################################
    ### BASELINE Inter-Region 
    #########################################
    ## Mean XCorr between all pairs of channels from both regions

    #base_temp_split_amp_list = [base_split_amplitude_list[x] for x in wanted_order]

    #base_inter_region_xcorr = np.array([\
    #        norm_zero_lag_xcorr(base_temp_split_amp_list[0][ind[0]],
    #                            base_temp_split_amp_list[1][ind[1]]) \
    #                    for ind in pair_inds])

    #base_shuffled_inter_region_xcorr = np.array([\
    #                norm_zero_lag_xcorr(\
    #                    base_temp_split_amp_list[0][ind[0]]\
    #                        [:,np.random.permutation(\
    #                                    np.arange(base_temp_split_amp_list[0].shape[2]))],
    #                    base_temp_split_amp_list[1][ind[1]]) \
    #                for ind in pair_inds])

    #########################################
    ### BASELINE INTRA(WITHIN)-Region 
    #########################################
    ## Perform xcorr between all pairs of channels within a region

    #base_intra_region_xcorr_list = [np.array([\
    #        norm_zero_lag_xcorr(region[ind[0]],region[ind[1]]) \
    #        for ind in this_pair_list]) \
    #        for this_pair_list, region in zip(pair_list, base_split_amplitude_list)]

    #base_shuffled_intra_region_xcorr_list = [np.array([\
    #        norm_zero_lag_xcorr(\
    #               region[ind[0]]
    #                    [:,np.random.permutation(\
    #                                np.arange(base_temp_split_amp_list[0].shape[2]))],
    #                region[ind[1]]) \
    #        for ind in this_pair_list]) \
    #        for this_pair_list, region in zip(pair_list, base_split_amplitude_list)]

    ########################################
    ## Save arrays 
    ########################################
    # HDF5 can't save list of different sized arrays
    # Save as Pandas DataFrame for consistency across all calculations
    # Save raw xcorr 
    # Means can be calculated when doing further preocessing

    inter_region_frame = pd.concat([\
            gen_df(inter_region_xcorr,'inter_region'),
            gen_df(shuffled_inter_region_xcorr,'shuffled_inter_region')])

    #base_inter_region_frame = pd.concat([\
    #        gen_df(base_inter_region_xcorr,'base_inter_region'),
    #        gen_df(base_shuffled_inter_region_xcorr,'base_shuffled_inter_region')])

    #binned_inter_region_frame = pd.concat([\
    #        gen_df_bin(binned_inter_region_xcorr,'binned_inter_region'),
    #        gen_df_bin(shuffled_binned_inter_region_xcorr,
    #                        'shuffled_binned_inter_region')])

    intra_region_frame = pd.concat(
            [gen_df(x,'intra_'+region_name) for x,region_name in \
                    zip(intra_region_xcorr_list, dat.region_names)] + \
            [gen_df(x,'shuffled_intra_'+region_name) for x,region_name \
                    in zip(shuffled_intra_region_xcorr_list, dat.region_names)])

    #base_intra_region_frame = pd.concat(
    #        [gen_df(x,'base_intra_'+region_name) for x,region_name in \
    #                zip(base_intra_region_xcorr_list, dat.region_names)] + \
    #        [gen_df(x,'base_shuffled_intra_'+region_name) for x,region_name \
    #                in zip(base_shuffled_intra_region_xcorr_list, dat.region_names)])

    #binned_intra_region_frame = pd.concat(
    #        [gen_df_bin(x,'intra_'+region_name) for x,region_name in \
    #            zip(binned_intra_region_xcorr_list, dat.region_names)] + \
    #        [gen_df_bin(x,'shuffled_intra_'+region_name) for x,region_name \
    #            in zip(shuffled_binned_intra_region_xcorr_list, dat.region_names)])

    frame_name_list = ['inter_region_frame',
                            #'binned_inter_region_frame',
                            'intra_region_frame']#,
                            #'binned_intra_region_frame',
                            #'base_inter_region_frame',
                            #'base_intra_region_frame']

    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        for frame_name in frame_name_list:
            # Will only remove if array already there
            remove_node(os.path.join(save_path, frame_name),hf5, recursive=True)

    for frame_name in frame_name_list:
        # Save transformed array to HDF5
        eval(frame_name).to_hdf(dat.hdf5_path,  
                os.path.join(save_path, frame_name))
