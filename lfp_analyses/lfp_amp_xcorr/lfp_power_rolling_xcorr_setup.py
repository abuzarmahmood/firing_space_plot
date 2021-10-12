"""
Cross correlations in LFP amplitudes between
brain regions

** NOTE ** : Instead of performing xcorr on entire trials,
        perform it on rolling windows to get time-wise coordination strength

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
import xarray

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
#import visualize

########################################
## Define Functions
########################################


def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def parallelize_args(func, iterator, args):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(args, this_iter) for this_iter in tqdm(iterator))

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

#@jit(nopython=True)
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

def rolling_norm_zero_lag_xcorr(vec1, vec2, window_size=250, step_size = 25):
    """
    Calculates normalized zero-lag cross correlation
    Returns a single number
    """
    starts = np.arange(0,(vec1.shape[-1] - window_size) + step_size,step_size)
    out = np.zeros((*vec1.shape[:-1],len(starts)))
    inds = list(zip(starts,starts+window_size))
    for num,this_ind in enumerate(tqdm(inds)):
        out[...,num] += \
                norm_zero_lag_xcorr(
                        vec1[...,this_ind[0]:this_ind[1]],
                        vec2[...,this_ind[0]:this_ind[1]])
    return out


##############################
## HDF5 I/O
##############################

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)
#
#def gen_df(array, label, time_vec):
#    inds = np.array(list(np.ndindex(array.shape)))
#    return pd.DataFrame({
#            'label' : [label] * inds.shape[0],
#            'pair' : inds[:,0],
#            'taste' : inds[:,1],
#            'trial' : inds[:,2],
#            'freq' : dat.freq_vec[inds[:,3]],
#            'time_bin' : inds[:,3],
#            'time' : time_vec[inds[:3]],
#            'xcorr' : array.flatten()})


################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

save_dir = '/media/bigdata/firing_space_plot/lfp_analyses/lfp_amp_xcorr/data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM12/AM12_4Tastes_191106_085215/'
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
window_size = 500
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
    dat.get_stft(recalculate=True)

    # Only pull STFT if transformation needs to be calculated
    amplitude_array = dat.amplitude_array.swapaxes(0,1)
    transformed_array = np.array(\
            parallelize(lambda x: rolling_zscore(x, window_size), amplitude_array))

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
#time_lims = [0,4000]
#transformed_array = transformed_array[...,time_lims[0]:time_lims[1]]

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

recalculate_xcorr = True
# If not there, or recalculate flag True, then calculate

#with tables.open_file(dat.hdf5_path,'r+') as hf5:
#    if os.path.join(save_path, 'inter_region_frame') not in hf5:
#        present_bool = False 
#    else:
#        present_bool = True

#if (not present_bool) or recalculate_xcorr:
if recalculate_xcorr:
    
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

    window_size = 250
    step_size =25
    t_vec = np.arange(0,(split_amplitude_list[0].shape[-1] - window_size) \
            + step_size,step_size)

    def par_rolling_inter_xcorr(amp_list, ind):
        return rolling_norm_zero_lag_xcorr(amp_list[0][ind[0]],
                                        amp_list[1][ind[1]],
                                        window_size, step_size)

    def par_rolling_shuffle_inter_xcorr(amp_list, ind):
        shuffle_inds = np.random.permutation(np.arange(amp_list[0].shape[2]))
        return rolling_norm_zero_lag_xcorr(
                                amp_list[0][ind[0]][:,shuffle_inds],
                                        amp_list[1][ind[1]],
                                        window_size, step_size)

    inter_outs = parallelize_args(
            par_rolling_inter_xcorr, pair_inds, temp_split_amp_list)
    shuffle_inter_outs = parallelize_args(
            par_rolling_shuffle_inter_xcorr, 
                                    pair_inds, temp_split_amp_list)

    inter_outs = np.array(inter_outs)
    shuffle_inter_outs = np.array(shuffle_inter_outs)

    inter_outs_x = xarray.DataArray(
            data = inter_outs[np.newaxis, np.newaxis],
            dims = ['region_type','order_type','pairs','tastes','trials','freqs','bins'],
            coords = dict(region_type = ['inter'], order_type = ['actual'],
                                    bins = t_vec, freqs = dat.freq_vec),
            attrs = dict(window_size = window_size, step_size = step_size))

    shuffle_inter_outs_x = xarray.DataArray(
            data = shuffle_inter_outs[np.newaxis, np.newaxis],
            dims = ['region_type','order_type','pairs','tastes','trials','freqs','bins'],
            coords = dict(region_type = ['inter'], order_type = ['shuffle'],
                                    bins = t_vec, freqs = dat.freq_vec),
            attrs = dict(window_size = window_size, step_size = step_size))

    fin_inter = xarray.concat([inter_outs_x, shuffle_inter_outs_x],
                                    dim = 'order_type')

    #inter_region_xcorr = np.array([\
    #        rolling_norm_zero_lag_xcorr(temp_split_amp_list[0][ind[0]],
    #                            temp_split_amp_list[1][ind[1]]) \
    #                    for ind in pair_inds])

    #shuffled_inter_region_xcorr = np.array([\
    #                norm_zero_lag_xcorr(\
    #                    temp_split_amp_list[0][ind[0]]\
    #                        [:,np.random.permutation(\
    #                                    np.arange(temp_split_amp_list[0].shape[2]))],
    #                    temp_split_amp_list[1][ind[1]]) \
    #                for ind in pair_inds])

    ########################################
    ## Within Region
    ########################################
    # Perform xcorr between all pairs of channels within a region
    pair_list = [list(it.combinations(np.arange(x.shape[0]),2)) \
                            for x in split_amplitude_list]

    def par_rolling_intra_xcorr(amp_list, ind):
        return rolling_norm_zero_lag_xcorr(amp_list[ind[0]],
                                        amp_list[ind[1]],
                                        window_size, step_size)

    def par_rolling_shuffle_intra_xcorr(amp_list, ind):
        shuffle_inds = np.random.permutation(np.arange(amp_list.shape[2]))
        return rolling_norm_zero_lag_xcorr(
                                amp_list[ind[0]][:,shuffle_inds],
                                        amp_list[ind[1]],
                                        window_size, step_size)
    intra_outs = \
            [parallelize_args(par_rolling_intra_xcorr, this_pairs, this_region)\
            for this_pairs, this_region in zip(pair_list, split_amplitude_list)]
    intra_outs = np.array(intra_outs)

    shuffle_intra_outs = \
            [parallelize_args(par_rolling_shuffle_intra_xcorr, this_pairs, this_region)\
            for this_pairs, this_region in zip(pair_list, split_amplitude_list)]

    intra_outs = [np.array(x) for x in intra_outs]
    shuffle_intra_outs = [np.array(x) for x in shuffle_intra_outs]

    intra_frame_list = []
    shuffle_intra_frame_list = []
    for num, name in enumerate(dat.region_names):
        intra_frame_list.append(
                xarray.DataArray(
                data = intra_outs[num][np.newaxis, np.newaxis],
                dims = ['region_type','order_type','pairs','tastes','trials','freqs','bins'],
                coords = dict(region_type = [name], order_type = ['actual'],
                                        bins = t_vec, freqs = dat.freq_vec),
                attrs = dict(window_size = window_size, step_size = step_size)))

        shuffle_intra_frame_list.append(
                xarray.DataArray(
                data = shuffle_intra_outs[num][np.newaxis, np.newaxis],
                dims = ['region_type','order_type','pairs','tastes','trials','freqs','bins'],
                coords = dict(region_type = [name], order_type = ['shuffle'],
                                        bins = t_vec, freqs = dat.freq_vec),
                attrs = dict(window_size = window_size, step_size = step_size)))

    #intra_region_xcorr_list = [np.array([\
    #        norm_zero_lag_xcorr(region[ind[0]],region[ind[1]]) \
    #        for ind in this_pair_list]) \
    #        for this_pair_list, region in zip(pair_list, split_amplitude_list)]
    #shuffled_intra_region_xcorr_list = [np.array([\
    #        norm_zero_lag_xcorr(\
    #               region[ind[0]]
    #                    [:,np.random.permutation(\
    #                                np.arange(temp_split_amp_list[0].shape[2]))],
    #                region[ind[1]]) \
    #        for ind in this_pair_list]) \
    #        for this_pair_list, region in zip(pair_list, split_amplitude_list)]


    ########################################
    ## Save arrays 
    ########################################
    # HDF5 can't save list of different sized arrays
    # Save as Pandas DataFrame for consistency across all calculations
    # Save raw xcorr 
    # Means can be calculated when doing further preocessing

    #rolling_inter_region_frame = pd.concat([\
    #        gen_df(inter_outs,'inter_region', t_vec),
    #        gen_df(shuffled_inter_region_xcorr,'shuffled_inter_region')])

    fin_inter.to_netcdf(
            path = os.path.join(
                save_dir, os.path.basename(dat.data_dir[:-1]) + "_" +\
                fin_inter.region_type.values[0]), 
            mode = 'w', engine="h5netcdf")

    fin_intra_frames = [xarray.concat([x,y], dim = 'order_type') \
            for x,y in zip(intra_frame_list, shuffle_intra_frame_list)]

    for this_frame in fin_intra_frames:
        this_frame.to_netcdf(
                path = os.path.join(
                    save_dir, os.path.basename(dat.data_dir[:-1]) + "_" + \
                    this_frame.region_type.values[0]), 
                mode = 'w', engine="h5netcdf")


    #rolling_intra_region_frame = pd.concat(
    #        [gen_df(x,'intra_'+region_name) for x,region_name in \
    #                zip(intra_region_xcorr_list, dat.region_names)] + \
    #        [gen_df(x,'shuffled_intra_'+region_name) for x,region_name \
    #                in zip(shuffled_intra_region_xcorr_list, dat.region_names)])

    #frame_name_list = ['rolling_inter_region_frame', 'rolling_intra_region_frame']

    #with tables.open_file(dat.hdf5_path,'r+') as hf5:
    #    for frame_name in frame_name_list:
    #        # Will only remove if array already there
    #        remove_node(os.path.join(save_path, frame_name),hf5, recursive=True)

    #for frame_name in frame_name_list:
    #    # Save transformed array to HDF5
    #    eval(frame_name).to_hdf(dat.hdf5_path,  
    #            os.path.join(save_path, frame_name))
