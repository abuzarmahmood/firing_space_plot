"""
Given a recording session with inter region changepoint fits
    1) Find matching fits
    2) Perform actual data and shuffle correlations 
"""

########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

import numpy as np
import json
from glob import glob
import os
import pandas as pd
import pickle 
import sys
from scipy import stats
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm 
import tables

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs')
from check_data import check_data 

def load_mode_tau(model_path):
    if os.path.exists(model_path):
        print('Trace loaded from cache')
        with open(model_path, 'rb') as buff:
            data = pickle.load(buff)
        tau_samples = data['tau']
        #int_tau = np.vectorize(int)(tau_samples)
        # Convert to int first, then take mode
        #int_mode_tau = stats.mode(int_tau,axis=0)[0][0]
        # Remove pickled data to conserve memory
        del data
    return tau_samples#, int_mode_tau

def parallelize_shuffles(func, args, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(*args) for this_iter in tqdm(iterator))

def calc_mse(x,y):
    return np.mean(np.abs(x-y))

def gen_shuffle(func, x, y):
    return func(np.random.permutation(x),y)

def moving_gen_shuffle(func,shuffle_num, x,y):
    return [func(np.random.permutation(x),y) for i in range(shuffle_num)]

def moving_parallelize_shuffles(func, args, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(*args, *this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/inter_region'

##################################################
#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           
##################################################

# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/multi_region_frame.pkl'
inter_frame = pd.read_pickle(data_dir_pkl)

#data_dir = inter_frame.path.iloc[0]
for data_dir in tqdm(inter_frame.path):

    dat = ephys_data(data_dir)
    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        if save_path not in hf5:
            hf5.create_group(os.path.dirname(save_path),
                    os.path.basename(save_path),
                    createparents = True)

    this_info = check_data(data_dir)
    this_info.run_all()
    inter_region_paths = [path for  num,path in enumerate(this_info.pkl_file_paths) \
                    if num in this_info.region_fit_inds]
    state4_models = [path for path in inter_region_paths if '4state' in path]
    split_basenames = [os.path.basename(x) for x in state4_models]

    full_tau = np.array([load_mode_tau(this_path).T for this_path in inter_region_paths])
    tau_vars = np.var(full_tau, axis = -1)
    # Use summed variance as a rank metric
    metric_tau_vars = np.sum(tau_vars, axis = 0)

    int_tau = np.vectorize(np.int)(full_tau)
    mode_tau = np.squeeze(stats.mode(int_tau, axis=3)[0])
    inds = list(np.ndindex(mode_tau.shape[:-1]))
    # Calculate spearman rho
    tau_corrs = [stats.spearmanr(x,y)[0] for x,y in zip(*mode_tau)] 
    # Calculate MSE
    tau_mse = [calc_mse(x,y) for x,y in zip(*mode_tau)]

    trial_ranks = np.argsort(metric_tau_vars,axis=-1) 
    sorted_mode_tau = np.array([mode_tau[:,num, this_ranks] \
            for num, this_ranks in enumerate(trial_ranks)])
    sorted_mode_tau = np.swapaxes(sorted_mode_tau, 0,1)

    # Calculate corrs and MSE for transitions sorted by their collective
    # variances (uncertainty)

    def moving_window_corrs(x,y, window_size, step_size, shuffle_num = 1000 ):
        window_count = (len(x) -  window_size)//step_size + 1
        window_inds = [np.array((0,window_size-1)) + x*step_size \
                for x in range(window_count)]
        xy_gen = [(x[this_inds[0]:this_inds[1]],
                                    y[this_inds[0]:this_inds[1]])\
                                for this_inds in window_inds]
        corrs_raw = [stats.spearmanr(x,y) for x,y in xy_gen] 
        #rho_out = moving_parallelize_shuffles(\
        #                func =  moving_gen_shuffle, 
        #                args = (stats.spearmanr, 1000),
        #                iterator = xy_gen)
        #rho_out = np.array(rho_out)[...,0]
        max_t = [x[1] for x in window_inds]
        rhos, ps = zip(*corrs_raw)
        #rho_percentiles = [stats.percentileofscore(shuffle, val) \
        #                for shuffle,val in zip(rho_out, rhos)]
        return rhos, ps, max_t#, rho_percentiles

    moving_corrs = [moving_window_corrs(x,y, 30,5, 1e4) \
            for x,y in zip(*sorted_mode_tau)]

    moving_rhos, moving_ps, max_t = zip(*moving_corrs)
    moving_rhos = np.array(moving_rhos)
    moving_ps = np.array(moving_ps)
    #moving_rho_percentiles = np.array(moving_rho_percentiles)/100
    #moving_rho_percentiles = 0.5 - np.abs(moving_rho_percentiles - 0.5)
    moving_t = max_t[0]

    # Calculate shuffles
    shuffle_num = 10000
    rho_shuffles = []
    mse_shuffles = []

    for num,(x,y) in enumerate(zip(*mode_tau)):

        rho_out = parallelize_shuffles(\
                        func =  gen_shuffle, 
                        args = (stats.spearmanr,x, y),
                        iterator = np.arange(shuffle_num))

        mse_out = parallelize_shuffles(\
                        func = gen_shuffle, 
                        args = (calc_mse,x, y),
                        iterator = np.arange(shuffle_num))

        rho_list = [x[0] for x in rho_out]
        mse_list = [x for x in mse_out]
        rho_shuffles.append(rho_list)
        mse_shuffles.append(mse_list)

    rho_percentiles = [stats.percentileofscore(shuffle, val) \
                    for shuffle,val in zip(rho_shuffles, tau_corrs)]
    mse_percentiles = [stats.percentileofscore(shuffle, val) \
                    for shuffle,val in zip(mse_shuffles, tau_mse)]

    save_names = ['mode_tau','tau_corrs','tau_mse',
            'rho_shuffles','mse_shuffles',
            'rho_percentiles','mse_percentiles',
            'moving_rhos','moving_ps','moving_t']

    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        for this_obj_name in save_names:
            remove_node(os.path.join(save_path, this_obj_name), hf5)
            hf5.create_array(save_path, 
                    this_obj_name, globals()[this_obj_name])
