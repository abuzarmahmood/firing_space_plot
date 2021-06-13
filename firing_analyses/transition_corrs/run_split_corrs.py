"""
Given a recording session with single region split changepoints 
    1) Find matching splits 
    2) Perform actual data and shuffle correlations 
"""

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
        mode_tau = stats.mode(tau_samples,axis=0)[0][0]
        int_mode_tau = np.vectorize(int)(mode_tau)
        # Remove pickled data to conserve memory
        del data
    return int_mode_tau

def parallelize_shuffles(func, args, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(*args) for this_iter in tqdm(iterator))

def calc_mse(x,y):
    return np.mean(np.abs(x-y))

def gen_shuffle(func, x, y):
    return func(np.random.permutation(x),y)

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/split_region'

# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/single_region_split_frame.pkl'
split_frame = pd.read_pickle(data_dir_pkl)

#data_dir = split_frame.path.iloc[0]
for data_dir in split_frame.path:

    dat = ephys_data(data_dir)
    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        if save_path not in hf5:
            hf5.create_group(os.path.dirname(save_path),
                    os.path.basename(save_path),
                    createparents = True)

    ## Find and load split pickle files
    this_info = check_data(data_dir)
    this_info.run_all()
    split_paths = [path for  num,path in enumerate(this_info.pkl_file_paths) \
                    if num in this_info.split_inds]
    state4_models = [path for path in split_paths if '4state' in path]
    split_basenames = [os.path.basename(x) for x in state4_models]

    # Match splits
    # Extract split nums
    split_basenames_split = [x.split('_') for x in split_basenames]
    split_nums = np.vectorize(int)\
            (np.array([[x[1],x[2]] for x in split_basenames_split]))
    split_inds = np.array([np.where(split_nums[:,1] == num)[0] \
            for num in range(split_nums[:,1].max())])


    for this_split in range(len(split_inds)):

        terminal_dir = f'split{this_split}'
        fin_save_path = os.path.join(save_path,terminal_dir)

        with tables.open_file(dat.hdf5_path,'r+') as hf5:
            # Will only remove if array already there
            remove_node(fin_save_path, hf5, recursive=True)
            hf5.create_group(save_path, terminal_dir, createparents = True)

        model_paths = [split_paths[i] for i in split_inds[this_split]]

        tau_list = [load_mode_tau(this_path).T for this_path in model_paths]
        # Calculate spearman rho
        tau_corrs = [stats.spearmanr(x,y)[0] for x,y in zip(*tau_list)] 
        # Calculate MSE
        tau_mse = [np.mean(np.abs(x-y)) for x,y in zip(*tau_list)]

        # Calculate shuffles
        shuffle_num = 10000
        rho_shuffles = []
        mse_shuffles = []

        for num,(x,y) in enumerate(zip(*tau_list)):
            rho_out = parallelize_shuffles(\
                    gen_shuffle, (stats.spearmanr,x, y),
                            np.arange(shuffle_num))
            mse_out = parallelize_shuffles(\
                    gen_shuffle, (calc_mse,x, y),
                            np.arange(shuffle_num))
            rho_list = [x[0] for x in rho_out]
            mse_list = [x for x in mse_out]
            rho_shuffles.append(rho_list)
            mse_shuffles.append(mse_list)

        rho_percentiles = [stats.percentileofscore(shuffle, val) \
                        for shuffle,val in zip(rho_shuffles, tau_corrs)]
        mse_percentiles = [stats.percentileofscore(shuffle, val) \
                        for shuffle,val in zip(mse_shuffles, tau_mse)]

        save_names = ['tau_list','tau_corrs','tau_mse',
                'rho_shuffles','mse_shuffles',
                'rho_percentiles','mse_percentiles']

        with tables.open_file(dat.hdf5_path,'r+') as hf5:
            for this_obj_name in save_names:
                hf5.create_array(fin_save_path, 
                        this_obj_name, globals()[this_obj_name])
