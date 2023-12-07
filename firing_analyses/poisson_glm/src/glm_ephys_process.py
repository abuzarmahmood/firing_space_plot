"""
First pass at applying glm framework to ephys data
"""

############################################################
## Imports
############################################################

import numpy as np
import pandas as pd
import sys
#sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm'
#base_path = '/home/exouser/Desktop/ABU/firing_space_plot/firing_analyses/poisson_glm'
sys.path.append(base_path)
from utils.glm_ephys_process_utils import (
        gen_spike_train,
        process_ind,
        try_process,
        )
import glm_tools as gt
from pandas import DataFrame as df
from pandas import concat
import os
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed, cpu_count
from glob import glob
import json
from pprint import pprint
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6

#base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm'
save_path = os.path.join(base_path,'artifacts')

############################################################
## Begin Process 
############################################################

# Check if previous runs present
run_list = sorted(glob(os.path.join(save_path, 'run*')))
run_basenames = sorted([os.path.basename(x) for x in run_list])
print(f'Present runs : {run_basenames}')
input_run_ind = int(input('Please specify current run (integer) :'))
fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03}')

if not os.path.exists(fin_save_path):
    os.makedirs(fin_save_path)
    run_exists_bool = False
else:
    run_exists_bool = True
    json_path = os.path.join(fin_save_path, 'fit_params.json')
    params_dict = json.load(open(json_path))
    print('Run exists with following parameters :')
    pprint(params_dict)

############################################################
if run_exists_bool:
    hist_filter_len = params_dict['hist_filter_len']
    stim_filter_len = params_dict['stim_filter_len']
    coupling_filter_len = params_dict['coupling_filter_len']

    trial_start_offset = params_dict['trial_start_offset']
    trial_lims = np.array(params_dict['trial_lims'])
    stim_t = params_dict['stim_t']

    bin_width = params_dict['bin_width']

    # Reprocess filter lens
    hist_filter_len_bin = params_dict['hist_filter_len_bin'] 
    stim_filter_len_bin = params_dict['stim_filter_len_bin']
    coupling_filter_len_bin = params_dict['coupling_filter_len_bin']

    # Define basis kwargs
    basis_kwargs = params_dict['basis_kwargs'] 

    # Number of fits on actual data (expensive)
    n_fits = params_dict['n_fits']
    n_max_tries = params_dict['n_max_tries']
    n_shuffles_per_fit = params_dict['n_shuffles_per_fit']

else:
    # Parameters
    hist_filter_len = 75 # ms
    stim_filter_len = 300 # ms
    coupling_filter_len = 75 # ms

    bin_width = 2 # ms
    # Reprocess filter lens
    hist_filter_len_bin = hist_filter_len // bin_width
    stim_filter_len_bin = stim_filter_len // bin_width
    coupling_filter_len_bin = coupling_filter_len // bin_width

    trial_start_offset = -2000 # ms
    trial_lims = np.array([1000,4500])
    stim_t = 2000

    # Define basis kwargs
    basis_kwargs = dict(
        n_basis = 15,
        basis = 'cos',
        basis_spread = 'log',
        )

    # Number of fits on actual data (expensive)
    n_fits = 5
    n_max_tries = 20
    # Number of shuffles tested against each fit
    n_shuffles_per_fit = 10

    # Save run parameters
    params_dict = dict(
            hist_filter_len = hist_filter_len,
            stim_filter_len = stim_filter_len,
            coupling_filter_len = coupling_filter_len,
            bin_width = bin_width,
            hist_filter_len_bin = hist_filter_len_bin,
            stim_filter_len_bin = stim_filter_len_bin,
            coupling_filter_len_bin = coupling_filter_len_bin,
            trial_start_offset = trial_start_offset,
            trial_lims = list(trial_lims),
            stim_t = stim_t,
            basis_kwargs = basis_kwargs,
            n_fits = n_fits,
            n_max_tries = n_max_tries,
            n_shuffles_per_fit = n_shuffles_per_fit,
            )

    params_save_path = os.path.join(fin_save_path, 'fit_params.json')
    with open(params_save_path, 'w') as outf:
        json.dump(params_dict, outf, indent = 4, default = int)
    print('Creating run with following parameters :')
    pprint(params_dict)

############################################################

reprocess_data = False 
spike_list_path = os.path.join(save_path,'spike_save')

if reprocess_data:

    file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
    file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
    basenames = [os.path.basename(x) for x in file_list]

    spike_list = []
    unit_region_list = []
    for ind in trange(len(file_list)):
        dat = ephys_data(file_list[ind])
        dat.get_spikes()
        spike_list.append(np.array(dat.spikes))
        # Calc mean firing rate
        mean_fr = np.array(dat.spikes).mean(axis=(0,1,3))
        dat.get_region_units()
        region_units = dat.region_units
        region_names = dat.region_names
        region_names = [[x]*len(y) for x,y in zip(region_names,region_units)]
        region_units = np.concatenate(region_units)
        region_names = np.concatenate(region_names)
        unit_region_frame = df(
                {'region':region_names,
                 'unit':region_units,
                 }
                )
        unit_region_frame['basename'] = basenames[ind]
        unit_region_frame['session'] = ind
        unit_region_frame = unit_region_frame.sort_values(by=['unit'])
        unit_region_frame['mean_rate'] = mean_fr
        unit_region_list.append(unit_region_frame)

    # Save spike_list as numpy arrays

    spike_inds_list = [np.stack(np.where(x)) for x in spike_list]

    if not os.path.exists(spike_list_path):
        os.makedirs(spike_list_path)
    for ind,spikes in enumerate(spike_list):
        #np.save(os.path.join(spike_list_path,f'{ind}_spikes.npy'),spikes)
        np.save(os.path.join(spike_list_path,f'{ind:03}_spike_inds.npy'),spike_inds_list[ind])

    unit_region_frame = concat(unit_region_list)
    unit_region_frame = unit_region_frame.reset_index(drop=True)
    unit_region_frame.to_csv(os.path.join(save_path,'unit_region_frame.csv'))

    # Make sure all sessions have 4 tastes
    assert all([x.shape[0] == 4 for x in spike_list])

    # Find neurons per session
    nrn_counts = [x.shape[2] for x in spike_list]

    # Process each taste separately
    inds = np.array(list(product(range(len(spike_list)),range(spike_list[0].shape[0]))))
    fin_inds = []
    for ind in tqdm(inds):
        this_count = nrn_counts[ind[0]]
        for nrn in range(this_count):
            fin_inds.append(np.hstack((ind,nrn)))
    fin_inds = np.array(fin_inds)

    ind_frame = df(fin_inds,columns=['session','taste','neuron'])
    ind_frame.to_csv(os.path.join(save_path,'ind_frame.csv'))

else:
    ############################################################
    # Reconstitute data
    spike_inds_paths = sorted(glob(os.path.join(spike_list_path,'*_spike_inds.npy')))
    spike_inds_list = [np.load(x) for x in spike_inds_paths]
    spike_list = [gen_spike_train(x) for x in spike_inds_list]

    # Load unit_region_frame
    unit_region_frame = pd.read_csv(os.path.join(save_path,'unit_region_frame.csv'),index_col=0)

    # Load ind_frame
    ind_frame = pd.read_csv(os.path.join(save_path,'ind_frame.csv'),index_col=0)
    fin_inds = ind_frame.values

# Sort inds by total number of neurons per session
# This is needed because larger sessions take a long time to fit
count_per_session = ind_frame.groupby(by='session').count().values[:,0]
ind_frame['count'] = count_per_session[ind_frame['session'].values]
ind_frame = ind_frame.sort_values(by='count')
fin_inds = ind_frame.values[:,:-1] # Drop Count

############################################################

# While iterating, will have to keep track of
# 1. Region each neuron belongs to
# 2. log-likehood for each data-type
# 3. p-values
stim_vec = np.zeros(spike_list[0].shape[-1])
stim_vec[stim_t] = 1

for num, this_ind in tqdm(enumerate(fin_inds)):
    process_ind(num, this_ind)

# TODO: Some large models max out ram and make parallelize crash. 
# These can be tracked and ignored till the end so the rest of the fits can be completed
