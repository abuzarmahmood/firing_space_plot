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
base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/src'
#base_path = '/home/exouser/Desktop/ABU/firing_space_plot/firing_analyses/poisson_glm'
sys.path.append(base_path)
# from utils.glm_ephys_process_utils import (
#         gen_stim_vec,
#         gen_spike_train,
#         process_ind,
#         try_process,
#         )
import utils.glm_ephys_process_utils as process_utils
from utils import utils
# import glm_tools as gt
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
save_path = os.path.join(
        os.path.dirname(base_path),
        'artifacts'
        )

############################################################
## Begin Process 
############################################################

# Check if previous runs present
run_list = sorted(glob(os.path.join(save_path, 'run*')))
run_basenames = sorted([os.path.basename(x) for x in run_list])
print(f'Present runs : {run_basenames}')
# input_run_ind = int(input('Please specify current run (integer) :'))
input_run_ind = 6
fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03}')

if not os.path.exists(fin_save_path):
    os.makedirs(fin_save_path)
    run_exists_bool = False
    params_dict = utils.generate_params_dict(fin_save_path)
else:
    run_exists_bool = True
    json_path = os.path.join(fin_save_path, 'fit_params.json')
    params_dict = json.load(open(json_path))
    print('Run exists with following parameters :')
    pprint(params_dict)

############################################################
############################################################

reprocess_data = False 
spike_list_path = os.path.join(save_path,'spike_save')

if reprocess_data:

    file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
    file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
    basenames = [os.path.basename(x) for x in file_list]

    spike_list, unit_region_list =\
            utils.get_unit_spikes_and_regions(file_list)

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
    inds = np.array(
            list(product(range(len(spike_list)),range(spike_list[0].shape[0]))))
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
    spike_inds_paths = sorted(
            glob(os.path.join(spike_list_path,'*_spike_inds.npy')))
    spike_inds_list = [np.load(x) for x in spike_inds_paths]
    spike_list = [process_utils.gen_spike_train(x) for x in spike_inds_list]

    # Load unit_region_frame
    unit_region_frame = pd.read_csv(
            os.path.join(save_path,'unit_region_frame.csv'),index_col=0)

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

stim_vec = process_utils.gen_stim_vec(spike_list, params_dict)

for num, this_ind in tqdm(enumerate(fin_inds)):
    args = (
            num, 
            this_ind,
            spike_list,
            stim_vec,
            params_dict,
            fin_save_path
            )
    # process_utils.process_ind(*args)
    process_utils.try_process(args)

    print()
    print('############################################################')
    print(f'Finished processing {num} of {len(fin_inds)} --- {num/len(fin_inds)*100:.2f}%')
    print('############################################################')
    print()

# TODO: Some large models max out ram and make parallelize crash. 
# These can be tracked and ignored till the end so the rest of the fits can be completed
