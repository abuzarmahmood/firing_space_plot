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
from functools import partial

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6

#base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm'
save_path = os.path.join(
        os.path.dirname(base_path),
        'artifacts'
        )

def parallelize(func, iterator, n_jobs = 8):
    return Parallel(n_jobs = n_jobs)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'

############################################################
## Begin Process 
############################################################
json_template_path = os.path.join(base_path, 'params', 'template_fit_params.json')
params_dict = json.load(open(json_template_path))
print('Run exists with following parameters :')
pprint(params_dict)

# input_run_ind = 8
run_params_path = os.path.join(base_path, 'params', 'run_params.json')
run_params = json.load(open(run_params_path))
input_run_ind = run_params['run_ind']
fraction_used = run_params['fraction_used']


fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03}')

# Copy template json to save path
json_save_path = os.path.join(fin_save_path, 'fit_params.json')
with open(json_save_path, 'w') as json_file:
    json.dump(params_dict, json_file)

# if not os.path.exists(fin_save_path):
#     os.makedirs(fin_save_path)
#     run_exists_bool = False
#     params_dict = utils.generate_params_dict(fin_save_path)
# else:
#     run_exists_bool = True
#     params_dict = json.load(open(json_path))
#     print('Run exists with following parameters :')
#     pprint(params_dict)

############################################################
############################################################

reprocess_data = False 
spike_list_path = os.path.join(save_path,'spike_save')

if reprocess_data:
    (
        spike_list, 
        unit_region_frame, 
        ind_frame, 
        fin_inds,
        ) = utils.load_dat_from_path_list(
                        file_list_path, save_path)
else:
    (
        spike_list, 
        unit_region_frame, 
        ind_frame, 
        fin_inds,
        ) = utils.load_dat_from_save(
                        spike_list_path, save_path)

# Sort inds by total number of neurons per session
# This is needed because larger sessions take a long time to fit
count_per_session = ind_frame.groupby(by='session').count().values[:,0]
ind_frame['count'] = count_per_session[ind_frame['session'].values]
ind_frame = ind_frame.sort_values(by='count')
fin_inds = ind_frame.values[:,:-1] # Drop Count

# Sample fin_inds according to fraction_used to speed up fitting
subsample_inds = np.random.choice(
        np.arange(len(fin_inds)),
        size=int(len(fin_inds)*fraction_used),
        replace=False,
        )
fin_inds = fin_inds[subsample_inds]
print()
print('============================================================')
print(f'=== Fitting only {fraction_used*100}% of data')
print('============================================================')
print()

############################################################

# While iterating, will have to keep track of
# 1. Region each neuron belongs to
# 2. log-likehood for each data-type
# 3. p-values

stim_vec = process_utils.gen_stim_vec(spike_list, params_dict)

# Some large models max out ram and make parallelize crash. 
# These can be tracked and ignored till the end so the rest 
# of the fits can be completed
# For now, just run parallel first, and then run sequential 
# to catch the ones that fail
try_process_parallel = partial(
        process_utils.try_process,
        spike_list=spike_list,
        stim_vec=stim_vec,
        params_dict=params_dict,
        fin_save_path=fin_save_path
        )

outs = parallelize(
        try_process_parallel,
        fin_inds,
        n_jobs=4,
        )


pbar = tqdm(total=len(fin_inds))
for ind_num, this_ind in tqdm(enumerate(fin_inds)):
    try_process_parallel(this_ind)

    # Manually update tqdm
    print()
    print('############################################################')
    print(f'Finished processing {ind_num} of {len(fin_inds)} --- {ind_num/len(fin_inds)*100:.2f}%')
    pbar.update(1)
    print('############################################################')
    print()

pbar.close()

