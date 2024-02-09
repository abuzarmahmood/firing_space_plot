from glob import glob
import os
import numpy as np
from tqdm import tqdm

############################################################
# Get dataset paths
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()
basename_list = [os.path.basename(d) for d in dir_list]

############################################################
# Get saved models
############################################################
base_dir = '/media/bigdata/projects/pytau'
import sys
sys.path.append(base_dir)

import pandas as pd
model_database_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/saved_models/model_database.csv'
model_database = pd.read_csv(model_database_path, index_col=0)
model_database.dropna(inplace=True)

pymc_version = '3.9.3'
wanted_database = \
        model_database[model_database['module.pymc3_version'] == pymc_version]

wanted_state_n = 4
wanted_database = wanted_database[\
        wanted_database['model.states'] == wanted_state_n]

wanted_region = 'gc'
wanted_database = wanted_database[\
        wanted_database['data.region_name'] == wanted_region]

wanted_database = wanted_database[\
        wanted_database['data.basename'].isin(basename_list)]

############################################################
# Get changepoint positions
############################################################
from pytau.changepoint_analysis import PklHandler

taste_nums = sorted(wanted_database['data.taste_num'].unique())
fin_basename_list = []
taste_nums_list = []
pkl_path_list = []
for this_basename in basename_list:
    for this_taste in taste_nums:
        print(f'Processing {this_basename} taste {this_taste}')
        this_database = wanted_database[\
                wanted_database['data.basename'] == this_basename]
        this_database = this_database[\
                this_database['data.taste_num'] == this_taste]
        this_pkl_path = this_database['exp.save_path'].values[0]
        fin_basename_list.append(this_basename)
        taste_nums_list.append(this_taste)
        pkl_path_list.append(this_pkl_path)

pkl_frame = pd.DataFrame({'basename':fin_basename_list,
                          'taste_num':taste_nums_list,
                          'pkl_path':pkl_path_list})

tau_list = []
present_bool_list = []
for idx, row in tqdm(pkl_frame.iterrows()):
    try:
        print(f'Processing {row["basename"]} taste {row["taste_num"]}')
        pkl_path = row['pkl_path']
        this_handler = PklHandler(pkl_path)
        scaled_mode_tau = this_handler.tau.scaled_mode_tau
        tau_list.append(scaled_mode_tau)
        present_bool_list.append(True)
    except:
        tau_list.append(np.nan)
        present_bool_list.append(False)

pkl_frame['tau'] = tau_list
pkl_frame['present'] = present_bool_list

artifact_path = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/granger_state_alignment/artifacts'

pkl_frame.to_pickle(os.path.join(artifact_path, 'tau_frame.pkl'))
