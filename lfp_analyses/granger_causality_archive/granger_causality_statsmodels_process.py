"""
Perform Granger Causality test using statsmodels library
"""

from glob import glob
import tables
import os
import numpy as np
import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
granger_causality_path = \
    '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/process_scripts'
sys.path.append(ephys_data_dir)
sys.path.append(granger_causality_path)
import granger_utils as gu
from ephys_data import ephys_data

############################################################
# Load Data
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

for dir_name in dir_list:
    try:
        dir_name = dir_list[0]
        basename = dir_name.split('/')[-1]
        # Assumes only one h5 file per dir
        h5_path = glob(dir_name + '/*.h5')[0]

        print(f'Processing {basename}')

        dat = ephys_data(dir_name)
        dat.get_info_dict()
        
        # # Pull out longer trial-durations for LFP
        # dat.default_lfp_params['trial_durations'] = [10000,10000]
        # dat.extract_lfps()

        taste_names = dat.info_dict['taste_params']['tastes']
        # Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
        lfp_channel_inds, region_lfps, region_names = \
            dat.return_representative_lfp_channels()

        taste_lfps = [x for x in region_lfps.swapaxes(0,1)]

        flat_region_lfps = np.reshape(
            region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))
        flat_taste_nums = np.repeat(np.arange(len(taste_names)),
                                    taste_lfps[0].shape[1])

        lfp_set_names = taste_names.copy()
        lfp_set_names.append('all')

        ############################################################
        # Preprocessing
        ############################################################

        # NOTE: Preprocess all tastes together so that separate
        # preprocessing does not introduce systematic differences
        # between tastes

        # 1) Remove trials with artifacts
        good_lfp_trials_bool = \
                dat.lfp_processing.return_good_lfp_trial_inds(flat_region_lfps)
        good_lfp_trials = flat_region_lfps[:,good_lfp_trials_bool]
        good_taste_nums = flat_taste_nums[good_lfp_trials_bool]

        
        # 2) Preprocess data
        this_granger = gu.granger_handler(good_lfp_trials)
        this_granger.preprocess_and_check_stationarity()
        preprocessed_data = this_granger.preprocessed_data

        # Make list of data according to lfp_set_names 
        preprocessed_lfp_data = [
                preprocessed_data[:, good_taste_nums == x]\
                for x in range(len(taste_names))
                ]
        preprocessed_lfp_data.append(preprocessed_data)

        # Only process 'all' trials together
        lfp_set_names = [lfp_set_names[-1]]
        preprocessed_lfp_data = [preprocessed_lfp_data[-1]]


        # 3) Calculate Granger Causality
