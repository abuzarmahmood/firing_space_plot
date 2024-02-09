from glob import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
granger_causality_path = \
    '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/process_scripts'
sys.path.append(ephys_data_dir)
sys.path.append(granger_causality_path)
import granger_utils as gu
from ephys_data import ephys_data

artifact_path = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/granger_state_alignment/artifacts'

############################################################
# Get dataset paths
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()
basename_list = [os.path.basename(d) for d in dir_list]

############################################################
# Get LFP 
############################################################
processed_lfp_list = []
present_bool = []
good_trial_bool_list = []
# dir_name = dir_list[0]
for dir_name in tqdm(dir_list):
    try:
        dat = ephys_data(dir_name)
        dat.get_info_dict()

        # # Pull out longer trial-durations for LFP
        # dat.default_lfp_params['trial_durations'] = [2000, 5000]
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

        ############################################################
        # Preprocessing
        ############################################################

        # NOTE: Preprocess all tastes together so that separate
        # preprocessing does not introduce systematic differences
        # between tastes

        # 1) Remove trials with artifacts
        good_lfp_trials_bool = \
                dat.lfp_processing.return_good_lfp_trial_inds(flat_region_lfps)

        # 2) Preprocess data
        this_granger = gu.granger_handler(flat_region_lfps)
        this_granger.preprocess_data()
        preprocessed_data = this_granger.preprocessed_data

        # Make list of data according to lfp_set_names 
        preprocessed_lfp_data = [
                preprocessed_data[:, flat_taste_nums == x]\
                for x in range(len(taste_names))
                ]

        good_taste_trials_bool = [
                good_lfp_trials_bool[flat_taste_nums == x]\
                for x in range(len(taste_names))
                ]

        processed_lfp_list.append(preprocessed_lfp_data)
        present_bool.append(True)
        good_trial_bool_list.append(good_taste_trials_bool)
    except:
        processed_lfp_list.append([None]*4)
        present_bool.append(False)
        good_trial_bool_list.append([[None]*30]*4)

##############################
taste_lens = [len(x) for x in processed_lfp_list]
fin_basename_list = [[basename_list[i]]*x for i,x in enumerate(taste_lens)]
fin_basename_list = [item for sublist in fin_basename_list for item in sublist]
fin_taste_list = [np.arange(x) for x in taste_lens]
fin_taste_list = [item for sublist in fin_taste_list for item in sublist]
fin_present_bool = [[present_bool[i]]*x for i,x in enumerate(taste_lens)]
fin_present_bool = [item for sublist in fin_present_bool for item in sublist]
fin_processed_lfp_list = [item for sublist in processed_lfp_list for item in sublist]
fin_good_trial_bool_list = [item for sublist in good_trial_bool_list for item in sublist]

# Save to dataframe
fin_df = pd.DataFrame({'dir_name':fin_basename_list,
                       'taste_num':fin_taste_list,
                       'present':fin_present_bool,
                       'lfp_data':fin_processed_lfp_list,
                       'good_trial_bool':fin_good_trial_bool_list
                       })

fin_df.to_pickle(os.path.join(artifact_path,'lfp_data.pkl'))
