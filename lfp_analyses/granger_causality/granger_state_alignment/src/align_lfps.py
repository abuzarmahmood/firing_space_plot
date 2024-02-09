from glob import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

artifact_path = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/granger_state_alignment/artifacts'

lfp_data_filename = 'lfp_data.pkl'
tau_frame_filename = 'tau_frame.pkl'

lfp_frame = pd.read_pickle(os.path.join(artifact_path,lfp_data_filename))
tau_frame = pd.read_pickle(os.path.join(artifact_path,tau_frame_filename))

lfp_frame['taste_num'] = lfp_frame['taste_num'].astype(int)
tau_frame['taste_num'] = tau_frame['taste_num'].astype(int)

fin_frame = pd.merge(
        lfp_frame,
        tau_frame,
        left_on=['dir_name','taste_num'],
        right_on=['basename','taste_num'],
        suffixes=('_lfp','_tau'),
        how='inner')

present_bool = np.logical_and(
        fin_frame['present_lfp'],
        fin_frame['present_tau'])

fin_frame = fin_frame[present_bool]

fin_frame.drop(
        columns=['present_lfp','present_tau','pkl_path', 'basename'],
        inplace=True)

############################################################
# Align LFP

# 1. Transition-aligned
# 2. Average-transition aligned

snippet_window_radius = 1000

aligned_frame_list = []
for i, this_row in tqdm(fin_frame.iterrows()):
    print(f'Processing {this_row["dir_name"]} taste {this_row["taste_num"]}')
    this_tau = this_row['tau']
    this_lfp = this_row['lfp_data']
    this_good_trial_bool = this_row['good_trial_bool']

    fin_tau = this_tau[this_good_trial_bool].T
    fin_lfp = this_lfp[:,this_good_trial_bool]

    ############################## 
    # Transition aligned
    ############################## 
    aligned_dat_list = []
    for this_transition in fin_tau:
        window_starts = this_transition - snippet_window_radius
        window_ends = this_transition + snippet_window_radius
        windowed_lfp = np.stack(
                [this_trial[:,start:end] for this_trial, start, end in \
                zip(fin_lfp, window_starts, window_ends)]
                )
        aligned_dat_list.append(windowed_lfp)
    this_frame = pd.DataFrame(
            dict(
                dir_name = this_row['dir_name'],
                taste_num = this_row['taste_num'],
                transition = np.arange(len(fin_tau)),
                aligned_lfp = aligned_dat_list,
                aligned = True,
                )
            )
    aligned_frame_list.append(this_frame)

    ############################## 
    # Average transition aligned 
    ############################## 
    unaligned_dat_list = [] 
    for this_transition in fin_tau:
        mean_transition = int(this_transition.mean())
        mean_start = mean_transition - snippet_window_radius
        mean_end = mean_transition + snippet_window_radius
        unaligned_dat = fin_lfp[:,:,mean_start:mean_end]
        unaligned_dat_list.append(unaligned_dat)

    this_frame = pd.DataFrame(
            dict(
                dir_name = this_row['dir_name'],
                taste_num = this_row['taste_num'],
                transition = np.arange(len(fin_tau)),
                aligned_lfp = unaligned_dat_list,
                aligned = False,
                )
            )
    aligned_frame_list.append(this_frame)


aligned_frame = pd.concat(aligned_frame_list)
# Write to file
aligned_frame.to_pickle(os.path.join(artifact_path,'aligned_lfp_frame.pkl'))
