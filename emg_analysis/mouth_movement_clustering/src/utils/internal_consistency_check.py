"""
Check that segments in gape_frame match with emg_array
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from scipy.ndimage import white_tophat
from tqdm import tqdm, trange

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
code_dir = os.path.join(base_dir, 'src')
sys.path.append(code_dir)

##############################
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

all_envs = all_data_frame['env'].tolist()
all_gape_frames = all_data_frame['gape_frame_raw'].tolist()

##############################

diff_dict_list = []
for session_ind in trange(len(all_envs)):
    for taste_ind in range(len(all_envs[session_ind])):
        for trial_ind in range(len(all_envs[session_ind][taste_ind])):
            this_env = all_envs[session_ind][taste_ind][trial_ind]
            session_gape_frame = all_gape_frames[session_ind]
            this_gape_frame = session_gape_frame.query(
                    f'taste == {taste_ind} and trial == {trial_ind}')
            filt_env = white_tophat(this_env, size=200)
            segment_array = np.nan*np.ones_like(this_env)
            for i, this_row in this_gape_frame.iterrows():
                segment_dat = this_row['segment_raw']
                segment_bounds = this_row['segment_bounds']
                segment_array[segment_bounds[0]+2000:segment_bounds[1]+2000] = segment_dat
                
            # Get difference between segment_array and filt_env
            diff_array = segment_array - filt_env
            dat_diff = np.nanmean(diff_array)

            diff_dict_list.append({'session':session_ind,
                                   'taste':taste_ind,
                                   'trial':trial_ind,
                                   'diff':dat_diff})

diff_frame = pd.DataFrame(diff_dict_list)
diff_frame['diff'].sum()

