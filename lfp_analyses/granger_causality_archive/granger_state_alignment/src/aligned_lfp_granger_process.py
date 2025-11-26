from glob import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

import sys
granger_causality_path = \
    '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/process_scripts/'
sys.path.append(granger_causality_path)
import granger_utils as gu

############################################################
# Get aligned LFP
artifact_path = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/granger_state_alignment/artifacts'

aligned_lfp_name = 'aligned_lfp_frame.pkl'
aligned_lfp_path = os.path.join(artifact_path,aligned_lfp_name)
aligned_lfp_frame = pd.read_pickle(aligned_lfp_path)
aligned_lfp_frame.reset_index(inplace=True)

############################################################

max_freq = 100
mean_granger_list = []
granger_mask_list = []
time_vec_list = []
freq_vec_list = []

for ind, this_row in tqdm(list(aligned_lfp_frame.iterrows())[len(mean_granger_list):]):
    try:
        print('=====================')
        print(f'Processing {ind}/{len(aligned_lfp_frame)}')
        print('=====================')

        this_dat = this_row['aligned_lfp']

        # Do NOT preprocess here as we are giving it 
        # preprocessed data
        this_granger = gu.granger_handler(
                                  this_dat,
                                  multitaper_time_halfbandwidth_product=1,
                                  preprocess=False,
                                  wanted_window = [0, 2000],
                                  warning_only=True
                                  )
        this_granger.get_granger_sig_mask()

        mean_granger = np.nanmean(this_granger.granger_actual,axis=0)
        granger_mask = this_granger.mask_array
        mean_granger = np.stack(
                [
                mean_granger[:,:,0,1],
                mean_granger[:,:,1,0],
                ]
            )
        granger_mask = np.stack(
                [
                granger_mask[:,:,0,1],
                granger_mask[:,:,1,0],
                ]
            )
        time_vec = this_granger.time_vec
        freq_vec = this_granger.freq_vec
        freq_bool = freq_vec <= max_freq
        mean_granger = mean_granger[...,freq_bool]
        granger_mask = granger_mask[...,freq_bool]
        freq_vec = freq_vec[freq_bool]
    except:
        print(f'Error processing {ind}')
        mean_granger = np.nan
        granger_mask = np.nan
        time_vec = np.nan
        freq_vec = np.nan

    mean_granger_list.append(mean_granger)
    granger_mask_list.append(granger_mask)
    time_vec_list.append(time_vec)
    freq_vec_list.append(freq_vec)

granger_frame = aligned_lfp_frame.copy()
granger_frame.drop(columns='aligned_lfp',inplace=True)
granger_frame['mean_granger'] = mean_granger_list
granger_frame['granger_mask'] = granger_mask_list
granger_frame['time_vec'] = time_vec_list
granger_frame['freq_vec'] = freq_vec_list

granger_frame.to_pickle(
        os.path.join(artifact_path,'granger_frame.pkl'))
