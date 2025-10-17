"""
Get EMG envs, and focus on -2000 -> -1000 ms post-stimulus
For every trial, plot:
    1) Full trial EMG with wanted window highlighted
    2) EMG in wanted window by itself
        2.1) y-axis shared with 1
        2.2) y-axis not shared with 1
    3) Average EMG for that taste

"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
code_dir = os.path.join(base_dir, 'src')
sys.path.append(code_dir)
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            find_segment,
                                            calc_peak_interval,
                                            JL_process,
                                            gen_gape_frame,
                                            threshold_movement_lengths,
                                            )


##############################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

all_envs = np.array(all_data_frame['env'].tolist())
x_vec = np.arange(-2000, 5000)
taste_map_list = all_data_frame.taste_map.tolist()
taste_list = [list(x.keys()) for x in taste_map_list]
basenames = all_data_frame.basename.tolist()

wanted_window = (-2000, -1000)
wanted_inds = np.where((x_vec >= wanted_window[0]) & (x_vec < wanted_window[1]))[0]
wanted_envs = all_envs[..., wanted_inds]


inds_path = os.path.join(artifact_dir, 'nothing_label_inds.npy')
if os.path.exists(inds_path):
    inds = np.load(inds_path)
else:
    inds = np.array(list(np.ndindex(all_envs.shape[:3])))
    # Randomize inds and save
    inds = np.random.permutation(inds)
    np.save(inds_path, inds)

nothing_label_plot_dir = os.path.join(plot_dir, 'nothing_label')
if not os.path.exists(nothing_label_plot_dir):
    os.makedirs(nothing_label_plot_dir)

# For all inds, extract movements so they can also be plotted
segment_dat_list = []
filtered_segment_dat_list = []
for this_ind in tqdm(inds):
    this_trial_dat = all_envs[tuple(this_ind)] 

    (
        segment_starts, 
        segment_ends, 
        segment_dat,
        filtered_segment_dat
        ) = extract_movements(this_trial_dat, size=200)

    segment_dat_list.append([segment_starts, segment_ends, segment_dat])
    filtered_segment_dat_list.append([segment_starts, segment_ends, filtered_segment_dat])


for i, this_ind in enumerate(tqdm(inds[:200])):
    
    save_path = os.path.join(nothing_label_plot_dir, f'sample_{i}.png')

    # if os.path.exists(save_path):
    #     continue

    this_env = wanted_envs[tuple(this_ind)]
    this_taste = taste_list[this_ind[0]][this_ind[1]]
    this_basename = basenames[this_ind[0]]
    this_trial_ind = this_ind[-1]

    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(2,3 , sharey=True, sharex=True, figsize=(15, 5))
    ax[0,0].plot(x_vec, all_envs[tuple(this_ind)].T, linewidth = 0.5, zorder = -1,
                 color='black')
    ax[0,0].axvspan(wanted_window[0], wanted_window[1], 
                    color='yellow', alpha=0.5, zorder=-1)
    ax[0,0].set_title('Full Trial EMG')
    ax[0,0].set_ylabel('EMG')
    ax[0,0].set_xlabel('Time (ms)')
    ax[0,1].plot(x_vec[wanted_inds], this_env.T, linewidth = 0.5, zorder = -1,
                 color='black')
    ax[0,1].set_title('Wanted Window EMG')
    ax[0,1].set_ylabel('EMG')
    ax[0,1].set_xlabel('Time (ms)')
    taste_envs = all_envs[this_ind[0], this_ind[1], :]
    ax[0,2].plot(x_vec, taste_envs.T, alpha=0.5, color='gray')
    ax[0,2].set_title('Average EMG for Taste')
    ax[0,2].set_ylabel('EMG')
    ax[0,2].set_xlabel('Time (ms)')
    
    segment_starts, segment_ends, segment_dat = segment_dat_list[i]
    filtered_segment_dat = filtered_segment_dat_list[i][2]
    for j, (start, end, dat) in enumerate(zip(segment_starts, segment_ends, segment_dat)):
        ax[0,0].plot(x_vec[start:end], dat, linewidth=2, zorder = 1, c = cmap(j%10))
        if x_vec[start] > wanted_window[0] and x_vec[end] < wanted_window[1]:
            ax[0,1].plot(x_vec[start:end], dat, linewidth=2, zorder = 1,
                         c = cmap(j%10))
            ax[1,1].plot(x_vec[start:end], filtered_segment_dat[j], linewidth=2, zorder = 1,
                         c = cmap(j%10))
        ax[1,0].plot(x_vec[start:end], filtered_segment_dat[j], linewidth=2, zorder = 1,
                     c = cmap(j%10))

    ax[1,0].set_title('Filtered Movements')

    fig.suptitle(f'{this_basename} - {this_taste} - {this_trial_ind}' +\
            '\n' + f'Sample {i}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
