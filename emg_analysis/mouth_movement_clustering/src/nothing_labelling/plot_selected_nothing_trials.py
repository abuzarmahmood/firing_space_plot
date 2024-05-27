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

##############################
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

nothing_selected_plot_dir = os.path.join(nothing_label_plot_dir, 'nothing_selected')
if not os.path.exists(nothing_selected_plot_dir):
    os.makedirs(nothing_selected_plot_dir)

# Get selected inds
selected_inds_path = os.path.join(artifact_dir, 'nothing_labels.csv')
selected_inds = pd.read_csv(selected_inds_path)

for i in tqdm(selected_inds.Abu):

    this_ind = inds[i]

    save_path = os.path.join(nothing_selected_plot_dir, f'sample_{i}.png')

    # if os.path.exists(save_path):
    #     continue

    this_env = wanted_envs[tuple(this_ind)]
    this_taste = taste_list[this_ind[0]][this_ind[1]]
    this_basename = basenames[this_ind[0]]
    this_trial_ind = this_ind[-1]

    all_trials_env = all_envs[this_ind[0], this_ind[1], :]

    fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True, sharex=True)
    ax[0].plot(x_vec, all_envs[tuple(this_ind)], color='black')
    ax[0].set_title(f'{this_basename} - {this_taste} - {this_trial_ind}' +\
                 '\n' + f'Sample {i} - Selected')
    ax[0].axvspan(-2000, -1000, color='yellow', alpha=0.5)
    ax[0].axhline(0, color='black', linestyle='--')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('EMG Value')
    ax[1].plot(x_vec, all_trials_env.T, color='black', alpha=0.1)
    ax[1].set_title(f'{this_basename} - {this_taste} - {this_trial_ind}' +\
                 '\n' + f'Sample {i} - All Trials')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
