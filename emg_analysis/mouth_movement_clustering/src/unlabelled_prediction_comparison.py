"""
Comparison of all models on unlabelled data to test the following:
    1) Do predictions appropriately predict palatability of trial?
    2) Are non-null (palatability-indicating) movements restricted to post-stimulus activity?

**Note: 
    JL algorithm takes pre-stimulus information into account
        1) by removing any mouth movements that occur before the stimulus
        2) by removing any mouth movements smaller than that mean+std of baseline activity
    These additional steps prevent us from asking how well the JL-QDA classifier performs on
    arbitrary mouth movements.
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import xgboost as xgb
import sys

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
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

sys.path.append(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier'))
from QDA_classifier import QDA

artifact_dir = os.path.join(base_dir, 'artifacts')

##############################
def modified_JL_process(
        left_end_list,
        right_end_list,
        peak_ind,
        ):
    """
    This function takes in a trial of data and returns the gapes
    according to Jenn Li's pipeline

    Inputs:
        left_end_list, right_end_list: list of start and end times of mouth movements
                for a given trial
        peak_ind: list of peak indices of mouth movements for a given trial

    Outputs:
        gape_peak_ind: list of peak indices of gapes for a given trial
    """

    left_end_list = np.array(left_end_list)
    right_end_list = np.array(right_end_list)
    peak_ind = np.array(peak_ind)

    dur = right_end_list - left_end_list 
    dur_bool = np.logical_and(dur > 20.0, dur <= 200.0)
    durations = dur[dur_bool]
    peak_ind = peak_ind[dur_bool]
    if len(peak_ind) == 0:
        return None, [np.nan, np.nan]

    # In case there aren't any peaks or just one peak
    # (very unlikely), skip this trial 
    if len(peak_ind) > 1:
        intervals = calc_peak_interval(peak_ind)

        gape_bool = [QDA(intervals[peak], durations[peak]) for peak in range(len(durations))]
        # Drop first one
        gape_bool = gape_bool[1:]
        peak_ind = peak_ind[1:]

        gape_peak_ind = peak_ind[gape_bool]
    else:
        gape_peak_ind = None

    return gape_peak_ind

############################################################
# Load data
############################################################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
session_specific_plot_dir = os.path.join(plot_dir, 'session_specific_plots')
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

pred_plot_dir = os.path.join(plot_dir, 'prediction_plots', 'jl_qda')
if not os.path.exists(pred_plot_dir):
    os.makedirs(pred_plot_dir)

##############################
# Gapes by JL
##############################

taste_map_list = all_data_frame.taste_map.tolist()
taste_list = [list(x.keys()) for x in taste_map_list]

gape_frame_list = all_data_frame.gape_frame_raw.tolist()
basenames = all_data_frame.basename.tolist()

# Recreate gape arrays from dataframes to make plotting easier
JL_gape_shape = (4,30,7000)
gape_array_list = []
for gape_frame in tqdm(gape_frame_list):
    this_gape_frame = gape_frame.copy()
    this_gape_frame = this_gape_frame.loc[this_gape_frame.classifier > 0]
    gape_array = np.zeros(JL_gape_shape)
    inds = this_gape_frame[['taste','trial','segment_bounds']]
    inds['segment_bounds'] += 2000
    for taste, trial, segment_bounds in inds.values:
        gape_array[taste,trial,segment_bounds[0]:segment_bounds[1]] = 1
    gape_array_list.append(gape_array)

x_vec = np.arange(-2000, 5000)

# Plot gapes for all trials
for i, (gape_array, basename) in enumerate(zip(gape_array_list, basenames)):
    fig, ax = plt.subplots(4,1,sharex=True,sharey=True)
    this_tastes = taste_list[i]
    for taste in range(4):
        ax[taste].imshow(gape_array[taste],aspect='auto',
                         cmap='gray',vmin=0,vmax=1,)
        ax[taste].set_title(f'{this_tastes[taste]}')
    fig.suptitle(basename)
    fig.savefig(os.path.join(pred_plot_dir, basename + '_JL_gapes.png'))
    plt.close(fig)

# Also make plots for mean gapes
mean_gapes = np.mean(np.array(gape_array_list),axis=2)

fig, ax = plt.subplots(len(mean_gapes),1,sharex=True,sharey=True,
                       figsize = (5, len(mean_gapes)*3))
for i, (this_mean_gape, basename) in enumerate(zip(mean_gapes, basenames)):
    this_tastes = taste_list[i]
    for taste in range(4):
        ax[i].plot(x_vec, this_mean_gape[taste], label=f'{this_tastes[taste]}')
    ax[i].set_title(basename)
    ax[i].axvline(0, color='r', linestyle='--')
    ax[i].legend()
fig.suptitle('Mean Gapes by JL')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
fig.savefig(os.path.join(pred_plot_dir, 'mean_gapes_JL.png'))
plt.close(fig)

##############################
# Gapes by JL (no amplitude info)
##############################

# Add session_number to gape_frame_list
for i, gape_frame in enumerate(gape_frame_list):
    gape_frame['session_number'] = i
    gape_frame['left_end'] = gape_frame['segment_bounds'].apply(lambda x: x[0])
    gape_frame['right_end'] = gape_frame['segment_bounds'].apply(lambda x: x[1])
    gape_frame['peak_ind'] = gape_frame['segment_center'].apply(lambda x: x*1000)

cat_gape_frame = pd.concat(gape_frame_list)
single_trial_groups = list(cat_gape_frame.groupby(['session_number','taste','trial']))
single_trial_ids = [x[0] for x in single_trial_groups]
single_trial_frames = [x[1] for x in single_trial_groups]

# For each trial, run modified JL process
gape_peak_inds = []
for trial_frame in tqdm(single_trial_frames):
    left_end_list = trial_frame.left_end.tolist()
    right_end_list = trial_frame.right_end.tolist()
    peak_ind = trial_frame.peak_ind.tolist()
    gape_peak_ind = modified_JL_process(left_end_list, right_end_list, peak_ind)
    gape_peak_inds.append(gape_peak_ind + 2000)

# Get fraction of gapes
gape_fractions = []
for this_gape_inds, this_trial_frame in zip(gape_peak_inds, single_trial_frames):
    this_gape_frac = len(this_gape_inds)/len(this_trial_frame)
    gape_fractions.append(this_gape_frac)

pred_plot_dir = os.path.join(plot_dir, 'prediction_plots', 'jl_qda_no_amp')
if not os.path.exists(pred_plot_dir):
    os.makedirs(pred_plot_dir)

# Plot histogram of gape fractions
fig, ax = plt.subplots()
sns.histplot(gape_fractions, ax=ax)
ax.set_title('Fraction of Gapes by JL (no amplitude info)')
ax.set_xlim(0,1)
ax.set_xlabel('Fraction of Gapes')
ax.set_ylabel('Count')
fig.savefig(os.path.join(pred_plot_dir, 'gape_fraction_hist.png'))
plt.close(fig)

##############################
# Recreate gape arrays from dataframes to make plotting easier

# Convert peak_inds to actual inds
fin_peak_inds = []
for this_trial_id, this_peak_inds in zip(single_trial_ids, gape_peak_inds):
    this_fin_ind = [(*this_trial_id, int(x)) for x in this_peak_inds]
    fin_peak_inds.extend(this_fin_ind)

fin_peak_frame = pd.DataFrame(
        fin_peak_inds, 
        columns = ['session_number','taste','trial','segment_center']
        )
    

JL_gape_shape = (4,30,7000)
gape_array_list = []
for session_num in tqdm(fin_peak_frame.session_number.unique()):
    this_peak_frame = fin_peak_frame.loc[fin_peak_frame.session_number == session_num]
    gape_array = np.zeros(JL_gape_shape)
    inds = this_peak_frame[['taste','trial','segment_center']]
    for taste, trial, segment_center in inds.values:
        gape_array[taste,trial,segment_center] = 1
    gape_array_list.append(gape_array)

x_vec = np.arange(-2000, 5000)

# Plot gapes for all trials
for i, (gape_array, basename) in enumerate(zip(gape_array_list, basenames)):
    fig, ax = plt.subplots(4,1,sharex=True,sharey=True)
    this_tastes = taste_list[i]
    for taste in range(4):
        this_array = gape_array[taste]
        array_inds = np.where(this_array)
        ax[taste].scatter(array_inds[1], array_inds[0], s=1)
        # ax[taste].imshow(gape_array[taste],aspect='auto',
        #                  cmap='gray',vmin=0,vmax=1,)
        ax[taste].set_title(f'{this_tastes[taste]}')
    fig.suptitle(basename)
    fig.savefig(os.path.join(pred_plot_dir, basename + '_JL_gapes_no_amp.png'))
    plt.close(fig)

# Also make plots for mean gapes
mean_gapes = np.mean(np.array(gape_array_list),axis=2)

kern_len = 250
kernel = np.ones(kern_len)/kern_len
fig, ax = plt.subplots(len(mean_gapes),1,sharex=True,sharey=True,
                       figsize = (5, len(mean_gapes)*3))
for i, (this_mean_gape, basename) in enumerate(zip(mean_gapes, basenames)):
    this_tastes = taste_list[i]
    for taste in range(4):
        taste_mean_gape = this_mean_gape[taste]
        smooth_gape = np.convolve(taste_mean_gape, kernel, mode='same')
        ax[i].plot(x_vec, smooth_gape, label=f'{this_tastes[taste]}')
    ax[i].set_title(basename)
    ax[i].axvline(0, color='r', linestyle='--')
    ax[i].legend(loc='upper right')
    ax[i].set_xlim(x_vec[0] + 2*kern_len, x_vec[-1] - 2*kern_len)
fig.suptitle('Mean Gapes by JL')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
fig.savefig(os.path.join(pred_plot_dir, 'mean_gapes_JL_no_amp.png'))
plt.close(fig)

##############################
# BSA Predictions 
##############################
wanted_bsa_p_list = []
for i, this_row in all_data_frame.iterrows():
    bsa_p_array = this_row['bsa_p']
    wanted_bsa_p_list.append(bsa_p_array)
wanted_bsa_p_list = np.array(wanted_bsa_p_list).astype('int')

def bsa_to_pred(x):
    if np.logical_and(x>=6, x<11):
        return 1
    elif x>=11:
        return 2
    else:
        return 0

bsa_pred = np.vectorize(bsa_to_pred)(wanted_bsa_p_list)

# Plot predictions
bsa_pred_plot_dir = os.path.join(plot_dir, 'prediction_plots', 'bsa')
if not os.path.exists(bsa_pred_plot_dir):
    os.makedirs(bsa_pred_plot_dir)

for i, this_pred in enumerate(bsa_pred):
    fig, ax = plt.subplots(4,1,sharex=True,sharey=True)
    this_tastes = taste_list[i]
    for taste in range(4):
        im = ax[taste].imshow(this_pred[taste],aspect='auto',
                         cmap='jet', vmin=0, vmax=2)
        ax[taste].set_ylabel(f'{this_tastes[taste]}')
    cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    basename = basenames[i]
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(basename)
    fig.savefig(os.path.join(bsa_pred_plot_dir, basename + '_BSA_pred.png'),
                bbox_inches='tight')
    plt.close(fig)

