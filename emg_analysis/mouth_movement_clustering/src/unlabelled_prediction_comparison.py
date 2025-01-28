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
from ast import literal_eval
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

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
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

pred_plot_dir = os.path.join(plot_dir, 'prediction_plots')


##############################
# Specificitiy of annotations 
##############################
merge_gape_pal_path = os.path.join(artifact_dir, 'merge_gape_pal.pkl')
merge_gape_pal = pd.read_pickle(merge_gape_pal_path)

feature_names_path = os.path.join(artifact_dir, 'merge_gape_pal_feature_names.npy')
feature_names = np.load(feature_names_path)

# Load scored_df anew to have the updated codes and event_types
# scored_df = merge_gape_pal[merge_gape_pal.scored == True]
scored_df_path = os.path.join(artifact_dir, 'scored_df.pkl')
scored_df = pd.read_pickle(scored_df_path)

###############
g = sns.displot(
        data = scored_df.loc[scored_df.event_type != 'lateral tongue protrusion'],
        x = 'segment_center',
        row = 'event_type',
        hue = 'event_type',
        legend = False,
        )
g.fig.suptitle('Histogram of Segment Centers for Annoteted Gapes and MTMs')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(
        os.path.join(pred_plot_dir, 'annotated_segment_center_hist.png'),
        bbox_inches='tight')
plt.close()

_, max_y = g.axes[0][0].get_ylim()
###############


# Over-write with updated event coes and types
scored_df['event_type'] = scored_df['updated_event_type']
scored_df['event_codes'] = scored_df['updated_codes']

# Correct event_types
types_to_drop = ['to discuss', 'other', 'unknown mouth movement','out of view']
scored_df = scored_df[~scored_df.event_type.isin(types_to_drop)]

# Remap event_types
event_type_map = {
        'mouth movements' : 'mouth or tongue movement',
        'tongue protrusion' : 'mouth or tongue movement',
        'mouth or tongue movement' : 'mouth or tongue movement',
        'lateral tongue movement' : 'lateral tongue protrusion',
        'lateral tongue protrusion' : 'lateral tongue protrusion',
        'gape' : 'gape',
        'no movement' : 'no movement',
        'nothing' : 'no movement',
        }

scored_df['event_type'] = scored_df['event_type'].map(event_type_map)
scored_df['event_codes'] = scored_df['event_type'].astype('category').cat.codes
scored_df['is_gape'] = (scored_df['event_type'] == 'gape')*1

scored_df.dropna(subset=['event_type'], inplace=True)

# plt.imshow(np.stack(scored_df.features.values), aspect='auto', interpolation='none')
# plt.xticks(np.arange(len(feature_names)), feature_names, rotation=90)
# plt.tight_layout()
# plt.show()

# Generate leave-one-animal-out predictions
# Convert 'lateral tongue protrustion' and 'no movement' to 'other'

# Drop LTPs for now as amplitude information of no_movement
# might be becoming confused by merging with LTPs

scored_df = scored_df[scored_df.event_type != 'lateral tongue protrusion']

# scored_df['event_type'] = scored_df['event_type'].replace(
#         ['lateral tongue protrusion','no movement'],
#         'other'
#         )

scored_df['event_type'] = scored_df['event_type'].replace(
        ['mouth or tongue movement'],
        'MTMs'
        )

event_code_dict = {
        'gape' : 1,
        'MTMs' : 2,
        # 'other' : 0,
        'no movement' : 0,
        }

scored_df['event_codes'] = scored_df['event_type'].map(event_code_dict)

# Plot histogram of segment centers for gape and MTMs

# plot_df = scored_df[scored_df.event_type.isin(['gape','MTMs'])]

g = sns.displot(
        data = scored_df,
        x = 'segment_center',
        row = 'event_type',
        hue = 'event_type',
        legend = False,
        )
for this_ax in g.axes.flatten():
    this_ax.set_ylim([0, max_y])
g.fig.suptitle('Histogram of Segment Centers for Annoteted Gapes and MTMs')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(
        os.path.join(pred_plot_dir, 'annotated_segment_center_hist_updated.png'),
        bbox_inches='tight')
plt.close()

# Calculate specificity of annotations
time_bins = [[-2000, 0], [0, 2000], [2000, 4000]]
x_vec = np.arange(-2000, 5000)
x_inds = np.stack([np.logical_and(x_vec >= x[0], x_vec < x[1]) for x in time_bins])

##############################
# Gapes by JL
##############################

pred_plot_dir = os.path.join(plot_dir, 'prediction_plots', 'jl_qda')
if not os.path.exists(pred_plot_dir):
    os.makedirs(pred_plot_dir)

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

gape_array_list = np.array(gape_array_list)


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
mean_gapes = np.mean(gape_array_list,axis=2)

fig, ax = plt.subplots(len(mean_gapes),1,sharex=True,sharey=True,
                       figsize = (5, len(mean_gapes)*1.5))
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

###############
# Calculate counts of events in each timebin 

binned_events_list = []
for this_inds in x_inds:
    binned_events = gape_array_list[..., this_inds]
    binned_events_list.append(binned_events)

mean_rate_list = np.array([np.mean(x, axis=-1) for x in binned_events_list])

# Calculate fraction of total gapes in 0-2000 ms bin
pred_specificity = mean_rate_list[1].sum(axis=(1,2)) / np.sum(mean_rate_list, axis=(0,2,3))
mean_pred_specificity = pred_specificity.mean()

pred_specificity_df = pd.DataFrame(
        dict(
            algorithm = 'JL',
            event_type = 'gapes',
            session_ind = np.arange(len(pred_specificity)),
            specificity = pred_specificity,
            ),
        )
###############

df_inds = np.array(list(np.ndindex(mean_rate_list.shape)))
mean_rate_df = pd.DataFrame(
        dict(
            time_bin = df_inds[:,0],
            session = df_inds[:,1],
            taste = df_inds[:,2],
            trial = df_inds[:,3],
            mean_rate = mean_rate_list.flatten()
            )
        )

g = sns.boxplot(
        x = 'time_bin',
        y = 'mean_rate',
        data = mean_rate_df,
        dodge = True
        )
g.set_xticklabels(time_bins)
plt.title('Mean Rate of Gapes by Time Bin\n' +\
        f'Gape Specificity: {mean_pred_specificity:.2f}')
plt.savefig(os.path.join(pred_plot_dir, 'mean_rate_gapes.png'))
plt.close()


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
ax.set_xlabel('Fraction of Gapes per trial')
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

gape_array_list = np.array(gape_array_list)
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

###############
# Calculate mean_rate of events in time-bins
binned_events_list = []
for this_inds in x_inds:
    binned_events = gape_array_list[..., this_inds]
    binned_events_list.append(binned_events)

mean_rate_list = np.array([np.mean(x, axis=-1) for x in binned_events_list])

# Calculate fraction of total gapes in 0-2000 ms bin
pred_specificity = mean_rate_list[1].sum(axis=None) / np.sum(mean_rate_list, axis=None)

this_specificity_df = pd.DataFrame(
        dict(
            algorithm = 'JL_no_amp',
            event_type = 'gapes',
            specificity = pred_specificity,
            ),
        index = [0]
        )
pred_specificity_df = pd.concat([pred_specificity_df, this_specificity_df])

df_inds = np.array(list(np.ndindex(mean_rate_list.shape)))
mean_rate_df = pd.DataFrame(
        dict(
            time_bin = df_inds[:,0],
            session = df_inds[:,1],
            taste = df_inds[:,2],
            trial = df_inds[:,3],
            mean_rate = mean_rate_list.flatten()
            )
        )

g = sns.boxplot(
        x = 'time_bin',
        y = 'mean_rate',
        data = mean_rate_df,
        dodge = True
        )
g.set_xticklabels(time_bins)
plt.title('Mean Rate of Gapes by Time Bin\n' +\
        f'Gape Specificity: {pred_specificity:.2f}')
plt.savefig(os.path.join(pred_plot_dir, 'mean_rate_gapes.png'))
plt.close()

###############

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

bsa_event_map = {
        0 : 'nothing',
        1 : 'gapes',
        2 : 'MTMs',
        }

bsa_pred = np.vectorize(bsa_to_pred)(wanted_bsa_p_list)

# Plot predictions
bsa_pred_plot_dir = os.path.join(plot_dir, 'prediction_plots', 'bsa')
if not os.path.exists(bsa_pred_plot_dir):
    os.makedirs(bsa_pred_plot_dir)

# Create cmap for BSA predictions
cmap = plt.cm.get_cmap('jet', 3)

for i, this_pred in enumerate(bsa_pred):
    fig, ax = plt.subplots(4,1,sharex=True,sharey=True)
    this_tastes = taste_list[i]
    for taste in range(4):
        im = ax[taste].pcolormesh(
                x_vec, np.arange(30), 
                this_pred[taste],
                  cmap=cmap,vmin=0,vmax=2,)
        ax[taste].set_ylabel(f'{this_tastes[taste]}' + '\nTrial #')
    ax[-1].set_xlabel('Time (ms)')
    cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.5,1,1.5])
    cbar.set_ticklabels(['nothing','gape','MTMs'])
    basename = basenames[i]
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(basename)
    fig.savefig(os.path.join(bsa_pred_plot_dir, basename + '_BSA_pred.png'),
                bbox_inches='tight')
    plt.close(fig)

# Mean BSA predictions
# First break down by category
bsa_pred_separated = []
for this_pred in bsa_pred:
    this_pred_separated = [this_pred == x for x in range(3)]
    this_pred_separated*=1
    bsa_pred_separated.append(this_pred_separated)

bsa_pred_separated = np.array(bsa_pred_separated)

binned_events_list = []
for this_inds in x_inds:
    binned_events = bsa_pred_separated[..., this_inds]
    binned_events_list.append(binned_events)

mean_rate_list = np.array([np.mean(x, axis=-1) for x in binned_events_list])
mean_rate_inds = np.stack(list(np.ndindex(mean_rate_list.shape)))

mean_rate_df = pd.DataFrame(
        dict(
            time_bin = mean_rate_inds[:,0],
            session = mean_rate_inds[:,1],
            event_type = mean_rate_inds[:,2],
            taste = mean_rate_inds[:,3],
            trial = mean_rate_inds[:,4],
            mean_rate = mean_rate_list.flatten()
            )
        )

mean_rate_df['time_bin_str'] = [str(time_bins[x]) for x in mean_rate_df.time_bin]
mean_rate_df['event_str'] = mean_rate_df.event_type.map(bsa_event_map)
mean_rate_df['taste_str'] = [taste_list[x][y] for x,y in zip(mean_rate_df.session, mean_rate_df.taste)]

# Calculate specificity per movement type
bin_sums = mean_rate_df.groupby(
        ['event_type', 'time_bin', 'event_str', 'time_bin_str','session']).sum()['mean_rate']
bin_sums = pd.DataFrame(bin_sums)
bin_sums.reset_index(inplace=True)

spec_list = []
event_name_list = []
for this_type in bin_sums['event_type'].unique():
    this_frame = bin_sums.loc[bin_sums.event_type == this_type]
    this_spec = this_frame.loc[this_frame.time_bin == 1].mean_rate.values / \
            this_frame.groupby('session').sum().mean_rate.values
    spec_list.append(this_spec)
    event_name_list.extend(this_frame.event_str.unique())

spec_list = [np.round(x, 3) for x in spec_list]
mean_spec_list = [np.mean(x) for x in spec_list]
mean_bsa_spec_dict = dict(zip(event_name_list, mean_spec_list))

event_list_long = [[x]*len(y) for x,y in zip(event_name_list, spec_list)]
session_list= [np.arange(len(y)) for y in spec_list]
session_list_long = [x for y in session_list for x in y]
event_list_long = [x for y in event_list_long for x in y]
spec_list_long = [x for y in spec_list for x in y]

g = sns.catplot(
        data = mean_rate_df,
        x = 'time_bin_str',
        y = 'mean_rate',
        col = 'event_str',
        row = 'taste_str',
        kind = 'box',
        dodge = True
        )
plt.suptitle('Mean Rate of Events by Time Bin\n' +\
         f'Specificities : {mean_bsa_spec_dict}')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig(os.path.join(bsa_pred_plot_dir, 'mean_rate_events.png'))
plt.close()

# Add to pred_specificity_df
this_spec_df = pd.DataFrame(
        dict(
            session_ind = session_list_long,
            event_type = event_list_long,
            specificity = spec_list_long,
            )
        )
this_spec_df['algorithm'] = 'BSA'

pred_specificity_df = pd.concat([pred_specificity_df, this_spec_df])

##############################
# Calculate fraction of non-null events in 0-2000 ms bin 
# Calculate specificity for each event-type separately

# # Drop 'nothing' events
# bsa_binary_pred = bsa_pred > 0
# 
# binned_events_list = []
# for this_inds in x_inds:
#     binned_events = bsa_binary_pred[..., this_inds]
#     binned_events_list.append(binned_events)

# mean_rate_list = np.array([np.mean(x, axis=-1) for x in binned_events_list])
# mean_rate_list = np.swapaxes(mean_rate_list, 1,2)
# 
# df_inds = np.array(list(np.ndindex(mean_rate_list.shape)))
# mean_rate_df = pd.DataFrame(
#         dict(
#             event_type = df_inds[:,0],
#             time_bin = df_inds[:,1],
#             session = df_inds[:,2],
#             taste = df_inds[:,3],
#             trial = df_inds[:,4],
#             mean_rate = mean_rate_list.flatten()
#             )
#         )
# mean_rate_df['event_type_name'] = mean_rate_df['event_type'].map(bsa_event_map)
# 
# g = sns.catplot(
#         x = 'time_bin',
#         y = 'mean_rate',
#         row = 'event_type_name',
#         col = 'session',
#         data = mean_rate_df,
#         dodge = True,
#         kind = 'box',
#         )
# g.set_xticklabels([str(x) for x in time_bins])
# specificity_str = str([np.round(x,3) for x in pred_specificity])
# plt.suptitle('Mean Rate of Events by Time Bin\n' +\
#         f'Specificities : {specificity_str}')
# plt.subplots_adjust(top = 0.9)
# plt.savefig(os.path.join(bsa_pred_plot_dir, 'mean_rate_events.png'))
# plt.close()

# # Calculate fraction of total gapes in 0-2000 ms bin
# # pred_specificity = mean_rate_list[1].sum(axis=None) / np.sum(mean_rate_list, axis=None)
# pred_specificity = [x[1].sum(axis=None) / np.sum(x, axis=None) \
#         for x in mean_rate_list]
# 
# this_specificity_df = pd.DataFrame(
#         dict(
#             algorithm = 'BSA',
#             event_type = 'gapes+MTMs',
#             specificity = pred_specificity,
#             ),
#         index = [0]
#         )
# pred_specificity_df = pd.concat([pred_specificity_df, this_specificity_df])

###############



##############################
# pred_specificity = [x.rate.iloc[1] / x.rate.sum() for x in mean_rate_dfs] 
# pred_specificity = [np.round(x,2) for x in pred_specificity]
# 
# this_specificity_df = pd.DataFrame(
#         dict(
#             event_type = ['nothing','gape','MTMs'],
#             specificity = pred_specificity,
#             ),
#         )
# this_specificity_df['algorithm'] = 'BSA'
# pred_specificity_df = pd.concat([pred_specificity_df, this_specificity_df])
# 
# 
# ###############
# 
# mean_bsa_pred = np.mean(bsa_pred_separated, axis=-2)
# 
# 
# fig, ax = plt.subplots(*mean_bsa_pred.shape[:2],
#                        sharex=True,sharey=True,
#                        figsize = (10,10)
#                        )
# for i in range(mean_bsa_pred.shape[0]):
#     for j in range(mean_bsa_pred.shape[1]):
#         ax[i,j].plot(mean_bsa_pred[i,j].T)
#         ax[i,j].set_title(f'Session {i}, {bsa_event_map[j]}')
# fig.suptitle('Mean BSA Predictions')
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# fig.savefig(os.path.join(bsa_pred_plot_dir, 'mean_bsa_pred.png'))
# plt.close(fig)

##############################
# XGB Predictions 
##############################

# Train on scored data but predict on all data
unique_animals = scored_df.animal_num.unique()

for i, this_session in enumerate(tqdm(unique_animals)):
    # Leave out this session
    train_df = scored_df[scored_df.animal_num != this_session]
    test_df = merge_gape_pal[merge_gape_pal.animal_num == this_session]

    # Train model
    X_train = np.stack(train_df.features.values)
    y_train = train_df.event_codes.values

    # Calculate sample weights and normalize weight for each class
    class_weights = train_df.event_codes.value_counts(normalize=True)
    inv_class_weights = 1 / class_weights
    sample_weights = inv_class_weights.loc[y_train].values 

    X_pred = np.stack(test_df.features.values)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_pred)

    merge_gape_pal.loc[merge_gape_pal.animal_num == this_session, 'xgb_pred'] = y_pred

    # For sanity checking, check accuracy on y_train
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred) 
    print(f'Train Accuracy: {train_acc:.2f}')

# Convert xgb_pred to int
merge_gape_pal['xgb_pred'] = merge_gape_pal['xgb_pred'].astype('int')

# Add event name to xgb_pred
merge_gape_pal['xgb_pred_event'] = merge_gape_pal['xgb_pred'].map(bsa_event_map)

###############
# Convert to array so downstream processing can be same as BSA
JL_gape_shape = (4,30,7000)
xgb_pred_array_list = []
for session_num in tqdm(merge_gape_pal.session_ind.unique()):
    this_peak_frame = merge_gape_pal.loc[merge_gape_pal.session_ind == session_num] 
    event_array = np.zeros(JL_gape_shape)
    inds = this_peak_frame[['taste','trial','segment_bounds']]
    pred_vals = this_peak_frame['xgb_pred'].values
    for ind, pred_val in enumerate(pred_vals):
        taste, trial, segment_bounds = inds.iloc[ind]
        updated_bounds = segment_bounds.copy()
        updated_bounds += 2000
        this_pred = pred_vals[ind]
        event_array[taste,trial,updated_bounds[0]:updated_bounds[1]] = this_pred

    xgb_pred_array_list.append(event_array)

xgb_pred_array_list = np.array(xgb_pred_array_list)
# Convert to int
xgb_pred_array_list = xgb_pred_array_list.astype('int')

# Plot
xgb_pred_plot_dir = os.path.join(plot_dir, 'prediction_plots', 'xgb')
if not os.path.exists(xgb_pred_plot_dir):
    os.makedirs(xgb_pred_plot_dir)

# Create cmap for BSA predictions
cmap = plt.cm.get_cmap('jet', 3)

for i, this_pred in enumerate(xgb_pred_array_list):
    fig, ax = plt.subplots(4,1,sharex=True,sharey=True)
    this_tastes = taste_list[i]
    for taste in range(4):
        im = ax[taste].pcolormesh(
                x_vec, np.arange(30), 
                this_pred[taste],
                  cmap=cmap,vmin=0,vmax=2,)
        ax[taste].set_ylabel(f'{this_tastes[taste]}' + '\nTrial #')
    ax[-1].set_xlabel('Time (ms)')
    cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.5,1,1.5])
    cbar.set_ticklabels(['nothing','gape','MTMs'])
    basename = basenames[i]
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.suptitle(basename)
    fig.savefig(os.path.join(xgb_pred_plot_dir, basename + '_xgb_pred.png'),
                bbox_inches='tight')
    plt.close(fig)


# Also make plots for mean events

xgb_pred_separated = []
for this_pred in xgb_pred_array_list:
    this_pred_separated = [this_pred == x for x in range(3)]
    this_pred_separated*=1
    xgb_pred_separated.append(this_pred_separated)

xgb_pred_separated = np.array(xgb_pred_separated)*1
mean_xgb_pred_separated = np.mean(xgb_pred_separated, axis=-2)

##############################
##############################
binned_events_list = []
for this_inds in x_inds:
    binned_events = xgb_pred_separated[..., this_inds]
    binned_events_list.append(binned_events)

mean_rate_list = np.array([np.mean(x, axis=-1) for x in binned_events_list])
mean_rate_inds = np.stack(list(np.ndindex(mean_rate_list.shape)))

mean_rate_df = pd.DataFrame(
        dict(
            time_bin = mean_rate_inds[:,0],
            session = mean_rate_inds[:,1],
            event_type = mean_rate_inds[:,2],
            taste = mean_rate_inds[:,3],
            trial = mean_rate_inds[:,4],
            mean_rate = mean_rate_list.flatten()
            )
        )

mean_rate_df['time_bin_str'] = [str(time_bins[x]) for x in mean_rate_df.time_bin]
mean_rate_df['event_str'] = mean_rate_df.event_type.map(bsa_event_map)
mean_rate_df['taste_str'] = [taste_list[x][y] for x,y in zip(mean_rate_df.session, mean_rate_df.taste)]

# Calculate specificity per movement type
bin_sums = mean_rate_df.groupby(
        ['event_type', 'time_bin', 'event_str', 'time_bin_str','session']).sum()['mean_rate']
bin_sums = pd.DataFrame(bin_sums)
bin_sums.reset_index(inplace=True)

spec_list = []
event_name_list = []
for this_type in bin_sums['event_type'].unique():
    this_frame = bin_sums.loc[bin_sums.event_type == this_type]
    this_spec = this_frame.loc[this_frame.time_bin == 1].mean_rate.values / \
            this_frame.groupby('session').sum().mean_rate.values
    spec_list.append(this_spec)
    event_name_list.extend(this_frame.event_str.unique())

spec_list = [np.round(x, 3) for x in spec_list]
mean_spec_list = [np.mean(x) for x in spec_list]
mean_xgb_spec_dict = dict(zip(event_name_list, mean_spec_list))

event_list_long = [[x]*len(y) for x,y in zip(event_name_list, spec_list)]
session_list= [np.arange(len(y)) for y in spec_list]
session_list_long = [x for y in session_list for x in y]
event_list_long = [x for y in event_list_long for x in y]
spec_list_long = [x for y in spec_list for x in y]

g = sns.catplot(
        data = mean_rate_df,
        x = 'time_bin_str',
        y = 'mean_rate',
        col = 'event_str',
        row = 'taste_str',
        kind = 'box',
        dodge = True
        )
plt.suptitle('Mean Rate of Events by Time Bin\n' +\
         f'Specificities : {mean_xgb_spec_dict}')
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig(os.path.join(xgb_pred_plot_dir, 'mean_rate_events.png'))
plt.close()

# Add to pred_specificity_df
this_spec_df = pd.DataFrame(
        dict(
            session_ind = session_list_long,
            event_type = event_list_long,
            specificity = spec_list_long,
            )
        )
this_spec_df['algorithm'] = 'XGB'

pred_specificity_df = pd.concat([pred_specificity_df, this_spec_df])

# Write out specificity_df to artifacts
pred_specificity_df.to_csv(
        os.path.join(artifact_dir, 'pred_specificity_df.csv')
        )

##############################
##############################
# Plot paired specificity data for BSA and XGB

plot_pred_spec_df = pred_specificity_df.loc[pred_specificity_df.event_type != 'nothing']
plot_pred_spec_df = plot_pred_spec_df.loc[plot_pred_spec_df.algorithm.isin(['BSA','XGB', 'JL'])]

event_type_list = [x[1] for x in list(plot_pred_spec_df.groupby('event_type'))]
event_type_list = [[x[1] for x in list(this_frame.groupby('algorithm'))] for this_frame in event_type_list]

event_type_list_values = [[x.specificity for x in y] for y in event_type_list]
event_type_events = [x[0].event_type.unique()[0] for x in event_type_list]
ttest_list = [ttest_rel(*x) for x in event_type_list_values]
pval_list = [np.round(x.pvalue,3) for x in ttest_list]
pval_dict = dict(zip(event_type_events, pval_list))

fig, ax = plt.subplots(1,2, sharex=True, sharey=True,
                       figsize = (5,5))
for i, this_type in enumerate(plot_pred_spec_df.event_type.unique()):
    ax[i].set_title(this_type)
    this_frame = plot_pred_spec_df.loc[plot_pred_spec_df.event_type == this_type]
    for session_ind, df in list(this_frame.groupby('session_ind')):
        bsa_value = df.loc[df.algorithm=='BSA'].specificity.values[0]
        xgb_value = df.loc[df.algorithm=='XGB'].specificity.values[0]
        try:
            jl_value = df.loc[df.algorithm=='JL'].specificity.values[0]
        except:
            jl_value = None
        ax[i].plot(['BSA','XGB'], [bsa_value, xgb_value], '-o', color = 'k')
        ax[i].scatter('JL-QDA', jl_value, color = 'k') 
        ax[i].set_xlabel('Algorithm')
ax[0].set_ylabel('Prediction Specificity')
fig.suptitle('Changes in specificty between algorithms\n' +\
        f'pvalues for BSA-XGB comparison = {pval_dict}')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'prediction_plots', 'specificity_comparison.png'),
            bbox_inches = 'tight')
plt.close(fig)
# plt.show()
        
##############################
##############################

fig, ax = plt.subplots(*mean_xgb_pred_separated.shape[:2],
                       figsize = (10,10),
                       sharex=True,sharey=True,
                       )
cut_len = 250
for i in range(mean_xgb_pred_separated.shape[0]):
    for j in range(mean_xgb_pred_separated.shape[1]):
        ax[i,j].plot(x_vec[cut_len:-cut_len], mean_xgb_pred_separated[i,j].T[cut_len:-cut_len])
        ax[i,j].set_title(f'Session {i}, {bsa_event_map[j]}')
fig.suptitle('Mean XGB Predictions')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
fig.savefig(os.path.join(xgb_pred_plot_dir, 'mean_xgb_pred.png'))
plt.close(fig)

############################################################
###############
# Make joint plots of bsa and xgb predictions
xgb_bsa_pred_plot_dir = os.path.join(plot_dir, 'prediction_plots', 'xgb_bsa')
if not os.path.exists(xgb_bsa_pred_plot_dir):
    os.makedirs(xgb_bsa_pred_plot_dir)

bsa_event_map = {
        0 : 'nothing',
        1 : 'gapes',
        2 : 'MTMs',
        }

event_color_map = {
        # 0 : '#ffffff',
        # # 1 : '#17becf',
        # 1 : '#843c39',
        # 2 : '#1f77b4',
        0 : '#D1D1D1',
        # 1 : '#17becf',
        1 : '#EF8636',
        2 : '#3B75AF',
        }

# Create segmented colormap
from matplotlib.colors import ListedColormap
cmap = ListedColormap(list(event_color_map.values()), name = 'NBT_cmap')

# Create cmap for BSA predictions
# cmap = plt.cm.get_cmap('jet', 3)

for i, (this_xgb_pred, this_bsa_pred) in enumerate(zip(xgb_pred_array_list, bsa_pred)):
    fig, ax = plt.subplots(4,2,sharex=True,sharey=True)
    this_tastes = taste_list[i]
    for taste in range(4):
        im = ax[taste,0].pcolormesh(
                x_vec, np.arange(30), 
                this_xgb_pred[taste],
                  cmap=cmap,vmin=0,vmax=2,)
        ax[taste,0].set_ylabel(f'{this_tastes[taste]}' + '\nTrial #')
        im = ax[taste,1].pcolormesh(
                x_vec, np.arange(30), 
                this_bsa_pred[taste],
                  cmap=cmap,vmin=0,vmax=2,)
        ax[0,0].set_title('XGB')
        ax[0,1].set_title('BSA')
        wanted_xlims = [-1000, 3000]
        ax[taste,0].set_xlim(wanted_xlims)
        ax[taste,1].set_xlim(wanted_xlims)
    ax[-1,0].set_xlabel('Time (ms)')
    ax[-1,1].set_xlabel('Time (ms)')
    cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.5,1,1.5])
    cbar.set_ticklabels(['nothing','gape','MTMs'])
    basename = basenames[i]
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(basename)
    fig.savefig(os.path.join(xgb_bsa_pred_plot_dir, basename + '_xgb_bsa_pred.png'),
                bbox_inches='tight', dpi = 300)
    plt.close(fig)

##############################
# Plot emg envelope traces colored by predictions
trial_group_list = list(merge_gape_pal.groupby(['session_ind','taste','trial']))

env_plot_dir = os.path.join(plot_dir, xgb_pred_plot_dir, 'env_traces')
if not os.path.exists(env_plot_dir):
    os.makedirs(env_plot_dir)

n_plots = 100
inds = np.random.choice(len(trial_group_list), n_plots, replace=False)

from matplotlib.lines import Line2D

cmap = plt.cm.get_cmap('tab10')
custom_lines = [Line2D([0],[0], color = cmap(x), lw=2) for x in range(3)]

for i in tqdm(inds):
    this_group_inds, this_group = trial_group_list[i]
    session_num, this_taste, this_trial = this_group_inds

    fig, ax = plt.subplots()
    for i, this_row in this_group.iterrows():
        dat = this_row['segment_raw']
        bounds = this_row['segment_bounds']
        ax.plot(np.arange(*bounds), dat,
                color = cmap(this_row['xgb_pred']),
                linewidth = 2)
        ax.legend(custom_lines, list(bsa_event_map.values()))
    ax.set_title(f'Session {session_num}, {this_taste}, Trial {this_trial}')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('EMG Envelope')
    fig.savefig(os.path.join(env_plot_dir, f'{session_num}_{this_taste}_{this_trial}_env.png'))
    plt.close(fig)
