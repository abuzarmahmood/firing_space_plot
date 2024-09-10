"""
Streamlined script to generate training dataset for the model.
"""
############################################################
############################################################
# get_data.py
############################################################
############################################################

import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from scipy.spatial.distance import mahalanobis

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
code_dir = os.path.join(base_dir, 'src')
sys.path.append(code_dir)  # noqa
from utils.extract_scored_data import return_taste_orders, process_scored_data  # noqa
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            JL_process,
                                            # gen_segment_frame,
                                            parse_segment_dat_list,
                                            parse_gapes_Li,
                                            threshold_movement_lengths,
                                            )  # noqa

# artifact_dir = os.path.join(base_dir, 'artifacts')
artifact_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering/src/classifier_pipeline/artifacts'
if not os.path.isdir(artifact_dir):
    os.makedirs(artifact_dir)

############################################################
# Helper functions
############################################################
def get_scored_data_paths(data_dir):
    """
    Get scored data from data_dir

    Inputs:
        data_dir (str): Directory containing data

    Outputs:
        scores_path_df (pd.DataFrame): Dataframe containing paths to scored data
    """
    scored_data_paths = glob(os.path.join(
        data_dir, '**', '*scores.csv'), recursive=True)
    scored_data_basenames = [os.path.basename(x).lower().split('.')[
        0] for x in scored_data_paths]
    scored_data_basenames = ["_".join(x.split('_')[:-1])
                             for x in scored_data_basenames]

    scores_path_df = pd.DataFrame({'path': scored_data_paths,
                                   'basename': scored_data_basenames,
                                   'scores': True})
    return scores_path_df

def get_emg_paths(data_dir):
    """
    Get emg paths from data_dir

    Inputs:
        data_dir (str): Directory containing data

    Outputs:
        emg_path_df (pd.DataFrame): Dataframe containing paths to emg data
    """

    emg_output_dirs = sorted(
        glob(os.path.join(data_dir, '*', '*', '*', 'emg_output')))

    # For each day of experiment, load env and table files
    data_subdirs = [glob(os.path.join(x, '*')) for x in emg_output_dirs]
    data_subdirs = [item for sublist in data_subdirs for item in sublist]
    # Make sure that path is a directory
    data_subdirs = [subdir for subdir in data_subdirs if os.path.isdir(subdir)]
    data_subdirs = [x for x in data_subdirs if 'emg' in os.path.basename(x)]
    # Make sure that subdirs are in order
    subdir_basenames = [subdir.split('/')[-3].lower() for subdir in data_subdirs]
    emg_name = [os.path.basename(x) for x in data_subdirs]

    emg_path_df = pd.DataFrame({'path': data_subdirs,
                                'basename': subdir_basenames,
                                'emg_name': emg_name,
                                'emg': True})

    # Only keep row if emg_name == 'emgad'
    emg_path_df['emg_name'] = emg_path_df['emg_name'].str.lower()
    emg_path_df = emg_path_df.loc[emg_path_df.emg_name == 'emgad']

    # Check for BSA output
    bsa_results_dirs = [glob(os.path.join(x, '*BSA_results'))[0]
                        for x in emg_path_df.path]
    emg_path_df['bsa_results_path'] = bsa_results_dirs

    return emg_path_df


def get_h5_paths(data_dir):
    """
    Get h5 paths from data_dir

    Inputs:
        data_dir (str): Directory containing data

    Outputs:
        h5_path_df (pd.DataFrame): Dataframe containing paths to h5 data
    """

    h5_files = glob(os.path.join(data_dir, '**', '*', '*', '*.h5'))
    h5_files = sorted(h5_files)
    h5_basenames = [os.path.basename(x).split('.')[0].lower() for x in h5_files]

    h5_path_df = pd.DataFrame({'path': h5_files,
                               'basename': h5_basenames,
                               'h5': True})

    ##############################
    # Also get info file paths
    h5_dirs = [os.path.dirname(x) for x in h5_files]
    info_file_paths = [glob(os.path.join(x, '*.info'))[0] for x in h5_dirs]
    h5_path_df['info_file_path'] = info_file_paths

    return h5_path_df

def load_raw_data(session_data_df):
    """
    Given session_data_df, use paths to load:
        1- scored data
        2- raw emg data
        3- bsa data
        4- taste orders
        5- info file

    Inputs:
        session_data_df (pd.DataFrame): Dataframe containing paths to data

    Outputs:
        session_data_df (pd.DataFrame): Dataframe with loaded data
    """

    # Unnecessary columns in scored data to delete
    scored_cols_to_delete = [
        'Observation id', 'Observation date', 'Description',
        'Observation duration', 'Observation type',
        'Source', 'Media duration (s)', 'FPS', 'Subject',
        'Behavioral category', 'Media file name', 'Image index',
        'Image file path', 'Comment',
    ]


    scored_data_list = []
    envs_list = []
    bsa_p_list = []
    omega_list = []
    taste_map_list = []
    for ind, row in tqdm(session_data_df.iterrows()):

        ###############
        # scored_data_path = scored_data_paths[ind]
        scored_data_path = row.path_scores
        scored_data = pd.read_csv(scored_data_path)
        scored_data.drop(columns=scored_cols_to_delete, inplace=True)
        # Indicate these scores were from video to differentiate from emg only
        scored_data['scoring_type'] = 'video'
        scored_data_list.append(scored_data)

        ###############
        # tastes x trials x time
        env_path = glob(os.path.join(row.path_emg, '*env.npy'))[0]
        env = np.load(env_path)
        envs_list.append(env)

        ###############
        bsa_p_filelist = glob(os.path.join(row.bsa_results_path, '*p.npy'))
        bsa_omega_filelist = glob(os.path.join(
            row.bsa_results_path, '*omega.npy'))[0]

        # Convert p to array
        # Extract inds from p filenames
        bsa_p_filelist.sort()
        bsa_p_basenames = [os.path.basename(x) for x in bsa_p_filelist]
        bsa_taste_inds = [int(x.split('_')[0][-2:]) for x in bsa_p_basenames]
        bsa_trial_inds = [int(x.split('_')[1][-2:]) for x in bsa_p_basenames]

        bsa_data = (np.stack([np.load(x)
                    for x in bsa_p_filelist]) > 0.1).astype(int)

        # Convert to 1D timeseries
        freq_inds = np.arange(bsa_data.shape[2])
        bsa_data_flat = freq_inds[np.argmax(bsa_data, axis=2)]

        # taste x trial x time
        p_array = np.zeros((
            np.max(bsa_taste_inds)+1,
            np.max(bsa_trial_inds)+1,
            bsa_data_flat.shape[-1]))

        for i, (taste_ind, trial_ind) in enumerate(zip(bsa_taste_inds, bsa_trial_inds)):
            p_array[taste_ind, trial_ind] = bsa_data_flat[i]

        bsa_p_list.append(p_array)

        # Omega
        omega_vec = np.load(bsa_omega_filelist)
        omega_list.append(omega_vec)

        ###############
        info_file_path = row.info_file_path
        info_file = json.load(open(info_file_path, 'r'))
        taste_names = info_file['taste_params']['tastes']
        pal_rankings = info_file['taste_params']['pal_rankings']
        taste_map = dict(zip(taste_names, pal_rankings))
        taste_map_list.append(taste_map)


    # Taste are delivered in a pseudorandom order
    # We need to know the order of taste deliveries for each session
    h5_files = session_data_df.path_h5.tolist()
    taste_order_list = []
    for h5_file in tqdm(h5_files):
        try:
            taste_orders = return_taste_orders([h5_file])
        except:
            taste_orders = None
        taste_order_list.append(taste_orders)

    ############################################################
    # Append to session_data_df and drop any rows with None
    session_data_df['scored_data'] = scored_data_list
    session_data_df['env'] = envs_list
    session_data_df['bsa_p'] = bsa_p_list
    session_data_df['bsa_omega'] = omega_list
    session_data_df['taste_orders'] = taste_order_list
    session_data_df['taste_map'] = taste_map_list

    # session_data_df.basename[merge_df.taste_orders.isna()]
    session_data_df.dropna(inplace=True)
    merge_keep_inds = session_data_df.index
    session_data_df.reset_index(inplace=True, drop=True)

    return session_data_df

def run_JL_process(envs, pre_stim=2000, post_stim=5000): 
    """
    Run Jenn Li process on envs

    Inputs:
        envs (np.array): Array of shape (tastes, trials, time)
         - Contains enveloped emg data
        pre_stim (int): Time before stimulus onset
         - Used to calculate baseline
        post_stim (int): Time after stimulus onset

    Outputs:
        gapes_Li (np.array): Array of shape (tastes, trials, time)
         - Binary array indicating predicted gapes
    """
    gapes_Li = np.zeros(envs.shape)
    inds = list(np.ndindex(envs.shape[:-1]))
    this_day_prestim_dat = envs[..., :pre_stim]

    for this_ind in inds:
        this_trial_dat = envs[this_ind]

        ### Jenn Li Process ###
        # Get peak indices
        gape_peak_inds, time_list = JL_process(
            this_trial_dat,
            this_day_prestim_dat,
            pre_stim,
            post_stim,
        )

        if gape_peak_inds is not None:
            gapes_Li[this_ind][gape_peak_inds] = 1

    return gapes_Li

def run_AM_process(envs, pre_stim=2000):
    """
    Run AM process on envs

    Inputs:
        envs (np.array): Array of shape (tastes, trials, time)

    Outputs:
        segment_dat_list (list): List of segment data
    """
    this_day_prestim_dat = envs[..., :pre_stim]
    mean_prestim = np.mean(this_day_prestim_dat, axis=None)

    segment_dat_list = []
    inds = list(np.ndindex(envs.shape[:-1]))

    for this_ind in inds:
        this_trial_dat = envs[this_ind]

        (
            segment_starts,
            segment_ends,
            segment_dat,
            filtered_segment_dat
        ) = extract_movements(
            this_trial_dat, size=200)

        # Threshold movement lengths
        segment_starts, segment_ends, segment_dat = threshold_movement_lengths(
            segment_starts, segment_ends, filtered_segment_dat,
            min_len=50, max_len=500)

        assert len(segment_starts) == len(segment_ends) == len(segment_dat), \
            'Mismatch in segment lengths'

        (feature_array,
         feature_names,
         segment_dat,
         segment_starts,
         segment_ends,
         norm_interp_segment_dat,
         ) = extract_features(
            segment_dat, segment_starts, segment_ends, mean_prestim = mean_prestim)

        assert len(feature_array) == len(segment_dat) == len(segment_starts) == len(segment_ends), \
            'Mismatch in feature array lengths'

        segment_bounds = list(zip(segment_starts, segment_ends))
        merged_dat = [feature_array, segment_dat, norm_interp_segment_dat, segment_bounds]
        segment_dat_list.append(merged_dat)

    return segment_dat_list, feature_names

############################################################
# Make sure we have 1) scored data, 2) emg env file, and 3) h5 files for each session
############################################################

data_dir = '/media/fastdata/Natasha_classifier_data'
scores_path_df = get_scored_data_paths(data_dir)
emg_path_df = get_emg_paths(data_dir)
h5_path_df = get_h5_paths(data_dir)

##############################
# Merge dataframes
# Keep all data in a "session_level" dataframe for organization
session_data_df = pd.merge(scores_path_df, emg_path_df, on='basename',
                    suffixes=('_scores', '_emg'),
                    how='outer')
session_data_df = pd.merge(session_data_df, h5_path_df, on='basename',
                    how='outer')
session_data_df = session_data_df.rename(columns={'path': 'path_h5'})

# Add more identifying information to session_data_df
session_data_df['animal_num'] = [
    x.split('_')[0] for x in session_data_df.basename.values]
session_data_df['session_ind'] = session_data_df.index

# Drop any rows with NaN
session_data_df.dropna(inplace=True)
session_data_df.reset_index(inplace=True, drop=True)

############################################################
# Load raw data
############################################################
session_data_df = load_raw_data(session_data_df)

# Confirm that there are as many 'trial start's in scored data
# as trials in the ephys data
# This is needed to ensure scoring and emg can be aligned later
for ind, row in session_data_df.iterrows():
    scored_data = row.scored_data
    taste_orders = row.taste_orders[0]
    n_trial_starts = len(
        scored_data.loc[scored_data.Behavior == 'trial start'])
    n_trials = len(taste_orders)
    if n_trials != n_trial_starts:
        print('Mismatch in trials for {}'.format(row.basename))

############################################################
# Add scored data to session_data_df
############################################################
fin_score_table_list = []
for ind, row in tqdm(session_data_df.iterrows()):
    scored_data = session_data_df.loc[ind, 'scored_data'].copy()
    taste_orders = session_data_df.loc[ind, 'taste_orders'][0].copy()
    # Convert scored data into more amenable format
    # See artifacts/explanatory_plots/scored_data_to_processed_score_table.png
    fin_scored_table = process_scored_data(scored_data, taste_orders)
    fin_score_table_list.append(fin_scored_table)

##############################
# Also add nothing scores from emg only
##############################

nothing_labels_path = os.path.join(artifact_dir, 'nothing_labels.csv')
nothing_labels = pd.read_csv(nothing_labels_path)
nothing_labels_ind_path = os.path.join(artifact_dir, 'nothing_label_inds.npy')
nothing_labels_inds = np.load(nothing_labels_ind_path)

# Break down by session
nothing_selected_samples = nothing_labels.Abu.values
fin_nothing_inds = nothing_labels_inds[nothing_selected_samples]
fin_nothing_inds_df = pd.DataFrame(fin_nothing_inds, columns=[
                                   'session', 'taste_num', 'taste_trial'])
nothing_groups_inds, nothing_groups_dfs = zip(
    *list(fin_nothing_inds_df.groupby('session')))

##############################

fin_scored_data_list = []
for this_session in range(len(fin_score_table_list)):
    this_scored_data = fin_score_table_list[this_session]
    if this_session in nothing_groups_inds:
        which_nothing_group = np.where(
            np.array(nothing_groups_inds) == this_session)[0][0]
        this_nothing_df = nothing_groups_dfs[which_nothing_group]

        nothing_frame_list = []
        for ind, row in this_nothing_df.iterrows():
            this_taste = row.taste_num
            this_trial = row.taste_trial
            this_nothing_frame = pd.DataFrame({
                'Behavior': ['nothing'],
                'scoring_type': ['emg'],
                'taste_num': [this_taste],
                'taste_trial': [this_trial],
                'rel_time': [(-2, -1)],
            })
            nothing_frame_list.append(this_nothing_frame)
        this_nothing_frame = pd.concat(nothing_frame_list)
        this_scored_data = pd.concat([this_scored_data, this_nothing_frame])
        this_scored_data.reset_index(inplace=True, drop=True)
        fin_scored_data_list.append(this_scored_data)
    else:
        fin_scored_data_list.append(this_scored_data)

##############################

session_data_df['scored_data'] = fin_scored_data_list

############################################################
# At this stage, all raw data has been loaded
# Following steps are for feature extraction + engineering
############################################################

############################################################
# Extract mouth movements
############################################################
pre_stim = 2000
post_stim = 5000

###############
# Run everything through JL Process
gapes_Li_list = [run_JL_process(row.env, pre_stim, post_stim) \
        for ind, row in tqdm(session_data_df.iterrows())]
session_data_df['gapes_Li'] = gapes_Li_list

###############
# Run everything through AM Process

# run_AM_process returns: 
# segment_dat_list, feature_names, norm_interp_segment_dat
segment_dat_list_raw = [run_AM_process(row.env) for ind, row in tqdm(session_data_df.iterrows())]
feature_names = segment_dat_list_raw[0][1]
# Segment_dat_list:
# Sessions
#  - Trials
#    - feature_array, segment_dat, norm_interp_segment_dat, segment_bounds
segment_dat_list = [x[0] for x in segment_dat_list_raw]
session_data_df['segment_dat_list'] = segment_dat_list

# Note, session_data_df is a "session level" frame
# Columns:
# ['path_scores', 'basename', 'scores', 'path_emg', 'emg_name', 'emg',
#        'bsa_results_path', 'path_h5', 'h5', 'info_file_path', 'scored_data',
#        'env', 'bsa_p', 'bsa_omega', 'taste_orders', 'taste_map', 'gapes_Li',
#        'segment_dat_list']
# Shape:
# (n_sessions, 18)

# We now have to extract data for individual segments and generate a gape frame

##############################
# Generate gape frame
# Extract features returns:
# feature_array, feature_names, segment_dat, segment_starts, segment_ends
# Convert these to a dataframe using gen_segment_frame

# segment_frame_list = [gen_segment_frame(row.segment_dat_list, row.gapes_Li)
#                    for ind, row in tqdm(session_data_df.iterrows())]

# First pass to parse segment_dat_list
segment_frame_list = [parse_segment_dat_list(row.segment_dat_list) \
                   for ind, row in tqdm(session_data_df.iterrows())]
# Second pass to add gapes_Li
segment_frame_list = [parse_gapes_Li(this_Li, segment_frame) \
                   for this_Li, segment_frame in tqdm(zip(session_data_df.gapes_Li, segment_frame_list))] 

# Bounds for segment_frame are in 0-7000 time
# Adjust to make -2000 -> 5000
# Adjust segment_bounds by removing pre_stim
for segment_frame in segment_frame_list:
    all_segment_bounds = segment_frame.segment_bounds.values
    adjusted_segment_bounds = [
        np.array(x)-pre_stim for x in all_segment_bounds]
    segment_frame['segment_bounds'] = adjusted_segment_bounds

    # Calculate segment centers
    segment_frame['segment_center'] = [
        np.mean(x)/1000 for x in segment_frame.segment_bounds]

# Save segment_frame with both raw and adjusted bounds
session_data_df['segment_frame'] = segment_frame_list
segment_frame_raw_list = segment_frame_list.copy()
session_data_df['segment_frame_raw'] = segment_frame_raw_list


##############################

# Make sure that each segment in segment_frame is in fin_score_table
# That is, all extracted segments are scored
# If not, remove the segment from segment_frame
segment_frame_list = []
for ind, row in tqdm(session_data_df.iterrows()):
    segment_frame = row.segment_frame
    segment_frame['scored'] = False
    scored_data = row.scored_data
    scored_data.rename(columns={'rel_time': 'segment_bounds'}, inplace=True)

    score_bounds_list = []
    for event_ind, event_row in tqdm(segment_frame.iterrows()):
        taste = event_row.taste
        taste_trial = event_row.trial
        segment_center = event_row.segment_center
        wanted_score_table = scored_data.loc[
            (scored_data.taste_num == taste) &
            (scored_data.taste_trial == taste_trial)]
        if len(wanted_score_table):
            # Check if segment center is in any of the scored segments
            for _, score_row in wanted_score_table.iterrows():
                min_bool = score_row.segment_bounds[0] <= segment_center
                max_bool = segment_center <= score_row.segment_bounds[1]
                if min_bool & max_bool:
                    segment_frame.loc[event_ind, 'scored'] = True
                    segment_frame.loc[event_ind,
                                   'event_type'] = score_row.Behavior
                    score_bounds_list.append(
                        np.array(score_row.segment_bounds)*1000)
                    break

    segment_frame = segment_frame.loc[segment_frame.scored]
    segment_frame['score_bounds'] = score_bounds_list
    segment_frame_list.append(segment_frame)

session_data_df['segment_frame'] = segment_frame_list


# scored_segment_frame.to_pickle(os.path.join(code_dir, 'data', 'scored_segment_frame.pkl'))
session_data_df.to_pickle(os.path.join(artifact_dir, 'session_data_df.pkl'))

############################################################
############################################################
# analysis.py
############################################################
############################################################

############################################################
# Preprocessing
############################################################
# Get all gape frames

fin_segment_frames = []
for i, (_, this_row) in enumerate(session_data_df.iterrows()):
    this_frame = this_row.segment_frame_raw
    this_frame['session_ind'] = i
    this_frame['animal_num'] = this_row.animal_num
    this_frame['basename'] = this_row.basename
    fin_segment_frames.append(this_frame)
cat_segment_frame = pd.concat(fin_segment_frames, axis=0).reset_index(drop=True)

# Match gape frames with palatability
pal_dicts = [{i: x for i, x in enumerate(this_dict.items())}
             for this_dict in session_data_df.taste_map.values]
pal_frames = [pd.DataFrame.from_dict(x, orient='index', columns=['taste_name', 'pal'])
              for x in pal_dicts]
pal_frames = [x.reset_index(drop=False) for x in pal_frames]
pal_frames = [x.rename(columns={'index': 'taste'}) for x in pal_frames]

fin_pal_frames = []
for i, this_frame in enumerate(pal_frames):
    this_frame['session_ind'] = i
    this_frame['animal_num'] = session_data_df.animal_num.values[i]
    this_frame['basename'] = session_data_df.basename.values[i]
    fin_pal_frames.append(this_frame)
cat_pal_frame = pd.concat(fin_pal_frames, axis=0).reset_index(drop=True)

cat_segment_frame = pd.merge(
    cat_segment_frame,
    cat_pal_frame,
    on=['session_ind', 'taste', 'animal_num', 'basename'],
    how='inner')

# ##############################
# # Baseline normalization
# ##############################
# # Also get baseline values to normalize amplitude
# all_envs = np.stack(session_data_df.env)
# all_basenames = session_data_df.basename.values
# animal_nums = [x.split('_')[0] for x in all_basenames]
# session_nums = [x.split('_')[1] for x in all_basenames]
# 
# baseline_lims = [0, 2000]
# baseline_envs = all_envs[:, baseline_lims[0]:baseline_lims[1]]
# 
# mean_baseline = np.mean(baseline_envs, axis=-1)
# std_baseline = np.std(baseline_envs, axis=-1)
# inds = np.array(list(np.ndindex(mean_baseline.shape)))
# 
# baseline_frame = pd.DataFrame(
#     data=np.concatenate(
#         [
#             inds,
#             mean_baseline.flatten()[:, None],
#             std_baseline.flatten()[:, None]
#         ],
#         axis=-1
#     ),
#     columns=['session', 'taste', 'trial', 'mean', 'std']
# )
# # Convert ['session','taste','trial'] to int
# baseline_frame = baseline_frame.astype(
#     {'session': 'int', 'taste': 'int', 'trial': 'int'})
# baseline_frame['animal'] = [animal_nums[i] for i in baseline_frame['session']]
# baseline_frame['session_day'] = [session_nums[i]
#                                  for i in baseline_frame['session']]
# baseline_frame['session_name'] = [animal + '\n' + day for
#                                   animal, day in zip(baseline_frame['animal'], baseline_frame['session_day'])]
# mean_baseline_frame = baseline_frame.groupby(
#     ['session_name', 'session_day', 'animal']).mean()
# mean_baseline_frame.reset_index(inplace=True)
# mean_baseline_frame = mean_baseline_frame.astype({'session': 'int'})
# mean_baseline_frame.rename(columns={'mean': 'baseline_mean'}, inplace=True)
# 
# cat_segment_frame = pd.merge(
#     cat_segment_frame,
#     mean_baseline_frame[['session', 'baseline_mean']],
#     left_on=['session_ind'],
#     right_on=['session'],
#     how='left')

############################################################
# Cross-validated classification accuracy
############################################################
# Test classifier by leaving out:
# 1) Random session
# 1) Session within animal
# 2) Session across animals
# 3) Whole animals

# feature_names = [
#     'duration',
#     # 'amplitude_rel',
#     'amplitude_abs',
#     'left_interval',
#     'right_interval',
#     # 'pca_1',
#     # 'pca_2',
#     # 'pca_3',
#     'max_freq',
# ]

all_features = np.stack(cat_segment_frame.features.values)

# Drop 'amplitude_abs' from features
drop_inds = [i for i, x in enumerate(feature_names) if 'amplitude_abs' in x]
all_features = np.delete(all_features, drop_inds, axis=1)
feature_names = np.delete(feature_names, drop_inds)

# # Drop amplitude_rel
# drop_inds = [i for i, x in enumerate(feature_names) if 'amplitude_rel' in x]
# all_features = np.delete(all_features, drop_inds, axis=1)
# feature_names = np.delete(feature_names, drop_inds)

# # Normalize amplitude by baseline
# baseline_vec = cat_segment_frame.baseline_mean.values
# amplitude_inds = np.array(
#     [i for i, x in enumerate(feature_names) if 'amplitude' in x])
# all_features[:, amplitude_inds] = all_features[:, amplitude_inds] / \
#     baseline_vec[:, None]

# # Drop PCA features
# # In future, we can re-calculate PCA features
# drop_inds = np.array(
#     [i for i, x in enumerate(feature_names) if 'pca' in x])
# all_features = np.delete(all_features, drop_inds, axis=1)

# # Recalculate PCA features
# # First scale all segments by amplitude and length
# all_segments = cat_segment_frame.segment_raw.values
# 
# # MinMax scale segments
# min_max_segments = [x-np.min(x) for x in all_segments]
# min_max_segments = [x/np.max(x) for x in min_max_segments]
# 
# # Scale all segments to same length
# # max_len = np.max([len(x) for x in min_max_segments])
# # scaled_segments = [np.interp(np.linspace(0,1,max_len), np.linspace(0,1,len(x)), x) \
# #         for x in min_max_segments]
# # scaled_segments = np.stack(scaled_segments)
# scaled_segments = normalize_segments(min_max_segments)

scaled_segments = np.stack(cat_segment_frame.segment_norm_interp.values)

# Get PCA features
pca_obj = PCA()
pca_obj.fit(scaled_segments)
pca_features = pca_obj.transform(scaled_segments)[:, :3]

# Add PCA features to all features
all_features = np.concatenate([all_features, pca_features], axis=-1)

# Scale features
scaled_features = StandardScaler().fit_transform(all_features)

# Correct feature_names
pca_feature_names = [feature_names[i] for i in drop_inds]
feature_names = np.delete(feature_names, drop_inds)
feature_names = np.concatenate([feature_names, pca_feature_names])

# Categorize animal_num
animal_codes = cat_segment_frame.animal_num.astype('category').cat.codes.values

##############################
cat_segment_frame['raw_features'] = list(all_features)
cat_segment_frame['features'] = list(scaled_features)

###############
# Also scale amplitudes of semgents from later plotting
scored_df = cat_segment_frame[cat_segment_frame.scored == True]

# Correct event_types
types_to_drop = ['to discuss', 'other',
                 'unknown mouth movement', 'out of view']
scored_df = scored_df[~scored_df.event_type.isin(types_to_drop)]

# Remap event_types
event_type_map = {
    'mouth movements': 'mouth or tongue movement',
    'tongue protrusion': 'mouth or tongue movement',
    'mouth or tongue movement': 'mouth or tongue movement',
    'lateral tongue movement': 'lateral tongue protrusion',
    'lateral tongue protrusion': 'lateral tongue protrusion',
    'gape': 'gape',
    'no movement': 'no movement',
    'nothing': 'no movement',
}

scored_df['event_type'] = scored_df['event_type'].map(event_type_map)
scored_df['event_codes'] = scored_df['event_type'].astype('category').cat.codes
scored_df['is_gape'] = (scored_df['event_type'] == 'gape')*1

scored_df.dropna(subset=['event_type'], inplace=True)

# Column descriptions
out_dict = {
    'taste': 'Taste of the trial',
    'trial': 'Trial number given taste',
    'features': 'Features of the segment (with normalized amplitude)',
    'segment_raw': 'Raw segment',
    'segment_bounds': 'Start and end of segment',
    'segment_num': 'Segment number within trial',
    'classifier': 'Classification of segment using Jenn Li algorithm (1 = gape)',
    'segment_center': 'Center of segment within trial',
    'scored': 'Whether segment was scored',
    'event_type': 'Type of event according to scoring',
    'event_codes': 'Event type as a code (from scoring)',
    'session_ind': 'Session number',
    'animal_num': 'Animal number',
    'basename': 'Basename of the session',
    'taste_name': 'Name of the taste',
    'pal': 'Palatability of the taste',
    'baseline_mean': 'Mean of baseline amplitude (for normalization of amplitudes)',
    'raw_features': 'Features of the segment (without amplitude normalization)',
    'baseline_scaled_segments': 'Segments with amplitude normalized by baseline',
    'is_gape': 'Whether event is a gape (1 = gape)',
}


##############################
# Feature clustering by event type

# Why do they have to be sorted?
features = np.stack(scored_df.features.values)
event_codes = scored_df.event_codes.values
event_code_map = scored_df[['event_codes', 'event_type']].drop_duplicates()

##############################
# UMAP / NCA plot to visualize distinction of clusters by event type
nca_obj = NeighborhoodComponentsAnalysis(n_components=2)
nca_features = nca_obj.fit_transform(features, event_codes)

##############################
# Fit a GMM to 3D NCA of 'no movement' data
# and assign all points within 3 mahalanobis distances
# as no movement

# Get no movement data
no_movement_code = event_code_map[event_code_map.event_type ==
                                  'no movement'].event_codes.values[0]
no_movement_inds = np.where(event_codes == no_movement_code)
no_movement_3d_nca = nca_features[no_movement_inds]

# Fit gaussian
mean_vec = np.mean(no_movement_3d_nca, axis=0)
cov_mat = np.cov(no_movement_3d_nca, rowvar=False)

# Calculate mahalanobis distance
mahal_dist = np.array(
    [mahalanobis(x, mean_vec, np.linalg.inv(cov_mat)) for x in nca_features])
mahal_thresh = 2

updated_codes = event_codes.copy()
updated_codes[mahal_dist < mahal_thresh] = no_movement_code

# Update codes in scored_df
scored_df['updated_codes'] = updated_codes
scored_df['updated_event_type'] = scored_df.updated_codes.map(
    event_code_map.set_index('event_codes').event_type)

############################################################
############################################################
# unlabelled_prediction_comparison.py
############################################################
############################################################

##############################
# Specificitiy of annotations
##############################
# Over-write with updated event coes and types
scored_df['event_type'] = scored_df['updated_event_type']

scored_df.dropna(subset=['event_type'], inplace=True)

# Remove lateral tongue protrusion
scored_df = scored_df[scored_df.event_type != 'lateral tongue protrusion']

# Abbreviate mouth or tongue movement
scored_df['event_type'] = scored_df['event_type'].replace(
    ['mouth or tongue movement'],
    'MTMs'
)

# Update event codes to match with BSA
event_code_dict = {
    'gape': 1,
    'MTMs': 2,
    'no movement': 0,
}

scored_df['event_codes'] = scored_df['event_type'].map(event_code_dict)

# Save scored_df
scored_df.to_pickle(os.path.join(artifact_dir, 'fin_training_dataset.pkl'))
