import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Patch
import seaborn as sns
import json
from time import time

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
code_dir = os.path.join(base_dir, 'src')
sys.path.append(code_dir)
from utils.extract_scored_data import return_taste_orders, process_scored_data
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            find_segment,
                                            calc_peak_interval,
                                            JL_process,
                                            gen_gape_frame,
                                            threshold_movement_lengths,
                                            )

artifact_dir = os.path.join(base_dir, 'artifacts')
if not os.path.isdir(artifact_dir):
    os.mkdir(artifact_dir)

############################################################
# Make sure we have 1) scored data, 2) emg env file, and 3) h5 files for each session
############################################################

data_dir = '/media/fastdata/Natasha_classifier_data' 
scored_data_paths = glob(os.path.join(data_dir, '**', '*scores.csv'), recursive = True)
scored_data_basenames = [os.path.basename(x).lower().split('.')[0] for x in scored_data_paths]
scored_data_basenames = ["_".join(x.split('_')[:-1]) for x in scored_data_basenames]

scores_path_df = pd.DataFrame({'path': scored_data_paths,
                               'basename': scored_data_basenames,
                               'scores': True})

##############################

emg_output_dirs = sorted(glob(os.path.join(data_dir,'*','*','*', 'emg_output')))

# For each day of experiment, load env and table files
data_subdirs = [glob(os.path.join(x,'*')) for x in emg_output_dirs]
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
                            'emg' : True})

# Only keep row if emg_name == 'emgad'
emg_path_df['emg_name'] = emg_path_df['emg_name'].str.lower()
emg_path_df = emg_path_df.loc[emg_path_df.emg_name == 'emgad']

##############################
# Check for BSA output
bsa_results_dirs = [glob(os.path.join(x, '*BSA_results'))[0] for x in emg_path_df.path]
emg_path_df['bsa_results_path'] = bsa_results_dirs

##############################

h5_files = glob(os.path.join(data_dir,'**','*','*', '*.h5'))
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


##############################
merge_df = pd.merge(scores_path_df, emg_path_df, on = 'basename',
                    suffixes = ('_scores', '_emg'),
                    how = 'outer')
merge_df = pd.merge(merge_df, h5_path_df, on = 'basename',
                    how = 'outer')
merge_df = merge_df.rename(columns = {'path':'path_h5'})

merge_df[['basename','scores','emg','h5']]

merge_df.dropna(inplace = True)
merge_df.reset_index(inplace = True, drop = True)

############################################################
cols_to_delete = [	
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
for ind, row in tqdm(merge_df.iterrows()): 

    ###############
    # scored_data_path = scored_data_paths[ind]
    scored_data_path = row.path_scores
    scored_data = pd.read_csv(scored_data_path)
    scored_data.drop(columns = cols_to_delete, inplace = True)	
    scored_data_list.append(scored_data)

    ###############
    # tastes x trials x time
    env_path = glob(os.path.join(row.path_emg, '*env.npy'))[0]
    env = np.load(env_path)
    envs_list.append(env)

    ###############
    bsa_p_filelist = glob(os.path.join(row.bsa_results_path, '*p.npy'))
    bsa_omega_filelist = glob(os.path.join(row.bsa_results_path, '*omega.npy'))[0]

    # Convert p to array
    # Extract inds from p filenames
    bsa_p_filelist.sort()
    bsa_p_basenames = [os.path.basename(x) for x in bsa_p_filelist]
    bsa_taste_inds = [int(x.split('_')[0][-2:]) for x in bsa_p_basenames]
    bsa_trial_inds = [int(x.split('_')[1][-2:]) for x in bsa_p_basenames]

    bsa_data = (np.stack([np.load(x) for x in bsa_p_filelist]) > 0.1).astype(int)

    # Convert to 1D timeseries
    freq_inds = np.arange(bsa_data.shape[2])
    bsa_data_flat = freq_inds[np.argmax(bsa_data, axis = 2)]

    # plt.imshow(bsa_data[0].T, interpolation = 'nearest', aspect = 'auto')
    # plt.plot(bsa_data_flat[0])
    # plt.show()

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
    taste_map = {taste:pal for taste, pal in zip(taste_names, pal_rankings)}
    taste_map_list.append(taste_map)

h5_files = merge_df.path_h5.tolist()
taste_order_list = []
for h5_file in tqdm(h5_files):
    try:
        taste_orders = return_taste_orders([h5_file])
    except:
        taste_orders = None
    taste_order_list.append(taste_orders)

############################################################
# Append to merge_df and drop any rows with None
merge_df['scored_data'] = scored_data_list
merge_df['env'] = envs_list
merge_df['bsa_p'] = bsa_p_list
merge_df['bsa_omega'] = omega_list
merge_df['taste_orders'] = taste_order_list
merge_df['taste_map'] = taste_map_list

# merge_df.basename[merge_df.taste_orders.isna()]

merge_df.dropna(inplace = True)
merge_df.reset_index(inplace = True, drop = True)

# Confirm that there are as many 'trial start's as trials in the ephys data
for ind, row in merge_df.iterrows():
    scored_data = row.scored_data
    taste_orders = row.taste_orders[0]
    n_trial_starts = len(scored_data.loc[scored_data.Behavior == 'trial start'])
    n_trials = len(taste_orders)
    if n_trials != n_trial_starts:
        print('Mismatch in trials for {}'.format(row.basename))

############################################################
fin_score_table_list = []
for ind, row in tqdm(merge_df.iterrows()):
    scored_data = merge_df.loc[ind, 'scored_data'].copy()
    taste_orders = merge_df.loc[ind, 'taste_orders'][0].copy()
    fin_scored_table = process_scored_data(scored_data, taste_orders)
    fin_score_table_list.append(fin_scored_table)

merge_df['scored_data'] = fin_score_table_list

############################################################
# Extract mouth movements 
############################################################
pre_stim = 2000
post_stim = 5000

###############
# Run everything through JL Process
gapes_Li_list = []
Li_time_taken_list = []
for ind, row in tqdm(merge_df.iterrows()):
    envs = row.env
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

        Li_time_taken_list.append(time_list)

        if gape_peak_inds is not None:
            gapes_Li[this_ind][gape_peak_inds] = 1

    gapes_Li_list.append(gapes_Li)

merge_df['gapes_Li'] = gapes_Li_list

Li_time_frame = pd.DataFrame(Li_time_taken_list, columns = ['preprocessing', 'classification'])
Li_time_frame.to_csv(os.path.join(artifact_dir, 'Li_time_frame.csv'))

###############
# Run everything through AM Process 
segment_dat_list_list = []
AM_time_taken_list = []
for ind, row in tqdm(merge_df.iterrows()):
    segment_dat_list = []
    envs = row.env
    inds = list(np.ndindex(envs.shape[:-1]))

    for this_ind in inds:
        this_trial_dat = envs[this_ind]

        start_time = time() 
        segment_starts, segment_ends, segment_dat = extract_movements(
            this_trial_dat, size=200)

        # Threshold movement lengths
        segment_starts, segment_ends, segment_dat = threshold_movement_lengths(
            segment_starts, segment_ends, segment_dat, 
            min_len = 50, max_len= 500)

        (feature_array,
         feature_names,
         segment_dat,
         segment_starts,
         segment_ends) = extract_features(
            segment_dat, segment_starts, segment_ends)

        end_feature_extraction_time = time()
        preprocessing_time = end_feature_extraction_time - start_time
        AM_time_taken_list.append(preprocessing_time)

        segment_bounds = list(zip(segment_starts, segment_ends))
        merged_dat = [feature_array, segment_dat, segment_bounds] 
        segment_dat_list.append(merged_dat)

    segment_dat_list_list.append(segment_dat_list)

merge_df['segment_dat_list'] = segment_dat_list_list

AM_time_frame = pd.DataFrame(AM_time_taken_list, columns = ['preprocessing'])
AM_time_frame.to_csv(os.path.join(artifact_dir, 'AM_time_frame.csv'))

##############################
# Generate gape frame
gape_frame_list = [gen_gape_frame(row.segment_dat_list, row.gapes_Li) \
        for ind, row in tqdm(merge_df.iterrows())]

# Bounds for gape_frame are in 0-7000 time
# Adjust to make -2000 -> 5000
# Adjust segment_bounds by removing pre_stim
for gape_frame in gape_frame_list:
    all_segment_bounds = gape_frame.segment_bounds.values
    adjusted_segment_bounds = [np.array(x)-pre_stim for x in all_segment_bounds]
    gape_frame['segment_bounds'] = adjusted_segment_bounds

    # Calculate segment centers
    gape_frame['segment_center'] = [np.mean(x)/1000 for x in gape_frame.segment_bounds]

merge_df['gape_frame'] = gape_frame_list
gape_frame_raw_list = gape_frame_list.copy()
merge_df['gape_frame_raw'] = gape_frame_raw_list


##############################

# Make sure that each segment in gape_frame is in fin_score_table
gape_frame_list = []
for ind, row in tqdm(merge_df.iterrows()):
    gape_frame = row.gape_frame
    scored_data = row.scored_data
    scored_data.rename(columns = {'rel_time' : 'segment_bounds'}, inplace = True)

    score_bounds_list = []
    for event_ind, event_row in tqdm(gape_frame.iterrows()):
        taste = event_row.taste
        taste_trial = event_row.trial
        segment_center = event_row.segment_center
        wanted_score_table = scored_data.loc[
            (scored_data.taste_num == taste) &
            (scored_data.taste_trial == taste_trial)]
        if len(wanted_score_table):
            # Check if segment center is in any of the scored segments
            for _, score_row in wanted_score_table.iterrows():
                if (score_row.segment_bounds[0] <= segment_center) & (segment_center <= score_row.segment_bounds[1]):
                    gape_frame.loc[event_ind, 'scored'] = True
                    # gape_frame.loc[event_ind, 'event_type'] = score_row.event  
                    gape_frame.loc[event_ind, 'event_type'] = score_row.Behavior  
                    score_bounds_list.append(np.array(score_row.segment_bounds)*1000)
                    break
                else:
                    gape_frame.loc[event_ind, 'scored'] = False

    gape_frame = gape_frame.loc[gape_frame.scored == True]
    gape_frame['score_bounds'] = score_bounds_list
    gape_frame_list.append(gape_frame)

merge_df['gape_frame'] = gape_frame_list

# scored_gape_frame.to_pickle(os.path.join(code_dir, 'data', 'scored_gape_frame.pkl'))
merge_df.to_pickle(os.path.join(artifact_dir, 'all_data_frame.pkl'))

############################################################
# Test plots 
############################################################
plot_dir = os.path.join(base_dir, 'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

# Plot scored, segmented data
#############################

segment_plot_dir = os.path.join(plot_dir, 'segment_plots')
if not os.path.isdir(segment_plot_dir):
    os.mkdir(segment_plot_dir)

# Get all event-types across all gape_frames
all_event_types = []
for this_gape_frame in gape_frame_list:
    all_event_types.extend(this_gape_frame.event_type.unique())
all_event_types = list(set(all_event_types))

cmap = plt.get_cmap('tab10')# len(event_types))
event_colors = {all_event_types[i]:cmap(i) for i in range(len(all_event_types))}

t = np.arange(-2000, 5000)
for i, this_basename in enumerate(tqdm(merge_df.basename.unique())):

    gape_frame = merge_df.loc[merge_df.basename == this_basename, 'gape_frame'].values[0]
    gape_frame.dropna(inplace = True)

    scored_gape_frame = gape_frame.copy()
    event_types = scored_gape_frame.event_type.unique()

    #event_types = ['mouth movements','unknown mouth movement','gape','tongue protrusion','lateral tongue protrusion']
    #scored_gape_frame = scored_gape_frame.loc[scored_gape_frame.event_type.isin(event_types)]


    plot_group = list(scored_gape_frame.groupby(['taste','trial']))
    plot_inds = [x[0] for x in plot_group]
    plot_dat = [x[1] for x in plot_group]

    # Generate custom legend
    legend_elements = [Patch(facecolor=event_colors[event], edgecolor='k',
                             label=event.title()) for event in event_types]

    # Plot with black horizontal lines over detected movements
    plot_n = np.min([30, len(plot_dat)])
    fig,ax = plt.subplots(plot_n, 1, sharex=True, sharey=True,
                          figsize = (10, plot_n*2))
    for i in range(plot_n):
        this_scores = plot_dat[i]
        this_inds = plot_inds[i]
        this_env = envs[this_inds]
        ax[i].plot(t, this_env, color = 'k')
        score_bounds_list = []
        for _, this_event in this_scores.iterrows():
            event_type = this_event.event_type
            score_start = this_event.score_bounds[0]
            score_stop = this_event.score_bounds[1]
            segment_start = this_event.segment_bounds[0]
            segment_stop = this_event.segment_bounds[1]
            segment_inds = np.logical_and(t >= segment_start, t <= segment_stop) 
            segment_t = t[segment_inds]
            segment_env = this_env[segment_inds]
            env_max = np.max(segment_env)
            h_line_y = env_max*1.3
            ax[i].plot(segment_t, segment_env, color='k')
            ax[i].hlines(h_line_y, segment_t[0], segment_t[-1], 
                         color = 'k', linewidth = 5, alpha = 0.7)
            this_event_c = event_colors[event_type]
            if tuple(this_event.score_bounds) not in score_bounds_list:
                ax[i].axvspan(score_start, score_stop, 
                              color=this_event_c, alpha=0.5, label=event_type)
                score_bounds_list.append(tuple(this_event.score_bounds))
    ax[0].legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1.5, 1.1))
    ax[0].set_xlim([-2000, 5000])
    fig.subplots_adjust(right=0.75)
    fig.savefig(os.path.join(
        segment_plot_dir, 
        f'{this_basename}_scored_segmented_overlay_black_lines'), 
                dpi = 150, bbox_inches='tight')
    plt.close(fig)
    #plt.show()
