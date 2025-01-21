import sys
import os
from tqdm import tqdm
import pandas as pd
blech_clust_dir = os.path.expanduser('~/Desktop/blech_clust')
sys.path.append(blech_clust_dir)
from utils.ephys_data.ephys_data import ephys_data
from pprint import pprint as pp
import numpy as np
from ast import literal_eval


# data_dir_file_path = '/media/fastdata/Thomas_Data/all_data_dirs.txt'
data_dir_file_path = '/media/fastdata/Thomas_Data/data/sorted_new/data_dir_list.txt'
data_dir_list = [x.strip() for x in open(data_dir_file_path, 'r').readlines()]

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/inter_region_rate_regression'
artifact_dir =  os.path.join(base_dir,'artifacts')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)
plot_dir = os.path.join(base_dir,'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

recollect_data = False

if recollect_data:
    for this_dir in tqdm(data_dir_list):
        try:
            this_ephys_data = ephys_data(this_dir)

            print(' ===================================== ')
            print(this_dir)
            print(' ===================================== ')

            this_ephys_data.get_region_units()
            this_ephys_data.get_region_units()
            region_dict = dict(
                    zip(
                        this_ephys_data.region_names,
                        this_ephys_data.region_units,
                        )
                    )
            inv_region_map = {}
            for k,v in region_dict.items():
                for unit in v:
                    inv_region_map[unit] = k
            this_ephys_data.get_spikes()
            n_tastes = len(this_ephys_data.spikes)

            # Get trial-changepoints
            qa_output_dir = os.path.join(this_dir, 'QA_output')
            best_change_path = os.path.join(qa_output_dir, 'best_change.txt')
            best_change = int(open(best_change_path, 'r').readlines()[1])

            session_artifact_dir = os.path.join(this_dir, 'QA_output', 'artifacts')
            # Get all files
            all_files = sorted(os.listdir(session_artifact_dir))
            # Keep only csv
            all_files = [x for x in all_files if 'csv' in x]
            # Look for pattern taste_* in each name
            # taste_ind = [int(x.split('taste_')[1][0]) for x in all_files if 'taste' in x]

            # Look for 'taste_trial'
            all_files = [x for x in all_files if 'taste_trial' in x][0]

            # Load pkl files
            # trial_change_dat = [pd.read_csv(os.path.join(artifact_dir, x), index_col = 0) \
            #         for x in all_files if 'taste' in x]
            trial_change_dat = pd.read_csv(os.path.join(session_artifact_dir, all_files), index_col = 0)

            # # Get median lowest elbo
            # # best_change = []
            # # for this_df in trial_change_dat:
            # med_elbo = trial_change_dat.groupby('changes').median()
            # # best_change.append(med_elbo['elbo'].idxmin())
            # best_change = med_elbo['elbo'].idxmin()

            best_change_df = trial_change_dat[trial_change_dat['changes'] == best_change]
            mode_list = np.stack([literal_eval(x) for x in best_change_df['mode']])
            median_mode_list = np.median(mode_list, axis=0)
            time_bins = trial_change_dat['time_bins'].iloc[0]
            time_bins = time_bins.replace(' ',',').replace('\n','').replace('[','').replace(']','').split(',')
            # Drop any empty strings
            time_bins = [x for x in time_bins if x]
            # Convert to float
            time_bins = [float(x) for x in time_bins]

            # Convert median_mode_list to timepoints using interpolated time_bins 
            interpolated_time = np.interp(median_mode_list, np.arange(len(time_bins)), time_bins)

            # Make "sections"
            time_sections = np.concatenate(([0], interpolated_time, [np.max(time_bins)]))

            # Get trial_info_frame and get sections for all trials
            trial_info_path = os.path.join(this_dir, 'trial_info_frame.csv') 
            trial_info_frame = pd.read_csv(trial_info_path)
            trial_info_frame['section'] = pd.cut(trial_info_frame['start_taste'], time_sections, labels=False)
            
            wanted_columns = ['dig_in_num_taste', 'start_taste', 'section', 'taste_rel_trial_num']
            trial_info_frame = trial_info_frame[wanted_columns]

            dig_in_num_taste_map = dict(zip(
                np.sort(trial_info_frame['dig_in_num_taste'].unique()), 
                range(n_tastes)))
            trial_info_frame['taste_ind'] = trial_info_frame['dig_in_num_taste'].map(dig_in_num_taste_map) 

            this_ephys_data.get_sequestered_firing()
            seq_firing_frame = this_ephys_data.sequestered_firing_frame
            seq_firing_frame.reset_index(inplace=True)

            merge_frame = pd.merge(
                    seq_firing_frame, 
                    trial_info_frame, 
                    left_on = ['taste_num','trial_num'],
                    right_on = ['taste_ind','taste_rel_trial_num'],
                    how = 'inner',
                    )
            merge_frame.drop(columns = ['index','taste_rel_trial_num','taste_ind'], inplace=True)
            # Add time values
            t_vec = np.vectorize(int)(np.linspace(-2000, 5000, merge_frame.time_num.max()+1))
            merge_frame['time'] = t_vec[merge_frame['time_num']]

            # Add region values
            merge_frame['region'] = merge_frame['neuron_num'].map(inv_region_map)

            basename = os.path.basename(this_dir)
            animal = os.path.basename(os.path.dirname(this_dir))
            merge_frame['basename'] = basename
            merge_frame['animal'] = animal

            # Save to pkl
            save_path = os.path.join(artifact_dir, f'{animal}_{basename}_section_rate.pkl')
            merge_frame.to_pickle(save_path)
        
        except Exception as e:
            print(f'Error with {this_dir}')
            print(e)
else:
    df_list_paths = os.listdir(artifact_dir)
    df_list_paths = [x for x in df_list if 'section_rate.pkl' in x]
    df_list = [pd.read_pickle(os.path.join(artifact_dir, x)) for x in df_list_paths]

############################################################
# Perform regression
############################################################
