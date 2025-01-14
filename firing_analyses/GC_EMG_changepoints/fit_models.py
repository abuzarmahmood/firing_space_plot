## Import modules
base_dir = '/media/bigdata/projects/pytau'
import sys
sys.path.append(base_dir)
from pytau.changepoint_io import FitHandler
import pylab as plt
from pytau.utils import plotting
from pytau.utils import ephys_data
from tqdm import tqdm
from pytau.changepoint_io import DatabaseHandler
# from pytau.changepoint_analysis import PklHandler
from pytau.changepoint_analysis import *
import os
from pprint import pprint as pp
import pandas as pd
from ast import literal_eval
import numpy as np


##############################
# Data Dirs
data_dir_file = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/data_dir_list.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

pp(data_dir_list)

base_dir = os.path.dirname(data_dir_file)
artifact_dir = os.path.join(base_dir, 'artifacts')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

error_list = []
for this_dir in tqdm(data_dir_list):
    this_ephys_data = ephys_data.EphysData(this_dir)
    this_ephys_data.get_spikes()
    n_tastes = len(this_ephys_data.spikes)

    # Get trial-changepoints
    artifact_dir = os.path.join(this_dir, 'QA_output', 'artifacts')
    # Get all files
    all_files = sorted(os.listdir(artifact_dir))
    # Keep only csv
    all_files = [x for x in all_files if 'csv' in x]
    # Look for pattern taste_* in each name
    # taste_ind = [int(x.split('taste_')[1][0]) for x in all_files if 'taste' in x]

    # Look for 'taste_trial'
    all_files = [x for x in all_files if 'taste_trial' in x][0]

    # Load pkl files
    # trial_change_dat = [pd.read_csv(os.path.join(artifact_dir, x), index_col = 0) \
    #         for x in all_files if 'taste' in x]
    trial_change_dat = pd.read_csv(os.path.join(artifact_dir, all_files), index_col = 0)

    # Get median lowest elbo
    # best_change = []
    # for this_df in trial_change_dat:
    med_elbo = trial_change_dat.groupby('changes').median()
    # best_change.append(med_elbo['elbo'].idxmin())
    best_change = med_elbo['elbo'].idxmin()

    # Get model with lowest elbo for each taste
    # median_mode_list = []
    # for this_taste_ind, this_change, this_df in zip(taste_ind, best_change, trial_change_dat):
    #     this_df = this_df[this_df['changes'] == this_change] 
    #     mode_list = np.stack([literal_eval(x) for x in this_df['mode']])
    #     median_mode = np.median(mode_list, axis=0) 
    #     median_mode = median_mode.astype(int)
    #     median_mode_list.append(median_mode)
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

    for i in range(n_tastes):
        # Shape: (n_trials, n_neurons, n_timepoints)
        this_taste = this_ephys_data.spikes[i]
        # If no spikes, skip
        if len(this_taste) == 0:
            print(f"Empty spikes in {this_dir}")
            error_list.append(this_dir)
            continue

        # Check if trials need to be split
        # trial_splits = []
        # split_bounds_list = []
        # if best_change[i] > 0:
        #     this_median_mode = median_mode_list[i]
        #     max_trials = this_taste.shape[0]
        #     split_bounds = np.concatenate(([0], this_median_mode, [max_trials]))
        #     for j in range(len(split_bounds) - 1):
        #         trial_splits.append(this_taste[split_bounds[j]:split_bounds[j+1]])
        #         this_split_bounds = (split_bounds[j], split_bounds[j+1])
        #         split_bounds_list.append(this_split_bounds)

        # else:
        #     trial_splits.append(this_taste)
        #     split_bounds_list.append((0, this_taste.shape[0]))

        # for split_ind, this_split in enumerate(trial_splits):
        
        this_taste_frame = trial_info_frame[trial_info_frame['taste_ind'] == i]

        for section_ind in np.sort(this_taste_frame['section'].unique()):
            
            section_frame = this_taste_frame[this_taste_frame['section'] == section_ind]
            # Get trial splits
            section_trials = section_frame['taste_rel_trial_num'].values
            this_split = this_taste[section_trials]

            # Check if already fit
            try:

                # Specify params for fit
                model_parameters = dict(
                    states=4,
                    fit=60000,
                    samples=20000,
                    model_kwargs={'None': None},
                        )

                preprocess_parameters = dict(
                    time_lims=[2000, 4000],
                    bin_width=50,
                    data_transform='None',  # Can also be 'spike_shuffled','trial_shuffled'
                    )

                FitHandler_kwargs = dict(
                    data_dir=this_dir,
                    taste_num=i,
                    region_name='all',  # Should match specification in info file
                    laser_type=None,
                    experiment_name='GC_EMG_changepoints_single_taste_drift_cut_3',
                    )

                ## Initialize handler, and feed paramters
                handler = FitHandler(**FitHandler_kwargs)
                handler.set_model_params(**model_parameters)
                handler.set_preprocess_params(**preprocess_parameters)

                # Overwrite handler.data with this_split
                handler.data = this_split

                # We can save taste_num as fractional for splits
                # Except the following two functions need integer taste_num
                handler.preprocess_data()
                handler.create_model()

                # Once we've run them, we can overwrite the taste_num
                # wanted_split_bounds = split_bounds_list[split_ind]
                wanted_split_bounds = (np.min(section_trials), np.max(section_trials))
                handler.database_handler.taste_num = f'{i}_split_ind{section_ind}_split_bounds{wanted_split_bounds}'

                # Perform inference and save output to model database
                handler.run_inference()
                handler.save_fit_output()
            except Exception as e:
                print(f"Error in {this_dir}")
                print(e)
                error_list.append(this_dir)
                continue

# plt.plot(-handler.inference_outs['approx'].hist)
# plt.show()


##############################
# Access fit results
# Directly from handler
# infernece_outs contains following attributes
# model : Model structure
# approx : Fitted model
# lambda : Inferred firing rates for each state
# tau : Inferred changepoints
# data : Data used for inference

# Can also get path to pkl file from model database

fit_database = DatabaseHandler()
# fit_database.drop_duplicates()
# fit_database.clear_mismatched_paths()

# Get fits for a particular experiment
dframe = fit_database.fit_database
wanted_exp_name = 'GC_EMG_changepoints_single_taste_drift_cut_3'
wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 

# Drop duplicates
wanted_frame = wanted_frame.drop_duplicates(subset=['data.basename', 'data.taste_num'])

# Only keep if 'data.taste_num' has 'split' in it
wanted_frame = wanted_frame[wanted_frame['data.taste_num'].str.contains('split')]
wanted_frame['base_taste'] = wanted_frame['data.taste_num'].apply(lambda x: x[0])

# Need to group 'tau.scaled_mode_tau' by 'data.basename'
grouped_frame = wanted_frame.groupby(['data.basename','base_taste']).agg(list)


class _tau():
    """Tau class to keep track of metadata and perform useful transformations
    """

    def __init__(self, tau_array, metadata):
        """Initialize tau class

        Args:
            tau_array ([type]): Array of samples from fitted model
            metadata (Dict): Dict containing metadata on fit
        """
        self.raw_tau = tau_array

        time_lims = metadata['preprocess']['time_lims']
        bin_width = metadata['preprocess']['bin_width']

        self.raw_int_tau = np.vectorize(np.int)(self.raw_tau)
        self.raw_mode_tau = mode(self.raw_int_tau)[0][0]

        self.scaled_tau = (self.raw_tau * bin_width) + time_lims[0]
        self.scaled_int_tau = np.vectorize(np.int)(self.scaled_tau)
        self.scaled_mode_tau = np.squeeze(mode(self.scaled_int_tau)[0])

class PklHandler():
    """Helper class to handle metadata and fit data from pkl file
    """

    def __init__(self, file_path):
        """Initialize PklHandler class

        Args:
            file_path (str): Path to pkl file
        """
        self.dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        self.file_name_base = file_name.split('.')[0]
        self.pkl_file_path = \
            os.path.join(self.dir_name, self.file_name_base + ".pkl")
        with open(self.pkl_file_path, 'rb') as this_file:
            self.data = pkl.load(this_file)

        model_keys = ['model', 'approx', 'lambda', 'tau', 'data']
        key_savenames = ['_model_structure', '_fit_model',
                         'lambda_array', 'tau_array', 'processed_spikes']
        data_map = dict(zip(model_keys, key_savenames))

        for key, var_name in data_map.items():
            setattr(self, var_name, self.data['model_data'][key])

        self.metadata = self.data['metadata']
        self.pretty_metadata = pd.json_normalize(self.data['metadata']).T

        self.tau = _tau(self.tau_array, self.metadata)
        # self.firing = _firing(self.tau, self.processed_spikes, self.metadata)

all_scaled_mode_tau = []
all_section_array = []
all_basename = []
all_base_taste = []
for (this_basename, this_base_taste), this_row in tqdm(grouped_frame.iterrows()):
    taste_num_list = this_row['data.taste_num']
    # Make sure taste_num_list is sorted
    sort_inds = np.argsort(taste_num_list)
    taste_num_list = np.array(taste_num_list)[sort_inds]
    exp_save_paths = np.array(this_row['exp.save_path'])[sort_inds]

    scaled_mode_tau_list = []
    section_list = []
    for pkl_path, taste_num in zip(exp_save_paths, taste_num_list):
        this_handler = PklHandler(pkl_path)
        # this_handler.pretty_metadata
        scaled_mode_tau = this_handler.tau.scaled_mode_tau
        if scaled_mode_tau.ndim == 1:
            scaled_mode_tau = scaled_mode_tau[None,:]
        scaled_mode_tau_list.append(scaled_mode_tau)
        section_num = int(taste_num.split('ind')[1][0])
        section_array = np.ones(len(scaled_mode_tau)) * section_num
        section_list.append(section_array)

    fin_scaled_mode_tau = np.concatenate(scaled_mode_tau_list)
    fin_section_array = np.concatenate(section_list)
    all_scaled_mode_tau.append(fin_scaled_mode_tau)
    all_section_array.append(fin_section_array)
    all_basename.append(this_basename)
    all_base_taste.append(this_base_taste)

df_dict = dict(
    basename=all_basename,
    base_taste=all_base_taste,
    scaled_mode_tau=all_scaled_mode_tau,
    section_array=all_section_array
    )
df = pd.DataFrame(df_dict)
df.to_pickle(os.path.join(artifact_dir, 'scaled_mode_tau_cut.pkl'))

# Pull out a single data_directory

scaled_mode_tau_list = []
basename_list = []
taste_num_list = []
for i, this_row in tqdm(wanted_frame.iterrows()):
    pkl_path = this_row['exp.save_path']
    basename = this_row['data.basename']
    basename_list.append(basename)
    taste_num = this_row['data.taste_num']
    taste_num_list.append(taste_num)

    # From saved pkl file
    this_handler = PklHandler(pkl_path)
    # this_handler.pretty_metadata

    scaled_mode_tau = this_handler.tau.scaled_mode_tau
    scaled_mode_tau_list.append(scaled_mode_tau)

# Save as pandas dataframe
df_dict = dict(
    basename=basename_list,
    taste_num=taste_num_list,
    scaled_mode_tau=scaled_mode_tau_list,
    )
df = pd.DataFrame(df_dict)
df.to_pickle(os.path.join(artifact_dir, 'scaled_mode_tau.pkl'))
