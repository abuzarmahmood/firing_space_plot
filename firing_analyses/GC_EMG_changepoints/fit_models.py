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
from pytau.changepoint_analysis import PklHandler
import os

fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()

##############################
# Data Dirs
data_dir_file = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/data_dir_list.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

base_dir = os.path.dirname(data_dir_file)
artifact_dir = os.path.join(base_dir, 'artifacts')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

error_list = []
for this_dir in tqdm(data_dir_list):
    this_ephys_data = ephys_data.EphysData(this_dir)
    this_ephys_data.get_spikes()
    n_tastes = len(this_ephys_data.spikes)
    for i in range(n_tastes):
        this_taste = this_ephys_data.spikes[i]
        # If no spikes, skip
        if len(this_taste) == 0:
            print(f"Empty spikes in {this_dir}")
            error_list.append(this_dir)
            continue
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
                experiment_name='GC_EMG_changepoints_single_taste',
                )

            ## Initialize handler, and feed paramters
            handler = FitHandler(**FitHandler_kwargs)
            handler.set_model_params(**model_parameters)
            handler.set_preprocess_params(**preprocess_parameters)

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

# Get fits for a particular experiment
dframe = fit_database.fit_database
wanted_exp_name = 'GC_EMG_changepoints_single_taste'
wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 
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
    this_handler.pretty_metadata

    scaled_mode_tau = this_handler.tau.scaled_mode_tau
    scaled_mode_tau_list.append(scaled_mode_tau)

# Save as pandas dataframe
import pandas as pd
df_dict = dict(
    basename=basename_list,
    taste_num=taste_num_list,
    scaled_mode_tau=scaled_mode_tau_list,
    )
df = pd.DataFrame(df_dict)
df.to_pickle(os.path.join(artifact_dir, 'scaled_mode_tau.pkl'))
