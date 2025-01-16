## Import modules
pytau_dir = '/media/bigdata/projects/pytau'
# base_dir = '/home/exouser/Desktop/pytau'
import sys
sys.path.append(pytau_dir)
import pylab as plt
import numpy as np
import glob
from itertools import product
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from glob import glob
import json


base_dir = '/media/bigdata/firing_space_plot/firing_analyses/TRG_GC_PC_changepoint'
artifact_dir = os.path.join(base_dir,'artifacts')
plot_dir = os.path.join(base_dir,'plots')

if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

############################################################
# Get ELBO values for all fits in a database
############################################################
elbo_frame_path = os.path.join(artifact_dir,'elbo_frame.csv')

if not os.path.exists(elbo_frame_path):
    from pytau.changepoint_io import DatabaseHandler
    from pytau.changepoint_analysis import PklHandler
    # Can also get path to pkl file from model database
    fit_database = DatabaseHandler()
    fit_database.drop_duplicates()
    fit_database.clear_mismatched_paths()

    # Get fits for a particular experiment
    dframe = fit_database.fit_database
    wanted_exp_name = 'TRG_GC_PC'
    wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name] 

    # Pull out a single data_directory
    pkl_path_list = wanted_frame['exp.save_path'].values
    # pkl_path = wanted_frame['exp.save_path'].iloc[0]


    elbo_list = []
    metadata_list = []
    for pkl_path in tqdm(pkl_path_list):
        this_handler = PklHandler(pkl_path, process_data=False)
        elbo =  -this_handler.data['model_data']['approx'].hist[-1] 
        # metadata = this_handler.metadata
        metadata = this_handler.pretty_metadata
        elbo_list.append(elbo)
        metadata_list.append(metadata)

    elbo_frame = pd.concat([pd.DataFrame(metadata).T for metadata in metadata_list])
    elbo_frame.reset_index(inplace=True)
    elbo_frame['elbo'] = elbo_list
    elbo_frame.to_csv(elbo_frame_path)
else:
    elbo_frame = pd.read_csv(elbo_frame_path)

###############
# Get tastant names
data_dir_list = elbo_frame['data.data_dir'].unique()
info_file_paths = [glob(os.path.join(data_dir,'*.info'))[0] for data_dir in data_dir_list]
info_list = [json.load(open(info_file_path)) for info_file_path in info_file_paths]
# Get taste_params/tastes, taste_params/dig_ins, and taste_params/filenames
basenames = [os.path.basename(info_file_path).split('.')[0] \
        for info_file_path in info_file_paths]
dig_ins = [info['taste_params']['dig_ins'] for info in info_list]
tastes = [info['taste_params']['tastes'] for info in info_list]
filenames = [info['taste_params']['filenames'] for info in info_list]

taste_frame = pd.DataFrame(
        dict(
            basename = basenames,
            dig_ins = dig_ins,
            tastes = tastes,
            filenames = filenames,
            )
        )
# Explode
taste_frame = taste_frame.explode(['dig_ins','tastes','filenames'])

###############

############################################################
# Analysis 
############################################################
wanted_cols = [
        'model.states',
        'preprocess.data_transform',
        'data.basename',
        'data.taste_num',
        'data.region_name',
        'elbo',
        ]

wanted_elbo_frame = elbo_frame[wanted_cols]

###############
# Shuffled comparison
###############
group_cols = ['model.states','data.region_name','data.basename', 'data.taste_num']
grouped_frame = wanted_elbo_frame.groupby(group_cols)
grouped_frame_inds, grouped_frame_list = zip(*list(grouped_frame))

# Calculate 'None' - 'spike_shuffled' for 'preprocess.data_transform'
# Positive values indicate that 'None' is better
# Negative values indicate that 'spike_shuffled' is better

diff_list = []
for group in grouped_frame_list:
    none_elbo = group.loc[group['preprocess.data_transform'] == 'None']['elbo'].values[0]
    spike_shuffled_elbo = group.loc[group['preprocess.data_transform'] == 'spike_shuffled']['elbo'].values[0]
    diff = none_elbo - spike_shuffled_elbo
    diff_list.append(diff)

diff_frame = pd.DataFrame(grouped_frame_inds, columns = group_cols) 
diff_frame['elbo_diff'] = diff_list

# Plot
g = sns.scatterplot(
        data = diff_frame, 
        x = 'model.states', 
        y = 'elbo_diff', 
        hue = 'data.region_name', 
        )
# Add lineplot
sns.lineplot(
        data = diff_frame, 
        x = 'model.states', 
        y = 'elbo_diff', 
        hue = 'data.region_name', 
        ax = g,
        )
plt.savefig(os.path.join(plot_dir,'elbo_diff.png'))
plt.close()

# x = 'model.states'
# y = 'elbo'
# row = 'data.taste_num'
# hue = 'data.region_name'
# col = 'data.basename'

###############
# States comparison 
###############
state_frame = wanted_elbo_frame.copy()
state_frame = state_frame.loc[state_frame['preprocess.data_transform'] == 'None']
state_frame.drop(columns = 'preprocess.data_transform', inplace=True)
grouped_frame = state_frame.groupby(list(np.delete(group_cols,0)))
grouped_frame_inds, grouped_frame_list = zip(*list(grouped_frame))
zscored_frame_list = []
for group in grouped_frame_list:
    group['elbo_zscore'] = zscore(group['elbo'])
    zscored_frame_list.append(group)
zscored_frame = pd.concat(zscored_frame_list)

g = sns.scatterplot(
        data = zscored_frame, 
        x = 'model.states', 
        y = 'elbo_zscore', 
        hue = 'data.region_name', 
        )
sns.lineplot(
        data = zscored_frame, 
        x = 'model.states', 
        y = 'elbo_zscore', 
        hue = 'data.region_name', 
        ax = g,
        ci = 'sd',
        )
plt.savefig(os.path.join(plot_dir,'elbo_zscore.png'))
plt.close()

g = sns.relplot(
        data = zscored_frame, 
        x = 'model.states', 
        y = 'elbo_zscore', 
        hue = 'data.region_name', 
        col = 'data.taste_num',
        row = 'data.basename',
        kind = 'line',
        )
plt.savefig(os.path.join(plot_dir,'elbo_zscore_grid.png'))
plt.close()
