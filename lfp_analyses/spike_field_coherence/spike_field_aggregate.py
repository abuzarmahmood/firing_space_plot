"""
Use Rayleigh test to generate plots of:
    1) zscored rayleigh statistic
    2) zscored p-value

"""

# Import required modules
from pycircstat.tests import rayleigh
from scipy.stats import mannwhitneyu, linregress
import seaborn as sns
# from astropy.stats import rayleightest
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import tables
import numpy as np
import numpy.ma as ma
from tqdm import tqdm, trange
import shutil
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
import pingouin as pg
from scipy.interpolate import interp1d
from scipy.optimize import minimize

def parallelize(func, iterator):
    return Parallel(n_jobs=cpu_count()-2)(delayed(func)(this_iter) for this_iter in tqdm(iterator))

def custom_rayleigh(x):
    return rayleigh(x.phases)
# return rayleightest(x.phases)

def rolling_rayleigh(x, start=-1000, stop=2500, window=250, step=25):
    window_starts = np.arange(start, stop-window+step, step)
    window_ends = window_starts + window
    windows = list(zip(window_starts, window_ends))
    p_val_list = []
    spike_count_list = []
    for this_window in windows:
        this_dat = x.loc[(x.time >= this_window[0]) &
                         (x.time < this_window[1])]
        spike_count_list.append(len(this_dat))
        p_val = custom_rayleigh(this_dat)
        p_val_list.append(p_val)
    p_val_frame = np.array(p_val_list)
    # Window ends is taken as time for window, otherwise -100 will get
    # data for post-stim if window > 100
    return_frame = pd.DataFrame(
            dict(
                time=window_ends,
                p_val=p_val_frame[:, 0],
                z_stat=p_val_frame[:, 1],
                time_bins = list(zip(window_starts, window_ends)),
                spike_counts = spike_count_list,
                     )
    )
    z_stat = return_frame['z_stat']
    return_frame['scaled_z_stat'] = (
        z_stat - z_stat.min()) / (z_stat.max() - z_stat.min())
    return return_frame
    # return pd.DataFrame(dict(time = window_starts, p_val = p_val_list))

def temp_rolling_rayleigh(x): 
    return rolling_rayleigh(x, step = step_size, window = window_size)

def process_rayleigh(dataframe, var = 'z_stat', add_id = None):
    """
    Process rayleigh dataframes

    Inputs:
        dataframe: dataframe with rayleigh data
        var: variable to use for processing
        add_id: additional id to add to dataframe

    Outputs:
        frame_pivot_list: list of dataframes with rayleigh data
        frame_inds: list of indices for each dataframe
    """
    ind_cols = ['basename', 'spikes_region', 'phase_region', 'nrn_num']
    # dataframe['ind'] = dataframe[ind_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1) 
    # Groupby ind and perform pivot
    #frame_list = list(dataframe.groupby('ind'))
    frame_list = list(dataframe.groupby(ind_cols))
    frame_inds = [x[0] for x in frame_list]
    if add_id is not None:
        frame_inds = [x + (add_id,) for x in frame_inds]
    frame_dat = [x[1] for x in frame_list]
    # Pivot on freq and time with z_stat as values
    frame_pivot_list = [x.pivot(index='freq', columns='time', values=var) for x in frame_dat]
    return frame_pivot_list, frame_inds


##################################################
## Params 
##################################################
time_lims = [-1000, 2500]
window_size = 150
step_size = 150

if window_size == step_size:
    rayleigh_time_vec = np.arange(time_lims[0], time_lims[1], step_size)
else:
    raise ValueError('Window size and step size must be equal')

##################################################
## Set up paths 
##################################################
base_dir = '/media/bigdata/firing_space_plot/lfp_analyses/spike_field_coherence'
plot_dir = os.path.join(base_dir, 'plots')
save_dir = os.path.join(base_dir, 'temp_data')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path, 'r').readlines()]
dir_basenames = [os.path.basename(x) for x in dir_list]

actual_save_path = '/stft/analyses/spike_phase_coherence/actual'
shuffled_save_path = '/stft/analyses/spike_phase_coherence/shuffled'

paths = [actual_save_path, shuffled_save_path]
basenames = [os.path.basename(x) for x in paths]

group_name_list = ['basename', 'spikes_region',
                   'phase_region', 'nrn_num', 'freq']


dat = ephys_data(dir_list[0])
dat.get_stft()
freq_vec = dat.freq_vec
##################################################
## Begin Processing 
##################################################

#i = 1
#this_path = paths[i]
#this_basename = basenames[i]

for i, (this_path, this_basename) in enumerate(zip(paths, basenames)):

    phase_list = []
    for this_dir in tqdm(dir_list):
        dat = ephys_data(this_dir)
        data_basename = os.path.basename(this_dir)

        if this_basename == 'actual':
            node_paths = [actual_save_path]
            node_basenames = [os.path.basename(x) for x in node_paths]
        elif this_basename == 'shuffled':
            with tables.open_file(dat.hdf5_path) as hf5:
                node_list = hf5.list_nodes(where=this_path) 
                node_paths = [x._v_pathname for x in node_list]
                node_basenames = [os.path.basename(x) for x in node_paths]

        for this_node_path, this_node_basename in tqdm(zip(node_paths, node_basenames)):
            key = '/' + os.path.join(this_node_basename, data_basename)
            with tables.open_file(os.path.join(save_dir, 'rolling_rayleigh.h5'), 'a') as hf5:
                if key in hf5:
                    print(f'=== {key} already exists ===')
                    continue

            phase_coherence_frame = pd.read_hdf(dat.hdf5_path, this_node_path)
            phase_coherence_frame['basename'] = os.path.basename(dat.data_dir)
            phase_coherence_frame = phase_coherence_frame.reset_index(drop=False)
            phase_coherence_frame['time'] = phase_coherence_frame['time'] - 2000
            time_bool = np.logical_and(
                phase_coherence_frame['time'] > time_lims[0], 
                phase_coherence_frame['time'] < time_lims[1]) 
            phase_coherence_frame = phase_coherence_frame.loc[time_bool.values]
            phase_coherence_frame['freq'] = freq_vec[phase_coherence_frame['freq']]
            phase_coherence_frame['freq'] = phase_coherence_frame['freq'].astype(int)
            # Drop 0 Hz
            phase_coherence_frame = phase_coherence_frame.loc[fin_phase_frame['freq'] != 0]

            # Group data for processing with Rayleigh
            group_obj = phase_coherence_frame.groupby(group_name_list)
            group_list = list(group_obj)
            group_inds = [x[0] for x in group_list]
            group_inds_str = ["_".join([str(y) for y in x]) for x in group_inds]
            group_save_str = [this_node_basename + '_' + x for x in group_inds_str]
            group_dat = [x[1] for x in group_list]

            # # Save all group_dat;s
            # for this_group_save_str, this_group_dat in zip(group_save_str, group_dat):
            #     this_save_path = os.path.join(save_dir, this_group_save_str)
            #     this_group_dat.to_csv(this_save_path)

            group_rolling_rayleigh = parallelize(temp_rolling_rayleigh, group_dat)

            for meta, this_dat in tqdm(zip(group_inds, group_rolling_rayleigh)):
                this_dat[group_name_list] = meta

            group_rayleigh = pd.concat(group_rolling_rayleigh)
            group_rayleigh = group_rayleigh.dropna()

            # Save group_rayleigh
            #group_rayleigh.to_csv(os.path.join(save_dir, data_basename + '_' + this_node_basename + '.csv'))
            # Save to hdf5
            print()
            print(f'Saving to {key}')
            print()
            group_rayleigh.to_hdf(os.path.join(save_dir, 'rolling_rayleigh.h5'), key=key)


# fin_phase_frame = pd.concat(phase_list)

# fin_phase_frame = fin_phase_frame.reset_index(level = 'time')
# fin_phase_frame['time'] = fin_phase_frame['time'] - 2000
# time_bool = np.logical_and(
#     fin_phase_frame['time'] > -1000,
#     fin_phase_frame['time'] < 2500)
# fin_phase_frame = fin_phase_frame.loc[time_bool.values]

# Convert freq_ind to actual freq
# fin_phase_frame.reset_index(inplace=True, drop=False, level = 'freq')
# fin_phase_frame['freq'] = freq_vec[fin_phase_frame['freq']]
# fin_phase_frame['freq'] = fin_phase_frame['freq'].astype(int)
# # Drop 0 Hz
# fin_phase_frame = fin_phase_frame.loc[fin_phase_frame['freq'] != 0]
# fin_phase_frame.set_index('freq', append=True, inplace=True)

##############################

# fin_phase_frame.reset_index(inplace=True, drop=False)

# For each neuron, roll over time to generate histograms,
# aggregate across trials for tastes separately (we might merge tastes later)
# group_name_list = ['basename', 'spikes_region',
#                    'phase_region', 'nrn_num', 'freq']
# wanted_cols = fin_phase_frame[group_name_list]
# # Convert cols to tuple
# wanted_cols = [tuple(x) for x in wanted_cols.values]
# unique_cols = list(set(wanted_cols)) 

# test_ind = unique_cols[0]
# # Query for this test_ind in fin_phase_frame
# test_frame = fin_phase_frame.loc[wanted_cols == test_ind]
# 
# def extract_values(frame, cols, values):
#     bools = []
#     for this_col, this_val in zip(cols, values):
#         bools.append(frame[this_col] == this_val)
#     fin_bool = np.logical_and.reduce(bools)
#     return frame.loc[fin_bool]
# 
# extract_values(fin_phase_frame, group_name_list, test_ind)
# 
# group_obj = fin_phase_frame.groupby(group_name_list)
# group_list = list(group_obj)

########################################
# Run rayleigh test on each neuron


# meta_list = [x[0] for x in group_list]
# group_rolling_rayleigh = parallelize(temp_rolling_rayleigh, group_list)
# 
# for meta, dat in tqdm(zip(meta_list, group_rolling_rayleigh)):
#     dat[group_name_list] = meta
# 
# group_rayleigh = pd.concat(group_rolling_rayleigh)
# group_rayleigh = group_rayleigh.dropna()

########################################
# For each session, load actual rayleigh and shuffles
# Calculate mean shuffle
# Save everything in a dataframe for easy access

hdf5_path = os.path.join(save_dir, 'rolling_rayleigh.h5')

with tables.open_file(hdf5_path, mode='r') as hf5:
    subdirs = hf5.list_nodes('/')
    subdirs = [x._v_name for x in subdirs]
shuffle_subdirs = [x for x in subdirs if 'sh' in x]

ind_cols = ['basename', 'spikes_region', 'phase_region', 'nrn_num']
all_pivot = []
all_inds = []
wanted_var = 'scaled_z_stat'
for this_basename in tqdm(dir_basenames):
    # Get actual data
    actual = pd.read_hdf(hdf5_path, key=f'/actual/{this_basename}')
    actual_pivot, actual_inds = process_rayleigh(actual, var=wanted_var, add_id='actual')
    all_pivot.extend(actual_pivot)
    all_inds.extend(actual_inds)
    # Get mean shuffle data
    # Try to load shuffles
    shuffle = []
    for x in shuffle_subdirs:
        try:
            this_load = pd.read_hdf(hdf5_path, key=f'/{x}/{this_basename}')
            shuffle.append(this_load)
        except:
            print(f'Could not load {x}/{this_basename}')
    for this_shuffle, this_sh_dir in zip(shuffle, shuffle_subdirs):
        this_shuffle['sh_dir'] = this_sh_dir
    shuffle = pd.concat(shuffle)
    # Group by sh_dir and take mean
    shuffle_mean = shuffle.groupby([*ind_cols, 'time', 'freq']).mean() 
    shuffle_mean.reset_index(inplace=True)
    shuffle_mean_pivot, shuffle_mean_inds = process_rayleigh(shuffle_mean, var='z_stat', add_id='shuffle')
    all_pivot.extend(shuffle_mean_pivot)
    all_inds.extend(shuffle_mean_inds)

inds_frame = pd.DataFrame(all_inds,
                          columns=[*ind_cols, 'data_type'])
inds_frame['pivots'] = all_pivot

# For each neuron, subtract mean shuffle from actual
grouped_pivot_frame = list(inds_frame.groupby(ind_cols))
grouped_inds = [x[0] for x in grouped_pivot_frame]
grouped_pivot = [x[1] for x in grouped_pivot_frame]
actual_pivot = [x.loc[x['data_type'] == 'actual', 'pivots'].values[0] for x in grouped_pivot]
shuffle_pivot = [x.loc[x['data_type'] == 'shuffle', 'pivots'].values[0] for x in grouped_pivot]
diff_pivot = [x - y for x, y in zip(actual_pivot, shuffle_pivot)]

diff_frame = pd.DataFrame(grouped_inds, columns=ind_cols)
diff_frame['actual'] = actual_pivot
diff_frame['shuffle'] = shuffle_pivot
diff_frame['diff'] = diff_pivot
# Save diff_frame
diff_frame.to_hdf(os.path.join(save_dir, 'diff_frame.h5'),
                  key=f'{wanted_var}_diff_frame')

# Group by ['spikes_region', 'phase_region']
group_rayleigh_list = list(diff_frame.groupby(['spikes_region', 'phase_region']))
group_rayleigh_inds = [x[0] for x in group_rayleigh_list]
group_rayleigh_str = [f'{x[0]}->{x[1]}' for x in group_rayleigh_inds]
group_rayleigh_dat = [x[1] for x in group_rayleigh_list]

# Keep only data with len(freq_vec)-1 (because 0 was dropped)
grouped_actual = [x['actual'].values for x in group_rayleigh_dat]
grouped_actual = [[y for y in x if y.shape == (len(freq_vec)-1, len(rayleigh_time_vec))] \
        for x in grouped_actual]
grouped_actual = [np.stack(x) for x in grouped_actual]
mean_actual = [np.mean(x, axis=0) for x in grouped_actual]

grouped_shuffle = [x['shuffle'].values for x in group_rayleigh_dat]
grouped_shuffle = [[y for y in x if y.shape == (len(freq_vec)-1, len(rayleigh_time_vec))] \
        for x in grouped_shuffle]
grouped_shuffle = [np.stack(x) for x in grouped_shuffle]
mean_shuffle = [np.mean(x, axis=0) for x in grouped_shuffle]

grouped_diff = [x['diff'].values for x in group_rayleigh_dat]
grouped_diff = [[y for y in x if y.shape == (len(freq_vec)-1, len(rayleigh_time_vec))] \
        for x in grouped_diff]
grouped_diff = [np.stack(x) for x in grouped_diff]
mean_diff = [np.mean(x, axis=0) for x in grouped_diff]

########################################
# Generate plots
########################################

# Make a 2D array of plots with grouped_rayleigh_str as rows and 
# data_type as columns

plot_data = np.array([mean_actual, mean_shuffle, mean_diff])
data_types = ['actual', 'shuffle', 'diff']
plot_data = np.swapaxes(plot_data, 0, 1)

# Plot
fig, axes = plt.subplots(nrows=plot_data.shape[0], ncols=plot_data.shape[1],
                         sharex=True, sharey=True, figsize=(10, 5))
for i, this_row in enumerate(axes):
    for j, this_ax in enumerate(this_row):
        im = this_ax.pcolormesh(rayleigh_time_vec, freq_vec[1:], plot_data[i, j, :, :],
                           cmap='jet') 
        this_ax.set_title(f'{group_rayleigh_str[i]}: {data_types[j]}')
        this_ax.set_ylabel('Frequency (Hz)')
        this_ax.set_xlabel('Time (s)')
        this_ax.set_ylim([0, 15])
        plt.colorbar(im, ax=this_ax)
        #this_ax.set_xlim([0, 1])
plt.tight_layout()
plt.suptitle(f'{wanted_var} Rayleigh plots')
plt.savefig(os.path.join(plot_dir, f'{wanted_var}_rayleigh_plots.png'),
            bbox_inches='tight')
plt.close()
#plt.show()

########################################
# Create unique identifier for each neuron
group_rayleigh['nrn_id'] = group_rayleigh['basename'] + '_' + \
    group_rayleigh['nrn_num'].astype(str) + group_rayleigh['spikes_region']
# Convert nrn_id to categorical number
group_rayleigh['nrn_id'] = pd.Categorical(group_rayleigh['nrn_id'])

nrn_id_map = dict(enumerate(group_rayleigh['nrn_id'].cat.categories))
group_rayleigh['nrn_id'] = group_rayleigh['nrn_id'].cat.codes

###############
# Significant p-values
alpha = 0.05
group_rayleigh['p_val_sig'] = group_rayleigh['p_val'] < alpha

###############
# Drop irrelevant columns
group_rayleigh.drop(columns=['basename', 'nrn_num'], inplace=True)

mean_group_rayleigh = group_rayleigh.groupby(
        ['spikes_region', 'phase_region', 'nrn_id']).mean().reset_index()
region_mean_group_rayleigh = [x[1] for x in list(mean_group_rayleigh.groupby('spikes_region'))]
region_group_rayleigh = [x[1] for x in list(group_rayleigh.groupby('spikes_region'))]

# Count number of unique neurons by spike region
nrn_count = group_rayleigh.groupby(
    ['spikes_region', 'nrn_id']).size().reset_index()
nrn_count = nrn_count.groupby('spikes_region').size().reset_index()
nrn_count.columns = ['spikes_region', 'nrn_count']

########################################
# Further analysis using scaled_z_stat
subset_cols = ['spikes_region', 'phase_region','freq', 
               'time',  'time_bins',
               'scaled_z_stat', 'p_val_sig',
               ]
imp_dat = group_rayleigh[subset_cols]
imp_dat = imp_dat.groupby(subset_cols[:-2]).mean().reset_index()
region_dat = [imp_dat[imp_dat['spikes_region'] == x] for x in imp_dat['spikes_region'].unique()]

# Generate timeseries of scaled_z_stat
region_z_stat_pivot = [x.pivot(index='freq', columns='time', values='scaled_z_stat') for x in region_dat]
region_pval_sig_pivot = [x.pivot(index='freq', columns='time', values='p_val_sig') for x in region_dat]
direction_names = [f'{x.spikes_region.unique()[0].upper()} spikes --> {x.phase_region.unique()[0].upper()} field' for x in region_dat]
