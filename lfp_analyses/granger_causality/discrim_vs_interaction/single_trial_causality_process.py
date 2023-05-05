"""
Resources:

https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
https://phdinds-aim.github.io/time_series_handbook/04_GrangerCausality/04_GrangerCausality.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html
"""

import os
from glob import glob
import numpy as np
import sys
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from itertools import product
from scipy.signal import butter, filtfilt
import tables

ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
granger_causality_path = \
    '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality'
sys.path.append(ephys_data_dir)
sys.path.append(granger_causality_path)
import granger_utils as gu
from ephys_data import ephys_data


def parallelize(func, arg_list, n_jobs=10):
    return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in tqdm(arg_list))


def filter_data(data, low_pass, high_pass, sampling_rate, axis=-1):
    m, n = butter(
        2,  # Order of filter
        [
            2.0*int(low_pass)/sampling_rate,
            2.0*int(high_pass)/sampling_rate
             ],
        btype='bandpass')
    filt_el = filtfilt(m, n, data, axis=axis)
    return filt_el

# Log all stdout and stderr to a log file in results folder
#sys.stdout = open(os.path.join(granger_causality_path, 'stdout.txt'), 'w')
#sys.stderr = open(os.path.join(granger_causality_path, 'stderr.txt'), 'w')

############################################################
# Define iterators
############################################################

epoch_lims = [[300,800], [800,1300]]
epoch_names = ['middle', 'late']

frequency_lims = \
        [[1,100],[1,30],[30,100],[4,8],[8,12],[12,30],[30,60],[60,100]]
frequency_names = \
        ['all','low','high','theta','alpha','beta','low_gamma','high_gamma']

dir_list_path = \
        '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

all_epoch_inds = np.arange(len(epoch_lims))
all_frequency_inds = np.arange(len(frequency_lims))
all_dir_inds = np.arange(len(dir_list))

iter_inds = list(product(all_dir_inds, all_frequency_inds, all_epoch_inds))

############################################################
# Load Data
############################################################

for this_iter in tqdm(iter_inds):
    #this_iter = iter_inds[0]

    dir_ind = this_iter[0]
    frequency_ind = this_iter[1]
    epoch_ind = this_iter[2]

    this_epoch = epoch_lims[epoch_ind]
    this_frequency = frequency_lims[frequency_ind]
    dir_name = dir_list[dir_ind]

    print(f'Processing {dir_name}')
    print(f'Epoch : {this_epoch}')
    print(f'Frequency : {this_frequency}')

    this_epoch_name = epoch_names[epoch_ind]
    this_frequency_name = frequency_names[frequency_ind]
    basename = dir_name.split('/')[-1]

    # Assumes only one h5 file per dir
    h5_path = glob(dir_name + '/*.h5')[0]

    print(f'Processing {basename}')

    dat = ephys_data(dir_name)
    dat.get_info_dict()
    taste_names = dat.info_dict['taste_params']['tastes']
    # Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
    lfp_channel_inds, region_lfps, region_names = \
        dat.return_representative_lfp_channels()

    # Sort everything by region_names
    sort_inds = np.argsort(region_names)
    region_names = np.array(region_names)[sort_inds]
    region_lfps = region_lfps[sort_inds]
    lfp_channel_inds = np.array(lfp_channel_inds)[sort_inds]

    flat_region_lfps = np.reshape(
        region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))

    ############################################################
    # Preprocessing
    ############################################################

    # 1) Remove trials with artifacts
    trial_labels = np.repeat(np.arange(region_lfps.shape[1]), region_lfps.shape[2])
    good_lfp_trials_bool = dat.lfp_processing.return_good_lfp_trial_inds(
        flat_region_lfps)
    good_lfp_trials_ind = np.where(good_lfp_trials_bool)[0]
    good_trial_labels = trial_labels[good_lfp_trials_bool]
    good_lfp_data = dat.lfp_processing.return_good_lfp_trials(
        flat_region_lfps)

    # 2) Bandpass according to frequency range
    filt_lfp_dat = filter_data(
        good_lfp_data, this_frequency[0], this_frequency[1], 1000, axis=-1)

    # 3) Preprocess for granger causality
    this_granger = gu.granger_handler(filt_lfp_dat)
    this_granger.preprocess_data()
    #this_granger.preprocess_and_check_stationarity()

    # 4) Cut to specified epoch
    input_data = this_granger.preprocessed_data
    input_data_epochs = input_data[...,this_epoch[0]:this_epoch[1]]
    # Reshape to (n_trials, n_channels, n_timepoints)
    input_data_epochs = np.transpose(input_data_epochs, (1,0,2))

    ############################################################
    # Compute Granger Causality
    ############################################################

    # Use BIC of VAR model to determine optimal lag
    max_lag = 100

    def temp_fit(x):
        try:
           return VAR(x.T).fit(maxlags=max_lag, ic='bic').k_ar
        except:
            return np.nan
    
    model_orders = parallelize(temp_fit, input_data_epochs) 
    chosen_model_order = int(np.nanmedian(model_orders))
    if np.isnan(chosen_model_order):
        k_determined = False
        chosen_model_order = 10
    else:
        k_determined = True

    ############################################################
    temp_gc_full = \
            lambda x: grangercausalitytests(x.T, [chosen_model_order], verbose=False)
    temp_gc_final = lambda x: temp_gc_full(x)[chosen_model_order][0]['params_ftest']

    # Process causality for both directions
    gc_results_out = [parallelize(temp_gc_final, this_dat) \
            for this_dat in \
            [input_data_epochs, input_data_epochs[:,::-1]]] 

    # Convert output to pandas dataframe for easier handling
    direction_names = [x.join(region_names) for x in ['-->','<--']]

    # gc_results_out : [epoch][direction][trial][f_stat, p_val, df_num, k_order]
    gc_results_flat = []
    for direction_num, direction in enumerate(gc_results_out):
        for trial_num, trial in enumerate(direction):
            gc_results_flat.append(
                    [direction_names[direction_num], trial_num] + list(trial))

    gc_results_df = pd.DataFrame(
            gc_results_flat,
            columns = [
                'direction',
                'trial',
                'f_stat',
                'p_val',
                'df_num',
                'k_order'
                ]
            )
    gc_results_df['k_determined'] = k_determined
    # Replace trial counter with actual trial number
    gc_results_df['trial'] = good_lfp_trials_ind[gc_results_df['trial']]
    gc_results_df['taste_num'] = trial_labels[gc_results_df['trial']]
    gc_results_df['epoch_name'] = this_epoch_name
    gc_results_df['frequency_name'] = this_frequency_name
    gc_results_df['epoch'] = str(this_epoch)
    gc_results_df['frequency'] = str(this_frequency)
    gc_results_df['basename'] = basename
    gc_results_df['taste'] = \
            gc_results_df['taste_num'].map(lambda x: taste_names[x])

    save_path = '/ancillary_analysis/granger_causality/single_trial'

    # If array present, append to it
    with tables.open_file(h5_path, 'r') as h5:
        # If this is the first of a new dataset, overwrite
        if frequency_ind + epoch_ind == 0:
            present_bool = False
        # If not, check whether frame is present
        if save_path in h5:
            present_bool = True
        else:
            present_bool = False
    if present_bool:
        df = pd.read_hdf(h5_path, save_path)
        gc_results_df = gc_results_df.append(df)
        gc_results_df.drop_duplicates(inplace=True)
    gc_results_df.to_hdf(h5_path, save_path)
