"""
Perform granger causality anaylsis on all data and save results to file
:
# 1) Load data
# 2) Preprocess data:
#     a) Remove trials with artifacts
#     b) Detrend single-trial data
#     c) Remove temporal mean from single-trial data
#     d) Dvidide by temporal standard deviation
#     e) Subtract mean across trials from each trial (for each timepoint)
#     f) Divide by standard deviation across trials (for each timepoint)
# 3) Perform Augmented Dickey-Fuller test on each channel to check for stationarity
# 4) Perform Granger Causality test on each channel pair
# 5) Test for good fitting by checking that residuals are white noise
# 6) Calculate significance of granger causality by shuffling trials
# 7) Plot results
#
# References:
# Ding, Mingzhou, et al. “Short-Window Spectral Analysis of Cortical Event-Related Potentials by Adaptive Multivariate Autoregressive Modeling: Data Preprocessing, Model Validation, and Variability Assessment.” Biological Cybernetics, vol. 83, no. 1, June 2000, pp. 35–45, https://doi.org/10.1007/s004229900137.

"""

from glob import glob
import tables
import os
import numpy as np
import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
granger_causality_path = \
    '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality'
sys.path.append(ephys_data_dir)
sys.path.append(granger_causality_path)
import granger_utils as gu
from ephys_data import ephys_data

# Log all stdout and stderr to a log file in results folder
sys.stdout = open(os.path.join(granger_causality_path, 'stdout.txt'), 'w')
sys.stderr = open(os.path.join(granger_causality_path, 'stderr.txt'), 'w')

############################################################
# Load Data
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

for dir_name in dir_list:
    try:
        #dir_name = dir_list[0]
        basename = dir_name.split('/')[-1]
        # Assumes only one h5 file per dir
        h5_path = glob(dir_name + '/*.h5')[0]

        print(f'Processing {basename}')

        dat = ephys_data(dir_name)
        # Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
        lfp_channel_inds, region_lfps, region_names = \
            dat.return_representative_lfp_channels()

        flat_region_lfps = np.reshape(
            region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))

        ############################################################
        # Preprocessing
        ############################################################

        # 1) Remove trials with artifacts
        #good_lfp_trials_bool = dat.lfp_processing.return_good_lfp_trial_inds(dat.all_lfp_array)
        good_lfp_data = dat.lfp_processing.return_good_lfp_trials(
            flat_region_lfps)

        ############################################################
        # Compute Granger Causality
        ############################################################
        this_granger = gu.granger_handler(good_lfp_data, 
                                          multitaper_time_halfbandwidth_product=1)
        this_granger.get_granger_sig_mask()

        ############################################################
        # Save Results to HDF5
        ############################################################
        with tables.open_file(h5_path, 'r+') as h5:
            save_path = '/ancillary_analysis/granger_causality'
            if save_path in h5:
                h5.remove_node(os.path.dirname(save_path),
                               'granger_causality',
                               recursive=True)
            h5.create_group(os.path.dirname(save_path), 'granger_causality')
            vals = [this_granger.granger_actual,
                    this_granger.masked_granger,
                    this_granger.mask_array,
                    this_granger.wanted_window,
                    this_granger.c_actual.time,
                    this_granger.c_actual.frequencies]
            names = ['granger_actual',
                     'masked_granger',
                     'mask_array',
                     'wanted_window',
                     'time_vec',
                     'freq_vec']
            for val, name in zip(vals, names):
                h5.create_array(save_path, name, val)
        del this_granger, dat
    except:
        print(f'Failed to process {basename}')
        continue