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

############################################################
# Load Data
############################################################

dir_name = sys.argv[1]
#dir_name = dir_list[0]
basename = dir_name.split('/')[-1]
# Assumes only one h5 file per dir
h5_path = glob(dir_name + '/*.h5')[0]

print(f'Processing {basename}')

dat = ephys_data(dir_name)
dat.get_info_dict()

# # Pull out longer trial-durations for LFP
# dat.default_lfp_params['trial_durations'] = [10000,10000]
# dat.extract_lfps()

taste_names = dat.info_dict['taste_params']['tastes']
# Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
lfp_channel_inds, region_lfps, region_names = \
    dat.return_representative_lfp_channels()

taste_lfps = [x for x in region_lfps.swapaxes(0,1)]

flat_region_lfps = np.reshape(
    region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))
flat_taste_nums = np.repeat(np.arange(len(taste_names)),
                            taste_lfps[0].shape[1])

lfp_set_names = taste_names.copy()
lfp_set_names.append('all')

############################################################
# Preprocessing
############################################################

# NOTE: Preprocess all tastes together so that separate
# preprocessing does not introduce systematic differences
# between tastes

# 1) Remove trials with artifacts
good_lfp_trials_bool = \
        dat.lfp_processing.return_good_lfp_trial_inds(flat_region_lfps)
good_lfp_trials = flat_region_lfps[:,good_lfp_trials_bool]
good_taste_nums = flat_taste_nums[good_lfp_trials_bool]


# 2) Preprocess data
this_granger = gu.granger_handler(good_lfp_trials)
this_granger.preprocess_and_check_stationarity()
preprocessed_data = this_granger.preprocessed_data

# Make list of data according to lfp_set_names 
preprocessed_lfp_data = [
        preprocessed_data[:, good_taste_nums == x]\
        for x in range(len(taste_names))
        ]
preprocessed_lfp_data.append(preprocessed_data)

############################################################
# Compute Granger Causality
############################################################
# Create group in h5 if needed
with tables.open_file(h5_path, 'r+') as h5:
    save_path = '/ancillary_analysis/granger_causality'
    if save_path in h5:
        h5.remove_node(os.path.dirname(save_path),
                       'granger_causality',
                       recursive=True)
    h5.create_group(os.path.dirname(save_path), 'granger_causality')


# Only process 'all' trials together
lfp_set_names = [lfp_set_names[-1]]
preprocessed_lfp_data = [preprocessed_lfp_data[-1]]

# We don't preprocess data again here because we provide
# preprocessed data from above
for num, this_dat in enumerate(preprocessed_lfp_data):
    print(f'Processing {lfp_set_names[num]}')
    this_granger = gu.granger_handler(
                              this_dat,
                              multitaper_time_halfbandwidth_product=1,
                              preprocess=False,
                              wanted_window = [9000, 12000],
                              )
    this_granger.get_granger_sig_mask()

    ############################################################
    # Save Results to HDF5
    ############################################################
    this_taste_name = lfp_set_names[num]

    with tables.open_file(h5_path, 'r+') as h5:
        h5.create_group(save_path, this_taste_name)
        fin_save_path = os.path.join(save_path, this_taste_name)
        vals = [this_granger.granger_actual,
                this_granger.masked_granger,
                this_granger.mask_array,
                this_granger.wanted_window,
                this_granger.time_vec,
                this_granger.freq_vec,
                region_names]
        names = ['granger_actual',
                 'masked_granger',
                 'mask_array',
                 'wanted_window',
                 'time_vec',
                 'freq_vec',
                 'region_names']
        for val, name in zip(vals, names):
            h5.create_array(fin_save_path, name, val)
    del this_granger
del dat, taste_lfps, region_lfps, flat_region_lfps
del this_dat, preprocessed_data, preprocessed_lfp_data
del good_lfp_trials, good_lfp_trials_bool, good_taste_nums
