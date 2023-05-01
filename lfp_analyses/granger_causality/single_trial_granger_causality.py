"""
Resources:

https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
https://phdinds-aim.github.io/time_series_handbook/04_GrangerCausality/04_GrangerCausality.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html
https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html
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
#import granger_utils as gu
from ephys_data import ephys_data
import pylab as plt
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import pingouin as pg
import seaborn as sns

def parallelize(func, arg_list, n_jobs=10):
    return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in tqdm(arg_list))

# Log all stdout and stderr to a log file in results folder
#sys.stdout = open(os.path.join(granger_causality_path, 'stdout.txt'), 'w')
#sys.stderr = open(os.path.join(granger_causality_path, 'stderr.txt'), 'w')

############################################################
# Load Data
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

dir_name = dir_list[0]
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

flat_region_lfps = np.reshape(
    region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))

############################################################
# Preprocessing
############################################################

# 1) Remove trials with artifacts
#good_lfp_trials_bool = dat.lfp_processing.return_good_lfp_trial_inds(dat.all_lfp_array)
trial_labels = np.repeat(np.arange(region_lfps.shape[1]), region_lfps.shape[2])
good_lfp_trials_bool = dat.lfp_processing.return_good_lfp_trial_inds(
    flat_region_lfps)
good_trial_labels = trial_labels[good_lfp_trials_bool]
good_lfp_data = dat.lfp_processing.return_good_lfp_trials(
    flat_region_lfps)

############################################################
# Compute Granger Causality
############################################################
this_granger = gu.granger_handler(good_lfp_data)
this_granger.preprocess_and_check_stationarity()

############################################################
# Parametric Granger Causality
# NOTE: It might be that the across trial preprocessing 
# is messing with single trial analysis

#epoch_names = ['pre','early', 'middle', 'late', 'all']
#epoch_lims = [[1500,2000],[2000,2500],[2500,3000],[3000,3500],[2000, 3500]]
epoch_starts = np.arange(1500, 3000, 25)
epoch_ends = epoch_starts + 300
epoch_lims = list(zip(epoch_starts, epoch_ends))
epoch_names = epoch_ends #[int(np.mean(x)) for x in epoch_lims]
input_data = this_granger.preprocessed_data
input_data_epochs = [input_data[...,inds[0]:inds[1]] for inds in epoch_lims]
# Reshape to (n_trials, n_channels, n_timepoints)
input_data_epochs = [np.transpose(x, (1,0,2)) for x in input_data_epochs]
#[x.shape for x in input_data_epochs]

# # Use BIC of VAR model to determine optimal lag
# max_lag = 100
# 
# temp_fit = lambda x: VAR(x.T).fit(maxlags=max_lag, ic='bic').k_ar
# model_orders = [parallelize(temp_fit, x) for x in input_data_epochs]
# chosen_model_order_list = [int(np.median(x)) for x in model_orders]
# chosen_model_order = int(np.median(chosen_model_order_list))

############################################################
temp_gc_full = \
        lambda x: grangercausalitytests(x.T, [chosen_model_order], verbose=False)
temp_gc_final = lambda x: temp_gc_full(x)[chosen_model_order][0]['params_ftest']

gc_results_out = [[parallelize(temp_gc_final, this_dat) \
        for this_dat in [x, x[:,::-1]]] \
        for x in input_data_epochs]

# Convert output to pandas dataframe for easier handling
direction_names = [x.join(region_names) for x in ['-->','<--']]

# gc_results_out : [epoch][direction][trial][f_stat, p_val, df_num, k_order]
gc_results_flat = []
for epoch_num, epoch in enumerate(gc_results_out):
    for direction_num, direction in enumerate(epoch):
        for trial_num, trial in enumerate(direction):
            gc_results_flat.append(
                    [epoch_names[epoch_num], direction_names[direction_num], trial_num] + list(trial))
gc_results_df = pd.DataFrame(gc_results_flat, columns = \
        ['epoch', 'direction', 'trial', 'f_stat', 'p_val', 'df_num', 'k_order'])
gc_results_df['taste_num'] = good_trial_labels[gc_results_df['trial']]
gc_results_df['taste'] = gc_results_df['taste_num'].map(lambda x: taste_names[x])
# Log transform f-statistic for parametric testing
gc_results_df['log_f_stat'] = np.log10(gc_results_df['f_stat'])

# Test difference in log f-statistic between epochs, direction, and taste
pg.normality(data=gc_results_df, dv='log_f_stat', group='epoch')
pg.normality(data=gc_results_df, dv='log_f_stat', group='direction')
pg.normality(data=gc_results_df, dv='log_f_stat', group='taste')

pg.anova(data=gc_results_df, dv='log_f_stat', 
         between=['epoch', 'direction', 'taste'])

# Perform pairwise comparisons between epochs, direction, and taste
pg.pairwise_ttests(data=gc_results_df, dv='log_f_stat',
                   between=['epoch', 'direction', 'taste'],
                   )

## Plot log f-statistic for each epoch, direction, and taste_names using seaborn
#sns.catplot(x='epoch', y='log_f_stat', hue='direction',
#            col='taste', data=gc_results_df, kind='violin')
#plt.show()

# Plot log f-statistic for each for each taste and direction over epochs as lineplots
sns.relplot(x='epoch', y='log_f_stat', hue='direction',
            col='taste', data=gc_results_df, kind='line')
plt.show()
