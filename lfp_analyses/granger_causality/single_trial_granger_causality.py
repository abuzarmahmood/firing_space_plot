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
this_granger = gu.granger_handler(good_lfp_data)
this_granger.preprocess_and_check_stationarity()

############################################################
# Parametric Granger Causality
# NOTE: It might be that the across trial preprocessing 
# is messing with single trial analysis

wanted_inds = [2000, 3500]
input_data = this_granger.preprocessed_data
input_data = input_data[...,wanted_inds[0]:wanted_inds[1]]
# Reshape to (n_trials, n_channels, n_timepoints)
input_data = np.transpose(input_data, (1,0,2))

# Use BIC of VAR model to determine optimal lag
max_lag = 100

temp_fit = lambda x: VAR(x.T).fit(maxlags=max_lag, ic='bic').k_ar
model_orders = parallelize(temp_fit, input_data)
chosen_model_order = int(np.median(model_orders))

############################################################
temp_gc_full = \
        lambda x: grangercausalitytests(x.T, [chosen_model_order], verbose=False)
temp_gc_final = lambda x: temp_gc_full(x)[chosen_model_order][0]['params_ftest']

gc_results_out = [parallelize(temp_gc_final, this_dat) \
        for this_dat in [input_data, input_data[:,::-1]]]
zipped_results = [list(zip(*x)) for x in gc_results_out]
cat_results = [[x,y] for x,y in zip(*zipped_results)]
f_stat, p_val, df_num, k_order = [np.stack(x) for x in cat_results]

############################################################
# Plots
############################################################

# Plot f-statistic of each direction against the other
# Check which side f-stats lean to
f_stat_ratio = f_stat[0]/f_stat[1]

fig,ax = plt.subplots(1,2)
ax[0].scatter(*f_stat)
ax[0].plot([0,np.max(f_stat)], [0,np.max(f_stat)], color='red')
ax[0].set_aspect('equal')
ax[1].hist(np.log10(f_stat_ratio), bins = 30)
ax[1].axvline(np.log10(1), color='red')
ax[1].axvline(np.log10(f_stat_ratio).mean(),
              color = 'black', linestyle = '--')
ax[1].set_title(f'Ration > 1: {np.mean(f_stat_ratio > 1):.2f}')
plt.show()

# Plot log p-value of each direction against the other
log10_p_val = np.log10(p_val)
alpha = 0.05
log10_alpha = np.log10(alpha)
sig_trials = log10_p_val < log10_alpha

mean_sig_trials = np.mean(sig_trials, axis=-1)

plt.scatter(*log10_p_val)
plt.axhline(log10_alpha, color='red')
plt.axvline(log10_alpha, color='red')
plt.show()


plt.scatter(log10_p_val, f_stat, s=5)
plt.axvline(log10_alpha, color='red')
plt.xlabel('log10 p value')
plt.ylabel('F statistic')
plt.title('Granger Causality' + '\n' + \
        f'Sig Trial Fraction: {np.mean(sig_trials):.2f}')
plt.show()
