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
import granger_utils as gu
from ephys_data import ephys_data
import pylab as plt
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

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

# ############################################################
# # Manual preprocessing 
# ############################################################
# 
# def plot_dat(this_dat, **kwargs):
#     """
#     this_dat : (n_channels, n_trials, n_timepoints)
#     """
#     fig,ax = plt.subplots(1,2, sharex=True, sharey=True)
#     ax[0].imshow(this_dat[0], aspect='auto', **kwargs)
#     ax[1].imshow(this_dat[1], aspect='auto', **kwargs)
#     plt.show()
# 
# plot_dat(good_lfp_data)
# 
# from scipy.signal import detrend
# detrended_data = detrend(good_lfp_data, axis=-1) 
# plot_dat(detrended_data)
# 
# mean_removed_data = detrended_data - np.mean(detrended_data, axis=-1, keepdims=True)
# plot_dat(mean_removed_data)
# 
# std_divided_data = mean_removed_data / np.std(mean_removed_data, axis=-1, keepdims=True)
# plot_dat(std_divided_data, vmin=-1, vmax=1)
# 
# mean_across_trials_subtracted_data = \
#     std_divided_data - np.mean(std_divided_data, axis=1, keepdims=True)
# plot_dat(mean_across_trials_subtracted_data, vmin=-1, vmax=1)
# 
# std_across_trials_divided_data = \
#     mean_across_trials_subtracted_data / \
#     np.std(mean_across_trials_subtracted_data, axis=1, keepdims=True)
# 
# wanted_time_range = [1500, 4000]
# plot_dat(
#         std_across_trials_divided_data[..., wanted_time_range[0]:wanted_time_range[1]], 
#         vmin=-1, vmax=1)

############################################################
# Compute Granger Causality
############################################################
this_granger = gu.granger_handler(
        good_lfp_data, 
        multitaper_time_halfbandwidth_product=1)
this_granger.preprocess_and_check_stationarity()
#this_granger.get_granger_sig_mask()

############################################################
# Parametric Granger Causality
# NOTE: It might be that the across trial preprocessing 
# is messing with single trial analysis

input_data = this_granger.preprocessed_data
wanted_inds = [2000, 1500]
input_data = input_data[...,wanted_inds[0]:wanted_inds[1]]

# Reshape to (n_trials, n_channels, n_timepoints)
input_data = np.transpose(input_data, (1,0,2))

# Use BIC of VAR model to determine optimal lag
max_lag = 100

trial_num = 0
this_dat = input_data[trial_num,...].T
plt.plot(this_dat); plt.show()

fig,ax = plt.subplots(1,2)
ax[0].imshow(input_data[:,0], aspect='auto', vmin=-1, vmax=1)
ax[1].imshow(input_data[:,1], aspect='auto', vmin=-1, vmax=1)
plt.show()

############################################################
temp_fit = lambda x: VAR(x.T).fit(maxlags=max_lag, ic='bic').k_ar
def parallelize(func, arg_list, n_jobs=10):
    return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in tqdm(arg_list))
#result = VAR(this_dat).fit(maxlags=max_lag, ic='bic')
model_orders = parallelize(temp_fit, input_data)
#plt.hist(model_orders); plt.show()
chosen_model_order = int(np.median(model_orders))

############################################################

temp_gc_full = \
        lambda x: grangercausalitytests(x.T, [chosen_model_order], verbose=False)
temp_gc_final = lambda x: temp_gc_full(x)[chosen_model_order][0]['params_ftest']

#gc_result = grangercausalitytests(this_dat, [chosen_model_order], verbose=False)
#gc_result = temp_gc_final(this_dat.T)

gc_results_out = [parallelize(temp_gc_final, this_dat) \
        for this_dat in [input_data, input_data[:,::-1]]]
zipped_results = [list(zip(*x)) for x in gc_results_out]
cat_results = [[x,y] for x,y in zip(*zipped_results)]
f_stat, p_val, df_num, k_order = [np.stack(x) for x in cat_results]

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

# Plot most significant trial and the least significant trial
most_sig_trial = np.argmax(log10_p_val)


# Plot mean single trial granger causality
mean_single_trial_granger = np.nanmean(this_granger.granger_single_trial, axis=0)

wanted_freq_range = [0,100]
freq_vec = this_granger.c_single_trial[0].frequencies
freq_inds = np.where(np.logical_and(freq_vec >= wanted_freq_range[0],
                                    freq_vec <= wanted_freq_range[1]))[0]
freq_vec = freq_vec[freq_inds]
mean_single_trial_granger_plot = mean_single_trial_granger[:,freq_inds]
all_freq_single_trial_granger = mean_single_trial_granger_plot.sum(axis=1)

wanted_time = [1.5,4]
time_vec = this_granger.c_single_trial[0].time
time_inds = np.where(np.logical_and(time_vec >= wanted_time[0],
                                    time_vec <= wanted_time[1]))[0]
time_vec = time_vec[time_inds]
mean_single_trial_granger_plot = mean_single_trial_granger_plot[time_inds,:]

fig,ax = plt.subplots(1,2, 
                      sharex=True, sharey=True,
                      )
ax[0].imshow(mean_single_trial_granger_plot[:,:,0,1].T, aspect='auto')
ax[1].imshow(mean_single_trial_granger_plot[:,:,1,0].T, aspect='auto')
ax[0].set_title('A->B')
ax[1].set_title('B->A')
plt.show()

fig,ax = plt.subplots(1,2,
                      sharex=True, sharey=True,
                      )
ax[0].plot(time_vec, all_freq_single_trial_granger[...,0,1])
ax[1].plot(time_vec, all_freq_single_trial_granger[...,1,0])
plt.show()
