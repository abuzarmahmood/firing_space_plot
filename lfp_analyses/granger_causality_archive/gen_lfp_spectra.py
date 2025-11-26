"""
Generate LFP spectra for:
    1. Raw LFP
    2. Granger preprocessed LFP

at different time points:
    1. Baseline
    2. Stimulus

"""

from glob import glob
import tables
import os
import numpy as np
import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
granger_causality_path = \
    '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality'
process_scripts_path = os.path.join(granger_causality_path,'process_scripts')
sys.path.append(ephys_data_dir)
sys.path.append(granger_causality_path)
sys.path.append(process_scripts_path)
import granger_utils as gu
from ephys_data import ephys_data
import multiprocessing as mp
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import linregress

plot_dir = os.path.join(granger_causality_path,'plots','lfp_spectra')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

artifacts_dir = os.path.join(granger_causality_path,'artifacts')
if not os.path.isdir(artifacts_dir):
    os.mkdir(artifacts_dir)

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

all_mean_raw_amplitude_array = []
all_mean_processed_amplitude_array = []
region_names_list = []

for dir_name in tqdm(dir_list):
    # dir_name = dir_list[0]

    basename = dir_name.split('/')[-1]
    # Assumes only one h5 file per dir
    h5_path = glob(dir_name + '/*.h5')[0]

    print(f'Processing {basename}')

    dat = ephys_data(dir_name)
    dat.get_info_dict()
    dat.stft_params['time_range_tuple'] = [0, 20] 
    dat.stft_params['max_freq'] = 100
    # dat.get_stft(recalculate=True, dat_type=['amplitude'])

    ##############################
    # Pull out wanted data

    taste_names = dat.info_dict['taste_params']['tastes']
    # Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
    lfp_channel_inds, region_lfps, region_names = \
        dat.return_representative_lfp_channels()
    region_names_list.append(region_names)

    taste_lfps = [x for x in region_lfps.swapaxes(0,1)]

    flat_region_lfps = np.reshape(
        region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))
    flat_taste_nums = np.repeat(np.arange(len(taste_names)),
                                taste_lfps[0].shape[1])

    lfp_set_names = taste_names.copy()
    lfp_set_names.append('all')

    # 1) Remove trials with artifacts
    good_lfp_trials_bool = \
            dat.lfp_processing.return_good_lfp_trial_inds(flat_region_lfps)
    good_lfp_trials = flat_region_lfps[:,good_lfp_trials_bool]
    good_taste_nums = flat_taste_nums[good_lfp_trials_bool]

    # Run STFT on good trials
    inds = list(np.ndindex(good_lfp_trials.shape[:2]))

    stft_list = Parallel(n_jobs = mp.cpu_count()-2)\
            (delayed(dat.calc_stft)(good_lfp_trials[this_iter],
                                **dat.stft_params)\
            for this_iter in tqdm(inds))

    freq_vec = stft_list[0][0]
    time_vec = stft_list[0][1]
    fin_stft_list = [x[-1] for x in stft_list]
    del stft_list
    amplitude_list = dat.parallelize(np.abs, fin_stft_list)
    amplitude_array = dat.convert_to_array(amplitude_list, inds)**2

    mean_raw_amplitude_array = amplitude_array.mean(axis=1)
    all_mean_raw_amplitude_array.append(mean_raw_amplitude_array)

    ############################################################
    # Preprocessing
    ############################################################

    # 2) Preprocess data
    this_granger = gu.granger_handler(good_lfp_trials)
    this_granger.preprocess_data()
    preprocessed_data = this_granger.preprocessed_data

    stft_list = Parallel(n_jobs = mp.cpu_count()-2)\
            (delayed(dat.calc_stft)(preprocessed_data[this_iter],
                                **dat.stft_params)\
            for this_iter in tqdm(inds))

    freq_vec = stft_list[0][0]
    time_vec = stft_list[0][1]
    fin_stft_list = [x[-1] for x in stft_list]
    del stft_list
    amplitude_list = dat.parallelize(np.abs, fin_stft_list)
    amplitude_array = dat.convert_to_array(amplitude_list, inds)**2

    mean_processed_amplitude_array = amplitude_array.mean(axis=1)
    all_mean_processed_amplitude_array.append(mean_processed_amplitude_array)

all_mean_raw_amplitude_array = np.array(all_mean_raw_amplitude_array)
all_mean_processed_amplitude_array = np.array(all_mean_processed_amplitude_array)
all_region_names = np.array(region_names_list)

np.save(os.path.join(artifacts_dir, 'all_mean_raw_amplitude_array.npy'),
        all_mean_raw_amplitude_array)
np.save(os.path.join(artifacts_dir, 'all_mean_processed_amplitude_array.npy'),
        all_mean_processed_amplitude_array)
np.save(os.path.join(artifacts_dir, 'all_region_names.npy'),
        all_region_names)

############################################################
# Plot baseline and stim spectra separately
############################################################

all_mean_raw_amplitude_array = np.load(
    os.path.join(artifacts_dir, 'all_mean_raw_amplitude_array.npy'))
all_mean_processed_amplitude_array = np.load(
    os.path.join(artifacts_dir, 'all_mean_processed_amplitude_array.npy'))
all_region_names = np.load(
    os.path.join(artifacts_dir, 'all_region_names.npy'))

# Make sure region names are the same across all animals
assert np.all(all_region_names[:,0] == all_region_names[0,0])
assert np.all(all_region_names[:,1] == all_region_names[0,1])

region_names = all_region_names[0]

stim_t = 10
stim_end_t = 12
baseline_inds = np.where(time_vec < stim_t)[0]
stim_inds = np.where((time_vec > stim_t) & (time_vec < stim_end_t))[0]

# Plot lineplots for average baseline and stim spectra for each region
plot_dat = [
    all_mean_raw_amplitude_array[...,baseline_inds].mean(axis=(-1)),
    all_mean_raw_amplitude_array[...,stim_inds].mean(axis=(-1)),
    all_mean_processed_amplitude_array[...,baseline_inds].mean(axis=(-1)),
    all_mean_processed_amplitude_array[...,stim_inds].mean(axis=(-1)),
    ]

mean_plot_dat = np.array(plot_dat).mean(axis=1)
sd_plot_dat = np.array(plot_dat).std(axis=1)

x_stack = [ 
    time_vec[baseline_inds],
    time_vec[stim_inds],
    time_vec[baseline_inds],
    time_vec[stim_inds]
    ]

titles = ['Raw Baseline', 'Raw Stimulus',
          'Processed Baseline', 'Processed Stimulus']

# Plot log and raw values for each region, for both baseline and stim
# Plot both regions together on each subplot
fig, ax = plt.subplots(4,2, sharex = True, #sharey = 'col',
                       figsize = (5,10))
for i, this_ax in enumerate(ax[:,0].flatten()):
    this_ax.plot(freq_vec, mean_plot_dat[i].T)
    # this_ax.fill_between(freq_vec,
    #                      y1 = mean_plot_dat[i].T - sd_plot_dat[i].T,
    #                      y2 = mean_plot_dat[i].T + sd_plot_dat[i].T,
    #                      alpha = 0.5)
    this_ax.set_title(titles[i])
    this_ax.set_yscale('log')
    this_ax.set_ylabel('Amplitude (log)')
    this_ax.set_xlabel('Time (s)')
for i, this_ax in enumerate(ax[:,1].flatten()):
    this_ax.plot(freq_vec, mean_plot_dat[i].T)
    this_ax.set_title(titles[i])
    this_ax.set_ylabel('Amplitude')
    this_ax.set_xlabel('Time (s)')
# Plot line-noise for all subplots
for this_ax in ax.flatten():
    this_ax.axvline(60, color = 'k', linestyle = '--')
# Create legend
ax[0,0].legend(region_names, loc = 'upper right')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'avg_baseline_stim_spectra.svg'))
plt.close(fig)

# Subtract trend from baseline and stim spectra
# Detect trend using spectrum for f>65Hz
trend_inds = np.where(freq_vec > 65)[0]
log_plot_dat = np.log(plot_dat)
trend_dat = log_plot_dat[...,trend_inds]

# Linear interpolation across full frequency range using trend 
# spectrum for f>65Hz
interp_freq_vec = freq_vec.copy() 
interp_trend_dat = np.zeros((
    trend_dat.shape[0], trend_dat.shape[1], interp_freq_vec.size))
iter_inds = list(np.ndindex(trend_dat.shape[:2]))
for i, j in tqdm(iter_inds):
    reg = linregress(freq_vec[trend_inds], trend_dat[i,j])
    interp_trend_dat[i,j,:] = reg.intercept + reg.slope*interp_freq_vec

# plt.plot(freq_vec[trend_inds], trend_dat[:,0].T)
# plt.plot(interp_freq_vec, interp_trend_dat[:,0].T)
# plt.show()

# Subtract interpolated trend from log spectra
detrended_log_plot_dat = log_plot_dat - interp_trend_dat

# Plot detrended log spectra
fig, ax = plt.subplots(2,2, sharex = True,
                       figsize = (10,10))
for i, this_ax in enumerate(ax.flatten()):
    im = this_ax.plot(freq_vec, detrended_log_plot_dat[i].T, '-x')
    this_ax.set_title(titles[i])
    this_ax.set_ylabel('Amplitude (log)')
    this_ax.set_xlabel('Time (s)')
# Plot line-noise for all subplots
for this_ax in ax.flatten():
    this_ax.axvline(60, color = 'k', linestyle = '--')
min_y = detrended_log_plot_dat.min() 
# Create transparent grey box below 0 amplitide
for this_ax in ax.flatten():
    this_ax.axhspan(min_y, 0, color = 'grey', alpha = 0.2)
# Mask out values between 56 and 64 Hz
for this_ax in ax.flatten():
    this_ax.axvspan(56, 64, color = 'white', zorder = 10)
# Create legend
ax[0,0].legend(region_names, loc = 'upper right')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'detrended_log_baseline_stim_spectra.svg'))
plt.close(fig)

# Plot data for individual sessions
for i, dir_name in enumerate(tqdm(dir_list)):
    # dir_name = dir_list[0]

    basename = dir_name.split('/')[-1]
    mean_raw_amplitude_array = all_mean_raw_amplitude_array[i]
    mean_processed_amplitude_array = all_mean_processed_amplitude_array[i]

    fin_plot_dat = [
                 mean_raw_amplitude_array[...,baseline_inds],
                 mean_raw_amplitude_array[...,stim_inds],
                 mean_processed_amplitude_array[...,baseline_inds],
                 mean_processed_amplitude_array[...,stim_inds]
                 ]

    fig, ax = plt.subplots(2,2)
    for i, this_ax in enumerate(ax.flatten()): 
        this_dat = fin_plot_dat[i]
        this_x = x_stack[i]
        this_title = titles[i]
        this_ax.pcolormesh(
                this_x, freq_vec, this_dat, shading='gouraud', cmap='viridis')
        this_ax.set_title(this_title)
        this_ax.set_xlabel('Time (s)')
        this_ax.set_ylabel('Frequency (Hz)')
    fig.suptitle(basename)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, basename + '_baseline_stim_spectra.png'))
    plt.close(fig)

    # Generate same plot in log scale
    fig, ax = plt.subplots(2,2)
    for i, this_ax in enumerate(ax.flatten()):
        this_dat = fin_plot_dat[i]
        this_x = x_stack[i]
        this_title = titles[i]
        this_ax.pcolormesh(
                this_x, freq_vec, this_dat, shading='gouraud', cmap='viridis',
                norm=LogNorm())
        this_ax.set_title(this_title)
        this_ax.set_xlabel('Time (s)')
        this_ax.set_ylabel('Frequency (Hz)')
    fig.suptitle(basename)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, basename + '_baseline_stim_spectra_log.png'))
    plt.close(fig)

    # Also plot lineplots of spectra by frequency
    mean_plot_dat = [x.mean(axis=-1) for x in fin_plot_dat]

