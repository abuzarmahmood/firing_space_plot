"""
On a single animal basis, perform changepoint modelling on the
spectral granger causality to check whether we see multiple
changes around the palatability transition
"""
import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from scipy import stats
import pymc3 as pm
import arviz as az
from sklearn.cluster import KMeans

import pandas as pd
import seaborn as sns
from collections import Counter

import sys
sys.path.append('/media/bigdata/projects/pytau')
import pytau.changepoint_model as models

plot_dir_base = '/media/bigdata/firing_space_plot/lfp_analyses/' +\
    'granger_causality/plots/aggregate_plots'

model_save_dir = '/media/bigdata/firing_space_plot/lfp_analyses/' +\
    'granger_causality/models/saved_models'
if not os.path.isdir(model_save_dir):
    os.makedirs(model_save_dir)

############################################################
# Load Data
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

basename_list = [os.path.basename(this_dir) for this_dir in dir_list]
animal_name = [this_name.split('_')[0] for this_name in basename_list]
animal_count = np.unique(animal_name, return_counts=True)
session_count = len(basename_list)

n_string = f'N = {session_count} sessions, {len(animal_count[0])} animals'

# Write out basenames to plot_dir
name_frame = pd.DataFrame(
        dict(
            animal_name = animal_name,
            session_name = basename_list
            )
        )
name_frame = name_frame.sort_values(by = list(name_frame.columns))
name_frame.reset_index(drop=True, inplace=True)
name_frame.to_csv(os.path.join(plot_dir_base, 'session_names.txt'),
                  sep = '\t', index=False)


save_path = '/ancillary_analysis/granger_causality/all'
names = ['granger_actual',
         'masked_granger',
         'mask_array',
         'wanted_window',
         'time_vec',
         'freq_vec']

loaded_dat_list = []
for this_dir in dir_list:
    h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
    with tables.open_file(h5_path, 'r') as h5:
        loaded_dat = [h5.get_node(save_path, this_name)[:]
                      for this_name in names]
        loaded_dat_list.append(loaded_dat)

dir_names = ['BLA-->GC', 'GC-->BLA']
zipped_dat = zip(*loaded_dat_list)
zipped_dat = [np.stack(this_dat) for this_dat in zipped_dat]

(
    granger_actual,
    masked_granger,
    mask_array,
    wanted_window,
    time_vec,
    freq_vec) = zipped_dat

wanted_window = np.array(wanted_window[0])/1000
stim_t = 2
corrected_window = wanted_window-stim_t
freq_vec = freq_vec[0]
time_vec = time_vec[0]
time_vec += corrected_window[0]

wanted_freq_range = [1, 100]
wanted_freq_inds = np.where(np.logical_and(freq_vec >= wanted_freq_range[0],
                                           freq_vec <= wanted_freq_range[1]))[0]
freq_vec = freq_vec[wanted_freq_inds]
granger_actual = granger_actual.mean(axis=1)
granger_actual = granger_actual[:, :, wanted_freq_inds]
masked_granger = masked_granger[:, :, wanted_freq_inds]
mask_array = mask_array[:, :, wanted_freq_inds]

# Extract directions from granger actual
granger_actual = np.stack([granger_actual[..., 0, 1],
                            granger_actual[..., 1, 0]], axis=1)

# And from mask_arrya
mask_array = np.stack([mask_array[..., 0, 1],
                       mask_array[..., 1, 0]], axis=1)

zscore_granger_actual = stats.zscore(granger_actual, axis=2)

# Plot granger causality
dat_names = ['raw','zscored']
dat_list = [granger_actual, zscore_granger_actual]
for this_dat, this_name in zip(dat_list, dat_names):
    inds = list(np.ndindex(this_dat.shape[:2]))
    fig, ax = plt.subplots(*this_dat.shape[:2], figsize=(7, 20),
                           sharex=True, sharey=True)
    for this_ind in inds:
        ax[this_ind].imshow(this_dat[this_ind].T,
                            origin='lower',
                            aspect='auto',
                            extent=[time_vec[0], time_vec[-1],
                                    freq_vec[0], freq_vec[-1]],
                            cmap='viridis')
        ax[this_ind].axvline(0, color='red')
        if this_ind[0] == ax.shape[0]:
            ax[this_ind].set_xlabel('Time (s)')
        if this_ind[0] == 0:
            ax[this_ind].set_title(f'{dir_names[this_ind[1]]}')
        if this_ind[1] == 0:
            ax[this_ind].set_ylabel('Freq (Hz)')
    plt.suptitle('Granger Causality')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plot_dir_base, f'all_granger_causality_{this_name}.png'))
    plt.close()


################################################################################
# Average by band and zscore 
################################################################################

band_names = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
band_ranges = [[1, 4], [4, 8], [8, 12], [12, 30], [30, 80], [80, 100]]
band_inds = []
for this_range in band_ranges:
    band_inds.append(np.where(np.logical_and(freq_vec >= this_range[0],
                                             freq_vec <= this_range[1]))[0])
granger_bands = []
for this_band in band_inds:
    granger_bands.append(np.nanmean(mask_array[..., this_band], axis=-1))
granger_bands = np.stack(granger_bands, axis=1)
granger_bands = np.moveaxis(granger_bands, 2, 0)

zscored_granger_bands = stats.zscore(granger_bands, axis=-1) 

# Plot zscore granger bands
# Plot granger causality
this_dat = np.swapaxes(zscored_granger_bands,0,1)
this_dat = np.swapaxes(this_dat,2,3)
inds = list(np.ndindex(this_dat.shape[:2]))
fig, ax = plt.subplots(*this_dat.shape[:2], figsize=(7, 20),
                       sharex=True, sharey=True)
for this_ind in inds:
    ax[this_ind].imshow(this_dat[this_ind].T,
                        origin='lower',
                        aspect='auto',
                        extent=[time_vec[0], time_vec[-1],
                                freq_vec[0], freq_vec[-1]],
                        cmap='viridis',
                        interpolation='none')
    ax[this_ind].axvline(0, color='red')
    if this_ind[0] == ax.shape[0]:
        ax[this_ind].set_xlabel('Time (s)')
    if this_ind[0] == 0:
        ax[this_ind].set_title(f'{dir_names[this_ind[1]]}')
    if this_ind[1] == 0:
        ax[this_ind].set_ylabel('Freq (Hz)')
plt.suptitle('Granger Causality')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(plot_dir_base, f'all_zscored_band_masks.png'))
plt.close()




############################################################
# Using Dirichlet Process Prior 
############################################################
max_states = 10
n_chains = 30

inds = list(np.ndindex(zscored_granger_bands.shape[:-2]))

model_list = []
trace_list = []
for this_ind in tqdm(inds):
    ind_str = "_".join([str(x) for x in this_ind])
    save_path = os.path.join(model_save_dir,
                            f'{ind_str}_dpp_trace_band_mask.nc')
    if os.path.isfile(save_path):
        # Load model netcdf using arviz
        dpp_trace = az.from_netcdf(save_path)
        dpp_model = []
    else:
        data_array = zscored_granger_bands[this_ind]
        dpp_model = models.gaussian_changepoint_mean_dirichlet(
                data_array, max_states = max_states)
        with dpp_model:
            dpp_trace = pm.sample(
                                tune = 500,
                                draws = 500, 
                                  target_accept = 0.95,
                                 chains = n_chains,
                                 cores = 30,
                                return_inferencedata=True)
            dpp_trace.to_netcdf(save_path)
    trace_list.append(dpp_trace)
    model_list.append(dpp_model)

tau_samples = np.stack([x.posterior.tau for x in trace_list], axis=0)

tau_array = np.zeros((*zscored_granger_bands.shape[:2], *tau_samples.shape[1:]))

for i, this_ind in enumerate(inds):
    tau_array[this_ind] = tau_samples[i]

# Plot tau for each direction and session
fig, ax = plt.subplots(*tau_array.shape[:2][::-1], sharex=True, sharey=True,)
for this_ind in inds:
    ax[this_ind[::-1]].hist(tau_array[this_ind].flatten(), bins=time_vec)
    ax[this_ind[::-1]].set_title(f'{this_ind}')
plt.tight_layout()
plt.show()
