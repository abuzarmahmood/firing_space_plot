"""
Perform changepoint detection on significance mask of granger causality
for each direction of interaction
"""
import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import trange
from scipy import stats
import pymc3 as pm

import sys
sys.path.append('/media/bigdata/projects/pytau')
import pytau.changepoint_model as models

plot_dir_base = '/media/bigdata/firing_space_plot/lfp_analyses/' +\
    'granger_causality/plots'

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
mean_mask = np.nanmean(mask_array, axis=0)

mean_mask = np.stack([mean_mask[...,0,1],mean_mask[...,1,0],], axis=-1).T

# Plot mean mask
fig, ax = plt.subplots(2,1)
for this_mask, this_ax in zip(mean_mask, ax):
    this_ax.imshow(this_mask, aspect='auto', origin='lower',
                   extent=[time_vec[0], time_vec[-1],
                           freq_vec[0], freq_vec[-1]],
                   cmap='viridis')
    this_ax.set_ylabel('Frequency (Hz)')
    this_ax.set_xlabel('Time (s)')
    this_ax.set_title(n_string)
plt.tight_layout()
plt.show()

############################################################
# Changepoint Detection
############################################################
# Create and fit model
n_fit = 80000
n_samples = 20000
state_range = np.arange(2, 8+1

best_model_list = []
model_list_list = []
elbo_values_list = []
for data_array in mean_mask:
    best_model, model_list, elbo_values = \
        models.find_best_states(
                data_array, 
                models.gaussian_changepoint_mean_2d,
                n_fit,
                n_samples,
                min_states = state_range.min(),
                max_states = state_range.max()
    )
    best_model_list.append(best_model)
    model_list_list.append(model_list)
    elbo_values_list.append(elbo_values)

# Plot ELBO values and mark best model
fig, ax = plt.subplots(2,1)
for this_elbo, this_ax in zip(elbo_values_list, ax):
    this_ax.plot(state_range, this_elbo)
    this_ax.axvline(state_range[np.argmin(this_elbo)], color='r',
                    linestyle='--', label='Best Model')
    this_ax.legend()
    this_ax.set_ylabel('ELBO')
    this_ax.set_xlabel('Number of States')
    this_ax.set_title(n_string)
plt.tight_layout()
plt.show()

best_state_nums = [5, 5]

model_list = []
trace_list = []
tau_list = []
for data_array, this_state_num in zip(mean_mask, best_state_nums):
    model = models.gaussian_changepoint_mean_2d(data_array, this_state_num)
    model, approx, mu_stack, sigma_stack, tau_samples, fit_data = \
            models.advi_fit(model = model, fit = n_fit, samples = n_samples)
    trace = approx.sample(2000)
    trace_list.append(trace)
    model_list.append(model)
    tau_list.append(tau_samples)

tau_array = np.stack(tau_list, axis=0)
int_tau = np.round(tau_array).astype(int)
mode_tau = np.squeeze(stats.mode(int_tau, axis=1))[0]

tau_reshape = np.moveaxis(tau_array, -1, 1)

fig, ax = plt.subplots(len(tau_reshape), 1, sharex=True)
for i, this_ax in enumerate(ax):
    this_dat = tau_reshape[i]
    for vals in this_dat:
        this_ax.hist(vals, bins=np.linspace(0,mean_mask.shape[-1]))
plt.show()

# Plot inferred changepoints from best models with mean_mask_bands
fig, ax = plt.subplots(2,1)
for i, this_ax in enumerate(ax):
    this_dat = mean_mask[i]
    this_tau = mode_tau[i]
    im = this_ax.imshow(stats.zscore(this_dat,axis=-1), aspect='auto', origin='lower',
                   extent=[time_vec[0], time_vec[-1],
                           freq_vec[0], freq_vec[-1]],
                   )
    plt.colorbar(im, ax=this_ax)
    this_ax.set_ylabel('Frequency (Hz)')
    this_ax.set_xlabel('Time (s)')
    this_ax.set_title(n_string)
    for tau_val in this_tau:
        this_ax.axvline(time_vec[tau_val], color='r', linestyle='--')
plt.tight_layout()
plt.show()

############################################################
# Repeat for average mask averaged over freq bands
band_names = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']
band_ranges = [[1, 4], [4, 8], [8, 12], [12, 30], [30, 80], [80, 100]]
band_inds = []
for this_range in band_ranges:
    band_inds.append(np.where(np.logical_and(freq_vec >= this_range[0],
                                             freq_vec <= this_range[1]))[0])
mean_mask_bands = []
for this_band in band_inds:
    mean_mask_bands.append(np.nanmean(mean_mask[:, this_band], axis=1))
mean_mask_bands = np.stack(mean_mask_bands, axis=1)

# Plot mean mask bands
fig, ax = plt.subplots(2,1)
for this_mask, this_ax in zip(mean_mask_bands, ax):
    this_ax.imshow(this_mask, aspect='auto', origin='lower',
                   extent=[time_vec[0], time_vec[-1],
                           band_ranges[0][0], band_ranges[-1][1]],
                   cmap='viridis')
    this_ax.set_ylabel('Frequency (Hz)')
    this_ax.set_xlabel('Time (s)')
    this_ax.set_title(n_string)
plt.tight_layout()
plt.show()

# Create and fit model
n_fit = 80000
n_samples = 20000
state_range = np.arange(2, 8+1)

best_model_list = []
model_list_list = []
elbo_values_list = []
for data_array in mean_mask_bands:
    best_model, model_list, elbo_values = \
        models.find_best_states(
                data_array, 
                models.gaussian_changepoint_mean_2d,
                n_fit,
                n_samples,
                min_states = state_range.min(),
                max_states = state_range.max()
    )
    best_model_list.append(best_model)
    model_list_list.append(model_list)
    elbo_values_list.append(elbo_values)

# Plot ELBO values and mark best model
fig, ax = plt.subplots(2,1)
for this_elbo, this_ax in zip(elbo_values_list, ax):
    this_ax.plot(state_range, this_elbo)
    this_ax.axvline(state_range[np.argmin(this_elbo)], color='r',
                    linestyle='--', label='Best Model')
    this_ax.legend()
    this_ax.set_ylabel('ELBO')
    this_ax.set_xlabel('Number of States')
    this_ax.set_title(n_string)
plt.tight_layout()
plt.show()

# Plot inferred changepoints from best models
#best_state_nums = [state_range[np.argmin(this_elbo)] for this_elbo in elbo_values_list]
best_state_nums = [4, 4]

model_list = []
trace_list = []
tau_list = []
for data_array, this_state_num in zip(mean_mask_bands, best_state_nums):
    model = models.gaussian_changepoint_mean_2d(data_array, this_state_num)
    model, approx, mu_stack, sigma_stack, tau_samples, fit_data = \
            models.advi_fit(model = model, fit = n_fit, samples = n_samples)
    trace = approx.sample(2000)
    trace_list.append(trace)
    model_list.append(model)
    tau_list.append(tau_samples)

############################################################
# Given each model, generate posterior samples of observations
ppc_list = []
for this_trace, this_model in zip(trace_list, model_list):
    with this_model:
        ppc = pm.sample_posterior_predictive(this_trace, samples=1000)
    ppc_list.append(ppc['obs'])

ppc_stack = np.stack(ppc_list, axis=0)
mean_ppc = np.mean(ppc_stack, axis=1)

# Plot ppc
fig, ax = plt.subplots(2,1)
for this_ppc, this_ax in zip(mean_ppc, ax):
    this_ax.imshow(this_ppc, aspect='auto', origin='lower',
                   extent=[time_vec[0], time_vec[-1],
                           band_ranges[0][0], band_ranges[-1][1]],
                   cmap='viridis')
    this_ax.set_ylabel('Frequency (Hz)')
    this_ax.set_xlabel('Time (s)')
    this_ax.set_title(n_string)
plt.tight_layout()
plt.show()
############################################################

tau_array = np.stack(tau_list, axis=0)
int_tau = np.round(tau_array).astype(int)
mode_tau = np.squeeze(stats.mode(int_tau, axis=1))[0]

tau_reshape = np.moveaxis(tau_array, -1, 1)

fig, ax = plt.subplots(len(tau_reshape), 1, sharex=True)
for i, this_ax in enumerate(ax):
    this_dat = tau_reshape[i]
    for vals in this_dat:
        this_ax.hist(vals, bins=np.linspace(0,mean_mask.shape[-1]))
plt.show()

# Plot inferred changepoints from best models with mean_mask_bands
dir_names = ['BLA-->GC', 'GC-->BLA']
col_names = ['Actual Data','Posterior Predictive']
fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
for i, this_ax in enumerate(ax):
    this_dat = mean_mask_bands[i]
    this_ppc = mean_ppc[i]
    this_tau = mode_tau[i]
    im = this_ax[0].imshow(stats.zscore(this_dat,axis=-1), aspect='auto', origin='lower',
                   extent=[time_vec[0], time_vec[-1],
                           freq_vec[0], freq_vec[-1]],
                   )
    this_ax[1].imshow(stats.zscore(this_ppc,axis=-1), aspect='auto', origin='lower',
                     extent=[time_vec[0], time_vec[-1],
                             freq_vec[0], freq_vec[-1]],
                      )
    #plt.colorbar(im, ax=this_ax)
    this_ax[0].set_ylabel(f'Frequency (Hz)\n{dir_names[i]}')
    this_ax[0].set_xlabel('Time (s)')
    for tau_val in this_tau:
        this_ax[0].axvline(time_vec[tau_val], color='r', linestyle='--')
        this_ax[1].axvline(time_vec[tau_val], color='r', linestyle='--')
    yticks = np.arange(len(band_ranges)) + 0.5
    yticks = yticks * ((freq_vec.max() - freq_vec.min()) / len(band_ranges))
    for sing_ax, this_name in zip(this_ax, col_names):
        sing_ax.set_yticks(yticks)
        sing_ax.set_yticklabels([str(x) for x in band_ranges])
        sing_ax.set_title(this_name)
fig.suptitle(n_string)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'granger_mask_changepoints.png'),
            dpi = 300)
plt.close(fig)
#plt.show()


fig, ax = plt.subplots(2,1)
for this_tau, this_ax in zip(tau_array, ax):
    this_ax.hist(this_tau.flatten(), bins=np.linspace(0,mean_mask.shape[-1], 100))
plt.show()

# Plot scatter plots of changepoints against on another
import itertools as it
inds = list(it.combinations(np.arange(tau_array.shape[-1]),2))

fig, ax = plt.subplots(len(tau_array), len(inds))
for i, this_tau in enumerate(tau_array):
    for j, this_ax in enumerate(ax[i]):
        this_ax.scatter(this_tau[:,inds[j][0]], this_tau[:,inds[j][1]])
plt.show()

# Plot mean mu vals per state to see unique states
mu_array = np.stack([this_trace['mu'] for this_trace in trace_list])
mean_mu = np.mean(mu_array, axis=1)

fig, ax = plt.subplots(2,1)
for this_mu, this_ax in zip(mean_mu, ax):
    this_ax.matshow(this_mu, aspect='auto', origin='lower',
                    )
plt.show()
