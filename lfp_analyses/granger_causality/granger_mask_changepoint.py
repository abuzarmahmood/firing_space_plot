"""
Perform changepoint detection on significance mask of granger causality
for each direction of interaction
"""

import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from scipy import stats
import pymc as pm
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
# import pickle
from cloudpickle import dump, load
from multiprocessing import cpu_count

import pandas as pd
import seaborn as sns
from collections import Counter

import sys
sys.path.append('/media/bigdata/projects/pytau')
import pytau.changepoint_model as models

plot_dir_base = '/media/bigdata/firing_space_plot/lfp_analyses/' +\
    'granger_causality/plots/aggregate_plots'

artifact_dir = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/artifacts' 

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
print(n_string)

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

# Add index to each animal name
name_frame['session_inds'] = name_frame.groupby('animal_name').cumcount()
name_frame['plot_name'] = name_frame['animal_name'] + '_' + name_frame['session_inds'].astype(str)

name_frame['animal_code'] = name_frame['animal_name'].astype('category').cat.codes


save_path = '/ancillary_analysis/granger_causality/all'
names = ['granger_actual',
         'masked_granger',
         'mask_array',
         'wanted_window',
         'time_vec',
         'freq_vec']

loaded_dat_list = []
loaded_name_list = []
for this_dir in tqdm(dir_list):
    try:
        h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
        with tables.open_file(h5_path, 'r') as h5:
            loaded_dat = [h5.get_node(save_path, this_name)[:]
                          for this_name in names]
            loaded_dat_list.append(loaded_dat)
            loaded_name_list.append(os.path.basename(this_dir))
    except:
        print(f'Error loading {this_dir}')
        continue

name_frame.set_index('session_name', inplace=True) 
name_frame = name_frame.loc[loaded_name_list]
name_frame.reset_index(inplace=True)

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

dir_names = ['BLA-->GC', 'GC-->BLA']
mean_mask = np.stack([mean_mask[...,0,1],mean_mask[...,1,0],], axis=-1).T

# Plot all masks
plot_mask_array = np.stack(
        [mask_array[..., 0, 1], mask_array[..., 1, 0]], axis=-1
        )
plot_mask_array = np.moveaxis(plot_mask_array, -1, 1)

fig, ax = plt.subplots(*plot_mask_array.shape[:2],
                       sharex=True, sharey=True,
                       figsize = (5,15))
inds = np.array(list(np.ndindex(*plot_mask_array.shape[:2])))
for this_ind, this_ax in zip(inds, ax.flatten()):
    this_mask = plot_mask_array[this_ind[0], this_ind[1]]
    this_ax.imshow(1-this_mask.T, aspect='auto', origin='lower',
                   extent=[time_vec[0], time_vec[-1],
                           freq_vec[0], freq_vec[-1]],
                   cmap='viridis')
    this_ax.set_ylabel('Freq (Hz)')
for i, this_ax in enumerate(ax[0]):
    this_ax.set_title(dir_names[i])
for i, this_ax in enumerate(ax[-1]):
    this_ax.set_xlabel('Time (s)')
fig.suptitle('Granger Masks')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'all_granger_masks.png'))
plt.close()
#plt.show()

##############################
# Focus on 0-20Hz ranger for GC-->BLA
##############################
freq_inds = np.where(np.logical_and(freq_vec >= 0, freq_vec <= 20))[0]

wanted_plot_mask_array = plot_mask_array[..., freq_inds]

fig, ax = plt.subplots(
        len(wanted_plot_mask_array) + 2, 2,
        sharex = True, sharey = True,
        figsize = (5,15)
        )
for dir_ind in range(2):
    for i in range(len(wanted_plot_mask_array)): 
        this_plot_dat = 1 - wanted_plot_mask_array[i, dir_ind].T
        ax[i, dir_ind].pcolorfast(time_vec, freq_vec[freq_inds],
                            this_plot_dat,
                           cmap = 'viridis')
        mean_plot_dat = np.nanmean(this_plot_dat, axis=0)
        scaled_mean = (mean_plot_dat / mean_plot_dat.max()) * 20
        ax[i, dir_ind].plot(
                time_vec, scaled_mean, color = 'red',
                linestyle = '--', linewidth = 2)
        ax[i, 0].set_ylabel(name_frame.loc[i, 'plot_name'])
    ax[-2, dir_ind].pcolorfast(time_vec, freq_vec[freq_inds],
                               1 - wanted_plot_mask_array[:, dir_ind].mean(axis=0).T,
                      cmap = 'jet')
    ax[-1, dir_ind].pcolorfast(time_vec, freq_vec[freq_inds],
                               stats.zscore(1 - wanted_plot_mask_array[:, dir_ind].mean(axis=0).T, axis=-1),
                      cmap = 'jet')
    ax[-1, dir_ind].set_xlabel('Time (s)')
    ax[0, dir_ind].set_title(dir_names[dir_ind])
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'gc_bla_granger_masks.png'))
plt.close()

# Plot single session masks for 0-20Hz
zscored_smoothed_mean_mask_list = []
dir_ind_list = []
for i in range(2):
    this_plot_dat = 1 - wanted_plot_mask_array[:,i].T
    mean_plot_dat = np.nanmean(this_plot_dat, axis=0).T
    scaled_mean = (mean_plot_dat / mean_plot_dat.max(axis=-1)[:, None]) * 20
    smoothed_mean = savgol_filter(mean_plot_dat, 11, 3, axis=-1)
    zscored_smooth_mean_mask = stats.zscore(smoothed_mean, axis=-1)
    zscored_smoothed_mean_mask_list.append(zscored_smooth_mean_mask)
    dir_ind_list.append(i)

##############################
# Calculate changepoints for zscored smoothed mean mask

wanted_model = models.gaussian_changepoint_mean_2d

zscored_smooth_mean_mask_list_flat = np.concatenate(zscored_smoothed_mean_mask_list, axis=0)
dir_ind_list_flat = np.concatenate(
        [np.repeat(x, y.shape[0]) for x, y in zip(dir_ind_list, zscored_smoothed_mean_mask_list)],
        axis=0
        )

change_frame_path = os.path.join(artifact_dir, 'granger_individual_mask_changepoints.pkl')
if not os.path.exists(change_frame_path):
    tau_list = []
    mean_ppc_list = []
    for this_dat in tqdm(zscored_smooth_mean_mask_list_flat):
        model = wanted_model(this_dat[None,:], 3)
        model, approx, mu_stack, sigma_stack, tau_samples, fit_data = \
                models.advi_fit(model = model, fit = 80000, samples = 20000)
        trace = approx.sample(2000)
        tau_samples = trace.tau.astype(int) 
        ppc = pm.sample_posterior_predictive(
                trace, model = model, var_names = ['obs'])['obs']
        mean_ppc = np.mean(ppc, axis=0)
        tau_list.append(tau_samples)
        mean_ppc_list.append(mean_ppc)

    basenames_list = name_frame['session_name'].tolist()
    basenames_list = np.stack(basenames_list*2) 

    change_frame = pd.DataFrame(
            dict(
                basenames = basenames_list,
                dir_inds = dir_ind_list_flat,
                dir_name = [dir_names[x] for x in dir_ind_list_flat],
                tau_samples = tau_list,
                mean_ppc = mean_ppc_list
                )
            )
    change_frame.to_pickle(change_frame_path) 
else:
    change_frame = pd.read_pickle(change_frame_path)

fig, ax = plt.subplots(2,5, sharex=True, sharey=True,
                       figsize = (12,5))
for i in range(2):
    this_plot_dat = 1 - wanted_plot_mask_array[:,i].T
    mean_plot_dat = np.nanmean(this_plot_dat, axis=0).T
    scaled_mean = (mean_plot_dat / mean_plot_dat.max(axis=-1)[:, None]) * 20
    this_tau = change_frame[change_frame['dir_inds'] == i]['tau_samples'].values
    this_tau = np.stack(this_tau)
    this_mode_tau = np.squeeze(stats.mode(this_tau, axis=1)[0])
    this_change_time = time_vec[this_mode_tau]
    ax[i, 0].pcolorfast(time_vec, np.arange(len(scaled_mean)),
                     scaled_mean,
                     cmap = 'jet')
    # Smooth using Savitzky-Golay filter
    smoothed_mean = savgol_filter(mean_plot_dat, 11, 3, axis=-1)
    ax[i, 1].pcolorfast(time_vec, np.arange(len(smoothed_mean)),
                        smoothed_mean,
                        cmap = 'jet')
    ax[i,0].set_ylabel(dir_names[i])
    ax[i,2].pcolorfast(time_vec, np.arange(len(scaled_mean)), 
                       stats.zscore(scaled_mean, axis=-1),
                       cmap = 'jet')
    zscored_smooth_mean_mask = stats.zscore(smoothed_mean, axis=-1)
    ax[i,3].pcolorfast(time_vec, np.arange(len(smoothed_mean)), 
                       zscored_smooth_mean_mask,
                       cmap = 'jet')
    for session_ind, this_times in enumerate(this_change_time):
        ax[i,3].scatter(
                this_times, 
                [session_ind]*len(this_times),
                color = 'red', alpha = 0.7)
    ax[i,-1].pcolorfast(time_vec, np.arange(len(scaled_mean)),
                       name_frame['animal_code'].values[:, None],
                       cmap = 'Set1')
ax[0,0].set_title('Summed Mask')
ax[0,1].set_title('Summed Smoothed Mask')
ax[0,2].set_title('Zscored Summed Mask')
ax[0,3].set_title('Zscored Smoothed Mask')
fig.suptitle('Summed Masks 0-20Hz')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'gc_bla_granger_masks_single.png'))
plt.close()

# Plot histograms of changepoints
fig, ax = plt.subplots(2,1, sharex=True)
for i in range(2):
    this_tau = change_frame[change_frame['dir_inds'] == i]['tau_samples'].values
    this_tau = np.stack(this_tau)
    this_mode_tau = np.squeeze(stats.mode(this_tau, axis=1)[0])
    this_plot_tau = this_mode_tau.flatten()
    ax[i].hist(time_vec[this_plot_tau], bins = 20)
    ax[i].set_title(dir_names[i])
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'gc_bla_granger_changepoints.png'))
plt.close()



##############################

## Plot mean mask
#fig, ax = plt.subplots(2,1)
#for this_mask, this_ax in zip(mean_mask, ax):
#    this_ax.imshow(this_mask, aspect='auto', origin='lower',
#                   extent=[time_vec[0], time_vec[-1],
#                           freq_vec[0], freq_vec[-1]],
#                   cmap='viridis')
#    this_ax.set_ylabel('Frequency (Hz)')
#    this_ax.set_xlabel('Time (s)')
#    this_ax.set_title(n_string)
#plt.tight_layout()
#plt.show()

############################################################
# Changepoint Detection
############################################################

#############################################################
## Using elbo
#############################################################
#
## Create and fit model
#n_fit = 80000
#n_samples = 20000
#state_range = np.arange(2, 8+1)
#
#best_model_list = []
#model_list_list = []
#elbo_values_list = []
#for data_array in mean_mask:
#    best_model, model_list, elbo_values = \
#        models.find_best_states(
#                data_array, 
#                models.gaussian_changepoint_mean_2d,
#                n_fit,
#                n_samples,
#                min_states = state_range.min(),
#                max_states = state_range.max()
#    )
#    best_model_list.append(best_model)
#    model_list_list.append(model_list)
#    elbo_values_list.append(elbo_values)
#
## Plot ELBO values and mark best model
#fig, ax = plt.subplots(2,1)
#for this_elbo, this_ax in zip(elbo_values_list, ax):
#    this_ax.plot(state_range, this_elbo)
#    this_ax.axvline(state_range[np.argmin(this_elbo)], color='r',
#                    linestyle='--', label='Best Model')
#    this_ax.legend()
#    this_ax.set_ylabel('ELBO')
#    this_ax.set_xlabel('Number of States')
#    this_ax.set_title(n_string)
#plt.tight_layout()
#plt.show()
#
#best_state_nums = [5, 5]
#
#model_list = []
#trace_list = []
#tau_list = []
#for data_array, this_state_num in zip(mean_mask, best_state_nums):
#    model = models.gaussian_changepoint_mean_2d(data_array, this_state_num)
#    model, approx, mu_stack, sigma_stack, tau_samples, fit_data = \
#            models.advi_fit(model = model, fit = n_fit, samples = n_samples)
#    trace = approx.sample(2000)
#    trace_list.append(trace)
#    model_list.append(model)
#    tau_list.append(tau_samples)
#
#tau_array = np.stack(tau_list, axis=0)
#int_tau = np.round(tau_array).astype(int)
#mode_tau = np.squeeze(stats.mode(int_tau, axis=1))[0]
#
#tau_reshape = np.moveaxis(tau_array, -1, 1)
#
#fig, ax = plt.subplots(len(tau_reshape), 1, sharex=True)
#for i, this_ax in enumerate(ax):
#    this_dat = tau_reshape[i]
#    for vals in this_dat:
#        this_ax.hist(vals, bins=np.linspace(0,mean_mask.shape[-1]))
#plt.show()
#
## Plot inferred changepoints from best models with mean_mask_bands
#fig, ax = plt.subplots(2,1)
#for i, this_ax in enumerate(ax):
#    this_dat = mean_mask[i]
#    this_tau = mode_tau[i]
#    im = this_ax.imshow(stats.zscore(this_dat,axis=-1), aspect='auto', origin='lower',
#                   extent=[time_vec[0], time_vec[-1],
#                           freq_vec[0], freq_vec[-1]],
#                   )
#    plt.colorbar(im, ax=this_ax)
#    this_ax.set_ylabel('Frequency (Hz)')
#    this_ax.set_xlabel('Time (s)')
#    this_ax.set_title(n_string)
#    for tau_val in this_tau:
#        this_ax.axvline(time_vec[tau_val], color='r', linestyle='--')
#plt.tight_layout()
#plt.show()

################################################################################
# Repeat for average mask averaged over freq bands
################################################################################

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

zscored_mean_mask_bands = np.stack([
    stats.zscore(x, axis=-1) for x in mean_mask_bands])

##############################
# Generate PCA of zscored spectra

# Plot explained_variance_ratio for zscored_mean_mask
zscored_mean_mask = np.stack(
        [
            stats.zscore(x, axis=-1) for x in mean_mask
            ]
        )

explained_variance = []
for this_dir in zscored_mean_mask:
    pca = PCA()
    pca.fit(this_dir.T)
    explained_variance.append(pca.explained_variance_ratio_)

pca_spectra_list = []
for this_dir in mean_mask:
    pca = PCA(n_components = 5)
    pca.fit(this_dir.T)
    print(len(pca.components_))
    pca_data = pca.transform(this_dir.T).T
    pca_data = stats.zscore(pca_data, axis=-1)
    pca_spectra_list.append(pca_data)

fig, ax = plt.subplots(2,3, figsize = (7.5,5))
for i in range(2):
    ax[i,0].imshow(zscored_mean_mask[i], aspect='auto', origin='lower',
                   interpolation='none',
                     extent=[time_vec[0], time_vec[-1],
                             freq_vec[0], freq_vec[-1]],
                        cmap='viridis')
    ax[i,1].plot(explained_variance[i], '-x')
    ax[i,2].imshow(pca_spectra_list[i], aspect='auto', origin='lower',
                   interpolation='none',
                   extent=[time_vec[0], time_vec[-1],
                           pca_spectra_list[i].shape[0], 0],
                   cmap='viridis')
    ax[i,0].set_ylabel('Frequency (Hz)')
    ax[i,0].set_xlabel('Time (s)')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'granger_mask_pca.png'))
plt.close()




## Plot mean mask bands
#fig, ax = plt.subplots(2,2)
#for i, this_ax in enumerate(ax):
#    this_ax[0].imshow(mean_mask_bands[i], aspect='auto', origin='lower',
#                   extent=[time_vec[0], time_vec[-1],
#                           band_ranges[0][0], band_ranges[-1][1]],
#                   cmap='viridis')
#    this_ax[1].imshow(zscored_mean_mask_bands[i], aspect='auto', origin='lower',
#                   extent=[time_vec[0], time_vec[-1],
#                           band_ranges[0][0], band_ranges[-1][1]],
#                   cmap='viridis')
#    this_ax[0].set_ylabel('Frequency (Hz)')
#    this_ax[0].set_xlabel('Time (s)')
#    this_ax[0].set_title(n_string)
#plt.tight_layout()
#plt.show()

############################################################
# Using Dirichlet Process Prior 
############################################################
max_states = 6
n_chains = 24
n_cores = np.min([n_chains, cpu_count()])


for max_states in np.arange(6,11):
    max_states = int(max_states)
    trace_path = os.path.join(artifact_dir, f'granger_mask_changepoints_dirichlet_{max_states}states.pkl')

    if not os.path.exists(trace_path):
        model_list = []
        trace_list = []
        for data_array in zscored_mean_mask_bands:
        # for data_array in pca_spectra_list:
            dpp_model = models.gaussian_changepoint_mean_dirichlet(
                    data_array, max_states = max_states)
            with dpp_model:
                # dpp_trace = pm.sample_smc(return_inferencedata=False)
                # Numpyro sampler seems to ignore return_inferencedata flag
                rng = np.random.default_rng(666)
                dpp_trace = pm.sample(
                                    tune = 1000,
                                    draws = 4000, 
                                    target_accept = 0.95,
                                    chains = int(n_chains),
                                    cores = int(n_cores),
                                    nuts_sampler = 'numpyro',
                                    return_inferencedata=True,
                                    random_seed = rng)
            trace_list.append(dpp_trace)
            model_list.append(dpp_model)

        # Save traces and models
        with open(trace_path, 'wb') as f:
            # pickle.dump((model_list, trace_list), f)
            dump((model_list, trace_list), f)
    else:
        with open(trace_path, 'rb') as f:
            model_list, trace_list = load(f)
            # model_list, trace_list = pickle.load(f)

    tau_samples = np.stack([x.posterior['tau'] for x in trace_list], axis=0)

    ppc_list = []
    for this_model, this_trace in zip(model_list, trace_list):
        this_ppc = pm.sample_posterior_predictive(
            this_trace, model = this_model, var_names = ['obs','w_latent', 'tau'])
        ppc_list.append(this_ppc)
    ppc_w_latent = np.stack([x.posterior_predictive['w_latent'] for x in ppc_list], axis=0)
    ppc_tau = np.stack([x.posterior_predictive['tau'] for x in ppc_list], axis=0)
    ppc_list = [x.posterior_predictive['obs'] for x in ppc_list]
    mean_ppc = np.stack([np.mean(x, axis=(0,1)) for x in ppc_list], axis=0)

    # Apparatus for extracting state counts
    w_latent_samples = np.stack([x.posterior['w_latent'] for x in trace_list])
    sorted_w_latent = np.sort(w_latent_samples, axis=-1)[...,::-1]
    # chain_w_latent = np.stack(np.array_split(sorted_w_latent, n_chains, axis=1))
    # mean_sorted = np.swapaxes(np.mean(chain_w_latent, axis=2), 0, 1)
    mean_sorted = np.mean(sorted_w_latent, axis=2)

    set_thresh = 0.05
    inds = np.array(list(np.ndindex(mean_sorted.shape)))
    state_frame = pd.DataFrame(
                            dict(
                                dirs = inds[:,0],
                                chains = inds[:,1],
                                states = inds[:,2]+1,
                                dur = mean_sorted.flatten()
                            )
                        )

    yticks = np.arange(len(band_ranges)) + 0.5
    yticks = yticks * ((freq_vec.max() - freq_vec.min()) / len(band_ranges))
    fig, ax = plt.subplots(6, 2, figsize = (10,15))
    for i in range(ax.shape[1]):
        this_frame = state_frame[state_frame['dirs'] == i]
        sns.stripplot(
            data = this_frame,
            x = 'states',
            y = 'dur',
            color = 'k',
            ax = ax[0,i],
            alpha = 0.5,
        );
        ax[0,i].set_title(dir_names[i])
        ax[0,i].plot(mean_sorted[i].T, alpha = 0.7, color = 'grey')
        ax[0,i].axhline(set_thresh, color = 'red', linestyle = '--',
                 label = f'Set thresh : {set_thresh}')
        ax[0,i].legend()
        ax[0,i].set_ylabel('State Durations')
        # ax[1,i].imshow(pca_spectra_list[i], aspect='auto', origin='lower',
        ax[1,i].imshow(1 - zscored_mean_mask_bands[i], aspect='auto', origin='lower',
                       interpolation='nearest',
                       extent=[time_vec[0], time_vec[-1],
                               freq_vec[0], freq_vec[-1]],
                       )
        ax[1,i].set_yticks(yticks)
        ax[1,i].set_yticklabels([str(x) for x in band_ranges])
        ax[1,i].set_ylabel('Frequency (Hz)')
        ax[1,i].set_xlabel('Time')
        ax[1,i].set_title('Zscored Mask')
        ax[2,i].imshow(1 - mean_ppc[i], aspect='auto', origin='lower',
                       interpolation='nearest',
                       extent=[time_vec[0], time_vec[-1],
                               freq_vec[0], freq_vec[-1]],
                       )
        ax[2,i].set_yticks(yticks)
        ax[2,i].set_yticklabels([str(x) for x in band_ranges])
        ax[2,i].set_title('Mean PPC')
        # tau_int = np.round(ppc_tau[i]).astype(int).flatten()
        tau_int = np.round(tau_samples[i]).astype(int).flatten()
        tau_int = np.clip(tau_int, 0, len(time_vec)-1)
        ax[3,i].hist(time_vec[tau_int], 
                     bins = time_vec[:-1], color = 'grey')
        ax[3,i].set_title('All State Tau Distribution')
        # peaks = [-0.07, 0.23, 0.73, 0.93]
        peaks = [-0.07, 0.23, 0.9]
        for this_peak in peaks:
            ax[3,i].axvline(this_peak, color = 'red', linestyle = '--')
        ax[3,i].sharex(ax[2,i])
        # ax[3,i].set_yscale('log')
        bins = np.arange(int(np.max(tau_samples)))
        tau_hist = np.stack([np.histogram(x.flatten(), bins=bins)[0] for x in  tau_samples[i]])
        ax[4,i].imshow(tau_hist, aspect='auto', origin='lower',
                       interpolation='nearest',)
        ax[4,i].set_xlabel('Time bin')
        ax[4,i].set_ylabel('Chain #')
        max_state_per_chain = this_frame.loc[this_frame.dur > set_thresh].groupby('chains').max()
        max_state_counts = max_state_per_chain.groupby('states').count()
        state_vec = np.arange(0,max_states+1)
        counts = [max_state_counts.loc[x].values[0] if x in max_state_counts.index else 0 for x in state_vec ]
        ax[5,i].bar(state_vec, counts)
        ax[5,i].set_xlabel("States")
        ax[5,i].set_ylabel('Count')
        ax[5,i].set_title(f'State Distribution, Thresh = {set_thresh}')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir_base, f'granger_mask_changepoints_dirichlet_{max_states}states.svg'))
    plt.close(fig)
    #plt.show()

##############################
# Get all models and plot flat histograms

model_list = sorted(glob(os.path.join(artifact_dir,'granger*dirichlet*pkl')))
model_states = [int(x.split('states')[0].split('_')[-1]) for x in model_list]
sort_inds = np.argsort(model_states)
model_list = [model_list[i] for i in sort_inds]
model_states = [model_states[i] for i in sort_inds]

peaks = [-0.07, 0.23, 0.9]
fig, ax = plt.subplots(len(model_list), 2, sharex=True, sharey=False,
                       figsize = (10,5))
for i, model_path in enumerate(model_list):
    with open(model_path,'rb') as f:
        this_model = load(f)[1]
    tau_samples = np.stack([x.posterior['tau'] for x in this_model], axis=0)
    for j, this_samples in enumerate(tau_samples):
        int_tau = np.vectorize(int)(this_samples.flatten())
        int_tau[int_tau > (len(time_vec)-1)] = len(time_vec) - 1
        time_samples = time_vec[int_tau]
        ax[i,j].hist(time_samples, bins = time_vec)
        for this_peak in peaks:
            ax[i,j].axvline(this_peak, color = 'red', linestyle = '--', alpha = 0.7)
    ax[i,0].set_ylabel(f'{model_states[i]} states')
fig.suptitle('Aggregate changepoints')
fig.savefig(os.path.join(plot_dir_base, 'changepoint_dirichlet_aggregrate.svg'),
            bbox_inches='tight')
plt.close(fig)
    
    
##############################

yticks = np.arange(len(band_ranges)) + 0.5
yticks = yticks * ((freq_vec.max() - freq_vec.min()) / len(band_ranges))
fig, ax = plt.subplots(5, 2, figsize = (10,15))
for i in range(ax.shape[1]):
    this_frame = state_frame[state_frame['dirs'] == i]
    sns.stripplot(
        data = this_frame,
        x = 'states',
        y = 'dur',
        color = 'k',
        ax = ax[0,i],
        alpha = 0.5,
    );
    ax[0,i].set_title(dir_names[i])
    ax[0,i].plot(mean_sorted[i].T, alpha = 0.7, color = 'grey')
    ax[0,i].axhline(set_thresh, color = 'red', linestyle = '--',
             label = f'Set thresh : {set_thresh}')
    ax[0,i].legend()
    ax[0,i].set_ylabel('State Durations')
    # ax[1,i].imshow(pca_spectra_list[i], aspect='auto', origin='lower',
    ax[1,i].imshow(1 - zscored_mean_mask_bands[i], aspect='auto', origin='lower',
                   interpolation='nearest',
                   extent=[time_vec[0], time_vec[-1],
                           freq_vec[0], freq_vec[-1]],
                   )
    ax[1,i].set_yticks(yticks)
    ax[1,i].set_yticklabels([str(x) for x in band_ranges])
    ax[1,i].set_ylabel('Frequency (Hz)')
    ax[1,i].set_xlabel('Time')
    ax[1,i].set_title('Zscored Mask')
    ax[2,i].imshow(1 - mean_ppc[i], aspect='auto', origin='lower',
                   interpolation='nearest',
                   extent=[time_vec[0], time_vec[-1],
                           freq_vec[0], freq_vec[-1]],
                   )
    ax[2,i].set_yticks(yticks)
    ax[2,i].set_yticklabels([str(x) for x in band_ranges])
    ax[2,i].set_title('Mean PPC')
    tau_int = np.round(ppc_tau[i]).astype(int).flatten()
    tau_int = np.clip(tau_int, 0, len(time_vec)-1)
    ax[3,i].hist(time_vec[tau_int], 
                 bins = time_vec[:-1], color = 'grey')
    ax[3,i].set_title('All State Tau Distribution')
    # peaks = [-0.07, 0.23, 0.73, 0.93]
    peaks = [-0.07, 0.23, 0.9]
    for this_peak in peaks:
        ax[3,i].axvline(this_peak, color = 'red', linestyle = '--')
    ax[3,i].sharex(ax[2,i])
    # ax[3,i].set_yscale('log')
    max_state_per_chain = this_frame.loc[this_frame.dur > set_thresh].groupby('chains').max()
    max_state_counts = max_state_per_chain.groupby('states').count()
    state_vec = np.arange(0,max_states+1)
    counts = [max_state_counts.loc[x].values[0] if x in max_state_counts.index else 0 for x in state_vec ]
    ax[4,i].bar(state_vec, counts)
    ax[4,i].set_xlabel("States")
    ax[4,i].set_ylabel('Count')
    ax[4,i].set_title(f'State Distribution, Thresh = {set_thresh}')
time_lims = [-0.5, time_vec[-1]]
for this_ax in ax[1:].flatten():
    this_ax.set_xlim(time_lims)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'granger_mask_changepoints_dirichlet_cut.svg'))
plt.close(fig)
#plt.show()

##############################
# Plot 5 state changepoint hists using ppc
wanted_n_states = 5
ppc_obs_array = np.stack(ppc_list)
scaled_ppc_tau_samples = time_vec[ppc_tau.astype('int')]
w_state_counts = np.sum(ppc_w_latent > set_thresh,axis=-1)
wanted_inds = np.where(w_state_counts == wanted_n_states)
wanted_tau = [this_scaled_tau[wanted_inds[1][wanted_inds[0]==i]] \
        for i, this_scaled_tau in enumerate(scaled_ppc_tau_samples)]
wanted_obs = [this_ppc_obs[wanted_inds[1][wanted_inds[0]==i]] \
        for i, this_ppc_obs in enumerate(ppc_obs_array)]

mean_wanted_obs = np.stack([np.mean(this_obs, axis=0) for this_obs in wanted_obs])
fig, ax = plt.subplots(2,2, sharex=True)
for i in range(ax.shape[0]):
    ax[0,i].hist(wanted_tau[i].flatten(), bins = time_vec, color = 'grey')
    ax[0,i].set_xlabel('Time (s)')
    ax[0,i].set_ylabel('Count')
    ax[0,i].set_title(dir_names[i])
    ax[1,i].imshow(zscored_mean_mask_bands[i], aspect='auto', origin='lower',
                   extent=[time_vec[0], time_vec[-1],
                           freq_vec[0], freq_vec[-1]],
                   interpolation='none')
    ax[1,i].set_yticks(yticks)
    ax[1,i].set_yticklabels([str(x) for x in band_ranges])
    ax[1,i].set_ylabel('Frequency (Hz)')
    ax[1,i].set_xlabel('Time (s)')
    ax[1,i].set_title('Mean PPC')
peaks = [-0.07, 0.23, 0.73, 0.93]
for this_ax in ax.flatten():
    for this_peak in peaks:
        this_ax.axvline(this_peak, color = 'red', linestyle = '--')
fig.suptitle('Tau Distribution, 5 State\n' + str(peaks) )
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'granger_mask_changepoints_dirichlet_5state.svg'),
            )
plt.close(fig)

# Plot per transition histograms for each tau
fig, ax = plt.subplots(10,2, sharex=True)
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        this_ax = ax[i,j]
        this_dat = wanted_tau[j][:,i]
        this_ax.hist(this_dat, bins = time_vec, color = 'grey')
plt.show()

# Plot pairs of state changepoint times
import itertools as it
inds = list(it.combinations(np.arange(wanted_n_states),2))
inds = [x for x in inds if np.diff(x) == 1]
fig, ax = plt.subplots(len(inds),2, sharex=True)
for i, this_ind in enumerate(inds):
    for j in range(ax.shape[1]):
        ax[i,j].scatter(
                wanted_tau[j][:,this_ind[0]].flatten(), 
                wanted_tau[j][:,this_ind[1]].flatten(),
                        color = 'grey', s = 1)
        ax[i,j].set_xlabel('Time (s)')
        ax[i,j].set_ylabel('Time (s)')
        ax[i,j].set_title(f'State {this_ind[j]} to {this_ind[j]+1}')
plt.tight_layout()
plt.show()

## Tau raster plot
#tau_raster = np.zeros((len(dir_names), tau_samples.shape[1], len(time_vec)))
#inds = np.array(list(np.ndindex(tau_samples.shape[:-1])))
#for this_ind in inds:
#    tau_vals = tau_samples[tuple(this_ind)]
#    tau_raster[tuple(this_ind)][tau_vals.astype('int')] = 1
#
## Plot tau raster
#fig, ax = plt.subplots(2,1, sharex=True)
#for i in range(ax.shape[0]):
#    this_ax = ax[i]
#    this_ax.scatter(np.where(tau_raster[i])[1], np.where(tau_raster[i])[0],
#                    color = 'grey', s = 1)
#    this_ax.set_xlabel('Time')
#    this_ax.set_ylabel('Sample')
#    this_ax.set_title(dir_names[i])
#plt.tight_layout()
#plt.show()



## Plot mode changepoints over mask and mean PPC
#fig, ax = plt.subplots(2,2)
#for i in range(ax.shape[1]):
#    this_frame = state_frame[state_frame['dirs'] == i]
#    max_state_per_chain = this_frame.loc[this_frame.dur > set_thresh].groupby('chains').max()
#    max_state_counts = max_state_per_chain.groupby('states').count()
#    state_vec = np.arange(1,max_states+1)
#    counts = [max_state_counts.loc[x].values[0] if x in max_state_counts.index else 0 for x in state_vec ]
#    best_state_num = state_vec[np.argmax(counts)]
#    this_tau = tau_samples[i]
#    this_w = sorted_w_latent[i]
#    w_state_counts = np.sum(this_w > set_thresh,axis=-1)
#    wanted_inds = np.where(w_state_counts == best_state_num)[0]
#    wanted_tau = this_tau[wanted_inds]
#    mode_tau = stats.mode(wanted_tau.astype('int'), axis=0)[0][0]
#    ax[0,i].imshow(zscored_mean_mask_bands[i], aspect='auto', origin='lower',
#                   interpolation='none',
#                   extent=[time_vec[0], time_vec[-1],
#                           freq_vec[0], freq_vec[-1]],
#                   )
#    ax[0,i].set_yticks(yticks)
#    ax[0,i].set_yticklabels([str(x) for x in band_ranges])
#    ax[0,i].set_ylabel('Frequency (Hz)')
#    ax[0,i].set_xlabel('Time (s)')
#    ax[0,i].set_title('Zscored Mask')
#    for this_mode in mode_tau:
#        ax[0,i].axvline(time_vec[this_mode], color = 'red', linestyle = '--',
#                        label = 'Mode Changepoint')
#    ax[1,i].imshow(mean_ppc[i], aspect='auto', origin='lower',
#                    interpolation='none',
#                   extent=[time_vec[0], time_vec[-1],
#                           freq_vec[0], freq_vec[-1]],
#                   )
#    ax[1,i].set_yticks(yticks)
#    ax[1,i].set_yticklabels([str(x) for x in band_ranges])
#    ax[1,i].set_ylabel('Frequency (Hz)')
#    ax[1,i].set_xlabel('Time (s)')
#    ax[1,i].set_title('Mean PPC')
#    for this_mode in mode_tau:
#        ax[1,i].axvline(time_vec[this_mode], color = 'red', linestyle = '--',
#                        label = 'Mode Changepoint')
#plt.tight_layout()
#plt.show()



############################################################
# Using ELBO 
############################################################
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

############################################################

mean_pca_mask = np.load(os.path.join(artifact_dir, 'mean_pca_mask.npy'))
max_states = 10
n_chains = 10
n_cores = np.min([n_chains, cpu_count()])
dpp_model = models.gaussian_changepoint_mean_dirichlet(
        mean_pca_mask, max_states = max_states)
with dpp_model:
    # dpp_trace = pm.sample_smc(return_inferencedata=False)
    # Numpyro sampler seems to ignore return_inferencedata flag
    rng = np.random.default_rng(666)
    dpp_trace = pm.sample(
                        tune = 50,
                        draws = 200, 
                        target_accept = 0.95,
                        chains = int(n_chains),
                        cores = int(n_cores),
                        nuts_sampler = 'numpyro',
                        return_inferencedata=True,
                        random_seed = rng)
tau_samples = dpp_trace.posterior['tau'].values 
np.save(
        os.path.join(artifact_dir, 'pca_mask_changepoints_tau.npy'),
        tau_samples
        )

fig, ax = plt.subplots(2,1, sharex=True)
n_bins = mean_pca_mask.shape[-1]
ax[0].imshow(mean_pca_mask, interpolation='none', aspect='auto')
ax[1].hist(tau_samples.flatten(), bins = n_bins) 
ax[1].hist(np.random.uniform(0, n_bins, len(tau_samples.flatten())), bins = n_bins,
        alpha = 0.3, color = 'k')
ax[1].axhline(len(tau_samples.flatten()) / n_bins, color = 'k')
ax.set_xlim([0, mean_pca_mask.shape[-1]])
plt.show()
