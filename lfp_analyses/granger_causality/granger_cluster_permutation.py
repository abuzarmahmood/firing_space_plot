"""
Perform cluster-permutation testing on granger causality data to determine
hotspots of significant causality.
"""

import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import pickle
from scipy.stats import percentileofscore
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from umap import UMAP
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib

import pandas as pd
import seaborn as sns
from collections import Counter

import sys
# sys.path.append('/media/bigdata/projects/pytau')
# import pytau.changepoint_model as models

# Get connected components
def connected_regions_1D(data):
    """
    Get connected components in 1D data

    Inputs:
        data : 1D np.array

    Outputs:
        labels : np.array
            Labels for connected components
    """
    binary_data = data > 0
    labels = []
    current_label = 0
    for i in binary_data: 
        if i:
            labels.append(current_label)
        else:
            current_label += 1
            labels.append(current_label)
    return np.array(labels)

def get_largest_cluster_sizes(data, n = 2, return_labels = False):
    """
    Get the n largest cluster sizes in 1D data

    Inputs:
        data : 1D np.array
            Timeseries of labels
        n : int

    Outputs:
        cluster_sizes : list of tuples
            tuple = (label, size)

    """
    labels = connected_regions_1D(data)
    cluster_sizes = Counter(labels)
    cluster_sizes = cluster_sizes.most_common(n)
    if return_labels:
        return cluster_sizes
    else:
        cluster_sizes = [x[1] for x in cluster_sizes]
        return cluster_sizes

def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                     edgecolor='none', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe, y - ye), xe * 2, ye * 2) 
                  for x, y, xe, ye in zip(xdata, ydata, xerror, yerror)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    # Add collection to Axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='none', ecolor='k', alpha=0.1)

    return artists

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

# Adjust time_vec to account for STFP/Granger Causality window offset
time_vec += 0.3
# Round time_vec to 2 decimal places
time_vec = np.round(time_vec, 2)

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
# shape : direction, freq, time
mean_mask = np.stack([mean_mask[...,0,1],mean_mask[...,1,0],], axis=-1).T

##############################
# Assess similarity in dynamics of either direction

reg = LinearRegression()
# dir_0 = mean_mask[0]
# dir_1 = mean_mask[1]
dir_0 = mean_mask[1]
dir_1 = mean_mask[0]
reg.fit(dir_0.T, dir_1.T)
dir_0_reg = reg.predict(dir_0.T).T
reg_mask = [dir_0_reg, dir_1]


# Get PCA of each direction
# pca_mean_mask = [PCA(n_components=3).fit_transform(this_data.T).T for this_data in mean_mask]
pca_mean_mask = [PCA(n_components=3).fit_transform(this_data.T).T for this_data in reg_mask]
# from sklearn.metrics import explained_variance_score
# pca_mean_mask = [UMAP(n_components=3).fit_transform(this_data.T).T for this_data in mean_mask]
# Center the projections 
pca_mean_mask = [x - x.mean(axis=-1)[:,None] for x in pca_mean_mask]
pca_mean_mask = [x / x.std(axis=-1)[:,None] for x in pca_mean_mask]

mean_pca_mask = np.stack(pca_mean_mask).mean(axis=0)
np.save(
        os.path.join(artifact_dir, 'mean_pca_mask.npy'),
        mean_pca_mask
        )

tau_samples = np.load(
        os.path.join(artifact_dir, 'pca_mask_changepoints_tau.npy'),
        )
# tau_time = [time_vec[int(x)] for x in tau_samples.flatten()]
# interpolate to get time from tau_samples
from scipy.interpolate import interp1d
f = interp1d(np.arange(len(time_vec)), time_vec)
tau_time = f(tau_samples.flatten()-1)

fig, ax = plt.subplots(2,1, sharex=True)
n_bins = mean_pca_mask.shape[-1]
ax[0].pcolormesh(
        time_vec,
        np.arange(len(mean_pca_mask)),
        mean_pca_mask, 
        )
ax[1].hist(tau_time, bins = n_bins) 
ax[1].hist(np.random.uniform(time_vec.min(), time_vec.max(), len(tau_samples.flatten())), 
           bins = n_bins,
        alpha = 0.3, color = 'k')
ax[1].axhline(len(tau_samples.flatten()) / n_bins, color = 'k')
ax[0].axvline(-0.3, color = 'k')
ax[1].axvline(-0.3, color = 'k')
ax[-1].set_xlabel('Time post-stim (s)')
ax[-1].set_ylabel('Count')
fig.suptitle('PCA Mean Mask Granger Changepoints')
fig.savefig(os.path.join(plot_dir_base, 'pca_mask_changepoints.png'),
            bbox_inches='tight')
plt.close(fig)
# plt.show()

# SG-Filter PCA
# pca_mean_mask_smooth = [savgol_filter(this_data, 5, 3) for this_data in pca_mean_mask]
pca_mean_mask_smooth = pca_mean_mask.copy() 


# fig, ax = plt.subplots(2, 1, # figsize=(10,5),
#                        sharex=True, sharey=True)
# for dir_ind, dir_name in enumerate(dir_names):
#     ax[dir_ind].imshow(pca_mean_mask_smooth[dir_ind], aspect='auto')
#     ax[dir_ind].set_title(dir_name)
# plt.tight_layout()
# plt.show()

# Align principal components
# reg = LinearRegression()
# dir_0 = pca_mean_mask[0]
# dir_1 = pca_mean_mask[1]
# reg.fit(dir_0.T, dir_1.T)
# dir_0_reg = reg.predict(dir_0.T).T
# reg = MLPRegressor(hidden_layer_sizes=(100,100,100), max_iter=10000)
reg = LinearRegression()
dir_0 = pca_mean_mask_smooth[0]
dir_1 = pca_mean_mask_smooth[1]
reg.fit(dir_0.T, dir_1.T)
dir_0_reg = reg.predict(dir_0.T).T


# Calculate MSE between directions compared to shuffled data
n_shuffles = 10000
mse_list = []
for i in range(n_shuffles):
    shuffle_inds = np.random.choice(np.arange(dir_0.shape[1]),
                                    size = dir_0.shape[1],
                                    replace = False)
    dir_0_sh = dir_0_reg[:, shuffle_inds]
    dist = np.linalg.norm(dir_0_sh - dir_1, axis=0)
    mse = np.mean(dist)
    mse_list.append(mse)

# Get percentile of actual data
mse = np.linalg.norm(dir_0_reg - dir_1, axis=0)
actual_mse = np.mean(mse)
percentile = percentileofscore(mse_list, actual_mse) 
print(f'Percentile of actual data: {percentile}')

# Perform shapiro-wilks on mse
stat, p = stats.shapiro(mse)
fig, ax = plt.subplots()
plt.plot(time_vec, mse)
ax.set_xlabel('Time post-stim')
ax.set_ylabel('MSE')
fig.suptitle('Linear Pred MSE\n' +\
        f'Shapiro-Wilks normality p : {np.round(p,2)}')
fig.savefig(os.path.join(plot_dir_base, 'mse_vs_time.png'),
            bbox_inches='tight')
plt.close(fig)

# Plot 3 2D projections
inds = [(0,1), (0,2), (1,2)]
fig, ax = plt.subplots(1, 3, figsize=(15,5))
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, dir_0.shape[1])]
stim_ind = np.argmin((time_vec - -0.185)**2)
stim_t = time_vec[stim_ind]
# time_markers = [stim_ind, 28, 40]
time_markers = [
                14
                # stim_ind, 
                # np.argmin((time_vec - 0)**2),
                # np.argmin((time_vec - 0.2)**2),
                # np.argmin((time_vec - 0.7)**2),
                ]
for i, (ind_0, ind_1) in enumerate(inds):
    im = ax[i].scatter(dir_0_reg[ind_0], dir_0_reg[ind_1], color = colors, alpha=0.5)
    ax[i].scatter(dir_1[ind_0], dir_1[ind_1], color = colors, alpha=0.9) 
    for timepoint in range(dir_0.shape[1]):
        ax[i].plot([dir_0_reg[ind_0, timepoint], dir_1[ind_0, timepoint]],
                   [dir_0_reg[ind_1, timepoint], dir_1[ind_1, timepoint]],
                   color = colors[timepoint], alpha=0.9, linewidth=1)
        if timepoint in time_markers:
            ax[i].scatter(
                    dir_0_reg[ind_0, timepoint], 
                    dir_0_reg[ind_1, timepoint], 
                          color='r', s=100, alpha=0.7, zorder = 10)
            ax[i].scatter(
                    dir_1[ind_0, timepoint], 
                    dir_1[ind_1, timepoint], 
                          color='r', s=100, alpha=0.7, zorder = 10)
    ax[i].plot(dir_0_reg[ind_0], dir_0_reg[ind_1], color='k', linestyle='--',
               label = 'BLA-->GC')
    ax[i].plot(dir_1[ind_0], dir_1[ind_1], color='k', label = 'GC-->BLA')
    ax[i].set_title(f'PCA {ind_0} vs {ind_1}')
plt.legend()
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
plt.colorbar(im, cax=cbar_ax, label='Time (s)')
fig.suptitle('PCA of Granger Causality Masks\n' +\
        f'MSE Percentile: {percentile}' + '\n' +\
        f'Time markers: {[np.round(time_vec[x],2) for x in time_markers]}')
plt.subplots_adjust(top=0.8)
fig.savefig(os.path.join(plot_dir_base, 'pca_granger_causality.svg'),
            bbox_inches='tight')
plt.close(fig)
# plt.tight_layout()
# plt.show()

# Plot 3D Projections
norm = matplotlib.colors.Normalize(vmin = time_vec.min(), vmax = time_vec.max())
colors = [cmap(norm(x)) for x in time_vec]
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
im = ax.scatter(*dir_0_reg, c = colors, norm = norm) 
ax.scatter(*dir_1, c = colors)
ax.plot(*dir_0_reg, c = 'k', alpha = 0.5)
ax.plot(*dir_1, c = 'k', alpha = 0.5)
for timepoint in range(dir_0.shape[1]):
    ax.plot(
           *list(zip(
           dir_0_reg[:, timepoint], 
           dir_1[:, timepoint],
           )),
           color = colors[timepoint], 
           alpha=0.9, linewidth=1)
    # Mark time_vec == 0 with a big black circle
    if timepoint in time_markers:
        ax.scatter(*dir_0_reg[:, timepoint], color='r', s=100, alpha=0.7, zorder = 10)
        ax.scatter(*dir_1[:, timepoint], color='r', s=100, alpha=0.7, zorder = 10)
# Put cbat on bottom
cbar = plt.colorbar(im, 
                    ax = ax, 
                    label = 'Time post-stimulus (s)', 
                    # pad = 0.1,
                    location = 'left',
                    # orientation = 'horizontal',
                    fraction = 0.03,
                    ) 
# Change cbar ticklabels
# cbar_ticks = cbar.get_ticks()
# time_inds = [np.round(time_vec[int(x*len(time_vec))-1], 2) for x in cbar_ticks]
wanted_tick_times = np.arange(-0.5, 2, 0.5)
tick_inds = [(x-time_vec.min())/(time_vec.max()-time_vec.min()) for x in wanted_tick_times]
# cbar.set_ticklabels(time_inds)
cbar.set_ticks(tick_inds)
cbar.set_ticklabels(wanted_tick_times)
cbar.ax.axhline((0-time_vec.min())/(time_vec.max()-time_vec.min()),
                color='r', linestyle='-', linewidth = 5)
ax.set_xlabel('PCA 0')
ax.set_ylabel('PCA 1')
ax.set_zlabel('PCA 2')
fig.suptitle('PCA of Granger Causality Masks\n' +\
        f'MSE Percentile: {percentile}' + '\n' +\
        f'Time markers: {[np.round(time_vec[x],2) for x in time_markers]}')
# plt.tight_layout()
plt.subplots_adjust(rightg=0.8)
fig.savefig(os.path.join(plot_dir_base, 'pca_granger_3d.svg'),
                        # bbox_inches = 'tight'
            )
plt.close(fig)

# Plot 3D with PC1, PC2, and Time as dimensions (each direction in different color)
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(projection='3d')

# Plot BLA-->GC in blue
ax.plot(dir_0_reg[0], dir_0_reg[1], time_vec, 
        color='blue', linewidth=3, alpha=0.8, label='BLA-->GC')
ax.scatter(dir_0_reg[0], dir_0_reg[1], time_vec, 
           color='blue', s=20, alpha=0.6)

# Plot GC-->BLA in red  
ax.plot(dir_1[0], dir_1[1], time_vec,
        color='red', linewidth=3, alpha=0.8, label='GC-->BLA')
ax.scatter(dir_1[0], dir_1[1], time_vec,
           color='red', s=20, alpha=0.6)

# Mark time markers
for timepoint in time_markers:
    ax.scatter(dir_0_reg[0, timepoint], dir_0_reg[1, timepoint], time_vec[timepoint],
               color='blue', s=100, alpha=1.0, zorder=10, edgecolors='black', linewidth=2)
    ax.scatter(dir_1[0, timepoint], dir_1[1, timepoint], time_vec[timepoint], 
               color='red', s=100, alpha=1.0, zorder=10, edgecolors='black', linewidth=2)

# Mark stimulus onset (time = 0)
stim_onset_ind = np.argmin(np.abs(time_vec))
ax.scatter(dir_0_reg[0, stim_onset_ind], dir_0_reg[1, stim_onset_ind], time_vec[stim_onset_ind],
           color='green', s=150, alpha=1.0, zorder=15, marker='*', edgecolors='black', linewidth=2)
ax.scatter(dir_1[0, stim_onset_ind], dir_1[1, stim_onset_ind], time_vec[stim_onset_ind],
           color='green', s=150, alpha=1.0, zorder=15, marker='*', edgecolors='black', linewidth=2)

ax.set_xlabel('PCA 0')
ax.set_ylabel('PCA 1') 
ax.set_zlabel('Time post-stimulus (s)')
ax.legend()
fig.suptitle('PCA Trajectories: PC1 vs PC2 vs Time\n' +\
        f'MSE Percentile: {percentile}' + '\n' +\
        f'Time markers: {[np.round(time_vec[x],2) for x in time_markers]}' + '\n' +\
        'Green star: Stimulus onset')
fig.savefig(os.path.join(plot_dir_base, 'pca_granger_pc1_pc2_time.svg'),
            bbox_inches='tight')
plt.close(fig)

# fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
# ax[0,0].imshow(dir_0_reg, interpolation='nearest', aspect='auto')
# ax[1,0].imshow(dir_1, interpolation='nearest', aspect='auto')
# ax[0,0].set_title('Registered')
# ax[0,1].imshow(stats.zscore(dir_0,axis=-1), interpolation='nearest', aspect='auto')
# ax[1,1].imshow(dir_1, interpolation='nearest', aspect='auto')
# ax[0,1].set_title('Un-registered')
# fig.savefig(os.path.join(plot_dir_base, 'pca_granger_3d_imshow.png'),
#                         bbox_inches = 'tight')
# plt.close(fig)
fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
ax[0].imshow(dir_0_reg, interpolation='nearest', aspect='auto')
ax[1].imshow(dir_1, interpolation='nearest', aspect='auto')
fig.savefig(os.path.join(plot_dir_base, 'pca_granger_3d_imshow.png'),
                        bbox_inches = 'tight')
plt.close(fig)

# Make a plot with time as the 3d dimension



# ##############################
# # Estimate switches using ARHMM
# import jax.numpy as jnp
# import jax.random as jr
# import matplotlib.pyplot as plt
# 
# from dynamax.hidden_markov_model import LinearAutoregressiveHMM
# from dynamax.utils.plotting import gradient_cmap
# from dynamax.utils.utils import random_rotation
# 
# # Initialize with K-Means
# num_states = 5
# emission_dim = 3
# num_lags = 1
# # emissions = dir_0_reg.copy().T
# emissions = dir_1.copy().T
# arhmm = LinearAutoregressiveHMM(num_states, emission_dim, num_lags=num_lags)
# params, props = arhmm.initialize(key=jr.PRNGKey(1), method="kmeans", emissions=emissions)
# 
# # Fit with EM
# inputs = arhmm.compute_inputs(emissions)
# fitted_params, lps = arhmm.fit_em(params, props, emissions, inputs=inputs)
# 
# posterior = arhmm.smoother(fitted_params, emissions, inputs=inputs)
# most_likely_states = arhmm.most_likely_states(fitted_params, emissions, inputs=inputs)
# 
# plt.plot(most_likely_states)
# plt.show()

###############
# Perform autoregressive prediction on dir_1
# Shape: pca, time
n_lags = 3

dir_1_lagged = [] 
for i in range(dir_1.shape[1] - n_lags):
    this_lagged = dir_1[:, i:i+n_lags]
    dir_1_lagged.append(this_lagged)
dir_1_lagged = np.stack(dir_1_lagged, axis=-1)
dir_1_lagged = dir_1_lagged.reshape(-1, dir_1_lagged.shape[-1]).T 
X = dir_1_lagged[:-1]
y = dir_1[:, n_lags+1:].T

# Fit model
n_repeats = 500
all_preds = []
for i in trange(n_repeats):
    regressor = LinearRegression()
    inds = np.random.choice(np.arange(X.shape[0]), size = X.shape[0], replace = True)
    regressor.fit(X[inds], y[inds])
    predicted_dir_1 = regressor.predict(X)
    all_preds.append(predicted_dir_1)
predicted_dir_1_array = np.stack(all_preds, axis=0)
lr_mse_auto_pred_list = np.mean(np.linalg.norm(y[None,:,:] - predicted_dir_1_array, axis=-1), axis = -1)


# Fit model
n_repeats = 500
all_preds = []
for i in trange(n_repeats):
    regressor = MLPRegressor(hidden_layer_sizes=(100,100,100), max_iter=1000)
    # regressor = LinearRegression()
    inds = np.random.choice(np.arange(X.shape[0]), size = X.shape[0], replace = True)
    regressor.fit(X[inds], y[inds])
    predicted_dir_1 = regressor.predict(X)
    all_preds.append(predicted_dir_1)
predicted_dir_1_array = np.stack(all_preds, axis=0)

# Calculate MSE of predicted_dir_1_array compared to y
# mse_auto_pred_list = [np.mean(np.linalg.norm(y - this_pred, axis=0)) for this_pred in predicted_dir_1_array] 
mlp_mse_auto_pred_list = np.mean(np.linalg.norm(y[None,:,:] - predicted_dir_1_array, axis=-1), axis = -1)

predicted_dir_1_array_flat = predicted_dir_1_array.reshape(-1, predicted_dir_1_array.shape[-1])

inds = [(0,1), (0,2), (1,2)]
fig, ax = plt.subplots(1, 3, figsize=(15,5))
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(y))] 
for i, (ind_0, ind_1) in enumerate(inds):
    ax[i].set_title(f'PCA {ind_0} vs {ind_1}')
    im = ax[i].hist2d(predicted_dir_1_array_flat[:,ind_0], predicted_dir_1_array_flat[:,ind_1], 
                 bins=20, cmap='viridis', norm = 'log')
    ax[i].plot(dir_0_reg[ind_0], dir_0_reg[ind_1], '-o', color='k')
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
plt.colorbar(im[3], cax=cbar_ax, label='Log Count')
# plt.legend()
# plt.show()
fig.suptitle('BLA-->GC vs Error in GC-->BLA Prediction')
plt.subplots_adjust(top=0.85)
fig.savefig(os.path.join(plot_dir_base, 'pca_granger_causality_pred_hist.svg'),
            bbox_inches='tight')
plt.close(fig)

inds = [(0,1), (0,2), (1,2)]
fig, ax = plt.subplots(1, 3, figsize=(15,5))
cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, len(y))] 
for i, (ind_0, ind_1) in enumerate(inds):
    ax[i].set_title(f'PCA {ind_0} vs {ind_1}')
    for line in predicted_dir_1_array:
        ax[i].plot(line[:,ind_0], line[:,ind_1], color='grey', alpha=0.1)
    ax[i].plot(dir_0_reg[ind_0], dir_0_reg[ind_1], '-o', color='k')
fig.suptitle('BLA-->GC vs Error in GC-->BLA Prediction')
plt.subplots_adjust(top=0.85)
fig.savefig(os.path.join(plot_dir_base, 'pca_granger_causality_pred_line.svg'),
            bbox_inches='tight')
plt.close(fig)

###############
plt.figure()
plt.hist(lr_mse_auto_pred_list, bins=40,
         label = 'Auto-predicted MSE')
plt.hist(mse_list, bins=40, alpha=0.5,
         label = 'Shuffled MSE')
plt.axvline(actual_mse, color='red', label = 'Other Direction Predicted MSE')
plt.yscale('log')
plt.xscale('log')
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.title('MSE of Auto-Regressive Prediction')
plt.savefig(os.path.join(plot_dir_base, 'mse_auto_pred.png'),
            bbox_inches='tight')
plt.close()

###############
# Bar plot

pred_dir_ind = 0

mse_df = pd.DataFrame(
        dict(
            # type = 'other direction pred shuffled',
            type = f'{dir_names[pred_dir_ind]} linear inter shuffled',
            mse = mse_list 
            )
        )
mse_df = pd.concat([
    pd.DataFrame(
        dict(
            # type = 'non-linear auto-predicted',
            type = f'{dir_names[pred_dir_ind]} non-linear auto',
            mse = mlp_mse_auto_pred_list,
            )
        ),
    mse_df
    ])
mse_df = pd.concat([
    pd.DataFrame(
        dict(
            # type = 'linear auto-predicted',
            type =f'{dir_names[pred_dir_ind]} linear auto',
            mse = lr_mse_auto_pred_list,
            )
        ),
    mse_df
    ])


fig, ax = plt.subplots(figsize = (5,5))
g = sns.boxenplot(
        data = mse_df,
        x = 'type',
        hue = 'type',
        y = 'mse',
        legend=True,
        ax = ax,
        )
ax.legend()
g.axes.axhline(actual_mse, color = 'r', linestyle = '--', linewidth = 2, 
               label = f'{dir_names[pred_dir_ind]} inter')
g.axes.text(0.1, actual_mse*0.75,f'{dir_names[pred_dir_ind]} inter', ha = 'center', c = 'r',
            weight = 'bold')
# Rotate xticklabels
xtick_labels = g.axes.get_xticklabels()
g.axes.set_xticklabels(xtick_labels, rotation=45)
g.figure.savefig(os.path.join(plot_dir_base, f'{dir_names[pred_dir_ind]}_mse_auto_pred_bar.svg'),
                 bbox_inches = 'tight')
plt.legend()
plt.close(g.figure)

# Make above plot but with fold-change relative to actual_mse
mse_df['mse_fold'] = mse_df['mse'] / actual_mse
fig, ax = plt.subplots(figsize = (3,4))
g = sns.boxenplot(
        data = mse_df,
        x = 'type',
        hue = 'type',
        y = 'mse_fold',
        legend=False,
        alpha = 0.5,
        ax = ax,
        )
g.axes.axhline(1, color = 'r', linestyle = '--', linewidth = 2,
               label = "Linear,\nInter")
# g.axes.text(0.1, 0.75,f'{dir_names[pred_dir_ind]} inter', ha = 'center', c = 'r',
#             weight = 'bold')
# Rotate xticklabels
xtick_labels = g.axes.get_xticklabels()
# Drop direction from xtick labels
xtick_labels = [
        " ".join([y for y in x.get_text().split(' ') if '-->' not in y])
        for x in xtick_labels]
# Convert to capitalized
xtick_labels = [x.title() for x in xtick_labels]
# Replace " "With "\n"
xtick_labels = [x.replace(" ", "\n") for x in xtick_labels]
g.axes.set_xticklabels(xtick_labels) 
g.axes.set_ylabel('MSE (fold-change)')
ax.set_xlabel('Prediction Type')
# Remove top and right spines
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
ax.legend()
g.figure.savefig(os.path.join(plot_dir_base, f'{dir_names[pred_dir_ind]}_mse_auto_pred_fold_bar.svg'),
                 bbox_inches = 'tight')
plt.close(g.figure)

###############
fig, ax = plt.subplots(1, 1, figsize=(2,5))
ax.hist(mse_list, bins=40, alpha=0.5,
         color='k',
         label = 'Shuffled MSE',
         orientation='horizontal')
ax.hist(mse_list, bins=40,
         histtype='step', color='k',
         orientation='horizontal'
         )
ax.axhline(actual_mse, color='red', label = 'Other Direction MSE',
           linestyle='--', linewidth=4)
ax.set_title('MSE between directions vs Shuffled')
ax.set_xlabel('Count')
ax.set_ylabel('MSE')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(plot_dir_base, 'mse_shuffled.svg'),
            bbox_inches='tight')
plt.close()

###############

fig, ax = plt.subplots(figsize = (6,4))
g = sns.boxenplot(
        data = mse_df,
        x = 'type',
        hue = 'type',
        y = 'mse',
        legend=False,
        ax = ax,
        )
g.axes.axhline(actual_mse, color = 'r', linestyle = '--', linewidth = 2, 
               label = 'Linear,\nInter')
# g.axes.text(0.1, actual_mse*0.75, 'Linear, Inter', ha = 'center', c = 'r',
#             weight = 'bold')
xtick_labels = g.axes.get_xticklabels()
g.axes.set_xticklabels(
        [
            'Linear,\nAuto', 
            'Non-linear,\nAuto',
            'Linear,\nInter,\nShuffled',
            ],
        # rotation=45
        )
# Remove top and right spines
g.spines['top'].set_visible(False)
g.spines['right'].set_visible(False)
g.axes.set_xlabel('Prediction Type')
g.axes.set_ylabel('MSE')
plt.legend()
g.figure.savefig(os.path.join(plot_dir_base, f'mse_auto_pred_bar.svg'),
                 bbox_inches = 'tight')
plt.close(g.figure)






##############################

##############################
# Shape: direction, frequency, time, session
mask_array = np.stack([mask_array[...,0,1],mask_array[...,1,0],], axis=-1).T

# freq_bands = [[0,17], [17, 60]]
freq_bands = [[0,12], [20, 60]]
freq_inds = [
        np.where(np.logical_and(freq_vec >= this_band[0], 
                                freq_vec <= this_band[1]))[0] 
        for this_band in freq_bands]

# Cut mask_array by frequencies
freq_mask_array = [
            mask_array[:,this_inds]
            for this_inds in freq_inds
            ]

# Make plot with each direction and sub-band
fig, ax = plt.subplots(len(dir_names), len(freq_bands), figsize=(10,3))
for dir_ind, dir_name in enumerate(dir_names):
    for band_ind, band_name in enumerate(freq_bands):
        this_freq_vec = freq_vec[freq_inds[band_ind]]
        this_data = freq_mask_array[band_ind][dir_ind]
        this_mean_data = 1-this_data.mean(axis=-1)
        ax[dir_ind, band_ind].pcolormesh(time_vec, this_freq_vec,
                                        stats.zscore(this_mean_data, axis=-1),
                                         # this_mean_data, 
                                         shading='auto', 
                                         cmap = 'jet',
                                         vmin = -2, vmax = 2)
        ax[dir_ind, band_ind].set_title(f'{dir_name} {band_name}')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'freq_band_masks.png'),
            bbox_inches='tight')
plt.close(fig)
# plt.show()

# Focusing on 0-20 band

wanted_freq_mask = freq_mask_array[0]
wanted_freq_vec = freq_vec[freq_inds[0]]
mean_wanted_freq_mask = wanted_freq_mask.mean(axis=-1)
zscored_mean_mask = stats.zscore(mean_wanted_freq_mask, axis=-1)

frame_list = []
zscore_thresh_vec = np.linspace(0.5, 3, 30)
n_shuffles = 1000
# zscore_thresh = 1.5
for zscore_thresh in tqdm(zscore_thresh_vec):
    zscore_thresh_mask = np.abs(zscored_mean_mask) > zscore_thresh
    zscored_thresh_sum = zscore_thresh_mask.sum(axis=1)
    connected_regions = [connected_regions_1D(this_data) for this_data in zscored_thresh_sum]
    largest_clusters_og = [get_largest_cluster_sizes(this_data, return_labels = True) for this_data in zscored_thresh_sum]
    if not all([len(this_dir) == 2 for this_dir in largest_clusters_og]):
        print(f'Not all clusters found for zscore_thresh {zscore_thresh}')
        continue
    largest_clusters = np.stack([[x[1] for x in this_dir] for this_dir in largest_clusters_og], axis=0)
    largest_cluster_labels = np.stack([[x[0] for x in this_dir] for this_dir in largest_clusters_og], axis=0)
    largest_cluster_inds = [[np.where(this_data == x)[0] for x in this_dir] \
            for this_dir, this_data in zip(largest_cluster_labels, connected_regions)]

    # Shuffle the data
    shuffle_inds_time = np.random.choice(np.arange(zscored_mean_mask.shape[-1]),
                                     size = (n_shuffles, wanted_freq_mask.shape[2], wanted_freq_mask.shape[-1]),
                                    replace = True)
    shuffle_inds_freq = np.random.choice(np.arange(zscored_mean_mask.shape[-2]),
                                     size = (n_shuffles, wanted_freq_mask.shape[1], wanted_freq_mask.shape[-1]),
                                    replace = True)
    wanted_freq_mask_sh = []
    for i in range(n_shuffles):
        sh_session_list = []
        for j in range(wanted_freq_mask.shape[-1]):
            this_session_data = wanted_freq_mask[..., j]
            this_inds = np.ix_(
                    np.arange(this_session_data.shape[0]),
                    shuffle_inds_freq[i,:,j],
                    shuffle_inds_time[i,:,j])
            this_shuffle = this_session_data[this_inds] 
            sh_session_list.append(this_shuffle)
        this_shuffle = np.stack(sh_session_list, axis=-1)
        wanted_freq_mask_sh.append(this_shuffle)
    wanted_freq_mask_sh = np.stack(wanted_freq_mask_sh, axis=0)
    wanted_freq_sh_mean = wanted_freq_mask_sh.mean(axis=-1)
    zscored_freq_sh_mean = stats.zscore(wanted_freq_sh_mean, axis=-1)
    zscore_thresh_mask_shuffle = np.abs(zscored_freq_sh_mean) > zscore_thresh
    zscored_thresh_sum_shuffle = zscore_thresh_mask_shuffle.sum(axis=2)

    # Get largest clusters in original data and shuffle
    # Nested list: n_shuffles, directions (2), n_clusters (2)
    wanted_clusters = 2
    largest_clusters_shuffle = [
            [get_largest_cluster_sizes(this_data, n=wanted_clusters) for this_data in this_shuffle]
            for this_shuffle in zscored_thresh_sum_shuffle]
    # Keep only if length is 2 
    cluster_sh_df = pd.DataFrame(largest_clusters_shuffle)
    cluster_sh_df = cluster_sh_df.melt(
            value_vars = cluster_sh_df.columns,
            var_name = 'direction',
            value_name = 'cluster_size')
    # Drop any rows wil cluster_size != 2
    cluster_sh_df = cluster_sh_df[cluster_sh_df['cluster_size'].apply(len) == wanted_clusters]
    # largest_clusters_shuffle = [
    #         [this_dir for this_dir in this_shuffle if len(this_dir) == 2] \
    #         for this_shuffle in largest_clusters_shuffle]
    min_len = cluster_sh_df.groupby('direction').count().min().values[0] 
    if min_len == 0:
        print(f'No clusters found for zscore_thresh {zscore_thresh}')
        continue
    else:
        print(f'Found {min_len} clusters for zscore_thresh {zscore_thresh}')
    # largest_clusters_shuffle = [this_dir[:min_len] for this_dir in largest_clusters_shuffle]
    # # Shape: n_shuffles, directions, n_clusters
    # largest_clusters_shuffle = np.stack(largest_clusters_shuffle, axis=0)
    # largest_clusters_shuffle = np.swapaxes(largest_clusters_shuffle, 0, 1)

    # Get percentiles of score
    percentiles = np.zeros_like(largest_clusters)
    for dir_ind in range(2):
        for cluster_ind in range(2):
            this_dir_sh = cluster_sh_df[cluster_sh_df['direction'] == dir_ind]
            this_shuffle = this_dir_sh['cluster_size'].apply(lambda x: x[cluster_ind]).values 
            this_actual = largest_clusters[dir_ind,cluster_ind]
            percentile = stats.percentileofscore(this_shuffle, this_actual)
            percentiles[dir_ind, cluster_ind] = percentile

    cluster_frame = pd.DataFrame(
            dict(
                direction = np.arange(2),
                zscore_thresh = zscore_thresh,
                largest_clusters = largest_clusters.tolist(),
                percentiles = percentiles.tolist(),
                cluster_inds = largest_cluster_inds,
                )
            )
    frame_list.append(cluster_frame)

cluster_sig_frame = pd.concat(frame_list)
percentile_lims = [2.5, 97.5]
cluster_sig_frame_long = cluster_sig_frame.explode(
        ['largest_clusters', 'percentiles', 'cluster_inds'])
# Keep only if percentile is outside of limits
cluster_sig_frame_long = cluster_sig_frame_long[
        cluster_sig_frame_long['percentiles'].apply(
            lambda x: x < percentile_lims[0] or x > percentile_lims[1]
            )
        ]

# For each direction, get median start and end points for each cluster
cluster_sig_frame_long['start_time'] = cluster_sig_frame_long['cluster_inds'].apply(
        lambda x: x[0])
cluster_sig_frame_long['end_time'] = cluster_sig_frame_long['cluster_inds'].apply(
        lambda x: x[-1])

##############################

# Create cmap from 0-100
norm = plt.Normalize(0, 100)
fig, ax = plt.subplots(2,2, sharex=True, sharey='row',
                       figsize=(10,4))
for dir_ind, dir_name in enumerate(dir_names):
    this_frame = cluster_sig_frame[cluster_sig_frame['direction'] == dir_ind]
    for i, row in this_frame.iterrows():
        for this_clust in range(2):
            this_percentile = row['percentiles'][this_clust]
            this_inds = row['cluster_inds'][this_clust]
            this_zscore_thresh = row['zscore_thresh']
            if this_percentile < percentile_lims[0] or this_percentile > percentile_lims[1]:
                im = ax[1, dir_ind].scatter(time_vec[this_inds], 
                                    np.ones_like(this_inds)*this_zscore_thresh,
                                    c = np.ones_like(this_inds)*this_percentile, 
                                    norm = norm, cmap = 'jet', alpha = 0.5)
            else:
                ax[1, dir_ind].scatter(time_vec[this_inds], 
                                    np.ones_like(this_inds)*this_zscore_thresh,
                                    c = 'grey',
                                    alpha = 0.5)
    ax[0, dir_ind].pcolormesh(time_vec, wanted_freq_vec,
                              1-zscored_mean_mask[dir_ind],
                              cmap='jet', shading='auto')
    ax[0, dir_ind].set_title(dir_name)
    ax[1, dir_ind].set_xlabel('Time (s)')
ax[0,0].set_ylabel('Frequency (Hz)')
ax[1,0].set_ylabel('Zscore Threshold')
plt.sca(ax[1, dir_ind])
cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
plt.colorbar(im, cax=cbar_ax, label='Shuffle Percentile')
fig.suptitle('Percentile of Cluster Size')
fig.savefig(os.path.join(plot_dir_base, 'cluster_percentiles.png'),
            bbox_inches='tight')
plt.close(fig)
# plt.show()

plot_cluster_sig_long = cluster_sig_frame_long.copy()
# Remove clusters in dir==1 starting before 30
plot_cluster_sig_long = plot_cluster_sig_long[
        np.logical_or(
            plot_cluster_sig_long['direction'] == 0,
            plot_cluster_sig_long['start_time'] > 30
            )
        ]
cluster_lims = plot_cluster_sig_long.groupby('direction').agg(
        {'start_time': 'median', 'end_time': 'median'})

# Make plot with each direction and sub-band
fig, ax = plt.subplots(len(dir_names), len(freq_bands), figsize=(10,3))
for dir_ind, dir_name in enumerate(dir_names):
    for band_ind, band_name in enumerate(freq_bands):
        this_freq_vec = freq_vec[freq_inds[band_ind]]
        this_data = freq_mask_array[band_ind][dir_ind]
        this_mean_data = 1-this_data.mean(axis=-1)
        ax[dir_ind, band_ind].pcolormesh(time_vec, this_freq_vec,
                                        stats.zscore(this_mean_data, axis=-1),
                                         # this_mean_data, 
                                         shading='auto', 
                                         cmap = 'jet',
                                         vmin = -2, vmax = 2,
                                         alpha = 0.7)
        ax[dir_ind, band_ind].set_title(f'{dir_name} {band_name}')
        for lims in cluster_lims.loc[dir_ind]:
            ax[dir_ind, 0].axvline(time_vec[int(lims)], color='k',
                                   linewidth=4, linestyle='dotted')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'freq_band_masks_sig.png'),
            bbox_inches='tight')
plt.close(fig)
# plt.show()

# Plot mean mask for each frequency and direction
# Shape: direction, frequency, time
mean_freq_mask = np.stack([this_data.mean(axis=(1,3)) for this_data in freq_mask_array])
std_freq_mask = np.stack([this_data.std(axis=(1,3)) for this_data in freq_mask_array])

fig, ax = plt.subplots(len(dir_names), len(freq_bands), figsize=(10,3),
                       sharex=True, sharey=True)
for dir_ind, dir_name in enumerate(dir_names):
    for band_ind, band_name in enumerate(freq_bands):
        this_freq_vec = freq_vec[freq_inds[band_ind]]
        this_data = 1 - mean_freq_mask[band_ind][dir_ind]
        ax[band_ind, dir_ind].plot(time_vec, this_data, color='k')
        ax[band_ind, dir_ind].set_title(f'{dir_name} {band_name}')
        if band_ind == 0:
            this_ind_lims = cluster_lims.loc[dir_ind]
            this_ind_lims = [int(x) for x in this_ind_lims]
            # this_time_lims = [time_vec[int(x)] for x in this_ind_lims] 
            this_time_vec = time_vec[this_ind_lims[0]:this_ind_lims[1]]
            ax[band_ind, dir_ind].fill_between(
                    this_time_vec,
                    np.zeros_like(this_time_vec),
                    this_data[this_ind_lims[0]:this_ind_lims[1]],
                    color = 'k',
                    alpha = 0.5)
        else:
            ax[band_ind, dir_ind].set_xlabel('Time post-stimulus (s)')
        if dir_ind == 0:
            ax[band_ind, dir_ind].set_ylabel('Sig fraction')
        ax[band_ind, dir_ind].set_xlim([-0.5, max(time_vec)])
        ax[band_ind, dir_ind].axvline(0, color='red', linestyle='--')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir_base, 'freq_band_masks_mean.svg'),
            bbox_inches='tight')
plt.close(fig)

