from glob import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pylab as plt
from itertools import product
from scipy.stats import zscore, ks_2samp, ecdf
from sklearn.decomposition import PCA, NMF
from scipy.signal import correlate
import seaborn as sns
import pingouin as pg
from statsmodels.distributions.empirical_distribution import ECDF

def norm_zero_lag_xcorr(vec1, vec2):
    """
    Calculates normalized zero-lag cross correlation
    Returns a single number
    """
    auto_v1 = np.sum(vec1**2,axis=-1)
    auto_v2 = np.sum(vec2**2,axis=-1)
    xcorr = np.sum(vec1 * vec2,axis=-1)
    denom = np.sqrt(np.multiply(auto_v1,auto_v2))
    return np.divide(xcorr, denom)


############################################################
# Get Granger Data 
base_path = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/granger_state_alignment'
artifact_path = os.path.join(base_path, 'artifacts')
plot_dir = os.path.join(base_path, 'plots')

granger_frame_path = os.path.join(artifact_path,'granger_frame.pkl')
granger_frame = pd.read_pickle(granger_frame_path)

freq_vec = granger_frame.iloc[0].freq_vec
time_vec_orig = granger_frame.iloc[0].time_vec

# Check for nans
granger_frame.dropna(inplace=True)

# Get tau_frame  
tau_frame_name = 'tau_frame.pkl'
tau_frame_path = os.path.join(artifact_path,tau_frame_name)
tau_frame = pd.read_pickle(tau_frame_path)
tau_frame.dropna(inplace=True)

tau_array = np.stack(tau_frame.tau)
mean_tau = tau_array.mean(axis=(0,1))

# Plot tau distributions
fig, ax = plt.subplots()
for this_tau in tau_array.T:
    ax.hist(this_tau.flatten(), bins=25, alpha=0.5)
    ax.axvline(this_tau.mean(), color='k', linestyle='--')
    ylims = ax.get_ylim()
    ax.text(int(this_tau.mean()), ylims[1]*0.1, f'Mean: {int(this_tau.mean())}', rotation=90)
ax.set_title('Tau Distributions')
ax.set_xlabel('Tau (ms)')
ax.set_ylabel('Count')
fig.savefig(os.path.join(plot_dir,'tau_distributions.png'))
plt.close(fig)

############################################################
grouped_granger = list(granger_frame.groupby(['transition', 'aligned']))
group_inds = [x[0] for x in grouped_granger]
group_data = [x[1] for x in grouped_granger]

###############
transition_inds = [x[0] for x in group_inds]
aligned_inds = [x[1] for x in group_inds]
unique_transition_inds = np.unique(transition_inds)
unique_aligned_inds = np.unique(aligned_inds)
aligned_inds_map = {k:v for v,k in enumerate(unique_aligned_inds)}
###############

grouped_mean_granger = [np.stack(x.mean_granger) for x in group_data]
grouped_mask = [np.stack(x.granger_mask) for x in group_data]

mean_granger_per_group = [np.nanmean(x,axis=0) for x in grouped_mean_granger] 
mean_mask_per_group = [np.nanmean(x, axis=0) for x in grouped_mask]

time_lims = [0.5, 1.5]
time_inds = np.logical_and(time_vec_orig >= time_lims[0], time_vec_orig <= time_lims[1])
time_vec = time_vec_orig[time_inds]
mean_granger_per_group = [x[:, time_inds] for x in mean_granger_per_group]
mean_mask_per_group = [x[:, time_inds] for x in mean_mask_per_group]

# Swap axes to make time as x axis
mean_granger_per_group = [np.swapaxes(x, 2, 1) for x in mean_granger_per_group]
mean_mask_per_group = [np.swapaxes(x, 2, 1) for x in mean_mask_per_group]
mean_mask_per_group = [1-x for x in mean_mask_per_group]

# Zscore
mean_granger_per_group = [zscore(x, axis=-1) for x in mean_granger_per_group]
mean_mask_per_group = [zscore(x, axis=-1) for x in mean_mask_per_group]

##############################
# Cross-correlation
xcorr_kern_len = int(0.7 / np.diff(time_vec)[0])
xcorr_kern = np.ones((len(freq_vec), xcorr_kern_len))
xcorr_kern[:, xcorr_kern_len//2:] = 0
xcorr_kern = xcorr_kern / xcorr_kern.sum() 

# Perform cross-correlation on non-meaned data
granger_xcorr_per_group = [
        [
            [np.abs(correlate(x.T, xcorr_kern, mode='valid')) for x in y] for y in z \
            ] for z in grouped_mean_granger
        ]

mask_xcorr_per_group = [
        [
            [np.abs(correlate(x.T, xcorr_kern, mode='valid')) for x in y] for y in z \
            ] for z in grouped_mask
        ]

##############################
# Test plot
test = grouped_mean_granger[0][0][0].T
test_corr = correlate(test, xcorr_kern, mode='valid').flatten()

# Convolve timevec with kern of same length
xcorr_time_vec = np.convolve(
        time_vec_orig, np.ones(xcorr_kern_len)/xcorr_kern_len, 
        mode='valid')


fig, ax = plt.subplots(3,1, sharex=True)
ax[0].pcolormesh(time_vec_orig[:xcorr_kern_len], freq_vec, xcorr_kern)
ax[1].pcolormesh(time_vec_orig, freq_vec[1:], test[1:])
ax[2].plot(xcorr_time_vec, test_corr)
fig.savefig(os.path.join(plot_dir, 'test_xcorr.png'))
plt.close(fig)
##############################

# mean_granger_xcorr = np.stack(
#         [np.squeeze(np.nanmax(x, axis=0)) for x in granger_xcorr_per_group]
#         )
# mean_mask_xcorr = np.stack(
#         [np.squeeze(np.nanmax(x, axis=0)) for x in mask_xcorr_per_group]
#         )

# Convert to dataframe
xcorr_frame_list = []
for i in range(len(transition_inds)):
    for k in range(len(granger_xcorr_per_group[i])):
        for j in range(len(granger_xcorr_per_group[i][k])):
            this_granger_xcorr = granger_xcorr_per_group[i][k][j]
            xcorr_frame_list.append(
                    pd.DataFrame(
                        dict(
                            transition=transition_inds[i],
                            aligned=aligned_inds[i],
                            granger_xcorr=granger_xcorr_per_group[i][k][j].flatten(),
                            mask_xcorr=mask_xcorr_per_group[i][k][j].flatten(),
                            time = xcorr_time_vec,
                            region = j, 
                            )
                        )
                    )

xcorr_frame = pd.concat(xcorr_frame_list)
xcorr_frame.reset_index(drop=True, inplace=True)
xcorr_frame.dropna(inplace=True)

sns.relplot(
        data=xcorr_frame,
        x='time',
        y='granger_xcorr',
        hue = 'aligned',
        row='transition',
        col='region',
        kind='line',
        errorbar='se',
        )
plt.savefig(os.path.join(plot_dir, 'granger_xcorr.png'))
plt.close()

# Plot only time==1.0 
xcorr_frame_1 = xcorr_frame[
        np.logical_and(
            xcorr_frame.time > 0.96,
            xcorr_frame.time < 1.05
            )
        ]

sns.catplot(
        data=xcorr_frame_1,
        x='region',
        y='granger_xcorr',
        hue='aligned',
        col='transition',
        kind='boxen',
        )
plt.savefig(os.path.join(plot_dir, 'granger_xcorr_1.png'))
plt.close()

sns.boxenplot(
        data=xcorr_frame_1,
        x = 'aligned',
        y = 'granger_xcorr',
        )
plt.savefig(os.path.join(plot_dir, 'granger_xcorr_1_aligned.png'))
plt.close()

sns.ecdfplot(
        data=xcorr_frame_1,
        x='granger_xcorr',
        hue='aligned',
        linewidth=5,
        alpha=0.7,
        )
plt.savefig(os.path.join(plot_dir, 'granger_xcorr_1_aligned_ecdf.png'))
plt.close()

pg.mwu(
        x=xcorr_frame_1.granger_xcorr[xcorr_frame_1.aligned==True],
        y=xcorr_frame_1.granger_xcorr[xcorr_frame_1.aligned==False],
        alternative='two-sided',
        )

ks_2samp(
        xcorr_frame_1.granger_xcorr[xcorr_frame_1.aligned==True],
        xcorr_frame_1.granger_xcorr[xcorr_frame_1.aligned==False],
        )

# Generate ecdf and linearly interpolate 
# to have them both on the same x-axis
ecdf_aligned = ECDF(xcorr_frame_1.granger_xcorr[xcorr_frame_1.aligned==True])
ecdf_unaligned = ECDF(xcorr_frame_1.granger_xcorr[xcorr_frame_1.aligned==False])

x = np.linspace(
        min(xcorr_frame_1.granger_xcorr),
        max(xcorr_frame_1.granger_xcorr),
        20,
        )
y_aligned = ecdf_aligned(x)
y_unaligned = ecdf_unaligned(x)

fig, ax = plt.subplots()
ax.plot(x, y_aligned, '-o', label='Aligned', linewidth=5, alpha=0.7)
ax.plot(x, y_unaligned, '-o', label='Unaligned', linewidth=5, alpha=0.7)
ax.legend()
ax.set_xlabel('Granger Step Function XCorr')
ax.set_ylabel('ECDF')
fig.savefig(os.path.join(plot_dir, 'granger_xcorr_1_aligned_ecdf_interp.png'))
plt.close(fig)

##############################
# Scaled XCorr at center point
# Test plot

# Only perform on time = [0.75,1.25]
window_lims = [0.75, 1.25]
window_inds = np.logical_and(
        time_vec_orig > window_lims[0],
        time_vec_orig < window_lims[1]
        )
window_time_vec = time_vec_orig[window_inds]

xcorr_kern_len = len(window_time_vec)
xcorr_kern = np.ones((len(freq_vec), xcorr_kern_len))
xcorr_kern[:, xcorr_kern_len//2:] = 0
xcorr_kern = xcorr_kern / xcorr_kern.sum() 


test = grouped_mean_granger[0][0][0].T[:, window_inds]
test_corr = norm_zero_lag_xcorr(test, xcorr_kern[:,::-1]) 

fig, ax = plt.subplots(1,3, sharey=True)
ax[0].pcolormesh(window_time_vec, freq_vec[1:], xcorr_kern[1:])
ax[1].pcolormesh(window_time_vec, freq_vec[1:], test[1:])
ax[2].plot(test_corr[1:], freq_vec[1:])
ax[0].set_ylabel('Frequency (Hz)')
ax[2].set_xlabel('Scaled XCorr')
fig.savefig(os.path.join(plot_dir, 'scaled_test_xcorr.png'))
plt.close(fig)

# Perform cross-correlation on non-meaned data
scaled_granger_xcorr_per_group = [
        [
            [np.abs(norm_zero_lag_xcorr(
                x.T[:, window_inds], 
                xcorr_kern, 
                )) for x in y] for y in z \
            ] for z in grouped_mean_granger
        ]

scaled_mask_xcorr_per_group = [
        [
            [np.abs(norm_zero_lag_xcorr(
                x.T[:, window_inds], 
                xcorr_kern, 
                )) for x in y] for y in z \
            ] for z in grouped_mask
        ]

# Convert to dataframe
scaled_xcorr_frame_list = []
for i in range(len(transition_inds)):
    for k in range(len(scaled_granger_xcorr_per_group[i])):
        for j in range(len(scaled_granger_xcorr_per_group[i][k])):
            this_granger_xcorr = scaled_granger_xcorr_per_group[i][k][j]
            scaled_xcorr_frame_list.append(
                    pd.DataFrame(
                        dict(
                            transition=transition_inds[i],
                            aligned=aligned_inds[i],
                            granger_xcorr=scaled_granger_xcorr_per_group[i][k][j].flatten(),
                            mask_xcorr=scaled_mask_xcorr_per_group[i][k][j].flatten(),
                            freq = freq_vec,
                            region = j, 
                            )
                        )
                    )

scaled_xcorr_frame = pd.concat(scaled_xcorr_frame_list)
scaled_xcorr_frame.reset_index(drop=True, inplace=True)
scaled_xcorr_frame.dropna(inplace=True)

###############
sns.relplot(
        x='freq',
        y='granger_xcorr',
        data=scaled_xcorr_frame,
        kind='line',
        hue = 'aligned',
        row='transition',
        col='region',
        errorbar = 'sd',
        )
plt.savefig(os.path.join(plot_dir, 'scaled_xcorr_per_region.png'))
plt.close('all')



##############################
##############################
# Extract first 3 PCs
pca_mean_granger = np.stack(
        [[NMF(n_components=3).fit_transform(x.T).T for x in y] for y in mean_granger_per_group]
        )
pca_mask = np.stack(
        [[NMF(n_components=3).fit_transform(x.T).T for x in y] for y in mean_mask_per_group]
        )

############################################################
# Plot


##############################
# Plot heatmaps 

granger_lims = [
        np.min([np.nanmin(x) for x in mean_granger_per_group]),
        np.max([np.nanmax(x) for x in mean_granger_per_group]),
        ]
mask_lim = [
        np.min([np.nanmin(x) for x in mean_mask_per_group]),
        np.max([np.nanmax(x) for x in mean_mask_per_group]),
        ]

fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.pcolormesh(
            time_vec, freq_vec[1:],
            mean_granger_per_group[i][0][1:],
            cmap='viridis',
            vmin=granger_lims[0],
            vmax=granger_lims[1],
            shading='auto',
            )
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('Mean Granger Causality, Dir 0')
fig.savefig(os.path.join(plot_dir, 'mean_granger_causality_dir_0.png'))
plt.close(fig)


fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.pcolormesh(
            time_vec, freq_vec[1:],
            mean_granger_per_group[i][1][1:],
            cmap='viridis',
            vmin=granger_lims[0],
            vmax=granger_lims[1],
            )
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('Mean Granger Causality, Dir 1')
fig.savefig(os.path.join(plot_dir, 'mean_granger_causality_dir_1.png'))
plt.close(fig)


fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.pcolormesh(
            time_vec, freq_vec[1:],
            mean_mask_per_group[i][0][1:], 
            cmap='viridis',
            vmin=mask_lim[0],
            vmax=mask_lim[1],
            )
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('Mean Mask, Dir 0')
fig.savefig(os.path.join(plot_dir, 'mean_mask_dir_0.png'))
plt.close(fig)


fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.pcolormesh(
            time_vec, freq_vec[1:],
            mean_mask_per_group[i][1][1:], 
            cmap='viridis',
            vmin=mask_lim[0],
            vmax=mask_lim[1],
            )
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('Mean Mask, Dir 1')
fig.savefig(os.path.join(plot_dir, 'mean_mask_dir_1.png'))
plt.close(fig)

##############################
# Plot PCA

fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.plot(
            time_vec, pca_mean_granger[i][0].T)
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('PCA Mean Granger Causality, Dir 0')
fig.savefig(os.path.join(plot_dir, 'pca_mean_granger_causality_dir_0.png'))
plt.close(fig)

fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.plot(
            time_vec, pca_mean_granger[i][1].T)
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('PCA Mean Granger Causality, Dir 1')
fig.savefig(os.path.join(plot_dir, 'pca_mean_granger_causality_dir_1.png'))
plt.close(fig)

fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.plot(
            time_vec, pca_mask[i][0].T)
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('PCA Mask, Dir 0')
fig.savefig(os.path.join(plot_dir, 'pca_mask_dir_0.png'))
plt.close(fig)

fig, ax = plt.subplots(
        len(unique_transition_inds),
        len(unique_aligned_inds),
        sharex=True, sharey=True, figsize=(10,10))
for i, (this_trans, this_align_bool) in enumerate(zip(transition_inds, aligned_inds)):
    this_align = aligned_inds_map[this_align_bool]
    this_ax = ax[this_trans, this_align]
    this_ax.plot(
            time_vec, pca_mask[i][1].T)
    this_ax.set_title(
            f'Trans: {this_trans}, Align: {this_align_bool}')
    this_ax.axvline(1, color='r', linestyle='--', linewidth=2)
fig.suptitle('PCA Mask, Dir 1')
fig.savefig(os.path.join(plot_dir, 'pca_mask_dir_1.png'))
plt.close(fig)
