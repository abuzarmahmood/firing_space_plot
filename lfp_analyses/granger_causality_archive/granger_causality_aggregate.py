"""
Aggregate the granger causality results from the individual files
and plot the results.
"""

import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from scipy.stats import zscore, mode, ttest_ind, ttest_1samp
from sklearn.decomposition import PCA

# import sys
# sys.path.append('/media/bigdata/projects/pytau')
# import pytau.changepoint_model as models

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

save_path = '/ancillary_analysis/granger_causality/all'
names = [
         'granger_actual',
         'masked_granger',
         'mask_array',
         'wanted_window',
         'time_vec',
         'freq_vec',
         'freq_summed_pvals',
         'freq_summed_sig',
         'freq_summed_actual_list',
         'freq_summed_shuffle_list',
         ]

loaded_dat_list = []
for this_dir in tqdm(dir_list):
    h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
    with tables.open_file(h5_path, 'r') as h5:
        if save_path not in h5:
            print(f'No {save_path} in {h5_path}')
            continue
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
    freq_vec,
    freq_summed_pvals,
    freq_summed_sig,
    freq_summed_actual_list,
    freq_summed_shuffle_list,
    ) = zipped_dat

# Shape: session, shuffles, time, freq, dir1, dir2
granger_actual_full = granger_actual.copy()
# Shape: session, time, freq, dir1, dir2
mask_array_full = mask_array.copy()
freq_vec_full = freq_vec.copy()[0]

# wanted_window = np.array(wanted_window[0])/1000
# stim_t = 10
# corrected_window = wanted_window-stim_t
freq_vec = freq_vec[0]
time_vec = time_vec[0]
time_vec += -1 # corrected_window[0]

wanted_freq_range = [1, 100]
wanted_freq_inds = np.where(np.logical_and(freq_vec >= wanted_freq_range[0],
                                           freq_vec <= wanted_freq_range[1]))[0]
freq_vec = freq_vec[wanted_freq_inds]
granger_actual = granger_actual.mean(axis=1)
granger_actual = granger_actual[:, :, wanted_freq_inds]
masked_granger = masked_granger[:, :, wanted_freq_inds]
mask_array = mask_array[:, :, wanted_freq_inds]

masked_granger = [np.ma.masked_array(x, mask=y)
                  for x, y in zip(granger_actual, mask_array)]
masked_granger = np.ma.stack(masked_granger)

############################################################
# Plot Data
############################################################
plot_dir_base = '/media/bigdata/firing_space_plot/lfp_analyses/' +\
    'granger_causality/plots'

plot_dir = os.path.join(plot_dir_base, 'all_tastes')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

aggregate_plot_dir = os.path.join(plot_dir_base, 'aggregate_plots')
if not os.path.isdir(aggregate_plot_dir):
    os.makedirs(aggregate_plot_dir)

#######################################
# # Plot of summed granger causality across all frequencies
# # summed_granger = granger_actual_full.sum(axis=3).mean(axis=1)
# summed_granger = np.nansum(granger_actual_full, axis=3)
# summed_granger = np.nanmean(summed_granger, axis=1)
# 
# fig, ax = plt.subplots(2,2,sharex=True, sharey='col')
# ax[0,0].plot(summed_granger[...,0,1].T, alpha = 0.5, color = 'k')
# ax[1,0].plot(summed_granger[...,1,0].T, alpha = 0.5, color = 'k')
# plt.show()

#######################################
# Plot summed granger causality across all frequencies

mean_freq_sum_shuffle = np.nanmean(freq_summed_shuffle_list, axis=2)
std_freq_sum_shuffle = np.nanstd(freq_summed_shuffle_list, axis=2)

std_mult = 3
fig, ax = plt.subplots(
        len(freq_summed_sig), 4, figsize=(15, 15),
        sharex=True, sharey='col')
for num, this_sig in enumerate(freq_summed_sig):
    ax[num, 0].plot(time_vec, freq_summed_actual_list[num, 0], color='black')
    ax[num, 0].plot(time_vec, mean_freq_sum_shuffle[num, 0], color='red')
    ax[num, 0].fill_between(
        time_vec,
        mean_freq_sum_shuffle[num, 0] - std_mult*std_freq_sum_shuffle[num, 0],
        mean_freq_sum_shuffle[num, 0] + std_mult*std_freq_sum_shuffle[num, 0],
        color='red', alpha=0.5)
    ax[num, 1].plot(time_vec, freq_summed_sig[num, 0], color='black')
    ax[num, 2].plot(time_vec, freq_summed_actual_list[num, 1], color='black')
    ax[num, 2].plot(time_vec, mean_freq_sum_shuffle[num, 1], color='red')
    ax[num, 2].fill_between(
        time_vec,
        mean_freq_sum_shuffle[num, 1] - std_mult*std_freq_sum_shuffle[num, 1],
        mean_freq_sum_shuffle[num, 1] + std_mult*std_freq_sum_shuffle[num, 1],
        color='red', alpha=0.5)
    ax[num, 3].plot(time_vec, freq_summed_sig[num, 1], color='black')
    for this_ax in ax[num, :]:
        this_ax.axvline(0, color='red', linestyle='--', linewidth=2)
for this_ax in ax[-1, :]:
    this_ax.set_xlabel('Time post-stimulus (s)')
ax[0, 0].set_title('P-Values GC-->BLA')
ax[0, 1].set_title('Significance GC-->BLA')
ax[0, 2].set_title('P-Values BLA-->GC')
ax[0, 3].set_title('Significance BLA-->GC')
fig.savefig(
        os.path.join(aggregate_plot_dir, 'summed_pvals_sig.png'),
        bbox_inches='tight')
plt.close(fig)


###############
# Plot mean summed granger actual + shuffle, and mean sig
mean_freq_sum_actual = np.nanmean(freq_summed_actual_list, axis=0)
# std_freq_sum_actual = np.nanstd(freq_summed_actual_list, axis=0)
std_freq_sum_actual = np.percentile(freq_summed_actual_list, [2.5, 97.5], axis=0)
mean_freq_sum_shuffle = np.nanmean(freq_summed_shuffle_list, axis=(0,2))
# std_freq_sum_shuffle = np.nanstd(freq_summed_shuffle_list, axis=(0,2))
std_freq_sum_shuffle = np.percentile(freq_summed_shuffle_list, [2.5, 97.5], axis=(0,2))
mean_freq_sum_sig = np.nanmean(freq_summed_sig, axis=0)
# std_freq_sum_sig = np.nanstd(freq_summed_sig, axis=0)
std_freq_sum_sig = np.percentile(freq_summed_sig*1, [2.5, 97.5], axis=0)


time_lims = [-0.5, 1.7]
time_vec_inds = np.where((time_vec >= time_lims[0]) & (time_vec <= time_lims[1]))[0]
time_vec_cut = time_vec[time_vec_inds]
mean_freq_sum_actual = mean_freq_sum_actual[..., time_vec_inds]
std_freq_sum_actual = std_freq_sum_actual[..., time_vec_inds]
mean_freq_sum_shuffle = mean_freq_sum_shuffle[..., time_vec_inds]
std_freq_sum_shuffle = std_freq_sum_shuffle[..., time_vec_inds]
mean_freq_sum_sig = mean_freq_sum_sig[..., time_vec_inds]
std_freq_sum_sig = std_freq_sum_sig[..., time_vec_inds]

# Smooth mean sig
from scipy.signal import savgol_filter
smooth_mean_freq_sum_sig = savgol_filter(mean_freq_sum_sig, 9, 3)

std_mult = 3
fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey='col') 
ax[0, 0].plot(time_vec_cut, mean_freq_sum_actual[0], color='black')
ax[0, 0].plot(time_vec_cut, mean_freq_sum_shuffle[0], color='red')
ax[0, 0].fill_between(
    time_vec_cut,
    # mean_freq_sum_actual[0] - std_mult*std_freq_sum_actual[0],
    # mean_freq_sum_actual[0] + std_mult*std_freq_sum_actual[0],
    std_freq_sum_actual[0][0], std_freq_sum_actual[1][0],
    color='black', alpha=0.5, label = 'Actual')
ax[0, 0].fill_between(
    time_vec_cut,
    # mean_freq_sum_shuffle[0] - std_mult*std_freq_sum_shuffle[0],
    # mean_freq_sum_shuffle[0] + std_mult*std_freq_sum_shuffle[0],
    std_freq_sum_shuffle[0][0], std_freq_sum_shuffle[1][0],
    color='red', alpha=0.5, label = 'Shuffled')
ax[0,0].legend()
ax[0, 1].plot(time_vec_cut, mean_freq_sum_sig[0], color='black', alpha=0.3)
ax[0, 1].plot(time_vec_cut, smooth_mean_freq_sum_sig[0], color='black')
# ax[0, 1].fill_between(
#     time_vec_cut,
#     mean_freq_sum_sig[0] - std_mult*std_freq_sum_sig[0],
#     mean_freq_sum_sig[0] + std_mult*std_freq_sum_sig[0],
#     color='black', alpha=0.5)
ax[1, 0].plot(time_vec_cut, mean_freq_sum_actual[1], color='black')
ax[1, 0].plot(time_vec_cut, mean_freq_sum_shuffle[1], color='red')
ax[1, 0].fill_between(
    time_vec_cut,
    # mean_freq_sum_actual[1] - std_mult*std_freq_sum_actual[1],
    # mean_freq_sum_actual[1] + std_mult*std_freq_sum_actual[1],
    std_freq_sum_actual[0][1], std_freq_sum_actual[1][1],
    color='black', alpha=0.5)
ax[1, 0].fill_between(
    time_vec_cut,
    # mean_freq_sum_shuffle[1] - std_mult*std_freq_sum_shuffle[1],
    # mean_freq_sum_shuffle[1] + std_mult*std_freq_sum_shuffle[1],
    std_freq_sum_shuffle[0][1], std_freq_sum_shuffle[1][1],
    color='red', alpha=0.5)
ax[1, 1].plot(time_vec_cut, mean_freq_sum_sig[1], color='black', alpha=0.3)
ax[1, 1].plot(time_vec_cut, smooth_mean_freq_sum_sig[1], color='black')
# ax[1, 1].fill_between(
#     time_vec_cut,
#     mean_freq_sum_sig[1] - std_mult*std_freq_sum_sig[1],
#     mean_freq_sum_sig[1] + std_mult*std_freq_sum_sig[1],
#     color='black', alpha=0.5)
for this_ax in ax.flatten():
    this_ax.axvline(0, color='red', linestyle='--', linewidth=2)
for this_ax in ax[-1, :]:
    this_ax.set_xlabel('Time post-stimulus (s)')
ax[1, 0].set_title('Mean Summed Granger GC-->BLA')
ax[1, 1].set_title('Mean Summed Sig GC-->BLA')
ax[0, 0].set_title('Mean Summed Granger BLA-->GC')
ax[0, 1].set_title('Mean Summed Sig BLA-->GC')
fig.suptitle('Mean +/- 95% percentile')
fig.savefig(
        os.path.join(aggregate_plot_dir, 'mean_summed_actual_shuffle_sig.png'),
        bbox_inches='tight', dpi = 300)
plt.close(fig)


#######################################

# Create plot with acrtual granger, masked granger and mask for both directions
cmap = plt.cm.viridis
cmap.set_bad(alpha=0.3)
mesh_kwargs = dict(
    # origin='lower',
    # aspect='auto',
    # levels=20,
    shading = 'nearest',
    cmap=cmap)

#for dat_ind in trange(len(basename_list)):
#    #dat_ind = 0
#    this_basename = basename_list[dat_ind]
#    fig, ax = plt.subplots(2, 3, figsize=(15, 7),
#                           sharex=True, sharey=True)
#    fig.suptitle(this_basename)
#    ax[0, 0].pcolormesh(time_vec, freq_vec,
#                        masked_granger[dat_ind][:, :, 0, 1].T.data, **mesh_kwargs)
#    ax[1, 0].pcolormesh(time_vec, freq_vec,
#                        masked_granger[dat_ind][:, :, 1, 0].T.data, **mesh_kwargs)
#    ax[0, 1].pcolormesh(time_vec, freq_vec,
#                        masked_granger[dat_ind][:, :, 0, 1].T, **mesh_kwargs)
#    ax[1, 1].pcolormesh(time_vec, freq_vec,
#                        masked_granger[dat_ind][:, :, 1, 0].T, **mesh_kwargs)
#    ax[0, 2].pcolormesh(time_vec, freq_vec,
#                        masked_granger[dat_ind][:, :, 0, 1].T.mask, **mesh_kwargs)
#    ax[1, 2].pcolormesh(time_vec, freq_vec,
#                        masked_granger[dat_ind][:, :, 1, 0].T.mask, **mesh_kwargs)
#    for this_ax in ax[:, 0]:
#        this_ax.set_ylabel('Freq. (Hz)')
#    for this_ax in ax[-1, :]:
#        this_ax.set_xlabel('Time post-stimulus (s)')
#    for this_ax in ax.flatten():
#        this_ax.axvline(0, color='red', linestyle='--', linewidth=2)
#    fig.savefig(os.path.join(
#        plot_dir, this_basename + '_contour.png'), dpi=300)
#    plt.close(fig)
#    # plt.show()

############################################################
# Aggregate Plots
############################################################
# Create plots of mean granger, mean_masked_granger,
# and summed mask for each direction

mean_granger_actual = np.nanmean(granger_actual, axis=0)
mean_granger_mask = np.nanmean(masked_granger, axis=0)

# Difference in mean_causality at different frequencies
time_inds = np.where((time_vec >= 0) & (time_vec <= 1))[0]
post_stim_granger_actual = granger_actual[:, time_inds]
post_stim_granger_actual = np.stack(
        [
            post_stim_granger_actual[..., 0, 1],
            post_stim_granger_actual[..., 1, 0],
            ], 
        axis=-1)

post_stim_mean_granger_actual = np.nanmean(post_stim_granger_actual, axis=1)
post_stim_granger_diff = post_stim_mean_granger_actual[..., 0] - post_stim_mean_granger_actual[..., 1]
mean_diff = np.nanmean(post_stim_granger_diff, axis=0)

fig = plt.figure(figsize=(5, 5))
plt.plot(freq_vec, post_stim_granger_diff.T,
         color = 'k', alpha = 0.1)
plt.plot(freq_vec, mean_diff, 
         '-x',
         color = 'r', linewidth = 2,
         label = 'Mean Difference')
plt.legend()
plt.axvline(12, color = 'r', linestyle = '--')
plt.text(12, 0.1, '12 Hz', color = 'r', rotation = 90)
plt.axvline(20, color = 'r', linestyle = '--')
plt.text(20, 0.1, '20 Hz', color = 'r', rotation = 90)
plt.axvline(60, color = 'r', linestyle = '--')
plt.text(60, 0.1, '60 Hz', color = 'r', rotation = 90)
plt.ylim(-0.1, 0.1)
plt.axhline(0, color = 'r', linestyle = '--')
plt.ylabel('<- GC to BLA | BLA to GC ->')
plt.suptitle('Granger Causality Difference\n' +\
        'Time-lims: [0, 1] s post-stimulus')
plt.xlabel('Frequency (Hz)')
plt.subplots_adjust(top=0.8)
fig.savefig(
        os.path.join(aggregate_plot_dir, 'granger_diff_per_freq.png'),
        bbox_inches='tight',
        )
plt.close(fig)

# Split by 0-17, and 17-60 Hz
lower_band = [0, 12]
upper_band = [20, 60]
lower_band_inds = np.where((freq_vec >= lower_band[0]) & (freq_vec <= lower_band[1]))[0]
upper_band_inds = np.where((freq_vec >= upper_band[0]) & (freq_vec <= upper_band[1]))[0]
lower_band_diff_mean = np.nanmean(post_stim_granger_diff[:, lower_band_inds], axis=1)
upper_band_diff_mean = np.nanmean(post_stim_granger_diff[:, upper_band_inds], axis=1)

lower_test = ttest_1samp(lower_band_diff_mean, popmean=0)
upper_test = ttest_1samp(upper_band_diff_mean, popmean=0)

fig = plt.figure(figsize=(3, 5))
plt.boxplot(
        [lower_band_diff_mean, upper_band_diff_mean],
        showfliers=False,
        )
for i, this_mean in enumerate([lower_band_diff_mean, upper_band_diff_mean]):
    this_mean = this_mean[this_mean > -0.1]
    plt.scatter(
            np.ones_like(this_mean) + i,
            this_mean,
            color = 'k',
            alpha = 0.3,
            )
plt.axhline(0, color = 'r', linestyle = '--')
plt.xticks([1, 2], [str(lower_band), str(upper_band)]) 
plt.xlabel('Frequency Band (Hz)')
plt.ylabel('<- GC to BLA | BLA to GC ->')
plt.title('Mean Granger Causality Difference\n' +\
        'p-values:\n ' +\
        f'Lower Band: {lower_test.pvalue:.3f}, Upper Band: {upper_test.pvalue:.3f}')
plt.savefig(
        os.path.join(aggregate_plot_dir, 'mean_granger_diff.png'),
        bbox_inches='tight',
        )
plt.close(fig)


##############################
mean_mask = 1 - np.nanmean(mask_array, axis=0)

mean_granger_c_lims = [
                    np.nanmin(mean_granger_actual),
                    np.nanmax(mean_granger_actual)]
mean_mask_c_lims = [
                    np.nanmin(mean_mask),
                    np.nanmax(mean_mask)]


fig, ax = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
fig.suptitle('Mean Granger and Mask' + '\n' + n_string)
ax[0, 0].pcolormesh(time_vec, freq_vec,
                    mean_granger_actual[:, :, 0, 1].T, **mesh_kwargs,
                    vmin=mean_granger_c_lims[0], vmax=mean_granger_c_lims[1])
ax[1, 0].pcolormesh(time_vec, freq_vec,
                    mean_granger_actual[:, :, 1, 0].T, **mesh_kwargs,
                    vmin=mean_granger_c_lims[0], vmax=mean_granger_c_lims[1])
ax[0, 1].pcolormesh(time_vec, freq_vec,
                    zscore(mean_granger_actual[:, :, 0, 1].T, axis=-1),
                    **mesh_kwargs)
ax[1, 1].pcolormesh(time_vec, freq_vec,
                    zscore(mean_granger_actual[:, :, 1, 0].T, axis=-1),
                    **mesh_kwargs)
ax[0, 2].pcolormesh(time_vec, freq_vec,
                    zscore(mean_mask[:, :, 0, 1].T, axis=-1),
                    **mesh_kwargs)
ax[1, 2].pcolormesh(time_vec, freq_vec,
                    zscore(mean_mask[:, :, 1, 0].T, axis=-1),
                    **mesh_kwargs)
im = ax[0, -1].pcolormesh(time_vec, freq_vec,
                          mean_mask[:, :, 0, 1].T, **mesh_kwargs,
                          vmin=mean_mask_c_lims[0], vmax=mean_mask_c_lims[1]
                          )
cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
plt.colorbar(im, cax=cbar_ax)
im = ax[1, -1].pcolormesh(time_vec, freq_vec,
                          mean_mask[:, :, 1, 0].T, **mesh_kwargs,
                          vmin=mean_mask_c_lims[0], vmax=mean_mask_c_lims[1],
                          )
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.35])
plt.colorbar(im, cax=cbar_ax)
for this_ax in ax[:, 0]:
    this_ax.set_ylabel('Freq. (Hz)')
for this_ax in ax[-1, :]:
    this_ax.set_xlabel('Time post-stimulus (s)')
for this_ax in ax.flatten():
    this_ax.axvline(0, color='red', linestyle='--', linewidth=2)
titles = ['Mean Granger', 'Zscore Mean Granger', 'Zscore Mean Mask', 'Mean Mask']
for this_ax, this_title in zip(ax[0, :], titles):
    this_ax.set_title(this_title)
direction_names = ['GC-->BLA','BLA-->GC']
for this_name, this_ax in zip(direction_names, ax[:,0]):
    this_ax.text(-0.3, 0.5, this_name, 
                 transform = this_ax.transAxes,
                 size = 'x-large',
                 va = 'center', rotation = 'vertical')
fig.savefig(os.path.join(aggregate_plot_dir, 'mean_granger_mask.png'), dpi=300)
plt.close(fig)
# plt.show()

##############################
# Plot distribution of mean mask across frequencies

wanted_freq_vec_inds = np.where((freq_vec > 0) & (freq_vec < 70))[0]
wanted_freq_vec = freq_vec[wanted_freq_vec_inds]
wanted_mean_mask = mean_mask[:, wanted_freq_vec_inds]

# only take mean mask for post stimulus time
wanted_mean_mask = wanted_mean_mask[time_vec > 0]

mean_mask_sum = np.sum(wanted_mean_mask, axis=0)
zscore_mean_mask_sum = zscore(mean_mask_sum, axis=0)

fig, ax = plt.subplots(2, 2, figsize=(2.5, 7), sharex = 'row', sharey=True) 
fig.suptitle('Summed Mean Mask' + '\n' + n_string)
ax[0, 0].plot(mean_mask_sum[:, 0, 1], wanted_freq_vec, color='black')
ax[0, 1].plot(mean_mask_sum[:, 1, 0], wanted_freq_vec, color='black')
ax[1, 0].plot(zscore_mean_mask_sum[:, 0, 1], wanted_freq_vec, color='black')
ax[1, 1].plot(zscore_mean_mask_sum[:, 1, 0], wanted_freq_vec, color='black')
for this_ax in ax[:, 0]:
    this_ax.set_ylabel('Freq. (Hz)')
for this_ax in ax[-1, :]:
    this_ax.set_xlabel('Summed Significant Bins')
titles = ['GC-->BLA', 'BLA-->GC']
for this_name, this_ax in zip(direction_names, ax[:, 0]):
    this_ax.set_title(this_name)
# Add ticks for [20, 40, 60] Hz
for this_ax in ax.flatten():
    this_ax.set_yticks([20, 40, 60])
    this_ax.set_yticklabels(['20', '40', '60'])
    this_ax.xaxis.set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(aggregate_plot_dir, 'summed_mean_mask.png'), dpi=300,
            bbox_inches='tight')
plt.close(fig)

# Make one of mean_mask_sum as image
fig, ax = plt.subplots(1, 2, figsize=(5, 7), sharey=True)
fig.suptitle('Summed Mean Mask' + '\n' + n_string)
ax[0].imshow(mean_mask_sum[:, 0, 1][np.newaxis, :].T, aspect='auto', origin='lower',)
ax[1].imshow(mean_mask_sum[:, 1, 0][np.newaxis, :].T, aspect='auto', origin='lower',)
# Correct y axis labels
titles = ['GC-->BLA', 'BLA-->GC'][::-1]
for this_name, this_ax in zip(direction_names, ax):
    this_ax.set_title(this_name)
    this_ax.xaxis.set_visible(False)
    this_ax.yaxis.set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(aggregate_plot_dir, 'summed_mean_mask_image.png'), dpi=300,
            bbox_inches='tight')
plt.close(fig)

wanted_mean_mask_sum = [
        mean_mask_sum[:, 0, 1],
        mean_mask_sum[:, 1, 0],
        ]

plt.plot(wanted_freq_vec, wanted_mean_mask_sum[0], '-x', color='black')
plt.plot(wanted_freq_vec, wanted_mean_mask_sum[1], '-x', color='red')
plt.show()

##############################

# Plot mean mask with inferred changepoints
mean_mask_cp = np.copy(mean_mask)
mean_mask_cp = np.stack([mean_mask_cp[:, :, 0, 1].T,
                         mean_mask_cp[:, :, 1, 0].T], axis=-1)

mean_mask_stack = np.concatenate(mean_mask_cp, axis = -1).T
zscore_mask_stack = zscore(mean_mask_stack, axis=-1)

plt.pcolormesh(time_vec, np.concatenate([freq_vec]*2), 
               zscore_mask_stack, **mesh_kwargs)
plt.show()

# Get components for each mask which give 95% variance explained
pca_dat_list = []
for this_mask in mean_mask_cp.T:
                    pca_obj = PCA().fit(this_mask)
                    transformed_dat = pca_obj.transform(this_mask)
                    wanted_dims = np.where(np.cumsum(pca_obj.explained_variance_ratio_) > 0.9)[0][0]
                    pca_dat_list.append(transformed_dat[:, :wanted_dims])
pca_dat = np.concatenate(pca_dat_list, axis=-1).T

plt.pcolormesh(time_vec, np.arange(len(pca_dat)), 
               zscore(pca_dat,axis=-1), **mesh_kwargs)
plt.show()

# # Fit changepoint model to full dataset
# n_fit = 40000
# n_samples = 20000
# n_states = 5
# model = models.gaussian_changepoint_2d(zscore_mask_stack, n_states)
# model, approx, mu_stack, sigma_stack, tau_samples, fit_data = \
#         models.advi_fit(model = model, fit = n_fit, samples = n_samples)
# 
# mean_mu = np.mean(mu_stack, axis=1)
# 
# # Extract changepoint values
# int_tau = np.vectorize(np.int)(tau_samples)
# mode_tau = np.squeeze(mode(int_tau, axis=0)[0])
# scaled_tau = time_vec[int_tau]
# 
# # Plot changepoints
# fig, ax = plt.subplots(2, 1, figsize=(5, 7), sharex=True, sharey=False)
# ax[0].pcolormesh(time_vec, np.concatenate([freq_vec]*2), 
#                zscore_mask_stack, **mesh_kwargs)
# for this_tau in mode_tau:
#                     ax[0].axvline(time_vec[this_tau], color='red', linestyle='--')
# for this_tau in scaled_tau.T:
#                     ax[1].hist(this_tau, 
#                                bins=np.linspace(time_vec.min(), time_vec.max()), alpha=0.5)
# fig = plt.figure()
# plt.matshow(mean_mu)
# plt.show()
# 
# # Plot changepoint for each direction
# fig, ax = plt.subplots(2, 1, figsize=(5, 7), sharex=True, sharey=True)
# fig.suptitle('Inferred Changepoints' + '\n' + n_string)
# ax[0].pcolormesh(time_vec, freq_vec,
#                  1-mean_mask_cp[:, :, 0], **mesh_kwargs)
# ax[1].pcolormesh(time_vec, freq_vec,
#                  1-mean_mask_cp[:, :, 1], **mesh_kwargs)
# for this_ax in ax:
#                     this_ax.set_ylabel('Freq. (Hz)')
# ax[-1].set_xlabel('Time post-stimulus (s)')
# for this_ax in ax.flatten():
#                     for this_tau in mode_tau:
#                                         this_ax.axvline(time_vec[this_tau], color='red', linestyle='--', linewidth=2)
# titles = ['GC-->BLA','BLA-->GC']
# for this_name, this_ax in zip(direction_names, ax):
#                     this_ax.set_title(this_name)
# plt.show()

############################################################

fig, ax = plt.subplots(2, 1, figsize=(5, 7), sharex=True, sharey=True)
fig.suptitle('Mean Mask with Inferred Changepoints' + '\n' + n_string)
ax[0].pcolormesh(time_vec, freq_vec,
                 1-mean_mask_cp[:, :, 0], **mesh_kwargs)
ax[1].pcolormesh(time_vec, freq_vec,
                 1-mean_mask_cp[:, :, 1], **mesh_kwargs)
for this_ax in ax:
                    this_ax.set_ylabel('Freq. (Hz)')
ax[-1].set_xlabel('Time post-stimulus (s)')
for this_ax in ax.flatten():
                    this_ax.axvline(0, color='red', linestyle='--', linewidth=2)
titles = ['GC-->BLA','BLA-->GC']
for this_name, this_ax in zip(direction_names, ax):
                    this_ax.text(-0.3, 0.5, this_name, 
                                 transform = this_ax.transAxes,
                                 size = 'x-large',
                                 va = 'center', rotation = 'vertical')
#fig.savefig(os.path.join(aggregate_plot_dir, 'mean_mask_cp.png'), dpi=300)
#plt.close(fig)
plt.show()

# For BLA , plot in smaller frequency range
zoom_freq_range = [0, 30]
zoom_freq_inds = np.where((freq_vec > zoom_freq_range[0]) &
                          (freq_vec < zoom_freq_range[1]))[0]

mean_granger_zoom = mean_granger_actual[:, zoom_freq_inds]
mean_mask_zoom = mean_mask[:, zoom_freq_inds]
freq_vec_zoom = freq_vec[zoom_freq_inds]

fig, ax = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
fig.suptitle('Zoomed Mean Granger and Mask' + '\n' + n_string)
ax[0, 0].pcolormesh(time_vec, freq_vec_zoom,
                    mean_granger_zoom[:, :, 0, 1].T, **mesh_kwargs)
ax[1, 0].pcolormesh(time_vec, freq_vec_zoom,
                    mean_granger_zoom[:, :, 1, 0].T, **mesh_kwargs)
ax[0, 1].pcolormesh(time_vec, freq_vec_zoom,
                    zscore(mean_granger_zoom[:, :, 0, 1].T, axis=-1),
                    **mesh_kwargs)
ax[1, 1].pcolormesh(time_vec, freq_vec_zoom,
                    zscore(mean_granger_zoom[:, :, 1, 0].T, axis=-1),
                    **mesh_kwargs)
ax[0, 2].pcolormesh(time_vec, freq_vec_zoom,
                    zscore(1-mean_mask_zoom[:, :, 0, 1].T, axis=-1),
                    **mesh_kwargs)
ax[1, 2].pcolormesh(time_vec, freq_vec_zoom,
                    zscore(1-mean_mask_zoom[:, :, 1, 0].T, axis=-1),
                    **mesh_kwargs)
im = ax[0, -1].pcolormesh(time_vec, freq_vec_zoom,
                          1-mean_mask_zoom[:, :, 0, 1].T, **mesh_kwargs)
cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
plt.colorbar(im, cax=cbar_ax)
im = ax[1, -1].pcolormesh(time_vec, freq_vec_zoom,
                          1-mean_mask_zoom[:, :, 1, 0].T, **mesh_kwargs)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.35])
plt.colorbar(im, cax=cbar_ax)
for this_ax in ax[:, 0]:
    this_ax.set_ylabel('Freq. (Hz)')
for this_ax in ax[-1, :]:
    this_ax.set_xlabel('Time post-stimulus (s)')
for this_ax in ax.flatten():
    this_ax.axvline(0, color='red', linestyle='--', linewidth=2)
titles = ['Mean Granger', 'Zscore Mean Granger', 'Zscore Mean Mask', 'Mean Mask']
for this_ax, this_title in zip(ax[0, :], titles):
    this_ax.set_title(this_title)
direction_names = ['GC-->BLA','BLA-->GC']
for this_name, this_ax in zip(direction_names, ax[:,0]):
    this_ax.text(-0.3, 0.5, this_name, 
                 transform = this_ax.transAxes,
                 size = 'x-large',
                 va = 'center', rotation = 'vertical')
fig.savefig(
        os.path.join(aggregate_plot_dir, 'zoomed_mean_granger_mask.png'),
        dpi=300)
plt.close(fig)
# plt.show()

# Plot mean mask for:
# 1. All frequencies
# 2. Frequencies < 20 Hz
# 3. As a line plot for frequencies < 20 Hz

summed_freq_range = [0,15]
wanted_freq_inds = np.where((freq_vec > summed_freq_range[0]) &
                            (freq_vec < summed_freq_range[1]))[0]
mean_mask_summed = np.sum(1-mean_mask[:, wanted_freq_inds], axis=1)

fig, ax  = plt.subplots(
                    len(granger_actual.T), 3, figsize=(15, 7),
                    sharex=True)
ax[0,0].pcolormesh(time_vec, freq_vec,
                   1-mean_mask[:, :, 0, 1].T, **mesh_kwargs)
ax[1,0].pcolormesh(time_vec, freq_vec,
                   1-mean_mask[:, :, 1, 0].T, **mesh_kwargs)
ax[0,1].pcolormesh(time_vec, freq_vec_zoom,
                   1-mean_mask_zoom[:, :, 0, 1].T, **mesh_kwargs)
ax[1,1].pcolormesh(time_vec, freq_vec_zoom,
                   1-mean_mask_zoom[:, :, 1, 0].T, **mesh_kwargs)
ax[0,2].plot(time_vec, mean_mask_summed[:, 0, 1], color='black')
ax[0,2].axvline(time_vec[np.argmax(mean_mask_summed[:, 0, 1])],
                color = 'k', linestyle = '--', label = 'Max value')
ax[0,2].text(time_vec[np.argmax(mean_mask_summed[:, 0, 1])],
             0.8, f'{time_vec[np.argmax(mean_mask_summed[:, 0, 1])]:.2f} s',
             transform = ax[0,2].transData,)
ax[1,2].plot(time_vec, mean_mask_summed[:, 1, 0], color='red')
ax[1,2].axvline(time_vec[np.argmax(mean_mask_summed[:, 1, 0])],
                color = 'k', linestyle = '--')
ax[1,2].text(time_vec[np.argmax(mean_mask_summed[:, 1,0])],
             1, f'{time_vec[np.argmax(mean_mask_summed[:, 1,0])]:.2f} s',
             transform = ax[1,2].transData,)
for num, this_ax in enumerate(ax[:, 0]):
                    this_ax.set_ylabel(direction_names[num] +\
                                                            '\nFreq. (Hz)')
for this_ax in ax[:,1]:
                    this_ax.set_ylabel('Freq. (Hz)')
for this_ax in ax[:,2]:
                    this_ax.set_ylabel('Summed Significant Bins')
for this_ax in ax[-1, :]:
                    this_ax.set_xlabel('Time post-stimulus (s)')
ax[0,-1].set_title(f'Summed Mask for Freqs = {summed_freq_range} Hz')
for this_ax in ax.flatten():
                    this_ax.axvline(0, color='red', linestyle='--', linewidth=2,
                                    label = 'Stimulus onset')
ax[0,-1].legend()
fig.savefig(
      os.path.join(aggregate_plot_dir, 'mean_mask_3_ways.png'),
      dpi=300)
plt.close(fig)


# Plot summed mask and granger across all frequencies
fig, ax = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
ax[0,0].plot(time_vec, 
             mean_granger_actual[:,:,0,1].sum(axis=1), color='black')
ax[1,0].plot(time_vec,
             mean_granger_actual[:,:,1,0].sum(axis=1), color='black')
ax[0,1].plot(time_vec, 
             mean_mask[:, :, 0, 1].sum(axis=1), color='black')
ax[1,1].plot(time_vec,
             mean_mask[:, :, 1, 0].sum(axis=1), color='black')
for num, this_ax in enumerate(ax[:, 0]):
                    this_ax.set_ylabel(direction_names[num] + '\nFreq. (Hz)')
ax[0,0].set_title('Summed Granger')
ax[0,1].set_title('Summed Mask')
fig.savefig(
    os.path.join(aggregate_plot_dir, 'summed_granger_and_mask.png'),
    dpi=300)
plt.close(fig)
