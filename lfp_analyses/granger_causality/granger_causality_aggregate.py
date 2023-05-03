"""
Aggregate the granger causality results from the individual files
and plot the results.
"""

import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import zscore

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
mean_mask = np.nanmean(mask_array, axis=0)

fig, ax = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
fig.suptitle('Mean Granger and Mask' + '\n' + n_string)
ax[0, 0].pcolormesh(time_vec, freq_vec,
                    mean_granger_actual[:, :, 0, 1].T, **mesh_kwargs)
ax[1, 0].pcolormesh(time_vec, freq_vec,
                    mean_granger_actual[:, :, 1, 0].T, **mesh_kwargs)
ax[0, 1].pcolormesh(time_vec, freq_vec,
                    zscore(mean_granger_actual[:, :, 0, 1].T, axis=-1),
                    **mesh_kwargs)
ax[1, 1].pcolormesh(time_vec, freq_vec,
                    zscore(mean_granger_actual[:, :, 1, 0].T, axis=-1),
                    **mesh_kwargs)
ax[0, 2].pcolormesh(time_vec, freq_vec,
                    zscore(1-mean_mask[:, :, 0, 1].T, axis=-1),
                    **mesh_kwargs)
ax[1, 2].pcolormesh(time_vec, freq_vec,
                    zscore(1-mean_mask[:, :, 1, 0].T, axis=-1),
                    **mesh_kwargs)
im = ax[0, -1].pcolormesh(time_vec, freq_vec,
                          1-mean_mask[:, :, 0, 1].T, **mesh_kwargs)
cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
plt.colorbar(im, cax=cbar_ax)
im = ax[1, -1].pcolormesh(time_vec, freq_vec,
                          1-mean_mask[:, :, 1, 0].T, **mesh_kwargs)
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
