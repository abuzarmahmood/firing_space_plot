"""
Pick sessions which have alpha power in the BLA-->GC direction
during the identity epoch
"""

import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

granger_path = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/discrim_vs_interaction'
plot_dir = os.path.join(granger_path, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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
         'freq_vec',
         'region_names']

loaded_dat_list = []
for this_dir in tqdm(dir_list):
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
    wanted_window_list,
    time_vec_list,
    freq_vec_list,
    region_names_list) = zipped_dat

# Invert mask to get True == Significant
mask_array = np.logical_not(mask_array)

wanted_window = np.array(wanted_window_list[0])/1000
stim_t = 2
corrected_window = wanted_window-stim_t
freq_vec = freq_vec_list[0]
time_vec = time_vec_list[0].copy()
time_vec += corrected_window[0]

#wanted_freq_inds = np.where(np.logical_and(freq_vec >= wanted_freq_range[0],
#                                           freq_vec <= wanted_freq_range[1]))[0]
#freq_vec = freq_vec[wanted_freq_inds]
#granger_actual = granger_actual.mean(axis=1)
#granger_actual = granger_actual[:, :, wanted_freq_inds]
#mask_array = mask_array[:, :, wanted_freq_inds]


# Get summed significant values in this rectangle
# and divide sessions accordingly

mask_array = np.stack([
    mask_array[:, :, :, 0, 1],
    mask_array[:, :, :, 1, 0],
    ])
mask_array = np.moveaxis(mask_array, 0, -1)

# Convert mask_array to an xarray
mask_array = xr.DataArray(mask_array,
                        dims = ['session', 'time', 'freq', 'direction'],
                          coords = dict(
                                  session = np.arange(mask_array.shape[0]),
                                    time = time_vec,
                                    freq = freq_vec,
                                    direction = ['GC-->BLA', 'BLA-->GC']
                                    ),
                          attrs = dict(
                              description = 'Mask array for granger causality',
                              units = 'boolean'
                              )
                          )
# Cut frequencies above 100 Hz
mask_array = mask_array.sel(freq = slice(1, 100))

# Plot to confirm
mean_mask_array = mask_array.mean(dim='session').sel(
    freq = slice(1, 100)
    )
mean_mask_array.plot(x='time', y='freq', col='direction', col_wrap=2)
plt.show()

wanted_freq_range = [1, 15]
wanted_time_range = [0.3,0.8]
wanted_mask_array = mask_array.copy()
wanted_mask_array = wanted_mask_array.sel(
    time = slice(wanted_time_range[0], wanted_time_range[1]),
    freq = slice(wanted_freq_range[0], wanted_freq_range[1])
    )

# Plot to confirm
wanted_mask_array.mean(dim='session').plot(x='time', y='freq', col='direction', col_wrap=2)
plt.show()


sum_mask_array = wanted_mask_array.sum(dim=['time', 'freq'])

fig, ax = plt.subplots(1,2)
for i in range(len(ax)):
    sum_mask_array.isel(direction = i).plot.hist(bins=10, ax =ax[i])
    ax[i].set_title(sum_mask_array.direction.values[i])
plt.show()

# Cluster BLA-->GC direction and pull out session with higher significance
wanted_direction = 'BLA-->GC'
this_dat = sum_mask_array.sel(direction = wanted_direction).values
this_dat = this_dat.reshape(-1,1)
cluster = KMeans(n_clusters=2, random_state=0).fit(this_dat)
cluster_labels = cluster.labels_
cluster_centers = cluster.cluster_centers_
wanted_cluster = np.argmax(cluster_centers)

# Recreate histogram with cluster colors
cmap = plt.cm.get_cmap('tab10')
fig, ax = plt.subplots(1,2)
for i in range(len(ax)):
    for j in range(2):
        this_dat = sum_mask_array.isel(direction = i).values[cluster_labels == j]
        ax[i].hist(this_dat, bins=5, color = cmap(j), alpha = 0.7)
    ax[i].set_title(sum_mask_array.direction.values[i])
plt.show()


wanted_session_inds = np.where(cluster_labels == wanted_cluster)[0]
print(len(wanted_session_inds)/len(cluster_labels))

mean_mask_array = mask_array.sel(
    session = wanted_session_inds,
    freq = slice(1, 100)).mean(dim='session')
mean_mask_array.plot(x='time', y='freq', col='direction', col_wrap=2)
plt.show()

############################################################
# Cluster sessions by spectra of both directions simultaneously
############################################################

mask_frame = mask_array.to_dataframe('sig')
# Drop frequencies above 100 Hz
mask_frame.reset_index(inplace=True)
#mask_frame = mask_frame[mask_frame.freq <= 100]
mask_frame['time_bins']  = pd.cut(mask_frame.time, 10)
mask_frame['freq_bins']  = pd.cut(mask_frame.freq, 10)
bin_mask_frame = mask_frame.groupby(['session', 'time_bins', 'freq_bins', 'direction']).mean()
bin_mask_frame.reset_index(inplace=True)
# Drop bins
bin_mask_frame.drop(['time_bins', 'freq_bins'], axis=1, inplace=True)

# Convert back to xarray
# Convert index to multiindex
bin_mask_frame.set_index(['session', 'time', 'freq', 'direction'], inplace=True)
bin_mask_array = xr.Dataset.from_dataframe(bin_mask_frame).sig

long_bin_array = bin_mask_array.stack(
    {'time_freq_dir': ['time', 'freq', 'direction']}
    )
long_bin_array = long_bin_array.dropna(dim='time_freq_dir', how='any')

# Extract components giving 95% variance
pca_obj = PCA()
pca_obj.fit(long_bin_array.values)
wanted_dims = np.where(np.cumsum(pca_obj.explained_variance_ratio_) > 0.95)[0][0]
pca_bin_dat = pca_obj.transform(long_bin_array.values)[:, :wanted_dims] 

# Cluster
cluster = KMeans(n_clusters=2, random_state=0).fit(pca_bin_dat)
cluster_labels = cluster.labels_

# Plot masks of each cluster
plot_order = np.argsort(cluster_labels)
fig, ax = plt.subplots(len(mask_array), 2, 
                       sharex=True, sharey=True,
                       figsize = (6, len(plot_order)))
for i in range(len(mask_array)):
    for j in range(2):
        this_mask = mask_array.isel(session = plot_order[i]).isel(direction = j)
        this_mask.plot(x='time', y='freq', ax = ax[i, j])
        # basename as title
        ax[i,j].set_title(basename_list[plot_order[i]])
        ax[i,j].set_ylabel(f'{cluster_labels[plot_order[i]]}')
fig.suptitle('Clustered masks\n' + " :: ".join(mask_array.direction.values))
#fig.subplots_adjust(top=0.9)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'clustered_masks.png'),
            bbox_inches='tight')
plt.close(fig)
#plt.show()

for i in range(len(cluster_labels)):
    print(f'{basename_list[plot_order[i]]}: {i}')
