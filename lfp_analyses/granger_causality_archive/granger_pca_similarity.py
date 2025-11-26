"""
Calculate cosine similarity between PCA of granger causality
and granger significance masks for each direction
"""

import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tqdm import trange
from scipy import stats
from sklearn.decomposition import PCA
from itertools import combinations, product

from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import seaborn as sns

plot_dir_base = '/media/bigdata/firing_space_plot/lfp_analyses/' +\
    'granger_causality/plots/aggregate_plots'

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

# Only take t > 0
time_inds = np.where(time_vec > 0)[0]
time_vec = time_vec[time_inds]
mean_mask = mean_mask[..., time_inds]

# Index granger_actual by direction, then time
granger_actual = np.stack(
        [granger_actual[...,0,1],
         granger_actual[...,1,0],], 
        axis=-1).T
granger_actual = granger_actual[..., time_inds,:]
granger_actual = np.moveaxis(granger_actual, -1, 1)

# Index mask_array by direction, then time
mask_array = np.stack(
        [mask_array[...,0,1],
         mask_array[...,1,0],],
        axis=-1).T
mask_array = mask_array[..., time_inds,:]
mask_array = np.moveaxis(mask_array, -1, 1)

############################################################
# PCA on mask

pca_objs = [PCA().fit(this_mask.T) for this_mask in mean_mask]
cumsum_var = [np.cumsum(this_pca.explained_variance_ratio_) \
        for this_pca in pca_objs]
transformed_data = [this_pca.transform(this_mask.T).T \
        for this_pca, this_mask in zip(pca_objs, mean_mask)]

# Summed variance of first 3 components
summed_var = [np.sum(this_pca.explained_variance_ratio_[:3]) \
        for this_pca in pca_objs]
summed_var = [np.round(this_var, 3) for this_var in summed_var]

# Plot explained variance ratio for each pca
fig, ax = plt.subplots(1,1)
for i, this_dat in enumerate(cumsum_var):
    ax.plot(this_dat, '-x', label = dir_names[i])
ax.set_xlabel('PC #')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Explained Variance Ratio for PCA on Granger Masks\n' + \
        str(list(zip(dir_names, summed_var))))
ax.legend()
plt.show()

# Plot transformed data
fig, ax = plt.subplots(2,1)
for i, this_dat in enumerate(transformed_data):
    ax[i].plot(this_dat[:3].T)
    ax[i].set_title(dir_names[i])
    ax[i].set_xlabel('PC #')
    ax[i].set_ylabel('PC Value')
plt.suptitle('Transformed Data')
plt.tight_layout()
plt.show()

# Plot pca loadings
fig, ax = plt.subplots(2,1)
for i, this_pca in enumerate(pca_objs):
    im = ax[i].matshow(this_pca.components_[:3], cmap = 'bwr')
    ax[i].set_title(dir_names[i])
    ax[i].set_xlabel('Frequency')
    ax[i].set_ylabel('PC Value')
    ax[i].set_xticks(np.arange(len(freq_vec)))
    ax[i].set_xticklabels(np.round(freq_vec, 2), rotation = 90)
    plt.colorbar(im, ax = ax[i])
plt.suptitle('PCA Components')
plt.tight_layout()
plt.show()

# Calculate dot product of first 3 components for each direction
stacked_comp = np.concatenate([this_pca.components_[:3] for this_pca in pca_objs])
dot_prod = np.dot(stacked_comp, stacked_comp.T)

# Dot product for random vectors of same size
vec_len = stacked_comp.shape[1]
n_vecs = 1000
rand_vecs = np.random.randn(n_vecs, vec_len)
# Convert to unit vectors
rand_vecs = rand_vecs/np.linalg.norm(rand_vecs, axis=1, keepdims=True)
rand_dot_prod = np.dot(rand_vecs, rand_vecs.T)
# Get values for lower triangle
inds = np.tril_indices(rand_dot_prod.shape[0], k=-1)
rand_dot_prod = rand_dot_prod[inds]
# Get abs
rand_dot_prod = np.abs(rand_dot_prod)
# Get 95th percentile corrected
alpha = 0.05
n_comparisons = 9 
corrected_alpha = alpha/n_comparisons
corrected_p = 1-corrected_alpha
rand_dot_prod_thresh = np.percentile(rand_dot_prod, corrected_p*100)

# Plot cosine similarity and boolean mask for dot product > 95th percentile
fig, ax = plt.subplots(2,1)
im = ax[0].matshow(dot_prod, cmap = 'bwr')
ax[0].set_title('Cosine Similarity')
plt.colorbar(im, ax = ax[0])
im = ax[1].matshow(dot_prod > rand_dot_prod_thresh, cmap = 'bwr')
ax[1].set_title('Dot Product > 95th Percentile')
plt.colorbar(im, ax = ax[1])
plt.suptitle('Dot Product of First 3 Components')
plt.tight_layout()
plt.show()

# Representation per frequency in first 3 components
comp_array = np.stack([this_pca.components_[:3] for this_pca in pca_objs])
summed_rep = np.sum(np.abs(comp_array), axis=1)

# Filter summed representation
savgol_rep = [savgol_filter(this_rep, 5, 3) for this_rep in summed_rep]

# plot raw and savgol filtered summed representation
fig, ax = plt.subplots(2,1, sharex=True)
for i, this_rep in enumerate(summed_rep):
    ax[0].plot(freq_vec, this_rep, label = dir_names[i])
    ax[1].plot(freq_vec, savgol_rep[i], label = dir_names[i])
ax[0].set_title('Raw Summed Representation')
ax[1].set_title('Savgol Filtered Summed Representation')
ax[0].legend()
ax[1].legend()
ax[1].set_xlabel('Frequency')
ax[0].set_ylabel('Summed Representation')
ax[1].set_ylabel('Summed Representation')
plt.tight_layout()
plt.show()

############################################################
# Repeat PCA dot product analysis on raw granger data
# Compare intra vs inter region using single sessions
############################################################

# Perform PCA on both directions and all sessions to 
# get first 3 components
granger_pca_comps = np.zeros((*mask_array.shape[:2], 3, len(freq_vec)))
inds = list(np.ndindex(mask_array.shape[:2]))
for i, this_ind in enumerate(inds):
    this_granger = mask_array[this_ind]
    this_pca = PCA().fit(this_granger.T)
    granger_pca_comps[this_ind] = this_pca.components_[:3]

# Calculate intra region dot product
# Get all combinations of sessions
n_sessions = granger_pca_comps.shape[1]
session_combs = list(combinations(range(n_sessions), 2))

# Calculate dot product for each session combination
intra_dot_prod = [[np.dot(
                    granger_pca_comps[i, this_comb[0]], 
                    granger_pca_comps[i, this_comb[1]].T) \
            for this_comb in session_combs]\
    for i in range(granger_pca_comps.shape[0])]
intra_dot_prod = np.array(intra_dot_prod)
mean_intra_dot_prod = np.mean(intra_dot_prod, axis=1)

# Repeat for inter region
# Get all combinations of regions
session_combs = list(product(range(n_sessions), range(n_sessions)))
inter_dot_prod = [np.dot(
                    granger_pca_comps[0, this_comb[0]],
                    granger_pca_comps[1, this_comb[1]].T) \
            for this_comb in session_combs]
inter_dot_prod = np.array(inter_dot_prod)
mean_inter_dot_prod = np.mean(inter_dot_prod, axis=0)
