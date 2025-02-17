"""
Unsupervised inference of movement types which maximally
differentiate between palatabilities
"""
import os
import sys
from glob import glob
from pickle import dump, load

import numpy as np
import tables
import pylab as plt
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from cv2 import pointPolygonTest
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy import stats
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import shap
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from scipy.spatial.distance import mahalanobis
import pingouin as pg

def calc_isotropic_boundary(X, center, n_points = 20):
    """
    Calculate the isotropic boundary of a set of points
    by calculating point with largest distance from centroid
    in a specific direction

    Input:
    X: np.array of shape (n_points, 2)
    center: np.array of shape (2,)
    n_points: number of points
    """

    angles = np.linspace(0, 2*np.pi, n_points)[:-1]
    dir_vecs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    diff_vecs = X - center

    max_point_list = []
    for i in range(n_points-1):
        this_dir = dir_vecs[i]
        dot_prods = np.dot(diff_vecs, this_dir)
        max_ind = np.argmax(dot_prods)
        max_point = X[max_ind]
        if not any([np.all(max_point == x) for x in max_point_list]): 
            max_point_list.append(max_point)
    max_point_list.extend([max_point_list[0]])
    max_point_list = np.stack(max_point_list)

    return max_point_list
############################################################
############################################################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')

all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

wanted_artifacts_paths = glob(os.path.join(artifact_dir, '*wanted_artifacts.pkl'))
wanted_artifacts_paths.sort()
wanted_artifacts = [load(open(path, 'rb')) for path in wanted_artifacts_paths] 
basenames = [x['basename'] for x in wanted_artifacts]
animal_nums = [x.split('_')[0] for x in basenames]

############################################################
# Comparison of JL accuracy vs ours
############################################################
JL_accuracy_list = [x['JL_accuracy'] for x in wanted_artifacts]
our_accuracy_list = [x['one_vs_all_accuracy'] for x in wanted_artifacts]
our_accuracy_mean = np.array([np.mean(x) for x in our_accuracy_list])
our_accuracy_std = np.array([np.std(x) for x in our_accuracy_list])

min_vals = np.min([JL_accuracy_list, our_accuracy_mean - our_accuracy_std])
max_vals = np.max([JL_accuracy_list, our_accuracy_mean + our_accuracy_std])

fig, ax = plt.subplots(1,1)
ax.errorbar(JL_accuracy_list, our_accuracy_mean, yerr=our_accuracy_std, fmt='o')
ax.plot([min_vals, max_vals], [min_vals, max_vals], 'k--')
ax.set_xlabel('JL Accuracy')
ax.set_ylabel('Our Accuracy')
ax.set_title('Comparison of Jenn Li vs Our Accuracy\n(One vs All Classifier)')
ax.set_aspect('equal')
plt.savefig(os.path.join(plot_dir, 'JL_vs_our_accuracy.png'))
plt.close()

############################################################
# Comparison of NM-BSA accuracy vs ours
############################################################

bsa_accuracy_list = [x['bsa_accuracy'] for x in wanted_artifacts]
bsa_f1_list = [x['bsa_f1'] for x in wanted_artifacts]
our_multiclass_accuracy_list = [x['gape_ltp_accuracy'] for x in wanted_artifacts]
our_multiclass_f1_list = [x['gape_ltp_f1'] for x in wanted_artifacts]

our_multiclass_accuracy_mean = np.array([np.mean(x) for x in our_multiclass_accuracy_list])
our_multiclass_accuracy_std = np.array([np.std(x) for x in our_multiclass_accuracy_list])
our_multiclass_f1_mean = np.array([np.mean(x) for x in our_multiclass_f1_list])
our_multiclass_f1_std = np.array([np.std(x) for x in our_multiclass_f1_list])

fig, ax = plt.subplots(1,2)
ax[0].errorbar(
        bsa_accuracy_list, 
        our_multiclass_accuracy_mean, 
        yerr=our_multiclass_accuracy_std, 
        fmt='o')
min_vals = np.min([bsa_accuracy_list, our_multiclass_accuracy_mean - our_multiclass_accuracy_std])
max_vals = np.max([bsa_accuracy_list, our_multiclass_accuracy_mean + our_multiclass_accuracy_std])
ax[0].plot([min_vals, max_vals], [min_vals, max_vals], 'k--')
ax[0].set_xlabel('NM-BSA Accuracy')
ax[0].set_ylabel('Our Accuracy')
ax[0].set_title('Accuracy Comparison\n(Gape + LTP Classifier)')
ax[0].set_aspect('equal')
ax[1].errorbar(
        bsa_f1_list, 
        our_multiclass_f1_mean, 
        yerr=our_multiclass_f1_std, 
        fmt='o')
min_vals = np.min([bsa_f1_list, our_multiclass_f1_mean - our_multiclass_f1_std])
max_vals = np.max([bsa_f1_list, our_multiclass_f1_mean + our_multiclass_f1_std])
ax[1].plot([min_vals, max_vals], [min_vals, max_vals], 'k--')
ax[1].set_xlabel('NM-BSA F1')
ax[1].set_ylabel('Our F1')
ax[1].set_title('F1 Comparison\n(Gape + LTP Classifier)')
ax[1].set_aspect('equal')
fig.suptitle('Single Session\nComparison of NM-BSA vs Our Accuracy\n(Gape + LTP Classifier)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'NM_BSA_vs_our_accuracy_single_session.png'),
            bbox_inches='tight')
plt.close()

############################################################
# Preprocessing 
############################################################
# Get all gape frames
gape_frame_list = all_data_frame.gape_frame_raw.values
fin_gape_frames = []
for i, this_frame in enumerate(gape_frame_list):
    this_frame['session_ind'] = i
    this_frame['animal_num'] = animal_nums[i]
    this_frame['basename'] = all_data_frame.basename.values[i]
    fin_gape_frames.append(this_frame)
cat_gape_frame = pd.concat(fin_gape_frames, axis=0).reset_index(drop=True)

# Match gape frames with palatability
pal_dicts = [{i: x for i, x in enumerate(this_dict.items())} \
                        for this_dict in all_data_frame.taste_map.values]
pal_frames = [pd.DataFrame.from_dict(x, orient='index', columns = ['taste_name','pal']) \
                        for x in pal_dicts]
pal_frames = [x.reset_index(drop=False) for x in pal_frames]
pal_frames = [x.rename(columns={'index':'taste'}) for x in pal_frames]

fin_pal_frames = []
for i, this_frame in enumerate(pal_frames):
    this_frame['session_ind'] = i
    this_frame['animal_num'] = animal_nums[i]
    this_frame['basename'] = all_data_frame.basename.values[i]
    fin_pal_frames.append(this_frame)
cat_pal_frame = pd.concat(fin_pal_frames, axis=0).reset_index(drop=True)

merge_gape_pal = pd.merge(
        cat_gape_frame, 
        cat_pal_frame, 
        on=['session_ind','taste','animal_num','basename'], 
        how='inner')

# Also get baseline values to normalize amplitude
all_envs = np.stack(all_data_frame.env)
all_basenames = all_data_frame.basename.values
animal_nums = [x.split('_')[0] for x in all_basenames]
session_nums = [x.split('_')[1] for x in all_basenames]

baseline_lims = [0, 2000]
baseline_envs = all_envs[:,baseline_lims[0]:baseline_lims[1]]

mean_baseline = np.mean(baseline_envs, axis = -1)
std_baseline = np.std(baseline_envs, axis = -1)
inds = np.array(list(np.ndindex(mean_baseline.shape)))

baseline_frame = pd.DataFrame(
        data = np.concatenate(
            [
                inds, 
                mean_baseline.flatten()[:,None],
                std_baseline.flatten()[:,None]
                ], 
            axis = -1
            ),
        columns = ['session','taste','trial', 'mean','std']
        )
# Convert ['session','taste','trial'] to int
baseline_frame = baseline_frame.astype({'session':'int','taste':'int','trial':'int'})
baseline_frame['animal'] = [animal_nums[i] for i in baseline_frame['session']]
baseline_frame['session_day'] = [session_nums[i] for i in baseline_frame['session']]
baseline_frame['session_name'] = [animal + '\n' + day for \
        animal,day in zip(baseline_frame['animal'],baseline_frame['session_day'])]
mean_baseline_frame = baseline_frame.groupby(
        ['session_name','session_day','animal']).mean()
mean_baseline_frame.reset_index(inplace=True)
mean_baseline_frame = mean_baseline_frame.astype({'session':'int'})
mean_baseline_frame.rename(columns={'mean':'baseline_mean'}, inplace=True)

merge_gape_pal = pd.merge(
        merge_gape_pal, 
        mean_baseline_frame[['session','baseline_mean']], 
        left_on=['session_ind'], 
        right_on=['session'],
        how='left')

merge_gape_pal_backup = merge_gape_pal.copy()

##############################
# Keep only if pal is 1 or 4
merge_gape_pal = merge_gape_pal[merge_gape_pal.pal.isin([1,4])]
pal_vec = merge_gape_pal.pal.values

feature_names = [
    'duration',
    'amplitude_rel',
    'amplitude_abs',
    'left_interval',
    'right_interval',
    'pca_1',
    'pca_2',
    'pca_3',
    'max_freq',
]

all_features = np.stack(merge_gape_pal.features.values)

# Drop amplitude_rel
drop_inds = [i for i, x in enumerate(feature_names) if 'amplitude_rel' in x]
all_features = np.delete(all_features, drop_inds, axis=1)
feature_names = np.delete(feature_names, drop_inds)

# Normalize amplitude by baseline
baseline_vec = merge_gape_pal.baseline_mean.values
amplitude_inds = np.array(
        [i for i, x in enumerate(feature_names) if 'amplitude' in x])
all_features[:,amplitude_inds] = all_features[:,amplitude_inds] / \
        baseline_vec[:,None]

# Drop PCA features
# In future, we can re-calculate PCA features
drop_inds = np.array(
        [i for i, x in enumerate(feature_names) if 'pca' in x])
all_features = np.delete(all_features, drop_inds, axis=1)

# Recalculate PCA features
# First scale all segments by amplitude and length
all_segments = merge_gape_pal.segment_raw.values

# MinMax scale segments
min_max_segments = [x-np.min(x) for x in all_segments]
min_max_segments = [x/np.max(x) for x in min_max_segments]

# Scale all segments to same length
max_len = np.max([len(x) for x in min_max_segments])
scaled_segments = [np.interp(np.linspace(0,1,max_len), np.linspace(0,1,len(x)), x) \
        for x in min_max_segments]
scaled_segments = np.stack(scaled_segments)

# Get PCA features
pca_obj = PCA()
pca_obj.fit(scaled_segments)
pca_features = pca_obj.transform(scaled_segments)[:,:3]

# Add PCA features to all features
all_features = np.concatenate([all_features, pca_features], axis=-1)

# Scale features
scaled_features = StandardScaler().fit_transform(all_features)

# Correct feature_names
pca_feature_names = [feature_names[i] for i in drop_inds]
feature_names = np.delete(feature_names, drop_inds)
feature_names = np.concatenate([feature_names, pca_feature_names])

###############
# Also scale amplitudes of semgents from later plotting
baseline_scaled_segments = [x/y for x,y in \
        tqdm(zip(all_segments, baseline_vec))]


############################################################
# Most separable mouth movements for palatability
############################################################
############################################################
# Unsupervised clustering of UMAP features
############################################################

# Get explained variance for features
pca_obj = PCA()
pca_obj.fit(scaled_features)
explained_variance = pca_obj.explained_variance_ratio_

# plt.plot(np.cumsum(explained_variance), '-o')
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance')
# plt.ylim([0,1])
# plt.show()

# Get UMAP features
umap_obj = UMAP(n_components=2)
umap_obj.fit(scaled_features)
umap_features = umap_obj.transform(scaled_features)


# K-Means cluster umap embedding to pull out notable segments
n_clusters = 30
kmeans_obj = KMeans(n_clusters=n_clusters)
kmeans_obj.fit(umap_features)
centroids = kmeans_obj.cluster_centers_
cluster_labels = kmeans_obj.labels_
cluster_colors = sns.color_palette('hsv', n_clusters)

# Sort clusters using agglomerative clustering on centroids
# from sklearn.cluster import AgglomerativeClustering
# agg_obj = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
# agg_obj.fit(centroids)
# cluster_order = np.argsort(agg_obj.labels_)
# centroids = centroids[cluster_order]
# cluster_labels = np.array([cluster_order[x] for x in cluster_labels])

# # Sort clusters using x-y position of centroids
# centroid_x = centroids[:,0]
# centroid_y = centroids[:,1]
# centroid_order = np.lexsort((centroid_y, centroid_x))
# cluster_labels = np.array([centroid_order[x] for x in cluster_labels])

# plt.scatter(centroid_x, centroid_y, c=range(n_clusters), cmap='tab20')
# for i in range(n_clusters):
#     plt.text(centroid_x[i], centroid_y[i], str(centroid_order[i]))
# plt.show()

# Find datapoints closest to each centroid
cluster_inds = []
for i in range(n_clusters):
    this_centroid = centroids[i]
    dists = np.linalg.norm(umap_features - this_centroid, axis=-1)
    cluster_inds.append(dists.argmin())
cluster_segments = [all_segments[x] for x in cluster_inds]

# Plot UMAP features
fig, ax = plt.subplots(1,1)
for this_cluster in np.unique(cluster_labels):
    inds = np.where(cluster_labels == this_cluster)
    ax.scatter(umap_features[inds,0], umap_features[inds,1], 
               c=cluster_colors[this_cluster], label=this_cluster, alpha=0.05)
    # ax.legend()
    ax.text(centroids[this_cluster,0], centroids[this_cluster,1], str(this_cluster))
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
fig.savefig(os.path.join(plot_dir, 'umap_features_clustered.png'),
            bbox_inches='tight')
plt.close()

# For each cluster, plot the segment and color by cluster
fig, ax = plt.subplots(len(cluster_segments),1, sharex=True, sharey=True,
        figsize=(5, len(cluster_segments)))
for i, this_segment in enumerate(cluster_segments):
    ax[i].plot(this_segment, color=cluster_colors[i],
               linewidth=4)
    ax[i].set_ylabel(str(i))
plt.xlabel('Time (ms)')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'umap_cluster_segments.png'),
            bbox_inches='tight')
plt.close()

# For each cluster, plot the segment and color by cluster
fig, ax = plt.subplots(len(cluster_segments),1, sharex=True, sharey=False,
        figsize=(5, len(cluster_segments)))
for i, this_segment in enumerate(cluster_segments):
    ax[i].plot(this_segment, color=cluster_colors[i],
               linewidth=4)
    ax[i].set_ylabel(str(i))
plt.xlabel('Time (ms)')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'umap_cluster_segments_unshared.png'),
            bbox_inches='tight')
plt.close()


# Plot palatable and unpalatable separately
fig, ax = plt.subplots(1,2, sharex=True, sharey=True,
                       figsize=(10,5))
pal_colors = ['r','b']
for i, this_pal in enumerate(np.unique(pal_vec)):
    inds = np.where(pal_vec == this_pal)
    ax[i].scatter(umap_features[inds,0], umap_features[inds,1], 
               c=pal_colors[i], label=this_pal, alpha=0.05)
    ax[i].legend()
    ax[i].set_xlabel('UMAP 1')
    ax[i].set_ylabel('UMAP 2')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'umap_pal_unpal.png'),
            bbox_inches='tight')
plt.close()

# Fit gaussian mixture models to both palatable and unpalatable
# Infer number of components using AIC

# n_components = np.arange(1, 50)
# models = [GaussianMixture(n, covariance_type='full', random_state=0).\
#         fit(umap_features) for n in tqdm(n_components)]
# aics = [model.aic(umap_features) for model in models]
# plt.plot(n_components, aics, '-o')
# plt.xlabel('Number of Components')
# plt.ylabel('AIC')
# plt.show()

# n_components = 50
# 
# model_list = []
# for i, this_pal in enumerate(np.unique(pal_vec)):
#     inds = np.where(pal_vec == this_pal)
#     this_data = umap_features[inds]
#     model = GaussianMixture(
#             n_components, covariance_type='full', random_state=0).\
#                     fit(this_data)
#     model_list.append(model)
# model_dict = dict(zip(np.unique(pal_vec), model_list))

# Evaluate both models over a meshgrid
x_lims = [np.min(umap_features[:,0]), np.max(umap_features[:,0])]
y_lims = [np.min(umap_features[:,1]), np.max(umap_features[:,1])]

n_bins = 25
x_vals = np.linspace(x_lims[0]*1.1, x_lims[1]*1.1, n_bins)
y_vals = np.linspace(y_lims[0]*1.1, y_lims[1]*1.1, n_bins)

x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)

xy_mesh = np.stack([x_mesh.flatten(), y_mesh.flatten()], axis=-1)

# Generate 2d histograms for both palatable and unpalatable
hist_list = []
for i, this_pal in enumerate(np.unique(pal_vec)):
    inds = np.where(pal_vec == this_pal)
    this_data = umap_features[inds]
    hist, x_edges, y_edges = np.histogram2d( 
            this_data[:,0], this_data[:,1], 
                    bins= [x_vals, y_vals],
                    density=True)
    hist_list.append(hist)

hist_x_mesh, hist_y_mesh = np.meshgrid(x_edges[:-1], y_edges[:-1])
hist_x_mesh = hist_x_mesh.T
hist_y_mesh = hist_y_mesh.T
hist_mesh_flat = np.stack([hist_x_mesh.flatten(), hist_y_mesh.flatten()], axis=-1)

# ll_eval_list = []
# for i, this_pal in enumerate(np.unique(pal_vec)):
#     model = model_dict[this_pal]
#     z = -model.score_samples(xy_mesh)
#     z = z.reshape(x_mesh.shape)
#     ll_eval_list.append(z)

vmin = np.min([
    np.min(hist_list[0]),
    np.min(hist_list[1])
    ])
vmax = np.max([
    np.max(hist_list[0]),
    np.max(hist_list[1])
    ])

# ll_diff = ll_eval_list[0] - ll_eval_list[1]
hist_diff = hist_list[0] / hist_list[1]
cmap = plt.cm.jet
norm = mpl.colors.CenteredNorm(0, 0.5)
fig, ax = plt.subplots(2,3, sharex=True, sharey=True,
                       figsize=(20,10))
pal_colors = ['r','b']
for i, this_pal in enumerate(np.unique(pal_vec)):
    inds = np.where(pal_vec == this_pal)
    ax[0, i].pcolormesh(hist_x_mesh, hist_y_mesh, hist_list[i],
                    cmap='jet', vmin=vmin, vmax=vmax)
    ax[0, i].set_xlabel('UMAP 1')
    ax[0, i].set_ylabel('UMAP 2')
    ax[1, i].scatter(umap_features[inds,0], umap_features[inds,1],
                    c=pal_colors[i], label=this_pal, alpha=0.1)
    ax[0, i].set_title('Palatability: ' + str(this_pal))
img = ax[0, 2].pcolormesh(hist_x_mesh, hist_y_mesh, np.log10(hist_diff),
               cmap=cmap, norm=norm)
plt.colorbar(img, ax=ax[0, 2],
             label = '<- Palatable | Unpalatable ->')
ax[0, 2].set_xlabel('UMAP 1')
ax[0, 2].set_ylabel('UMAP 2')
ax[0, 2].set_title('Difference')
# plt.show()
fig.suptitle('UMAP Features\nPalatability Comparison')
fig.savefig(os.path.join(plot_dir, 'umap_palatability_comparison.png'),
            bbox_inches='tight')
plt.close()

###############
# Take difference hist and segment by kmeans clusters

# Interpolaste hist_mesh to get subpixel resolution
# n_subpixels = 10
# subpixel_x_vals = np.linspace(x_lims[0]*1.1, x_lims[1]*1.1, n_bins*n_subpixels)
# subpixel_y_vals = np.linspace(y_lims[0]*1.1, y_lims[1]*1.1, n_bins*n_subpixels)
# subpixel_x_mesh, subpixel_y_mesh = np.meshgrid(subpixel_x_vals, subpixel_y_vals)
# flat_subpixel_mesh = np.stack([subpixel_x_mesh.flatten(), subpixel_y_mesh.flatten()], axis=-1)
# flat_subpixel_mesh = flat_subpixel_mesh.astype('float32')
# 
# subpixel_clusters = kmeans_obj.predict(flat_subpixel_mesh)
# 
# plt.scatter(flat_subpixel_mesh[:,0], flat_subpixel_mesh[:,1], c=subpixel_clusters, alpha=0.1,
#             cmap='tab20')
# plt.show()

# Calculate cluster boundaries
cluster_boundaries = []
for i in range(n_clusters):
    this_cluster = umap_features[np.where(cluster_labels == i)]
    this_center = centroids[i]
    this_boundary = calc_isotropic_boundary(this_cluster, this_center)
    cluster_boundaries.append(this_boundary)

# For each cluster, get average diff value
# Fit Gaussian process to get smooth representation of hist diff
gp = GaussianProcessRegressor()
log_hist_diff = np.log10(hist_diff)
y = log_hist_diff.flatten()
X = hist_mesh_flat

# Replace nans with 0 
nan_inds = np.where(np.isnan(y))
y[nan_inds] = 0

# Replae infs with 0 
inf_inds = np.where(np.isinf(y))
y[inf_inds] = 0

gp.fit(X, y)

# Predict over meshgrid
y_pred, sigma = gp.predict(hist_mesh_flat, return_std=True)
y_pred = y_pred.reshape(hist_x_mesh.shape)

# fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
# img = ax[0].pcolormesh(hist_x_mesh, hist_y_mesh, np.log10(hist_diff),
#                 cmap='jet')
# plt.colorbar(img, ax=ax[0])
# img = ax[1].pcolormesh(hist_x_mesh, hist_y_mesh, y_pred,
#                 cmap='jet')
# plt.colorbar(img, ax=ax[1])
# plt.show()

###############
# Take n points from each cluster and evaluate histdiff (via GP)
# Then take average
n_points = 100
cluster_diffs_list = []
cluster_points_list = []
for i in range(n_clusters):
    cluster_inds = np.where(cluster_labels == i)
    cluster_inds = np.random.choice(cluster_inds[0], n_points)
    cluster_points = umap_features[cluster_inds]
    cluster_diff = gp.predict(cluster_points)
    cluster_diffs_list.append(cluster_diff)
    cluster_points_list.append(cluster_points)
mean_cluster_diffs = np.array([np.mean(x) for x in cluster_diffs_list])

# plot scatter
cmap = plt.cm.jet
norm = mpl.colors.CenteredNorm(0, 1)
fig, ax = plt.subplots(1,2, sharex=True, sharey=True,
                       figsize=(15,5))
img = ax[1].pcolormesh(hist_x_mesh, hist_y_mesh, np.log10(hist_diff),
                cmap=cmap, norm=norm)
plt.colorbar(img, ax=ax[1])
for i in range(n_clusters):
    ax[0].scatter(cluster_points_list[i][:,0], cluster_points_list[i][:,1],
               c=cmap(norm(cluster_diffs_list[i])), alpha=0.5) 
    ax[0].text(centroids[i,0], centroids[i,1], str(i))
    ax[1].plot(cluster_boundaries[i][:,0], cluster_boundaries[i][:,1],
            color = 'k')
    ax[0].plot(cluster_boundaries[i][:,0], cluster_boundaries[i][:,1],
            color = 'k')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'umap_cluster_mean_inference.png'),)
plt.close()


# Plot UMAP features
fig, ax = plt.subplots(1,3, sharex=True, sharey=True,
                       figsize=(20,5))
img = ax[1].pcolormesh(hist_x_mesh, hist_y_mesh, np.log10(hist_diff),
                cmap=cmap, norm=norm)
plt.colorbar(img, ax=ax[1])
cmap = plt.cm.jet
norm = mpl.colors.CenteredNorm(0, 0.5)
for this_cluster in np.unique(cluster_labels):
    inds = np.where(cluster_labels == this_cluster)
    ax[0].scatter(umap_features[inds,0], umap_features[inds,1], 
               c=cluster_colors[this_cluster], label=this_cluster,
                  s = 2)
    ax[0].plot(cluster_boundaries[this_cluster][:,0], cluster_boundaries[this_cluster][:,1],
            color = 'k')
    ax[1].plot(cluster_boundaries[this_cluster][:,0], cluster_boundaries[this_cluster][:,1],
            color = 'k')
    ax[0].text(centroids[this_cluster,0], centroids[this_cluster,1], str(this_cluster),
               fontweight='bold')
    ax[2].text(centroids[this_cluster,0], centroids[this_cluster,1], str(this_cluster),
               fontweight='bold')
    # Create filled polygons with mean cluster diff
    this_diff = mean_cluster_diffs[this_cluster]
    ax[2].fill(cluster_boundaries[this_cluster][:,0], cluster_boundaries[this_cluster][:,1],
               c = cmap(norm(this_diff)))
    ax[2].plot(cluster_boundaries[this_cluster][:,0], cluster_boundaries[this_cluster][:,1],
            color = 'k')
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[2])
fig.savefig(os.path.join(plot_dir, 'umap_features_clustered_boundaries.png'),)
plt.close()

# Sort clusters by mean diff, and plot segments
cluster_order = np.argsort(mean_cluster_diffs)

fig, ax = plt.subplots(len(cluster_segments),1, sharex=True, sharey=True,
        figsize=(5, len(cluster_segments)))
for i, ind in enumerate(cluster_order):
    this_segment = cluster_segments[ind]
    this_diff = mean_cluster_diffs[ind]
    ax[i].plot(this_segment, c=cmap(norm(this_diff)),
               linewidth=4, label = str(np.round(this_diff,3)))
    ax[i].legend()
    ax[i].set_ylabel(str(ind))
plt.xlabel('Time (ms)')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'umap_cluster_segments_sorted.png'),
            bbox_inches='tight')
plt.close()

# Check cluster constituency by session
cluster_label_session_frame = pd.DataFrame(
        dict(
            labels = cluster_labels,
            session = merge_gape_pal.session_ind.values
            )
        )

# Create stacked bar plots for each cluster
# Normalize height of all plots
for i in range(n_clusters):
    this_cluster = cluster_label_session_frame[cluster_label_session_frame.labels == i]
    this_counts = this_cluster.session.value_counts()
    this_counts = this_counts / np.sum(this_counts)
    if i == 0:
        stacked_frame = this_counts
    else:
        stacked_frame = pd.concat([stacked_frame, this_counts], axis=1)


stacked_frame.fillna(0, inplace=True)
stacked_frame = stacked_frame.T
stacked_frame.reset_index(inplace=True, drop=True)

# Sort columns
stacked_frame = stacked_frame[cluster_label_session_frame.session.unique()]

# Any relationship between mean diff and cluster constituency?
cluster_constituency_var = stacked_frame.var(axis=1)
abs_mean_cluster_diffs = np.abs(mean_cluster_diffs)
rho, p = stats.spearmanr(abs_mean_cluster_diffs, cluster_constituency_var)
rho, p = np.round(rho, 3), np.round(p, 3)

plt.scatter(abs_mean_cluster_diffs, cluster_constituency_var)
plt.xlabel('Abs Mean Cluster Diff')
plt.ylabel('Cluster Constituency Variance')
plt.title('Cluster Constituency Variance vs Mean Diff\n' + \
        f'Spearman Rho: {rho}, p: {p}')
plt.savefig(os.path.join(plot_dir, 'cluster_constituency_vs_mean_diff.png'),)
plt.close()

fig, ax = plt.subplots(1,2, sharex=True, sharey=True,
                       figsize=(10,5))
stacked_frame.plot(kind='bar', stacked=True, legend=False,
                   width = 1, ax=ax[0])
# Sort by variance
stacked_frame_sorted = stacked_frame.iloc[cluster_constituency_var.argsort().values]
stacked_frame_sorted.plot(kind='bar', stacked=True, legend=False,
                   width = 1, ax=ax[1])
plt.xlabel('Cluster')
plt.ylabel('Fraction of Sessions')
plt.title('Cluster Constituency by Session')
plt.xticklabels = np.arange(n_clusters)
# plt.show()
plt.savefig(os.path.join(plot_dir, 'cluster_constituency.png'),)
plt.close()

sns.clustermap(stacked_frame_sorted.T, row_cluster=False, col_cluster=True,
               cmap='viridis', figsize=(10,5))
plt.savefig(os.path.join(plot_dir, 'cluster_constituency_clustermap.png'),
            bbox_inches='tight')
plt.close()
