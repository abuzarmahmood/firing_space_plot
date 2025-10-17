"""
Analysis of mouth movement classification

1) Comparison with previous methods
    a) Jenn Li (gapes)
    b) Narendra (gapes + LTPs)

"""

import os
import sys
from glob import glob
from pickle import dump, load
import json

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

############################################################
# Cross-validated classification accuracy 
############################################################
# Test classifier by leaving out:
# 1) Random session
# 1) Session within animal
# 2) Session across animals
# 3) Whole animals
merge_gape_pal = merge_gape_pal_backup.copy()

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

np.save(os.path.join(artifact_dir, 'merge_gape_pal_feature_names.npy'),
        feature_names)

# Categorize animal_num
animal_codes = merge_gape_pal.animal_num.astype('category').cat.codes.values

fig,ax = plt.subplots(1,4, sharey=True)
ax[0].imshow(scaled_features, aspect='auto', interpolation='nearest')
ax[0].set_xticks(np.arange(len(feature_names)))
ax[0].set_xticklabels(feature_names, rotation=45, ha='right')
ax[0].set_xlabel('Feature')
ax[0].set_ylabel('Segment')
ax[1].imshow(merge_gape_pal.session_ind.values[:,None], 
             aspect='auto', interpolation='nearest',
             cmap = 'tab20')
ax[2].imshow(animal_codes[:,None],
             aspect='auto', interpolation='nearest',
             cmap = 'tab20')
scored_data = merge_gape_pal.scored.fillna(False).values
ax[3].imshow(scored_data[:,None],
             aspect='auto', interpolation='nearest',
             )
ax[0].set_title('Features')
ax[1].set_title('Session')
ax[2].set_title('Animal')
ax[3].set_title(f'Scored, n={np.sum(scored_data)}')
fig.savefig(os.path.join(plot_dir, 'scaled_features_for_crossval.png'),
            bbox_inches='tight')
plt.close()

##############################
merge_gape_pal['raw_features'] = list(all_features)
merge_gape_pal['features'] = list(scaled_features)

###############
# Also scale amplitudes of semgents from later plotting
baseline_scaled_segments = [x/y for x,y in \
        tqdm(zip(all_segments, baseline_vec))]

merge_gape_pal['baseline_scaled_segments'] = baseline_scaled_segments

merge_gape_pal.to_pickle(os.path.join(artifact_dir, 'merge_gape_pal.pkl'))

scored_df = merge_gape_pal[merge_gape_pal.scored == True]

# Correct event_types
types_to_drop = ['to discuss', 'other', 'unknown mouth movement','out of view']
scored_df = scored_df[~scored_df.event_type.isin(types_to_drop)]

# Remap event_types
event_type_map = {
        'mouth movements' : 'mouth or tongue movement',
        'tongue protrusion' : 'mouth or tongue movement',
        'mouth or tongue movement' : 'mouth or tongue movement',
        'lateral tongue movement' : 'lateral tongue protrusion',
        'lateral tongue protrusion' : 'lateral tongue protrusion',
        'gape' : 'gape',
        'no movement' : 'no movement',
        'nothing' : 'no movement',
        }

scored_df['event_type'] = scored_df['event_type'].map(event_type_map)
scored_df['event_codes'] = scored_df['event_type'].astype('category').cat.codes
scored_df['is_gape'] = (scored_df['event_type'] == 'gape')*1

scored_df.dropna(subset=['event_type'], inplace=True)

##############################
# Output copy for MTM clustering analysis
out_copy = scored_df.copy()
out_copy.drop(columns = 'session', inplace=True)
out_copy.reset_index(drop=True, inplace=True)

# Output frame to pickle and output json with description of each column
out_copy.to_pickle(os.path.join(artifact_dir, 'mtm_clustering_df.pkl'))
out_cols = out_copy.columns

out_dict = {
        'taste': 'Taste of the trial',
        'trial': 'Trial number given taste',
        'features': 'Features of the segment (with normalized amplitude)',
        'segment_raw': 'Raw segment',
        'segment_bounds': 'Start and end of segment',
        'segment_num': 'Segment number within trial',
        'classifier': 'Classification of segment using Jenn Li algorithm (1 = gape)',
        'segment_center': 'Center of segment within trial',
        'scored': 'Whether segment was scored',
        'event_type': 'Type of event according to scoring',
        'event_codes': 'Event type as a code (from scoring)',
        'session_ind': 'Session number',
        'animal_num': 'Animal number',
        'basename': 'Basename of the session',
        'taste_name': 'Name of the taste',
        'pal': 'Palatability of the taste',
        'baseline_mean': 'Mean of baseline amplitude (for normalization of amplitudes)',
        'raw_features': 'Features of the segment (without amplitude normalization)',
        'baseline_scaled_segments': 'Segments with amplitude normalized by baseline',
        'is_gape': 'Whether event is a gape (1 = gape)',
        }

json.dump(
        out_dict, open(os.path.join(artifact_dir, 'mtm_clustering_df_description.json'), 
                       'w'),
        indent=4)

# Write out feature names as txt
with open(os.path.join(artifact_dir, 'mtm_clustering_feature_names.txt'), 'w') as f:
    for item in feature_names:
        f.write("%s\n" % item)



##############################

print(scored_df.event_type.value_counts())

event_type_counts = scored_df.event_type.value_counts().to_dict()

def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d})"

color_map = {
        'gape' : '#EF8636',
        'mouth or tongue movement' : '#3B75AF',
        'lateral tongue protrusion' : '#AFC7E8',
        'no movement' : '#D1D1D1'
        }

fig, ax = plt.subplots(figsize = (4,4))
ax.pie(event_type_counts.values(), labels=event_type_counts.keys(),
       autopct=lambda pct: func(pct, np.array(list(event_type_counts.values()))),
       explode = [0.1]*len(event_type_counts),
       colors = [color_map[x] for x in event_type_counts.keys()], 
       )
ax.set_title('Event Type Distribution\n' + f'n={len(scored_df)}')
fig.savefig(os.path.join(plot_dir, 'event_type_distribution_bsa.png'),
            bbox_inches='tight', dpi = 300)
plt.close()

# Make pie-chart for how many 'no-movements' are video labelled
# vs emg-labelled
# If segment_center < 0, them emg_labelled

no_movement_scored_df = scored_df[scored_df.event_type == 'no movement']
no_movement_scored_df['emg_labelled'] = (no_movement_scored_df.segment_center < 0)

emg_labelled_counts = no_movement_scored_df.emg_labelled.value_counts().to_dict()
fig, ax = plt.subplots()
ax.pie(emg_labelled_counts.values(), labels=emg_labelled_counts.keys(),
       autopct=lambda pct: func(pct, np.array(list(emg_labelled_counts.values()))),
       explode = [0.1]*len(emg_labelled_counts),
       )
ax.set_title('No Movement Labelled by EMG\n' + f'n={len(no_movement_scored_df)}')
fig.savefig(os.path.join(plot_dir, 'event_no_movement_labelled_by_emg.png'),
            bbox_inches='tight')
plt.close()

##############################
# Overlayed line-plots for each type

n_types = len(event_type_counts)
fig, ax = plt.subplots(len(event_type_counts),1, sharex=True, sharey=True,
                       figsize=(5, 5*len(event_type_counts)))
for i, this_type in enumerate(event_type_counts.keys()):
    this_df = scored_df[scored_df.event_type == this_type]
    # this_segments = this_df.segment_raw.values
    this_segments = this_df.baseline_scaled_segments.values
    for this_segment in this_segments:
        # ax[i].plot(this_segment, color='k', alpha=0.1)
        ax[i].plot(this_segment, color= color_map[this_type], 
                   # alpha= 0.1)
                   alpha=50/len(this_segments))
    ax[i].set_title(f'{this_type}, n={len(this_segments)}')

    # Also plot mean
    this_features = np.stack(this_df.features.values)
    mean_feature_vec = this_features.mean(axis=0)
    rep_mean_ind = np.argmin(np.linalg.norm(this_features - mean_feature_vec, axis=-1))
    rep_mean_segment = this_segments[rep_mean_ind]

    ax[i].plot(rep_mean_segment, 
               # color = color_map[this_type],
               color = 'k', 
               linewidth = 5,
               linestyle = '--')

plt.xlabel('Time (ms)')
fig.suptitle('Raw Segments by Event Type')
fig.savefig(os.path.join(plot_dir, 'raw_segments_by_event_type.png'),
            bbox_inches='tight')
plt.close()

##############################
# Feature clustering by event type 
features = np.stack(scored_df.features.values)
event_codes = scored_df.event_codes.values
sort_order = np.argsort(event_codes)
sorted_features = features[sort_order]
sorted_event_codes = event_codes[sort_order]

sorted_features_df = pd.DataFrame(sorted_features, columns=feature_names)

event_code_map = scored_df[['event_codes', 'event_type']].drop_duplicates() 

cmap = plt.cm.tab10
row_colors = [cmap(x) for x in sorted_event_codes] 
g = sns.clustermap(sorted_features_df, row_cluster=False, col_cluster=False,
               row_colors = row_colors, cmap='viridis')
legend_elements = [mpl.lines.Line2D([0], [0], color=cmap(i), label=event_type,
                                    linewidth = 5) \
        for i, event_type in zip(event_code_map.event_codes, event_code_map.event_type)] 
g.ax_heatmap.legend(handles=legend_elements, title='Event Type',
                    bbox_to_anchor=(1.04,1), loc='upper left')
plt.suptitle('Feature Clustering by Event Type')
plt.savefig(os.path.join(plot_dir, 'feature_clustering_by_event_type.png'),
            bbox_inches='tight')
plt.close()

##############################
# UMAP / NCA plot to visualize distinction of clusters by event type
nca_obj = NeighborhoodComponentsAnalysis(n_components=2)
nca_features = nca_obj.fit_transform(features, event_codes)

plt.figure()
for i in range(n_types):
    inds = np.where(event_codes == i)
    event_name = event_code_map[event_code_map.event_codes == i].event_type.values[0]
    plt.scatter(nca_features[inds,0], nca_features[inds,1], label=event_name, alpha=0.5)
plt.legend()
plt.xlabel('NCA 1')
plt.ylabel('NCA 2')
plt.title('NCA Features by Event Type')
plt.savefig(os.path.join(plot_dir, 'nca_features_by_event_type.png'),
            bbox_inches='tight')
plt.close()

umap_obj = UMAP(n_components=2)
umap_features = umap_obj.fit_transform(features)

plt.figure()
for i in range(n_types):
    inds = np.where(event_codes == i)
    event_name = event_code_map[event_code_map.event_codes == i].event_type.values[0]
    plt.scatter(umap_features[inds,0], umap_features[inds,1], label=event_name, alpha=0.5)
plt.legend()
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP Features by Event Type')
plt.savefig(os.path.join(plot_dir, 'umap_features_by_event_type.png'),
            bbox_inches='tight')
plt.close()

# # 3D NCA
# nca_obj = NeighborhoodComponentsAnalysis(n_components=3)
# nca_features = nca_obj.fit_transform(features, event_codes)
# 
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(n_types):
#     inds = np.where(event_codes == i)
#     event_name = event_code_map[event_code_map.event_codes == i].event_type.values[0]
#     ax.scatter(nca_features[inds,0], nca_features[inds,1], nca_features[inds,2], label=event_name, alpha=0.5)
# ax.legend()
# ax.set_xlabel('NCA 1')
# ax.set_ylabel('NCA 2')
# ax.set_zlabel('NCA 3')
# plt.title('NCA Features by Event Type')
# plt.show()

##############################
# Fit a GMM to 3D NCA of 'no movement' data
# and assign all points within 3 mahalanobis distances
# as no movement

# Get no movement data
no_movement_code = event_code_map[event_code_map.event_type == 'no movement'].event_codes.values[0]
no_movement_inds = np.where(event_codes == no_movement_code)
no_movement_3d_nca = nca_features[no_movement_inds] 

# Fit gaussian
mean_vec = np.mean(no_movement_3d_nca, axis=0)
cov_mat = np.cov(no_movement_3d_nca, rowvar=False)

# # Calculate contours for standard deviations
# def calc_std_contour(mean_vec, cov_mat, n_std):
#     eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#     eig_vals = np.diag(eig_vals)
#     std_mat = np.dot(eig_vecs, np.sqrt(eig_vals))
#     std_mat = std_mat * n_std
#     return std_mat
# 
# std_vec = np.array([1,2,3])
# std_contours = [calc_std_contour(mean_vec, cov_mat, x) for x in std_vec]

# Calculate mahalanobis distance
mahal_dist = np.array([mahalanobis(x, mean_vec, np.linalg.inv(cov_mat)) for x in nca_features])
mahal_thresh = 2

updated_codes = event_codes.copy()
updated_codes[mahal_dist < mahal_thresh] = no_movement_code

fig, ax = plt.subplots(1,2, sharex=True, sharey=True, 
                       figsize=(10,5))
for i in range(n_types):
    inds = np.where(event_codes == i)
    event_name = event_code_map[event_code_map.event_codes == i].event_type.values[0]
    ax[0].scatter(nca_features[inds,0], nca_features[inds,1], label=event_name, alpha=0.1)
ax[0].legend()
ax[0].set_xlabel('NCA 1')
ax[0].set_ylabel('NCA 2')
ax[0].set_title('NCA Features by Event Type')
# Overlay scatter with GMM
for i in range(n_types):
    inds = np.where(updated_codes == i)
    event_name = event_code_map[event_code_map.event_codes == i].event_type.values[0]
    ax[1].scatter(nca_features[inds,0], nca_features[inds,1], label=event_name, alpha=0.1)
ax[1].legend()
ax[1].set_xlabel('NCA 1')
ax[1].set_ylabel('NCA 2')
ax[1].set_title('NCA Features by Event Type')
plt.savefig(os.path.join(plot_dir, 'no_movement_reassignment.png'),
            bbox_inches='tight')
plt.close()

# Update codes in scored_df
scored_df['updated_codes'] = updated_codes
scored_df['updated_event_type'] = scored_df.updated_codes.map(
        event_code_map.set_index('event_codes').event_type)
scored_df.to_pickle(os.path.join(artifact_dir, 'scored_df.pkl'))

# Plot raw segments by updated codes
fig, ax = plt.subplots(len(event_type_counts),1, sharex=True, sharey=True,
                       figsize=(5, 5*len(event_type_counts)))
for i, this_type in enumerate(event_type_counts.keys()):
    this_code = event_code_map[event_code_map.event_type == this_type].event_codes.values[0]
    this_df = scored_df[scored_df.updated_codes == this_code]
    # this_segments = this_df.segment_raw.values
    this_segments = this_df.baseline_scaled_segments.values
    for this_segment in this_segments:
        # ax[i].plot(this_segment, color='k', alpha=0.1)
        ax[i].plot(this_segment, color= color_map[this_type], 
                   # alpha= 0.1)
                   alpha=45/len(this_segments))
    ax[i].set_title(f'{this_type}, n={len(this_segments)}')

    # Also plot mean
    this_features = np.stack(this_df.features.values)
    mean_feature_vec = this_features.mean(axis=0)
    rep_mean_ind = np.argmin(np.linalg.norm(this_features - mean_feature_vec, axis=-1))
    rep_mean_segment = this_segments[rep_mean_ind]

    ax[i].plot(rep_mean_segment, 
               # color = color_map[this_type],
               color = 'k', 
               linewidth = 5,
               linestyle = '--')

plt.xlabel('Time (ms)')
fig.suptitle('Raw Segments by Event Type')
fig.savefig(os.path.join(plot_dir, 'raw_segments_by_updated_events_black_mean.png'),
            bbox_inches='tight')
plt.close()

# Plot raw segments by updated codes
fig, ax = plt.subplots(len(event_type_counts),1, sharex=True, sharey=True,
                       figsize=(5, 5*len(event_type_counts)))
for i, this_type in enumerate(event_type_counts.keys()):
    this_code = event_code_map[event_code_map.event_type == this_type].event_codes.values[0]
    this_df = scored_df[scored_df.updated_codes == this_code]
    # this_segments = this_df.segment_raw.values
    this_segments = this_df.baseline_scaled_segments.values
    for this_segment in this_segments:
        # ax[i].plot(this_segment, color='k', alpha=0.1)
        ax[i].plot(this_segment, color= color_map[this_type], 
                   # alpha= 0.1)
                   alpha=45/len(this_segments))
    ax[i].set_title(f'{this_type}, n={len(this_segments)}')

    # Also plot mean
    this_features = np.stack(this_df.features.values)
    mean_feature_vec = this_features.mean(axis=0)
    rep_mean_ind = np.argmin(np.linalg.norm(this_features - mean_feature_vec, axis=-1))
    rep_mean_segment = this_segments[rep_mean_ind]

    ax[i].plot(rep_mean_segment, 
               color = color_map[this_type],
               linewidth = 5,
               linestyle = '--')

plt.xlabel('Time (ms)')
fig.suptitle('Raw Segments by Event Type')
fig.savefig(os.path.join(plot_dir, 'raw_segments_by_updated_events_color_mean.png'),
            bbox_inches='tight')
plt.close()


############################################################
# JL comparison
############################################################

jl_comparison_plot_dir = os.path.join(plot_dir, 'jl_comparison')
os.makedirs(jl_comparison_plot_dir, exist_ok=True)

##############################
# Leave one session out
##############################

# Get all unique sessions
unique_sessions = scored_df.session_ind.unique()

test_y_list = []
pred_y_list = []
jl_pred_y_list = []
mean_abs_shap_list = []
for i, this_session in enumerate(tqdm(unique_sessions)):
    # Leave out this session
    train_df = scored_df[scored_df.session_ind != this_session]
    test_df = scored_df[scored_df.session_ind == this_session]

    # Train classifier
    X_train = np.stack(train_df.features.values)
    y_train = train_df.is_gape.values
    X_test = np.stack(test_df.features.values)
    y_test = test_df.is_gape.values

    jl_pred_y = test_df.classifier.values
    jl_pred_y_list.append(jl_pred_y)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_y_list.append(y_test)
    pred_y_list.append(y_pred)

    # Get shap values
    # explainer = shap.Explainer(clf)
    # shap_values = explainer(X_train)
    # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # mean_abs_shap_list.append(mean_abs_shap)
    mean_abs_shap_list.append(clf.feature_importances_)


# Get accuracy and f1
jl_accuracy = [accuracy_score(y_test, jl_pred_y) for y_test, jl_pred_y in zip(test_y_list, jl_pred_y_list)]
accuracy = [accuracy_score(y_test, pred_y) for y_test, pred_y in zip(test_y_list, pred_y_list)]
jl_f1 = [f1_score(y_test, jl_pred_y) for y_test, jl_pred_y in zip(test_y_list, jl_pred_y_list)]
f1 = [f1_score(y_test, pred_y) for y_test, pred_y in zip(test_y_list, pred_y_list)]

fig, ax = plt.subplots(1,2)
ax[0].scatter(jl_accuracy, accuracy)
ax[0].plot([0,1],[0,1],'k--')
ax[0].set_xlabel('JL Accuracy')
ax[0].set_ylabel('Our Accuracy')
ax[0].set_title('Accuracy Comparison')
ax[0].set_aspect('equal')
ax[1].scatter(jl_f1, f1)
ax[1].plot([0,1],[0,1],'k--')
ax[1].set_xlabel('JL F1')
ax[1].set_ylabel('Our F1')
ax[1].set_title('F1 Comparison')
ax[1].set_aspect('equal')
fig.suptitle('Leave One Session Out\nComparison of JL vs Our Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(jl_comparison_plot_dir, 
                         'leave_one_session_out_comparison.png'),
            bbox_inches='tight')
plt.close()

# Plot shap values
shap_df = pd.DataFrame(mean_abs_shap_list, columns=feature_names)
# Melt
shap_df = shap_df.melt(var_name='Feature', value_name='Mean Abs SHAP')
# Sort features by mean abs shap
shap_df.sort_values('Mean Abs SHAP', ascending=False, inplace=True)


sns.boxplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
            fill = False, showfliers=False)
sns.stripplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
              color='k', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Abs SHAP')
plt.title('Mean Abs SHAP Comparison')
plt.tight_layout()
plt.savefig(os.path.join(jl_comparison_plot_dir, 'mean_abs_shap_session_comparison.png'),
            bbox_inches='tight')
plt.close()

##############################
# Leave on animal out 
##############################

# Get all unique sessions
scored_df['animal_code'] = scored_df.animal_num.astype('category').cat.codes
unique_animals = scored_df.animal_code.unique()

test_y_list = []
pred_y_list = []
jl_pred_y_list = []
mean_abs_shap_list = []
for i, this_session in enumerate(tqdm(unique_animals)):
    # Leave out this session
    train_df = scored_df[scored_df.animal_code != this_session]
    test_df = scored_df[scored_df.animal_code == this_session]

    # Train classifier
    X_train = np.stack(train_df.features.values)
    y_train = train_df.is_gape.values
    X_test = np.stack(test_df.features.values)
    y_test = test_df.is_gape.values

    jl_pred_y = test_df.classifier.values
    jl_pred_y_list.append(jl_pred_y)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_y_list.append(y_test)
    pred_y_list.append(y_pred)

    # Get shap values
    # explainer = shap.Explainer(clf)
    # shap_values = explainer(X_train)
    # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # mean_abs_shap_list.append(mean_abs_shap)
    mean_abs_shap_list.append(clf.feature_importances_)


# Get accuracy and f1
jl_accuracy = [accuracy_score(y_test, jl_pred_y) for y_test, jl_pred_y in zip(test_y_list, jl_pred_y_list)]
accuracy = [accuracy_score(y_test, pred_y) for y_test, pred_y in zip(test_y_list, pred_y_list)]
jl_f1 = [f1_score(y_test, jl_pred_y) for y_test, jl_pred_y in zip(test_y_list, jl_pred_y_list)]
f1 = [f1_score(y_test, pred_y) for y_test, pred_y in zip(test_y_list, pred_y_list)]

fig, ax = plt.subplots(1,2)
ax[0].scatter(jl_accuracy, accuracy)
ax[0].plot([0,1],[0,1],'k--')
ax[0].set_xlabel('JL Accuracy')
ax[0].set_ylabel('Our Accuracy')
ax[0].set_title('Accuracy Comparison')
ax[0].set_aspect('equal')
ax[1].scatter(jl_f1, f1)
ax[1].plot([0,1],[0,1],'k--')
ax[1].set_xlabel('JL F1')
ax[1].set_ylabel('Our F1')
ax[1].set_title('F1 Comparison')
ax[1].set_aspect('equal')
fig.suptitle('Leave One Session Out\nComparison of JL vs Our Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(jl_comparison_plot_dir, 
                         'leave_one_animal_out_comparison.png'),
            bbox_inches='tight')
plt.close()

# Plot shap values
shap_df = pd.DataFrame(mean_abs_shap_list, columns=feature_names)
# Melt
shap_df = shap_df.melt(var_name='Feature', value_name='Mean Abs SHAP')
# Sort features by mean abs shap
shap_df.sort_values('Mean Abs SHAP', ascending=False, inplace=True)


sns.boxplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
            fill = False, showfliers=False)
sns.stripplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
              color='k', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Abs SHAP')
plt.title('Mean Abs SHAP Comparison')
plt.tight_layout()
plt.savefig(os.path.join(jl_comparison_plot_dir, 'mean_abs_shap_animal_comparison.png'),
            bbox_inches='tight')
plt.close()

############################################################
# NM-BSA Comparison 
############################################################
bsa_comparison_plot_dir = os.path.join(plot_dir, 'bsa_comparison')
os.makedirs(bsa_comparison_plot_dir, exist_ok=True)

# Get BSA for given events
# gape = 6:11
# LTP = 11:
wanted_bsa_p_list = []
for i, this_event_row in scored_df.iterrows():
    taste = this_event_row['taste']
    trial = this_event_row['trial']
    time_lims = np.array(this_event_row['segment_bounds'])+2000
    all_data_ind = all_data_frame.loc[all_data_frame.basename == this_event_row['basename']].index[0]
    bsa_p_array = all_data_frame.loc[all_data_ind,'bsa_p']
    bsa_dat = bsa_p_array[taste, trial, time_lims[0]:time_lims[1]]
    bsa_mode = stats.mode(bsa_dat, axis = 0).mode
    wanted_bsa_p_list.append(bsa_mode)
wanted_bsa_p_list = np.array(wanted_bsa_p_list).astype('int')

# Convert bsa_p to predictions
def bsa_to_pred(x):
    if np.logical_and(x>=6, x<11):
        return 0
    elif x>=11:
        return 1
    else:
        return 2

bsa_aligned_event_map = {
        'gape' : 0,
        'mouth or tongue movement' : 1,
        'lateral tongue protrusion' : 2,
        'no movement' : 2,
        }

bsa_labels = ['gape', 'mouth or tongue movement', 'LTP/Nothing']

scored_df['bsa_aligned_event_codes'] = scored_df['event_type'].map(bsa_aligned_event_map)

# Get metrics for BSA
wanted_bsa_pred_list = np.array([bsa_to_pred(x) for x in wanted_bsa_p_list]).astype('int')
scored_df['bsa_pred'] = wanted_bsa_pred_list

#######################################
# plot frequency ranges for all movements
max_freq_ind = np.where(feature_names == 'max_freq')[0][0]
freq_vec = scored_df.raw_features.apply(lambda x: x[max_freq_ind])
event_vec = scored_df.event_type.values

bsa_omega = all_data_frame.bsa_omega.values[0]
omega_cutoff_inds = [6,11]
omega_cutoff_freqs = bsa_omega[omega_cutoff_inds]

bsa_inferred_freq = bsa_omega[wanted_bsa_p_list]

freq_df = pd.DataFrame(
        dict(
            max_freq = freq_vec,
            bsa_freq = bsa_inferred_freq,
            event = event_vec
            )
        )

fig, ax = plt.subplots(freq_df.event.nunique(), 2, sharex=True,
                       figsize = (10, 3*freq_df.event.nunique()))
for i, this_event in enumerate(freq_df.event.unique()):
    this_freqs = freq_df[freq_df.event == this_event].max_freq
    ax[i,0].hist(this_freqs, bins = 25, alpha=0.7)
    ax[i,0].set_ylabel(this_event)
    for this_cutoff in omega_cutoff_freqs:
        ax[i,0].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
    this_bsa_freqs = freq_df[freq_df.event == this_event].bsa_freq
    ax[i,1].hist(this_bsa_freqs, bins = 25, alpha=0.7)
    ax[i,1].set_ylabel(this_event)
    for this_cutoff in omega_cutoff_freqs:
        ax[i,1].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
ax[0,0].set_title('Max Freq')
ax[0,1].set_title('BSA Freq')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count')
plt.suptitle('Frequency Distribution of Mouth Movements')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'event_frequency_ranges.png'),
            bbox_inches='tight')
plt.close()

# Direct overlays of gape and 'mouth or tongue movement'
freq_regions = [[0, 4.15], [4.15, 6.4], [6.4, 20]]
freq_region_labels = ['Null', 'Gape', 'LTP']
fig,ax = plt.subplots(2,1, sharex=True, sharey=True)
for event_type in ['gape', 'mouth or tongue movement']:
    dat_inds = np.where(scored_df.event_type == event_type)
    max_freq = freq_vec.values[dat_inds]
    bsa_freq = bsa_inferred_freq[dat_inds]
    bsa_freq += np.random.normal(0, 0.1, len(bsa_freq))
    ax[0].hist(max_freq, bins = 25, alpha=0.7, label = event_type,
               density=True)
    ax[1].hist(bsa_freq, bins = 25, alpha=0.7, label = event_type,
               density=True)
ax[0].legend()
for this_cutoff in omega_cutoff_freqs:
    ax[0].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
    ax[1].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
ax1_y_lims = ax[1].get_ylim()
for label, region in zip(freq_region_labels, freq_regions):
    ax[1].text(np.mean(region), ax1_y_lims[1]*0.75, label, fontweight='bold',
               ha='center')
ax[0].set_title('Max Freq')
ax[1].set_title('BSA Freq')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count')
plt.suptitle('Frequency Distribution of Gape and Mouth Movements')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'gape_mouth_freq_comparison.png'),
            bbox_inches='tight')
plt.close()


# Correlate max freq with bsa freq
rho, p = stats.spearmanr(freq_df.max_freq, freq_df.bsa_freq)
rho, p = np.round(rho, 3), np.round(p, 3)

fig, ax = plt.subplots()
ax.scatter(freq_df.max_freq, freq_df.bsa_freq,
            alpha=0.1)
ax.plot([0,15],[0,15],'k--')
ax.set_xlabel('Max Freq')
ax.set_ylabel('BSA Freq')
ax.set_title('Max Freq vs BSA Freq\n' + \
        f'Spearman Rho: {rho}, p: {p}')
ax.set_aspect('equal')
plt.savefig(os.path.join(plot_dir, 'max_freq_vs_bsa_freq.png'),)
plt.close()

##############################
# Leave one session out
##############################
test_y_list = []
pred_y_list = []
bsa_pred_y_list = []
mean_abs_shap_list = []
for i, this_session in enumerate(tqdm(unique_sessions)):
    # Leave out this session
    train_df = scored_df[scored_df.session_ind != this_session]
    test_df = scored_df[scored_df.session_ind == this_session]

    # Train classifier
    X_train = np.stack(train_df.features.values)
    y_train = train_df.bsa_aligned_event_codes.values
    X_test = np.stack(test_df.features.values)
    y_test = test_df.bsa_aligned_event_codes.values

    bsa_pred_y = test_df.bsa_pred.values
    bsa_pred_y_list.append(bsa_pred_y)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_y_list.append(y_test)
    pred_y_list.append(y_pred)

    # Get shap values
    # explainer = shap.Explainer(clf)
    # shap_values = explainer(X_train)
    # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # mean_abs_shap_list.append(mean_abs_shap)
    mean_abs_shap_list.append(clf.feature_importances_)


# Get accuracy and f1
bsa_accuracy = [accuracy_score(y_test, bsa_pred_y) \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
accuracy = [accuracy_score(y_test, pred_y) for y_test, pred_y in zip(test_y_list, pred_y_list)]
bsa_f1 = [f1_score(y_test, bsa_pred_y, average='macro') \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
f1 = [f1_score(y_test, pred_y, average='macro') \
        for y_test, pred_y in zip(test_y_list, pred_y_list)]
bsa_confusion = [confusion_matrix(y_test, bsa_pred_y, normalize='true') \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
confusion = [confusion_matrix(y_test, pred_y, normalize='true') \
        for y_test, pred_y in zip(test_y_list, pred_y_list)]

fig, ax = plt.subplots(1,2)
ax[0].scatter(bsa_accuracy, accuracy)
ax[0].plot([0,1],[0,1],'k--')
ax[0].set_xlabel('BSA Accuracy')
ax[0].set_ylabel('Our Accuracy')
ax[0].set_title('Accuracy Comparison')
ax[0].set_aspect('equal')
ax[1].scatter(bsa_f1, f1)
ax[1].plot([0,1],[0,1],'k--')
ax[1].set_xlabel('BSA F1')
ax[1].set_ylabel('Our F1')
ax[1].set_title('F1 Comparison')
ax[1].set_aspect('equal')
fig.suptitle('Leave One Session Out\nComparison of BSA vs Our Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(bsa_comparison_plot_dir, 
                         'leave_one_session_out_comparison.png'),
            bbox_inches='tight')
plt.close()


# Plot mean+std confusion matrices 
mean_bsa_confusion = np.mean(bsa_confusion, axis=0)
std_bsa_confusion = np.std(bsa_confusion, axis=0)
mean_confusion = np.mean(confusion, axis=0)
std_confusion = np.std(confusion, axis=0)

confusion_diff = [x-y for x,y in zip(confusion, bsa_confusion)]
# norm_confusion_diff = [x/np.sum(np.abs(x)) for x in confusion_diff]
mean_confusion_diff = np.mean(confusion_diff, axis=0)
std_confusion_diff = np.std(confusion_diff, axis=0)


fig, ax = plt.subplots(1,3, figsize=(20,5))
img = ax[0].matshow(mean_bsa_confusion, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(img, ax=ax[0])
for i in range(mean_bsa_confusion.shape[0]):
    for j in range(mean_bsa_confusion.shape[1]):
        ax[0].text(j, i, f'{mean_bsa_confusion[i,j]:.2f}\n±{std_bsa_confusion[i,j]:.2f}',
                 ha='center', va='center', color='k', fontweight='bold')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')
ax[0].set_xticks(np.arange(3))
ax[0].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[0].set_yticks(np.arange(3))
ax[0].set_yticklabels(bsa_labels)
ax[0].set_title('Mean+Std BSA Confusion Matrix')
img = ax[1].matshow(mean_confusion, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(img, ax=ax[1])
for i in range(mean_confusion.shape[0]):
    for j in range(mean_confusion.shape[1]):
        ax[1].text(j, i, f'{mean_confusion[i,j]:.2f}\n±{std_confusion[i,j]:.2f}',
                 ha='center', va='center', color='k', fontweight='bold')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')
ax[1].set_xticks(np.arange(3))
ax[1].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[1].set_yticks(np.arange(3))
ax[1].set_yticklabels(bsa_labels)
ax[1].set_title('Mean+Std Our Confusion Matrix')

cmap = plt.cm.jet
norm = mpl.colors.CenteredNorm(0, 0.5)
im = ax[2].matshow(mean_confusion_diff, cmap=cmap, norm=norm)
plt.colorbar(label = '<- BSA Predicted More | Our Predicted More ->', ax=ax[2], mappable=im)
for i in range(mean_confusion_diff.shape[0]):
    for j in range(mean_confusion_diff.shape[1]):
        ax[2].text(j, i, f'{mean_confusion_diff[i,j]:.2f}\n±{std_confusion_diff[i,j]:.2f}',
                 ha='center', va='center', color='k', fontweight='bold')
ax[2].set_xlabel('Predicted')
ax[2].set_ylabel('True')
ax[2].set_xticks(np.arange(3))
ax[2].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[2].set_yticks(np.arange(3))
ax[2].set_yticklabels(bsa_labels)
ax[2].set_title('Mean+Std Confusion Difference')
plt.tight_layout()
plt.savefig(os.path.join(bsa_comparison_plot_dir, 'mean_std_confusion_diff_session.png'),
            bbox_inches='tight')
plt.close()


# Plot shap values
shap_df = pd.DataFrame(mean_abs_shap_list, columns=feature_names)
# Melt
shap_df = shap_df.melt(var_name='Feature', value_name='Mean Abs SHAP')
# Sort features by mean abs shap
shap_df.sort_values('Mean Abs SHAP', ascending=False, inplace=True)


sns.boxplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
            fill = False, showfliers=False)
sns.stripplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
              color='k', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Abs SHAP')
plt.title('Mean Abs SHAP Comparison')
plt.tight_layout()
plt.savefig(os.path.join(bsa_comparison_plot_dir, 'mean_abs_shap_session_comparison.png'),
            bbox_inches='tight')
plt.close()

##############################
# Leave one animal out
##############################
test_y_list = []
pred_y_list = []
bsa_pred_y_list = []
mean_abs_shap_list = []
for i, this_session in enumerate(tqdm(unique_animals)):
    # Leave out this session
    train_df = scored_df[scored_df.animal_code != this_session]
    test_df = scored_df[scored_df.animal_code == this_session]

    # Train classifier
    X_train = np.stack(train_df.features.values)
    y_train = train_df.bsa_aligned_event_codes.values
    X_test = np.stack(test_df.features.values)
    y_test = test_df.bsa_aligned_event_codes.values

    bsa_pred_y = test_df.bsa_pred.values
    bsa_pred_y_list.append(bsa_pred_y)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_y_list.append(y_test)
    pred_y_list.append(y_pred)

    # Get shap values
    # explainer = shap.Explainer(clf)
    # shap_values = explainer(X_train)
    # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # mean_abs_shap_list.append(mean_abs_shap)
    mean_abs_shap_list.append(clf.feature_importances_)


# Get accuracy and f1
bsa_accuracy = [accuracy_score(y_test, bsa_pred_y) \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
accuracy = [accuracy_score(y_test, pred_y) for y_test, pred_y in zip(test_y_list, pred_y_list)]
bsa_f1 = [f1_score(y_test, bsa_pred_y, average='macro') \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
f1 = [f1_score(y_test, pred_y, average='macro') \
        for y_test, pred_y in zip(test_y_list, pred_y_list)]
bsa_confusion = [confusion_matrix(y_test, bsa_pred_y, normalize='true') \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
confusion = [confusion_matrix(y_test, pred_y, normalize='true') \
        for y_test, pred_y in zip(test_y_list, pred_y_list)]

fig, ax = plt.subplots(1,2)
ax[0].scatter(bsa_accuracy, accuracy)
ax[0].plot([0,1],[0,1],'k--')
ax[0].set_xlabel('BSA Accuracy')
ax[0].set_ylabel('Our Accuracy')
ax[0].set_title('Accuracy Comparison')
ax[0].set_aspect('equal')
ax[1].scatter(bsa_f1, f1)
ax[1].plot([0,1],[0,1],'k--')
ax[1].set_xlabel('BSA F1')
ax[1].set_ylabel('Our F1')
ax[1].set_title('F1 Comparison')
ax[1].set_aspect('equal')
fig.suptitle('Leave One Session Out\nComparison of BSA vs Our Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(bsa_comparison_plot_dir, 
                         'leave_one_animal_out_comparison.png'),
            bbox_inches='tight')
plt.close()

# Plot mean+std confusion matrices 
mean_bsa_confusion = np.mean(bsa_confusion, axis=0)
std_bsa_confusion = np.std(bsa_confusion, axis=0)
mean_confusion = np.mean(confusion, axis=0)
std_confusion = np.std(confusion, axis=0)

confusion_diff = [x-y for x,y in zip(confusion, bsa_confusion)]
# norm_confusion_diff = [x/np.sum(np.abs(x)) for x in confusion_diff]
mean_confusion_diff = np.mean(confusion_diff, axis=0)
std_confusion_diff = np.std(confusion_diff, axis=0)


fig, ax = plt.subplots(1,3, figsize=(20,5))
img = ax[0].matshow(mean_bsa_confusion, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(img, ax=ax[0])
for i in range(mean_bsa_confusion.shape[0]):
    for j in range(mean_bsa_confusion.shape[1]):
        ax[0].text(j, i, f'{mean_bsa_confusion[i,j]:.2f}\n±{std_bsa_confusion[i,j]:.2f}',
                 ha='center', va='center', color='k', fontweight='bold')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')
ax[0].set_xticks(np.arange(3))
ax[0].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[0].set_yticks(np.arange(3))
ax[0].set_yticklabels(bsa_labels)
ax[0].set_title('Mean+Std BSA Confusion Matrix')
img = ax[1].matshow(mean_confusion, cmap='viridis', vmin=0, vmax=1)
plt.colorbar(img, ax=ax[1])
for i in range(mean_confusion.shape[0]):
    for j in range(mean_confusion.shape[1]):
        ax[1].text(j, i, f'{mean_confusion[i,j]:.2f}\n±{std_confusion[i,j]:.2f}',
                 ha='center', va='center', color='k', fontweight='bold')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')
ax[1].set_xticks(np.arange(3))
ax[1].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[1].set_yticks(np.arange(3))
ax[1].set_yticklabels(bsa_labels)
ax[1].set_title('Mean+Std Our Confusion Matrix')

cmap = plt.cm.jet
norm = mpl.colors.CenteredNorm(0, 0.5)
im = ax[2].matshow(mean_confusion_diff, cmap=cmap, norm=norm)
plt.colorbar(label = '<- BSA Predicted More | Our Predicted More ->', ax=ax[2], mappable=im)
for i in range(mean_confusion_diff.shape[0]):
    for j in range(mean_confusion_diff.shape[1]):
        ax[2].text(j, i, f'{mean_confusion_diff[i,j]:.2f}\n±{std_confusion_diff[i,j]:.2f}',
                 ha='center', va='center', color='k', fontweight='bold')
ax[2].set_xlabel('Predicted')
ax[2].set_ylabel('True')
ax[2].set_xticks(np.arange(3))
ax[2].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[2].set_yticks(np.arange(3))
ax[2].set_yticklabels(bsa_labels)
ax[2].set_title('Mean+Std Confusion Difference')
plt.tight_layout()
plt.savefig(os.path.join(bsa_comparison_plot_dir, 'mean_std_confusion_diff_animal.png'),
            bbox_inches='tight')
plt.close()


# Plot shap values
shap_df = pd.DataFrame(mean_abs_shap_list, columns=feature_names)
# Melt
shap_df = shap_df.melt(var_name='Feature', value_name='Mean Abs SHAP')
# Sort features by mean abs shap
shap_df.sort_values('Mean Abs SHAP', ascending=False, inplace=True)


sns.boxplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
            fill = False, showfliers=False)
sns.stripplot(data=shap_df, x='Feature', y='Mean Abs SHAP',
              color='k', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Abs SHAP')
plt.title('Mean Abs SHAP Comparison')
plt.tight_layout()
plt.savefig(os.path.join(bsa_comparison_plot_dir, 'mean_abs_shap_animal_comparison.png'),
            bbox_inches='tight')
plt.close()

##############################
# Mean values for features across different movements

reverse_label_map = {
        0 : 'gape',
        1 : 'MTM',
        2 : 'nothing'
        }

feature_mat = np.stack(scored_df.features)
labels = scored_df.bsa_aligned_event_codes.values
feature_df = pd.DataFrame(
        columns = feature_names,
        data = feature_mat
        )
feature_df['labels'] = labels
feature_df['label_names'] = feature_df['labels'].map(reverse_label_map)
feature_df_long = feature_df.melt(
        id_vars = ['labels', 'label_names'],
        value_vars = feature_df.columns.difference(['labels'])
        )

g = sns.catplot(
        data = feature_df_long,
        x = 'label_names',
        y = 'value',
        col = 'variable',
        kind = 'box'
        )
# plt.show()
g.savefig(os.path.join(plot_dir, 'zscored_movement_features_box.png'),
          dpi = 300, bbox_inches = 'tight')
plt.close(g)

# Plot with subset of features
wanted_features = ['amplitude_abs','max_freq']
feature_df_long = feature_df_long.loc[feature_df_long['variable'].isin(wanted_features)]

# For each feature, perform anova
feature_groups = [x[1] for x in list(feature_df_long.groupby(['variable']))]

anova_list = []
for this_group in feature_groups:
    anova_out = pg.anova(
            data = this_group,
            dv = 'value',
            between = 'label_names'
            )
    anova_list.append(anova_out)

anova_p_list = [x['p-unc'].values[0] for x in anova_list]

g = sns.catplot(
        data = feature_df_long,
        x = 'label_names',
        y = 'value',
        col = 'variable',
        kind = 'box'
        )
for i, this_ax in enumerate(g.axes.flatten()):
    this_title = this_ax.title.get_text()
    new_title = this_title + f' ::: p-value = {anova_p_list[i]:.2E}'
    this_ax.set_title(new_title)
g.savefig(os.path.join(plot_dir, 'zscored_movement_wanted_features_box.png'),
          dpi = 300, bbox_inches = 'tight')
plt.close(g)

