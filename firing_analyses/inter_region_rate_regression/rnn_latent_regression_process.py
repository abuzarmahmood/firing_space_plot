"""
Instead of calculating PCA from neural activity,
calculate it from latents inferred using the trained RNN.
"""

from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
import tables
import json
import hashlib
import base64
import seaborn as sns
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import spearmanr, pearsonr, wilcoxon, zscore, percentileofscore
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_validate
import matplotlib.pyplot as plt
from ast import literal_eval
import numpy as np
from pprint import pprint as pp
import sys
import os
from tqdm import tqdm, trange
import pandas as pd
from glob import glob
import pingouin as pg


blech_clust_dir = os.path.expanduser('~/Desktop/blech_clust')
sys.path.append(blech_clust_dir)
from utils.ephys_data import ephys_data
from utils.ephys_data import visualize as vz

base_dir =  '/media/bigdata/firing_space_plot/firing_analyses/inter_region_rate_regression/'
base_plot_dir = os.path.join(base_dir, 'plots')
artifact_dir = os.path.join(base_dir, 'artifacts', 'rnn_latent_regression')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

data_dir_list_path = '/media/fastdata/Thomas_Data/data/sorted_new/data_dir_list.txt'
data_dir_list = open(data_dir_list_path, 'r').read().splitlines()

animal_name = [x.split('/')[-2] for x in data_dir_list]
basename_list = [os.path.basename(x) for x in data_dir_list]
base_to_animal_map = dict(zip(basename_list, animal_name))

# rnn_latents_list = []
error_list = []
for this_dir in tqdm(data_dir_list):
    try:
        basename = os.path.basename(this_dir)
        print('==========================')
        print(f'Processing {basename}')
        print('==========================')
        data = ephys_data.ephys_data(this_dir)
        data.get_spikes()
        data.get_region_units()
        trial_counts = [len(x) for x in data.spikes]
        cum_trial_counts = np.concatenate([[0], np.cumsum(trial_counts)])
        with tables.open_file(data.hdf5_path, 'r') as h5:
            regions_list = h5.list_nodes('/rnn_output/regions')
            latent_arrays = [x.latent_out[:].T for x in regions_list]
            pred_x = [x.pred_x[:] for x in regions_list][0]

        latent_long = [x.reshape(x.shape[0], -1) for x in latent_arrays]

        # pls = PLSRegression(n_components=3)
        # pls.fit(latent_long[0].T, latent_long[1].T)
        # pls_latents = pls.transform(latent_long[0].T, latent_long[1].T)
        # # pred_y = pls.predict(latent_long[0].T)
        # pls_latents = [x.T for x in pls_latents]
        # # pls_latents = [latent_long[0].T, pred_y]
        # pls_latents_trial = [np.reshape(x, (3, *latent_arrays[0].shape[1:])) for x in pls_latents]
        #
        # mean_pls_latents_trial = np.stack([x.mean(axis=1) for x in pls_latents_trial])
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(*mean_pls_latents_trial[0], color='r')
        # ax.plot(*mean_pls_latents_trial[1], color='b')
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_zlabel('PC3')
        # plt.show()
        #
        # ind = 2
        # fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
        # ax[0].imshow(pls_latents_trial[0][ind])
        # ax[1].imshow(pls_latents_trial[1][ind])
        # plt.show()
        #
        ##############################
        # Reduce dims to 90% variance
        # Then perform MLP regression
        down_latents_long = []
        for this_latent in latent_long:
            pca = PCA(0.9)
            pca.fit(this_latent.T)
            down_latent = pca.transform(this_latent.T).T
            down_latents_long.append(down_latent)

        # fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
        # ax[0].imshow(down_latents_long[0], aspect='auto', interpolation='none')
        # ax[1].imshow(down_latents_long[1], aspect='auto', interpolation='none')
        # plt.show()

        n_tastes = len(data.spikes)
        trial_len = latent_arrays[0].shape[-1]
        # split_latent_long = [np.stack(np.array_split(x, n_tastes, axis=1)) for x in down_latents_long]
        split_latent_long = [
                [x[:, trial_len*cum_trial_counts[i]:trial_len*cum_trial_counts[i+1]] for i in range(len(cum_trial_counts)-1)] \
                    for x in down_latents_long
                ]

        n_shuffles = 100

        for i in range(n_tastes):

            out_frame_path = os.path.join(artifact_dir, f'{basename}_taste_{i}.pkl')
            if os.path.exists(out_frame_path):
                continue

            x = split_latent_long[0][i].T
            y = split_latent_long[1][i].T
            orig_mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
            orig_mlp.fit(x, y)
            y_pred = orig_mlp.predict(x)
            actual_score = orig_mlp.score(x, y)

            # Difference over time
            diff = y - y_pred

            n_trials = data.spikes[i].shape[0]
            diff_trial = np.stack(np.array_split(diff, n_trials, axis=0))
            # mean_abs_diff = np.mean(np.abs(diff_trial), axis=0)
            # norm_mean_abs_diff = np.linalg.norm(mean_abs_diff, axis=1)

            # Shuffle trials
            raw_shuffle_scores = []
            n_trials = data.spikes[i].shape[0]
            for j in range(n_shuffles):
                # Shuffle trials
                y_split = np.stack(np.array_split(y, n_trials, axis=0))
                y_split = np.random.permutation(y_split)
                y_shuffled = np.concatenate(y_split, axis=0)
                shuffle_score = orig_mlp.score(x, y_shuffled)
                raw_shuffle_scores.append(shuffle_score)

            retrained_shuffle_scores = []
            for j in range(n_shuffles // 10):
                # Shuffle trials
                y_split = np.stack(np.array_split(y, n_trials, axis=0))
                y_split = np.random.permutation(y_split)
                y_shuffled = np.concatenate(y_split, axis=0)
                mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
                mlp.fit(x, y_shuffled)
                shuffle_score = mlp.score(x, y_shuffled)
                retrained_shuffle_scores.append(shuffle_score)

            out_frame = pd.DataFrame(
                    dict(
                        basename=basename,
                        taste_idx=i,
                    ),
                    index=[0]
                    )
            out_frame['diff_trial'] = [diff_trial]
            out_frame['actual_score'] = actual_score
            out_frame['raw_shuffle_scores'] = [raw_shuffle_scores]
            out_frame['retrained_shuffle_scores'] = [retrained_shuffle_scores]

            out_frame.to_pickle(out_frame_path)

    except Exception as e:
        print(f'Error in {this_dir}: {e}')

# Load all saved pickles
pkl_list = glob(os.path.join(artifact_dir, '*.pkl'))

##############################
# Analyze timeseries of error in predictions
basename_list = []
taste_idx_list = []
norm_mean_abs_diff_list = []
for this_pkl in tqdm(pkl_list):
    this_frame = pd.read_pickle(this_pkl)
    wanted_data = this_frame[['basename', 'taste_idx', 'diff_trial']]
    diff_trial = wanted_data['diff_trial'].values[0]
    mean_abs_diff = np.mean(np.abs(diff_trial), axis=0)
    norm_mean_abs_diff = np.linalg.norm(mean_abs_diff, axis=1)
    basename_list.append(wanted_data['basename'].values[0])
    taste_idx_list.append(wanted_data['taste_idx'].values[0])
    norm_mean_abs_diff_list.append(norm_mean_abs_diff)

animal_list = [base_to_animal_map[x] for x in basename_list]
mean_norm_mean_abs_diff = np.stack(norm_mean_abs_diff_list).mean(axis=0)
zscored_norm_mean_abs_diff_list = [zscore(x) for x in norm_mean_abs_diff_list] 
mean_zscored_norm_mean_abs_diff = np.stack(zscored_norm_mean_abs_diff_list).mean(axis=0)

zscored_norm_mean_abs_diff_frame = pd.concat(
        [pd.DataFrame(
            dict(
                basename = [basename_list[i]]*len(zscored_norm_mean_abs_diff_list[i]),
                taste_idx = [taste_idx_list[i]]*len(zscored_norm_mean_abs_diff_list[i]),
                zscored_norm_mean_abs_diff = zscored_norm_mean_abs_diff_list[i],
                time = pred_x - 500,
                )
            ) for i in range(len(zscored_norm_mean_abs_diff_list))]
         )

wanted_time = [0, 2000]
zscored_norm_mean_abs_diff_frame = zscored_norm_mean_abs_diff_frame[
        (zscored_norm_mean_abs_diff_frame['time'] >= wanted_time[0]) & \
        (zscored_norm_mean_abs_diff_frame['time'] <= wanted_time[1])
        ]
bins_width = 250
zscored_norm_mean_abs_diff_frame['time_bin'] = pd.cut(
        zscored_norm_mean_abs_diff_frame['time'],
        bins=np.arange(wanted_time[0], wanted_time[1] + bins_width, bins_width),
        right=False
        )
# Reset index
zscored_norm_mean_abs_diff_frame.reset_index(drop=True, inplace=True)
# Set time_bin to the start of the bin
zscored_norm_mean_abs_diff_frame['time_bin'] = zscored_norm_mean_abs_diff_frame['time_bin'].apply(lambda x: x.left)
# Drop nans
zscored_norm_mean_abs_diff_frame.dropna(inplace=True)
# Convert to int
zscored_norm_mean_abs_diff_frame['time_bin'] = zscored_norm_mean_abs_diff_frame['time_bin'].astype(int)

# Get mean and std by time bin 
mean_zscored_norm_mean_abs_diff_frame = zscored_norm_mean_abs_diff_frame.groupby(
        ['basename', 'taste_idx', 'time_bin']
        ).mean().reset_index()
std_zscored_norm_mean_abs_diff_frame = zscored_norm_mean_abs_diff_frame.groupby(
        ['basename', 'taste_idx', 'time_bin']
        ).std().reset_index()
stats_zscored_norm_mean_abs_diff_frame = pd.merge(
        mean_zscored_norm_mean_abs_diff_frame,
        std_zscored_norm_mean_abs_diff_frame,
        on=['basename', 'taste_idx', 'time_bin'],
        suffixes=('_mean', '_std')
        )
# Rename
stats_zscored_norm_mean_abs_diff_frame.rename(
        columns={'zscored_norm_mean_abs_diff_mean': 'mean', 'zscored_norm_mean_abs_diff_std': 'std'},
        inplace=True
        )

unique_animals = np.unique(animal_list)
cmap = plt.cm.get_cmap('tab10', len(unique_animals))
fig, ax = plt.subplots(3,1, figsize=(7,7))
for i in range(len(norm_mean_abs_diff_list)):
    color_ind = np.where(unique_animals == animal_list[i])[0][0]
    ax[0].plot(pred_x - 500, norm_mean_abs_diff_list[i], color = cmap(color_ind), alpha=0.5) 
    ax[1].plot(pred_x - 500, zscored_norm_mean_abs_diff_list[i], color = cmap(color_ind), alpha=0.5)
ax[0].plot(pred_x - 500, mean_norm_mean_abs_diff, color='k', linewidth=2)
ax[0].set_title('Mean Abs Diff')
ax[1].plot(pred_x - 500, mean_zscored_norm_mean_abs_diff, color='k', linewidth=2)
ax[1].set_title('Zscored Mean Abs Diff')
ax[0].set_ylabel('Norm Mean Abs Diff')
ax[1].set_ylabel('Zscored Norm Mean Abs Diff')
ax[0].set_xlim(wanted_time)
ax[1].set_xlim(wanted_time)
sns.barplot(
        data=zscored_norm_mean_abs_diff_frame,
        x='time_bin',
        y='zscored_norm_mean_abs_diff',
        ax=ax[2]
        )
ax[2].set_title('Zscored Mean Abs Diff by Time Bin (error bars = SEM)\nNo significance in any error bin')
ax[2].set_xlabel('Time post-stimulus')
fig.suptitle('RNN Latent Regression Error')
plt.tight_layout()
fig.savefig(os.path.join(base_plot_dir, 'rnn_latent_regression_error.png'),
            bbox_inches='tight')
plt.close(fig)
# plt.show()


##############################
# Compile everything into a single dataframe for plotting
long_frame_list = []
for this_pkl in tqdm(pkl_list):
    this_frame = pd.read_pickle(this_pkl)
    this_frame.drop('diff_trial', axis=1, inplace=True)
    long_frame = this_frame.melt(id_vars=['basename', 'taste_idx']).explode('value')
    long_frame_list.append(long_frame)

long_frame = pd.concat(long_frame_list)

##############################
g = sns.catplot(
        data=long_frame,
        col = 'basename',
        x = 'taste_idx',
        hue='variable',
        y='value',
        kind='box',
        )
for ax in g.axes.flat:
    ax.set_title(ax.get_title(), rotation=45, ha='left')
fig = plt.gcf()
fig.set_size_inches(20, 10)
fig.suptitle('RNN Latent Regression Scores')
fig.savefig(os.path.join(base_plot_dir, 'rnn_latent_regression.png'),
            bbox_inches='tight')
plt.close(fig)
# plt.show()

##############################
# Min-max scale scores to allow easier comparison
def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())

group_list = list(long_frame.groupby(['basename', 'taste_idx',]))
updated_frames_list = []
for group in group_list:
    this_frame = group[1]
    this_frame['scaled_value'] = min_max_scale(this_frame['value'])
    updated_frames_list.append(this_frame)

# Tally of actual score being the highest
actual_score_max_bool = []
for frame in updated_frames_list:
    actual_score = frame[frame['variable'] == 'actual_score']['scaled_value'].values[0]
    max_score = frame['scaled_value'].max()
    actual_score_max_bool.append(actual_score == max_score)

group_inds = [x[0] for x in group_list]
actual_score_max_frame = pd.DataFrame(
        group_inds,
        columns=['basename', 'taste_idx']
        )
actual_score_max_frame['actual_score_max'] = actual_score_max_bool
actual_score_max_frame.groupby('basename').mean()
actual_score_max_frame['animal'] = actual_score_max_frame['basename'].map(base_to_animal_map)
mean_actual_score_max = actual_score_max_frame.groupby(['animal', 'basename']).mean()
mean_actual_score_max = mean_actual_score_max.reset_index()

# Get counts per basename
count_frame = actual_score_max_frame.groupby(['animal', 'basename']).count()
count_frame = count_frame.reset_index()

count_frame.sort_values(['animal', 'basename'], inplace=True)

g = sns.barplot(
        data=mean_actual_score_max,
        x='basename',
        y='actual_score_max',
        hue='animal',
        )
fig = plt.gcf()
fig.set_size_inches(5,7)
# Annotate with counts
for i, row in count_frame.iterrows():
    plt.text(row['basename'], 1.1, row['taste_idx'], ha='center') 
plt.ylim(0, 1.2)
# Rotate x labels
plt.xticks(rotation=90, ha='right')
# Put legend outside of plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Fraction of Trials where Actual Score is Max')
plt.tight_layout()
# plt.show()
fig.savefig(os.path.join(base_plot_dir, 'rnn_latent_regression_actual_score_max.svg'),
            bbox_inches='tight')
plt.close(fig)

##############################
# Calculate percentile of actual score vs retrained_shuffle_scores
percentile_list = []
for frame in updated_frames_list:
    actual_score = frame[frame['variable'] == 'actual_score']['scaled_value'].values[0]
    retrained_shuffle_scores = frame[frame['variable'] == 'retrained_shuffle_scores']['scaled_value'].values
    percentile = percentileofscore(retrained_shuffle_scores, actual_score)
    percentile_list.append(percentile) 

percentile_frame = pd.DataFrame(
        group_inds,
        columns=['basename', 'taste_idx']
        )
percentile_frame['percentile'] = percentile_list

percentile_frame['animal'] = percentile_frame['basename'].map(base_to_animal_map)

percentile_frame.sort_values(['animal', 'basename'], inplace=True)

fig, ax = plt.subplots(1,2, sharey=True)
g = sns.swarmplot(
        data=percentile_frame,
        x='basename',
        y='percentile',
        hue='animal',
        ax=ax[0]
        )
fig.set_size_inches(5,7)
# Rotate x labels
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].yscale('log')
ax[0].axhline(95, color='r', linestyle='--')
# Put legend outside of plot
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].hist(percentile_frame['percentile'], bins=20, orientation='horizontal') 
ax[1].axhline(95, color='r', linestyle='--')
ax[1].set_xlabel('Count')
fig.suptitle('Percentile of Actual Score vs Retrained Shuffle Scores')
plt.tight_layout()
# plt.show()
fig.savefig(os.path.join(base_plot_dir, 'rnn_latent_regression_percentile.svg'),
            bbox_inches='tight')
plt.close(fig)

##############################

# long_frame['scaled_value'] = np.concatenate(scaled_values)
long_frame = pd.concat(updated_frames_list)

g = sns.catplot(
        data=long_frame,
        col = 'basename',
        x = 'taste_idx',
        hue='variable',
        y='scaled_value',
        kind='box',
        )
for ax in g.axes.flat:
    ax.set_title(ax.get_title(), rotation=45, ha='left')
fig = plt.gcf()
fig.set_size_inches(20, 10)
fig.suptitle('RNN Latent Regression Scores (Scaled)')
fig.savefig(os.path.join(base_plot_dir, 'rnn_latent_regression_scaled.png'),
            bbox_inches='tight')
plt.close(fig)
# plt.show()

##############################
# Plot pooled scaled values
g = sns.boxenplot(
        data=long_frame,
        x = 'variable',
        hue='variable',
        y='scaled_value',
        )
fig = plt.gcf()
fig.set_size_inches(3, 5)
# Rotate x labels
plt.xticks(rotation=45, ha='right')
fig.suptitle('RNN Latent Regression Scores (Scaled - Pooled)')
fig.savefig(os.path.join(base_plot_dir, 'rnn_latent_regression_pooled.svg'),
            bbox_inches='tight')
plt.close(fig)

# Also plot mean pooled scaled values
mean_frame = long_frame.groupby(['basename', 'taste_idx', 'variable']).mean().reset_index()
g = sns.boxenplot(
        data=mean_frame,
        x = 'variable',
        hue='variable',
        y='scaled_value',
        )
fig = plt.gcf()
fig.set_size_inches(3, 5)
# Rotate x labels
plt.xticks(rotation=45, ha='right')
fig.suptitle('Mean RNN Latent Regression Scores (Scaled - Pooled)')
fig.savefig(os.path.join(base_plot_dir, 'rnn_latent_regression_pooled_mean.svg'),
            bbox_inches='tight')
plt.close(fig)

# Pairwise comparisons for pooled scaled values
long_frame.dropna(inplace=True)
long_frame['scaled_value'] = long_frame['scaled_value'].astype(float)
pg.pairwise_ttests(
        data=long_frame.groupby(['basename', 'taste_idx', 'variable']).mean().reset_index(),
        dv='scaled_value',
        within='variable',
        padjust='bonf',
        )
