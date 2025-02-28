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
from scipy.stats import spearmanr, pearsonr, wilcoxon, zscore
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
        split_latent_long = [np.stack(np.array_split(x, n_tastes, axis=1)) for x in down_latents_long]

        n_shuffles = 100

        for i in range(n_tastes):

            out_frame_path = os.path.join(artifact_dir, f'{basename}_taste_{i}.pkl')

            x = split_latent_long[0][i].T
            y = split_latent_long[1][i].T
            orig_mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
            orig_mlp.fit(x, y)
            actual_score = orig_mlp.score(x, y)

            # Shuffle trials
            raw_shuffle_scores = []
            n_trials = data.spikes[i].shape[0]
            for j in trange(n_shuffles):
                # Shuffle trials
                y_split = np.stack(np.array_split(y, n_trials, axis=0))
                y_split = np.random.permutation(y_split)
                y_shuffled = np.concatenate(y_split, axis=0)
                shuffle_score = orig_mlp.score(x, y_shuffled)
                raw_shuffle_scores.append(shuffle_score)

            retrained_shuffle_scores = []
            for j in trange(n_shuffles // 10):
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
            out_frame['actual_score'] = actual_score
            out_frame['raw_shuffle_scores'] = [raw_shuffle_scores]
            out_frame['retrained_shuffle_scores'] = [retrained_shuffle_scores]

            out_frame.to_pickle(out_frame_path)

    except Exception as e:
        print(f'Error in {this_dir}: {e}')

            # plt.hist(raw_shuffle_scores, label='Raw Shuffled', density=True)
            # plt.hist(retrained_shuffle_scores, label='Retrained Shuffled', density=True)
            # plt.axvline(actual_score, color='r', label='Actual')
            # plt.show()

            # fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
            # ax[0].imshow(y, aspect='auto', interpolation='none')
            # ax[1].imshow(y_shuffled, aspect='auto', interpolation='none')
            # plt.show()



        # # Remove first 250ms to aovid visible transients
        # cut_ind = np.where(pred_x > 100)[0][0]
        # latent_arrays = [x[..., cut_ind:] for x in latent_arrays]

        # fig, ax = vz.firing_overview(latent_arrays[0])
        # for this_ax in ax.flatten():
        #     for this_count in cum_trial_counts:
        #         this_ax.axhline(this_count, color='k', linestyle='--')
        # plt.show()
        #
    #     taste_idx_array = np.zeros(latent_arrays[0].shape[1:])
    #     for i in range(len(cum_trial_counts)-1):
    #         taste_idx_array[cum_trial_counts[i]:cum_trial_counts[i+1]] = i
    #
    #     taste_idx_long = np.reshape(taste_idx_array, -1)
    #
    #     n_components = 3
    #     nca_latents_long_list = []
    #     nca_latents_taste_list = []
    #     for this_latent in latent_arrays:
    #         this_latent_long = np.reshape(
    #             this_latent, (this_latent.shape[0], -1))
    #         this_nca = NCA(n_components=n_components)
    #         this_nca.fit(this_latent_long.T[::10], taste_idx_long[::10])
    #         nca_latents = this_nca.transform(this_latent_long.T).T
    #         nca_latents_long_list.append(nca_latents)
    #         nca_latents_trial = np.reshape(
    #             nca_latents, (n_components, *this_latent.shape[1:]))
    #         nca_latents_taste = []
    #         for i in range(len(cum_trial_counts)-1):
    #             nca_latents_taste.append(
    #                 nca_latents_trial[:, cum_trial_counts[i]:cum_trial_counts[i+1]])
    #         # nca_latents_taste = np.stack(nca_latents_taste, axis=0)
    #         nca_latents_taste_list.append(nca_latents_taste)
    #
    #     mean_nca_latents_taste = [
    #         [x.mean(axis=1) for x in y] for y in nca_latents_taste_list]
    #
    #     # Shape: (n_regions, n_tastes, n_components, n_trials, n_timepoints)
    #     # nca_latents_taste_array = np.stack(nca_latents_taste_list, axis=0)
    #
    #     # mean_nca_taste = np.mean(nca_latents_taste_array, axis=3)
    #     # Shape: (n_regions, n_tastes, n_components, n_timepoints)
    #     mean_nca_taste = np.stack(mean_nca_latents_taste, axis=0)
    #
    #     # Perform linear regression to align latents
    #     mean_nca_long = mean_nca_taste.swapaxes(1, 2).reshape(
    #         mean_nca_taste.shape[0], n_components, -1)
    #     X = mean_nca_long[0].T
    #     # Add bias
    #     X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=-1)
    #     Y = mean_nca_long[1].T
    #     align_mat = np.linalg.lstsq(X, Y, rcond=None)[0]
    #
    #     # Align mean latents
    #     aligned_mean_nca_taste = []
    #     taste_X = mean_nca_taste[0]
    #     # Add bias
    #     taste_X = np.concatenate(
    #         [taste_X, np.ones((taste_X.shape[0], 1, taste_X.shape[-1]))], axis=1)
    #     taste_Y = np.tensordot(taste_X, align_mat, axes=[1, 0]).swapaxes(1, 2)
    #     aligned_mean_nca_taste.append(taste_Y)
    #     aligned_mean_nca_taste.append(mean_nca_taste[1])
    #
    # except Exception as e:
    #     print(f'Error in {this_dir}: {e}')
    #     error_list.append(this_dir)
    #     continue
