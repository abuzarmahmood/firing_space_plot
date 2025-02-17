"""
Generate NCA plots of RNN latents to highlight separation of tastes
"""

import os
import tables
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from tqdm import tqdm

blech_clust_path = os.path.expanduser('~/Desktop/blech_clust') 
import sys
sys.path.append(blech_clust_path)

from utils.ephys_data import ephys_data
from utils.ephys_data import visualize as vz

base_plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/inter_region_rate_regression/plots'
this_plot_dir = os.path.join(base_plot_dir, 'rnn_nca_plots')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

data_dir_list_path = '/media/fastdata/Thomas_Data/data/sorted_new/data_dir_list.txt'
data_dir_list = open(data_dir_list_path, 'r').read().splitlines()

def make_ax_transparent(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return ax

# rnn_latents_list = []
error_list = []
for this_dir in tqdm(data_dir_list):
    try:
        basename = os.path.basename(this_dir)
        data = ephys_data.ephys_data(this_dir)
        data.get_spikes()
        data.get_region_units()
        trial_counts = [len(x) for x in data.spikes]
        cum_trial_counts = np.concatenate([[0], np.cumsum(trial_counts)])
        with tables.open_file(data.hdf5_path,'r') as h5:
            regions_list = h5.list_nodes('/rnn_output/regions')
            latent_arrays = [x.latent_out[:].T for x in regions_list]
            pred_x = [x.pred_x[:] for x in regions_list][0]

        # Remove first 250ms to aovid visible transients
        cut_ind = np.where(pred_x > 100)[0][0]
        latent_arrays = [x[...,cut_ind:] for x in latent_arrays]

        # fig, ax = vz.firing_overview(latent_arrays[0])
        # for this_ax in ax.flatten():
        #     for this_count in cum_trial_counts:
        #         this_ax.axhline(this_count, color='k', linestyle='--')
        # plt.show()
        #
        taste_idx_array = np.zeros(latent_arrays[0].shape[1:])
        for i in range(len(cum_trial_counts)-1):
            taste_idx_array[cum_trial_counts[i]:cum_trial_counts[i+1]] = i

        taste_idx_long = np.reshape(taste_idx_array, -1)

        n_components = 3
        nca_latents_long_list = []
        nca_latents_taste_list = []
        for this_latent in latent_arrays:
            this_latent_long = np.reshape(this_latent, (this_latent.shape[0], -1))
            this_nca = NCA(n_components=n_components)
            this_nca.fit(this_latent_long.T[::10], taste_idx_long[::10])
            nca_latents = this_nca.transform(this_latent_long.T).T
            nca_latents_long_list.append(nca_latents)
            nca_latents_trial = np.reshape(nca_latents, (n_components, *this_latent.shape[1:])) 
            nca_latents_taste = []
            for i in range(len(cum_trial_counts)-1):
                nca_latents_taste.append(nca_latents_trial[:,cum_trial_counts[i]:cum_trial_counts[i+1]])
            # nca_latents_taste = np.stack(nca_latents_taste, axis=0)
            nca_latents_taste_list.append(nca_latents_taste)


        mean_nca_latents_taste = [[x.mean(axis=1) for x in y] for y in nca_latents_taste_list]

        # Shape: (n_regions, n_tastes, n_components, n_trials, n_timepoints)
        # nca_latents_taste_array = np.stack(nca_latents_taste_list, axis=0)

        # mean_nca_taste = np.mean(nca_latents_taste_array, axis=3)
        # Shape: (n_regions, n_tastes, n_components, n_timepoints)
        mean_nca_taste = np.stack(mean_nca_latents_taste, axis=0)

        # Perform linear regression to align latents
        mean_nca_long = mean_nca_taste.swapaxes(1,2).reshape(mean_nca_taste.shape[0], n_components, -1)
        X = mean_nca_long[0].T
        # Add bias
        X = np.concatenate([X, np.ones((X.shape[0],1))], axis=-1)
        Y = mean_nca_long[1].T 
        align_mat = np.linalg.lstsq(X,Y, rcond=None)[0]

        # Align mean latents
        aligned_mean_nca_taste = []
        taste_X = mean_nca_taste[0]
        # Add bias
        taste_X = np.concatenate([taste_X, np.ones((taste_X.shape[0],1, taste_X.shape[-1]))], axis=1)
        taste_Y = np.tensordot(taste_X, align_mat, axes=[1,0]).swapaxes(1,2)
        aligned_mean_nca_taste.append(taste_Y)
        aligned_mean_nca_taste.append(mean_nca_taste[1])

        fig = plt.figure(figsize=(10,5))
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        ax_list = [ax0, ax1]
        for region_ind in range(len(nca_latents_taste_list)):
            for taste_ind in range(len(nca_latents_taste_list[region_ind])):
                ax_list[region_ind].plot(*mean_nca_taste[region_ind, taste_ind], label=f'Taste {taste_ind}')
        for i, region_name in enumerate(data.region_names):
            ax_list[i].set_title(region_name)
            ax_list[i].legend()
            # ax_list[i].set_axis_off()
            ax_list[i] = make_ax_transparent(ax_list[i])
        # plt.show()
        fig.suptitle(f'{basename} RNN-NCA Latents Unaligned')
        plt.tight_layout()
        plt.savefig(os.path.join(this_plot_dir, f'{basename}_rnn_nca_latents_unaligned.png'))
        plt.close()

        # Also plot aligned
        fig = plt.figure(figsize=(10,5))
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        ax_list = [ax0, ax1]
        for region_ind in range(len(nca_latents_taste_list)):
            for taste_ind in range(len(nca_latents_taste_list[region_ind])):
                ax_list[region_ind].plot(
                        *aligned_mean_nca_taste[region_ind][taste_ind], 
                        label=f'Taste {taste_ind}')
        for i, region_name in enumerate(data.region_names):
            ax_list[i].set_title(region_name)
            ax_list[i].legend()
            # ax_list[i].set_axis_off()
            ax_list[i] = make_ax_transparent(ax_list[i])
        # Set same lims
        # ax1.set_xlim(ax0.get_xlim())
        # ax1.set_ylim(ax0.get_ylim())
        # ax1.set_zlim(ax0.get_zlim())
        # plt.show()
        fig.suptitle(f'{basename} RNN-NCA Latents Aligned')
        plt.tight_layout()
        plt.savefig(os.path.join(this_plot_dir, f'{basename}_rnn_nca_latents_aligned.png'))
        plt.close()

    except Exception as e:
        print(f'Error in {this_dir}: {e}')
        error_list.append(this_dir)
        continue


############################################################
# Do the same with pca
############################################################
error_list = []
for this_dir in tqdm(data_dir_list):
    try:
        basename = os.path.basename(this_dir)
        data = ephys_data.ephys_data(this_dir)
        data.get_spikes()
        data.get_region_units()
        trial_counts = [len(x) for x in data.spikes]
        cum_trial_counts = np.concatenate([[0], np.cumsum(trial_counts)])
        with tables.open_file(data.hdf5_path,'r') as h5:
            regions_list = h5.list_nodes('/rnn_output/regions')
            latent_arrays = [x.latent_out[:].T for x in regions_list]
            pred_x = [x.pred_x[:] for x in regions_list][0]

        # Remove first 250ms to aovid visible transients
        cut_ind = np.where(pred_x > 100)[0][0]
        latent_arrays = [x[...,cut_ind:] for x in latent_arrays]

        # fig, ax = vz.firing_overview(latent_arrays[0])
        # for this_ax in ax.flatten():
        #     for this_count in cum_trial_counts:
        #         this_ax.axhline(this_count, color='k', linestyle='--')
        # plt.show()
        #
        taste_idx_array = np.zeros(latent_arrays[0].shape[1:])
        for i in range(len(cum_trial_counts)-1):
            taste_idx_array[cum_trial_counts[i]:cum_trial_counts[i+1]] = i

        taste_idx_long = np.reshape(taste_idx_array, -1)

        n_components = 3
        nca_latents_long_list = []
        nca_latents_taste_list = []
        for this_latent in latent_arrays:
            this_latent_long = np.reshape(this_latent, (this_latent.shape[0], -1))
            this_nca = PCA(n_components=n_components)
            this_nca.fit(this_latent_long.T[::10], taste_idx_long[::10])
            nca_latents = this_nca.transform(this_latent_long.T).T
            nca_latents_long_list.append(nca_latents)
            nca_latents_trial = np.reshape(nca_latents, (n_components, *this_latent.shape[1:])) 
            nca_latents_taste = []
            for i in range(len(cum_trial_counts)-1):
                nca_latents_taste.append(nca_latents_trial[:,cum_trial_counts[i]:cum_trial_counts[i+1]])
            # nca_latents_taste = np.stack(nca_latents_taste, axis=0)
            nca_latents_taste_list.append(nca_latents_taste)


        mean_nca_latents_taste = [[x.mean(axis=1) for x in y] for y in nca_latents_taste_list]

        # Shape: (n_regions, n_tastes, n_components, n_trials, n_timepoints)
        # nca_latents_taste_array = np.stack(nca_latents_taste_list, axis=0)

        # mean_nca_taste = np.mean(nca_latents_taste_array, axis=3)
        # Shape: (n_regions, n_tastes, n_components, n_timepoints)
        mean_nca_taste = np.stack(mean_nca_latents_taste, axis=0)

        # Perform linear regression to align latents
        mean_nca_long = mean_nca_taste.swapaxes(1,2).reshape(mean_nca_taste.shape[0], n_components, -1)
        X = mean_nca_long[0].T
        # Add bias
        X = np.concatenate([X, np.ones((X.shape[0],1))], axis=-1)
        Y = mean_nca_long[1].T 
        align_mat = np.linalg.lstsq(X,Y, rcond=None)[0]

        # Align mean latents
        aligned_mean_nca_taste = []
        taste_X = mean_nca_taste[0]
        # Add bias
        taste_X = np.concatenate([taste_X, np.ones((taste_X.shape[0],1, taste_X.shape[-1]))], axis=1)
        taste_Y = np.tensordot(taste_X, align_mat, axes=[1,0]).swapaxes(1,2)
        aligned_mean_nca_taste.append(taste_Y)
        aligned_mean_nca_taste.append(mean_nca_taste[1])

        fig = plt.figure(figsize=(10,5))
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        ax_list = [ax0, ax1]
        for region_ind in range(len(nca_latents_taste_list)):
            for taste_ind in range(len(nca_latents_taste_list[region_ind])):
                ax_list[region_ind].plot(*mean_nca_taste[region_ind, taste_ind], label=f'Taste {taste_ind}')
        for i, region_name in enumerate(data.region_names):
            ax_list[i].set_title(region_name)
            ax_list[i].legend()
            # ax_list[i].set_axis_off()
            ax_list[i] = make_ax_transparent(ax_list[i])
        # plt.show()
        fig.suptitle(f'{basename} RNN-PCA Latents Unaligned')
        plt.tight_layout()
        plt.savefig(os.path.join(this_plot_dir, f'{basename}_rnn_pca_latents_unaligned.png'))
        plt.close()

        # Also plot aligned
        fig = plt.figure(figsize=(10,5))
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        ax_list = [ax0, ax1]
        for region_ind in range(len(nca_latents_taste_list)):
            for taste_ind in range(len(nca_latents_taste_list[region_ind])):
                ax_list[region_ind].plot(
                        *aligned_mean_nca_taste[region_ind][taste_ind], 
                        label=f'Taste {taste_ind}')
        for i, region_name in enumerate(data.region_names):
            ax_list[i].set_title(region_name)
            ax_list[i].legend()
            # ax_list[i].set_axis_off()
            ax_list[i] = make_ax_transparent(ax_list[i])
        # Set same lims
        # ax1.set_xlim(ax0.get_xlim())
        # ax1.set_ylim(ax0.get_ylim())
        # ax1.set_zlim(ax0.get_zlim())
        # plt.show()
        fig.suptitle(f'{basename} RNN-PCA Latents Aligned')
        plt.tight_layout()
        plt.savefig(os.path.join(this_plot_dir, f'{basename}_rnn_pca_latents_aligned.png'))
        plt.close()

    except Exception as e:
        print(f'Error in {this_dir}: {e}')
        error_list.append(this_dir)
        continue
