"""
Analyze timecourse of effect of perturbation on evoked response
from Narendra's 2500 ms perturbation data.
"""

blech_clust_path = '/home/abuzarmahmood/Desktop/blech_clust'
import sys
sys.path.append(blech_clust_path)
from utils.ephys_data import ephys_data
from utils.ephys_data import visualize as vz

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns
import os
import pandas as pd
from scipy.stats import zscore

data_dir_list_path = '/media/storage/NM_resorted_data/laser_2500_dirs.txt'
data_dir_list = np.sort(open(data_dir_list_path, 'r').readlines())
data_dir_list = [data_dir.strip() for data_dir in data_dir_list]

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/NM_opto_gwt'
plot_dir = os.path.join(base_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Get firing rates for all cells
for data_dir in data_dir_list:
    basename = os.path.basename(data_dir)
    dat = ephys_data.ephys_data(data_dir)
    dat.get_spikes()
    dat.get_firing_rates()
    dat.get_sequestered_firing()

    # vz.firing_overview(dat.all_normalized_firing)
    # plt.show()
    #
    # vz.firing_overview(dat.trial_inds_frame.loc[0, 'firing'].swapaxes(0, 1))
    # vz.firing_overview(dat.trial_inds_frame.loc[1, 'firing'].swapaxes(0, 1))
    # plt.show()

    mean_seq_firing = dat.trial_inds_frame.copy()
    mean_seq_firing['mean_firing'] = mean_seq_firing['firing'].apply(lambda x: np.mean(x, axis=0))
    mean_seq_firing['nrn_num'] = [np.arange(mean_seq_firing.mean_firing[0].shape[0])]*len(mean_seq_firing)

    mean_seq_firing = mean_seq_firing.explode(['mean_firing', 'nrn_num'])
    mean_seq_firing.reset_index(inplace=True)
    t_vec = np.arange(-2000, 4775, 25)+250
    mean_seq_firing['time'] = [t_vec]*len(mean_seq_firing)
    mean_seq_firing = mean_seq_firing.explode(['mean_firing', 'time'])

    time_lims = [-500, 2500]
    mean_seq_firing = mean_seq_firing[(mean_seq_firing.time >= time_lims[0]) & (mean_seq_firing.time <= time_lims[1])]

    g = sns.relplot(
        x='time',
        y='mean_firing',
        hue='laser_duration_ms',
        row = 'dig_in_num_taste',
        col = 'nrn_num',
        data=mean_seq_firing,
        kind='line',
        palette = 'tab10',
        facet_kws={'sharex': True, 'sharey': False},
    )
    # Plot line at x=0
    for ax in g.axes.flatten():
        ax.axvline(0, color='r', linestyle='--')
    # plt.show()
    g.savefig(os.path.join(plot_dir, f'perturbation_timecourse_{basename}.png'),
              bbox_inches='tight')
    plt.close('all')

    # Cut into 100ms bins, and calculate relative change in firing rate for every bin
    off_firing = mean_seq_firing[mean_seq_firing.laser_duration_ms == 0]
    on_firing = mean_seq_firing[mean_seq_firing.laser_duration_ms == 2500]

    diff_frame = pd.merge(
        off_firing,
        on_firing,
        on=['dig_in_num_taste', 'nrn_num', 'time'],
        suffixes=('_off', '_on')
    )
    diff_frame.drop(
            columns=[
                'laser_duration_ms_off', 
                'laser_duration_ms_on',
                'firing_off',
                'firing_on',
                'trial_inds_off',
                'trial_inds_on',
                'laser_lag_ms_off',
                ], 
            inplace=True)
    diff_frame['firing_diff'] = diff_frame['mean_firing_on'] - diff_frame['mean_firing_off']
    diff_frame['abs_firing_diff'] = np.abs(diff_frame['firing_diff'])
    # Add noise to off firing to avoid division by zero
    diff_frame['mean_firing_off'] += np.abs(np.random.randn(diff_frame.shape[0])*1e-2)
    diff_frame['rel_firing_diff'] = diff_frame['abs_firing_diff']/diff_frame['mean_firing_off']

    # For each dig_in_taste_num, create a pivot with col = time, row = nrn_num
    t_vec_cut = t_vec[(t_vec >= time_lims[0]) & (t_vec <= time_lims[1])]
    all_pivots_list = []
    for dig_in_num in diff_frame.dig_in_num_taste.unique():
        dig_in_diff_frame = diff_frame[diff_frame.dig_in_num_taste == dig_in_num]
        dig_in_diff_pivot = dig_in_diff_frame.pivot(index='nrn_num', columns='time', values='rel_firing_diff')
        all_pivots_list.append(dig_in_diff_pivot)


    fig, ax = plt.subplots(len(all_pivots_list), 1, 
                           sharex=True, sharey=True,
                           figsize = (5, 5*len(all_pivots_list)))
    for idx, dig_in_diff_pivot in enumerate(all_pivots_list):
        im = ax[idx].pcolormesh(
            t_vec_cut,
            np.arange(dig_in_diff_pivot.shape[0]),
            dig_in_diff_pivot.values.astype(float), 
            cmap = 'viridis',
            shading='auto',
            vmin=0,
            vmax=1
            )
        plt.colorbar(im, ax=ax[idx], label='Relative Firing Rate Change')
        ax[idx].set_ylabel('Neuron Number')
        ax[idx].set_title(f'Dig_in_num: {dig_in_num}')
        ax[idx].axvline(0, color='r', linestyle='--')
    ax[0].set_title(f'Firing rate change / No Laser Firing Rate')
    ax[-1].set_xlabel('Time post stimulus (ms)')
    fig.suptitle(f'{basename}')
    # plt.show()
    fig.savefig(os.path.join(plot_dir, f'perturbation_timecourse_rel_diff_{basename}.png'),
                bbox_inches='tight')
    plt.close('all')

    # Also plot absolute firing rate change
    all_pivots_list = []
    for dig_in_num in diff_frame.dig_in_num_taste.unique():
        dig_in_diff_frame = diff_frame[diff_frame.dig_in_num_taste == dig_in_num]
        dig_in_diff_pivot = dig_in_diff_frame.pivot(index='nrn_num', columns='time', values='abs_firing_diff')
        all_pivots_list.append(dig_in_diff_pivot)

    fig, ax = plt.subplots(len(all_pivots_list), 2, 
                           sharex=True, 
                           figsize = (10, 5*len(all_pivots_list)))
    for idx, dig_in_diff_pivot in enumerate(all_pivots_list):
        zscored_vals = zscore(dig_in_diff_pivot.values.astype(float), axis=-1)
        im = ax[idx, 0].pcolormesh(
            t_vec_cut,
            np.arange(dig_in_diff_pivot.shape[0]),
            dig_in_diff_pivot.values.astype(float), 
            cmap = 'viridis',
            shading='auto',
            vmin = -3,
            vmax = 3
            )
        plt.colorbar(im, ax=ax[idx,0], label='Zscored Firing Rate Change')
        ax[idx,0].set_ylabel('Neuron Number')
        ax[idx,0].set_title(f'Dig_in_num: {dig_in_num}')
        ax[idx,0].axvline(0, color='r', linestyle='--')
        # Plot mean value
        ax[idx,1].plot(t_vec_cut, np.mean(zscored_vals, axis=0))
        ax[idx,1].axvline(0, color='r', linestyle='--')
    ax[0,0].set_title(f'Zscored Absolute Firing Rate Change')
    ax[-1,0].set_xlabel('Time post stimulus (ms)')
    fig.suptitle(f'{basename}')
    # plt.show()
    fig.savefig(os.path.join(plot_dir, f'perturbation_timecourse_zscored_diff_{basename}.png'),
                bbox_inches='tight')
    plt.close('all')

