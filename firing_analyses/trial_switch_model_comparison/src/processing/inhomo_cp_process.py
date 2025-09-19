"""
Switch changepoint:
    - Fit switch changepoint model over fixed number of states
        - Variable number of states in other models is to account
            for trial variability
    - Fit with both MCMC and VI
    - Calculate error in state transitions
        - Each trial will have different number of transitions

For each model:
    - Record time taken to fit model (and across all models if for multiple states)

For changepoint models:
    - Save variables as histograms with 1 percentile bins

"""

from tqdm import tqdm, trange
import os
import sys
import pandas as pd
import numpy as np
from time import time
from pickle import dump, load
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/trial_switch_model_comparison'

plot_dir = f'{base_dir}/plots'
inhomo_plot_dir = f'{plot_dir}/inhomo_cp_plots'

if not os.path.exists(inhomo_plot_dir):
    os.makedirs(inhomo_plot_dir)

# src_dir = f'{base_dir}/src'
# sys.path.append(src_dir)
# from data_gen_utils import return_poisson_data_switch 

artifact_dir = f'{base_dir}/artifacts'

# from dynamax.hidden_markov_model import PoissonHMM
# import jax.numpy as jnp
# import jax.random as jr

data_path = f'{artifact_dir}/data_dict.pkl'
df = pd.read_pickle(data_path)
df['data_index'] = df.index

out_path = f'{artifact_dir}/inhomo_change_fits'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Get list of all pkl files in out_path
pkl_files = glob(f'{out_path}/*.pkl')
inhomo_cp_dict_list = [load(open(f, 'rb')) for f in tqdm(pkl_files)]
inhomo_cp_frame = pd.DataFrame(inhomo_cp_dict_list)
# Drop duplicate data_index
inhomo_cp_frame = inhomo_cp_frame.drop_duplicates(subset='data_index')

############################################################
# Merge fit_df with inhomo_cp_frame 
############################################################
fit_df = pd.merge(
        df[['data_index','n_states', 'n_trial_states',
            'nrn_count', 'mean_rate', 'trial_section_list',
            'tau_array_list']],
        inhomo_cp_frame,
        on='data_index'
        )
fit_df.to_pickle(f'{artifact_dir}/inhomo_cp_best_fit_df.pkl')

##############################
##############################
# Plot inferred state transitions
for row_ind, row in fit_df.iterrows():
    n_states = row['n_states']
    n_trial_states = row['n_trial_states']
    trial_section_list = row['trial_section_list']
    flat_trial_sections = np.unique(
            [item for sublist in trial_section_list for item in sublist]
            )
    actual_tau_array = row['tau_array_list']
    tau_trial_samples = row['tau_trial_samples']
    tau_samples = row['tau_samples']
    tau_samples_long = np.reshape(tau_samples, (-1, *tau_samples.shape[2:]))
    mode_tau = stats.mode(tau_samples_long, axis=0)[0]

    # Plot actual tau as filled circles
    # and inferred tau as lines
    fig, ax = plt.subplots(1,2, sharey=True, figsize=(10,10))

    ax[0].hist(
            tau_trial_samples.flatten(), 
            bins=np.max(flat_trial_sections), 
            orientation='horizontal', 
            label='Trial Tau Samples')
    for i, section in enumerate(flat_trial_sections-0.5):
        if i == 0:
            ax[0].axhline(section, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Trial Section')
        else:
            ax[0].axhline(section, color='r', linestyle='--', linewidth=2, alpha=0.7) 
    # ax[0].set_title(f'Data Index: {row["data_index"]}, n_trial_states: {n_trial_states}')
    ax[0].legend()
    ax[0].set_ylabel('Trial #')
    ax[0].set_xlabel('Count')

    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, n_states))
    for trial_ind in range(mode_tau.shape[0]):
        for change_ind in range(n_states-1):
            ax[1].plot(
                    actual_tau_array[trial_ind, change_ind],
                    trial_ind,
                    'o',
                    color=colors[change_ind],
                    alpha=0.5,
                    )
            ax[1].scatter(
                    mode_tau[trial_ind, change_ind],
                    trial_ind,
                    # color=colors[change_ind],
                    c = 'k',
                    s = 50,
                    marker = '|',
                    )
    for i, section in enumerate(flat_trial_sections-0.5):
        if i == 0:
            ax[1].axhline(section, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Trial Section')
        else:
            ax[1].axhline(section, color='r', linestyle='--', linewidth=2, alpha=0.7)
    fig.suptitle(f'Data Index: {row["data_index"]}, n_states: {n_states}, n_trial_states: {n_trial_states}')
    ax[1].legend()
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Trial #')
    # plt.show()
    fig.savefig(f'{inhomo_plot_dir}/data_index_{row["data_index"]}_tau_comparison.png')
    plt.close(fig)

##############################
# For each data_index, infer number of trial sections
for row_ind, row in fit_df.iterrows():
    n_states = row['n_states']
    n_trial_states = row['n_trial_states']
    trial_section_list = row['trial_section_list']
    flat_trial_sections = np.unique(
            [item for sublist in trial_section_list for item in sublist]
            )
    actual_tau_array = row['tau_array_list']
    tau_trial_samples = row['tau_trial_samples']
    tau_samples = row['tau_samples']
    tau_samples_long = np.reshape(tau_samples, (-1, *tau_samples.shape[2:]))
    mode_tau = stats.mode(tau_samples_long, axis=0)[0]

    # For each trace, take mode across samples
    mode_tau_trial = stats.mode(tau_trial_samples, axis=1)[0]

    temp_df = pd.DataFrame(mode_tau_trial)
    temp_df = temp_df.sort_values(by=temp_df.columns.tolist())
    mode_tau_trial = temp_df.to_numpy()

    for trace_ind, this_trace in enumerate(mode_tau_trial):
        plt.scatter(
                this_trace,
                np.ones(len(this_trace))*trace_ind, 
                c = 'k',
                s = 50,
                marker = '|',
                )
    plt.xlim([-1, mode_tau.shape[0]])
    plt.ylabel('Trace')
    plt.xlabel('Trial #')
    # plt.show()
    plt.suptitle(f'Data Index: {row["data_index"]}, n_states: {n_states}, n_trial_states: {n_trial_states}')
    plt.savefig(f'{inhomo_plot_dir}/data_index_{row["data_index"]}_trial_section_inference.png')
    plt.close()
