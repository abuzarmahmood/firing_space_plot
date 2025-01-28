"""
HMM:
    - Fit HMM over min - max states
    - Calc inferred number of states using cross-validation
    - Calculate state transitions per trial
    - Calculate error in state transitions

For each model:
    - Record time taken to fit model (and across all models if for multiple states)
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

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/trial_switch_model_comparison'

plot_dir = f'{base_dir}/plots'
hmm_plot_dir = f'{plot_dir}/hmm_plots'

if not os.path.exists(hmm_plot_dir):
    os.makedirs(hmm_plot_dir)

src_dir = f'{base_dir}/src'
sys.path.append(src_dir)
from data_gen_utils import return_poisson_data_switch 

artifact_dir = f'{base_dir}/artifacts'

# from dynamax.hidden_markov_model import PoissonHMM
# import jax.numpy as jnp
# import jax.random as jr

data_path = f'{artifact_dir}/data_dict.pkl'
df = pd.read_pickle(data_path)
df['data_index'] = df.index

out_path = f'{artifact_dir}/hmm_fits'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Get list of all pkl files in out_path
pkl_files = glob(f'{out_path}/*.pkl')
hmm_dict_list = [load(open(f, 'rb')) for f in tqdm(pkl_files)]
hmm_frame = pd.DataFrame(hmm_dict_list)

############################################################
# Get best fit 
############################################################
fit_df = pd.merge(
        df[['data_index','n_states', 'n_trial_states','nrn_count', 'mean_rate']],
        hmm_frame[['data_index','fit_states', 'mean_marg_log_prob', 'repeat', 'time_taken']],
        on='data_index'
        )
# Get mean across repeats
# fit_df = fit_df.groupby(['data_index','n_states','n_trial_states','fit_states']).mean().reset_index()

# For each data_index, only keep the fit_states with the highest log_prob 
fit_df = fit_df.loc[fit_df.groupby('data_index')['mean_marg_log_prob'].idxmax()]
fit_df.to_pickle(f'{artifact_dir}/hmm_best_fit_df.pkl')

############################################################
# Calculate state transitions per trial 
############################################################
for i, this_row in tqdm(fit_df.iterrows()):

    # n_states = this_row['n_states']
    n_nrns = int(this_row['nrn_count'])
    actual_states = int(this_row['n_states'])
    fit_states = int(this_row['fit_states'])
    data_index = int(this_row['data_index'])
    this_repeat = int(this_row['repeat'])
    this_tau_array = df.loc[data_index, 'tau_array_list']
    this_trial_sections = df.loc[data_index, 'trial_section_list']
    trial_sections_flat = np.unique(
            [item for sublist in this_trial_sections for item in sublist]
            )

    # Shape : trials, time, fit_states
    inferred_states = hmm_frame.loc[
            np.logical_and(
                np.logical_and(
                    hmm_frame['data_index'] == data_index, 
                    hmm_frame['fit_states'] == fit_states,  
                    ),
                hmm_frame['repeat'] == this_repeat
                )
            ].Ez.values[0]

    fig, ax = plt.subplots()
    im = ax.imshow(inferred_states.mean(axis=1).T, aspect='auto')
    ax.set_title(f'Inferred States Mass, actual states: {actual_states}, fit states: {fit_states}')
    ax.set_xlabel('Trials')
    ax.set_ylabel('States')
    for this_section in trial_sections_flat:
        ax.axvline(this_section-0.5, color='r')
    plt.colorbar(im, label = 'State Fraction')
    fig.savefig(f'{hmm_plot_dir}/data_{data_index}_fit_{fit_states}_repeat_{this_repeat}_inferred_states_mass.png',
                bbox_inches='tight')
    plt.close(fig)
    
    max_inferred_state = inferred_states.argmax(axis=2)
    fig, ax = plt.subplots()
    im = ax.imshow(max_inferred_state, aspect='auto', cmap = 'tab20')
    for i, this_trial in enumerate(this_tau_array):
        ax.scatter(this_trial-0.5, i*np.ones_like(this_trial), color='k', marker='|',
                   linewidth=5)
    ax.set_title(f'Inferred States Mass, actual states: {actual_states}, fit states: {fit_states}')
    for this_section in trial_sections_flat:
        ax.axhline(this_section-0.5, color='r', linewidth=2)
    fig.savefig(f'{hmm_plot_dir}/data_{data_index}_fit_{fit_states}_repeat_{this_repeat}_inferred_states.png',
                bbox_inches='tight')
    plt.close(fig)
    # plt.show()

############################################################
# Plot actual states vs fit states 
############################################################
# Plot 
# 1) n_states vs fit_states
# 2) n_trial_states * n_states vs fit_states
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].scatter(
        fit_df['n_states'],
        fit_df['fit_states'],
        c=fit_df['n_trial_states'],
        cmap='viridis',
        )
min_val = min(fit_df['n_states'].min(), fit_df['fit_states'].min())
max_val = max(fit_df['n_states'].max(), fit_df['fit_states'].max())
ax[0].plot([min_val,max_val],[min_val,max_val], 'k--')
ax[0].set_xlabel('Actual States')
ax[0].set_ylabel('Fit States')
ax[0].set_title('Actual States vs Fit States')
im = ax[1].scatter(
        fit_df['n_states']*fit_df['n_trial_states'],
        fit_df['fit_states'],
        c=fit_df['n_trial_states'],
        cmap='viridis',
        )
min_val = min(fit_df['n_states'].min()*fit_df['n_trial_states'].min(), fit_df['fit_states'].min())
max_val = max(fit_df['n_states'].max()*fit_df['n_trial_states'].max(), fit_df['fit_states'].max())
ax[1].plot([min_val,max_val],[min_val,max_val], 'k--')
ax[1].set_xlabel('(n_trial_states * n_states)')
ax[1].set_ylabel('Fit States')
ax[1].set_title('Actual Trial States vs Fit States')
# make cbar detached from subplots 
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='n_trial_states') 
plt.suptitle('HMM Fit States vs Actual States')
# plt.tight_layout()
# plt.show()
fig.savefig(f'{plot_dir}/hmm_fit_states_vs_actual_states.png',
        bbox_inches='tight')
plt.close(fig)
