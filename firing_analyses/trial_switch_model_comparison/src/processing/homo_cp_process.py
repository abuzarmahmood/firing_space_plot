"""
Non-switch changepoint:
    - Fit non-switch changepoint model over min - max states
    - Calculate inferred number of states using ELBO
    - Calculate error in state transitions
        - Each trial will have same number of transitions

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

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/trial_switch_model_comparison'

plot_dir = f'{base_dir}/plots'
hmm_plot_dir = f'{plot_dir}/homo_cp_plots'

if not os.path.exists(hmm_plot_dir):
    os.makedirs(hmm_plot_dir)

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

out_path = f'{artifact_dir}/homo_change_fits'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Get list of all pkl files in out_path
pkl_files = glob(f'{out_path}/*.pkl')
homo_cp_dict_list = [load(open(f, 'rb')) for f in tqdm(pkl_files)]
homo_cp_frame = pd.DataFrame(homo_cp_dict_list)

############################################################
# Get best fit 
############################################################
fit_df = pd.merge(
        df[['data_index','n_states', 'n_trial_states','nrn_count', 'mean_rate']],
        homo_cp_frame[['data_index','fit_states', 'elbo', 'time_taken']],
        on='data_index'
        )
# Get mean across repeats
# fit_df = fit_df.groupby(['data_index','n_states','n_trial_states','fit_states']).mean().reset_index()

# For each data_index, only keep the fit_states with the highest log_prob 
fit_df = fit_df.loc[fit_df.groupby('data_index')['elbo'].idxmax()]
fit_df.to_pickle(f'{artifact_dir}/homo_cp_best_fit_df.pkl')
