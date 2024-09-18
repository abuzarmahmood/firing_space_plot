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

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/trial_switch_model_comparison'

src_dir = f'{base_dir}/src'
sys.path.append(src_dir)
from data_gen_utils import return_poisson_data_switch 

artifact_dir = f'{base_dir}/artifacts'

from dynamax.hidden_markov_model import PoissonHMM
import jax.numpy as jnp
import jax.random as jr

data_path = f'{artifact_dir}/data_dict.pkl'
df = pd.read_pickle(data_path)

out_path = f'{artifact_dir}/hmm_fits'
if not os.path.exists(out_path):
    os.makedirs(out_path)

############################################################

train_test_split = 0.8
min_fit_states = df.n_states.min()
n_repeats = 5

for i, this_row in tqdm(df.iterrows()):

    n_states = this_row['n_states']
    n_nrns = int(this_row['nrn_count'])
    # Don't waste resources fitting models with max states based on all data
    max_fit_states = n_states * 4 # Since we're using 3 max trial_states

    spike_array = this_row['spike_array_list']
    # needs to be of shape: trials, time, neurons
    spike_array = spike_array.swapaxes(1,2)

    for fit_states in range(min_fit_states, max_fit_states+1):
        for r in range(n_repeats):
            trial_inds = np.arange(len(spike_array))
            train_inds = np.random.choice(trial_inds, int(train_test_split*len(trial_inds)), replace=False)
            test_inds = np.setdiff1d(trial_inds, train_inds)
            train_spike_array = spike_array[train_inds]
            test_spike_array = spike_array[test_inds]

            start_time = time()
            hmm = PoissonHMM(fit_states, n_nrns, emission_prior_concentration=3.0, emission_prior_rate=1.0)
            initial_probs = jnp.ones((fit_states,)) / fit_states
            transition_matrix = 0.90 * jnp.eye(fit_states) + 0.10 * jnp.ones((fit_states, fit_states)) / fit_states
            true_params, param_props = hmm.initialize(jr.PRNGKey(0), initial_probs=initial_probs, transition_matrix=transition_matrix)

            params, param_props = hmm.initialize(jr.PRNGKey(1234))
            params, lps = hmm.fit_em(params, param_props, train_spike_array, num_iters=1000)
            end_time = time()
            time_taken = end_time - start_time

            # Calculate cross-validated log probability
            marg_log_prob = np.array([np.float32(hmm.marginal_log_prob(params, x)) for x in test_spike_array])
            mean_marg_log_prob = np.mean(marg_log_prob)

            # Calculate posterior state probabilities
            posterior = [hmm.smoother(params, this_trial) for this_trial in spike_array]
            Ez = np.stack([x.smoothed_probs for x in posterior])

            out_dict = dict(
                    data_index = i,
                    fit_states = fit_states,
                    repeat = r,
                    mean_marg_log_prob = mean_marg_log_prob, 
                    time_taken = time_taken,
                    Ez = Ez,
                    params = params,
                    )
            # out_df = pd.DataFrame(out_dict, index=[0])
            # out_df.to_pickle(f'{out_path}/data_{i}_fit_{fit_states}_repeat_{r}.pkl')
            file_name = f'{out_path}/data_{i}_fit_{fit_states}_repeat_{r}.pkl'
            with open(file_name, 'wb') as f:
                dump(out_dict, f)

