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
import pymc as pm
import pytensor.tensor as tt

import os
# os.environ["MKL_NUM_THREADS"]='1'
# os.environ["OMP_NUM_THREADS"]='1'

base_dir = '/media/bigdata/projects/pytau'
sys.path.append(base_dir)
import pytau.changepoint_model as models

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/trial_switch_model_comparison'

src_dir = f'{base_dir}/src'
sys.path.append(src_dir)

artifact_dir = f'{base_dir}/artifacts'

data_path = f'{artifact_dir}/data_dict.pkl'
df = pd.read_pickle(data_path)

out_path = f'{artifact_dir}/inhomo_change_fits'
if not os.path.exists(out_path):
    os.makedirs(out_path)

############################################################
def stick_breaking(beta):
    portion_remaining = tt.concatenate(
        [[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

def single_taste_poisson_dpp_trial_switch(
        spike_array,
        switch_components,
        states):
    """
    Assuming only emissions change across trials
    Changepoint distribution remains constant

    spike_array :: trials x nrns x time
    states :: number of states to include in the model 
    """

    trial_num, nrn_num, time_bins = spike_array.shape
    
    with pm.Model() as model:

        # Define Emissions

        # nrns
        nrn_lambda = pm.Exponential('nrn_lambda', 10, shape=(nrn_num))

        # nrns x switch_comps
        trial_lambda = pm.Exponential('trial_lambda',
                                      nrn_lambda.dimshuffle(0, 'x'),
                                      shape=(nrn_num, switch_components))

        # nrns x switch_comps x states
        state_lambda = pm.Exponential('state_lambda',
                                      trial_lambda.dimshuffle(0, 1, 'x'),
                                      shape=(nrn_num, switch_components, states))
        # Define Changepoints
        # Assuming distribution of changepoints remains
        # the same across all trials

        a = pm.HalfCauchy('a_tau', 3., shape=states - 1)
        b = pm.HalfCauchy('b_tau', 3., shape=states - 1)

        even_switches = np.linspace(0, 1, states+1)[1:-1]
        tau_latent = pm.Beta('tau_latent', a, b,
                             # testval=even_switches,
                             shape=(trial_num, states-1)).sort(axis=-1)

        # Trials x Changepoints
        tau = pm.Deterministic('tau', time_bins * tau_latent)

        # Hyperpriors on alpha
        a_gamma = pm.Gamma('a_gamma', 10, 1)
        b_gamma = pm.Gamma('b_gamma', 1.5, 1)

        # Concentration parameter for beta
        alpha = pm.Gamma('alpha', a_gamma, b_gamma)

        # Draw beta's to calculate stick lengths
        beta = pm.Beta('beta', 1, alpha, shape=switch_components)

        # Calculate stick lengths using stick_breaking process
        w_raw = pm.Deterministic('w_raw', stick_breaking(beta))

        # Make sure lengths add to 1, and scale to length of data
        tau_trial_latent = pm.Deterministic('w_latent', tt.cumsum(w_raw / w_raw.sum())[:-1])
        # w_latent = pm.Deterministic('w_latent', w_raw / w_raw.sum())
        # tau = pm.Deterministic('tau', tt.cumsum(w_latent * length)[:-1])

        # Trial_changepoints
        tau_trial = pm.Deterministic('tau_trial', trial_num * tau_trial_latent)

        trial_idx = np.arange(trial_num)
        trial_selector = tt.math.sigmoid(
            trial_idx[np.newaxis, :] - tau_trial.dimshuffle(0, 'x'))

        trial_selector = tt.concatenate(
            [np.ones((1, trial_num)), trial_selector], axis=0)
        inverse_trial_selector = 1 - trial_selector[1:, :]
        inverse_trial_selector = tt.concatenate([inverse_trial_selector,
                                                np.ones((1, trial_num))], axis=0)

        # First, we can "select" sets of emissions depending on trial_changepoints
        # switch_comps x trials
        trial_selector = np.multiply(trial_selector, inverse_trial_selector)

        # state_lambda: nrns x switch_comps x states

        # selected_trial_lambda : nrns x states x trials
        selected_trial_lambda = pm.Deterministic('selected_trial_lambda',
                                                 tt.sum(
                                                     # "nrns" x switch_comps x "states" x trials
                                                     trial_selector.dimshuffle(
                                                         'x', 0, 'x', 1) * state_lambda.dimshuffle(0, 1, 2, 'x'),
                                                     axis=1)
                                                 )

        # Then, we can select state_emissions for every trial
        idx = np.arange(time_bins)

        # tau : Trials x Changepoints
        weight_stack = tt.math.sigmoid(
            idx[np.newaxis, :]-tau[:, :, np.newaxis])
        weight_stack = tt.concatenate(
            [np.ones((trial_num, 1, time_bins)), weight_stack], axis=1)
        inverse_stack = 1 - weight_stack[:, 1:]
        inverse_stack = tt.concatenate(
            [inverse_stack, np.ones((trial_num, 1, time_bins))], axis=1)

        # Trials x states x Time
        weight_stack = np.multiply(weight_stack, inverse_stack)

        # Convert selected_trial_lambda : nrns x trials x states x "time"

        # nrns x trials x time
        lambda_ = tt.sum(selected_trial_lambda.dimshuffle(0, 2, 1, 'x') * weight_stack.dimshuffle('x', 0, 1, 2),
                         axis=2)

        # Convert to : trials x nrns x time
        lambda_ = lambda_.dimshuffle(1, 0, 2)

        # Add observations
        observation = pm.Poisson("obs", lambda_, observed=spike_array)

    return model

############################################################

n_chains = 24
for i, this_row in tqdm(df.iterrows()):

    n_states = this_row['n_states']
    n_nrns = int(this_row['nrn_count'])

    spike_array = this_row['spike_array_list']
    time_bins = spike_array.shape[-1]

    start_time = time()
    model = single_taste_poisson_dpp_trial_switch(
            spike_array,
            switch_components = 5,
            states = n_states,
            )
    dpp_trace = models.dpp_fit(model, n_cores = int(np.min([n_chains, 24])), 
                               n_chains = n_chains, use_numpyro=True,
                              tune = 125, draws = 125)
    end_time = time()
    time_taken = end_time - start_time

    # traces, samples, trials, transitions
    tau_samples = trace.posterior['tau'].values
    bins = np.arange(time_bins) 
    inds = list(np.ndindex(tau_samples.shape[2:]))
    tau_hist = np.zeros((tau_samples.shape[2], tau_samples.shape[3], time_bins-1))
    for i, j in inds:
        tau_hist[i,j], _ = np.histogram(tau_samples[:,:,i,j], bins=bins)

    out_dict = dict(
            data_index = i,
            fit_states = fit_states,
            time_taken = time_taken,
            elbo = elbo,
            tau_hist = tau_hist,
            )
    # out_df = pd.DataFrame(out_dict, index=[0])
    # out_df.to_pickle(f'{out_path}/data_{i}_fit_{fit_states}_repeat_{r}.pkl')
    file_name = f'{out_path}/data_{i}_fit_{fit_states}.pkl'
    with open(file_name, 'wb') as f:
        dump(out_dict, f)

