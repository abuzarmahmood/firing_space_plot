"""
Non-switch changepoint:
    - Fit non-switch changepoint model over min - max states
    - Calculate inferred number of states using ELBO
    - Calculate error in state transitions
        - Each trial will have same number of transitions

For each model:
    - Record time taken to fit model (and across all models if for multiple states)
"""
import os
os.environ["MKL_NUM_THREADS"]='1'
os.environ["OMP_NUM_THREADS"]='1'

from tqdm import tqdm, trange
import sys
import pandas as pd
import numpy as np
from time import time
from pickle import dump, load
import pymc as pm
import pytensor.tensor as tt


base_dir = '/media/bigdata/projects/pytau'
sys.path.append(base_dir)
import pytau.changepoint_model as models

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/trial_switch_model_comparison'

src_dir = f'{base_dir}/src'
sys.path.append(src_dir)
from data_gen_utils import return_poisson_data_switch 

artifact_dir = f'{base_dir}/artifacts'

data_path = f'{artifact_dir}/data_dict.pkl'
df = pd.read_pickle(data_path)

out_path = f'{artifact_dir}/homo_change_fits'
if not os.path.exists(out_path):
    os.makedirs(out_path)

############################################################

def single_taste_poisson(
        spike_array,
        states,
        **kwargs):
    """Model for changepoint on single taste

    ** Largely taken from "non_hardcoded_changepoint_test_3d.ipynb"
    ** Note : This model does not have hierarchical structure for emissions

    Args:
        spike_array (3D Numpy array): trials x neurons x time
        states (int): Number of states to model

    Returns:
        pymc3 model: Model class containing graph to run inference on
    """

    mean_vals = np.array([np.mean(x, axis=-1)
                          for x in np.array_split(spike_array, states, axis=-1)]).T
    mean_vals = np.mean(mean_vals, axis=1)
    mean_vals += 0.01  # To avoid zero starting prob

    nrns = spike_array.shape[1]
    trials = spike_array.shape[0]
    idx = np.arange(spike_array.shape[-1])
    length = idx.max() + 1

    with pm.Model() as model:
        lambda_latent = pm.Exponential('lambda',
                                       1/mean_vals,
                                       shape=(nrns, states))

        if states -1 == 1:
            a_tau = pm.HalfCauchy('a_tau', 3.)
            b_tau = pm.HalfCauchy('b_tau', 3.)
        else:
            a_tau = pm.HalfCauchy('a_tau', 3., shape=states - 1)
            b_tau = pm.HalfCauchy('b_tau', 3., shape=states - 1)

        even_switches = np.linspace(0, 1, states+1)[1:-1]
        tau_latent = pm.Beta('tau_latent', a_tau, b_tau,
                             # testval=even_switches,
                             shape=(trials, states-1)).sort(axis=-1)

        tau = pm.Deterministic('tau',
                               idx.min() + (idx.max() - idx.min()) * tau_latent)

        weight_stack = tt.math.sigmoid(
            idx[np.newaxis, :]-tau[:, :, np.newaxis])
        weight_stack = tt.concatenate(
            [np.ones((trials, 1, length)), weight_stack], axis=1)
        inverse_stack = 1 - weight_stack[:, 1:]
        inverse_stack = tt.concatenate(
            [inverse_stack, np.ones((trials, 1, length))], axis=1)
        weight_stack = np.multiply(weight_stack, inverse_stack)

        lambda_ = tt.tensordot(weight_stack, lambda_latent, [
                               1, 1]).swapaxes(1, 2)
        observation = pm.Poisson("obs", lambda_, observed=spike_array)

    return model

############################################################

min_fit_states = df.n_states.min()
n_fit = int(1e5)
n_samples = int(2e4)

for i, this_row in tqdm(df.iterrows()):

    n_states = this_row['n_states']
    n_nrns = int(this_row['nrn_count'])
    # Don't waste resources fitting models with max states based on all data
    max_fit_states = n_states * 4 # Since we're using 3 max trial_states

    spike_array = this_row['spike_array_list']
    time_bins = spike_array.shape[-1]

    for fit_states in range(min_fit_states, max_fit_states+1):
        file_name = f'{out_path}/data_{i}_fit_{fit_states}.pkl'

        if not os.path.exists(file_name):

            start_time = time()
            model = single_taste_poisson(
                    spike_array,
                    fit_states,
                    )
            with model:
                inference = pm.ADVI('full-rank')
                approx = pm.fit(n=n_fit, method=inference)
                trace = approx.sample(draws=n_samples)
            end_time = time()
            time_taken = end_time - start_time

            elbo = -inference.hist[-1]
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
            with open(file_name, 'wb') as f:
                dump(out_dict, f)

