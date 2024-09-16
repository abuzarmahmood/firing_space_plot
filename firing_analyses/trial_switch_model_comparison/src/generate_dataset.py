"""
Generate datasets for the experiments.

Grid over the following parameters:
    - Number of neurons
    - Mean firing rate

Uniform sampling over the following parameters:
    - Number of states
    - Number of trial states
"""

import pickle
import scipy.stats as stats
import pymc as pm
import pytensor.tensor as tt
import numpy as np
import pylab as plt
import arviz as az
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange

nrn_dist = np.random.lognormal(5, 4, 10000)
nrn_dist = nrn_dist[nrn_dist<40]
nrn_dist = nrn_dist[nrn_dist>5]
plt.hist(nrn_dist, bins = 20);

rate_dist = np.random.lognormal(2, 1, 10000)
rate_dist = rate_dist[rate_dist<20]
plt.hist(rate_dist, bins = 20);

min_len = np.min([len(rate_dist), len(nrn_dist)])
rate_dist_cut = rate_dist[:min_len]
nrn_dist_cut = nrn_dist[:min_len]
plt.hist2d(rate_dist_cut, nrn_dist_cut, bins = 20);
plt.xlabel('Mean rate')
plt.ylabel('Nrn count')

n_samples = 100
mean_r_list = []
r_array_list = []
spike_array_list = []
tau_array_list = []
trial_section_list = []
for i in trange(n_samples):
    n_trial_states = np.random.choice([1,2,3])
    n_states = np.random.choice(np.arange(2,8))
    r_array, spike_array, tau_array, trial_sections = return_poisson_data_switch(
        nrns = int(nrn_dist_cut[i]), 
        states = n_states, 
        rate_scaler = rate_dist_cut[i],
        trial_count = 30, 
        length = 40,
        n_trial_states = n_trial_states,
    )
    mean_r_list.append(spike_array.mean(axis=None))
    r_array_list.append(r_array)
    spike_array_list.append(spike_array)
    tau_array_list.append(tau_array)
    trial_section_list.append(trial_sections)

fig, ax = plt.subplots(1,2, sharey=True)
ax[0].scatter(rate_dist_cut[:n_samples], mean_r_list, 
         alpha = 0.3)
ax[0].set_xlabel('Scaler')
ax[0].set_ylabel('Mean spikes per bin')
ax[1].scatter(nrn_dist_cut[:n_samples], mean_r_list, 
         alpha = 0.3)
ax[1].set_xlabel('Nrn counts')

# Save as dataframe to pkl
# Save:
# - r_array_list
# - spike_array_list
# - tau_array_list
# - trial_section_list
# - mean_rate
# - nrn_count
# - n_states
# - n_trial_states
