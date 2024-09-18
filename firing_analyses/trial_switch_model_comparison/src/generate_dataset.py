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
import os
import sys

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/trial_switch_model_comparison'

src_dir = f'{base_dir}/src'
sys.path.append(src_dir)
from data_gen_utils import return_poisson_data_switch 

artifact_dir = f'{base_dir}/artifacts'
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

plot_dir = f'{base_dir}/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

nrn_dist = np.random.lognormal(5, 4, 10000)
nrn_dist = nrn_dist[nrn_dist<40]
nrn_dist = nrn_dist[nrn_dist>5]

rate_dist = np.random.lognormal(2, 1, 10000)
rate_dist = rate_dist[rate_dist<20]

min_len = np.min([len(rate_dist), len(nrn_dist)])
rate_dist_cut = rate_dist[:min_len]
nrn_dist_cut = nrn_dist[:min_len]

fig, ax = plt.subplots(1,3, figsize = (15,5))
ax[0].hist(nrn_dist_cut, bins = 20);
ax[0].set_xlabel('Nrn count')
ax[0].set_ylabel('Count')
ax[1].hist(rate_dist_cut, bins = 20);
ax[1].set_xlabel('Mean rate')
ax[1].set_ylabel('Count')
ax[2].hist2d(rate_dist_cut, nrn_dist_cut, bins = 20);
ax[2].set_xlabel('Mean rate')
ax[2].set_ylabel('Nrn count')
fig.suptitle('Distributions of neuron count and mean rate')
fig.savefig(f'{plot_dir}/nrn_rate_distributions.png')
plt.close(fig)

n_samples = 100
mean_r_list = []
r_array_list = []
spike_array_list = []
tau_array_list = []
trial_section_list = []
states_list = []
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
    states_list.append(n_states)

fig, ax = plt.subplots(1,2, sharey=True)
ax[0].scatter(rate_dist_cut[:n_samples], mean_r_list, 
         alpha = 0.3)
ax[0].set_xlabel('Scaler')
ax[0].set_ylabel('Mean spikes per bin')
ax[1].scatter(nrn_dist_cut[:n_samples], mean_r_list, 
         alpha = 0.3)
ax[1].set_xlabel('Nrn counts')
fig.suptitle('Mean rate vs mean spikes per bin')
fig.savefig(f'{plot_dir}/rate_vs_spikes.png')
plt.close(fig)

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

data_dict = {
    'r_array_list' : r_array_list,
    'spike_array_list' : spike_array_list,
    'tau_array_list' : tau_array_list,
    'trial_section_list' : trial_section_list,
    'mean_rate' : mean_r_list,
    'nrn_count' : nrn_dist_cut[:n_samples],
    'n_states' :  states_list, 
    'n_trial_states' : [len(x) for x in trial_section_list],
}
df = pd.DataFrame(data_dict)
df.to_pickle(f'{artifact_dir}/data_dict.pkl')

sns.pairplot(df.iloc[:,-4:], corner = True)
plt.savefig(f'{plot_dir}/pairplot.png')
plt.close()
