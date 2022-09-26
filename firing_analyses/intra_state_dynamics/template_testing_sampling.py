"""
Updates can also be made using top n values rather than percentile cutoff
"""

import numpy as np
from tqdm import tqdm, trange
from numpy.linalg import norm
import pandas as pd
import pylab as plt
from scipy.ndimage import gaussian_filter1d as gauss_filt1d

# Sample transitions with weights
grid_len = 100
n_transitions = n_states - 1
transition_inds = [10,20,30,40,50,60] 
n_states = len(transition_inds)+1#4
cost_per_transition = np.zeros((n_states-1,grid_len))
for i,val in enumerate(transition_inds):
    cost_per_transition[i,val] = 1
cost_per_transition = gauss_filt1d(cost_per_transition, 3, axis=-1)
grid = np.arange(grid_len)
prior = np.ones(cost_per_transition.shape)*10

samples_per_batch = 50
batches = 1000
alpha = 10
# For each sample, draw transitions sequentially according to the
# normalized prior
# Then update prior using top "alpha" points in terms of cost 
for this_batch in range(batches):
    if this_batch == 0:
        fig,ax = plt.subplots(2,1, sharex=True)
        ax[1].plot(cost_per_transition.T)
    if ((this_batch+1) % 100) == 0:
        #plt.imshow(prior, aspect='auto');plt.colorbar();plt.show()
        normalized_prior = (prior/prior.sum(axis=-1)[:,np.newaxis])
        ax[0].plot(normalized_prior.T);
        ax[0].set_ylim((0, normalized_prior.max(axis=None)))
    trans_max_list = [len(grid) - n_transitions + i for i in range(n_transitions)]
    all_transition_inds = []
    for s in range(samples_per_batch):
        this_transition_inds = []
        for transition in range(n_states-1):
            if transition == 0:
                this_prior = prior[transition] 
                cut_prior = this_prior[:trans_max_list[transition]]
                this_trans = np.random.choice(
                        a = grid[:trans_max_list[transition]],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        )[0]
            else:
                this_prior = prior[transition] 
                cut_min = this_transition_inds[transition-1]
                cut_max = trans_max_list[transition]
                cut_prior = this_prior[cut_min:cut_max]
                this_trans = np.random.choice(
                        a = grid[cut_min:cut_max],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        ).flatten()[0]
            this_transition_inds.append(this_trans)
        all_transition_inds.append(this_transition_inds)
    all_transition_inds = np.stack(all_transition_inds).T
    # Evaluate costs
    cost_array = np.stack([this_cost[this_inds] for this_cost,this_inds \
            in zip(cost_per_transition, all_transition_inds)] )
    # Make sure cost_array normalization doesnt' create zeros
    if all(cost_array.max(axis=1)):
        cost_array = cost_array / cost_array.max(axis=1)[:,np.newaxis]
    #critical_vals = np.percentile(cost_array, 100-alpha, axis=-1)
    #update_bool = cost_array > critical_vals[:,np.newaxis]
    #update_inds = np.where(update_bool)
    #prior[update_inds[0], all_transition_inds[update_bool]] += 1
    inds = np.array(list(np.ndindex(all_transition_inds.shape)))
    prior[inds[:,0], all_transition_inds.flatten()] += cost_array.flatten()
plt.show()
