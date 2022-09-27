import numpy as np
from tqdm import tqdm, trange
from numpy.linalg import norm
import pandas as pd
import pylab as plt
from itertools import product
from joblib import Parallel, delayed, cpu_count
import os

import sys
code_dir = '/media/bigdata/firing_space_plot/firing_analyses/intra_state_dynamics'
sys.path.append(code_dir)

from template_testing_tools import *

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/intra_state_dynamics/plots/sequential_sampling'

############################################################
# Create sampler
####################
# Test template projection
############################################################
## Simulation
############################################################
max_len = 300
min_dur = 40
def cost_func(firing_array, trans_points):
    basis_funcs = return_template_mat(trans_points, max_len) 
    basis_funcs = basis_funcs / norm(basis_funcs,axis=-1)[:,np.newaxis] 
    return np.mean(return_similarities(basis_funcs,firing_array))

states = 6
trans_points = return_transition_pos(
        n_states = states, 
        max_len = max_len,
        min_state_dur = min_dur,
        n_samples = 1).flatten()
print(trans_points)

nrns = 10
trials = 15
sim_w = np.random.random(size = (nrns,states))
basis_funcs = return_template_mat(trans_points, max_len) 
basis_funcs = basis_funcs / norm(basis_funcs,axis=-1)[:,np.newaxis] 
firing = np.matmul(sim_w, basis_funcs)*10
firing_array = np.tile(firing[:,np.newaxis], (1,trials,1))
firing_array = firing_array + np.random.randn(*firing_array.shape)*0.1

max_cost = cost_func(firing_array, trans_points)

plt.imshow(firing_array.mean(axis=1), interpolation = 'nearest', aspect = 'auto')
plt.show()

fin_cost_func = lambda trans_points : cost_func(firing_array, trans_points)

############################################################
# Estimate states and positions for simulated data
############################################################
n_states = np.arange(2,10)
n_transitions = n_states - 1
min_trans = n_transitions.min()
grid = np.arange(max_len)

samples_per_batch = 100
batches = 5000

base_transition_prior = [np.ones((x-1,max_len))*10 for x in n_states]
base_states_prior = np.ones(len(n_states)) * 30
transition_prior = [np.ones((x-1,max_len))*10 for x in n_states]
states_prior = np.ones(len(n_states)) * 30

cost_per_state = [[np.random.random()*0.1] for i in range(len(n_states))]

for this_batch in trange(batches):
    this_n_states = np.random.choice(
            a = n_states,
            size = 1,
            replace = False,
            p = states_prior / states_prior.sum()
            )[0]
    this_n_transitions = this_n_states - 1
    trans_max_list = [len(grid) - this_n_transitions + i \
            for i in range(this_n_transitions)]
    all_transition_inds = []
    for s in range(samples_per_batch):
        this_transition_inds = []
        for transition in range(this_n_transitions):
            if transition == 0:
                this_prior = transition_prior[this_n_transitions - min_trans][transition] 
                cut_prior = this_prior[:trans_max_list[transition]]
                this_trans = np.random.choice(
                        a = grid[:trans_max_list[transition]],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        )[0]
            else:
                this_prior = transition_prior[this_n_transitions - min_trans][transition] 
                cut_min = this_transition_inds[transition-1] + 1
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
    # Make sure cost_array normalization doesnt' create zeros or nans
    cost_array = []
    for x in all_transition_inds.T:
        try:
            cost_array.append(fin_cost_func(x))
        except:
            cost_array.append(0)
    cost_array = np.array(cost_array)
    norm_cost_array = cost_array.copy()
    norm_cost_array = norm_cost_array - np.nanmin(norm_cost_array)
    norm_cost_array = norm_cost_array / np.nanmax(norm_cost_array)
    if sum(np.isnan(cost_array)):
        cost_array[np.isnan(cost_array)] = 0
        norm_cost_array[np.isnan(norm_cost_array)] = 0
    #plt.scatter(all_transition_inds.flatten(), cost_array);plt.show()
    #cost_array = [fin_cost_func(x) for x in all_transition_inds.T]
    #if cost_array.max():
    #    norm_cost_array = cost_array / cost_array.max()
    # Update transition prior
    inds = np.array(list(np.ndindex(all_transition_inds.shape)))
    broadcasted_cost = np.tile(norm_cost_array, (len(inds) // len(cost_array)))
    transition_prior[this_n_transitions-min_trans][inds[:,0], 
            all_transition_inds.flatten()] += broadcasted_cost 
    # Update states_prior
    # Update with max cost for now
    # Might change to single samples (without batch)
    mean_state_costs = np.array([np.mean(x) for x in cost_per_state])
    current_mean_cost = cost_array.mean()
    # Use max cost for update because it is likely that simpler models
    # have a higher mean cost
    current_max_cost = cost_array.max()
    if current_max_cost > mean_state_costs.min():
        update_state_cost = current_max_cost - mean_state_costs.min()
        update_state_cost = \
                update_state_cost / (mean_state_costs.max() - mean_state_costs.min())
    else:
        update_state_cost = 0
    # Keep track of mean cost
    cost_per_state[this_n_transitions - min_trans].append(current_mean_cost)
    # Update with scaled max cost
    states_prior[this_n_transitions - min_trans] += update_state_cost 

#plt.show()
sub_transition_prior = [x-y for x,y in zip(transition_prior,base_transition_prior)]
sub_states_prior = states_prior - base_states_prior
fig,ax = plt.subplots(len(n_states) +1, figsize = (5,15))
ax[0].bar(n_states, sub_states_prior / sub_states_prior.sum())
for num in range(len(sub_transition_prior)):
    temp_prior = sub_transition_prior[num]
    temp_normal_prior = temp_prior / temp_prior.sum()
    ax[num+1].plot(temp_normal_prior.T)
    ax[num+1].set_title(f'States = {n_states[num]}')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,f'dummy_{states}state_data_sequential_sampling.png'))
plt.close(fig)
#plt.show()

############################################################
# Sample from updated priors
############################################################
n_new_samples = 5000
new_states= []
new_transitions = []
for s in trange(n_new_samples):
    #for this_batch in trange(batches):
    this_n_states = np.random.choice(
            a = n_states,
            size = 1,
            replace = False,
            #p = base_states_prior / base_states_prior.sum()
            p = states_prior / states_prior.sum()
            )[0]
    this_n_transitions = this_n_states - 1
    trans_max_list = [len(grid) - this_n_transitions + i \
            for i in range(this_n_transitions)]
    this_transition_inds = []
    for transition in range(this_n_transitions):
        if transition == 0:
            this_prior = transition_prior[this_n_transitions - min_trans][transition] 
            cut_prior = this_prior[:trans_max_list[transition]]
            this_trans = np.random.choice(
                    a = grid[:trans_max_list[transition]],
                    size = 1,
                    replace = False,
                    p = cut_prior / cut_prior.sum() 
                    )[0]
        else:
            this_prior = transition_prior[this_n_transitions - min_trans][transition] 
            cut_min = this_transition_inds[transition-1] + 1
            cut_max = trans_max_list[transition]
            cut_prior = this_prior[cut_min:cut_max]
            this_trans = np.random.choice(
                    a = grid[cut_min:cut_max],
                    size = 1,
                    replace = False,
                    p = cut_prior / cut_prior.sum() 
                    ).flatten()[0]
        this_transition_inds.append(this_trans)
    # Make sure cost_array normalization doesnt' create zeros or nans
    new_states.append(this_n_states)
    new_transitions.append(this_transition_inds)

cost_array = []
for x in tqdm(new_transitions):
    try:
        cost_array.append(fin_cost_func(np.array(x)))
    except:
        cost_array.append(0)
cost_array = np.array(cost_array)
if sum(np.isnan(cost_array)):
    cost_array[np.isnan(cost_array)] = 0
new_costs = cost_array.copy() 

new_states = np.array(new_states)
inds = np.argsort(new_states)
new_states = new_states[inds]
new_costs = [new_costs[x] for x in inds]
new_transitions = [new_transitions[x] for x in inds]

unique_new_states = np.unique(new_states)
fig,ax = plt.subplots(len(unique_new_states) + 1, 2, sharex = 'col', figsize = (5,10))
this_ax = fig.add_subplot(len(unique_new_states) + 1, 2,1)
ax[0,0].axis('off')
this_ax.hist(new_states)
for num, state in enumerate(unique_new_states):
    wanted_inds = np.where(new_states == state)[0]
    min_ind, max_ind = wanted_inds.min(), wanted_inds.max()
    wanted_costs = new_costs[min_ind:max_ind]
    cost_inds = np.argsort(wanted_costs)
    sorted_costs = np.array(wanted_costs)[cost_inds]
    if len(wanted_inds) > 1:
        wanted_transitions = np.stack(new_transitions[min_ind:max_ind])
    sorted_transitions = wanted_transitions[cost_inds]
    x = np.tile(np.arange(sorted_transitions.shape[0]), (sorted_transitions.shape[1],1))
    ax[num+1,0].set_title(f'States = {state}')
    ax[num+1, 0].scatter(sorted_transitions, x.T, 
            color = 'grey', alpha = 0.3, s = 2)
    ax[num+1, 0].set_xlim([0, len(grid)])
    ax[num+1, 1].plot(sorted_costs, x[0])
    ax[num+1, 1].axvline(max_cost, color = 'red', linestyle = '--')
    #for x in wanted_transitions.T:
    #    ax[num+1].hist(x, bins = 50)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,f'sampling_updated_prior_{states}_state.png'))
plt.close(fig)
#plt.show()

############################################################
# Sample from base priors
############################################################
n_new_samples = 5000
new_states= []
new_transitions = []
for s in trange(n_new_samples):
    #for this_batch in trange(batches):
    this_n_states = np.random.choice(
            a = n_states,
            size = 1,
            replace = False,
            p = base_states_prior / base_states_prior.sum()
            )[0]
    this_n_transitions = this_n_states - 1
    trans_max_list = [len(grid) - this_n_transitions + i \
            for i in range(this_n_transitions)]
    this_transition_inds = []
    for transition in range(this_n_transitions):
        if transition == 0:
            this_prior = base_transition_prior[this_n_transitions - min_trans][transition] 
            cut_prior = this_prior[:trans_max_list[transition]]
            this_trans = np.random.choice(
                    a = grid[:trans_max_list[transition]],
                    size = 1,
                    replace = False,
                    p = cut_prior / cut_prior.sum() 
                    )[0]
        else:
            this_prior = base_transition_prior[this_n_transitions - min_trans][transition] 
            cut_min = this_transition_inds[transition-1] + 1
            cut_max = trans_max_list[transition]
            cut_prior = this_prior[cut_min:cut_max]
            this_trans = np.random.choice(
                    a = grid[cut_min:cut_max],
                    size = 1,
                    replace = False,
                    p = cut_prior / cut_prior.sum() 
                    ).flatten()[0]
        this_transition_inds.append(this_trans)
    # Make sure cost_array normalization doesnt' create zeros or nans
    new_states.append(this_n_states)
    new_transitions.append(this_transition_inds)

cost_array = []
for x in tqdm(new_transitions):
    try:
        cost_array.append(fin_cost_func(np.array(x)))
    except:
        cost_array.append(0)
cost_array = np.array(cost_array)
if sum(np.isnan(cost_array)):
    cost_array[np.isnan(cost_array)] = 0
new_costs = cost_array.copy() 

new_states = np.array(new_states)
inds = np.argsort(new_states)
new_states = new_states[inds]
new_costs = [new_costs[x] for x in inds]
new_transitions = [new_transitions[x] for x in inds]

unique_new_states = np.unique(new_states)
fig,ax = plt.subplots(len(unique_new_states) + 1, 2, sharex = 'col', figsize = (5,10))
this_ax = fig.add_subplot(len(unique_new_states) + 1, 2,1)
ax[0,0].axis('off')
this_ax.hist(new_states)
for num, state in enumerate(unique_new_states):
    wanted_inds = np.where(new_states == state)[0]
    min_ind, max_ind = wanted_inds.min(), wanted_inds.max()
    wanted_costs = new_costs[min_ind:max_ind]
    cost_inds = np.argsort(wanted_costs)
    sorted_costs = np.array(wanted_costs)[cost_inds]
    if len(wanted_inds) > 1:
        wanted_transitions = np.stack(new_transitions[min_ind:max_ind])
    sorted_transitions = wanted_transitions[cost_inds]
    x = np.tile(np.arange(sorted_transitions.shape[0]), (sorted_transitions.shape[1],1))
    ax[num+1,0].set_title(f'States = {state}')
    ax[num+1, 0].scatter(sorted_transitions, x.T, 
            color = 'grey', alpha = 0.3, s = 2)
    ax[num+1, 0].set_xlim([0, len(grid)])
    ax[num+1, 1].plot(sorted_costs, x[0])
    ax[num+1, 1].axvline(max_cost, color = 'red', linestyle = '--')
    #for x in wanted_transitions.T:
    #    ax[num+1].hist(x, bins = 50)
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,f'sampling_base_prior_{states}_state.png'))
plt.close(fig)
#plt.show()
