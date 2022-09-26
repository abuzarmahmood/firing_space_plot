"""
Updates can also be made using top n values rather than percentile cutoff
"""

import numpy as np
from tqdm import tqdm, trange
from numpy.linalg import norm
import pandas as pd
import pylab as plt
from scipy.ndimage import gaussian_filter1d as gauss_filt1d
from tqdm import tqdm, trange
import seaborn as sns

############################################################
# Sample transitions with weights
grid_len = 100
transition_inds = [5,30,45,60,75] 
n_states = len(transition_inds)+1#4
n_transitions = n_states - 1
cost_per_transition = np.zeros((n_states-1,grid_len))
for i,val in enumerate(transition_inds):
    cost_per_transition[i,val] = 1
cost_per_transition = gauss_filt1d(cost_per_transition, 3, axis=-1)
grid = np.arange(grid_len)

# Cost function
# Note, a single draw will have a single cost
# Different transitions will not have different costs
def cost_func(transition_array):
    """transition_array : transitions x samples
    """
    # Evaluate costs
    cost_array = np.stack([this_cost[this_inds] for this_cost,this_inds \
            in zip(cost_per_transition, transition_array)] )
    return cost_array.sum(axis=0)

############################################################
# Updating priors 
############################################################
prior = np.ones(cost_per_transition.shape)*10
samples_per_batch = 50
batches = 1000
#alpha = 10
# For each sample, draw transitions sequentially according to the
# normalized prior
# Then update prior using top "alpha" points in terms of cost 
max_cost_dat = []
max_cost = []
for this_batch in trange(batches):
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
                this_prior = transition_prior[this_n_transitions - 2][transition] 
                cut_prior = this_prior[:trans_max_list[transition]]
                this_trans = np.random.choice(
                        a = grid[:trans_max_list[transition]],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        )[0]
            else:
                this_prior = transition_prior[this_n_transitions - 2][transition] 
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
    # Make sure cost_array normalization doesnt' create zeros
    cost_array = cost_func(all_transition_inds)
    max_cost_dat.append(all_transition_inds[:,np.argmax(cost_array)])
    max_cost.append(np.max(cost_array))
    if cost_array.max():
        cost_array = cost_array / cost_array.max()
    #critical_vals = np.percentile(cost_array, 100-alpha, axis=-1)
    #update_bool = cost_array > critical_vals[:,np.newaxis]
    #update_inds = np.where(update_bool)
    #prior[update_inds[0], all_transition_inds[update_bool]] += 1
    inds = np.array(list(np.ndindex(all_transition_inds.shape)))
    broadcasted_cost = np.tile(cost_array, (len(inds) // len(cost_array)))
    prior[inds[:,0], all_transition_inds.flatten()] += broadcasted_cost 
plt.show()

max_cost_dat = np.array(max_cost_dat)
x = np.tile(np.arange(len(max_cost_dat))[:,np.newaxis], (1, max_cost_dat.shape[1]))
fig,ax = plt.subplots(1,2, sharey=True)
ax[0].scatter(max_cost_dat, x, color = 'grey', alpha = 0.3)
ax[1].plot(max_cost, x[:,0])
plt.show()

x = np.tile(np.arange(len(max_cost_dat))[:,np.newaxis], (1, max_cost_dat.shape[1]))
fig,ax = plt.subplots(1,2, sharey=True)
ax[0].scatter(max_cost_dat[np.argsort(max_cost)], x, color = 'grey', alpha = 0.3)
ax[1].plot(np.sort(max_cost), x[:,0])
plt.show()

fig,ax = plt.subplots(2,1)
im = ax[0].imshow(prior, aspect='auto');
fig.colorbar(im, ax=ax[0]);
im = ax[1].imshow(normalized_prior, aspect='auto');
fig.colorbar(im, ax=ax[1]);
plt.show()

############################################################
# Comparison with non-updating priors
############################################################
def return_transition_pos(
        n_states,
        max_len = 2000,
        min_state_dur = 50,
        n_samples = 1000
        ):

    grid = np.arange(0, max_len, min_state_dur)
    # Iteratively select transitions
    n_transitions = n_states - 1
    # We can select next transition using randint,
    # Need to know min and max
    # First max needs to allow "n_transitions" more transitions

    trans_max_list = [len(grid) - n_transitions + i for i in range(n_transitions)]

    transition_ind_list = []
    for i in range(n_transitions):
        if i == 0:
            temp_trans = np.random.randint(0, np.ones(n_samples)*trans_max_list[i])
        else:
            temp_trans = np.random.randint(
                    transition_ind_list[i-1] + 1,
                    trans_max_list[i])
        transition_ind_list.append(temp_trans)

    transition_inds_array = np.stack(transition_ind_list).T
    transition_array = grid[transition_inds_array]
    return transition_array

trans_points = return_transition_pos(
        n_states = n_states, 
        max_len = grid_len,
        min_state_dur = 1,
        n_samples = samples_per_batch*batches)

all_costs = [cost_func(x) for x in tqdm(trans_points)]

x = np.tile(np.arange(len(all_costs))[:,np.newaxis], (1, trans_points.shape[1]))
fig,ax = plt.subplots(1,2, sharey=True)
ax[0].scatter(trans_points[np.argsort(all_costs)], x, color = 'grey', alpha = 0.3)
ax[1].plot(np.sort(all_costs), x[:,0])
plt.show()

############################################################
## Compare with random sampling
plt.plot(np.sort(all_costs), np.arange(len(all_costs)), label = 'Random')
plt.plot(np.sort(max_cost), 
        np.arange(0,len(max_cost)*samples_per_batch, samples_per_batch), 
        label = 'Sequential')
plt.axvline(cost_func(transition_inds), 
        color = 'red', linestyle = '--', label = 'Max Cost')
plt.xlabel('Cost')
plt.ylabel('Sorted Indices')
plt.legend()
plt.show()

############################################################
# __  __       _ _   _      ____  _        _       
#|  \/  |_   _| | |_(_)    / ___|| |_ __ _| |_ ___ 
#| |\/| | | | | | __| |____\___ \| __/ _` | __/ _ \
#| |  | | |_| | | |_| |_____|__) | || (_| | ||  __/
#|_|  |_|\__,_|_|\__|_|    |____/ \__\__,_|\__\___|
#                                                  
############################################################

############################################################
# Sample transitions with weights
grid_len = 100
transition_inds = [
        [30,60],
        [20,40,80],
        [10,30,50,80],
        [5,30,45,60,75],
        [5,30,45,60,75, 90],
        ]
n_states = [len(x)+1 for x in transition_inds]#4
n_transitions = [x - 1 for x in n_states]

cost_per_transition = [np.zeros((x,grid_len)) for x in n_transitions]
for state_i,transition_list in enumerate(transition_inds):
    for i, val in enumerate(transition_list):
        cost_per_transition[state_i][i,val] = 1
state_multipliers = [0.1,1,5,1.5,20]
cost_per_transition = [gauss_filt1d(x, 3, axis=-1)*mult\
        for x,mult in zip(cost_per_transition, state_multipliers)]
fig,ax = plt.subplots(len(cost_per_transition), 1, sharey=True)
for dat, this_ax in zip(cost_per_transition, ax):
    #this_ax.imshow(dat, aspect = 'auto')
    this_ax.plot(dat.T)
plt.show()

grid = np.arange(grid_len)

# Cost function
# Note, a single draw will have a single cost
# Different transitions will not have different costs
def cost_func(transition_array):
    """transition_array : transitions x samples
    """
    # Evaluate costs
    this_cost_per_transition = cost_per_transition[len(transition_array)-2]
    cost_array = np.stack([this_cost[this_inds] for this_cost,this_inds \
            in zip(this_cost_per_transition, transition_array)] )
    return cost_array.sum(axis=0)

#transition_array = np.array(transition_inds[0]).T[:,np.newaxis]
max_cost_per_state = np.array([cost_func(np.array([x]).T) for x in transition_inds])



############################################################
## In batches
samples_per_batch = 100
batches = 500
base_transition_prior = [np.ones(x.shape)*10 for x in cost_per_transition]
base_states_prior = np.ones(len(n_states)) * 30
transition_prior = [np.ones(x.shape)*10 for x in cost_per_transition]
states_prior = np.ones(len(n_states)) * 30
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
                this_prior = transition_prior[this_n_transitions - 2][transition] 
                cut_prior = this_prior[:trans_max_list[transition]]
                this_trans = np.random.choice(
                        a = grid[:trans_max_list[transition]],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        )[0]
            else:
                this_prior = transition_prior[this_n_transitions - 2][transition] 
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
    # Make sure cost_array normalization doesnt' create zeros
    cost_array = cost_func(all_transition_inds)
    if cost_array.max():
        norm_cost_array = cost_array / cost_array.max()
    # Update transition prior
    inds = np.array(list(np.ndindex(all_transition_inds.shape)))
    broadcasted_cost = np.tile(norm_cost_array, (len(inds) // len(cost_array)))
    transition_prior[this_n_transitions-2][inds[:,0], all_transition_inds.flatten()] += broadcasted_cost 
    # Update states_prior
    # Update with max cost for now
    # Might change to single samples (without batch)
    states_prior[this_n_transitions - 2] += cost_array.mean()
#plt.show()
sub_transition_prior = [x-y for x,y in zip(transition_prior,base_transition_prior)]
sub_states_prior = states_prior - base_states_prior
fig,ax = plt.subplots(len(transition_inds) +1, 2)
ax[0,0].bar(n_states, sub_states_prior / sub_states_prior.sum())
ax[0,1].bar(n_states, max_cost_per_state.flatten() / max_cost_per_state.sum())
for num in range(len(sub_transition_prior)):
    temp_prior = sub_transition_prior[num]
    temp_normal_prior = temp_prior / temp_prior.sum()
    ax[num+1,0].plot(temp_normal_prior.T)
    ax[num+1,1].plot((cost_per_transition[num] / cost_per_transition[num].sum()).T)
plt.show()

############################################################
## In batches
## Normalize updates ACROSS states
samples_per_batch = 100
batches = 500
transition_prior = [np.ones(x.shape)*10 for x in cost_per_transition]
states_prior = np.ones(len(n_states)) * 30
for this_batch in trange(batches):
    state_list = np.random.choice(
            a = n_states,
            size = samples_per_batch,
            replace = True,
            p = states_prior / states_prior.sum()
            ) 
    #for s in range(samples_per_batch):
    all_transition_inds = []
    for this_n_states in state_list: 
        #this_n_states = np.random.choice(
        #        a = n_states,
        #        size = 1,
        #        replace = False,
        #        p = states_prior / states_prior.sum()
        #        )[0]
        this_n_transitions = this_n_states - 1
        trans_max_list = [len(grid) - this_n_transitions + i \
                for i in range(this_n_transitions)]
        this_transition_inds = []
        for transition in range(this_n_transitions):
            this_prior = transition_prior[this_n_transitions - 2][transition] 
            if transition == 0:
                cut_prior = this_prior[:trans_max_list[transition]]
                this_trans = np.random.choice(
                        a = grid[:trans_max_list[transition]],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        )[0]
            else:
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
    # Make sure cost_array normalization doesnt' create zeros
    cost_array = np.array([cost_func(x) for x in all_transition_inds])
    #cost_frame = pd.DataFrame(
    #        dict(
    #            states = state_list,
    #            costs = cost_array,
    #            )
    #        )
    #fig,ax = plt.subplots(1,2)
    #sns.barplot(
    #        data = cost_frame,
    #        x = 'states',
    #        y = 'costs',
    #        ax = ax[0],
    #        ci = None,
    #        )
    #ax[1].bar(n_states, state_multipliers)
    #plt.show()
    if cost_array.max():
        norm_cost_array = cost_array / cost_array.max()
    # Update transition prior
    #broadcasted_cost = np.tile(norm_cost_array, (len(inds) // len(cost_array)))
    for this_cost, this_inds in zip(norm_cost_array, all_transition_inds):
        state_inds = np.arange(len(this_inds))
        transition_prior[len(this_inds)-2][state_inds, this_inds] += [this_cost]*len(this_inds) 
        # Update states_prior
        # Update with max cost for now
        # Might change to single samples (without batch)
        states_prior[len(this_inds) - 2] += this_cost 
#plt.show()
fig,ax = plt.subplots(len(transition_inds) +1, 2)
ax[0,0].bar(n_states, states_prior / states_prior.sum())
ax[0,1].bar(n_states, max_cost_per_state.flatten() / max_cost_per_state.sum())
for num in range(len(transition_prior)):
    temp_prior = transition_prior[num]
    temp_normal_prior = temp_prior / temp_prior.sum()
    ax[num+1,0].plot(temp_normal_prior.T)
    ax[num+1,1].plot((cost_per_transition[num] / cost_per_transition[num].sum()).T)
plt.show()

############################################################
## Online updates 
total_samples = 50000
transition_prior = [np.ones(x.shape)*10 for x in cost_per_transition]
states_prior = np.ones(len(n_states)) * 10
#for this_batch in trange(batches):
for s in trange(total_samples):
    this_n_states = np.random.choice(
            a = n_states,
            size = 1,
            replace = False,
            p = states_prior / states_prior.sum()
            )[0]
    this_n_transitions = this_n_states - 1
    trans_max_list = [len(grid) - this_n_transitions + i \
            for i in range(this_n_transitions)]
    #all_transition_inds = []
    #for s in range(samples_per_batch):
    this_transition_inds = []
    for transition in range(this_n_transitions):
        if transition == 0:
            this_prior = transition_prior[this_n_transitions - 2][transition] 
            cut_prior = this_prior[:trans_max_list[transition]]
            this_trans = np.random.choice(
                    a = grid[:trans_max_list[transition]],
                    size = 1,
                    replace = False,
                    p = cut_prior / cut_prior.sum() 
                    )[0]
        else:
            this_prior = transition_prior[this_n_transitions - 2][transition] 
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
        #all_transition_inds.append(this_transition_inds)
    this_transition_inds = np.array(this_transition_inds)
    #all_transition_inds = np.stack(all_transition_inds).T
    # Make sure cost_array normalization doesnt' create zeros
    #cost_array = cost_func(all_transition_inds)
    cost_array = cost_func(this_transition_inds)
    #if cost_array.max():
    #    norm_cost_array = cost_array / cost_array.max()
    # Update transition prior
    #inds = np.array(list(np.ndindex(all_transition_inds.shape)))
    #broadcasted_cost = np.tile(norm_cost_array, (len(inds) // len(cost_array)))
    #transition_prior[this_n_transitions-2][inds[:,0], all_transition_inds.flatten()] += broadcasted_cost 
    inds = np.arange(len(this_transition_inds))
    broadcasted_cost = np.tile(cost_array, (len(inds))) 
    transition_prior[this_n_transitions-2][tuple((inds, this_transition_inds))] += broadcasted_cost 
    # Update states_prior
    # Update with max cost for now
    # Might change to single samples (without batch)
    #states_prior[this_n_transitions - 2] += cost_array.mean()
    states_prior[this_n_transitions - 2] += cost_array
#plt.show()
fig,ax = plt.subplots(len(transition_inds) +1, 2)
ax[0,0].bar(n_states, states_prior / states_prior.sum())
ax[0,1].bar(n_states, max_cost_per_state.flatten() / max_cost_per_state.sum())
for num in range(len(transition_prior)):
    temp_prior = transition_prior[num]
    temp_normal_prior = temp_prior / temp_prior.sum()
    ax[num+1,0].plot(temp_normal_prior.T)
    ax[num+1,1].plot((cost_per_transition[num] / cost_per_transition[num].sum()).T)
plt.show()
