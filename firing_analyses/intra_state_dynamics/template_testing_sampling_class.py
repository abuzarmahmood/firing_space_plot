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
## Define class
############################################################

class sequential_template_regression():
    def __init__(
            self,
            cost_function,
            firing_array,
            max_len,
            n_states,
            batches = 5000,
            samples_per_batch = 100,
            ):

        # Save inputs
        self.cost_func = cost_function
        self.firing_array = firing_array
        self.max_len = max_len
        self.n_states = n_states
        self.batches = batches
        self.samples_per_batch = samples_per_batch

        # Derivative parameters
        self.n_transitions = n_states - 1
        self.min_trans = self.n_transitions.min()
        self.grid = np.arange(max_len)
        self.base_transition_prior = [np.ones((x-1,max_len))*10 for x in n_states]
        self.base_states_prior = np.ones(len(n_states)) * 30
        self.transition_prior = [np.ones((x-1,max_len))*10 for x in n_states]
        self.states_prior = np.ones(len(n_states)) * 30

        self.cost_per_state = [[np.random.random()*0.1] for i in range(len(n_states))]

        # History
        self.state_history = []

    def state_step(
            self,
            prior = 'updated'
            ):
        if prior == 'updated':
            states_prior = self.states_prior
        elif prior == 'base':
            states_prior = self.base_states_prior
        else:
            state_prior = prior
        this_n_states = np.random.choice(
                a = self.n_states,
                size = 1,
                replace = False,
                p = states_prior / states_prior.sum()
                )[0]
        this_n_transitions = this_n_states - 1
        trans_max_list = [len(self.grid) - this_n_transitions + i \
                for i in range(this_n_transitions)]
        return this_n_states, this_n_transitions, trans_max_list

    def transition_step(self,
            this_n_transitions,
            trans_max_list,
            prior = 'updated', 
            ):
        if prior == 'updated':
            transition_prior = self.transition_prior
        elif prior == 'base':
            transition_prior = self.base_transition_prior
        else:
            transition_prior = prior
        this_transition_inds = []
        for transition in range(this_n_transitions):
            if transition == 0:
                this_prior = transition_prior[this_n_transitions - self.min_trans][transition] 
                cut_prior = this_prior[:trans_max_list[transition]]
                this_trans = np.random.choice(
                        a = self.grid[:trans_max_list[transition]],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        )[0]
            else:
                this_prior = transition_prior[this_n_transitions - self.min_trans][transition] 
                cut_min = this_transition_inds[transition-1] + 1
                cut_max = trans_max_list[transition]
                cut_prior = this_prior[cut_min:cut_max]
                this_trans = np.random.choice(
                        a = self.grid[cut_min:cut_max],
                        size = 1,
                        replace = False,
                        p = cut_prior / cut_prior.sum() 
                        ).flatten()[0]
            this_transition_inds.append(this_trans)
        return this_transition_inds

    def transition_batch(
            self,
            state_prior = 'updated',
            transition_prior = 'updated',
            n_samples = None,
            ):
        this_n_states, this_n_transitions, trans_max_list = self.state_step(prior = state_prior)
        self.state_history.append(this_n_states)
        if n_samples is None:
            n_samples = self.samples_per_batch
        all_transition_inds = [
                self.transition_step(
                   this_n_transitions,
                   trans_max_list,
                   prior = state_prior
                    )
               for s in range(n_samples) 
                ]
        return np.stack(all_transition_inds).T

    def calculate_cost(
            self,
            all_transition_inds,
            ):
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
        return cost_array, norm_cost_array

    def update_priors(self,
            all_transition_inds,
            cost_array,
            norm_cost_array):
        inds = np.array(list(np.ndindex(all_transition_inds.shape)))
        broadcasted_cost = np.tile(norm_cost_array, (len(inds) // len(cost_array)))
        self.transition_prior[all_transition_inds.shape[0]-self.min_trans][inds[:,0], 
                all_transition_inds.flatten()] += broadcasted_cost 
        # Update states_prior
        # Update with max cost for now
        # Might change to single samples (without batch)
        mean_state_costs = np.array([np.mean(x) for x in self.cost_per_state])
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
        self.cost_per_state[all_transition_inds.shape[0] - self.min_trans].append(current_mean_cost)
        # Update with scaled max cost
        self.states_prior[all_transition_inds.shape[0] - self.min_trans] += update_state_cost 

    def run_fit(self):
        for this_batch in trange(self.batches):
            all_transition_inds = self.transition_batch()
            cost_array, norm_cost_array = self.calculate_cost(all_transition_inds)
            self.update_priors(all_transition_inds, cost_array, norm_cost_array)

    def sample_priors(
            self, 
            n_new_samples = 5000,
            state_prior = 'updated', 
            transition_prior = 'updated' 
            ):

        all_transition_inds = []
        for i in trange(n_new_samples):
            this_transition_ind = self.transition_batch(
                    state_prior = state_prior,
                    transition_prior = transition_prior,
                    n_samples = 1,
                    )
            all_transition_inds.append(this_transition_ind)
        state_count = np.array([len(x)+1 for x in all_transition_inds])
        outs = [self.calculate_cost(x) for x in tqdm(all_transition_inds)]
        cost_array, norm_cost_array = list(zip(*outs)) 
        cost_array = np.stack(cost_array).flatten()
        return all_transition_inds, state_count, cost_array

    def return_top_nth(
            self,
            nth,
            all_transition_inds,
            state_count,
            cost_array
            ):
        cost_critical_val = np.percentile(cost_array, 100-nth)
        cost_inds = np.where(cost_array > cost_critical_val)[0]
        selected_cost = cost_array[cost_inds]
        selected_states = state_count[cost_inds]
        selected_inds = [all_transition_inds[i] for i in cost_inds]
        return selected_inds, selected_states, selected_cost

    def return_sample_plots(
            self,
            all_transition_inds,
            state_count,
            cost_array
            ):

        new_states = np.array(state_count).copy() 
        inds = np.argsort(new_states)
        new_states = new_states[inds]
        new_costs = cost_array[inds]
        new_transitions = [all_transition_inds[x] for x in inds]

        max_cost_ind = np.argmax(new_costs)
        max_cost = new_costs[max_cost_ind]
        max_state = new_states[max_cost_ind]

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
            ax[num+1, 0].set_xlim([0, len(self.grid)])
            ax[num+1, 1].plot(sorted_costs, x[0])
            ax[num+1, 1].axvline(max_cost, color = 'red', linestyle = '--')
            if state == max_state:
                ax[num+1,1].axvline(max_cost, color = 'k', linestyle = '--')
                ax[num+1,1].text(max_cost, 0, str(np.round(max_cost,3)), 
                    va = 'bottom', rotation = 'vertical',
                    ha = 'right')
        plt.tight_layout()
        #plt.show()
        #fig.savefig(os.path.join(plot_dir,f'sampling_updated_prior_{states}_state.png'))
        #plt.close(fig)
        return fig,ax

############################################################
## Simulation
############################################################
max_len = 300
min_dur = 40
def cost_func(firing_array, trans_points):
    basis_funcs = return_template_mat(trans_points, max_len) 
    basis_funcs = basis_funcs / norm(basis_funcs,axis=-1)[:,np.newaxis] 
    return np.mean(return_similarities(basis_funcs,firing_array))

states = 4
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
## Run Fit 
############################################################
n_states = np.arange(2,8)
test = sequential_template_regression(
       cost_func,
       firing_array,
       max_len,
       n_states,
       batches = 5000
        )
test.run_fit()

new_samples = 5000
updated_prior_samples = test.sample_priors(n_new_samples = new_samples)
base_prior_samples = test.sample_priors(
        state_prior = 'base', transition_prior = 'base',
        n_new_samples = new_samples
        )
fig,ax = test.return_sample_plots(*updated_prior_samples)
fig.suptitle('Updated')
plt.tight_layout()
fig,ax = test.return_sample_plots(*base_prior_samples)
fig.suptitle('Base')
plt.tight_layout()
plt.show()

nth = 5
top_updated_samples = test.return_top_nth(nth, *updated_prior_samples)
top_base_samples = test.return_top_nth(nth, *base_prior_samples)

fig,ax = test.return_sample_plots(*top_updated_samples)
fig.suptitle('Updated')
plt.tight_layout()
fig,ax = test.return_sample_plots(*top_base_samples)
fig.suptitle('Base')
plt.tight_layout()
plt.show()
