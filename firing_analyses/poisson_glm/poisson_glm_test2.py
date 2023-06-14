"""
Simulate and infer models with different filters

1) Simple history filter
2) Stimulus and history filter
3) Stimulus, history, and coupling filter (2 neurons)
4) Stimulus, history, and coupling filter (n neurons)
"""

import numpy as np
import pylab as plt
from scipy.stats import zscore
import pandas as pd
import sys
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
import makeRaisedCosBasis as cb
import glm_tools as gt
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
import statsmodels.formula.api as smf
from pandas import DataFrame as df
from pandas import concat


############################################################
## History filter
############################################################
# Need to look specifically at exponential filters for
# visual comparison because things are weird in log space
hist_filter_len = 80
basis_kwargs = dict(
    n_basis = 20,
    basis = 'cos',
    basis_spread = 'linear',
    )
hist_filter = gt.generate_random_filter(hist_filter_len)
spike_data,prob = gt.generate_history_data(hist_filter, 10000)
res = gt.fit_history_glm(spike_data, hist_filter_len, **basis_kwargs)
lag_params = gt.process_glm_res(
        res, 
        hist_filter_len,
        **basis_kwargs, 
        param_key = 'hist')
fig, ax = plt.subplots(4, 1, figsize = (10, 5), sharey = True)
ax[0].plot(spike_data)
ax[0].set_title(f'Mean firing rate: {np.mean(spike_data)*1000}')
ax[1].plot(np.exp(prob))
ax[1].set_title('Prob')
ax[2].plot(np.exp(hist_filter))
ax[2].set_title('True filter (exp)')
ax[3].plot(np.exp(lag_params))
ax[3].set_title('Estimated filter (exp)')
plt.show()

############################################################
## Stim filter 
############################################################
n = 10000 
basis_kwargs = dict(
    n_basis = 10,
    basis = 'cos',
    basis_spread = 'linear',
    )
stim_filter_len = 100
stim_filter = gt.generate_stim_filter(stim_filter_len)
spike_data, prob, stim_data = \
        gt.generate_stim_data(
                n = n, 
                stim_filter = stim_filter,
                stim_count = 30
)
res,pred = gt.fit_stim_glm(
        spike_data, 
        stim_data,
        stim_filter_len,
        **basis_kwargs
        )

stim_params = gt.process_glm_res(
        res, 
        stim_filter_len,
        **basis_kwargs,
        param_key = 'stim')

fig, ax = plt.subplots(4, 1, figsize = (5, 10), sharey = False)
ax[0].plot(spike_data, label = 'spikes', linewidth = 0.5)
ax[0].plot(stim_data, label = 'stim', linewidth = 3)
ax[0].legend()
ax[0].set_title(f'Mean firing rate: {np.mean(spike_data)*1000}')
ax[1].plot(np.exp(prob))
ax[1].set_title('Probability (exp)')
ax[2].plot(np.exp(stim_filter))
ax[2].set_title('Stim filter (exp)')
ax[3].plot(np.exp(stim_params))
ax[3].set_title('Estimated stim filter (exp)')
plt.tight_layout()
plt.show()

# Plot predicted vs actual
fig,ax = plt.subplots(1,1, sharex=True, sharey=True)
ax.plot(spike_data, label = 'True spikes')
ax.plot(pred, label = 'Predicted prob')
plt.legend()
plt.show()

# Plot actual and predicted PSTHs
psth_window = [-200,200]
stim_inds = np.where(stim_data)[0]
wanted_psth_inds = [(x+psth_window[0], x+psth_window[1]) \
        for x in stim_inds]
wanted_psth_inds = [x for x in wanted_psth_inds if \
        all(np.array(x) > 0) and all(np.array(x) < n)]

spike_trains = np.stack([spike_data[x[0]:x[1]] for x in wanted_psth_inds])
pred_trains = np.stack([pred[x[0]-stim_filter_len:x[1]-stim_filter_len] for x in wanted_psth_inds])

kern_len = 20
kern = np.ones(kern_len)/kern_len
firing_rates = np.stack([np.convolve(x, kern, mode = 'same') for x in spike_trains])

fig,ax = plt.subplots(3,1, sharex=True, sharey=True)
ax[0].plot(spike_trains.T, color = 'k', alpha = 0.5)
ax[1].plot(firing_rates.T, color = 'k', alpha = 0.5)
ax[2].plot(pred_trains.T, color = 'k', alpha = 0.5)
plt.show()


############################################################
## Stim + History Filter 
############################################################
n = 10000
basis_kwargs = dict(
    n_basis = 30,
    basis = 'cos',
    basis_spread = 'linear',
    )
hist_filter_len = 80
stim_filter_len = 100

hist_filter = gt.generate_random_filter(hist_filter_len)
stim_filter = gt.generate_stim_filter(stim_filter_len)
spike_data, prob, stim_data = \
        gt.generate_stim_history_data(
                hist_filter, 
                stim_filter,
                n = 10000, 
                stim_count = 10)

res,pred = gt.fit_stim_history_glm(
        spike_data, 
        stim_data,
        hist_filter_len, 
        stim_filter_len,
        **basis_kwargs
        )

fig, ax = plt.subplots(3,1, sharey = True, sharex = True)
ax[0].plot(pred)
ax[0].set_title('Predicted')
ax[1].plot(np.exp(prob))
ax[1].set_title('Probability (exp)')
ax[2].plot(spike_data)
ax[2].set_title('Spikes')
plt.show()

hist_params = gt.process_glm_res(
        res, 
        hist_filter_len,
        **basis_kwargs,
        param_key = 'hist')
stim_params = gt.process_glm_res(
        res, 
        stim_filter_len,
        **basis_kwargs,
        param_key = 'stim')

fig, ax = plt.subplots(6, 1, figsize = (5, 10), sharey = False)
ax[0].plot(spike_data, label = 'spikes', linewidth = 0.5)
ax[0].plot(stim_data, label = 'stim', linewidth = 3)
ax[0].legend()
ax[0].set_title(f'Mean firing rate: {np.mean(data)*1000}')
ax[1].plot(np.exp(prob))
ax[1].set_title('Probability')
ax[2].plot(np.exp(hist_filter))
ax[2].set_title('True filter (exp)')
ax[3].plot(np.exp(hist_params))
ax[3].set_title('Estimated filter (exp)')
ax[4].plot(np.exp(stim_filter))
ax[4].set_title('Stim filter (exp)')
ax[5].plot(np.exp(stim_params))
ax[5].set_title('Estimated stim filter (exp)')
plt.tight_layout()
plt.show()

# Plot predicted vs actual
fig,ax = plt.subplots(1,1, sharex=True, sharey=True)
ax.plot(spike_data, label = 'True spikes')
ax.plot(pred, label = 'Predicted prob')
plt.legend()
plt.show()

# Plot actual and predicted PSTHs
psth_window = [-200,200]
stim_inds = np.where(stim_data)[0]
wanted_psth_inds = [(x+psth_window[0], x+psth_window[1]) \
        for x in stim_inds]
wanted_psth_inds = [x for x in wanted_psth_inds if \
        all(np.array(x) > 0) and all(np.array(x) < n)]

spike_trains = np.stack([spike_data[x[0]:x[1]] for x in wanted_psth_inds])
pred_trains = np.stack([pred[x[0]-stim_filter_len:x[1]-stim_filter_len] for x in wanted_psth_inds])

kern_len = 20
kern = np.ones(kern_len)/kern_len
firing_rates = np.stack([np.convolve(x, kern, mode = 'same') for x in spike_trains])

fig,ax = plt.subplots(3,1, sharex=True, sharey=True)
ax[0].plot(spike_trains.T, color = 'k', alpha = 0.5)
ax[1].plot(firing_rates.T, color = 'k', alpha = 0.5)
ax[2].plot(pred_trains.T, color = 'k', alpha = 0.5)
plt.show()

############################################################
# Coupling filter only
############################################################
n = 10000
basis_kwargs = dict(
    n_basis = 10,
    basis = 'cos',
    basis_spread = 'linear',
    )
hist_filter_len = 40
coupling_filter_len = 40
n_coupled_neurons = 5
# Note: This hist filter will be to generate data for OTHER neuron
hist_filter_list = [gt.generate_random_filter(hist_filter_len) \
        for _ in range(n_coupled_neurons)]
coupling_filter_list = [gt.generate_random_filter(coupling_filter_len) \
        for _ in range(n_coupled_neurons)]
# Divide coupling filters by 2 to make sure they are not too large
coupling_filter_list = [cf/n_coupled_neurons for cf in coupling_filter_list]

spike_data, prob, coupling_probs, coupled_spikes = \
        gt.generate_coupling_data(
                hist_filter_list,
                coupling_filter_list,
                n = n,
                )

res,pred = gt.fit_coupled_glm(
        spike_data, 
        coupled_spikes,
        coupling_filter_len = coupling_filter_len,
        **basis_kwargs
        ) 
coupling_params_stack = np.stack(
    [gt.process_glm_res(
        res, 
        coupling_filter_len,
        **basis_kwargs,
        param_key = f'lag_{i}') \
                for i in range(n_coupled_neurons)]
    )

fig, ax = plt.subplots(n_coupled_neurons,1, sharex=True, sharey=False)
for i in range(n_coupled_neurons):
    ax[i].plot(np.exp(coupling_filter_list[i]), label = 'true')
    ax[i].plot(np.exp(coupling_params_stack[i]), label = 'estimated')
ax[-1].legend()
plt.show()

# Plot actual spikes and predicted prob
fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
ax.plot(spike_data)
ax.set_title('Spikes')
ax.plot(pred)
ax.set_title('Predicted')
fig.suptitle(f'Mean firing rate: {np.round(np.mean(spike_data)*1000,2)}')
plt.show()

############################################################
# Coupled history 
############################################################
n = 10000
basis_kwargs = dict(
    n_basis = 10,
    basis = 'cos',
    basis_spread = 'linear',
    )
hist_filter_len = 80
coupling_filter_len = 80
n_coupled_neurons = 5

# Note: This hist filter will be to generate data for OTHER neuron
hist_filter = gt.generate_random_filter(hist_filter_len)
hist_filter_list = [gt.generate_random_filter(hist_filter_len) \
        for _ in range(n_coupled_neurons)]
coupling_filter_list = [gt.generate_random_filter(coupling_filter_len) \
        for _ in range(n_coupled_neurons)]
# Divide coupling filters by 2 to make sure they are not too large
coupling_filter_list = [cf/n_coupled_neurons for cf in coupling_filter_list]

# Divide both by two to maintain firing rates
coupling_filter_list = [x/2 for x in coupling_filter_list]
hist_filter = hist_filter / 2
#hist_filter = np.zeros(hist_filter_len)

spike_data, prob, coupling_probs, coupled_spikes = \
        gt.generate_history_coupling_data(
                hist_filter,
                hist_filter_list,
                coupling_filter_list,
                n = n,
                )

# Chop off up to max filter len
max_filter_len = np.max([hist_filter_len, coupling_filter_len])
spike_data = spike_data[:-max_filter_len]
coupled_spikes = coupled_spikes[:,:-max_filter_len]

res,pred = gt.fit_history_coupled_glm(
        spike_data, 
        coupled_spikes,
        hist_filter_len = hist_filter_len,
        coupling_filter_len = coupling_filter_len,
        **basis_kwargs
        ) 
coupling_params_stack = np.stack(
    [gt.process_glm_res(
        res, 
        coupling_filter_len,
        **basis_kwargs,
        param_key = f'lag_{i}') \
                for i in range(n_coupled_neurons)]
    )
hist_params_stack = gt.process_glm_res(
        res, 
        hist_filter_len,
        **basis_kwargs, 
        param_key = 'hist')

fig, ax = plt.subplots(n_coupled_neurons + 1,1, sharex=True, sharey=True)
for i in range(n_coupled_neurons):
    ax[i].plot(np.exp(coupling_filter_list[i]), label = 'true')
    ax[i].plot(np.exp(coupling_params_stack[i]), label = 'estimated')
ax[-1].plot(np.exp(hist_filter), label = 'True')
ax[-1].plot(np.exp(hist_params_stack), label = 'Estimated')
ax[-1].legend()
plt.show()

# Plot actual spikes and predicted prob
fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
ax.plot(spike_data)
ax.set_title('Spikes')
ax.plot(pred)
ax.set_title('Predicted')
fig.suptitle(f'Mean firing rate: {np.round(np.mean(spike_data)*1000,2)}')
plt.show()

############################################################
# Coupled-stim-history 
############################################################
n = 10000
basis_kwargs = dict(
    n_basis = 10,
    basis = 'cos',
    basis_spread = 'linear',
    )
stim_filter_len = 500
stim_count = 10
hist_filter_len = 80
coupling_filter_len = 80
n_coupled_neurons = 5

# Note: This hist filter will be to generate data for OTHER neuron
hist_filter = gt.generate_random_filter(hist_filter_len)
hist_filter_list = [gt.generate_random_filter(hist_filter_len) \
        for _ in range(n_coupled_neurons)]
coupling_filter_list = [gt.generate_random_filter(coupling_filter_len) \
        for _ in range(n_coupled_neurons)]
# Divide coupling filters by 2 to make sure they are not too large
coupling_filter_list = [cf/n_coupled_neurons for cf in coupling_filter_list]

# Divide both by two to maintain firing rates
#coupling_filter_list = [x/2 for x in coupling_filter_list]
coupling_filter_list = [np.zeros(coupling_filter_len) for i in range(n_coupled_neurons)]
hist_filter = hist_filter / 2
#hist_filter = np.zeros(hist_filter_len)
stim_filter = gt.generate_stim_filter(stim_filter_len)

spike_data, prob, coupling_probs, coupled_spikes, stim_vec = \
        gt.generate_stim_history_coupling_data(
                hist_filter,
                hist_filter_list,
                coupling_filter_list,
                stim_filter = stim_filter,
                stim_count = stim_count,
                n = n,
                )

# Chop off up to max filter len
max_filter_len = np.max([hist_filter_len, coupling_filter_len, stim_filter_len])
spike_data = spike_data[:-max_filter_len]
coupled_spikes = coupled_spikes[:,:-max_filter_len]

res,pred = gt.fit_stim_history_coupled_glm(
        spike_data, 
        coupled_spikes,
        stim_vec,
        hist_filter_len = hist_filter_len,
        coupling_filter_len = coupling_filter_len,
        stim_filter_len= stim_filter_len,
        **basis_kwargs
        ) 
coupling_params_stack = np.stack(
    [gt.process_glm_res(
        res, 
        coupling_filter_len,
        **basis_kwargs,
        param_key = f'lag_{i}') \
                for i in range(n_coupled_neurons)]
    )
hist_params_stack = gt.process_glm_res(
        res, 
        hist_filter_len,
        **basis_kwargs, 
        param_key = 'hist')
stim_params_stack = gt.process_glm_res(
        res, 
        hist_filter_len,
        **basis_kwargs, 
        param_key = 'stim')

fig, ax = plt.subplots(n_coupled_neurons + 2,1, sharex=True, sharey=False)
for i in range(n_coupled_neurons):
    ax[i].plot(np.exp(coupling_filter_list[i]), label = 'true')
    ax[i].plot(np.exp(coupling_params_stack[i]), label = 'estimated')
ax[-2].plot(np.exp(hist_filter), label = 'True')
ax[-2].plot(np.exp(hist_params_stack), label = 'Estimated')
stim_filter = gt.generate_stim_filter(filter_len = stim_filter_len)
ax[-1].plot(np.exp(stim_filter), label = 'True')
ax[-1].plot(np.exp(stim_params_stack), label = 'Estimated')
ax[-1].legend()
plt.show()

# Plot actual spikes and predicted prob
fig, ax = plt.subplots(1,1, sharex=True, sharey=True)
ax.plot(spike_data)
ax.set_title('Spikes')
ax.plot(pred)
ax.set_title('Predicted')
fig.suptitle(f'Mean firing rate: {np.round(np.mean(spike_data)*1000,2)}')
plt.show()

# Plot actual and predicted PSTHs
psth_window = [-200,200]
stim_inds = np.where(stim_vec)[0]
wanted_psth_inds = [(x+psth_window[0], x+psth_window[1]) \
        for x in stim_inds]
wanted_psth_inds = [x for x in wanted_psth_inds if \
        all(np.array(x) > 0) and all(np.array(x) < n)]

spike_trains = np.stack([spike_data[x[0]:x[1]] for x in wanted_psth_inds])
pred_trains = np.stack([pred[x[0]-stim_filter_len:x[1]-stim_filter_len] for x in wanted_psth_inds])

kern_len = 50
kern = np.ones(kern_len)/kern_len
firing_rates = np.stack([np.convolve(x, kern, mode = 'same') for x in spike_trains])

imshow_kwargs = dict(aspect = 'auto',
                     interpolation = 'nearest',
                     vmin = 0,
                     vmax = 0.2)
fig,ax = plt.subplots(1,2, sharex=True, sharey=False)
#ax[0].plot(spike_trains.T, color = 'k', alpha = 0.5)
ax[0].imshow(firing_rates, **imshow_kwargs)
ax[1].imshow(pred_trains, **imshow_kwargs)
plt.show()

