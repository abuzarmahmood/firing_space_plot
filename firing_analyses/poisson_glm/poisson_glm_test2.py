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
import os
plot_dir=  '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/tests/plots'
from tqdm import tqdm, trange
from seaborn import sns


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
n = 20000
basis_kwargs = dict(
    n_basis = 10,
    basis = 'cos',
    basis_spread = 'linear',
    )
stim_filter_len = 500
stim_count = 20
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

# Chop off up to max filter len because something weird happens at the end
max_filter_len = np.max([hist_filter_len, coupling_filter_len, stim_filter_len])
spike_data = spike_data[:-max_filter_len]
coupled_spikes = coupled_spikes[:,:-max_filter_len]
stim_vec = stim_vec[:-max_filter_len]

res,pred = gt.fit_stim_history_coupled_glm(
        spike_data, 
        coupled_spikes,
        stim_vec,
        hist_filter_len = hist_filter_len,
        coupling_filter_len = coupling_filter_len,
        stim_filter_len= stim_filter_len,
        regularized=False,
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

fig, ax = plt.subplots(n_coupled_neurons + 2,1, sharex=True, sharey=False,
                       figsize = (5,10))
for i in range(n_coupled_neurons):
    ax[i].plot(np.exp(coupling_filter_list[i]), label = 'true')
    ax[i].plot(np.exp(coupling_params_stack[i]), label = 'estimated')
    ax[i].set_ylabel('Coupling')
ax[-2].plot(np.exp(hist_filter), label = 'True')
ax[-2].plot(np.exp(hist_params_stack), label = 'Estimated')
ax[-2].set_ylabel('History')
stim_filter = gt.generate_stim_filter(filter_len = stim_filter_len)
ax[-1].plot(np.exp(stim_filter), label = 'True')
ax[-1].plot(np.exp(stim_params_stack), label = 'Estimated')
ax[-1].set_ylabel('Stimulus')
ax[-1].legend()
fig.suptitle('Filter comparison')
plt.tight_layout()
#fig.savefig(os.path.join(plot_dir,'full_model_filters.png'),
#            dpi = 300, bbox_inches = 'tight')
#plt.close(fig)
plt.show()

# Plot actual spikes and predicted prob
fig, ax = plt.subplots(1,1, sharex=True, sharey=True,
                       figsize = (10,3))
ax.plot(spike_data, alpha = 0.5)
ax.set_title('Spikes')
ax.plot(pred)
ax.set_title('Predicted')
fig.suptitle(f'Mean firing rate: {np.round(np.mean(spike_data)*1000,2)}')
plt.tight_layout()
#fig.savefig(os.path.join(plot_dir,'actual_vs_pred_prob.png'),
#            dpi = 300, bbox_inches = 'tight')
#plt.close(fig)
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

kern_len = 25
kern = np.ones(kern_len)/kern_len
firing_rates = np.stack([np.convolve(x, kern, mode = 'same') for x in spike_trains])
smooth_pred_rates = np.stack([np.convolve(x, kern, mode = 'same') for x in pred_trains])

imshow_kwargs = dict(aspect = 'auto',
                     interpolation = 'nearest',
                     vmin = 0,
                     vmax = 0.3)
fig,ax = plt.subplots(1,2, sharex=True, sharey=False)
#ax[0].plot(spike_trains.T, color = 'k', alpha = 0.5)
im = ax[0].imshow(firing_rates, **imshow_kwargs)
ax[0].set_title('Actual Rates')
plt.colorbar(im, ax = ax[0])
im = ax[1].imshow(smooth_pred_rates, **imshow_kwargs)
ax[1].set_title('Predicted Rates')
plt.colorbar(im, ax = ax[1])
plt.suptitle(f'PSTH Window: {psth_window}, smoothing kern len: {kern_len}')
plt.tight_layout()
#fig.savefig(os.path.join(plot_dir,'actual_vs_pred_trial_img.png'),
#            dpi = 300, bbox_inches = 'tight')
#plt.close(fig)
plt.show()

fig,ax = plt.subplots(len(firing_rates), 1, sharex=True, sharey=False,
                      figsize=(5,15))
for f, p, this_ax in zip(firing_rates, smooth_pred_rates, ax):
    this_ax.plot(f, label = 'Firing')
    this_ax.plot(p, label = 'Pred')
this_ax.legend()
plt.suptitle(f'PSTH Window: {psth_window}, smoothing kern len: {kern_len}')
plt.tight_layout()
#fig.savefig(os.path.join(plot_dir,'actual_vs_pred_trial_lines.png'),
#            dpi = 300, bbox_inches = 'tight')
#plt.close(fig)
plt.show()


# Plot p-values
pval_frame = pd.DataFrame(res.pvalues)
pval_frame = pval_frame.rename(columns = {0 : 'pval'})
pval_frame['pval'] += 1e-20
groups = [x.split('_')[0] if '_' in x else x for x in pval_frame.index]
pval_frame['group'] = groups
grouped_pval_frame = [x[1] for x in list(pval_frame.groupby('group'))]

thresh = 0.001
fig, ax = plt.subplots(len(grouped_pval_frame), 1, sharey=False,
                       figsize = (5,10))
ylabels = ['Intercept', 'Coupling','History','Stim']
for this_label, this_ax, this_dat in zip(ylabels, ax, grouped_pval_frame):
    sig_inds = this_dat.pval < thresh
    not_sig_inds = this_dat.pval > thresh
    this_dat = this_dat.sort_index()
    this_ax.scatter(this_dat.index[sig_inds], 
                    -np.log10(this_dat.pval[sig_inds]),
                    label = 'Sig')
    this_ax.scatter(this_dat.index[not_sig_inds], 
                    -np.log10(this_dat.pval[not_sig_inds]),
                    label = 'Not Sig')
    this_ax.axhline(-np.log10(thresh), color = 'red', label = str(thresh))
    this_ax.set_ylabel(this_label + ' (log10)')
this_ax.legend()
plt.suptitle('P values per parameter')
plt.tight_layout()
#fig.savefig(os.path.join(plot_dir,'p_value_comparison.png'),
#            dpi = 300, bbox_inches = 'tight')
#plt.close(fig)
plt.show()

############################################################
############################################################

# Cross validated prediction on trial-matched vs shuffled-data
# 1) Shuffle trials
# 2) Shuffle timebins across trials

# Generate data frame and calculate cross validated prediction
# Repeat for shuffled datasets
# If model has learned something useful, then shuffling dv-iv either
# across trials or circularly per time bin should diminish log likelihood

# Note: Shuffling needs to happen at the design_matrix level since
# history will always be a strong predictor and history can't be 
# shuffled without destroying the actual spike trains

data_frame =  gt.gen_data_frame(
        spike_data, 
        coupled_spikes,
        stim_vec,
        )

actual_input_dat = data_frame.copy()
actual_design_mat = gt.dataframe_to_design_mat(actual_input_dat)
# Generate train test splits
actual_train_dat, actual_test_dat = gt.return_train_test_split(actual_design_mat)

# Fit model to actual data
res,pred = gt.fit_stim_history_coupled_glm(
        glmdata = actual_train_dat,
        hist_filter_len = hist_filter_len,
        coupling_filter_len = coupling_filter_len,
        stim_filter_len= stim_filter_len,
        regularized=False,
        **basis_kwargs
        )

# Test fit
actual_test_pred = res.predict(actual_test_dat)
actual_test_ll = gt.poisson_ll(actual_test_pred, actual_test_dat['spikes'].values)
actual_test_ll = np.round(actual_test_ll, 2)

# Generate shuffles and repeat testing
# Note: No need to refit as we're simply showing that destroying different
# parts of the predictors destroys the model's ability to predict actual data
# i.e. model has learned TRIAL-SPECIFIC features
trial_sh_design_mat = gt.gen_trial_shuffle(actual_design_mat)
circ_sh_design_mat = gt.gen_circular_shuffle(actual_design_mat)
rand_sh_design_mat = gt.gen_random_shuffle(actual_design_mat)
trial_sh_train_dat, trial_sh_test_dat = gt.return_train_test_split(trial_sh_design_mat)
circ_sh_train_dat, circ_sh_test_dat = gt.return_train_test_split(circ_sh_design_mat)
rand_sh_train_dat, rand_sh_test_dat = gt.return_train_test_split(rand_sh_design_mat)

trial_sh_test_pred = res.predict(trial_sh_test_dat)
trial_sh_test_ll = gt.poisson_ll(trial_sh_test_pred, actual_test_dat['spikes'].values)
trial_sh_test_ll = np.round(trial_sh_test_ll, 2)

circ_sh_test_pred = res.predict(circ_sh_test_dat)
circ_sh_test_ll = gt.poisson_ll(circ_sh_test_pred, actual_test_dat['spikes'].values)
circ_sh_test_ll = np.round(circ_sh_test_ll, 2)

rand_sh_test_pred = res.predict(rand_sh_test_dat)
rand_sh_test_ll = gt.poisson_ll(rand_sh_test_pred, actual_test_dat['spikes'].values)
rand_sh_test_ll = np.round(rand_sh_test_ll, 2)

# Plot all 3 conditions against actual data
fig, ax = plt.subplots(4,1, figsize=(10,10), sharex=True, sharey=True)
ax[0].plot(actual_test_dat['spikes'],  label='Actual')
ax[0].plot(actual_test_pred,  label='Predicted')
ax[0].set_title('Actual, LL: {}'.format(actual_test_ll))
ax[0].legend()
ax[1].plot(trial_sh_test_dat['spikes'],  label='Actual')
ax[1].plot(trial_sh_test_pred,  label='Predicted')
ax[1].set_title('Trial shuffled, LL: {}'.format(trial_sh_test_ll))
ax[1].legend()
ax[2].plot(circ_sh_test_dat['spikes'],  label='Actual')
ax[2].plot(circ_sh_test_pred,  label='Predicted')
ax[2].set_title('Circular shuffled, LL: {}'.format(circ_sh_test_ll))
ax[2].legend()
ax[3].plot(rand_sh_test_dat['spikes'],  label='Actual')
ax[3].plot(rand_sh_test_pred,  label='Predicted')
ax[3].set_title('Random shuffled, LL: {}'.format(rand_sh_test_ll))
ax[3].legend()
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,'cross_val_shuffle_LL.png'),
            dpi = 300, bbox_inches = 'tight')
plt.close(fig)
#plt.show()

############################################################
############################################################
# Convert to testing machine
# Number of fits on actual data (expensive)
n_fits = 10
n_max_tries = 20
# Number of shuffles tested against each fit
n_shuffles_per_fit = 50

data_frame =  gt.gen_data_frame(
        spike_data, 
        coupled_spikes,
        stim_vec,
        max_filter_len = max_filter_len,
        )

# Reload glm_tools
import importlib
importlib.reload(gt)

fit_outs = []
for i in trange(n_max_tries):
    if len(fit_outs) < n_fits:
        try:
            outs = gt.gen_actual_fit(
                data_frame,
                hist_filter_len = hist_filter_len,
                coupling_filter_len = coupling_filter_len,
                stim_filter_len= stim_filter_len,
                basis_kwargs = basis_kwargs,
                ) 
            fit_outs.append(outs)
        except:
            print('Failed fit')
    else:
        print('Finished fitting')
        break

fit_list = [fit_out[0] for fit_out in fit_outs]
actual_design_mat = fit_outs[0][1]

ll_names = ['actual','trial_sh','circ_sh','rand_sh']
ll_outs = [gt.calc_loglikelihood(actual_design_mat, res) for res in tqdm(fit_list)]

ll_frame = pd.DataFrame(ll_outs, columns=ll_names)

plt.boxplot(ll_frame)
plt.ylim([-4000,0])
plt.xticks(np.arange(1,1+len(ll_names)),labels = ll_names)
plt.ylabel('Log Likelihood')
plt.show()
