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
from sklearn.model_selection import train_test_split


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

# Cross validated prediction on trial-matched vs shuffled-data
# 1) Shuffle trials
# 2) Shuffle timebins across trials
def gen_data_frame(
        spike_data, 
        coupled_spikes,
        stim_vec,
        ):
    stacked_data = np.concatenate([
        spike_data[None,:], coupled_spikes, stim_vec[None,:]], 
                            axis=0)
    labels = ['spikes',*[f'coup_{i}' for i in range(len(coupled_spikes))], 'stim']
    data_frame = pd.DataFrame(
            data = stacked_data.T,
            columns = labels)
    trial_starts = np.where(stim_vec[:-max_filter_len])[0]
    dat_len = len(spike_data)
    trial_labels = np.zeros(dat_len)
    trial_time = np.zeros(dat_len)
    counter = 0
    for i in range(len(trial_starts)):
        if i != len(trial_starts)-1:
            trial_labels[trial_starts[i]:trial_starts[i+1]] = counter
            counter +=1
            trial_time[trial_starts[i]:trial_starts[i+1]] = \
                    np.arange(0 , trial_starts[i+1] - trial_starts[i])
        else:
            trial_labels[trial_starts[i]:dat_len] = counter
            trial_time[trial_starts[i]:dat_len] = \
                    np.arange(0, dat_len - trial_starts[i])

    data_frame['trial_labels'] = trial_labels
    data_frame['trial_time'] = trial_time
    data_frame = data_frame.astype('int')
    return data_frame

def gen_trial_shuffle(data_frame, dv = 'spikes'):
    """
    Mismatch trials between dv and iv
    """
    spike_dat = data_frame[dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x != dv]]
    unique_trials = iv_dat['trial_labels'].unique()
    trial_map = dict(zip(unique_trials, np.random.permutation(unique_trials)))
    iv_dat['trial_labels'] = [trial_map[x] for x in iv_dat['trial_labels']]
    iv_dat = iv_dat.sort_values(by = ['trial_labels', 'trial_time'])
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat], axis=1)
    return out_frame

def gen_circular_shuffle(data_frame, dv = 'spikes'):
    """
    Shuffle timebins across trials (i.e. maintain the position of time bins but
                                    change trial indices)
    """
    spike_dat = data_frame[dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x != dv]]
    time_grouped_dat = [x[1] for x in list(iv_dat.groupby('trial_time'))]
    for this_dat in time_grouped_dat:
        this_dat['trial_labels'] = np.random.permutation(this_dat['trial_labels'])
    iv_dat = pd.concat(time_grouped_dat)
    iv_dat = iv_dat.sort_values(by = ['trial_labels', 'trial_time'])
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat], axis=1)
    return out_frame

def gen_random_shuffle(data_frame, dv = 'spikes'):
    """
    Randomly shuffled IV and DV separately
    """
    trial_cols = ['trial_labels','trial_time']
    spike_dat = data_frame[dv]
    trial_dat = data_frame[trial_cols]
    rm_cols = trial_cols + [dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x not in rm_cols]]
    iv_dat = iv_dat.sample(frac = 1, replace=False)
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat, trial_dat.reset_index(drop=True)], axis=1)
    return out_frame

def dataframe_to_design_mat(data_frame, test_size = 0.2):
    """
    Split data into training and testing sets
    This NEEDS to be done at the design matrix level because
    temporal structure no longer matters then
    """
    coup_cols = [x for x in data_frame.columns if 'coup' in x]
    glmdata = gt.gen_stim_history_coupled_design(
                    spike_data = data_frame['spikes'].values, 
                    coupled_spikes = data_frame[coup_cols].values.T,
                    stim_data = data_frame['stim'].values,
                    hist_filter_len = 10,
                    coupling_filter_len = 10,
                    stim_filter_len = 500,
                    n_basis = 10,
                    basis = 'cos',
                    basis_spread = 'log',
                    )
    # Re-add trial_labels and trial_time
    trial_cols = ['trial_labels','trial_time']
    glmdata = pd.concat([glmdata, data_frame[trial_cols]], axis=1)
    glmdata = glmdata.dropna()
    glmdata.reset_index(inplace=True, drop=True)

    # Drop trials which are short
    trial_list = [x[1] for x in list(glmdata.groupby('trial_labels'))]
    trial_lens = [len(x) for x in trial_list]
    med_len = np.median(trial_lens)
    unwanted_trials = [i for i, this_len in enumerate(trial_lens) \
            if this_len != med_len]
    remaining_lens = [x for i,x in enumerate(trial_lens) \
            if i not in unwanted_trials]
    assert all([[x==y for x in remaining_lens] for y in remaining_lens]), \
            'Trial lengths are not equal'
    glmdata = glmdata.loc[~glmdata.trial_labels.isin(unwanted_trials)]
    glmdata.reset_index(inplace=True, drop=True)
    return glmdata
    

def return_train_test_split(data_frame, test_size = 0.2):
    train_dat, test_dat = train_test_split(
            data_frame, test_size=test_size, random_state=42)
    return train_dat.sort_index(), test_dat.sort_index() 

from scipy.special import gammaln
def poisson_ll(lam, k):
    """
    Poisson log likelihood
    """
    assert len(lam) == len(k), 'lam and k must be same length'
    assert all(lam > 0), 'lam must be non-negative'
    assert all(k >= 0), 'k must be non-negative'
    return np.sum(k*np.log(lam) - lam - gammaln(k+1))

# Generate data frame and calculate cross validated prediction
# Repeat for shuffled datasets
# If model has learned something useful, then shuffling dv-iv either
# across trials or circularly per time bin should diminish log likelihood

# Note: Shuffling needs to happen at the design_matrix level since
# history will always be a strong predictor and history can't be 
# shuffled without destroying the actual spike trains

data_frame =  gen_data_frame(
        spike_data, 
        coupled_spikes,
        stim_vec,
        )

actual_input_dat = data_frame.copy()
actual_design_mat = dataframe_to_design_mat(actual_input_dat)

# Generate shuffles
trial_sh_design_mat = gen_trial_shuffle(actual_design_mat)
circ_sh_design_mat = gen_circular_shuffle(actual_design_mat)
rand_sh_design_mat = gen_random_shuffle(actual_design_mat)

# Generate train test splits
actual_train_dat, actual_test_dat = return_train_test_split(actual_design_mat)
trial_sh_train_dat, trial_sh_test_dat = return_train_test_split(trial_sh_design_mat)
circ_sh_train_dat, circ_sh_test_dat = return_train_test_split(circ_sh_design_mat)
rand_sh_train_dat, rand_sh_test_dat = return_train_test_split(rand_sh_design_mat)

# Fit model to actual data
res,pred = gt.fit_stim_history_coupled_glm(
        glmdata = actual_design_mat,
        hist_filter_len = hist_filter_len,
        coupling_filter_len = coupling_filter_len,
        stim_filter_len= stim_filter_len,
        regularized=False,
        **basis_kwargs
        )

actual_test_pred = res.predict(actual_test_dat)
actual_test_ll = poisson_ll(actual_test_pred, actual_test_dat['spikes'].values)
actual_test_ll = np.round(actual_test_ll, 2)

trial_sh_test_pred = res.predict(trial_sh_test_dat)
trial_sh_test_ll = poisson_ll(trial_sh_test_pred, actual_test_dat['spikes'].values)
trial_sh_test_ll = np.round(trial_sh_test_ll, 2)

circ_sh_test_pred = res.predict(circ_sh_test_dat)
circ_sh_test_ll = poisson_ll(circ_sh_test_pred, actual_test_dat['spikes'].values)
circ_sh_test_ll = np.round(circ_sh_test_ll, 2)

rand_sh_test_pred = res.predict(rand_sh_test_dat)
rand_sh_test_ll = poisson_ll(rand_sh_test_pred, actual_test_dat['spikes'].values)
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
plt.show()

