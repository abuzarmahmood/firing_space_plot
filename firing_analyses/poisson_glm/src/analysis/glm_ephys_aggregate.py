"""
Questions:
    1) Are significant fits (actual LL > shuffled LL) related to firing rate?
    2) Intra-region vs Inter-region connectivity?
        2.1) Can we pull out neurons with one vs the other?
"""

import numpy as np
import pylab as plt
import pandas as pd
import sys
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/src')
import utils.makeRaisedCosBasis as cb
from analysis import aggregate_utils
from pandas import DataFrame as df
from pandas import concat
import os
from tqdm import tqdm, trange
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from itertools import product
from glob import glob
from scipy.stats import mannwhitneyu as mwu
from scipy.stats import wilcoxon
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import pingouin as pg
import json
import matplotlib_venn as venn

#run_str = 'run_004'
save_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
# Check if previous runs present
run_list = sorted(glob(os.path.join(save_path, 'run*')))
run_basenames = sorted([os.path.basename(x) for x in run_list])
print(f'Present runs : {run_basenames}')
# input_run_ind = int(input('Please specify current run (integer) :'))
input_run_ind = 4
run_str = f'run_{input_run_ind:03d}'
plot_dir=  f'/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/plots/{run_str}'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

from importlib import reload
reload(aggregate_utils)

(unit_region_frame,
    fin_pval_frame, 
    fin_ll_frame, 
    pred_spikes_list, 
    design_spikes_list, 
    ind_frame,
    ) = aggregate_utils.return_data(save_path, run_str)

############################################################
############################################################

# How many neurons are significant
# Can perform paired test as all were tested on same model
grouped_ll_frame = list(fin_ll_frame.groupby(['session','taste', 'neuron']))
grouped_ll_inds, grouped_ll_frame = zip(*grouped_ll_frame)
sh_cols = [x for x in grouped_ll_frame[0].columns if 'sh' in x]

ll_pval_list = []
ll_stat_list = []
for i in trange(len(grouped_ll_frame)):
    this_frame = grouped_ll_frame[i]
    pval_dict = {}
    stat_dict = {}
    for this_col in sh_cols:
        try:
            this_pval = wilcoxon(this_frame[this_col], this_frame['actual'])
            pval_dict[this_col] = this_pval.pvalue
            stat_dict[this_col] = this_pval.statistic
        except ValueError:
            pval_dict[this_col] = np.nan
            stat_dict[this_col] = np.nan
    ll_pval_list.append(pval_dict)
    ll_stat_list.append(stat_dict)

ll_pval_frame = np.log10(pd.DataFrame(ll_pval_list))
grouped_ll_inds_frame = pd.DataFrame(grouped_ll_inds, columns = ['session','taste','neuron'])
ll_pval_frame = pd.concat([grouped_ll_inds_frame, ll_pval_frame], axis=1)

ll_stat_frame = pd.DataFrame(ll_stat_list)
ll_stat_frame = pd.concat([grouped_ll_inds_frame, ll_stat_frame], axis=1)

# Drop nan rows
ll_pval_frame = ll_pval_frame.dropna()
ll_stat_frame = ll_stat_frame.dropna()

# Sort by session, taste, neuron
ll_pval_frame = ll_pval_frame.sort_values(by=['session','taste','neuron'])
ll_stat_frame = ll_stat_frame.sort_values(by=['session','taste','neuron'])

# log10(0.005) = -2.3
wanted_cols = [x for x in ll_pval_frame.columns if 'sh' in x]
plot_dat = ll_pval_frame[wanted_cols]

thresh = np.round(np.log10(0.001),2)
sig_frac = np.round((plot_dat< thresh).mean(axis=0),2)
# Fraction significant for all 3 shuffles
all_sig_frac = np.round(np.all((plot_dat < thresh).values, axis=-1).mean(),2)

# Sort frame by KMeans and plot
kmeans = KMeans(n_clusters=4, random_state=0).fit(plot_dat.values)
plot_dat = plot_dat.iloc[kmeans.labels_.argsort()] 

plt.imshow(plot_dat.values, interpolation = 'none', aspect = 'auto')
plt.colorbar(label = 'Log10 P-Value')
plt.title('Log10 P-Values for Wilcoxon Signed Rank Test' \
        + '\n' + f'Significant Fraction {thresh} (log)-> {np.round(10**thresh,3)}: ' \
        + str(sig_frac.values) + '\n' + f'All Significant Fraction: {all_sig_frac}')
plt.xlabel('Shuffle Type')
plt.xticks(np.arange(len(plot_dat.columns)), plot_dat.columns, rotation = 90)
fig=plt.gcf()
fig.set_size_inches(4,10)
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_dir,'log10_pval_frame.png'), dpi = 300, bbox_inches = 'tight')
plt.close()

############################################################
############################################################
# Toss out neurons that are not significant for all 3 shuffles
sig_rows = np.all((ll_pval_frame[wanted_cols] < thresh).values, axis=-1) 
ll_pval_frame = ll_pval_frame[sig_rows]

############################################################
############################################################
# Is firing rate related to significance?

# Mean firing rate for each region
#sns.histplot(data = unit_region_frame, x = 'mean_rate', 
#             hue = 'region', bins = 50,
#             cumulative = True, stat = 'probability', element = 'step',
#             common_norm = False, fill = False)
#plt.title('Mean Firing Rate for Each Region')
#plt.xlabel('Mean Firing Rate')
#plt.ylabel('Count')
#plt.show()

# Merge ll_pval_frame with unit_region_frame
ll_pval_frame = ll_pval_frame.merge(unit_region_frame, on = ['session','neuron'])

sns.jointplot(data = ll_pval_frame, x = 'mean_rate', y = 'trial_sh', 
              hue = 'region', kind = 'hist')
plt.axhline(y = thresh, color = 'r', linestyle = '--', label = '0.05')
plt.suptitle('Mean Firing Rate vs. Significance')
plt.xlabel('Mean Firing Rate')
plt.ylabel('Log10 P-Value')
#plt.show()
plt.savefig(os.path.join(plot_dir,'mean_rate_vs_significance.png'), dpi = 300, bbox_inches = 'tight')
plt.close()

############################################################
############################################################
# Pretty examples --> High firing rate and high log-likelihood
# Actual and predicted PSTHs

n_top = 100
# Plot log_ll vs mean_rate
mean_nrn_ll_frame = fin_ll_frame.groupby(ind_names).median().reset_index(drop=False)
mean_nrn_ll_frame = mean_nrn_ll_frame.sort_values(by = ['mean_rate','actual'], ascending = False)
# Take out rows which don't mathc with ll_pval_frame
mean_nrn_ll_frame = mean_nrn_ll_frame.merge(ll_pval_frame[ind_names], on = ind_names)
mean_nrn_ll_frame['top'] = False
mean_nrn_ll_frame.loc[mean_nrn_ll_frame.index[:n_top],'top'] = True

# Remove outliers
mean_nrn_ll_frame = mean_nrn_ll_frame[mean_nrn_ll_frame['actual'] > -1e10]

sns.jointplot(data = mean_nrn_ll_frame, 
              x = 'mean_rate', y = 'actual',
              hue = 'top', palette = ['b','r'],)
plt.suptitle('Mean Firing Rate vs. Log Likelihood')
plt.xlabel('Mean Firing Rate')
plt.ylabel('Log Likelihood')
plt.savefig(os.path.join(plot_dir,'mean_rate_vs_log_ll.png'), dpi = 300, bbox_inches = 'tight')
plt.close()
#plt.show()

# Extract top inds
top_inds_frame = mean_nrn_ll_frame[mean_nrn_ll_frame['top']]
top_inds_frame = top_inds_frame.sort_values(by = 'actual', ascending = False)
top_inds = top_inds_frame[ind_names].values[:n_top]

##############################
# Also find neurons which have high likelihood averaged across all tastes
mean_nrn_taste_ll_frame = mean_nrn_ll_frame.groupby(['session','neuron']).mean().reset_index(drop=False)
mean_nrn_taste_ll_frame = mean_nrn_taste_ll_frame.sort_values(by = ['mean_rate','actual'], ascending = False)
mean_nrn_taste_ll_frame['top'] = False
mean_nrn_taste_ll_frame.loc[mean_nrn_taste_ll_frame.index[:n_top],'top'] = True

# Extract top inds
taste_top_inds_frame = mean_nrn_taste_ll_frame[mean_nrn_taste_ll_frame['top']]
taste_top_inds_frame = taste_top_inds_frame.sort_values(by = 'actual', ascending = False)
taste_top_inds = taste_top_inds_frame[['session','neuron']].values[:n_top]

# Recalculate PSTHs for top inds
############################################################
# Parameters

# Load parameters from run
json_path = os.path.join(save_path, run_str,'fit_params.json')

params_dict = json.load(open(json_path))

hist_filter_len = params_dict['hist_filter_len']
stim_filter_len = params_dict['stim_filter_len']
coupling_filter_len = params_dict['coupling_filter_len']

trial_start_offset = params_dict['trial_start_offset']
trial_lims = np.array(params_dict['trial_lims'])
stim_t = params_dict['stim_t']

bin_width = params_dict['bin_width']

# Reprocess filter lens
hist_filter_len_bin = params_dict['hist_filter_len_bin'] 
stim_filter_len_bin = params_dict['stim_filter_len_bin']
coupling_filter_len_bin = params_dict['coupling_filter_len_bin']

# Define basis kwargs
basis_kwargs = params_dict['basis_kwargs'] 

# Number of fits on actual data (expensive)
n_fits = params_dict['n_fits']
n_max_tries = params_dict['n_max_tries']
n_shuffles_per_fit = params_dict['n_shuffles_per_fit']

############################################################
make_example_plots = False

# Generate PSTHs for all tastes
psth_plot_dir = os.path.join(plot_dir, 'example_psths')
if not os.path.exists(psth_plot_dir):
    os.makedirs(psth_plot_dir)

for idx, dat_inds in tqdm(zip(session_taste_inds, all_taste_inds)):
    this_spike_dat = np.stack([design_spikes_list[i] for i in dat_inds])
    this_pred_dat = np.stack([pred_spikes_list[i] for i in dat_inds])

    actual_psth = np.mean(this_spike_dat, axis = 1)
    pred_psth = np.mean(this_pred_dat, axis = 1)

    # Smoothen PSTH
    kern_len = 200
    kern = np.ones(kern_len)/kern_len
    actual_psth_smooth = np.apply_along_axis(
            lambda m: np.convolve(m, kern, mode = 'same'), 
            axis = -1, 
            arr = actual_psth)
    pred_psth_smooth = np.apply_along_axis(
            lambda m: np.convolve(m, kern, mode = 'same'),
            axis = -1,
            arr = pred_psth)

    # Plot PSTHs
    fig,ax = plt.subplots(1,2, sharex=True, sharey=True, figsize = (7,2))
    ax[0].plot(actual_psth_smooth.T)
    ax[1].plot(pred_psth_smooth.T)
    ax[0].set_title('Actual')
    ax[1].set_title('Predicted')
    fig.suptitle('Session {}, Neuron {}'.format(idx[0], idx[1]))
    plt.savefig(os.path.join(psth_plot_dir, 'psth_{}.png'.format(idx)), dpi = 300, bbox_inches = 'tight')
    plt.close()


############################################################
# Process inferred filters 
############################################################
# Length of basis is adjusted because models were fit on binned data
hist_cosine_basis = cb.gen_raised_cosine_basis(
        hist_filter_len_bin,
        n_basis = basis_kwargs['n_basis'],
        spread = basis_kwargs['basis_spread'],
        )
stim_cosine_basis = cb.gen_raised_cosine_basis(
        stim_filter_len_bin,
        n_basis = basis_kwargs['n_basis'],
        spread = basis_kwargs['basis_spread'],
        )
coup_cosine_basis = cb.gen_raised_cosine_basis(
        coupling_filter_len_bin,
        n_basis = basis_kwargs['n_basis'],
        spread = basis_kwargs['basis_spread'],
        )

#test_basis = gt.cb.gen_raised_cosine_basis(200, n_basis = 20, spread = 'log')
#plt.plot(test_basis.sum(axis=0), color = 'red', linewidth = 2)
#plt.plot(test_basis.T);plt.show()

# Throw out all rows which don't have significant differences in likelihood
# between actual and shuffled
fin_pval_frame = fin_pval_frame.merge(ll_pval_frame, on = ind_names)

# Only take fit_num with highest likelihood
max_ll_frame = fin_ll_frame[['fit_num','actual',*ind_names]]
max_inds = max_ll_frame.groupby(ind_names).actual.idxmax().reset_index().actual
max_vals = max_ll_frame.loc[max_inds].drop(columns = 'actual') 

fin_pval_frame = fin_pval_frame.merge(max_vals, on = ['fit_num',*ind_names])
fin_pval_frame['agg_index'] = ["_".join([str(x) for x in y]) for y in fin_pval_frame[ind_names].values]

sig_alpha = 0.05
############################################################
# Extract history filters
hist_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('hist')]
#hist_frame.drop(columns = ['trial_sh','circ_sh','rand_sh'], inplace = True)
hist_frame = hist_frame[['fit_num','param','p_val','values', *ind_names, 'agg_index']]
hist_frame['lag'] = hist_frame.param.str.extract('(\d+)').astype(int)
#hist_groups = [x[1] for x in list(hist_frame.groupby(ind_names))]
#hist_groups = [x.sort_values('lag') for x in hist_groups]

hist_val_pivot = hist_frame.pivot_table(
        index = 'agg_index',
        columns = 'lag',
        values = 'values',
        )
hist_pval_pivot = hist_frame.pivot_table(
        index = 'agg_index',
        columns = 'lag',
        values = 'p_val',
        )

hist_val_array = hist_val_pivot.values 
hist_pval_array = hist_pval_pivot.values

sig_hist_filters = np.where((hist_pval_array < sig_alpha).sum(axis=1))[0]
frac_sig_hist_filters = np.round(len(sig_hist_filters) / len(hist_pval_array), 2)
print(f'Fraction of significant history filters: {frac_sig_hist_filters}')

# Cluster using Kmeans
kmeans = KMeans(n_clusters = 4, random_state = 0).fit(hist_val_array)
hist_val_array = hist_val_array[kmeans.labels_.argsort()]

# Reconstruct hist filters
hist_recon = np.dot(hist_val_array, hist_cosine_basis)

## Plot
#plt.imshow(hist_val_array, aspect = 'auto', interpolation = 'none')
#plt.colorbar()
#plt.show()

# plot principle components
pca = PCA(n_components = 5)
pca.fit(hist_recon.T)
pca_array = pca.transform(hist_recon.T)
hist_tvec = np.arange(hist_recon.shape[1]) * bin_width

fig, ax = plt.subplots(2,1, sharex=True)
for i, dat in enumerate(pca_array.T):
    ax[0].plot(hist_tvec, dat, 
               label = f'PC {i}, {np.round(pca.explained_variance_ratio_[i],2)}', 
               linewidth = 5, alpha = 0.7)
ax[0].legend()
ax[0].set_ylabel('PC Magnitude')
ax[1].pcolormesh(hist_tvec, np.arange(pca_array.shape[1]), pca_array.T) 
ax[1].set_ylabel('PC #')
ax[1].set_xlabel('Time (ms)')
fig.suptitle(f'PCA of history filters, \n Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}')
fig.savefig(os.path.join(plot_dir, 'hist_filter_pca.png'), 
            dpi = 300, bbox_inches = 'tight')
plt.close(fig)
#plt.show()

# Plot each filter in it's own subplot
plot_cutoff = 50
fig, ax = plt.subplots(len(pca_array.T), 1, sharex=True, sharey=True,
                       figsize = (3,10))
peak_markers = [3,6,11,19]
for i, (this_dat, this_ax) in enumerate(zip(pca_array.T[:,:plot_cutoff], ax)):
    this_ax.plot(hist_tvec, this_dat)
    this_ax.set_ylabel(f'PC {i} : {np.round(pca.explained_variance_ratio_[i],2)}')
    for this_peak in peak_markers:
        this_ax.axvline(this_peak, linestyle = '--', color = 'k', alpha = 0.5)
ax[-1].set_xlabel('Time (ms)')
pca_str = f'Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}'
marker_str = f'Peak markers: {peak_markers} ms'
fig.suptitle(f'PCA of history filters (zoomed) \n' + pca_str + '\n' + marker_str)
fig.savefig(os.path.join(plot_dir, 'hist_filter_pca2_zoom.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)


############################################################
############################################################
# Extract stimulus filters
stim_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('stim')]
#stim_frame.drop(columns = ['trial_sh','circ_sh','rand_sh'], inplace = True)
stim_frame = stim_frame[['fit_num','param','p_val','values', *ind_names, 'agg_index']]
stim_frame['lag'] = stim_frame.param.str.extract('(\d+)').astype(int)
#stim_groups = [x[1] for x in list(stim_frame.groupby(ind_names))]
#stim_groups = [x.sort_values('lag') for x in stim_groups]

stim_val_pivot = stim_frame.pivot_table(
        index = 'agg_index',
        columns = 'lag',
        values = 'values',
        )
stim_pval_pivot = stim_frame.pivot_table(
        index = 'agg_index',
        columns = 'lag',
        values = 'p_val',
        )

stim_val_array = stim_val_pivot.values #np.stack([x['values'].values for x in stim_groups])
stim_pval_array = stim_pval_pivot.values #np.stack([x['p_val'].values for x in stim_groups])

sig_stim_filters = np.where((stim_pval_array < sig_alpha).sum(axis=1))[0]
frac_sig_stim_filters = np.round(len(sig_stim_filters) / len(stim_pval_array), 2)
print(f'Fraction of significant stimory filters: {frac_sig_stim_filters}')


# Reconstruct stim filters
stim_recon = np.dot(stim_val_array, stim_cosine_basis)
zscore_stim_recon = zscore(stim_recon,axis=-1)

# Cluster using Kmeans
kmeans = KMeans(n_clusters = 4, random_state = 0).fit(zscore_stim_recon)
zscore_stim_recon = zscore_stim_recon[kmeans.labels_.argsort()]
stim_recon = stim_recon[kmeans.labels_.argsort()]

stim_tvec = np.arange(stim_recon.shape[1]) * bin_width

## Plot
#fig, ax = plt.subplots(1,2)
#ax[0].imshow(stim_recon, aspect = 'auto', interpolation = 'none')
#ax[1].imshow(zscore_stim_recon, aspect = 'auto', interpolation = 'none')
#plt.colorbar()
#plt.show()

# plot principle components
pca = PCA(n_components = 5)
pca.fit(stim_recon.T)
pca_array = pca.transform(stim_recon.T)

fig, ax = plt.subplots(2,1, sharex=True)
for i, dat in enumerate(pca_array.T):
    ax[0].plot(stim_tvec, dat, label = f'PC {i}, {np.round(pca.explained_variance_ratio_[i],2)}', 
               linewidth = 2, alpha = 0.7)
ax[0].legend(loc='right')
ax[0].set_ylabel('PC Magnitude')
ax[1].pcolormesh(stim_tvec, np.arange(pca_array.shape[1]), pca_array.T) 
ax[1].set_ylabel('PC #')
ax[1].set_xlabel('Time (ms)')
fig.suptitle(f'PCA of stimulus filters, \n Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}')
fig.savefig(os.path.join(plot_dir, 'stim_filter_pca.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)
#plt.show()

# Plot each filter in it's own subplot
fig, ax = plt.subplots(len(pca_array.T), 1, sharex=True, sharey=True,
                       figsize = (3,10))
for i, (this_dat, this_ax) in enumerate(zip(pca_array.T, ax)):
    this_ax.plot(stim_tvec, this_dat)
    this_ax.set_ylabel(f'PC {i} : {np.round(pca.explained_variance_ratio_[i],2)}')
ax[-1].set_xlabel('Time (ms)')
fig.suptitle(f'PCA of stimulus filters, \n Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}')
fig.savefig(os.path.join(plot_dir, 'stim_filter_pca2.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

## Plot each filter in it's own subplot
#plot_cutoff = 50
#fig, ax = plt.subplots(len(pca_array.T), 1, sharex=True, sharey=True,
#                       figsize = (3,10))
#peak_markers = [2,4,8,16]
#for i, (this_dat, this_ax) in enumerate(zip(pca_array.T[:,:plot_cutoff], ax)):
#    this_ax.plot(this_dat)
#    this_ax.set_ylabel(f'PC {i} : {np.round(pca.explained_variance_ratio_[i],2)}')
#    for this_peak in peak_markers:
#        this_ax.axvline(this_peak, linestyle = '--', color = 'k', alpha = 0.5)
#ax[-1].set_xlabel('Time (ms)')
#pca_str = f'Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}'
#marker_str = f'Peak markers: {peak_markers} ms'
#fig.suptitle(f'PCA of Stim filters (zoomed) \n' + pca_str + '\n' + marker_str)
#fig.savefig(os.path.join(plot_dir, 'stim_filter_pca2_zoom.png'), dpi = 300, bbox_inches = 'tight')
#plt.close(fig)

############################################################

# Extract coupling filters
############################################################
coupling_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('coup')]
coupling_frame.drop(columns = ['trial_sh','circ_sh','rand_sh'], inplace = True)
# Make sure there are no 0 pvals
coupling_frame.p_val += 1e-20

# Fraction of significant coupling filter values per threshold
alpha_vec = np.round(np.logspace(-1,-3,5),3)
frac_sig = [(coupling_frame.p_val < alpha).mean() for alpha in alpha_vec]
frac_ratio = np.round(np.array(frac_sig) / alpha_vec, 2)
print(dict(zip(alpha_vec, frac_ratio)))

# Assuming one significant value is enough, how many significant filters
coupling_frame = coupling_frame[['fit_num','param','p_val','values', *ind_names]]

coupling_frame['lag'] = [int(x.split('_')[-1]) for x in coupling_frame.param]
coupling_frame['other_nrn'] = [int(x.split('_')[-2]) for x in coupling_frame.param]

coupling_grouped_list = list(coupling_frame.groupby(ind_names))
coupling_grouped_inds = [x[0] for x in coupling_grouped_list]
coupling_grouped = [x[1] for x in coupling_grouped_list] 

# For each group, pivot to have other_nrn as row and lag as column
coupling_pivoted_vals = [x.pivot(index = 'other_nrn', columns = 'lag', values = 'values') \
        for x in coupling_grouped]
coupling_pivoted_pvals = [x.pivot(index = 'other_nrn', columns = 'lag', values = 'p_val') \
        for x in coupling_grouped]

# Count each filter as significant if a value is below alpha
# Note, these are the neuron inds as per the array of each session
base_alpha = 0.05
#alpha = 0.001
alpha = base_alpha / len(coupling_frame['lag'].unique()) # Bonferroni Correction 
coupling_pivoted_raw_inds = [np.where((x < alpha).sum(axis=1))[0] \
        for x in coupling_pivoted_pvals]
coupling_pivoted_frame_index = [x.index.values for x in coupling_pivoted_vals]
coupling_pivoted_sig_inds = [y[x] for x,y in zip(coupling_pivoted_raw_inds, coupling_pivoted_frame_index)]

########################################
# Coupling filter profiles
coupling_val_array = np.concatenate(coupling_pivoted_vals, axis = 0)

# Reconstruct coupling filters
coupling_recon = np.dot(coupling_val_array, coup_cosine_basis)
coupling_tvec = np.arange(coupling_recon.shape[1]) * bin_width

# plot principle components
pca = PCA(n_components = 5)
pca.fit(coupling_recon.T)
pca_array = pca.transform(coupling_recon.T)

fig, ax = plt.subplots(2,1, sharex=True)
for i, dat in enumerate(pca_array.T):
    ax[0].plot(coupling_tvec, dat, 
               label = f'PC {i}, {np.round(pca.explained_variance_ratio_[i],2)}', 
               linewidth = 2, alpha = 0.7)
ax[0].legend(loc='right')
ax[0].set_ylabel('PC Magnitude')
ax[1].pcolormesh(coupling_tvec, np.arange(pca_array.shape[1]), pca_array.T) 
ax[1].set_ylabel('PC #')
ax[1].set_xlabel('Time (ms)')
fig.suptitle(f'PCA of coupling filters, \n Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}')
fig.savefig(os.path.join(plot_dir, 'coupling_filter_pca.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)
#plt.show()

# Plot each filter in it's own subplot
fig, ax = plt.subplots(len(pca_array.T), 1, sharex=True, sharey=True,
                       figsize = (3,10))
for i, (this_dat, this_ax) in enumerate(zip(pca_array.T, ax)):
    this_ax.plot(coupling_tvec, this_dat)
    this_ax.set_ylabel(f'PC {i} : {np.round(pca.explained_variance_ratio_[i],2)}')
ax[-1].set_xlabel('Time (ms)')
fig.suptitle(f'PCA of coupling filters, \n Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}')
fig.savefig(os.path.join(plot_dir, 'coupling_filter_pca2.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

# Plot each filter in it's own subplot
plot_cutoff = 50
fig, ax = plt.subplots(len(pca_array.T), 1, sharex=True, sharey=True,
                       figsize = (3,10))
peak_markers = [6,11,19]
for i, (this_dat, this_ax) in enumerate(zip(pca_array.T[:,:plot_cutoff], ax)):
    this_ax.plot(coupling_tvec, this_dat)
    this_ax.set_ylabel(f'PC {i} : {np.round(pca.explained_variance_ratio_[i],2)}')
    for this_peak in peak_markers:
        this_ax.axvline(this_peak, linestyle = '--', color = 'k', alpha = 0.5)
ax[-1].set_xlabel('Time (ms)')
pca_str = f'Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}'
marker_str = f'Peak markers: {peak_markers} ms'
fig.suptitle(f'PCA of coupling filters (zoomed) \n' + pca_str + '\n' + marker_str)
fig.savefig(os.path.join(plot_dir, 'coupling_filter_pca2_zoom.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

########################################
# Do filters from BLA-->GC and GC-->BLA have different shapes

coupling_list_frame = coupling_frame.groupby([*ind_names, 'other_nrn']).\
        agg({'values' : lambda x : x.tolist(),
             'p_val' : lambda x: x.tolist()}).reset_index()
coupling_list_frame = coupling_list_frame.merge(unit_region_frame[['neuron','region','session']],
                                how = 'left', on = ['session','neuron'])
coupling_list_frame = coupling_list_frame.merge(unit_region_frame[['neuron','region', 'session']],
                                how = 'left', left_on = ['session', 'other_nrn'], 
                                right_on = ['session','neuron'])
coupling_list_frame.drop(columns = 'neuron_y', inplace = True)
coupling_list_frame.rename(columns = {
    'neuron_x':'neuron', 
    'region_x' : 'region',
    'region_y' : 'input_region'}, 
                   inplace = True)

coupling_io_groups_list = list(coupling_list_frame.groupby(['region','input_region']))
coupling_io_group_names = [x[0] for x in coupling_io_groups_list]

coupling_io_group_filters = [np.stack(x[1]['values']) for x in coupling_io_groups_list]
coupling_io_group_filter_recon = [x.dot(coup_cosine_basis) for x in coupling_io_group_filters]
coupling_io_group_filter_pca = np.stack([PCA(n_components=3).fit_transform(x.T) \
        for x in coupling_io_group_filter_recon])

vmin,vmax = np.min(coupling_io_group_filter_pca), np.max(coupling_io_group_filter_pca)
fig,ax = plt.subplots(2,2, sharex=True, sharey=True)
for i, (this_dat, this_ax) in enumerate(zip(coupling_io_group_filter_pca, ax.flatten())):
    #this_ax.plot(coupling_tvec, this_dat)
    im = this_ax.pcolormesh(coupling_tvec, np.arange(len(this_dat.T)),
                       this_dat.T, vmin = vmin, vmax = vmax)
    this_ax.set_ylabel('PC #')
    this_ax.set_title("<--".join(coupling_io_group_names[i]))
cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
ax[-1,-1].set_xlabel('Time (ms)')
fig.suptitle('PCA of coupling filters')
#plt.show()
fig.savefig(os.path.join(plot_dir, 'coupling_by_connection.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

fig,ax = plt.subplots(2,2, sharex=True, sharey=True)
for i, (this_dat, this_ax) in enumerate(zip(coupling_io_group_filter_pca, ax.flatten())):
    for num, this_pc in enumerate(this_dat.T):
        this_ax.plot(coupling_tvec, this_pc, label = f'PC{num}')
    this_ax.set_title("<--".join(coupling_io_group_names[i]))
ax[-1,-1].set_xlabel('Time (ms)')
ax[-1,-1].legend()
fig.suptitle('PCA of coupling filters')
fig.savefig(os.path.join(plot_dir, 'coupling_by_connection2.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

# Check whether filter shapes and pvalue distributions are different
coupling_pval_dat = coupling_list_frame.drop(columns = 'values') 
coupling_pval_dat = coupling_pval_dat.explode('p_val')
coupling_pval_dat['log_pval'] = np.vectorize(np.log10)(coupling_pval_dat['p_val'])
coupling_pval_dat.reset_index(inplace=True)
coupling_pval_dat['group_str'] = coupling_pval_dat.apply(lambda x: f'{x.region} <-- {x.input_region}', axis = 1)

g = sns.displot(
        data = coupling_pval_dat,
        x = 'log_pval',
        kind = 'ecdf',
        hue = 'group_str',
        )
this_ax = g.axes[0][0]
this_ax.set_yscale('log')
this_ax.set_ylim([0.005,1])
this_ax.set_ylabel('Fraction of filters')
this_ax.set_xlabel('log10(p-value)')
this_ax.set_title('Cumulative distribution of p-values')
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'coupling_pval_dist.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)
#plt.show()

########################################

# Total filters
total_filters = [x.shape[0] for x in coupling_pivoted_vals]
total_sig_filters = [len(x) for x in coupling_pivoted_sig_inds]

# Fraction of significant filters
frac_sig_coup_filters = np.round(sum(total_sig_filters) / sum(total_filters), 3)
print(f'Fraction of significant coupling filters: {frac_sig_coup_filters}') 

# Match inds to actuals neurons
# First collate connectivity matrices
tuple_dat = [tuple([*x,y]) for x,y in zip(coupling_grouped_inds, coupling_pivoted_sig_inds)]
tuple_frame = pd.DataFrame(tuple_dat, columns = [*ind_names, 'sig_inds'])

# Convert tuple frame to long-form
tuple_frame = tuple_frame.explode('sig_inds')

# Merge with unit_region_frame to obtain neuron region
tuple_frame = tuple_frame.rename(columns = {'sig_inds':'input_neuron'})
tuple_frame = tuple_frame.merge(unit_region_frame[['neuron','region','session']],
                                how = 'left', on = ['session','neuron'])
# Merge again to assign region to input_neuron
tuple_frame = tuple_frame.merge(unit_region_frame[['neuron','region', 'session']],
                                how = 'left', left_on = ['session', 'input_neuron'], 
                                right_on = ['session','neuron'])
tuple_frame.drop(columns = 'neuron_y', inplace = True)
tuple_frame.rename(columns = {
    'neuron_x':'neuron', 
    'region_x' : 'region',
    'region_y' : 'input_region'}, 
                   inplace = True)

# per session and neuron, what is the distribution of intra-region
# vs inter-region connections
count_per_input = tuple_frame.groupby([*ind_names, 'region', 'input_region']).count()
count_per_input.reset_index(inplace = True)

total_count_per_region = unit_region_frame[['region','neuron','session']]\
        .groupby(['session','region']).count()
total_count_per_region.reset_index(inplace = True)

# Merge to get total count per region
count_per_input = count_per_input.merge(total_count_per_region, how = 'left',
                                        left_on = ['session','input_region'],
                                        right_on = ['session','region'])
count_per_input.rename(columns = {
    'neuron_x':'neuron', 
    'region_x':'region',
    'neuron_y' : 'region_total'}, inplace = True)
count_per_input.drop(columns = ['region_y'], inplace = True)

count_per_input['input_fraction'] = count_per_input.input_neuron / count_per_input.region_total 

# Is there an interaction between region and input region
input_fraction_anova = pg.anova(count_per_input, dv = 'input_fraction', between = ['region','input_region'])
sns.boxplot(data = count_per_input, x = 'region', y = 'input_fraction', hue = 'input_region',
              dodge=True)
plt.title(str(input_fraction_anova[['Source','p-unc']].dropna().round(2)))
plt.suptitle('Comparison of Input Fraction')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'input_fraction_boxplot.png'), dpi = 300)
plt.close()
#plt.show()

#sns.displot(data = count_per_input, col = 'region', x = 'input_fraction', hue = 'input_region',
#            kind = 'ecdf')
#plt.show()

# Region preference index
region_pref_frame = count_per_input[['session','taste','neuron','region','input_region','input_fraction']]
region_pref_frame['region_pref'] = region_pref_frame.groupby(['session','taste','neuron'])['input_fraction'].diff().dropna()
region_pref_frame.dropna(inplace = True)
# region_pref = bla - gc (so positive is more bla than gc)
region_pref_frame.drop(columns = ['input_fraction', 'input_region'], inplace = True)

sns.swarmplot(data = region_pref_frame, x = 'region', y = 'region_pref')
plt.ylabel('Region preference index, \n (BLA frac - GC frac) \n <-- More GC Input | More BLA Input -->')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'preference_index_swarm.png'), dpi = 300)
plt.close()
#plt.show()

##############################
# Segregation of projecting populations
tuple_frame['neuron_idx'] = ["_".join([str(y) for y in x]) for x in tuple_frame[ind_names].values] 
tuple_frame['input_neuron_idx'] = ["_".join([str(y) for y in x]) \
        for x in tuple_frame[['session','taste','input_neuron']].values] 
tuple_frame['cxn_type'] = ["<--".join([str(y) for y in x]) \
        for x in tuple_frame[['region','input_region']].values]

inter_region_frame = tuple_frame.loc[tuple_frame.cxn_type.isin(['gc<--bla','bla<--gc'])]

gc_neurons_rec = inter_region_frame.loc[inter_region_frame.cxn_type == 'gc<--bla']['neuron_idx'].unique()
gc_neurons_send = inter_region_frame.loc[inter_region_frame.cxn_type == 'bla<--gc']['input_neuron_idx'].unique()

bla_neurons_rec = inter_region_frame.loc[inter_region_frame.cxn_type == 'bla<--gc']['neuron_idx'].unique()
bla_neurons_send = inter_region_frame.loc[inter_region_frame.cxn_type == 'gc<--bla']['input_neuron_idx'].unique()

fig,ax = plt.subplots(2,1)
venn.venn2(
        [set(gc_neurons_rec), set(gc_neurons_send)], 
        set_labels = ('GC rec', 'GC send'),
        ax = ax[0])
venn.venn2(
        [set(bla_neurons_rec), set(bla_neurons_send)],
        set_labels = ('BLA rec', 'BLA send'),
        ax = ax[1])
ax[0].set_title('GC neurons')
ax[1].set_title('BLA neurons')
fig.suptitle('Overlap in neurons sending and receiving input')
fig.savefig(os.path.join(plot_dir, 'projecting_neuron_venn.png'), dpi = 300)
plt.close()
#plt.show()

# Do these groups receive more or less input than the general population
# e.g. do the bla_to_gc projecting BLA neurons receive more or less input from 
# GC than the rest of the BLA population

##############################
# Perform similar analysis, but for magnitude of filter

# Check relationship between values and p_val
plt.scatter(
        np.log10(coupling_frame['p_val']), 
        np.log(coupling_frame['values']),
        alpha = 0.01
        )
plt.xlabel('log10(p_val)')
plt.ylabel('log(values)')
plt.title('Coupling filters pvalues vs values')
plt.savefig(os.path.join(plot_dir, 'coupling_pval_vs_val.png'), dpi = 300)
plt.close()

# Filter energy
# Not sure whether to take absolute or not
# Because with absolute, flucutations about 0 will add up to something
# HOWEVER, IF FITS ARE ACCURATE, it shouldn't really matter
#coupling_filter_energy = [np.sum(np.abs(x.values),axis=-1) for x in coupling_pivoted_vals]
coupling_energy_frame = coupling_frame.copy()
coupling_energy_frame['pos_values'] = np.abs(coupling_frame['values'])
coupling_energy_frame = coupling_energy_frame.groupby([*ind_names, 'other_nrn']).sum()['pos_values'].reset_index()
coupling_energy_frame.rename(columns = {'pos_values' : 'energy'}, inplace=True)

# Merge with unit_region_frame to obtain neuron region
coupling_energy_frame = coupling_energy_frame.rename(columns = {'other_nrn':'input_neuron'})
coupling_energy_frame = coupling_energy_frame.merge(unit_region_frame[['neuron','region','session']],
                                how = 'left', on = ['session','neuron'])
# Merge again to assign region to input_neuron
coupling_energy_frame = coupling_energy_frame.merge(unit_region_frame[['neuron','region', 'session']],
                                how = 'left', left_on = ['session', 'input_neuron'], 
                                right_on = ['session','neuron'])
coupling_energy_frame.drop(columns = 'neuron_y', inplace = True)
coupling_energy_frame.rename(columns = {
    'neuron_x':'neuron', 
    'region_x' : 'region',
    'region_y' : 'input_region'}, 
                   inplace = True)

input_energy_anova = pg.anova(
        coupling_energy_frame, 
        dv = 'energy', 
        between = ['region','input_region'])

sns.boxplot(data = coupling_energy_frame, x = 'region', y = 'energy', hue = 'input_region',
              dodge=True, showfliers = False)
plt.suptitle('Comparison of Input Filter Energy')
plt.title(str(input_energy_anova[['Source','p-unc']].dropna().round(2)))
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(plot_dir, 'input_energy_boxplot.png'), dpi = 300)
plt.close()
