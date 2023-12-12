############################################################
# Imports 
############################################################
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

def set_params_to_globals(save_path, run_str):
    json_path = os.path.join(save_path, run_str,'fit_params.json')
    params_dict = json.load(open(json_path))

    param_names = ['hist_filter_len',
                   'stim_filter_len',
                   'coupling_filter_len',
                   'trial_start_offset',
                   'trial_lims',
                   'stim_t',
                   'bin_width',
                   'hist_filter_len_bin',
                   'stim_filter_len_bin',
                   'coupling_filter_len_bin',
                   'basis_kwargs',
                   'n_fits',
                   'n_max_tries',
                   'n_shuffles_per_fit',
                   ]

    for param_name in param_names:
        globals()[param_name] = params_dict[param_name]

############################################################
# Setup 
############################################################
#run_str = 'run_004'
save_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
# Check if previous runs present
run_list = sorted(glob(os.path.join(save_path, 'run*')))
run_basenames = sorted([os.path.basename(x) for x in run_list])
print(f'Present runs : {run_basenames}')
# input_run_ind = int(input('Please specify current run (integer) :'))
input_run_ind = 6
run_str = f'run_{input_run_ind:03d}'
plot_dir=  f'/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/plots/{run_str}'

# Parameters
set_params_to_globals(save_path, run_str)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# from importlib import reload
# reload(aggregate_utils)

sig_alpha = 0.05

############################################################
# Load Data
############################################################
(unit_region_frame,
    fin_pval_frame, 
    fin_ll_frame, 
    pred_spikes_list, 
    design_spikes_list, 
    ind_frame,
    session_taste_inds,
     all_taste_inds,
    ) = aggregate_utils.return_data(save_path, run_str)


############################################################
## Distribution of p-values per filter type for actual vs shuffled
############################################################
fin_pval_frame['filt_type'] = fin_pval_frame['param'].apply(
        lambda x : x.split('_')[0])

sns.displot(
        data = fin_pval_frame,
        x = 'p_val',
        hue = 'fit_type',
        kind = 'kde',
        fill = True,
        col = 'filt_type',
        facet_kws=dict(sharey=False)
        )
plt.savefig(os.path.join(plot_dir, 'filter_p_val_distributions.png'))
plt.close()
# plt.show()

############################################################
# Preprocessing
############################################################
# Pull out actual fit-type from fin_pval_frame
fin_pval_frame = fin_pval_frame.loc[fin_pval_frame['fit_type'] == 'actual']

# Mark rows where loglikelihood for actual fits > shuffle
fin_ll_frame['actual>shuffle'] = \
        fin_ll_frame['actual'] > fin_ll_frame['trial_shuffled']

############################################################
# Process inferred filters 
############################################################
# Only take fit_num with highest likelihood
max_ll_frame = fin_ll_frame[['fit_num','actual',*ind_names]]
max_inds = max_ll_frame.groupby(ind_names).actual.idxmax().reset_index().actual
max_vals = max_ll_frame.loc[max_inds].drop(columns = 'actual') 

fin_pval_frame = fin_pval_frame.merge(max_vals, on = ['fit_num',*ind_names])
fin_pval_frame['agg_index'] = ["_".join([str(x) for x in y]) for y in fin_pval_frame[ind_names].values]
# fin_pval_frame.dropna(inplace=True)

filter_plot_dir = os.path.join(plot_dir, 'filter_analysis')
if not os.path.exists(filter_plot_dir):
    os.makedirs(filter_plot_dir)

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

# Plot filters
fig,ax = plt.subplots(3,1, sharex=True)
dat_list = [hist_cosine_basis, stim_cosine_basis, coup_cosine_basis]
dat_names = ['History','Stimulus','Coupling']
for i in range(len(dat_list)): 
    this_ax = ax[i]
    this_dat = dat_list[i]
    x = np.arange(this_dat.shape[1])*bin_width
    this_ax.plot(x, this_dat.T, '-x')
    this_ax.set_ylabel(dat_names[i] + '\n' + f'n={len(this_dat)}')
ax[-1].set_xlabel('Time (ms)')
fig.suptitle('Cosine basis filters')
plt.savefig(os.path.join(filter_plot_dir, 'cosine_basis_filters.png'), dpi = 300, bbox_inches = 'tight')
plt.close()
# plt.show()

############################################################

class single_filter_handler:
    def __init__(self, data_frame):
        self.data_frame = data_frame

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
fig.savefig(os.path.join(filter_plot_dir, 'hist_filter_pca.png'), 
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
fig.savefig(os.path.join(filter_plot_dir, 'hist_filter_pca2_zoom.png'), dpi = 300, bbox_inches = 'tight')
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
fig.savefig(os.path.join(filter_plot_dir, 'stim_filter_pca.png'), dpi = 300, bbox_inches = 'tight')
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
fig.savefig(os.path.join(filter_plot_dir, 'stim_filter_pca2.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

############################################################

# Extract coupling filters
############################################################
coupling_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('coup')]
# coupling_frame.drop(columns = ['trial_sh','circ_sh','rand_sh'], inplace = True)
# Make sure there are no 0 pvals
coupling_frame.p_val += 1e-20

# Fraction of significant coupling filter values per threshold
alpha_vec = np.round(np.logspace(-1,-4,10),3)
frac_sig = [(coupling_frame.p_val < alpha).mean() for alpha in alpha_vec]
frac_ratio = np.round(np.array(frac_sig) / alpha_vec, 2)
print(dict(zip(alpha_vec, frac_ratio)))

plt.plot(alpha_vec, frac_sig, '-x', label = 'Actual')
# Plot x=y
plt.plot(alpha_vec, alpha_vec, '--k', label = 'Expected')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Alpha Threshold')
plt.ylabel('Significant Fraction')
ax = plt.gca()
plt.legend()
ax.set_aspect('equal')
plt.tight_layout()
plt.title('Fraction of significant coupling filter values per threshold')
plt.savefig(
        os.path.join(filter_plot_dir, 'coupling_filter_significance.png'), 
        dpi = 300, bbox_inches = 'tight')
plt.close()
#plt.show()

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
fig.savefig(os.path.join(filter_plot_dir, 'coupling_filter_pca.png'), dpi = 300, bbox_inches = 'tight')
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
fig.savefig(os.path.join(filter_plot_dir, 'coupling_filter_pca2.png'), dpi = 300, bbox_inches = 'tight')
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
fig.savefig(os.path.join(filter_plot_dir, 'coupling_filter_pca2_zoom.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)
