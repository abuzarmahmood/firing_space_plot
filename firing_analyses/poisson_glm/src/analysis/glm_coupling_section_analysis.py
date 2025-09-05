"""
Investigate whether different sections in the GLM
connectivity venn diagram have different properties.

These include:
    - Firing rates (mean post-stimulus)
    - Palatability (max rho post-stimulus)
    - Identity (max effect size post-stimulus) 
    - Responsiveness (max pre-stim to post-stim ratio)
    - Inhibitory vs excitatory connectivity
    - Connectivity strength / significance (for each lag)
    - On average, how many connections are there per neuron
        for each connection type? e.g. how many intra-region
        vs inter-region connections are there per neuron?

We can also compare properties of connections/neurons 
that are not significant vs significant.
"""
############################################################
# In the future, this code will need to be updated to
# be more modular, but for now, just run glm_couplung_analysis.py 


import sys
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/src')
from analysis.glm_coupling_analysis import *
from sklearn.decomposition import NMF
from scipy import stats
from scipy.stats import (
        chisquare, percentileofscore, f_oneway, zscore,
        spearmanr, ks_2samp, mannwhitneyu, ttest_ind
        )
from glob import glob
import json
from itertools import combinations, product
import seaborn as sns
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
from matplotlib import colors
import matplotlib.patches as mpatches
import tables
import pingouin as pg
import pickle

def calc_firing_rates(spike_array, kern):
    """
    Calculate firing rates from spike array

    Inputs:
        spike_array: array of spike times
                     shape: (n_trials, n_timepoints)   

    Outputs:
        firing_rates: array of firing rates
                      shape: (n_trials, n_timepoints)
    """
    return np.apply_along_axis(
            lambda m: np.convolve(m, kern, mode = 'same'), 
            axis = -1, 
            arr = spike_array)

def taste_oneway_anova(firing_array):
    """
    One-way ANOVA for taste discriminability

    Inputs:
        firing_array: array of firing rates for all_tastes
            shape: (n_tastes, n_trials, n_timepoints)

    Outputs:
        f_stat: F statistic for one-way ANOVA
        p_val: p-value for one-way ANOVA
    """

    try:
        temp_rates = np.moveaxis(firing_array, -1, -0)
        outs = [f_oneway(*x) for x in temp_rates]
        return list(zip(*outs))
    except:
        data_len = firing_array.shape[-1]
        return [[np.nan]*data_len, [np.nan]*data_len]

def taste_pal_corr(firing_array, ranks):
    """
    Calculate correlation between firing rates and palatability

    Inputs:
        firing_array: array of firing rates for all_tastes
            shape: (n_tastes, n_trials, n_timepoints)
        ranks: palatability rankings for all tastes

    Outputs:
        rho: correlation between firing rates and palatability
        pval: p-value of correlation
    """

    try:
        temp_rates = np.moveaxis(firing_array, -1, -0)
        ranks_array = np.broadcast_to(
                np.array(ranks)[:,None], 
                temp_rates[0].shape) 
        outs = [spearmanr(x.flatten(), ranks_array.flatten()) 
                for x in temp_rates]
        return list(zip(*outs))
    except:
        data_len = firing_array.shape[-1]
        return [[np.nan]*data_len, [np.nan]*data_len]

############################################################
coupling_analysis_plot_dir = os.path.join(plot_dir, 'coupling_analysis')
if not os.path.exists(coupling_analysis_plot_dir):
    os.makedirs(coupling_analysis_plot_dir)
############################################################
data_inds_frame = pd.read_csv(os.path.join(fin_save_path, 'data_inds_frame.csv'),
                              index_col = 0)

# To perform this analysis, we need to know connection information for 
# every neuron, that is:
#  - Significance
#  - Connection strength
#  for each connection type

# Maybe easiest way to handle this is just to have 
# connectivity matrices for each neuron per session
# As that will tell us ALL connections

# Collapse coupling frame across lags

# Change 'coeffs' to 'coeffs' to avoid confusion
coupling_frame = coupling_frame.rename(columns = {'values':'coeffs'})

############################################################
############################################################
## Connection Properties
############################################################
############################################################

group_cols = ['session','taste','neuron','other_nrn']
grouped_coupling_frame_list = \
        list(coupling_frame.groupby(group_cols))

grouped_coupling_inds = [x[0] for x in grouped_coupling_frame_list]
grouped_coupling_frames = [x[1] for x in grouped_coupling_frame_list]
p_val_list = [x['p_val'].values for x in grouped_coupling_frames]
values_list = [x['coeffs'].values for x in grouped_coupling_frames]

list_coupling_frame = pd.DataFrame(
        columns = group_cols,
        data = grouped_coupling_inds)
list_coupling_frame['p_vals'] = p_val_list
list_coupling_frame['coeffs'] = values_list

# Mark neuron and other_nrn regions

list_coupling_frame = list_coupling_frame.merge(
        unit_region_frame[['session','neuron','region']],
        left_on = ['session','neuron'],
        right_on = ['session','neuron'],
        how = 'left')

list_coupling_frame = list_coupling_frame.merge(
        unit_region_frame[['session','neuron','region']],
        left_on = ['session','other_nrn'],
        right_on = ['session','neuron'],
        how = 'left',
        suffixes = ['','_input'])

list_coupling_frame = list_coupling_frame.drop(columns = ['other_nrn'])

# Mark significant connections
base_alpha = 0.05
# Perform bonferroni correction
bonferroni_alpha = base_alpha / len(list_coupling_frame.iloc[0]['p_vals'])
any_sig = [np.any(x < bonferroni_alpha) for x in list_coupling_frame['p_vals']]

list_coupling_frame['sig'] = any_sig

# Mark connection type
list_coupling_frame['connection_type'] = \
        list_coupling_frame.region + '<-' + list_coupling_frame.region_input

# Also mark inter vs intra region
list_coupling_frame['inter_region'] = \
        list_coupling_frame.region != list_coupling_frame.region_input

# Mark significant filters
sig_inds = [np.where(x<bonferroni_alpha)[0] \
        for x in list_coupling_frame['p_vals']]
list_coupling_frame['sig_inds'] = sig_inds

# Convert values to filters (multiply by cosine basis)
list_coupling_frame['actual_filter'] = \
        [x@coup_cosine_basis for x in list_coupling_frame['coeffs']]

############################################################
# 1) Histogram of significant filter inds per connection type
############################################################

cxn_type_group = list(list_coupling_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_inds = [np.concatenate(x['sig_inds'].values) for x in cxn_type_frames]

# Plot histograms
fig, ax = plt.subplots(len(cxn_type_names),1,
                       sharex=True, sharey=True,
                       figsize = (5,10))
for ind, (cxn_type, cxn_inds) in enumerate(zip(cxn_type_names, cxn_type_inds)):
    ax[ind].hist(cxn_inds, bins = 10, density = True)
    ax[ind].set_title(cxn_type)
    ax[ind].set_ylabel('Count')
ax[-1].set_xlabel('Significant Filter Inds')
plt.suptitle('Significant Filter Inds per Connection Type')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'sig_filter_inds_per_cxn_type.png'))
plt.close()

############################################################
# 2) Histogram of significant filter inds for inter vs intra region
############################################################

# Get filter peaks
coup_filter_peaks = np.array([np.argmax(x) for x in coup_cosine_basis])+1

cxn_type_group = list(list_coupling_frame.groupby('inter_region'))
# cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_names = ['Intra Region', 'Inter Region']
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_inds = [np.concatenate(x['sig_inds'].values) for x in cxn_type_frames]

# Plot histograms
fig, ax = plt.subplots(2,1, sharex=True)
hist_outs = []
for cxn_type, cxn_inds in zip(cxn_type_names, cxn_type_inds):
    this_hist = ax[0].hist(cxn_inds, bins = np.arange(11)-0.5, label = cxn_type,
            alpha = 0.5, density = True)
    hist_outs.append(this_hist)
# Find difference between hists
diff_hist = hist_outs[0][0] - hist_outs[1][0]

# Perform bootstrapping to get confidence intervals on difference 
n_boot = int(1e4)
set_lens = [len(x) for x in cxn_type_inds]
merged_sets = np.concatenate(cxn_type_inds)
boot_sets = [[np.random.choice(merged_sets, size = this_size, replace = True) \
        for this_size in set_lens] for _ in trange(n_boot)]
boot_hists = [[np.histogram(x, bins = np.arange(11)-0.5, density=True)[0] \
        for x in this_set] \
        for this_set in boot_sets]
boot_hist_diffs = np.stack([x[0] - x[1] for x in boot_hists])

# plt.imshow(boot_hist_diffs, aspect = 'auto', cmap = 'RdBu_r')
# plt.colorbar()
# plt.show()

# Get 95% confidence intervals for difference
# conf_int = np.percentile(boot_hist_diffs, [2.5, 97.5], axis = 0)  
conf_int = np.percentile(boot_hist_diffs.flatten(), [2.5, 97.5])

# Plot difference
ax[1].bar(np.arange(len(diff_hist)), diff_hist, width = 1, color = 'k')
ax[-1].set_xticks(np.arange(len(coup_filter_peaks)), coup_filter_peaks)
# Plot gray band for confidence interval
ax[1].fill_between(np.arange(len(diff_hist)), conf_int[0], conf_int[1],
        color = 'gray', alpha = 0.5, zorder = -1, 
                   label = '95% Confidence Interval')
ax[0].set_title('Overlay')
ax[0].set_ylabel('Count')
ax[0].legend()
ax[1].legend()
ax[1].set_xlabel('Filter Peak (ms)')
ax[1].set_ylabel('Difference in Density\n' +\
        '<- Inter Region | Intra Region ->')
plt.suptitle('Significant Filter Inds per Connection Type')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'sig_filter_inds_inter_vs_intra.png'),
            bbox_inches = 'tight')
plt.close()

# Prob of summed differenes for
# Inds 0+1
# Inds 2-8
wanted_ranges = [[0,2], [2,9]]
wanted_sum_diffs = [np.sum(diff_hist[x[0]:x[1]]) for x in wanted_ranges]

# Calculate prob of getting this difference
boot_sum_diffs = [np.sum(boot_hist_diffs[:,x[0]:x[1]], axis = 1) \
        for x in wanted_ranges]

# Calculate percentiles
boot_sum_diff_perc = [percentileofscore(x, y) \
        for x,y in zip(boot_sum_diffs, wanted_sum_diffs)]

# Plot
fig, ax = plt.subplots(1,2)
for ind, (this_range, this_sum_diff, this_perc) \
        in enumerate(zip(wanted_ranges, wanted_sum_diffs, boot_sum_diff_perc)):
    ax[ind].hist(boot_sum_diffs[ind], bins = 100, label = 'Bootstrapped')
    ax[ind].axvline(this_sum_diff, color = 'k', linestyle = '--', 
                    linewidth = 2, label = 'Observed')
    ax[ind].set_title(f'Inds {this_range[0]}-{this_range[1]-1}' + \
            '\n' + f'Observed: {this_sum_diff:.3f}' + \
            '\n' + f'perc = {this_perc:.3f}')
    # Put legend on bottom of subplot
    handles, labels = ax[ind].get_legend_handles_labels()
    ax[ind].legend(handles[::-1], labels[::-1], loc = 'upper left',
            bbox_to_anchor = (0,-0.5,1,0.2))
    ax[ind].set_xlabel('Summed Difference in Density')
    ax[ind].set_ylabel('Count')
    ax[ind].set_xlim([-0.2, 0.2])
plt.suptitle('Summed Differences in Density\n' +\
        f'Bootstrapped from {n_boot} samples')
plt.tight_layout()
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'summed_diffs_inter_vs_intra.png'),
            bbox_inches = 'tight')
plt.close()

############################################################
# 3) PCA of signfiicant filters
############################################################
sig_filter_frame = list_coupling_frame[list_coupling_frame['sig']]

cxn_type_group = list(sig_filter_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_coeffs = [np.stack(x['coeffs'].values) for x in cxn_type_frames]

# Convert coeffs to filters
cxn_type_filters = [np.stack(x['actual_filter'].values) for x in cxn_type_frames]

# Perform PCA
cxn_type_pca_obj = [PCA(n_components = 3).fit(x.T) for x in cxn_type_filters]
var_explained = [x.explained_variance_ratio_ for x in cxn_type_pca_obj]

cxn_type_pca = [x.transform(data.T) for x, data in \
        zip(cxn_type_pca_obj, cxn_type_filters)]

# # Plot PCA
# fig, ax = plt.subplots(len(cxn_type_names),1,
#                        sharex=True, sharey=True,
#                        figsize = (5,10))
# for ind, (cxn_type, cxn_pca) in enumerate(zip(cxn_type_names, cxn_type_pca)):
#     for i, cxn in enumerate(cxn_pca.T):
#         ax[ind].plot(cxn, 
#                      label = f'PC{i+1}:{np.round(var_explained[ind][i],2)}',
#                      alpha = 0.7, linewidth = 2)
#     ax[ind].set_title(cxn_type)
#     ax[ind].set_ylabel('PCA Magnitude')
#     # Put legend on right of each plot
#     ax[ind].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax[ind].set_xscale('log')
#     # Plot all coup filter peaks
#     for coup_peak in coup_filter_peaks:
#         ax[ind].axvline(coup_peak, color = 'k', linestyle = '--', 
#                         alpha = 0.3, zorder = -1)
# ax[-1].set_xlabel('Time (ms)')
# plt.suptitle('PCA of Significant Filters')
# # Put legend at bottom of figure
# # fig.legend(*ax[-1].get_legend_handles_labels(), loc='lower center', ncol=3)
# plt.savefig(os.path.join(
#     coupling_analysis_plot_dir, 'sig_filter_pca.png'),
#             bbox_inches = 'tight')
# plt.close()


############################################################
# 4) Energy distribution of signfiicant filters 
############################################################
sig_filter_frame['filter_energy'] = \
        sig_filter_frame['actual_filter'].apply(np.linalg.norm)

cxn_type_group = list(sig_filter_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_energies = [x['filter_energy'].values for x in cxn_type_frames]

# # Plot histograms
# fig, ax = plt.subplots(len(cxn_type_names),1,
#                        sharex=True, sharey=True,
#                        figsize = (5,10))
# for ind, (cxn_type, cxn_energies) in \
#         enumerate(zip(cxn_type_names, cxn_type_energies)):
#     ax[ind].hist(cxn_energies, bins = 10)
#     ax[ind].set_title(cxn_type)
#     ax[ind].set_ylabel('Count')
#     ax[ind].set_yscale('log')
# ax[-1].set_xlabel('Filter Energy')
# # Plot zero line
# for a in ax:
#     a.axvline(0, color = 'k', linestyle = '--')
# plt.suptitle('Filter Significant Energy per Connection Type')
# plt.savefig(os.path.join(
#     coupling_analysis_plot_dir, 'sig_filter_energy_per_cxn_type.png'))
# plt.close()

############################################################
# 5) Filter magnitude across filter length 
############################################################
# This should agree with significant filter inds histogram
sig_filter_frame['abs_filter'] = \
        np.abs(sig_filter_frame['actual_filter'].values)

cxn_type_group = list(sig_filter_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]

# Convert coeffs to filters
cxn_type_filters = [np.stack(x['abs_filter'].values) for x in cxn_type_frames]

# Perform non-negative matrix factorization
cxn_type_nmf_obj = [NMF(n_components = 3).fit(x.T) for x in cxn_type_filters]
cxn_type_nmf = [x.transform(data.T) for x, data in \
        zip(cxn_type_nmf_obj, cxn_type_filters)]

# # Plot NMF
# fig, ax = plt.subplots(len(cxn_type_names),1,
#                        sharex=True, sharey=True,
#                        figsize = (5,10))
# for ind, (cxn_type, cxn_nmf) in enumerate(zip(cxn_type_names, cxn_type_nmf)):
#     for i, cxn in enumerate(cxn_nmf.T):
#         ax[ind].plot(cxn, label = f'NMF{i+1}', alpha = 0.7, linewidth = 2)
#     ax[ind].set_title(cxn_type)
#     ax[ind].set_ylabel('NMF Magnitude')
#     # Put legend on right of each plot
#     ax[ind].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     ax[ind].set_xscale('log')
#     # Plot all coup filter peaks
#     for coup_peak in coup_filter_peaks:
#         ax[ind].axvline(coup_peak, color = 'k', linestyle = '--', 
#                         alpha = 0.3, zorder = -1)
# ax[-1].set_xlabel('Time (ms)')
# plt.suptitle('NMF of ABSOLUTE Significant Filters')
# plt.savefig(os.path.join(
#     coupling_analysis_plot_dir, 'abs_sig_filter_nmf.png'),
#             bbox_inches = 'tight')
# plt.close()

# Also plot mean_abs_filter across cxn types
mean_abs_filter = [np.mean(x, axis = 0) for x in cxn_type_filters]
sd_abs_filter = [np.std(x, axis = 0) for x in cxn_type_filters]

# fig, ax = plt.subplots(len(cxn_type_names),1,
#                        sharex=True, sharey=True,
#                        figsize = (5,10))
# for ind, (cxn_type, cxn_mean, cxn_sd) in \
#         enumerate(zip(cxn_type_names, mean_abs_filter, sd_abs_filter)):
#     ax[ind].plot(cxn_mean, label = 'Mean')
#     ax[ind].fill_between(np.arange(len(cxn_mean)),
#                          y1 = cxn_mean - cxn_sd,
#                          y2 = cxn_mean + cxn_sd,
#                          alpha = 0.5, label = 'SD')
#     ax[ind].set_title(cxn_type)
#     ax[ind].set_ylabel('Mean Filter Magnitude')
#     ax[ind].set_xscale('log')
#     # Plot all coup filter peaks
#     for coup_peak in coup_filter_peaks:
#         ax[ind].axvline(coup_peak, color = 'k', linestyle = '--', 
#                         alpha = 0.3, zorder = -1)
# ax[-1].set_xlabel('Time (ms)')
# plt.suptitle('Mean ABSOLUTE Significant Filters')
# plt.savefig(os.path.join(
#     coupling_analysis_plot_dir, 'mean_abs_sig_filter.png'),
#             bbox_inches = 'tight')
# plt.close()

############################################################
# 6) Summed effect distribution of signfiicant filters 
############################################################
sig_filter_frame['summed_filter'] = \
        sig_filter_frame['actual_filter'].apply(np.sum)

cxn_type_group = list(sig_filter_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_energies = [x['summed_filter'].values for x in cxn_type_frames]

# # Plot histograms
# fig, ax = plt.subplots(len(cxn_type_names),1,
#                        sharex=True, sharey=True,
#                        figsize = (5,10))
# for ind, (cxn_type, cxn_energies) in \
#         enumerate(zip(cxn_type_names, cxn_type_energies)):
#     ax[ind].hist(cxn_energies, bins = 10)
#     ax[ind].set_title(cxn_type)
#     ax[ind].set_ylabel('Count')
#     ax[ind].set_yscale('log')
# ax[-1].set_xlabel('Summed Filter')
# # Plot zero line
# for a in ax:
#     a.axvline(0, color = 'k', linestyle = '--')
# plt.suptitle('Summed Significant Filter per Connection Type')
# plt.savefig(os.path.join(
#     coupling_analysis_plot_dir, 'sig_filter_sum_per_cxn_type.png'))
# plt.close()

############################################################
# 7) Coefficient distrubution of significant inds 
############################################################
# Both summed per filter, and unsummed
sig_filter_frame['sig_coeffs'] = \
        [x['coeffs'][x['sig_inds']] for \
        i,x in sig_filter_frame.iterrows()]

sig_filter_frame['summed_sig_coeffs'] = \
        sig_filter_frame['sig_coeffs'].apply(np.sum)

cxn_type_group = list(sig_filter_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_sig_coeffs = [x['sig_coeffs'].values for x in cxn_type_frames]
cxn_type_summed_sig_coeffs = [x['summed_sig_coeffs'].values \
        for x in cxn_type_frames]

############################################################
############################################################
## Neuron Properties
############################################################
############################################################

# Add min p-value to list_coupling_frame so signficance can easily
# be evaluated
list_coupling_frame['min_p_val'] = \
        [np.min(x) for x in list_coupling_frame['p_vals']]

# Generate frame containing all neurons with significant connections
wanted_cols = ['session','taste','neuron', 'region', 'neuron_input',
               'region_input','connection_type','inter_region', 'sig',
               'min_p_val']

# Primary index are receiving neurons
receive_neurons = list_coupling_frame[wanted_cols]
receive_neurons['nrn_type'] = 'receive'
receive_neurons.drop(columns = ['neuron_input','region_input'], 
                     inplace = True)

# neuron_input and region_input are sending neurons
send_neurons = list_coupling_frame[wanted_cols]
send_neurons.drop(columns = ['neuron','region'], inplace = True)
send_neurons.rename(columns = {'neuron_input':'neuron',
                               'region_input':'region'},
                    inplace = True)
send_neurons['nrn_type'] = 'send'

cxn_neurons = pd.concat([receive_neurons, send_neurons],
                            ignore_index = True)
# Replace true/false in inter_region column with inter/intra
cxn_neurons['inter_region'] = \
    cxn_neurons['inter_region'].replace(
            {True:'inter', False:'intra'})

cxn_neurons['fin_cxn_type'] = \
    cxn_neurons['region'] + '_' + \
    cxn_neurons['inter_region'] + '_' + \
    cxn_neurons['nrn_type']

cxn_neurons['nrn_id'] = \
    cxn_neurons['session'].astype('str') + '_' + \
    cxn_neurons['neuron'].astype('str')   

###############
plt.scatter(
        np.log10(cxn_neurons['min_p_val']),
        1*cxn_neurons['sig'],
        alpha = 0.5)
plt.axvline(np.log10(bonferroni_alpha), color = 'k', linestyle = '--')
plt.xlabel('Log10 Min P-Value')
plt.ylabel('Significance')
plt.show()

############################################################
############################################################

############################################################
# Venn diagrams of each connection type
############################################################
n_comparisons = np.unique(list_coupling_frame.p_vals.apply(len))[0]
# Put 0.05 at the end so downstream analyses work with 0.05
base_alpha_vec = [0.01, 0.005, 0.05]

def pop_selector(region, inter_region = None, nrn_type = None):
    region_bool = sig_cxn_neurons['region'] == region
    if inter_region is not None:
        inter_bool = sig_cxn_neurons['inter_region'] == inter_region
    else:
        inter_bool = True
    if nrn_type is not None:
        type_bool = sig_cxn_neurons['nrn_type'] == nrn_type
    else:
        type_bool = True
    bool_frame = sig_cxn_neurons.loc[region_bool & inter_bool & type_bool]
    return set(bool_frame['nrn_id'].unique())

for this_base_alpha in base_alpha_vec:
    bonf_alpha = this_base_alpha / n_comparisons

    cxn_neurons['sig'] = cxn_neurons['min_p_val'] < bonf_alpha

    sig_cxn_neurons = cxn_neurons[cxn_neurons['sig']]

    ###############
    # Inter-region connections
    gc_inter_send = pop_selector('gc','inter','send')
    gc_inter_receive = pop_selector('gc','inter','receive')
    bla_inter_send = pop_selector('bla','inter','send')
    bla_inter_receive = pop_selector('bla','inter','receive')

    gc_inter_send_receive = list(gc_inter_send & gc_inter_receive)
    bla_inter_send_receive = list(bla_inter_send & bla_inter_receive)

    gc_inter_send_only = list(set(gc_inter_send) - set(gc_inter_receive))
    gc_inter_receive_only = list(set(gc_inter_receive) - set(gc_inter_send))
    bla_inter_send_only = list(set(bla_inter_send) - set(bla_inter_receive))
    bla_inter_receive_only = list(set(bla_inter_receive) - set(bla_inter_send))


    # Intra only
    gc_inter_all = list(gc_inter_send | gc_inter_receive)
    bla_inter_all = list(bla_inter_send | bla_inter_receive)

    gc_intra = pop_selector('gc', 'intra')
    bla_intra = pop_selector('bla', 'intra')

    gc_intra_only = list(set(gc_intra) - set(gc_inter_all))
    bla_intra_only = list(set(bla_intra) - set(bla_inter_all))

    fig, ax = plt.subplots(2,1, figsize = (10,10))
    venn.venn3([set(gc_inter_send),
                set(gc_inter_receive),
                set(gc_intra)],
               set_labels = ['GC Inter Send', 'GC Inter Receive', 'GC Intra'],
               ax = ax[0])
    ax[0].set_title('GC Connection Types')
    venn.venn3([set(bla_inter_send),
                set(bla_inter_receive),
                set(bla_intra)],
               set_labels = ['BLA Inter Send', 'BLA Inter Receive', 'BLA Intra'],
               ax = ax[1])
    ax[1].set_title('BLA Connection Types\n' +\
            f'Base Alpha: {this_base_alpha}, Corrected Alpha: {bonf_alpha}')
    plt.savefig(os.path.join(
        coupling_analysis_plot_dir, f'gc_bla_cxn_types_venn_{this_base_alpha}.png'),
                bbox_inches='tight')
    plt.close()

    # Calculate probability of overlap b/w send and receive populations per region 
    gc_all = set(cxn_neurons.loc[cxn_neurons['region'] == 'gc']['nrn_id'])
    bla_all = set(cxn_neurons.loc[cxn_neurons['region'] == 'bla']['nrn_id'])

    gc_all_len = len(gc_all)
    gc_send_frac = len(gc_inter_send) / gc_all_len
    gc_receive_frac = len(gc_inter_receive) / gc_all_len
    gc_bi_frac = len(gc_inter_send_receive) / gc_all_len

    bla_all_len = len(bla_all)
    bla_send_frac = len(bla_inter_send) / bla_all_len
    bla_receive_frac = len(bla_inter_receive) / bla_all_len
    bla_bi_frac = len(bla_inter_send_receive) / bla_all_len

    n_samples = 10000

    gc_send_boot = np.random.random(size = (n_samples, gc_all_len)) < gc_send_frac
    gc_receive_boot = np.random.random(size = (n_samples, gc_all_len)) < gc_receive_frac
    gc_bi_boot = np.logical_and(gc_send_boot, gc_receive_boot) 
    gc_bi_boot_frac = np.mean(gc_bi_boot, axis = 1)

    bla_send_boot = np.random.random(size = (n_samples, bla_all_len)) < bla_send_frac
    bla_receive_boot = np.random.random(size = (n_samples, bla_all_len)) < bla_receive_frac
    bla_bi_boot = np.logical_and(bla_send_boot, bla_receive_boot)
    bla_bi_boot_frac = np.mean(bla_bi_boot, axis = 1)

    gc_bi_perc = percentileofscore(gc_bi_boot_frac, gc_bi_frac)
    bla_bi_perc = percentileofscore(bla_bi_boot_frac, bla_bi_frac)

    gc_bi_pval = ((100 - gc_bi_perc) / 100)*2
    bla_bi_pval = ((100 - bla_bi_perc) / 100)*2

    gc_bi_boot_ci = np.percentile(gc_bi_boot_frac, [2.5, 97.5])
    bla_bi_boot_ci = np.percentile(bla_bi_boot_frac, [2.5, 97.5])

    # Plot results
    fig, ax = plt.subplots(figsize = (3,5))
    ax.errorbar(['gc', 'bla'], [gc_bi_frac, bla_bi_frac],
                yerr = [[gc_bi_frac - gc_bi_boot_ci[0], bla_bi_frac - bla_bi_boot_ci[0]],
                        [gc_bi_boot_ci[1] - gc_bi_frac, bla_bi_boot_ci[1] - bla_bi_frac]],
                fmt = 'o', linestyle = '--', linewidth = 5,
                label = 'Shuffle 95% CI')
    ax.scatter(['gc', 'bla'], [gc_bi_frac, bla_bi_frac], color = 'r', s = 100,
               zorder = 10, label = 'Observed Bi-directional Fraction')
    ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, -0.3))
    ax.set_ylabel('Fraction of Bi-directional Connections')
    ax.set_title('Bi-directional Connection Fraction\n' +\
            f'GC: p = {gc_bi_pval:.3f}, BLA: p = {bla_bi_pval:.3f}' + \
            '\n' + f'Base Alpha: {this_base_alpha}, Corrected Alpha: {bonf_alpha}')
    plt.savefig(os.path.join(
        coupling_analysis_plot_dir, f'bi_directional_frac_{this_base_alpha}.png'),
                bbox_inches='tight')
    plt.close()


    ###############

    cxn_groups = list(
            sig_cxn_neurons.loc[sig_cxn_neurons['inter_region'] == 'inter']\
                    .groupby('fin_cxn_type'))
    cxn_type_names = [x[0] for x in cxn_groups]
    cxn_type_frames = [x[1] for x in cxn_groups]
    cxn_type_neurons = [x['nrn_id'].unique() for x in cxn_type_frames]


    # Plot venn diagrams
    fig,ax = plt.subplots(2,1)
    venn.venn2([set(cxn_type_neurons[0]), set(cxn_type_neurons[1])],
               set_labels = cxn_type_names[0:2],
               ax = ax[0])
    venn.venn2([set(cxn_type_neurons[2]), set(cxn_type_neurons[3])],
               set_labels = cxn_type_names[2:4],
               ax = ax[1])
    plt.suptitle('Inter-region Connection Types\n' +\
            f'Base Alpha: {this_base_alpha}, Corrected Alpha: {bonf_alpha}')
    plt.savefig(os.path.join(
        coupling_analysis_plot_dir, f'inter_region_cxn_types_venn_{this_base_alpha}.png'),
                bbox_inches='tight')
    plt.close()


    # Inter-region vs intra-region connections
    cxn_groups = list(sig_cxn_neurons.groupby(['region','inter_region']))
    cxn_type_names = [x[0] for x in cxn_groups]
    cxn_type_frames = [x[1] for x in cxn_groups]
    cxn_type_neurons = [x['nrn_id'].unique() for x in cxn_type_frames]

    # Plot venn diagrams
    fig,ax = plt.subplots(2,1)
    venn.venn2([set(cxn_type_neurons[0]), set(cxn_type_neurons[1])],
               set_labels = cxn_type_names[0:2],
               ax = ax[0])
    venn.venn2([set(cxn_type_neurons[2]), set(cxn_type_neurons[3])],
               set_labels = cxn_type_names[2:4],
               ax = ax[1])
    plt.suptitle('Inter vs Intra-region Connection Types\n' +\
            f'Base Alpha: {this_base_alpha}, Corrected Alpha: {bonf_alpha}')
    plt.savefig(os.path.join(
        coupling_analysis_plot_dir, 'inter_vs_intra_region_cxn_types_venn.png'),
                bbox_inches='tight')
    plt.close()



############################################################
# Do inter-region projection neurons have different
# encoding properties than intra-region projection neurons?
############################################################

data_inds_frame = pd.DataFrame(data_inds,
                               columns = ['session','taste','neuron'],
                               )

#     1- Firing rates (mean post-stimulus)
#     2- Responsiveness (max pre-stim to post-stim ratio)
#     3- Identity (max effect size post-stimulus) 
#     4- Palatability (max rho post-stimulus)

# Convert spikes to firing rates
kern_len = 200
kern = np.ones(kern_len)/kern_len
design_rates_list = [calc_firing_rates(x, kern) \
        for x in tqdm(design_spikes_list)]

# Calculate encoding properties for every neuron
pre_stim_lims = [250, 750]
post_stim_lims = [1000, 3000]

##############################
# Compare firing profiles of neurons in each group
mean_design_rates = np.stack([np.mean(x, axis = (0)) for x in design_rates_list])

nrn_groups = [
        gc_inter_send_only, gc_inter_receive_only, gc_inter_send_receive,
        bla_inter_send_only, bla_inter_receive_only, bla_inter_send_receive,
        gc_intra_only, bla_intra_only]
nrn_group_names = [
        'gc_inter_send_only', 'gc_inter_receive_only', 'gc_inter_send_receive',
        'bla_inter_send_only', 'bla_inter_receive_only', 'bla_inter_send_receive',
        'gc_intra_only', 'bla_intra_only']

recreate_group_rates = False
if recreate_group_rates:
    gc_inter_sig_frame = pd.concat([
        pd.DataFrame(data = x, columns = ['nrn_id']).assign(group_label = y) \
                for x,y in zip(nrn_groups[:3], nrn_group_names[:3])],
        ignore_index = True)
    data_ind_temps_frame = data_inds_frame.copy()
    data_ind_temps_frame['data_index'] = data_ind_temps_frame.index
    gc_inter_sig_frame['session'] = gc_inter_sig_frame['nrn_id'].apply(lambda x:x.split('_')[0])
    gc_inter_sig_frame['nrn'] = gc_inter_sig_frame['nrn_id'].apply(lambda x:x.split('_')[1])
    # Make all columns int64
    gc_inter_sig_frame['session'] = gc_inter_sig_frame['session'].astype('int64')
    gc_inter_sig_frame['nrn'] = gc_inter_sig_frame['nrn'].astype('int64')
    gc_inter_sig_frame = gc_inter_sig_frame.merge(
            data_ind_temps_frame,
            how = 'left',
            left_on = ['session','nrn'],
            right_on = ['session','neuron'])
    gc_inter_sig_frame.sort_values('data_index', inplace = True)

    # Get mean firing rates for each group
    wanted_rates = []
    for this_row in gc_inter_sig_frame.itertuples():
        this_nrn_id = this_row.nrn_id
        this_data_index = this_row.data_index
        this_group_label = this_row.group_label
        # this_mean_rates = mean_design_rates[this_data_index]
        this_mean_rates = design_rates_list[this_data_index] 
        wanted_rates.append(this_mean_rates)

    gc_inter_sig_frame['mean_rates'] = wanted_rates

    group_rates_dict = {}
    group_rates_dict_list = []
    for this_group_name, this_group_df in gc_inter_sig_frame.groupby('group_label'):
        group_rates_list = []
        for (this_session, this_nrn), this_df in this_group_df.groupby(['session','nrn']):
            # gc_inter_sig_frame does not have all tastes
            # Get rates for all tastes using data_inds_frame
            this_inds = data_inds_frame[
                    (data_inds_frame['session'] == this_session) & \
                    (data_inds_frame['neuron'] == this_nrn)].index
            if len(this_inds) < 4:
                print(f'Warning: {this_session}_{this_nrn} has only {len(this_inds)} tastes')
                continue
            # rate_array = np.stack([mean_design_rates[x] for x in this_inds])
            rate_array = np.stack([design_rates_list[x] for x in this_inds])
            group_rates_list.append(rate_array)
            group_rates_dict_list.append(
                    dict(
                        group = this_group_name,
                        session = this_session,
                        neuron = this_nrn,
                        rates = rate_array
                    ))
        group_rates_dict[this_group_name] = np.stack(group_rates_list)

    for key, val in group_rates_dict.items():
        print(f'{key}: {val.shape}')

    # Write out as pickle
    group_rates_save_path = os.path.join(
            coupling_analysis_plot_dir, 'group_rates_dict.pkl')
    with open(group_rates_save_path, 'wb') as f:
        pickle.dump(group_rates_dict, f)

else:
    group_rates_save_path = os.path.join(
            coupling_analysis_plot_dir, 'group_rates_dict.pkl')
    with open(group_rates_save_path, 'rb') as f:
        group_rates_dict = pickle.load(f)
    for key, val in group_rates_dict.items():
        print(f'{key}: {val.shape}')

group_rates_df = pd.DataFrame(group_rates_dict_list)

# Normalize the rates for each neuron
norm_group_rates_dict = {}
for key, val in group_rates_dict.items():
    norm_rates_list = []
    for this_nrn in val:
        norm_rates = stats.zscore(this_nrn, axis = None)
        norm_rates_list.append(norm_rates)
    norm_group_rates_dict[key] = np.stack(norm_rates_list)

# Plot all mean rates
wanted_lims = [500, 3000]
max_nrn_count = np.max([x.shape[0] for x in norm_group_rates_dict.values()])
fig, ax = plt.subplots(max_nrn_count, len(norm_group_rates_dict),
                       sharex=True, sharey=False, 
                       figsize = (10,15))
for ind, (key, val) in enumerate(norm_group_rates_dict.items()):
    for i in range(val.shape[0]):
        ax[i, ind].plot(val[i].mean(axis=1)[:, wanted_lims[0]:wanted_lims[1]].T) 
        # Mark stimulus time
        ax[i, ind].axvline(500, color = 'k', linestyle = '--')
        # Remove y labels
        ax[i, ind].set_yticklabels([])
        ax[i, ind].set_ylabel(str(i))
    ax[0, ind].set_title(key)
# plt.show()
fig.savefig(os.path.join(
    coupling_analysis_plot_dir, 'all_group_mean_rates.png'),
            bbox_inches = 'tight')
plt.close()

##############################
# Get timecourses for 
# 1) responsiveness
# 2) identity
# 3) palatability
##############################

def calc_resp_timeseries(nrn_rates, pre_stim, progress=False):
    """
    Calculate responsiveness timeseries for a neuron
    Concatenate all tastes
    Perform t-test for each timepoint
    nrn_rates: (n_tastes, n_trials, n_timepoints)
    pre_stim: numeric

    Returns:
    resp_timeseries: p-values (n_timepoints,)
    """
    cat_rates = np.concatenate(nrn_rates, axis = 0)
    pre_stim_rates = cat_rates[..., :pre_stim].flatten()
    resp_timeseries = []
    if progress:
        iterable = trange(cat_rates.shape[-1])
    else:
        iterable = range(cat_rates.shape[-1])
    for t in iterable: 
        this_time_rates = cat_rates[..., t].flatten()
        # this_time_mean = this_time_rates.mean()
        # percentile = percentileofscore(pre_stim_rates, this_time_mean)
        # # Two-tailed p-value
        # if percentile > 50:
        #     p_val = 2 * (100 - percentile) / 100
        # else:
        #     p_val = 2 * percentile / 100
        t_stat, p_val = stats.ttest_ind(this_time_rates, pre_stim_rates)
        resp_timeseries.append(p_val)
    resp_timeseries = np.array(resp_timeseries)
    return resp_timeseries

def calc_identity_timeseries(nrn_rates, progress=False):
    """
    Calculate identity timeseries for a neuron
    Perform one-way anova for each timepoint
    nrn_rates: (n_tastes, n_trials, n_timepoints)
    Returns:
        id_timeseries: p-values (n_timepoints,)
    """
    cat_rates = np.concatenate(nrn_rates, axis = 0)
    group_labels = np.concatenate([
        np.ones(x.shape[0])*i for i,x in enumerate(nrn_rates)], axis = 0)
    if progress:
        iterable = trange(cat_rates.shape[-1])
    else:
        iterable = range(cat_rates.shape[-1])
    id_timeseries = []
    for t in iterable:
        this_time_rates = cat_rates[..., t].flatten()
        groups = [this_time_rates[group_labels == i] for i in np.unique(group_labels)]
        f_stat, p_val = stats.f_oneway(*groups)
        id_timeseries.append(p_val)
    id_timeseries = np.array(id_timeseries)
    return id_timeseries

def calc_pal_timeseries(nrn_rates, pal_ranks, progress=False):
    """ 
    Calculate palatability timeseries for a neuron
    Perform spearman correlation for each timepoint
    nrn_rates: (n_tastes, n_trials, n_timepoints)
    pal_ranks: list of palatability ranks for each taste
    Returns:
        pal_timeseries: p-values (n_timepoints,)
    """
    cat_rates = np.concatenate(nrn_rates, axis = 0)
    pal_rank_vec = np.concatenate([
        np.ones(x.shape[0])*pal_ranks[i] for i,x in enumerate(nrn_rates)], axis = 0)

    if progress:
        iterable = trange(cat_rates.shape[-1])
    else:
        iterable = range(cat_rates.shape[-1])

    pal_timeseries = []
    for t in iterable:
        this_time_rates = cat_rates[..., t].flatten()
        rho, p_val = stats.spearmanr(this_time_rates, pal_rank_vec)
        pal_timeseries.append(p_val)
    pal_timeseries = np.array(pal_timeseries)
    return pal_timeseries

def get_unit_desc_timeseries(nrn_rates, pre_stim, pal_ranks):
    resp_timeseries = calc_resp_timeseries(nrn_rates, pre_stim)
    id_timeseries = calc_identity_timeseries(nrn_rates)
    pal_timeseries = calc_pal_timeseries(nrn_rates, pal_ranks)
    return resp_timeseries, id_timeseries, pal_timeseries


    fig, ax = plt.subplots(2,1, figsize = (5,5), sharex=True)
    ax[0].plot(nrn_rates.mean(axis=1).T)
    ax[1].plot(pal_timeseries)
    ax[1].plot(pal_timeseries < 0.05, color = 'k')
    plt.show()

# We will need palatability ranks for every session
file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]

# Load pal_rankings for each session
pal_rankings = []
for this_dir in tqdm(file_list):
    this_info_path = glob(os.path.join(this_dir,'*.info'))[0] 
    this_info_dict = json.load(open(this_info_path,'r')) 
    this_rankings = this_info_dict['taste_params']['pal_rankings']
    pal_rankings.append(this_rankings)

# Check that order is same for all
assert all([x == pal_rankings[0] for x in pal_rankings])

from joblib import Parallel, delayed

def paralleize(func, iterable, n_jobs = 8):
    results = Parallel(n_jobs = n_jobs)(
            delayed(func)(args) for args in tqdm(iterable))
    return results


group_rates_df['cut_rates'] = [x[...,wanted_lims[0]:wanted_lims[1]] for x in group_rates_df['rates']]

def parallelize_ts_calc(nrn_rates):
    return get_unit_desc_timeseries(nrn_rates, 500, pal_rankings[0])


# Get timeseries for each neuron
group_rates_df_path = os.path.join(
        coupling_analysis_plot_dir, 'group_rates_dict.pkl')
if not os.path.exists(group_rates_df_path):
    # desc_timeseries_list = []
    # for i, this_row in tqdm(group_rates_df.iterrows(), total = group_rates_df.shape[0]):
    #     this_session = this_row['session']
    #     this_nrn = this_row['neuron']
    #     this_rates = this_row['rates'][...,wanted_lims[0]:wanted_lims[1]]
    #     this_pal_ranks = pal_rankings[this_session]
    #     this_desc_timeseries = get_unit_desc_timeseries(
    #             this_rates, 500, this_pal_ranks)
    #     desc_timeseries_list.append(this_desc_timeseries)
    # group_rates_df['resp_timeseries'] = [x[0] for x in desc_timeseries_list]
    # group_rates_df['id_timeseries'] = [x[1] for x in desc_timeseries_list]
    # group_rates_df['pal_timeseries'] = [x[2] for x in desc_timeseries_list]
    all_ts_out = paralleize(
            parallelize_ts_calc,
            [x for x in group_rates_df['cut_rates']],
            n_jobs = 24)

    # Add to dataframe
    group_rates_df['resp_timeseries'] = [x[0] for x in all_ts_out]
    # group_rates_df['resp_timeseries'] = [calc_resp_timeseries(x, 500) for x in tqdm(group_rates_df['cut_rates'])]
    group_rates_df['id_timeseries'] = [x[1] for x in all_ts_out]
    group_rates_df['pal_timeseries'] = [x[2] for x in all_ts_out]

    # Write out group_rates_df as pickle
    with open(group_rates_df_path, 'wb') as f:
        pickle.dump(group_rates_df, f)
else:
    with open(group_rates_df_path, 'rb') as f:
        group_rates_df = pickle.load(f)

# Plot timeseries for each group
alpha = 0.05
wanted_ts = ['resp_timeseries', 'id_timeseries', 'pal_timeseries']
for this_ts in wanted_ts:
    max_nrn_count = np.max(group_rates_df.groupby('group').size())
    group_names = group_rates_df['group'].unique()
    fig, ax = plt.subplots(1, len(group_names),
                           sharex=True, sharey=True,
                           figsize = (5,5))
    for ind, (this_group, this_df) in enumerate(group_rates_df.groupby('group')):
        # Stack wanted timeseries
        this_ts_stack = np.stack(this_df[this_ts].values)
        # Pad to have max_nrn_count rows
        if this_ts_stack.shape[0] < max_nrn_count:
            temp_stack = np.empty((max_nrn_count, this_ts_stack.shape[1]))
            temp_stack[:] = np.nan
            temp_stack[:this_ts_stack.shape[0], :] = this_ts_stack
            this_ts_stack = temp_stack
        print(f'{this_group}: {this_ts_stack.shape}')
        ax[ind].imshow(this_ts_stack < alpha, aspect = 'auto',
                       interpolation = 'none',
                       vmin = 0, vmax = 1,
                       cmap = 'Greys');
        ax[ind].axvline(500, color = 'r', linestyle = '--')
        ax[ind].set_title(this_group)
    plt.suptitle(this_ts)
    fig.savefig(os.path.join(
        coupling_analysis_plot_dir, f'all_group_{this_ts}_timeseries.png'),
            bbox_inches = 'tight')
    plt.close()
# plt.show()
    
# Perform PCA on all significance timeseries and see if there are distinctions by group
from sklearn.decomposition import PCA, NMF

for this_ts in wanted_ts:
    all_ts_stack = np.stack(group_rates_df[this_ts].values) < 0.05
    pca = NMF(n_components = 3, max_iter = 1000, init = 'random', random_state = 0)
    pca_object = pca.fit(all_ts_stack)
    pca_result = pca.transform(all_ts_stack) 
    # print(f'{this_ts}: Explained variance ratios: {pca.explained_variance_ratio_}')
    # Get pca factors
    eigen_vectors = pca.components_

    # Sort pca_result by group
    group_names = group_rates_df['group'].values
    sorted_inds = np.argsort(group_names)

    sorted_group_names = group_names[sorted_inds]
    group_code_map = {name:code for code, name in enumerate(np.unique(sorted_group_names))}
    sorted_group_codes = np.array([group_code_map[x] for x in sorted_group_names])
    cmap = plt.get_cmap('brg', len(np.unique(sorted_group_names)))
    sorted_pca_result = pca_result[sorted_inds]

    fig = plt.figure(figsize = (15,5)) 
    ax = []
    ax.append(fig.add_subplot(1,4,1))
    # Second ax shares y axis with first
    ax.append(fig.add_subplot(1,4,2, sharey = ax[0]))
    ax.append(fig.add_subplot(1,4,3))
    ax[0].scatter(sorted_group_names, np.arange(sorted_group_names.shape[0]))
    ax[0].set_xticklabels(np.unique(sorted_group_names), rotation = 90)
    im = ax[1].imshow(sorted_pca_result, aspect = 'auto',
                 interpolation = 'none')
    plt.colorbar(im, ax = ax[1])
    ax[1].set_title(f'{this_ts} PCA components')
    ax[1].set_xlabel('PCA Dimension')
    for i in range(3):
        ax[2].plot(np.arange(len(eigen_vectors[i])) -  500, eigen_vectors[i], label = f'PC {i}')
        ax[2].axvline(0, color = 'k', linestyle = '--')
    ax[2].legend()
    ax[2].set_title(f'{this_ts} PCA eigenvectors')
    # Add pca scatter plot
    ax.append(fig.add_subplot(1,4,4, projection = '3d'))
    for this_code in np.unique(sorted_group_codes):
        wanted_inds = sorted_group_codes == this_code
        wanted_pca_data = sorted_pca_result[wanted_inds]
        ax[3].scatter(wanted_pca_data[:,0], wanted_pca_data[:,1], wanted_pca_data[:,2],
                      label = list(group_code_map.keys())[list(group_code_map.values()).index(this_code)],
                      c = cmap(this_code),
                      alpha = 1,
                      linewidth = 0.5, edgecolor = 'k'
                      )
    ax[3].set_xlabel('PC 1')
    ax[3].set_ylabel('PC 2')
    ax[3].set_zlabel('PC 3')
    ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # im =  ax[3].scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2],
    #               c = [cmap(x) for x in sorted_group_codes],
    #               linewidth = 0.5, edgecolor = 'k'
    #               )
    plt.suptitle(this_ts)
    plt.tight_layout()
    fig.savefig(os.path.join(
        coupling_analysis_plot_dir, f'all_group_{this_ts}_timeseries_PCA.png'),
            bbox_inches = 'tight')
    plt.close()




##############################
# Perform tensor decomposition on each group
import tensorly as tl


group_tucker_dict = {}
for key, val in norm_group_rates_dict.items():
    # val.shape = (n_neurons, n_tastes, n_timepoints)
    # Decompose to obtain taste x time factor
    wanted_val = val[...,wanted_lims[0]:wanted_lims[1]]
    # tucker_decomp = tl.decomposition.tucker(val, rank = [4, 3, 3])
    # group_tucker_dict[key] = tucker_decomp
    # print(f'{key}: core shape = {tucker_decomp[0].shape}, ' + \
    #         f'factor shapes = {[x.shape for x in tucker_decomp[1]]}')

    cp_decomp = tl.decomposition.parafac(wanted_val, rank = 3)
    print(f'{key}: CP factor shapes = {[x.shape for x in cp_decomp.factors]}')
    # Compute taste x time interaction 
    taste_time_interaction = np.tensordot(
            cp_decomp.factors[1],
            cp_decomp.factors[2],
            axes = [1,1])
    group_tucker_dict[key] = taste_time_interaction

    plt.plot(cp_decomp.factors[-1])
    plt.show()

# Plot taste x time factors
fig, ax = plt.subplots(len(group_tucker_dict),1,
                       sharex=True, 
                       figsize = (5,10))
for ind, (key, val) in enumerate(group_tucker_dict.items()):
    for i in range(val.shape[0]):
        ax[ind].plot(val[i], label = f'Taste {i+1}', alpha = 0.7, linewidth = 2)
    ax[ind].set_title(key)
    ax[ind].set_ylabel('Factor Magnitude')
    ax[ind].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


##############################
# 1- Firing rates (mean post-stimulus)
mean_post_stim_rates = [np.mean(x[:,post_stim_lims[0]:post_stim_lims[1]],
                                axis = (0,1)) for x in design_rates_list]

##############################
# 2- Responsiveness (max pre-stim to post-stim ratio)
mean_pre_stim_rates = [np.mean(x[:,pre_stim_lims[0]:pre_stim_lims[1]],
                               axis = (0,1)) for x in design_rates_list]
responsiveness = [x/y for x,y in zip(mean_post_stim_rates, mean_pre_stim_rates)]

##############################
# 3- Identity (max effect size post-stimulus)

# Get inds for every neuron
nrn_inds = data_inds_frame.groupby(['session','neuron']).groups

nrn_rates = []
for x in nrn_inds.values():
    this_rates = np.stack([design_rates_list[i] for i in x.values]) 
    nrn_rates.append(this_rates)

post_stim_nrn_rates = [x[...,post_stim_lims[0]:post_stim_lims[1]] \
        for x in nrn_rates]

discrim_save_path = os.path.join(
        coupling_analysis_plot_dir, 'discrimination_stats.npy')
if not os.path.exists(discrim_save_path):
    nrn_discrim = [taste_oneway_anova(x) for x in tqdm(post_stim_nrn_rates)]
    np.save(discrim_save_path, nrn_discrim)
else:
    nrn_discrim = np.load(discrim_save_path, allow_pickle=True)

nrn_discrim_stat, nrn_discrim_pval = list(zip(*nrn_discrim))
nrn_discrim_stat = np.stack(nrn_discrim_stat)
nrn_discrim_pval = np.stack(nrn_discrim_pval)

nrn_discrim_mean_stat = [np.mean(x) for x in nrn_discrim_stat]

##############################
# 4- Palatability (max rho post-stimulus)

pal_save_path = os.path.join(
        coupling_analysis_plot_dir, 'palatability_stats.npy')
if not os.path.exists(pal_save_path):
    nrn_pal_corr = [taste_pal_corr(x, pal_rankings[0]) \
            for x in tqdm(post_stim_nrn_rates)]
    np.save(pal_save_path, nrn_pal_corr)
else:
    nrn_pal_corr = np.load(pal_save_path, allow_pickle=True)

nrn_pal_rho, nrn_pal_pval = list(zip(*nrn_pal_corr))
nrn_pal_rho = np.stack(nrn_pal_rho)
nrn_pal_pval = np.stack(nrn_pal_pval)

# Take mean of absolute
nrn_pal_mean_rho = [np.mean(np.abs(x)) for x in nrn_pal_rho]

##############################
# Merge everything into an "encoding_frame"
# Firing rates and responsiveness will need to be averaged
# on a per-neuron basis

encoding_frame = data_inds_frame.copy()
encoding_frame['mean_post_stim_rates'] = mean_post_stim_rates
encoding_frame['responsiveness'] = responsiveness

# Average across tastes
encoding_frame = encoding_frame.groupby(['session','neuron']).mean()
encoding_frame.drop(columns = ['taste'], inplace = True)
encoding_frame.reset_index(inplace = True)

pal_iden_frame = pd.DataFrame(
        data = nrn_inds.keys(),
        columns = ['session','neuron'])
pal_iden_frame['mean_discrim_stat'] = nrn_discrim_mean_stat
pal_iden_frame['mean_pal_rho'] = nrn_pal_mean_rho

# Merge with encoding_frame
encoding_frame = encoding_frame.merge(pal_iden_frame,
                                      how = 'outer',
                                      left_on = ['session','neuron'],
                                      right_on = ['session','neuron'])
encoding_frame['nrn_id'] = encoding_frame['session'].astype('str') + '_' + \
        encoding_frame['neuron'].astype('str')

encoding_frame.dropna(inplace = True)
encoding_frame['region'] = encoding_frame['group_label'].apply(lambda x: x.split('_')[0])
encoding_frame['cxn_type'] = \
        encoding_frame['group_label'].apply(
                lambda x: "_".join(x.split('_')[1:]))

###############
# Calculate tastiness
# l2 norm of normalized discrimin and palatability
encoding_frame['log_discrim'] = np.log(encoding_frame['mean_discrim_stat'])
encoding_frame['norm_log_discrim'] = \
        (encoding_frame['log_discrim'] / np.max(encoding_frame['log_discrim']))
encoding_frame['norm_pal'] = \
        (encoding_frame['mean_pal_rho'] / np.max(encoding_frame['mean_pal_rho']))
encoding_frame['tastiness'] = \
        np.sqrt(encoding_frame['norm_log_discrim']**2 + \
                encoding_frame['norm_pal']**2)
encoding_frame['region'] = encoding_frame.region.apply(lambda x: x.upper())

discrim_pal_corr = spearmanr(
        encoding_frame['norm_log_discrim'],
        encoding_frame['norm_pal'])
# plt.scatter(
#         encoding_frame.mean_discrim_stat,
#         encoding_frame.mean_pal_rho,
#         alpha = 0.5)
fig, ax = plt.subplots(figsize = (3,5))
sns.scatterplot(data = encoding_frame,
                x = 'norm_log_discrim', 
                y = 'norm_pal',
                # hue = 'region',
                style = 'region',
                markers = True,
                s = 50,
                color = 'k',
                alpha = 0.7,
                # edgecolor = 'k',
                # linewidth = 1,
                # palette = 'grey')
)
# Make contours for tastiness
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
x_vals = np.linspace(x_min, x_max, 100)
y_vals = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.sqrt(X**2 + Y**2)
im = ax.contourf(X, Y, Z, zorder = -1, alpha = 0.5,
                 cmap = 'viridis') 
plt.colorbar(im, ax = ax, label = 'Tastiness')
# im = ax.contour(X, Y, Z, zorder = -1, alpha = 0.7,
#                 colors = 'k')
# ax.clabel(im, inline = True, fmt = '%.1f',)
# ax.set_aspect('equal')
plt.legend()
# plt.xscale('log')
plt.xlabel('Norm-Log Mean Discrimination')
plt.ylabel('Norm Palatability')
plt.suptitle(
        'Spearman Correlation between Discrimination and Palatability\n' +\
        f'Corr: {discrim_pal_corr.statistic:.3f}, p = {discrim_pal_corr.pvalue:.3f}')
# plt.show()
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'discrim_pal_correlation.svg'),
            bbox_inches = 'tight')
plt.close()

# Write out encoding_frame
encoding_frame.to_csv(os.path.join(
    coupling_analysis_plot_dir, 'encoding_frame.csv'),
    index = False)

############################################################
# For each group, add labels to encoding frame
# First check that no neuron has already been labeled
encoding_frame['group_label'] = None
for group_name, group in zip(nrn_group_names, nrn_groups):
    wanted_rows = encoding_frame.loc[encoding_frame['nrn_id'].isin(group)]
    assert all(wanted_rows['group_label'].isna())
    encoding_frame.loc[encoding_frame['nrn_id'].isin(group), 'group_label'] = group_name

# Melt encoding_frame by encoding metric
metric_list = [
        # 'mean_post_stim_rates',
        # 'responsiveness',
        'mean_discrim_stat',
        'mean_pal_rho',
        'norm_log_discrim',
        'norm_pal',
        'tastiness'
        ]
encoding_frame_melt = pd.melt(encoding_frame,
                              id_vars = ['session','neuron','nrn_id',
                                         'group_label'],
                              value_vars = metric_list,
                              var_name = 'metric',
                              value_name = 'value')
encoding_frame_melt.dropna(inplace = True)
encoding_frame_melt['region'] = \
        encoding_frame_melt['group_label'].apply(lambda x: x.split('_')[0])

# Write out encoding_frame_melt
encoding_frame_melt.to_csv(os.path.join(
    coupling_analysis_plot_dir, 'encoding_frame_melt.csv'),
    index = False)

# Plot boxen plots using sns
wanted_metrics = [
        # 'mean_post_stim_rates',
        'mean_discrim_stat',
        'mean_pal_rho',
        'tastiness'
        ]

# For each metric, calculate all p-values
out_list = []
for this_metric in wanted_metrics:
    wanted_frame = encoding_frame_melt.loc[
            encoding_frame_melt['metric'] == this_metric]
    wanted_frame = wanted_frame.loc[
            wanted_frame['region'] == 'gc']
    out = pg.pairwise_ttests(data = wanted_frame, 
                       dv = 'value', between = 'group_label',
                       )
    out['metric'] = this_metric
    out_list.append(out)
out_frame = pd.concat(out_list, ignore_index = True)
out_frame.to_csv(os.path.join(
    coupling_analysis_plot_dir, 'encoding_metric_gc_pvals.csv'),
    index = False)


for this_region in ['gc','bla']:
    wanted_frame = encoding_frame_melt.loc[
            encoding_frame_melt['region'] == this_region]
    wanted_frame = wanted_frame.loc[
            wanted_frame['metric'].isin(wanted_metrics)]
    # Remove group_labels with 'inter_send_receive'
    wanted_frame = wanted_frame.loc[
            ~wanted_frame['group_label'].str.contains('inter_send_receive')]
    g = sns.catplot(data = wanted_frame, 
                x = 'group_label', y = 'value',
                palette = [
                    'red',
                    # 'orange',
                    'green',
                    'blue'
                    ],
                order = [
                    f'{this_region}_inter_receive_only',
                    # f'{this_region}_inter_send_receive',
                    f'{this_region}_inter_send_only',
                    f'{this_region}_intra_only',
                    ],
                hue_order = [
                    f'{this_region}_inter_receive_only',
                    # f'{this_region}_inter_send_receive',
                    f'{this_region}_inter_send_only',
                    f'{this_region}_intra_only',
                    ],
                kind = 'boxen', hue = 'group_label',
                col = 'metric', 
                aspect = 2, sharey = False, showfliers = False,
                    alpha = 0.7,
                    )
    g.fig.set_size_inches(8,5)
    y_ax_labels = [
            'Discrimination',
            'Palatability',
            'Tastiness'
            ]
    for ax_ind, ax in enumerate(g.axes.flatten()):
        plt.sca(ax)
        # plt.xticks(rotation = 45, ha = 'right')
        this_title = ax.get_title()
        x_labels = [
                'Inter\nReceive',
                'Inter\nSend',
                'Intra'
                ]
        ax.set_xticklabels(x_labels)
        y_label = this_title.split('=')[1].strip()
        plt.ylabel(y_label)
        plt.title(None)
        ax.set_ylabel(y_ax_labels[ax_ind])
        ax.set_xlabel(None)
    temp_frame = wanted_frame.loc[wanted_frame.metric == 'mean_discrim_stat']
    g.fig.suptitle(f'{this_region.upper()} Encoding Metrics by Group' +\
            f'\n{temp_frame.group_label.value_counts().reset_index().values}')
    plt.tight_layout()
    plt.savefig(os.path.join(
        coupling_analysis_plot_dir, 
        f'encoding_metrics_by_group_{this_region}.svg'),
                bbox_inches = 'tight')
    plt.close()

##############################
# Close look at GC inter-receive + GC inter-send-receive
# for mean_discrim and mean_pal_rho
# wanted_metrics = ['mean_discrim_stat','mean_pal_rho']
# wanted_groups = ['gc_inter_receive_only','gc_inter_send_receive']
# wanted_region = 'gc'
# wanted_frame = encoding_frame_melt.loc[
#         (encoding_frame_melt['region'] == wanted_region) & \
#         (encoding_frame_melt['metric'].isin(wanted_metrics)) & \
#         (encoding_frame_melt['group_label'].isin(wanted_groups))]
# 
# fig, ax = plt.subplots(1, len(wanted_metrics), figsize = (5,5))
# for ind, this_metric in enumerate(wanted_metrics):
#     plot_frame = wanted_frame.loc[wanted_frame['metric'] == this_metric]
#     sns.boxenplot(data = plot_frame,
#                   x = 'group_label', y = 'value',
#                   palette = ['red','orange'],
#                   hue = 'group_label',
#                   showfliers = False,
#                   ax = ax[ind])
#     sns.stripplot(data = plot_frame,
#                   x = 'group_label', y = 'value',
#                   dodge = True,
#                   ax = ax[ind],
#                   facecolors = 'none',
#                   edgecolor = 'black',
#                   alpha = 0.5,
#                   linewidth = 2,
#                   )
#     # Perform test between groups for each metric
#     group_vals = [plot_frame.loc[plot_frame['group_label'] == x]['value'] for x in wanted_groups]
#     t_stat, p_val = ttest_ind(*group_vals)
#     ax[ind].set_title(this_metric + '\n' + f'p = {p_val:.3f}')
#     ax[ind].set_ylabel(None)
#     ax[ind].set_xlabel(None)
#     if ind == 0:
#         ax[ind].set_ylim([0, 40])
#     ax[ind].set_xticklabels(['Inter-Receive','Inter-Send-Receive'],
#                             rotation = 45, ha = 'right')
#     # Add a bit of whitespace at the top
#     ax[ind].set_ylim([0, ax[ind].get_ylim()[1]*1.1])
#     ax[ind].set_ylabel(this_metric)
# plt.suptitle('GC Inter-Receive vs Inter-Send-Receive\n' +\
#         f'{wanted_metrics}')
# plt.tight_layout()
# plt.savefig(os.path.join(
#     coupling_analysis_plot_dir, f'encoding_metrics_by_group_{wanted_region}_close.png'),
#             bbox_inches = 'tight')
# plt.close()


# Make 2D plot with each metric on the axis, and color by group
# Remove outlier

# wanted_frame = wanted_frame.loc[wanted_frame['value'] < 40]
# 
# 
# discrim_frame = wanted_frame.loc[wanted_frame['metric'] == 'mean_discrim_stat']
# pal_frame = wanted_frame.loc[wanted_frame['metric'] == 'mean_pal_rho']
# merged_frame = discrim_frame.merge(pal_frame,
#                                    how = 'outer',
#                                    on = ['session','neuron','nrn_id','group_label'])
# merged_frame.rename(columns = {'value_x':'mean_discrim_stat',
#                                'value_y':'mean_pal_rho'},
#                     inplace = True)
# merged_frame['norm_mean_discrim_stat'] = \
#         MinMaxScaler().fit_transform(merged_frame['mean_discrim_stat'].values.reshape(-1,1))
# merged_frame['norm_mean_pal_rho'] = \
#         MinMaxScaler().fit_transform(merged_frame['mean_pal_rho'].values.reshape(-1,1))
# merged_frame['norm_tastiness'] = \
#     np.linalg.norm(merged_frame[['norm_mean_discrim_stat',
#                                  'norm_mean_pal_rho']].values, axis = 1)
# norm_tastiness_group = [merged_frame.loc[merged_frame['group_label'] == x]['norm_tastiness'] 
#                         for x in wanted_groups]
# # Remove nan
# norm_tastiness_group = [x.dropna() for x in norm_tastiness_group]

wanted_frame = encoding_frame_melt.loc[
        (encoding_frame_melt['region'] == 'gc')
        ]
# Test for difference between groups
wanted_groups = ['gc_inter_receive_only','gc_inter_send_receive']
wanted_metrics = ['norm_log_discrim','norm_pal', 'tastiness']
wanted_frame = wanted_frame.loc[wanted_frame['metric'].isin(wanted_metrics)]
wanted_frame = wanted_frame.loc[wanted_frame['group_label'].isin(wanted_groups)]
# norm_tastiness_group = [x[1] for x in list(tastiness_frame.groupby('group_label')['value'])]

norm_tastiness_group = [
        x[1] for x in list(
            wanted_frame.loc[wanted_frame['metric'] == 'tastiness'].groupby('group_label')['value'])]
# Remove outliers
t_stat, p_val = ttest_ind(
        *[x[x<1.2] for x in norm_tastiness_group])

merged_frame = wanted_frame.pivot_table(
        index = ['session','neuron','nrn_id','group_label'],
        columns = 'metric',
        values = 'value').reset_index()

# fig, ax = plt.subplots(1,2, figsize = (6,6))
fig, ax = plt.subplots(figsize = (3,4))
# sns.scatterplot(data = merged_frame,
#                 x = 'mean_discrim_stat', y = 'mean_pal_rho',
#                 hue = 'group_label', ax = ax[0])
# sns.scatterplot(
#                 data = merged_frame,
#                 # x = 'norm_mean_discrim_stat', y = 'norm_mean_pal_rho',
#                 # x = 'mean_discrim_stat', y = 'mean_pal_rho',
#                 x = 'norm_log_discrim', y = 'norm_pal',
#                 hue = 'group_label', ax = ax[0],
#                 palette = ['red','orange'],
#                 s = 50,
#                 alpha = 0.5,
#                 edgecolor = 'black',
#                 linewidth = 2)
sns.boxplot(data = merged_frame,
            # x = 'group_label', y = 'norm_tastiness',
            x = 'group_label', y = 'tastiness',
            # ax = ax[1],
            ax = ax,
            hue = 'group_label',
            palette = ['red','orange'],
            linewidth = 2,
            showfliers = False)
sns.stripplot(data = merged_frame,
              x = 'group_label', y = 'tastiness',
              # ax = ax[1],
              ax = ax,
              hue = 'group_label',
              palette = ['red','orange'],
              s = 7,
              alpha = 0.5,
              edgecolor = 'black',
              linewidth = 1,
              )
ax.set_ylabel('Tastiness')
ax.set_xlabel('Group')
# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax[1].set_title(f'T-Test p-value: {p_val:.3f}')
ax.set_title(f'T-Test p-value: {p_val:.3f}')
# Add a bit of space at the top of the plot
# ax[1].set_ylim([0, ax[1].get_ylim()[1]*1.1])
ax.set_ylim([0, 1.2]) 
# Put legend on bottom
# L = ax[0].legend(loc = 'lower center', bbox_to_anchor = (0.5, 1))
# wanted_labels = ['Receive','Send+Receive']
# for i, this_text in enumerate(L.get_texts()):
#     this_text.set_text(wanted_labels[i])
# ax[1].set_xticklabels(['Inter-Receive','Inter-Send-Receive'],
ax.set_xticklabels(
    ['Inter\nReceive','Inter\nSend+Receive'],
    )
#                       rotation = 45, ha = 'right')
plt.tight_layout()
plt.subplots_adjust(top = 0.8)
plt.suptitle('GC Inter-Receive vs Inter-Send-Receive\nWithout Outlier')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, f'encoding_metrics_by_group_{wanted_region}_scatter.svg'),
            bbox_inches = 'tight')
plt.close()

############################################################
# Check enrichment of putative neuron type in GC subpopulations
############################################################
# Confirm:
# 1) Each HDF5 file has waveforms
# 2) Each HDF5 has epected number of neurons as per data_inds_frame
old_new_map_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts/single_neuron_match.csv'
old_new_map = pd.read_csv(old_new_map_path, index_col = 0)

rec_bool = encoding_frame['group_label'].str.contains('rec')

##############################
# GC
##############################
# Get number of gc_rec neurons in each session
gc_bool = encoding_frame['group_label'].str.contains('gc')
gc_bool.fillna(False, inplace = True)

# gc_rec_neurons = encoding_frame.loc[gc_bool & rec_bool]
# gc_rec_neurons = gc_rec_neurons.groupby('session')['neuron'].count().reset_index()
# gc_rec_neurons.sort_values('neuron', ascending = False, inplace = True)

wanted_cols = ['session','neuron','group_label']
gc_neurons = encoding_frame.loc[gc_bool]
gc_neurons = gc_neurons[wanted_cols]

# Cross match with old-new neuron ids

# Add session_name using unit_region_frame
gc_neurons = gc_neurons.merge(unit_region_frame[['session','neuron','basename']],
                              how = 'left',
                              on = ['session','neuron'])
gc_neurons['nrn_id'] = gc_neurons['basename'].astype('str') + '_' + \
        gc_neurons['neuron'].astype('str')

old_new_map['nrn_id'] = old_new_map['session'].astype('str') + '_' + \
        old_new_map['unit_number_old'].astype('str')

gc_neurons = gc_neurons.merge(
        old_new_map[['session','unit_number_old','unit_number_new', 'nrn_id']],
        how = 'left',
        on = 'nrn_id')

gc_neurons.dropna(inplace = True)
gc_neurons.drop(columns = ['session_y','nrn_id','unit_number_old'], inplace = True)
gc_neurons['unit_number_new'] = gc_neurons['unit_number_new'].astype(int)

##############################
# bla
##############################
# Get number of bla_rec neurons in each session
bla_bool = encoding_frame['group_label'].str.contains('bla')
bla_bool.fillna(False, inplace = True)

# bla_rec_neurons = encoding_frame.loc[bla_bool & rec_bool]
# bla_rec_neurons = bla_rec_neurons.groupby('session')['neuron'].count().reset_index()
# bla_rec_neurons.sort_values('neuron', ascending = False, inplace = True)

wanted_cols = ['session','neuron','group_label']
bla_neurons = encoding_frame.loc[bla_bool]
bla_neurons = bla_neurons[wanted_cols]

# Cross match with old-new neuron ids

# Add session_name using unit_region_frame
bla_neurons = bla_neurons.merge(unit_region_frame[['session','neuron','basename']],
                              how = 'left',
                              on = ['session','neuron'])
bla_neurons['nrn_id'] = bla_neurons['basename'].astype('str') + '_' + \
        bla_neurons['neuron'].astype('str')

old_new_map['nrn_id'] = old_new_map['session'].astype('str') + '_' + \
        old_new_map['unit_number_old'].astype('str')

bla_neurons = bla_neurons.merge(
        old_new_map[['session','unit_number_old','unit_number_new', 'nrn_id']],
        how = 'left',
        on = 'nrn_id')

bla_neurons.dropna(inplace = True)
bla_neurons.drop(columns = ['session_y','nrn_id','unit_number_old'], inplace = True)
bla_neurons['unit_number_new'] = bla_neurons['unit_number_new'].astype(int)


##############################
##############################
# For remaining neurons, pull out mean waveforms
new_data_list_path = '/media/storage/for_transfer/bla_gc/data_dir_list.txt' 
new_data_list = [x.strip() for x in open(new_data_list_path,'r').readlines()] 

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import tables

gc_session_list = []
gc_unit_list = []
gc_mean_waveform_list = []
gc_median_waveform_list = []
gc_waveform_counts_list = []
for this_session in tqdm(gc_neurons['basename'].unique()):
    this_data_path = [x for x in new_data_list if this_session in x][0]
    this_data = ephys_data(this_data_path)
    this_hdf_path = this_data.hdf5_path
    this_gc_frame = gc_neurons.loc[gc_neurons['basename'] == this_session]
    wanted_units = this_gc_frame['unit_number_new'].values
    # Convert to unit_strs
    wanted_unit_str_list = [f'unit{i:03}' for i in wanted_units]
    with tables.open_file(this_hdf_path, 'r') as h5_file:
        for this_unit, this_unit_str in zip(wanted_units, wanted_unit_str_list): 
            this_waveforms = h5_file.get_node(f'/sorted_units/{this_unit_str}/waveforms')[:]
            mean_waveform = np.mean(this_waveforms, axis = 0)
            median_waveform = np.median(this_waveforms, axis = 0)
            waveform_counts = this_waveforms.shape[0]
            gc_session_list.append(this_session)
            gc_unit_list.append(this_unit)
            gc_mean_waveform_list.append(mean_waveform)
            gc_median_waveform_list.append(median_waveform)
            gc_waveform_counts_list.append(waveform_counts)

bla_session_list = []
bla_unit_list = []
bla_mean_waveform_list = []
bla_median_waveform_list = []
bla_waveform_counts_list = []
for this_session in tqdm(bla_neurons['basename'].unique()):
    this_data_path = [x for x in new_data_list if this_session in x][0]
    this_data = ephys_data(this_data_path)
    this_hdf_path = this_data.hdf5_path
    this_bla_frame = bla_neurons.loc[bla_neurons['basename'] == this_session]
    wanted_units = this_bla_frame['unit_number_new'].values
    # Convert to unit_strs
    wanted_unit_str_list = [f'unit{i:03}' for i in wanted_units]
    with tables.open_file(this_hdf_path, 'r') as h5_file:
        for this_unit, this_unit_str in zip(wanted_units, wanted_unit_str_list): 
            this_waveforms = h5_file.get_node(f'/sorted_units/{this_unit_str}/waveforms')[:]
            mean_waveform = np.mean(this_waveforms, axis = 0)
            median_waveform = np.median(this_waveforms, axis = 0)
            waveform_counts = this_waveforms.shape[0]
            bla_session_list.append(this_session)
            bla_unit_list.append(this_unit)
            bla_mean_waveform_list.append(mean_waveform)
            bla_median_waveform_list.append(median_waveform)
            bla_waveform_counts_list.append(waveform_counts)

# Normalize waveforms using
# 1) Mean of first 10 samples == 0
# 2) Trough == -1
gc_norm_mean_waveforms_list = []
for this_mean_waveform in gc_mean_waveform_list:
    this_mean_waveform -= np.mean(this_mean_waveform[:10])
    this_mean_waveform /= -np.min(this_mean_waveform)
    gc_norm_mean_waveforms_list.append(this_mean_waveform)

bla_norm_mean_waveforms_list = []
for this_mean_waveform in bla_mean_waveform_list:
    this_mean_waveform -= np.mean(this_mean_waveform[:10])
    this_mean_waveform /= -np.min(this_mean_waveform)
    bla_norm_mean_waveforms_list.append(this_mean_waveform)

# Plot mean and median waveforms
waveform_plot_dir = os.path.join(plot_dir, 'waveforms')
if not os.path.exists(waveform_plot_dir):
    os.makedirs(waveform_plot_dir)

# Extract PC1 from waveforms
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
pca.fit(mean_waveform_list)
pc1_waveforms = pca.transform(mean_waveform_list)

# Plot overlay of normalized waveforms
fig, ax = plt.subplots(2,1, figsize = (7,10))
for this_waveform in norm_mean_waveforms_list:
    ax[0].plot(this_waveform, alpha = 0.5, c='k')
ax[0].set_title('Normalized Mean Waveforms')
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Normalized Amplitude')
ax[1].hist(pc1_waveforms, bins = 20)
ax[1].set_title('PC1 of Mean Waveforms')
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(
    waveform_plot_dir, 'norm_mean_waveforms.png'),
            bbox_inches = 'tight')
plt.close()

##############################
# gc
##############################
# For each mean_waveform, calculate trough-to-next-peak time
trough_times = []
next_peak_times = []
trough_peak_times = []
for this_mean_waveform in gc_mean_waveform_list:
    trough_ind = np.argmin(this_mean_waveform)
    next_peak_ind = np.argmax(this_mean_waveform[trough_ind:]) + trough_ind
    trough_times.append(trough_ind)
    next_peak_times.append(next_peak_ind)
    trough_peak_times.append(next_peak_ind - trough_ind)

# Sampling rate is 30 kHz
sampling_rate = 30000
trough_to_peak_ms = [x/sampling_rate*1000 for x in trough_peak_times]
unit_labels = np.array(trough_to_peak_ms) < 0.45

gc_neurons['trough_peak_time'] = trough_to_peak_ms
gc_neurons['unit_label'] = unit_labels
gc_neurons['waveform_counts'] = gc_waveform_counts_list
gc_neurons['norm_mean_waveform'] = gc_norm_mean_waveforms_list

##############################
# bla
##############################
trough_times = []
next_peak_times = []
trough_peak_times = []
for this_mean_waveform in bla_mean_waveform_list:
    trough_ind = np.argmin(this_mean_waveform)
    next_peak_ind = np.argmax(this_mean_waveform[trough_ind:]) + trough_ind
    trough_times.append(trough_ind)
    next_peak_times.append(next_peak_ind)
    trough_peak_times.append(next_peak_ind - trough_ind)

# Sampling rate is 30 kHz
sampling_rate = 30000
trough_to_peak_ms = [x/sampling_rate*1000 for x in trough_peak_times]
unit_labels = np.array(trough_to_peak_ms) < 0.45

bla_neurons['trough_peak_time'] = trough_to_peak_ms
bla_neurons['unit_label'] = unit_labels
bla_neurons['waveform_counts'] = bla_waveform_counts_list
bla_neurons['norm_mean_waveform'] = bla_norm_mean_waveforms_list


##############################
##############################
all_neurons = pd.concat([gc_neurons, bla_neurons])
all_neurons['region'] = all_neurons['group_label'].apply(lambda x: x.split('_')[0])
all_neurons['cxn_type'] = all_neurons['group_label'].apply(
        lambda x: '_'.join(x.split('_')[1:]))

# Replace unit_labels with False=Pyr, True=Int
all_neurons['unit_label'] = all_neurons['unit_label'].apply(
        lambda x: 'Pyr' if x else 'Int')

# Replace group_labels
cxn_type_dict = dict(
        inter_receive_only = 'Inter Receive',
        inter_send_receive = 'Inter Send + Receive',
        inter_send_only = 'Inter Send',
        intra_only = 'Intra')
all_neurons['cxn_type'] = all_neurons['cxn_type'].apply(
        lambda x: cxn_type_dict[x])


# Pivot tables
region_pivot = []
for this_region in ['gc','bla']:
    this_frame = all_neurons.loc[all_neurons['region'] == this_region]
    this_pivot = this_frame.pivot_table(index = 'cxn_type',
                                        columns = 'unit_label',
                                        values = 'neuron',
                                        aggfunc = 'count')
    this_pivot.fillna(0, inplace = True)
    region_pivot.append(this_pivot)

# Save to waveform_plot_dir
gc_pivot_path = os.path.join(waveform_plot_dir, 'gc_neuron_types.csv')
bla_pivot_path = os.path.join(waveform_plot_dir, 'bla_neuron_types.csv')
region_pivot[0].to_csv(gc_pivot_path)
region_pivot[1].to_csv(bla_pivot_path)

# Plot heatmap
fig, ax = plt.subplots(1,2, figsize = (7,3),)
# Annotated heatmap
vmin = np.min([region_pivot[0].values, region_pivot[1].values])
vmax = np.max([region_pivot[0].values, region_pivot[1].values])
sns.heatmap(region_pivot[0], 
            ax = ax[0], cmap = 'viridis',
            annot = True, fmt = 'g',
            vmin = vmin, vmax = vmax,
            cbar = False)
g = sns.heatmap(region_pivot[1], 
            ax = ax[1], cmap = 'viridis',
            annot = True, fmt = 'g',
            vmin = vmin, vmax = vmax,
            cbar = False)
ax[0].set_title('GC Neuron Types')
ax[1].set_title('BLA Neuron Types')
# x-axis label = 'Putative Neuron Type'
# y-axis label = 'Connection Type'
for this_ax in ax:
    this_ax.set_xlabel('Putative Neuron Type')
    this_ax.set_ylabel('Connection Type')
    this_ax.set_xticklabels(['Pyramidal','Interneuron'], rotation = 45)
    this_ax.set_yticklabels(this_ax.get_yticklabels(), rotation = 0)
plt.tight_layout()
plt.savefig(os.path.join(
    waveform_plot_dir, 'neuron_types_heatmap.svg'),
            bbox_inches = 'tight')
plt.close()

##############################
##############################
# scatter of trough_peak_times vs waveform_counts,
# with marker-type indicating region
# and hue indicating putative neuron type
bins = np.linspace(all_neurons['trough_peak_time'].min(),
                   all_neurons['trough_peak_time'].max(), 20)
# fig, ax = plt.subplots(2,1, sharex=True,
#                        figsize = (5,7))
fig = plt.figure(figsize = (5,4))
cmap = sns.color_palette('tab10')
# Add a large axis, and a small axis for histogram on top
# Share x-axes
ax = [
      fig.add_axes([0.1,0.85,0.8,0.15]),
      ]
ax.append(
        fig.add_axes([0.1,0.1,0.8,0.7]),
        )
markers = ['o','x']
for i, nrn_type in enumerate(['Pyr','Int'][::-1]):
    this_frame = all_neurons.loc[all_neurons['unit_label'] == nrn_type]
    ax[0].hist(this_frame['trough_peak_time'], bins = bins, alpha = 0.5, 
               label = nrn_type, color = cmap[i],)
    ax[0].hist(this_frame['trough_peak_time'], bins = bins, alpha = 1, 
               histtype = 'step', color = cmap[i],)
    for region_ind, region in enumerate(['gc','bla']):
        this_region_frame = this_frame.loc[this_frame['region'] == region]
        ax[1].scatter(this_region_frame['trough_peak_time'], 
                      this_region_frame['waveform_counts'],
                      marker = markers[region_ind], color = cmap[i], 
                      label = f'{region.upper()} {nrn_type}',
                      alpha = 0.7)
    # ax[1].scatter(this_frame['trough_peak_time'], this_frame['waveform_counts'],
    #               marker = markers[i], color = cmap[i], label = nrn_type)
ax[0].set_yscale('log')
# Put legend outside
ax[1].legend(loc = 'upper left', bbox_to_anchor = (1,1))
# g = sns.scatterplot(
#                 data = all_neurons,
#                 x = 'trough_peak_time', 
#                 y = 'waveform_counts',
#                 hue = 'unit_label', 
#                 style = 'region',
#                 # palette = ['red','blue'],
#                 # markers = ['o','x'],
#                 ax = ax[1])
# ax.set_xlabel('Trough-to-Peak Time (ms)')
# ax.set_ylabel('Waveform Counts')
ax[0].set_xticklabels([])
ax[0].set_ylabel('Log Count')
ax[1].set_xlabel('Trough-to-Peak Time (ms)')
ax[1].set_ylabel('Waveform Counts')
plt.tight_layout()
plt.savefig(os.path.join(
    waveform_plot_dir, 'regions_waveform_counts_vs_trough_peak_time.svg'),
            bbox_inches = 'tight')
plt.close()

# Plot histogram by itself
fig, ax = plt.subplots(figsize = (5,4))
for i, nrn_type in enumerate(['Pyr','Int'][::-1]):
    this_frame = all_neurons.loc[all_neurons['unit_label'] == nrn_type]
    # Pick oppostite label
    label = ['Pyr','Int'][i]
    ax.hist(this_frame['trough_peak_time'], bins = bins, alpha = 0.5, 
            label = label, color = cmap[i],)
    ax.hist(this_frame['trough_peak_time'], bins = bins, alpha = 1, 
            histtype = 'step', color = cmap[i], linewidth = 2)
ax.set_yscale('log')
ax.set_xlabel('Trough-to-Peak Time (ms)')
ax.set_ylabel('Log Count')
ax.legend(loc = 'upper left')
# Remoev top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(
    waveform_plot_dir, 'trough_peak_time_histogram.svg'),
            bbox_inches = 'tight')
plt.close()



##############################
##############################
gc_group_frac = gc_neurons.groupby('group_label')['unit_label'].count()
frac_inter = gc_neurons.groupby('group_label')['unit_label'].mean()
gc_group_frac = pd.DataFrame(gc_group_frac)
gc_group_frac['frac_inter'] = frac_inter
gc_group_frac.reset_index(inplace = True)
gc_group_frac.rename(columns = {'unit_label':'unit_count'}, inplace = True)
gc_group_frac['count_inter'] = gc_group_frac['unit_count'] * gc_group_frac['frac_inter']
gc_group_frac['count_inter'] = gc_group_frac['count_inter'].astype(int)
gc_group_frac.to_csv(os.path.join(
    waveform_plot_dir, 'gc_group_neuron_types.csv'), index = False)

for i in range(len(mean_waveform_list)):
    fig, ax = plt.subplots()
    ax.plot(mean_waveform_list[i], label = 'Mean')
    ax.plot(median_waveform_list[i], label = 'Median')
    ax.axvline(trough_times[i], color = 'r', linestyle = '--', label = 'Trough')
    ax.axvline(next_peak_times[i], color = 'g', linestyle = '--', label = 'Next Peak')
    ax.legend()
    ax.set_title(f'{session_list[i]}: Unit {unit_list[i]}')
    plt.savefig(os.path.join(
        waveform_plot_dir, f'{session_list[i]}_unit_{unit_list[i]}_waveforms.png'),
                bbox_inches = 'tight')
    plt.close()

# Average +/- SD waveform per cluster
group_mean_waveforms = []
group_std_waveforms = []
norm_mean_waveform_array = np.array(norm_mean_waveforms_list) 
for label in np.unique(unit_labels):
    wanted_inds = np.where(unit_labels == label)[0]
    this_mean_waveform = np.mean(norm_mean_waveform_array[wanted_inds], axis = 0)
    this_std_waveform = np.std(norm_mean_waveform_array[wanted_inds], axis = 0)
    group_mean_waveforms.append(this_mean_waveform)
    group_std_waveforms.append(this_std_waveform)

# Plot group waveforms
fig, ax = plt.subplots()
for i, (this_mean_waveform, this_std_waveform) \
        in enumerate(zip(group_mean_waveforms, group_std_waveforms)):
    ax.plot(this_mean_waveform, label = f'Group {i}')
    ax.fill_between(np.arange(len(this_mean_waveform)),
                    this_mean_waveform - this_std_waveform,
                    this_mean_waveform + this_std_waveform,
                    alpha = 0.5) 
ax.legend()
ax.set_xlabel('Sample')
ax.set_ylabel('Amplitude')
fig.suptitle('Group Waveforms\nMean +/- SD')
plt.tight_layout()
plt.savefig(os.path.join(
    waveform_plot_dir, 'group_waveforms.svg'),
            bbox_inches = 'tight')
plt.close()

# Make scatter plot of trough_peak_times vs waveform_counts
trough_to_peak_ms = np.array(trough_to_peak_ms)
waveform_counts_list = np.array(waveform_counts_list)
fig, ax = plt.subplots()
color_thresh = 0.4
above_inds = np.where(np.array(trough_to_peak_ms) > color_thresh)[0]
ax.scatter(trough_to_peak_ms[above_inds], waveform_counts_list[above_inds])
below_inds = np.where(np.array(trough_to_peak_ms) < color_thresh)[0]
ax.scatter(trough_to_peak_ms[below_inds], waveform_counts_list[below_inds])
ax.set_xlabel('Trough-to-Peak Time (ms)')
ax.set_ylabel('Waveform Counts')
plt.savefig(os.path.join(
    waveform_plot_dir, 'waveform_counts_vs_trough_peak_time.svg'),
            bbox_inches = 'tight')
plt.close()

###############
###############
# Plot waveforms for all neurons
linestyles = ['-', '-x'][::-1]
fig, ax = plt.subplots(figsize = (3,3))
for inds, this_frame in all_neurons.groupby(['region','unit_label']):
    waveforms = np.stack(this_frame['norm_mean_waveform'].values)
    mean_waveform = np.mean(waveforms, axis = 0)
    std_waveform = np.std(waveforms, axis = 0)
    if inds[0] == 'gc':
        ax.plot(mean_waveform, 
                linestyles[inds[0] == 'gc'],
                label = f'{inds[0]} {inds[1]}',
                color = cmap[inds[1] == 'Pyr'],
                # linestyle = linestyles[inds[0] == 'gc'],
                linewidth = 3,
                alpha = 0.7)
    else:
        ax.scatter(
                np.arange(len(mean_waveform)),
                mean_waveform,
                marker = 'x',
                label = f'{inds[0]} {inds[1]}',
                color = cmap[inds[1] == 'Pyr'],
                alpha = 0.7,
                s = 50)
        ax.plot(mean_waveform, 
                label = f'{inds[0]} {inds[1]}',
                color = cmap[inds[1] == 'Pyr'],
                linewidth = 1,
                alpha = 0.7)
    # ax.fill_between(np.arange(len(mean_waveform)),
    #                 mean_waveform - std_waveform,
    #                 mean_waveform + std_waveform,
    #                 alpha = 0.5,
    #                 color = cmap[inds[1] == 'Pyr'],
    #                 )
ax.legend(loc = 'upper left', bbox_to_anchor = (1,1))
ax.set_xlabel('Sample')
ax.set_ylabel('Amplitude')
plt.savefig(os.path.join(
    waveform_plot_dir, 'all_neurons_waveforms.svg'),
            bbox_inches = 'tight')
plt.close()

###############
###############
# Get waveform for all gc neurons for comparison
new_data_basenames = [x.split('/')[-1] for x in new_data_list]

all_gc_waveforms = []
for this_dir in tqdm(new_data_list):
    try:
        this_ephys_data = ephys_data(this_dir)
        this_ephys_data.get_region_units()
        gc_ind = np.where([x == 'gc' for x in this_ephys_data.region_names])[0][0]
        gc_nrn_inds = this_ephys_data.region_units[gc_ind]
        gc_nrn_strs = [f'unit{i:03}' for i in gc_nrn_inds]
        h5_path = this_ephys_data.hdf5_path
        with tables.open_file(h5_path, 'r') as h5_file:
            for this_nrn_str in gc_nrn_strs:
                this_waveform = h5_file.get_node(f'/sorted_units/{this_nrn_str}/waveforms')[:]
                mean_waveform = np.mean(this_waveform, axis = 0)
                all_gc_waveforms.append(mean_waveform)
    except:
        continue

# get trough_peak_times
trough_times = []
next_peak_times = []
all_gc_trough_peak_times = []
for this_mean_waveform in all_gc_waveforms:
    trough_ind = np.argmin(this_mean_waveform)
    next_peak_ind = np.argmax(this_mean_waveform[trough_ind:]) + trough_ind
    trough_times.append(trough_ind)
    next_peak_times.append(next_peak_ind)
    all_gc_trough_peak_times.append(next_peak_ind - trough_ind)

# Sampling rate is 30 kHz
sampling_rate = 30000
all_gc_trough_peak_ms = [x/sampling_rate*1000 for x in all_gc_trough_peak_times]

# Plot histograms for wanted_gc_neurons and all_gc_neurons
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].hist(trough_to_peak_ms, bins = 20)
ax[0].set_title(f'Wanted GC Neurons (n = {len(trough_to_peak_ms)})')
ax[1].hist(all_gc_trough_peak_ms, bins = 20)
ax[1].set_title(f'All GC Neurons, n = {len(all_gc_trough_peak_ms)}')
plt.xlabel('Trough-to-Peak Time (ms)')
plt.savefig(os.path.join(
    waveform_plot_dir, 'trough_peak_time_comparison.png'),
            bbox_inches = 'tight')
plt.close()


############################################################
# GC<-BLA neurons privileged transitions 
############################################################
# Are GC neurons receiving input from BLA privileged in 
# transition timing or coherence?
gc_neurons_all = encoding_frame.loc[gc_bool]
gc_neurons_all = gc_neurons_all[wanted_cols]
gc_neurons_all = gc_neurons_all.merge(
        unit_region_frame[['session','neuron','region','basename']],
        how = 'left',
        on = ['session','neuron'])

# Save to artifact_dir and continue analysis in new file
artifact_dir = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
gc_neurons_all.to_csv(os.path.join(
    artifact_dir, 'gc_neurons_all.csv'), index = False)

############################################################
# Convergence/Divergence of inter-region connections
############################################################
# Distribution of:
# 1) # of outgoing connections
# 2) # of incoming connections

# Only focus on inter-region connections
# inter_sig_cxn_neurons = sig_cxn_neurons[sig_cxn_neurons['inter_region'] == 'inter']
inter_region_couplings = list_coupling_frame[
        list_coupling_frame['inter_region'] == True]
sig_inter_region_couplings = inter_region_couplings[
        inter_region_couplings['sig']]

sig_inter_region_couplings['send_nrn_id'] = \
        sig_inter_region_couplings['session'].astype('str') + '_' + \
        sig_inter_region_couplings['neuron_input'].astype('str')
sig_inter_region_couplings['receive_nrn_id'] = \
        sig_inter_region_couplings['session'].astype('str') + '_' + \
        sig_inter_region_couplings['neuron'].astype('str')
sig_inter_region_couplings.drop(columns = ['inter_region','sig', 'connection_type'],
                                inplace = True)

# Also get total number of bla and gc neurons in each session
region_counts = unit_region_frame.groupby('session')['region'].value_counts().reset_index()

# Get index for each region
unit_region_frame['region_nrn_ind'] = unit_region_frame.groupby(
        ['session','region']).cumcount()

# Collapse across taste
sig_inter_region_couplings = sig_inter_region_couplings.groupby(
        ['session','send_nrn_id','receive_nrn_id']).first()
sig_inter_region_couplings.reset_index(inplace = True)
sig_inter_region_couplings.drop(
        columns = ['taste'], inplace = True)
        # columns = ['taste', 'neuron', 'neuron_input'], inplace = True)
sig_inter_region_couplings.rename(
        columns = {'region' : 'receive_region',
                   'region_input' : 'send_region',
                   'neuron' : 'receive_nrn',
                   'neuron_input' : 'send_nrn',
                   },
        inplace = True)

# # Send neurons
# Generate adjacency matrices for each session
region_matrix_list = {}
for session in sig_inter_region_couplings['session'].unique():
    this_frame = region_counts[
            region_counts['session'] == session]
    gc_count = this_frame.loc[this_frame['region'] == 'gc']['count'].values[0]
    bla_count = this_frame.loc[this_frame['region'] == 'bla']['count'].values[0]
    blank_matrix = np.zeros((gc_count, bla_count))
    region_matrix_list[session] = blank_matrix

# Fill in adjacency matrices
region_val_map = {'gc':1, 'bla':2}
for i, row in sig_inter_region_couplings.iterrows():
    session = row['session']
    send_region = row['send_region']
    receive_region = row['receive_region']
    send_nrn = row['send_nrn']
    receive_nrn = row['receive_nrn']

    # Reference absolute indices against region_nrn_ind in unit_region_frame
    send_nrn_region_ind = unit_region_frame[
            (unit_region_frame['session'] == session) & \
            (unit_region_frame['neuron'] == send_nrn)
            ]['region_nrn_ind'].values[0]
    receive_nrn_region_ind = unit_region_frame[
            (unit_region_frame['session'] == session) & \
            (unit_region_frame['neuron'] == receive_nrn)
            ]['region_nrn_ind'].values[0]


    cxn_val = region_val_map[send_region]
    coupling_tuple = (send_nrn_region_ind, receive_nrn_region_ind)
    # Since adjacency matrix is gc x bla, we need to switch the order
    # of the tuple if the connection is from bla to gc
    if cxn_val == 2:
        coupling_tuple = (receive_nrn_region_ind, send_nrn_region_ind)
    this_matrix = region_matrix_list[session]
    this_matrix[coupling_tuple] += cxn_val

# Plot adjacency matrices
adj_mat_plot_dir = os.path.join(coupling_analysis_plot_dir, 'adjacency_matrices')
if not os.path.exists(adj_mat_plot_dir):
    os.makedirs(adj_mat_plot_dir)

# Make colormap
# -1 = blue, 0 = white, 1 = red
cmap = colors.ListedColormap(['white','red','blue','purple'])

for session, adj_mat in region_matrix_list.items():
    fig, ax = plt.subplots()
    im = ax.matshow(adj_mat, cmap =cmap,
               vmin = 0, vmax = 3)
    # Create legend
    legend_obj = [mpatches.Patch(color = 'white', label = 'No connection'),
                  mpatches.Patch(color = 'red', label = 'GC send'),
                  mpatches.Patch(color = 'blue', label = 'BLA send'),
                  mpatches.Patch(color = 'purple', label = 'Bidirectional')]
    ax.legend(handles = legend_obj, loc = 'center left',
              bbox_to_anchor = (1,0.5))
    ax.set_xticks(np.arange(adj_mat.shape[1]))
    ax.set_yticks(np.arange(adj_mat.shape[0]))
    # Draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_title(f'{session} adjacency matrix')
    ax.set_xlabel('BLA Neurons')
    ax.set_ylabel('GC Neurons')
    plt.tight_layout()
    plt.savefig(os.path.join(adj_mat_plot_dir, f'{session}_adjacency_matrix.png'),
                bbox_inches = 'tight')
    plt.close()

##############################
# Analysis of adjacency matrices


# 1) Fraction of neurons in other region a single neuron projects to
send_frame_list = []
for session, adj_mat in region_matrix_list.items():
    bla_send = np.mean(adj_mat == 2, axis = 0)
    gc_send = np.mean(adj_mat == 1, axis = 1)
    bla_frame = pd.DataFrame(
            {
                'session':session,
                'type':'send',
                'region':'bla',
                'values':bla_send
                }
            )
    gc_frame = pd.DataFrame(
            {
                'session':session,
                'type':'send',
                'region':'gc',
                'values':gc_send
                }
            )
    this_send_frame = pd.concat([bla_frame, gc_frame], ignore_index = True)
    send_frame_list.append(this_send_frame)

send_frame = pd.concat(send_frame_list, ignore_index = True)

# 2) Fraction of neurons in other region a single neuron receives from
receive_frame_list = []
for session, adj_mat in region_matrix_list.items():
    bla_receive = np.mean(adj_mat == 1, axis = 0)
    gc_receive = np.mean(adj_mat == 2, axis = 1)
    bla_frame = pd.DataFrame(
            {
                'session':session,
                'type':'receive',
                'region':'bla',
                'values':bla_receive
                }
            )
    gc_frame = pd.DataFrame(
            {
                'session':session,
                'type':'receive',
                'region':'gc',
                'values':gc_receive
                }
            )
    this_receive_frame = pd.concat([bla_frame, gc_frame], ignore_index = True)
    receive_frame_list.append(this_receive_frame)

receive_frame = pd.concat(receive_frame_list, ignore_index = True)

fin_adj_frame = pd.concat([send_frame, receive_frame], ignore_index = True)

# make plots
g = sns.catplot(data = fin_adj_frame,
                x = 'region', y = 'values',
                hue = 'type', row = 'type',
                kind = 'box', sharey = True)
g.axes[0,0].set_title('Fraction of Neurons in Other \nRegion a Single Neuron Projects to')
g.axes[1,0].set_title('Fraction of Neurons in Other \nRegion a Single Neuron Receives from')
# Turn legend off
g._legend.remove()
g.fig.set_size_inches(3,5)
plt.tight_layout()
g.savefig(os.path.join(
    coupling_analysis_plot_dir, 'fraction_neuron_connections.png'),
            bbox_inches = 'tight')
plt.close()

# make plots
g = sns.catplot(data = fin_adj_frame,
                x = 'region', y = 'values',
                hue = 'type', row = 'type',
                kind = 'violin', sharey = True)
g.axes[0,0].set_title('Fraction of Neurons in Other \nRegion a Single Neuron Projects to')
g.axes[1,0].set_title('Fraction of Neurons in Other \nRegion a Single Neuron Receives from')
# Turn legend off
g._legend.remove()
g.fig.set_size_inches(3,5)
plt.tight_layout()
g.savefig(os.path.join(
    coupling_analysis_plot_dir, 'fraction_neuron_connections_violin.png'),
            bbox_inches = 'tight')
plt.close()

# Manually create the violin plot so we can overlay points on it
fig, ax = plt.subplots(2,1, figsize = (3,5), sharey = True)
for i, this_type in enumerate(fin_adj_frame['type'].unique()):
    this_data = fin_adj_frame.loc[fin_adj_frame['type'] == this_type]
    g = sns.violinplot(data = this_data,
                       x = 'region', y = 'values',
                       ax = ax[i],
                       inner = 'point',
                       hue = 'type',
                       split = True,
                       alpha = 0.5,
                       inner_kws = {'s':20, 'facecolors':'none'},
                       legend = False)
    ax[i].set_title(f'Fraction of Neurons in Other \nRegion a Single Neuron {this_type}s to')
    ax[i].set_ylabel('Fraction')
    ax[i].set_xlabel('Region')
plt.tight_layout()
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'fraction_neuron_connections_violin_overlay.png'),
            bbox_inches = 'tight')
plt.close()


# also plot other way
g = sns.catplot(data = fin_adj_frame,
                x = 'type', y = 'values',
                hue = 'region', row = 'region',
                kind = 'box', sharey = True)
# g.axes[0,0].set_title('Fraction of Neurons in Other Region a Single Neuron Projects to')
# g.axes[1,0].set_title('Fraction of Neurons in Other Region a Single Neuron Receives from')
# Move legend out
g._legend.set_bbox_to_anchor([1,0.5])
g._legend.set_loc('center left')
g.axes[1,0].set_xticklabels([
    'Fraction of other region neurons\na single neuron projects to',
    'Fraction of other region neurons\na single neuron receives from'],
                            rotation = 45, ha = 'right')
g.fig.set_size_inches(3,5)
plt.tight_layout()
g.savefig(os.path.join(
    coupling_analysis_plot_dir, 'fraction_neuron_connections_other_way.png'),
            bbox_inches = 'tight')
plt.close()

###############
# Plot schematizing each projection
# from graphviz import Digraph

# Make 2 vertical layers
# 1) GC neurons
# 2) BLA neurons



##############################
# Density of connections between input and output populations in GC
##############################
# First calculate probability of connection between intra-only
# GC neurons
sig_intra_gc_couplings = list_coupling_frame[
        (list_coupling_frame['region'] == 'gc') & \
        (list_coupling_frame['region_input'] == 'gc') & \
        (list_coupling_frame['sig'] == True)
        ]

sig_intra_gc_couplings['send_nrn_id'] = \
        sig_intra_gc_couplings['session'].astype('str') + '_' + \
        sig_intra_gc_couplings['neuron_input'].astype('str')
sig_intra_gc_couplings['receive_nrn_id'] = \
        sig_intra_gc_couplings['session'].astype('str') + '_' + \
        sig_intra_gc_couplings['neuron'].astype('str')
sig_intra_gc_couplings.drop(columns = ['inter_region','sig', 'connection_type'],
                                inplace = True)

# Collapse across taste
sig_intra_gc_couplings = sig_intra_gc_couplings.groupby(
        ['session','send_nrn_id','receive_nrn_id']).first()
sig_intra_gc_couplings.reset_index(inplace = True)
sig_intra_gc_couplings.drop(
        columns = ['taste'], inplace = True)
        # columns = ['taste', 'neuron', 'neuron_input'], inplace = True)
sig_intra_gc_couplings.rename(
        columns = {'region' : 'receive_region',
                   'region_input' : 'send_region',
                   'neuron' : 'receive_nrn',
                   'neuron_input' : 'send_nrn',
                   },
        inplace = True)

# # Send neurons
# Generate adjacency matrices for each session
gc_matrix_list = {}
for session in sig_inter_region_couplings['session'].unique():
    this_frame = region_counts[
            region_counts['session'] == session]
    gc_count = this_frame.loc[this_frame['region'] == 'gc']['count'].values[0]
    blank_matrix = np.zeros((gc_count, gc_count))
    gc_matrix_list[session] = blank_matrix

# Fill in adjacency matrices
for i, row in sig_intra_gc_couplings.iterrows():
    session = row['session']
    send_region = row['send_region']
    receive_region = row['receive_region']
    send_nrn = row['send_nrn']
    receive_nrn = row['receive_nrn']

    # Reference absolute indices against region_nrn_ind in unit_region_frame
    send_nrn_region_ind = unit_region_frame[
            (unit_region_frame['session'] == session) & \
            (unit_region_frame['neuron'] == send_nrn)
            ]['region_nrn_ind'].values[0]
    receive_nrn_region_ind = unit_region_frame[
            (unit_region_frame['session'] == session) & \
            (unit_region_frame['neuron'] == receive_nrn)
            ]['region_nrn_ind'].values[0]


    cxn_val = region_val_map[send_region]
    coupling_tuple = (send_nrn_region_ind, receive_nrn_region_ind)
    this_matrix = gc_matrix_list[session]
    this_matrix[coupling_tuple] += cxn_val

# Break these matrices down into connections between
# intra-only and send_only-receive_only neurons
gc_intra_only_nrn_frame = pd.DataFrame(
        dict(
            session = [int(x.split('_')[0]) for x in gc_intra_only],
            abs_nrn_id = [int(x.split('_')[1]) for x in gc_intra_only],
            )
        )
gc_intra_only_nrn_frame['gc_nrn_ind'] = [unit_region_frame[
                            (unit_region_frame['session'] == x) & \
                            (unit_region_frame['neuron'] == y)
                            ]['region_nrn_ind'].values[0] for x,y in \
                        zip(gc_intra_only_nrn_frame['session'],
                            gc_intra_only_nrn_frame['abs_nrn_id'])]
gc_intra_only_nrn_frame.sort_values(by = ['session','gc_nrn_ind'],
                                    inplace = True)
gc_intra_only_nrn_frame.reset_index(drop = True, inplace = True)

gc_rec_only_plus_send_only = list(set(gc_inter_send_only) | set(gc_inter_receive_only))
gc_rec_only_plus_send_only_nrn_frame = pd.DataFrame(
        dict(
            session = [int(x.split('_')[0]) for x in gc_rec_only_plus_send_only],
            abs_nrn_id = [int(x.split('_')[1]) for x in gc_rec_only_plus_send_only],
            )
        )
gc_rec_only_plus_send_only_nrn_frame['gc_nrn_ind'] = [unit_region_frame[
    (unit_region_frame['session'] == x) & \
    (unit_region_frame['neuron'] == y)
    ]['region_nrn_ind'].values[0] for x,y in \
    zip(gc_rec_only_plus_send_only_nrn_frame['session'],
        gc_rec_only_plus_send_only_nrn_frame['abs_nrn_id'])]
gc_rec_only_plus_send_only_nrn_frame.sort_values(by = ['session','gc_nrn_ind'],
                                                   inplace = True)
gc_rec_only_plus_send_only_nrn_frame.reset_index(drop = True, inplace = True)

gc_intra_only_mat_list = []
gc_send_receive_mat_list = []
for session, adj_mat in gc_matrix_list.items():

    intra_nrn_inds = gc_intra_only_nrn_frame[
            gc_intra_only_nrn_frame['session'] == session]['gc_nrn_ind'].values
    send_receive_nrn_inds = gc_rec_only_plus_send_only_nrn_frame[
            gc_rec_only_plus_send_only_nrn_frame['session'] == session]['gc_nrn_ind'].values

    this_intra_mat = adj_mat[np.ix_(intra_nrn_inds, intra_nrn_inds)]
    this_send_receive_mat = adj_mat[np.ix_(send_receive_nrn_inds, send_receive_nrn_inds)]

    gc_intra_only_mat_list.append(this_intra_mat)
    gc_send_receive_mat_list.append(this_send_receive_mat)


# Calculate probability of non-self intra-region connection
gc_intra_cxn_prob = []
for adj_mat in gc_intra_only_mat_list:
    temp_mat = adj_mat.copy()
    # Set diagonal to nan
    np.fill_diagonal(temp_mat, np.nan)
    mean_prob = np.nanmean(temp_mat, axis = 0)
    gc_intra_cxn_prob.extend(mean_prob)

gc_send_receive_cxn_prob = []
for adj_mat in gc_send_receive_mat_list:
    temp_mat = adj_mat.copy()
    # Set diagonal to nan
    np.fill_diagonal(temp_mat, np.nan)
    mean_prob = np.nanmean(temp_mat, axis = 0)
    gc_send_receive_cxn_prob.extend(mean_prob)

# Make plots
fig, ax = plt.subplots(2,1,sharex = True)
ax[0].hist(gc_intra_cxn_prob, bins = np.linspace(0,1,20)) 
ax[0].set_title('GC Intra-only Connection Probability')
ax[0].set_ylabel('Density')
ax[1].hist(gc_send_receive_cxn_prob, bins = np.linspace(0,1,20))
ax[1].set_title('GC Send-Receive Connection Probability')
ax[1].set_xlabel('Connection Probability')
ax[1].set_ylabel('Density')
plt.tight_layout()
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'gc_connection_probabilities.png'),
            bbox_inches = 'tight')
plt.close()

# Calculate bootstrapped distributions of means for each set
n_boot = 10000
gc_intra_cxn_prob_boot = [
        np.random.choice(gc_intra_cxn_prob, size = len(gc_intra_cxn_prob), replace = True) \
        for _ in range(n_boot)]
gc_send_receive_cxn_prob_boot = [
        np.random.choice(gc_send_receive_cxn_prob, size = len(gc_send_receive_cxn_prob), replace = True) \
        for _ in range(n_boot)]

gc_intra_cxn_prob_boot_means = np.mean(gc_intra_cxn_prob_boot, axis = 1)
gc_send_receive_cxn_prob_boot_means = np.mean(gc_send_receive_cxn_prob_boot, axis = 1)

# Plot bootstrapped distributions
fig, ax = plt.subplots()
ax.hist(gc_intra_cxn_prob_boot_means, bins = np.linspace(0,1,50), log = False, density = True,
        alpha = 0.5, label = 'Intra-only')
ax.hist(gc_send_receive_cxn_prob_boot_means, bins = np.linspace(0,1,50), log = False, density = True,
        alpha = 0.5, label = 'Send-Receive')
ax.set_title('GC Connection Probability Bootstrapped Means')
ax.set_xlabel('Connection Probability')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'gc_connection_probabilities_bootstrapped_means.png'),
            bbox_inches = 'tight')
plt.close()


# Plot adjacency matrices
gc_adj_mat_plot_dir = os.path.join(coupling_analysis_plot_dir, 'gc_adjacency_matrices')
if not os.path.exists(gc_adj_mat_plot_dir):
    os.makedirs(gc_adj_mat_plot_dir)

# Make colormap
for session, adj_mat in gc_matrix_list.items():
    fig, ax = plt.subplots()
    im = ax.matshow(adj_mat, cmap = 'binary',
                vmin = 0, vmax = 1)
    ax.plot([0,adj_mat.shape[1]],[0,adj_mat.shape[0]], color = 'black',
            linewidth = 1, linestyle = '--')
    # Create legend
    legend_obj = [mpatches.Patch(color = 'black', label = 'connection')]
    ax.legend(handles = legend_obj, loc = 'center left',
              bbox_to_anchor = (1,0.5))
    ax.set_xticks(np.arange(adj_mat.shape[1]))
    ax.set_yticks(np.arange(adj_mat.shape[0]))
    # Draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_title(f'{session} adjacency matrix')
    ax.set_xlabel('BLA Neurons')
    ax.set_ylabel('GC Neurons')
    plt.tight_layout()
    plt.savefig(os.path.join(gc_adj_mat_plot_dir, f'{session}_adjacency_matrix.png'),
                bbox_inches = 'tight')
    plt.close()

############################################################
############################################################
############################################################
############################################################

##############################
# Now merge wth sig_cxn_neurons


# But first, aggregate 'fin_cxn_type' in sig_cxn_neurons
inter_sig_cxn_neurons_agg = \
        inter_sig_cxn_neurons.groupby(['session','neuron','region']).agg(
        {'fin_cxn_type': lambda x: list(set(x))})
inter_sig_cxn_neurons_agg.reset_index(inplace = True)

# Convert 'fin_cxn_type' to a string
inter_sig_cxn_neurons_agg['fin_cxn_type'] = \
        inter_sig_cxn_neurons_agg['fin_cxn_type'].apply(lambda x: ' '.join(x))

# Merge with encoding_frame
inter_sig_cxn_neurons_agg = inter_sig_cxn_neurons_agg.merge(encoding_frame, 
                                      how = 'left',
                                      left_on = ['session','neuron'],
                                      right_on = ['session','neuron'])


# Aggregate by 'fin_cxn_type'
cxn_groups = list(inter_sig_cxn_neurons_agg.groupby(['fin_cxn_type']))
cxn_type_names = [x[0] for x in cxn_groups]
cxn_type_frames = [x[1] for x in cxn_groups]

# For each cxn_type plot histograms of:
# 1- firing rates
# 2- responsiveness
# 3- max discrim stat
# 4- max pal rho

data_col_names = ['mean_post_stim_rates','responsiveness',
                  'mean_discrim_stat','mean_pal_rho']

# For each variable, manually calculate bins
n_bins = 20
bin_dict = {}
for col_name in data_col_names:
    bin_dict[col_name] = np.linspace(
            np.min(inter_sig_cxn_neurons_agg[col_name]),
            np.max(inter_sig_cxn_neurons_agg[col_name]),
            n_bins)

fig, ax = plt.subplots(len(cxn_type_names),4,figsize = (20,20),
                       sharex = 'col',sharey = 'col')
for i, cxn_type in enumerate(cxn_type_frames):
    for j, col_name in enumerate(data_col_names):
        ax[i,j].hist(cxn_type[col_name], bins = bin_dict[col_name],
                     log = False, density = True)
        ax[i,j].set_title(col_name)
        ax[i,j].set_xlabel('value')
        ax[i,j].set_ylabel('count')
    ax[i,0].set_ylabel(cxn_type_names[i], fontweight = 'bold') 
fig.suptitle('Distribution of encoding metrics by connection type')
fig.tight_layout()
fig.savefig(os.path.join(
    coupling_analysis_plot_dir, 'encoding_metrics_by_cxn_type.png'),
            bbox_inches = 'tight')
plt.close(fig)

# Repeat above plot for gc only and bla only
for wanted_region in ['gc','bla']:
    fig, ax = plt.subplots(len(cxn_type_names)//2,4,figsize = (20,10),
                           sharex = 'col',sharey = 'col')
    wanted_frames_inds = [i for i,x in enumerate(cxn_type_names)\
            if wanted_region in x]
    wanted_frames = [cxn_type_frames[i] for i in wanted_frames_inds]
    wanted_cxn_names = [cxn_type_names[i] for i in wanted_frames_inds]
    for i, cxn_type in enumerate(wanted_frames):
        for j, col_name in enumerate(data_col_names):
            ax[i,j].hist(cxn_type[col_name], bins = bin_dict[col_name],
                         log = False, density = True)
            ax[i,j].set_title(col_name)
            ax[i,j].set_xlabel('value')
            ax[i,j].set_ylabel('count')
        ax[i,0].set_ylabel(wanted_cxn_names[i], fontweight = 'bold') 
    fig.suptitle(f'{wanted_region.upper()}' + \
            '\nDistribution of encoding metrics by connection type')
    fig.tight_layout()
    fig.savefig(os.path.join(
        coupling_analysis_plot_dir, 
        f'{wanted_region}_encoding_metrics_by_cxn_type.png'),
                bbox_inches = 'tight')
    plt.close(fig)

##############################
# Perform KS tests to determine if distributions are different
# for each region, for each condition
stats_frame_list = []
for wanted_region in ['gc','bla']:
    wanted_frames_inds = [i for i,x in enumerate(cxn_type_names)\
            if wanted_region in x]
    wanted_frames = [cxn_type_frames[i] for i in wanted_frames_inds]
    wanted_cxn_names = [cxn_type_names[i] for i in wanted_frames_inds]
    cxn_name_combos = list(combinations(np.arange(len(wanted_cxn_names)),2))
    run_inds = list(product(cxn_name_combos, data_col_names) )
    for this_ind in run_inds:
        frame_inds = this_ind[0]
        col_name = this_ind[1]
        dat1 = wanted_frames[frame_inds[0]][col_name]
        dat2 = wanted_frames[frame_inds[1]][col_name]
        stat, p_val = ks_2samp(dat1, dat2)
        this_stat_frame = pd.DataFrame(
                dict(
                    region = [wanted_region],
                    cxn1 = [wanted_cxn_names[frame_inds[0]]],
                    cxn2 = [wanted_cxn_names[frame_inds[1]]],
                    col_name = [col_name],
                    stat = [stat],
                    p_val = [p_val]
                    )
                )
        stats_frame_list.append(this_stat_frame)

stats_frame = pd.concat(stats_frame_list)
stats_frame['joint_comparison'] = [list(set([x.cxn1, x.cxn2])) for i,x in stats_frame.iterrows()]
alpha = 0.05
stats_frame['sig'] = stats_frame['p_val'] < alpha
# sig_frame = stats_frame[stats_frame['sig'] == True]
# sig_frame.drop(columns = ['cxn1','cxn2'], inplace = True)
# Output stats frame to csv in plot_dir
stats_frame.to_csv(os.path.join(
    coupling_analysis_plot_dir, 'encoding_metrics_by_cxn_type_stats.csv'),
                   index = False)

##############################
# For each region and variable, plot p-val matrices across comparisons
alpha = 0.05

stats_groups_list = list(stats_frame.groupby(['region','col_name']))
stats_groups_names = [x[0] for x in stats_groups_list]
stats_groups_frames = [x[1] for x in stats_groups_list]


stats_pivot_plot_dir = os.path.join(coupling_analysis_plot_dir,
                                    'encoding_metrics_by_cxn_type_stats_pivot')
if not os.path.isdir(stats_pivot_plot_dir):
    os.mkdir(stats_pivot_plot_dir)

for i in range(len(stats_groups_names)):
    this_name = stats_groups_names[i]
    this_frame = stats_groups_frames[i]

    pivot_frame = this_frame.pivot(index = 'cxn1', columns = 'cxn2', 
                                   values = 'p_val')

    fig, ax = plt.subplots(figsize = (5,5))
    # Plot heamtap with binary colormap using alpha as cutoff
    ax.imshow(pivot_frame.values, cmap = 'binary', vmin = 0, vmax = 1)
    ax.set_xticks(np.arange(len(pivot_frame.columns)))
    ax.set_yticks(np.arange(len(pivot_frame.index)))
    ax.set_xticklabels(pivot_frame.columns, rotation = 90)
    ax.set_yticklabels(pivot_frame.index)
    ax.set_title(f'{this_name[0].upper()}' + '\n' + f'{this_name[1]}')
    for i in range(len(pivot_frame.index)):
        for j in range(len(pivot_frame.columns)):
            pval = pivot_frame.values[i,j]
            if pval < alpha:
                pre_str = '*\n'
            else:
                pre_str = ''
            ax.text(j, i, pre_str + str(np.round(pval,3)), 
                    ha="center", va="center", color="r",
                    fontweight = 'bold')
    fig.tight_layout()
    fig.savefig(os.path.join(stats_pivot_plot_dir,
                             f'{this_name[0]}_{this_name[1]}_pval_matrix.png'),
                bbox_inches = 'tight')
    plt.close(fig)

##############################
# It seems like rec and rec+send are jointly different populations
# than send for GC across:
# mean_post_stim_rates, mean_discrim_stat, mean_pal_rho

# Are these variables correlated on a single neuron basis?


# Perform NCA to see if these population collectively 
gc_inter_sig_agg = inter_sig_cxn_neurons_agg.loc[\
        inter_sig_cxn_neurons_agg.region == 'gc']
gc_inter_sig_agg.dropna(inplace=True)
gc_inter_sig_agg['cxn_cat'] = gc_inter_sig_agg['fin_cxn_type'].astype('category').cat.codes

# Generate code to name map
cxn_type_names = {x:y for x,y in zip(
    gc_inter_sig_agg['cxn_cat'], gc_inter_sig_agg['fin_cxn_type'])}

X_raw = gc_inter_sig_agg[data_col_names].values 
# X = StandardScaler().fit_transform(X_raw)
X = MinMaxScaler().fit_transform(X_raw)
y = gc_inter_sig_agg['cxn_cat'].values

##############################
# Generate correlation matrix
corr_mat = np.corrcoef(X.T)

im = plt.matshow(corr_mat, cmap = 'binary')
plt.xticks(range(len(data_col_names)), data_col_names, rotation = 90)
plt.yticks(range(len(data_col_names)), data_col_names)
# Annotate with values
for i in range(len(data_col_names)):
    for j in range(len(data_col_names)):
        plt.text(j, i, str(np.round(corr_mat[i,j],2)), 
                 ha="center", va="center", color="r",
                 fontweight = 'bold')
plt.colorbar(im, label = 'Correlation')
plt.title('GC features correlation matrix')
plt.tight_layout()
plt.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_features_corr_mat.png'),
            bbox_inches = 'tight')
plt.close()

# Also make scatter plots of
# 1) firing rate vs discrimination
# 2) firing rate vs palatability
# Color by subgroup
plot_cols = ['mean_discrim_stat', 'mean_pal_rho']

# Perform robust regression and calculate explained variance score
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import HuberRegressor, LinearRegression

# Cut-off at firing < 0.02
# explained_var_scores = []
r2_score_list = []
fit_linear_model = []
for this_col in plot_cols:
    x = gc_inter_sig_agg['mean_post_stim_rates'].values
    y = gc_inter_sig_agg[this_col].values
    cut_bool = x < 0.02
    x = x[cut_bool]
    y = y[cut_bool]
    x = x[:,None]
    y = y[:,None]
    # reg = HuberRegressor()
    reg = LinearRegression()
    reg.fit(x,y)
    fit_linear_model.append(reg)
    y_pred = reg.predict(x)
    # explained_var_scores.append(explained_variance_score(y, y_pred))
    r2_score_list.append(r2_score(y, y_pred))

cmap = mpl.colors.ListedColormap(sns.color_palette("husl", 3))

for use_log in [True,False]:
    if use_log:
        add_str = 'log_'
    else:
        add_str = ''
    fig, ax = plt.subplots(1,2)
    for i in range(len(plot_cols)):
        for j in np.unique(gc_inter_sig_agg['cxn_cat'].values):
            this_frame = gc_inter_sig_agg.loc[gc_inter_sig_agg.cxn_cat == j]
            ax[i].scatter(
                this_frame['mean_post_stim_rates'],
                this_frame[plot_cols[i]],
                c = cmap(this_frame['cxn_cat'].values),
                label = cxn_type_names[j],
                alpha = 0.5)
        # Plot regression line
        this_reg = fit_linear_model[i]
        x_range = [np.min(gc_inter_sig_agg['mean_post_stim_rates'].values),
                   np.max(gc_inter_sig_agg['mean_post_stim_rates'].values)]
        x_vec = np.linspace(x_range[0], x_range[1], 100)
        y_vec = this_reg.predict(x_vec[:,None])
        ax[i].plot(x_vec, y_vec, color = 'k', linewidth = 2,
                   alpha = 0.5, linestyle = '--')
        ax[i].set_xlabel('mean_post_stim_rates')
        ax[i].set_ylabel(plot_cols[i])
        ax[i].set_title(f'GC {plot_cols[i]} vs firing rate')
        if use_log:
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
    # Legend at bottom
    ax[i].legend(list(cxn_type_names.values()), loc='upper center',
                 bbox_to_anchor=(0.5, -0.2))
    fig.suptitle('r2_scores:\n'+str(dict(zip(plot_cols, np.round(r2_score_list,2)))))
    fig.tight_layout()
    fig.savefig(os.path.join(coupling_analysis_plot_dir,
                             add_str + 'gc_features_vs_firing_scatter.png'),
                bbox_inches = 'tight')
    plt.close(fig)

    # Same plots as above but zoomed in
    fig, ax = plt.subplots(1,2)
    plot_cols = ['mean_discrim_stat', 'mean_pal_rho']
    for i in range(len(plot_cols)):
        for j in np.unique(gc_inter_sig_agg['cxn_cat'].values):
            this_frame = gc_inter_sig_agg.loc[gc_inter_sig_agg.cxn_cat == j]
            ax[i].scatter(
                this_frame['mean_post_stim_rates'],
                this_frame[plot_cols[i]],
                c = cmap(this_frame['cxn_cat'].values),
                label = cxn_type_names[j],
                alpha = 0.5)
        # Plot regression line
        this_reg = fit_linear_model[i]
        x_range = [np.min(gc_inter_sig_agg['mean_post_stim_rates'].values),
                   np.max(gc_inter_sig_agg['mean_post_stim_rates'].values)]
        x_vec = np.linspace(x_range[0], x_range[1], 100)
        y_vec = this_reg.predict(x_vec[:,None])
        ax[i].plot(x_vec, y_vec, color = 'k', linewidth = 2,
                   alpha = 0.5, linestyle = '--')
        ax[i].set_xlabel('mean_post_stim_rates')
        ax[i].set_ylabel(plot_cols[i])
        ax[i].set_title(f'GC {plot_cols[i]} vs firing rate')
        if use_log:
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
    # Legend at bottom
    ax[i].legend(list(cxn_type_names.values()), loc='upper center',
                 bbox_to_anchor=(0.5, -0.2))
    fig.suptitle('r2_scores:\n'+str(dict(zip(plot_cols, np.round(r2_score_list,2)))))
    fig.tight_layout()
    ax[0].set_xlim([-0.005,0.02])
    ax[1].set_xlim([-0.005,0.02])
    ax[0].set_ylim([-0.005,20])
    fig.savefig(os.path.join(coupling_analysis_plot_dir,
                             add_str + 'gc_features_vs_firing_scatter_zoomed.png'),
                bbox_inches = 'tight')
    plt.close(fig)


##############################

# # Project data to 1d and plot histograms
# pca = PCA(n_components=1)
# pca.fit(X)
# X_pca = pca.transform(X)
# 
# fig, ax = plt.subplots(figsize = (5,5))
# for i in range(len(np.unique(y))):
#     ax.hist(X_pca[y == i], label = cxn_type_names[i], alpha = 0.5)
# # legend on bottom
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
# ax.set_xlabel('PCA 1')
# ax.set_ylabel('Count')
# ax.set_title('PCA results')
# fig.tight_layout()
# fig.savefig(os.path.join(coupling_analysis_plot_dir,
#                          'gc_features_pca.png'),
#             bbox_inches = 'tight')
# plt.close(fig)

##############################
# Perform NCA
nca = NCA(random_state=42, n_components=2)
nca.fit(X, y)
X_embedded = nca.transform(X)

# plt.matshow(nca.components_)
# plt.show()

# Plot NCA results
fig, ax = plt.subplots(figsize = (5,5))
for i in range(len(np.unique(y))):
    ax.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], 
               label = cxn_type_names[i], s = 50, alpha = 0.5)
# legend on bottom
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
ax.set_xlabel('NCA 1')
ax.set_ylabel('NCA 2')
ax.set_title('NCA results')
ax.set_xscale('log')
ax.set_yscale('log')
fig.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_nca_results.png'),
            bbox_inches = 'tight')
plt.close(fig)

# Test against KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=3)
clf = LogisticRegression(random_state=0, max_iter = 1000)
clf.fit(X_embedded, y)
print(clf.score(X_embedded, y))
y_pred = clf.predict(X_embedded)

pred_confusion = confusion_matrix(y, y_pred,
                                  normalize = 'true')
# Plot confusion matrix
fig, ax = plt.subplots(figsize = (5,5))
im = ax.imshow(pred_confusion, cmap = 'binary')
ax.set_xticks(range(len(np.unique(y))))
ax.set_yticks(range(len(np.unique(y))))
ax.set_xticklabels([cxn_type_names[i] for i in np.unique(y)], rotation = 90)
ax.set_yticklabels([cxn_type_names[i] for i in np.unique(y)])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix')
# Annotate with values
for i in range(len(np.unique(y))):
    for j in range(len(np.unique(y))):
        ax.text(j, i, str(np.round(pred_confusion[i,j],2)), 
                 ha="center", va="center", color="r",
                 fontweight = 'bold')
fig.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_nca_confusion.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################
# Plot decision boundary
cmap = mpl.colors.ListedColormap(sns.color_palette("husl", 3))

_, ax = plt.subplots(figsize = (7,7))
DecisionBoundaryDisplay.from_estimator(
    clf,
    X_embedded,
    alpha=0.3,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    cmap=cmap,
)
for i in range(len(np.unique(y))):
    ax.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], 
               label = cxn_type_names[i], s = 50,
               edgecolors = 'k', c = cmap(i))
fig = plt.gcf()
# Legend on bottom
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('NCA 1')
ax.set_ylabel('NCA 2')
ax.set_aspect('equal')
# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
fig.suptitle('GC subpopulations decision boundaries')
plt.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_nca_decision_boundary.png'),
            bbox_inches = 'tight')
plt.close(fig)
