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

############################################################
nrn_groups = [
        gc_inter_send_only, gc_inter_receive_only, gc_inter_send_receive,
        bla_inter_send_only, bla_inter_receive_only, bla_inter_send_receive,
        gc_intra_only, bla_intra_only]
nrn_group_names = [
        'gc_inter_send_only', 'gc_inter_receive_only', 'gc_inter_send_receive',
        'bla_inter_send_only', 'bla_inter_receive_only', 'bla_inter_send_receive',
        'gc_intra_only', 'bla_intra_only']

# For each group, add labels to encoding frame
# First check that no neuron has already been labeled
encoding_frame['group_label'] = None
for group_name, group in zip(nrn_group_names, nrn_groups):
    wanted_rows = encoding_frame.loc[encoding_frame['nrn_id'].isin(group)]
    assert all(wanted_rows['group_label'].isna())
    encoding_frame.loc[encoding_frame['nrn_id'].isin(group), 'group_label'] = group_name

# Melt encoding_frame by encoding metric
metric_list = ['mean_post_stim_rates','responsiveness',
               'mean_discrim_stat','mean_pal_rho']
encoding_frame_melt = pd.melt(encoding_frame,
                              id_vars = ['session','neuron','nrn_id',
                                         'group_label'],
                              value_vars = metric_list,
                              var_name = 'metric',
                              value_name = 'value')
encoding_frame_melt.dropna(inplace = True)
encoding_frame_melt['region'] = \
        encoding_frame_melt['group_label'].apply(lambda x: x.split('_')[0])

# Plot boxen plots using sns
wanted_metrics = ['mean_post_stim_rates',
                  'mean_discrim_stat','mean_pal_rho']
for this_region in ['gc','bla']:
    wanted_frame = encoding_frame_melt.loc[
            encoding_frame_melt['region'] == this_region]
    wanted_frame = wanted_frame.loc[
            wanted_frame['metric'].isin(wanted_metrics)]
    g = sns.catplot(data = wanted_frame, 
                x = 'group_label', y = 'value',
                palette = ['red','orange','green','blue'],
                order = [
                    f'{this_region}_inter_receive_only',
                    f'{this_region}_inter_send_receive',
                    f'{this_region}_inter_send_only',
                    f'{this_region}_intra_only',
                    ],
                hue_order = [
                    f'{this_region}_inter_receive_only',
                    f'{this_region}_inter_send_receive',
                    f'{this_region}_inter_send_only',
                    f'{this_region}_intra_only',
                    ],
                kind = 'boxen', hue = 'group_label',
                col = 'metric', 
                aspect = 2, sharey = False, showfliers = False,
                    )
    g.fig.set_size_inches(8,5)
    for ax in g.axes.flatten():
        plt.sca(ax)
        plt.xticks(rotation = 45, ha = 'right')
        this_title = ax.get_title()
        y_label = this_title.split('=')[1].strip()
        plt.ylabel(y_label)
        plt.title(None)
    temp_frame = wanted_frame.loc[wanted_frame.metric == 'mean_post_stim_rates']
    g.fig.suptitle(f'{this_region.upper()} Encoding Metrics by Group' +\
            f'\n{temp_frame.group_label.value_counts().reset_index().values}')
    plt.tight_layout()
    plt.savefig(os.path.join(
        coupling_analysis_plot_dir, 
        f'encoding_metrics_by_group_{this_region}.png'),
                bbox_inches = 'tight')
    plt.close()


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
# send_nrn_counts = sig_inter_region_couplings.groupby(
#         ['session', 'send_region', 'send_nrn_id'])['receive_nrn_id'].count()
# send_nrn_counts = send_nrn_counts.reset_index()
# send_nrn_counts.rename(columns = {'receive_nrn_id':'send_count'}, inplace = True)
# send_nrn_counts = send_nrn_counts.merge(region_counts,
#                                         how = 'left',
#                                         left_on = ['session','send_region'],
#                                         right_on = ['session','region'])
# send_nrn_counts.drop(columns = ['region'], inplace = True)
# send_nrn_counts.rename(columns = {'count':'region_total_count'}, inplace = True)

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
from matplotlib import colors
import matplotlib.patches as mpatches
cmap = colors.ListedColormap(['white','red','blue','purple'])

for session, adj_mat in region_matrix_list.items():
    fig, ax = plt.subplots()
    im = ax.matshow(adj_mat, cmap =cmap, alpha = 0.7,
               vmin = 0, vmax = 3)
    # Create legend
    # legend_obj = [mpatches.Patch(color = 'blue', label = 'BLA send'),
    #               mpatches.Patch(color = 'white', label = 'No connection'),
    #               mpatches.Patch(color = 'red', label = 'GC send')]
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
                kind = 'box', sharey = False)
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

# also plot other way
g = sns.catplot(data = fin_adj_frame,
                x = 'type', y = 'values',
                hue = 'region', row = 'region',
                kind = 'box', sharey = False)
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
# First calculate probability of connection between two random neurons
# in GC




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
