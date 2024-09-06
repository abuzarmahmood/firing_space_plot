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
from scipy.stats import linregress
from sklearn.ensemble import IsolationForest
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import HuberRegressor, LinearRegression
from scipy.spatial.distance import mahalanobis, euclidean

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
# temp_coupling_frame_frame_list = \
grouped_coupling_frame_list = \
        list(coupling_frame.groupby(group_cols))

temp_coupling_frame_inds = [x[0] for x in grouped_coupling_frame_list]
temp_coupling_frame_frames = [x[1] for x in grouped_coupling_frame_list]
p_val_list = [x['p_val'].values for x in temp_coupling_frame_frames]
values_list = [x['coeffs'].values for x in temp_coupling_frame_frames]

list_coupling_frame = pd.DataFrame(
        columns = group_cols,
        data = temp_coupling_frame_inds)
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

# Find
# 1) Total number of neurons analyzed
# 2) Number of neurons with significant connections for each connection type
temp_coupling_frame = list_coupling_frame[
        ['session','taste','neuron','region','region_input', 
         'sig','inter_region', 'neuron_input']
        ]

temp_coupling_frame['send_id'] = \
        temp_coupling_frame['session'].astype('str') + '_' + \
        temp_coupling_frame['region_input'].astype('str') + '_' + \
        temp_coupling_frame['neuron_input'].astype('str')

temp_coupling_frame['receive_id'] = \
        temp_coupling_frame['session'].astype('str') + '_' + \
        temp_coupling_frame['region'].astype('str') + '_' + \
        temp_coupling_frame['neuron'].astype('str')

pooled_ids = np.concatenate(
        [temp_coupling_frame['send_id'].values,
         temp_coupling_frame['receive_id'].values])

unique_ids = np.unique(pooled_ids)
region_keys = ['gc','bla']
region_unique_ids = [[x for x in unique_ids if y in x] for y in region_keys]

# Total number of neurons per region
region_nrn_counts = [len(x) for x in region_unique_ids]
region_counts_dict = dict(zip(region_keys, region_nrn_counts))

##############################
temp_coupling_frame['cxn_type'] = \
        temp_coupling_frame['region'] + '<-' + \
        temp_coupling_frame['region_input']

sig_temp_coupling_frame = temp_coupling_frame[temp_coupling_frame.sig]
sig_temp_coupling_frame = sig_temp_coupling_frame.loc[\
        sig_temp_coupling_frame.inter_region]
sig_cxn_frame = pd.concat(
        [
        pd.DataFrame(
            dict(
                nrn_id = sig_temp_coupling_frame['send_id'].values,
                cxn_type = 'send'),
            ),
        pd.DataFrame(
            dict(
                nrn_id = sig_temp_coupling_frame['receive_id'].values,
                cxn_type = 'receive'),
            )
        ]
        )

# sig_cxn_frame['nrn_id'] = sig_cxn_frame['nrn_id'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])

sig_unique_ids = np.unique(sig_cxn_frame['nrn_id'])
sig_region_unique_ids = [[x for x in sig_unique_ids if y in x] for y in region_keys]
sig_region_nrn_counts = [len(x) for x in sig_region_unique_ids]
sig_region_counts_dict = dict(zip(region_keys, sig_region_nrn_counts))

############################################################

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

# Plot PCA
fig, ax = plt.subplots(len(cxn_type_names),1,
                       sharex=True, sharey=True,
                       figsize = (5,10))
for ind, (cxn_type, cxn_pca) in enumerate(zip(cxn_type_names, cxn_type_pca)):
    for i, cxn in enumerate(cxn_pca.T):
        ax[ind].plot(cxn, 
                     label = f'PC{i+1}:{np.round(var_explained[ind][i],2)}',
                     alpha = 0.7, linewidth = 2)
    ax[ind].set_title(cxn_type)
    ax[ind].set_ylabel('PCA Magnitude')
    # Put legend on right of each plot
    ax[ind].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[ind].set_xscale('log')
    # Plot all coup filter peaks
    for coup_peak in coup_filter_peaks:
        ax[ind].axvline(coup_peak, color = 'k', linestyle = '--', 
                        alpha = 0.3, zorder = -1)
ax[-1].set_xlabel('Time (ms)')
plt.suptitle('PCA of Significant Filters')
# Put legend at bottom of figure
# fig.legend(*ax[-1].get_legend_handles_labels(), loc='lower center', ncol=3)
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'sig_filter_pca.png'),
            bbox_inches = 'tight')
plt.close()


############################################################
# 4) Energy distribution of signfiicant filters 
############################################################
sig_filter_frame['filter_energy'] = \
        sig_filter_frame['actual_filter'].apply(np.linalg.norm)

cxn_type_group = list(sig_filter_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_energies = [x['filter_energy'].values for x in cxn_type_frames]

# Plot histograms
fig, ax = plt.subplots(len(cxn_type_names),1,
                       sharex=True, sharey=True,
                       figsize = (5,10))
for ind, (cxn_type, cxn_energies) in \
        enumerate(zip(cxn_type_names, cxn_type_energies)):
    ax[ind].hist(cxn_energies, bins = 10)
    ax[ind].set_title(cxn_type)
    ax[ind].set_ylabel('Count')
    ax[ind].set_yscale('log')
ax[-1].set_xlabel('Filter Energy')
# Plot zero line
for a in ax:
    a.axvline(0, color = 'k', linestyle = '--')
plt.suptitle('Filter Significant Energy per Connection Type')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'sig_filter_energy_per_cxn_type.png'))
plt.close()

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

# Plot NMF
fig, ax = plt.subplots(len(cxn_type_names),1,
                       sharex=True, sharey=True,
                       figsize = (5,10))
for ind, (cxn_type, cxn_nmf) in enumerate(zip(cxn_type_names, cxn_type_nmf)):
    for i, cxn in enumerate(cxn_nmf.T):
        ax[ind].plot(cxn, label = f'NMF{i+1}', alpha = 0.7, linewidth = 2)
    ax[ind].set_title(cxn_type)
    ax[ind].set_ylabel('NMF Magnitude')
    # Put legend on right of each plot
    ax[ind].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[ind].set_xscale('log')
    # Plot all coup filter peaks
    for coup_peak in coup_filter_peaks:
        ax[ind].axvline(coup_peak, color = 'k', linestyle = '--', 
                        alpha = 0.3, zorder = -1)
ax[-1].set_xlabel('Time (ms)')
plt.suptitle('NMF of ABSOLUTE Significant Filters')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'abs_sig_filter_nmf.png'),
            bbox_inches = 'tight')
plt.close()

# Also plot mean_abs_filter across cxn types
mean_abs_filter = [np.mean(x, axis = 0) for x in cxn_type_filters]
sd_abs_filter = [np.std(x, axis = 0) for x in cxn_type_filters]

fig, ax = plt.subplots(len(cxn_type_names),1,
                       sharex=True, sharey=True,
                       figsize = (5,10))
for ind, (cxn_type, cxn_mean, cxn_sd) in \
        enumerate(zip(cxn_type_names, mean_abs_filter, sd_abs_filter)):
    ax[ind].plot(cxn_mean, label = 'Mean')
    ax[ind].fill_between(np.arange(len(cxn_mean)),
                         y1 = cxn_mean - cxn_sd,
                         y2 = cxn_mean + cxn_sd,
                         alpha = 0.5, label = 'SD')
    ax[ind].set_title(cxn_type)
    ax[ind].set_ylabel('Mean Filter Magnitude')
    ax[ind].set_xscale('log')
    # Plot all coup filter peaks
    for coup_peak in coup_filter_peaks:
        ax[ind].axvline(coup_peak, color = 'k', linestyle = '--', 
                        alpha = 0.3, zorder = -1)
ax[-1].set_xlabel('Time (ms)')
plt.suptitle('Mean ABSOLUTE Significant Filters')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'mean_abs_sig_filter.png'),
            bbox_inches = 'tight')
plt.close()

############################################################
# 6) Summed effect distribution of signfiicant filters 
############################################################
sig_filter_frame['summed_filter'] = \
        sig_filter_frame['actual_filter'].apply(np.sum)

cxn_type_group = list(sig_filter_frame.groupby('connection_type'))
cxn_type_names = [x[0] for x in cxn_type_group]
cxn_type_frames = [x[1] for x in cxn_type_group]
cxn_type_energies = [x['summed_filter'].values for x in cxn_type_frames]

# Plot histograms
fig, ax = plt.subplots(len(cxn_type_names),1,
                       sharex=True, sharey=True,
                       figsize = (5,10))
for ind, (cxn_type, cxn_energies) in \
        enumerate(zip(cxn_type_names, cxn_type_energies)):
    ax[ind].hist(cxn_energies, bins = 10)
    ax[ind].set_title(cxn_type)
    ax[ind].set_ylabel('Count')
    ax[ind].set_yscale('log')
ax[-1].set_xlabel('Summed Filter')
# Plot zero line
for a in ax:
    a.axvline(0, color = 'k', linestyle = '--')
plt.suptitle('Summed Significant Filter per Connection Type')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'sig_filter_sum_per_cxn_type.png'))
plt.close()

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

# Plot histograms
# Left column is summed, right column is unsummed
fig, ax = plt.subplots(len(cxn_type_names),2,
                       sharex=True, sharey=True,
                       figsize = (5,10))
for ind, (cxn_type, cxn_sig_coeffs, cxn_summed_sig_coeffs) in \
        enumerate(zip(cxn_type_names, cxn_type_sig_coeffs,
            cxn_type_summed_sig_coeffs)):
    ax[ind][0].hist(cxn_summed_sig_coeffs)
    ax[ind][0].set_title(cxn_type)
    ax[ind][0].set_ylabel('Count')
    ax[ind][0].set_yscale('log')
    ax[ind][1].hist(np.concatenate(cxn_sig_coeffs))
    ax[ind][1].set_title(cxn_type)
    ax[ind][1].set_ylabel('Count')
    ax[ind][1].set_yscale('log')
ax[-1][0].set_xlabel('Summed Significant Coefficients')
ax[-1][1].set_xlabel('Significant Coefficients')
# Plot 0 line
for this_ax in ax.flatten():
    this_ax.axvline(0, color = 'k', linestyle = '--')
plt.suptitle('Significant Coefficients per Connection Type')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'sig_coeffs_per_cxn_type.png'),
            bbox_inches='tight')
plt.close()

############################################################
############################################################
## Neuron Properties
############################################################
############################################################

# Generate frame containing all neurons with significant connections
wanted_cols = ['session','taste','neuron', 'region', 'neuron_input',
               'region_input','connection_type','inter_region']
sig_receive_neurons = sig_filter_frame[wanted_cols]
sig_receive_neurons['nrn_type'] = 'receive'
sig_receive_neurons.drop(columns = ['neuron_input','region_input'], 
                         inplace = True)

sig_send_neurons = sig_filter_frame[wanted_cols]
sig_send_neurons.drop(columns = ['neuron','region'], inplace = True)
sig_send_neurons.rename(columns = {'neuron_input':'neuron',
                                   'region_input':'region'},
                        inplace = True)
sig_send_neurons['nrn_type'] = 'send'

sig_cxn_neurons = pd.concat([sig_receive_neurons, sig_send_neurons],
                            ignore_index = True)
# Replace true/false in inter_region column with inter/intra
sig_cxn_neurons['inter_region'] = \
        sig_cxn_neurons['inter_region'].replace(
                {True:'inter', False:'intra'})

sig_cxn_neurons['fin_cxn_type'] = \
        sig_cxn_neurons['region'] + '_' + \
        sig_cxn_neurons['inter_region'] + '_' + \
        sig_cxn_neurons['nrn_type']

sig_cxn_neurons['nrn_id'] = \
        sig_cxn_neurons['session'].astype('str') + '_' + \
        sig_cxn_neurons['neuron'].astype('str')   

############################################################
# Venn diagrams of each connection type
############################################################
# Only iner-region connections
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
plt.suptitle('Inter-region Connection Types')
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'inter_region_cxn_types_venn.png'),
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
plt.suptitle('Inter vs Intra-region Connection Types')
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

##############################
# Now merge wth sig_cxn_neurons

# Only focus on inter-region connections
 inter_sig_cxn_neurons = sig_cxn_neurons[sig_cxn_neurons['inter_region'] == 'inter']

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
cxn_type_names = [x[0][0] for x in cxn_groups]
cxn_type_frames = [x[1] for x in cxn_groups]

# For each cxn_type plot histograms of:
# 1- firing rates
# 2- responsiveness
# 3- max discrim stat
# 4- max pal rho

# data_col_names = ['mean_post_stim_rates','responsiveness',
#                   'mean_discrim_stat','mean_pal_rho']
data_col_names = ['mean_post_stim_rates',
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

# Create barplots instead of distributions
gc_inter_sig_cxn_neurons_agg = inter_sig_cxn_neurons_agg.loc[\
        inter_sig_cxn_neurons_agg.region == 'gc']
gc_inter_sig_cxn_neurons_agg['nrn_id'] = \
        gc_inter_sig_cxn_neurons_agg['session'].astype('str') + '_' + \
        gc_inter_sig_cxn_neurons_agg['neuron'].astype('str')

# Get non-sig inter-region GC neurons
gc_inter_non_sig_cxn_neurons = unit_region_frame.merge(
        encoding_frame,
        how = 'left',
        left_on = ['session','neuron'],
        right_on = ['session','neuron'])
gc_inter_non_sig_cxn_neurons = gc_inter_non_sig_cxn_neurons[\
        gc_inter_non_sig_cxn_neurons['region'] == 'gc']
# Drop sig neurons
gc_inter_non_sig_cxn_neurons['nrn_id'] = \
        gc_inter_non_sig_cxn_neurons['session'].astype('str') + '_' + \
        gc_inter_non_sig_cxn_neurons['neuron'].astype('str')
gc_inter_non_sig_cxn_neurons = gc_inter_non_sig_cxn_neurons[\
        ~gc_inter_non_sig_cxn_neurons['nrn_id'].isin(
            gc_inter_sig_cxn_neurons_agg['nrn_id'])]

# # Standardize data columns
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
# scaler = RobustScaler
# for this_col in data_col_names:
#     this_data = gc_inter_sig_cxn_neurons_agg[this_col].values
#     this_data = this_data.reshape(-1,1)
#     this_scaler = scaler().fit(this_data)
#     this_data = this_scaler.transform(this_data)
#     gc_inter_sig_cxn_neurons_agg[this_col] = this_data


# Convert to long frame on 'fin_cxn_type'
gc_inter_sig_cxn_neurons_agg_long = gc_inter_sig_cxn_neurons_agg.melt(
        id_vars = ['fin_cxn_type'],
        value_vars = data_col_names,
        var_name = 'metric',
        value_name = 'value')

# g = sns.boxenplot(
#         data = gc_inter_sig_cxn_neurons_agg_long,
#         x = 'metric',
#         y = 'value',
#         hue = 'fin_cxn_type',
#         showfliers = True,
#         fill = True,
#         palette = ['red','green','orange'],
#         hue_order = ['gc_inter_receive',
#         'gc_inter_send gc_inter_receive',
#         'gc_inter_send'])
g = sns.catplot(
        kind = 'boxen',
        data = gc_inter_sig_cxn_neurons_agg_long,
        x = 'fin_cxn_type',
        col = 'metric',
        y = 'value',
        hue = 'fin_cxn_type',
        palette = ['red','orange','green'],
        order = [
            'gc_inter_receive',
            'gc_inter_send gc_inter_receive',
            'gc_inter_send'
            ],
        hue_order = [
            'gc_inter_receive',
            'gc_inter_send gc_inter_receive',
            'gc_inter_send'
            ],
        sharey = False,
        showfliers = False,
        )
# Rotate xticks
for ax in g.axes.flatten():
    plt.sca(ax)
    plt.xticks(rotation = 45, ha = 'right')
    this_title = ax.get_title()
    y_label = this_title.split('=')[1].strip()
    plt.ylabel(y_label)
    plt.title(None)
plt.suptitle('GC Encoding Metrics by Connection Type')
ymin, ymax = plt.ylim()
# plt.ylim([-1, 5])
g.fig.set_size_inches(6,5)
# sns.move_legend(g, "upper left", bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig(os.path.join(
    coupling_analysis_plot_dir, 'gc_encoding_metrics_by_cxn_type_bar.png'),
            bbox_inches = 'tight')
plt.close()


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

# Cut-off at firing < 0.02
# explained_var_scores = []
reg_pval_list = []
r2_score_list = []
fit_linear_model = []
for this_col in plot_cols:
    x = gc_inter_sig_agg['mean_post_stim_rates'].values
    y = gc_inter_sig_agg[this_col].values
    # cut_bool = x < 0.02
    # x = x[cut_bool]
    # y = y[cut_bool]
    x = x[:,None]
    y = y[:,None]
    # reg = HuberRegressor()
    reg = LinearRegression()
    reg.fit(x,y)
    fit_linear_model.append(reg)
    y_pred = reg.predict(x)
    # explained_var_scores.append(explained_variance_score(y, y_pred))
    r2_score_list.append(r2_score(y, y_pred))
    preg = linregress(x.ravel(), y.ravel())
    reg_pval_list.append(preg.pvalue)

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
    fig.suptitle(
            'r2_scores:\n'+\
                    str(dict(zip(plot_cols, np.round(r2_score_list,2)))) +\
            '\nregression p-values:\n'+\
                    str(dict(zip(plot_cols, np.round(reg_pval_list,2)))) 
                    )
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
    fig.suptitle(
            'r2_scores:\n'+\
                    str(dict(zip(plot_cols, np.round(r2_score_list,2)))) +\
            '\nregression p-values:\n'+\
                    str(dict(zip(plot_cols, np.round(reg_pval_list,2)))) 
                    )
    fig.tight_layout()
    ax[0].set_xlim([-0.005,0.02])
    ax[1].set_xlim([-0.005,0.02])
    ax[0].set_ylim([-0.005,20])
    fig.savefig(os.path.join(coupling_analysis_plot_dir,
                             add_str + 'gc_features_vs_firing_scatter_zoomed.png'),
                bbox_inches = 'tight')
    plt.close(fig)

##############################

do_outlier = True 

if do_outlier:
    outlier_bool_list = []
    fig, ax = plt.subplots(1,2)
    for i, this_col in enumerate(plot_cols):
        x = gc_inter_sig_agg[['mean_post_stim_rates', this_col]].values
        clf = IsolationForest(contamination=0.05).fit(x)
        pred = clf.predict(x)
        outlier_bool_list.append(pred)
        ax[i].scatter(x[:,0], x[:,1], c = pred)	
        ax[i].set_xlabel('mean_post_stim_rates')
        ax[i].set_ylabel(this_col)
        ax[i].set_title(f'GC {this_col} vs firing rate')
    fig.suptitle('Isolation Forest outlier detection')
    fig.tight_layout()
    fig.savefig(os.path.join(coupling_analysis_plot_dir,
                             'gc_features_vs_firing_scatter_outliers.png'),
                bbox_inches = 'tight')
    plt.close(fig)

    keep_bool = np.logical_and(outlier_bool_list[0]>0, outlier_bool_list[1]>0)
else:
    keep_bool = np.ones(len(outlier_bool_list[0])) > 0

gc_inter_sig_keep = gc_inter_sig_agg.iloc[keep_bool]

# Find max value for send population to use
# as cut threshold
send_frame = gc_inter_sig_keep.loc[gc_inter_sig_keep.fin_cxn_type == 'gc_inter_send']
max_send_rate = send_frame['mean_post_stim_rates'].max()

do_cut = False

# Determine significance of regression for each
reg_pval_dict = {}
for this_col in plot_cols:
    col_dict = {}
    for this_group in list(cxn_type_names.keys()):
        this_frame = gc_inter_sig_keep.loc[gc_inter_sig_keep.cxn_cat == this_group]
        x = this_frame['mean_post_stim_rates'].values
        y = this_frame[this_col].values
        if do_cut:
            cut_bool = x < max_send_rate
            x = x[cut_bool]
            y = y[cut_bool]
        reg = linregress(x,y)
        col_dict[this_group] = np.round(reg.pvalue,2)
    reg_pval_dict[this_col] = col_dict

# Regress each population against discrim and pal separately
fit_linear_model = {}
for this_col in plot_cols:
    col_dict = {}
    for this_group in list(cxn_type_names.keys()):
        this_frame = gc_inter_sig_keep.loc[gc_inter_sig_keep.cxn_cat == this_group]
        x = this_frame['mean_post_stim_rates'].values
        y = this_frame[this_col].values
        if do_cut:
            cut_bool = x < max_send_rate
            x = x[cut_bool]
            y = y[cut_bool]
        x = x[:,None]
        y = y[:,None]
        reg = LinearRegression()
        reg.fit(x,y)
        col_dict[this_group] = reg
    fit_linear_model[this_col] = col_dict

for do_log in [True, False]:
    if do_log:
        add_str = 'log_'
    else:
        add_str = ''
    if do_outlier:
        add_str += 'outlier_removed_'
    # Make plots with regression lines
    fig, ax = plt.subplots(1,2, figsize = (7,7))
    for i, this_col in enumerate(plot_cols): 
        for j in np.unique(gc_inter_sig_keep['cxn_cat'].values):
            this_frame = gc_inter_sig_keep.loc[gc_inter_sig_keep.cxn_cat == j]
            ax[i].scatter(
                this_frame['mean_post_stim_rates'],
                this_frame[plot_cols[i]],
                c = cmap(this_frame['cxn_cat'].values),
                label = cxn_type_names[j],
                alpha = 1)
            # Plot regression line
            this_reg = fit_linear_model[this_col][j]
            x_range = [np.min(gc_inter_sig_keep['mean_post_stim_rates'].values),
                       np.max(gc_inter_sig_keep['mean_post_stim_rates'].values)]
            x_vec = np.linspace(x_range[0], x_range[1], 100)
            y_vec = this_reg.predict(x_vec[:,None])
            ax[i].plot(x_vec, y_vec, linewidth = 2,
                       alpha = 1, linestyle = '--',
                       color = cmap(j))
        if do_log:
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
        if do_cut:
            ax[i].axvline(max_send_rate, color = 'k', alpha = 0.3, label = 'max_send_rate')
        ax[i].set_xlabel('mean_post_stim_rates')
        ax[i].set_ylabel(plot_cols[i])
        this_col_pvals = "\n".join([f'{cxn_type_names[k]}: {v}' for k,v in reg_pval_dict[this_col].items()])
        ax[i].set_title(f'GC {plot_cols[i]} vs firing rate' + '\n' +\
                'reg_pvals: \n' + f'{this_col_pvals}')
    # Legend at bottom
    ax[i].legend(loc='upper center',
                 bbox_to_anchor=(0.5, -0.2))
    fig.suptitle('r2_scores:\n'+str(dict(zip(plot_cols, np.round(r2_score_list,2)))))
    fig.tight_layout()
    fig.savefig(os.path.join(coupling_analysis_plot_dir,
                             add_str + 'gc_features_vs_firing_regression.png'),
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

##############################
# Do a direct classification (1 vs 1) between
# all pairs of classes
# Use Logistic regression

X_raw = gc_inter_sig_agg[data_col_names]
X = StandardScaler().fit_transform(X_raw)
group_labels = gc_inter_sig_agg.cxn_cat.values

clf_score_list = []
for this_pair in cxn_name_combos:
    dat1 = X[group_labels == this_pair[0]]
    dat2 = X[group_labels == this_pair[1]]
    input_X = np.vstack((dat1, dat2))
    input_y = np.hstack((np.zeros(len(dat1)), np.ones(len(dat2))))
    clf = LogisticRegression()
    clf.fit(input_X, input_y)
    clf_score = clf.score(input_X, input_y)
    clf_score_list.append(clf_score)

##############################
# Fit gaussians to each group and check
# correlation in prob of each datapoint

from sklearn.mixture import GaussianMixture

gmm_list = []
for this_label in np.unique(group_labels):
    this_dat = X[group_labels == this_label] 
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(this_dat)
    gmm_list.append(gmm)

# Get the probability of each datapoint
# from each gmm
gmm_prob_list = []
for this_gmm in gmm_list:
    gmm_prob_list.append(this_gmm.score_samples(X))

gmm_prob_array = np.squeeze(np.stack(gmm_prob_list))

# Distances between probs
prob_euc_dists = np.zeros(
        (len(gmm_list), len(gmm_list))
        )
for i in range(len(gmm_list)):
    for j in range(len(gmm_list)):
        prob_euc_dists[i,j] = euclidean(
                gmm_prob_array[i], gmm_prob_array[j]
                )
prob_euc_dists = euclidean(gmm_prob_array, gmm_prob_array)


# Plot prob_euc_dists
fig, ax = plt.subplots(figsize = (5,5))
im = ax.matshow(prob_euc_dists, cmap = 'binary')
ax.set_xticks(range(len(np.unique(group_labels))))
ax.set_yticks(range(len(np.unique(group_labels))))
ax.set_xticklabels([cxn_type_names[i] for i in np.unique(group_labels)],
                   rotation = 90)
ax.set_yticklabels([cxn_type_names[i] for i in np.unique(group_labels)])
ax.set_title('Euclidean distance between GMM probabilities')
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_gmm_prob_euc_dists.png'),
            bbox_inches = 'tight')
plt.close(fig)

# For each datapoint, check difference in likelihood
prob_diffs = np.zeros(
        (len(gmm_list), len(gmm_list))
        )
for i in range(len(gmm_list)):
    for j in range(len(gmm_list)):
        prob_diffs[i,j] = np.mean( 
                gmm_prob_array[i] - gmm_prob_array[j]
                                  )

# Plot prob_euc_dists
# Use symmetric binary colormap centrered at 0
cmap = plt.cm.get_cmap('RdBu_r')
norm = plt.Normalize(vmin=prob_diffs.min(), vmax=prob_diffs.max())

fig, ax = plt.subplots(figsize = (5,5))
im = ax.matshow(prob_diffs, cmap = cmap, norm = norm)
ax.set_xticks(range(len(np.unique(group_labels))))
ax.set_yticks(range(len(np.unique(group_labels))))
ax.set_xticklabels([cxn_type_names[i] for i in np.unique(group_labels)],
                   rotation = 90)
ax.set_yticklabels([cxn_type_names[i] for i in np.unique(group_labels)])
ax.set_title('Average difference in log likelihoods\nacross all datapoints')
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_gmm_prob_ll_diffs.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################
corr_mat = np.corrcoef(gmm_prob_array)

# Plot prob array with group labels
sort_inds = np.argsort(group_labels)
fig, ax = plt.subplots(figsize = (5,10))
im = ax.imshow(np.log10(np.abs(gmm_prob_array.T[sort_inds])), 
               aspect='auto', interpolation = 'none')
ax.set_xticks(range(len(np.unique(group_labels))))
ax.set_xticklabels([cxn_type_names[i] for i in np.unique(group_labels)], 
                   rotation = 90)
ax.set_title('Gaussian Likelihoods')
ax.set_xlabel('Group')
ax.set_ylabel('Datapoint')
fig.colorbar(im)
fig.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_gmm_likelihoods.png'),
            bbox_inches = 'tight')
plt.close(fig)

# Plot correlation matrix
fig, ax = plt.subplots(figsize = (5,5))
im = ax.matshow(corr_mat, cmap = 'binary')
ax.set_xticks(range(len(np.unique(group_labels))))
ax.set_yticks(range(len(np.unique(group_labels))))
ax.set_xticklabels([cxn_type_names[i] for i in np.unique(group_labels)], 
                   rotation = 90)
ax.set_yticklabels([cxn_type_names[i] for i in np.unique(group_labels)])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Gaussian Likelihood Correlation matrix')
# Annotate with values
for i in range(len(np.unique(group_labels))):
    for j in range(len(np.unique(group_labels))):
        ax.text(j, i, str(np.round(corr_mat[i,j],2)), 
                 ha="center", va="center", color="r",
                 fontweight = 'bold')
fig.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_gmm_corr.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################
# Calculate average mahalanobis distance of all components from 
# all other arrays


# Get the mean and covariance of each group
gmm_mean_list = []
gmm_cov_list = []
for this_gmm in gmm_list:
    gmm_mean_list.append(this_gmm.means_[0])
    gmm_cov_list.append(this_gmm.covariances_[0])

# Calculate inverse of each covairance matrix
# (needed for mahalanobis calculation)
gmm_cov_inv_list = []
for this_cov in gmm_cov_list:
    gmm_cov_inv_list.append(np.linalg.inv(this_cov))

# Calculate mahalanobis distance of each point
# from the mean of each group
mahal_dist_list = []
for i in range(len(gmm_mean_list)):
    this_mean = gmm_mean_list[i]
    this_cov_inv = gmm_cov_inv_list[i]
    mahal_list = [mahalanobis(x, this_mean, this_cov_inv) \
            for x in X]
    mahal_dist_list.append(mahal_list)

mahal_array = np.stack(mahal_dist_list)

mean_mahal_dists = np.zeros((len(gmm_mean_list), len(gmm_mean_list)))
for this_clust in range(len(gmm_mean_list)):
    for other_clust in range(len(gmm_mean_list)):
        wanted_rows = np.where(group_labels == this_clust)[0] 
        wanted_dists = mahal_array[this_clust, wanted_rows]
        mean_mahal_dists[this_clust, other_clust] = np.mean(wanted_dists)

# Plot mahalanobis distance matrix
fig, ax = plt.subplots(figsize = (5,5))
im = ax.matshow(mean_mahal_dists, cmap = 'binary')
ax.set_xticks(range(len(np.unique(group_labels))))
ax.set_yticks(range(len(np.unique(group_labels))))
ax.set_xticklabels([cxn_type_names[i] for i in np.unique(group_labels)],
                   rotation = 90)
ax.set_yticklabels([cxn_type_names[i] for i in np.unique(group_labels)])
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Mahalanobis Distance matrix')
# Annotate with values
for i in range(len(np.unique(group_labels))):
    for j in range(len(np.unique(group_labels))):
        ax.text(j, i, str(np.round(mean_mahal_dists[i,j],2)), 
                 ha="center", va="center", color="r",
                 fontweight = 'bold')
fig.tight_layout()
fig.savefig(os.path.join(coupling_analysis_plot_dir,
                         'gc_gmm_mean_mahal_dists.png'),
            bbox_inches = 'tight')
plt.close(fig)
