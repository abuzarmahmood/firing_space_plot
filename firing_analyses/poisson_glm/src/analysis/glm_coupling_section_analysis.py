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
        spearmanr
        )
from glob import glob
import json

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

nrn_discrim = [taste_oneway_anova(x) for x in tqdm(post_stim_nrn_rates)]
nrn_discrim_stat, nrn_discrim_pval = list(zip(*nrn_discrim))
nrn_discrim_stat = np.stack(nrn_discrim_stat)
nrn_discrim_pval = np.stack(nrn_discrim_pval)

nrn_discrim_max_stat = [np.max(x) for x in nrn_discrim_stat]

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

nrn_pal_corr = [taste_pal_corr(x, pal_rankings[0]) \
        for x in tqdm(post_stim_nrn_rates)]
nrn_pal_rho, nrn_pal_pval = list(zip(*nrn_pal_corr))
nrn_pal_rho = np.stack(nrn_pal_rho)
nrn_pal_pval = np.stack(nrn_pal_pval)

nrn_pal_max_rho = [np.max(x) for x in nrn_pal_rho]

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
pal_iden_frame['max_discrim_stat'] = nrn_discrim_max_stat
pal_iden_frame['max_pal_rho'] = nrn_pal_max_rho

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
cxn_type_names = [x[0] for x in cxn_groups]
cxn_type_frames = [x[1] for x in cxn_groups]

# For each cxn_type plot histograms of:
# 1- firing rates
# 2- responsiveness
# 3- max discrim stat
# 4- max pal rho

data_col_names = ['mean_post_stim_rates','responsiveness',
                  'max_discrim_stat','max_pal_rho']

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
                     log = True, density = True)
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
                         log = True, density = True)
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
# Group by session, neuron, and region
sig_cxn_neurons_group = list(sig_cxn_neurons.groupby(
    ['session','neuron','region']))
sig_cxn_neuron_inds = [x[0] for x in sig_cxn_neurons_group]
sig_cxn_neuron_frames = [x[1] for x in sig_cxn_neurons_group]

unique_cxn_types = [x['fin_cxn_type'].unique() for x in sig_cxn_neuron_frames]
