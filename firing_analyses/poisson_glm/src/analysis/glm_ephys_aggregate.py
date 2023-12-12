"""
Questions:
    1) Are significant fits (actual LL > shuffled LL) related to firing rate?
    2) Intra-region vs Inter-region connectivity?
        2.1) Can we pull out neurons with one vs the other?
"""

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
# Preprocessing
############################################################
# Pull out actual fit-type from fin_pval_frame
fin_pval_frame = fin_pval_frame.loc[fin_pval_frame['fit_type'] == 'actual']

# Mark rows where loglikelihood for actual fits > shuffle
fin_ll_frame['actual>shuffle'] = \
        fin_ll_frame['actual'] > fin_ll_frame['trial_shuffled']

ind_names = ['session','taste', 'neuron']
ll_sig_inds = fin_ll_frame.loc[fin_ll_frame['actual>shuffle'], ind_names]

# To avoid crazy numbers in ll, pull out a specific range of the distribution
perc_range = 0.95
perc_lims = np.array([(1-0.95)/2, 1-(1-0.95)/2])*100
# Adjust x and y lims to include percentile range indicated
actual_lims = np.percentile(fin_ll_frame['actual'], perc_lims)*1.1
shuffle_lims = np.percentile(fin_ll_frame['trial_shuffled'], perc_lims)*1.1

# Take the higher of the lower limits
low_lim = np.max([x[0] for x in [actual_lims, shuffle_lims]])

inds = np.logical_and(
        fin_ll_frame['actual'] > low_lim, 
        fin_ll_frame['trial_shuffled'] > low_lim, 
        )
pretty_ll_data = fin_ll_frame.loc[inds]

############################################################
############################################################
# Show that paired actual ll > trial shuffled ll
# Generate scatter of actual vs trial shuffled
# Also plot histogram along diagonal

fig, ax = plt.subplots(figsize = (5,5))
ax.scatter(
        -fin_ll_frame['actual'],
        -fin_ll_frame['trial_shuffled'],
        s = 2, 
        c = 'k',
        alpha = 0.3,
        )
x_lims = actual_lims.copy()
y_lims = shuffle_lims.copy()
x_lims[1] = 0
y_lims[1] = 0
ax.set_xlim(-np.array(y_lims)[::-1])
ax.set_ylim(-np.array(y_lims)[::-1])
ax.plot(-y_lims, -y_lims, color = 'red', linestyle = '--', linewidth = 2)
ax.set_ylabel('-LL : Actual')
ax.set_xlabel('-LL : Trial Shuffled')
ax.set_aspect('equal')
plt.suptitle('Actual vs Trial Shuffled Log Likelihood')
plt.title(f'N = {len(fin_ll_frame)}')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(plot_dir, 'actual_vs_trial_shuffled_ll.svg'),
            bbox_inches='tight')
plt.close()


# Histogram along diagonal
# Convert data to vectors
vec_data = np.array([
    pretty_ll_data['actual'],
    pretty_ll_data['trial_shuffled']
    ]).T
proj_vec = np.array([-1,1])
proj_data = np.dot(vec_data, proj_vec)

# Check that median is significantly different from 0
wil_stat, wil_p = wilcoxon(proj_data)

fig, ax = plt.subplots(figsize = (5,3))
ax.hist(proj_data, bins = 50, density = True, alpha = 0.5, color = 'k');
ax.hist(proj_data, bins = 50, histtype = 'step', color = 'k', density = True);
ax.axvline(0, color = 'red', linestyle = '--', linewidth = 2)
ax.set_xlabel('Actual - Trial Shuffled Likelihood')
# ax.set_ylabel('Count')
plt.suptitle('Actual vs Trial Shuffled Log Likelihood\n' +\
        f'Wilcoxon p = {wil_p:.3f}')
# Mark median with arrow
med_val = np.median(proj_data)
ax.annotate(
        '',
        xy = (med_val, ax.get_ylim()[0]),
        xytext = (med_val, 0.1*ax.get_ylim()[1]),
        arrowprops = dict(
            facecolor = 'white', 
            edgecolor = 'black',
            shrink = 0.05
            ),
        label = 'Median',
        )
#ax.legend()
# Remove all axes except bottom
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'actual_vs_trial_shuffled_ll_hist.svg'),
            bbox_inches='tight')
plt.close()
#plt.show()


# Pie char showing actual > shuffle, and shuffle > actual

fig,ax = plt.subplots(figsize=(3,3))
ax.pie(
        fin_ll_frame['actual>shuffle'].value_counts().values,
        labels = ['Actual > Shuffle', 'Shuffle > Actual'],
        autopct = '%1.1f%%',
        )
plt.title('Actual vs Trial Shuffled Log Likelihood')
plt.savefig(os.path.join(plot_dir, 'actual_vs_trial_shuffled_ll_pie.png'),
            bbox_inches='tight')
plt.close()
#plt.show()

# Mean firing rate per category
# Is firing rate related to significance?
cat_vals = [x[1]['mean_rate'] \
        for x in list(fin_ll_frame.groupby(['actual>shuffle']))]

test_stat, p_val = mwu(*cat_vals)

fig,ax = plt.subplots(figsize=(3,3))
sns.boxplot(
        data = fin_ll_frame,
        x = 'actual>shuffle',
        y = 'mean_rate',
        ax=ax
        )
# sns.ecdfplot(
#         data = fin_ll_frame,
#         hue = 'actual>shuffle',
#         x = 'mean_rate',
#         )
plt.title('Mean Firing Rate per Category\n' + \
        f'MWU pval: {p_val:.3f}')
plt.savefig(os.path.join(plot_dir, 'mean_firing_rate_per_category.png'),
            bbox_inches='tight')
plt.close()
#plt.show()

############################################################
############################################################
# Relationship between firing rate and loglikelihood per region

pretty_ll_data['ll_diff'] = pretty_ll_data['actual'] - pretty_ll_data['trial_shuffled']

fig, ax = plt.subplots(2,1, sharex=True, figsize = (5,10))
sns.histplot(data = pretty_ll_data, x = 'mean_rate', y = 'll_diff', 
              hue = 'region', bins = 40, ax = ax[0])
# plt.axhline(y = thresh, color = 'r', linestyle = '--', label = '0.05')
ax[0].set_title('Mean Firing Rate vs. Significance')
ax[0].set_xlabel('Mean Firing Rate')
ax[0].set_ylabel('Log10 Likelihood Difference\n <--Shuffled better | Actual better-->')
sns.histplot(data = pretty_ll_data, 
              x = 'mean_rate', y = 'actual',
             hue = 'region', 
             bins = 40, ax = ax[1])
ax[1].set_title('Mean Firing Rate vs. Log Likelihood')
ax[1].set_xlabel('Mean Firing Rate')
ax[1].set_ylabel('Actual Log Likelihood')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir,'mean_rate_vs_log_ll.png'), dpi = 300, bbox_inches = 'tight')
plt.close()
#plt.show()

############################################################
############################################################
# Generate PSTHs for all tastes
psth_plot_dir = os.path.join(plot_dir, 'example_psths')
if not os.path.exists(psth_plot_dir):
    os.makedirs(psth_plot_dir)

n_plots = 10
for idx, dat_inds in \
        tqdm(zip(session_taste_inds[:n_plots], all_taste_inds[:n_plots])):
    this_spike_dat = np.stack([design_spikes_list[i] for i in dat_inds])
    this_pred_dat = np.stack([pred_spikes_list[i] for i in dat_inds])

    actual_psth = np.mean(this_spike_dat, axis = 1)
    pred_psth = np.mean(this_pred_dat, axis = 1)

    # Smoothen PSTH
    kern_len = 20
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

########################################
# Do filters from BLA-->GC and GC-->BLA have different shapes
coupling_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('coup')]
coupling_frame = coupling_frame[['fit_num','param','p_val','values', *ind_names]]

coupling_frame['lag'] = [int(x.split('_')[-1]) for x in coupling_frame.param]
coupling_frame['other_nrn'] = [int(x.split('_')[-2]) for x in coupling_frame.param]

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
# this_ax.set_ylim([0.005,1])
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
