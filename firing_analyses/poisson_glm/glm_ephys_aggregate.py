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
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
import glm_tools as gt
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
from sklearn.cluster import KMeans
import seaborn as sns

save_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
plot_dir=  '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/plots'

############################################################
# Get Spikes

file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in file_list]

spike_list = []
unit_region_list = []
for ind in trange(len(file_list)):
    dat = ephys_data(file_list[ind])
    dat.get_spikes()
    spike_list.append(np.array(dat.spikes))
    # Calc mean firing rate
    mean_fr = np.array(dat.spikes).mean(axis=(0,1,3))
    dat.get_region_units()
    region_units = dat.region_units
    region_names = dat.region_names
    region_names = [[x]*len(y) for x,y in zip(region_names,region_units)]
    region_units = np.concatenate(region_units)
    region_names = np.concatenate(region_names)
    unit_region_frame = df(
            {'region':region_names,
             'unit':region_units,
             }
            )
    unit_region_frame['basename'] = basenames[ind]
    unit_region_frame['session'] = ind
    unit_region_frame = unit_region_frame.sort_values(by=['unit'])
    unit_region_frame['mean_rate'] = mean_fr
    unit_region_list.append(unit_region_frame)
############################################################

############################################################
# Load Data

unit_region_frame = pd.read_csv(os.path.join(save_path,'unit_region_frame.csv'), index_col = 0)
unit_region_frame.rename(columns = {'unit' : 'neuron'}, inplace=True)
ind_frame = pd.read_csv(os.path.join(save_path,'ind_frame.csv'), index_col = 0)

p_val_frame_paths = sorted(glob(os.path.join(save_path,'*p_val_frame.csv')))
ll_frame_paths = sorted(glob(os.path.join(save_path,'*ll_frame.csv')))
p_val_basenames = [os.path.basename(x) for x in p_val_frame_paths]
ll_basenames = [os.path.basename(x) for x in ll_frame_paths]
p_val_inds_str = [x.split('_p_val')[0] for x in p_val_basenames]
ll_inds_str = [x.split('_ll')[0] for x in ll_basenames]
p_val_inds = np.array([list(map(int,x.split('_'))) for x in p_val_inds_str])
ll_inds = np.array([list(map(int,x.split('_'))) for x in ll_inds_str])
# Make sure inds match up
assert np.all(p_val_inds == ll_inds), "Mismatched inds"
p_val_frame_list = [pd.read_csv(x, index_col=0) for x in p_val_frame_paths]
ll_frame_list = [pd.read_csv(x, index_col = 0) for x in ll_frame_paths]

# Add inds to frames
# Order : [sessio, taste, neurons]
p_val_frame_list_fin = []
ll_frame_list_fin = []
for i in trange(len(p_val_frame_list)):
    this_ind = p_val_inds[i]
    this_pval_frame = p_val_frame_list[i]
    this_ll_frame = ll_frame_list[i]
    this_pval_frame['session'] = this_ind[0]
    this_pval_frame['taste'] = this_ind[1]
    this_pval_frame['neuron'] = this_ind[2]
    this_ll_frame['session'] = this_ind[0]
    this_ll_frame['taste'] = this_ind[1]
    this_ll_frame['neuron'] = this_ind[2]
    p_val_frame_list_fin.append(this_pval_frame)
    ll_frame_list_fin.append(this_ll_frame)

fin_pval_frame = pd.concat(p_val_frame_list_fin)
fin_pval_frame = fin_pval_frame.sort_values(by=['session','taste','neuron'])
# Reset index
fin_pval_frame = fin_pval_frame.reset_index(drop=True)

fin_ll_frame = pd.concat(ll_frame_list_fin)
# Sort by inds
fin_ll_frame = fin_ll_frame.sort_values(by=['session','taste','neuron'])
# Merge fin_ll_frame and unit_region_frame
fin_ll_frame = pd.merge(fin_ll_frame, unit_region_frame, on = ['session','neuron'])


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

thresh = np.round(np.log10(0.005),2)
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
ind_names = ['session','taste', 'neuron']
mean_nrn_ll_frame = fin_ll_frame.groupby(ind_names).median().reset_index(drop=False)
mean_nrn_ll_frame = mean_nrn_ll_frame.sort_values(by = ['mean_rate','actual'], ascending = False)
# Take out rows which don't mathc with ll_pval_frame
mean_nrn_ll_frame = mean_nrn_ll_frame.merge(ll_pval_frame[ind_names], on = ind_names)
mean_nrn_ll_frame['top'] = False
mean_nrn_ll_frame.loc[mean_nrn_ll_frame.index[:n_top],'top'] = True

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

# Recalculate PSTHs for top inds
############################################################
# Parameters
hist_filter_len = 200
stim_filter_len = 500
coupling_filter_len = 200

trial_start_offset = -2000
trial_lims = np.array([1000,4000])
stim_t = 2000

bin_width = 10

# Reprocess filter lens
hist_filter_len_bin = hist_filter_len // bin_width
stim_filter_len_bin = stim_filter_len // bin_width
coupling_filter_len_bin = coupling_filter_len // bin_width

# Define basis kwargs
basis_kwargs = dict(
    n_basis = 10,
    basis = 'cos',
    basis_spread = 'log',
    )

# Number of fits on actual data (expensive)
n_fits = 1
n_max_tries = 20
############################################################
stim_vec = np.zeros(spike_list[0].shape[-1])
stim_vec[stim_t] = 1

for num, this_ind in tqdm(enumerate(top_inds)):
    #this_ind = fin_inds[0]
    this_ind_str = '_'.join([str(x) for x in this_ind])
    this_session_dat = spike_list[this_ind[0]]
    this_taste_dat = this_session_dat[this_ind[1]]
    this_nrn_ind = this_ind[2]
    other_nrn_inds = np.delete(np.arange(this_session_dat.shape[2]),
            this_nrn_ind)
    n_coupled_neurons = len(other_nrn_inds)

    this_nrn_dat = this_taste_dat[:,this_nrn_ind]
    other_nrn_dat = this_taste_dat[:,other_nrn_inds]
    stim_dat = np.tile(stim_vec,(this_taste_dat.shape[0],1))

    this_nrn_flat = np.concatenate(this_nrn_dat)
    other_nrn_flat = np.concatenate(np.moveaxis(other_nrn_dat, 1, -1)).T
    stim_flat = np.concatenate(stim_dat)

    # To convert to dataframe, make sure trials are not directly
    # concatenated as that would imply temporal continuity
    data_frame = gt.gen_data_frame(
            this_nrn_flat,
            other_nrn_flat,
            stim_flat,
            stim_filter_len = stim_filter_len,
            trial_start_offset = trial_start_offset,
            )

    # Bin data
    data_frame['time_bins'] = data_frame.trial_time // bin_width
    data_frame = data_frame.groupby(['trial_labels','time_bins']).sum()
    data_frame['trial_time'] = data_frame['trial_time'] // bin_width
    data_frame = data_frame.reset_index()

    # Create design mat
    actual_design_mat = gt.dataframe_to_design_mat(
            data_frame,
            hist_filter_len = hist_filter_len_bin,
            stim_filter_len = stim_filter_len_bin,
            coupling_filter_len = coupling_filter_len_bin,
            basis_kwargs = basis_kwargs,
            )


    # Cut to trial_lims
    # Note, this needs to be done after design matrix is created
    # so that overlap of history between trials is avoided
    trial_lims_vec = np.arange(*trial_lims)
    actual_design_mat = actual_design_mat.loc[actual_design_mat.trial_time.isin(trial_lims_vec)]

    #plt.imshow(actual_design_mat.iloc[:,:-2].values, aspect = 'auto')
    #plt.show()

    fit_list = []
    for i in trange(n_max_tries):
        if len(fit_list) < n_fits:
            try:
                res, _ = gt.gen_actual_fit(
                        data_frame, # Not used if design_mat is provided
                        hist_filter_len = hist_filter_len_bin,
                        stim_filter_len = stim_filter_len_bin,
                        coupling_filter_len = coupling_filter_len_bin,
                        basis_kwargs = basis_kwargs,
                        actual_design_mat = actual_design_mat,
                        )
                fit_list.append(res)
            except:
                print('Failed fit')
        else:
            print('Finished fitting')
            break

    ll_names = ['actual','trial_sh','circ_sh','rand_sh']
    ll_outs = [gt.calc_loglikelihood(actual_design_mat, res) for res in tqdm(fit_list)]
    ll_frame = pd.DataFrame(ll_outs, columns=ll_names)

    # Grab best fit
    best_fit_ind = ll_frame.actual.idxmax()
    best_fit = fit_list[best_fit_ind]

    # PSTH
    time_bins = actual_design_mat.trial_time.unique()
    design_trials = list(actual_design_mat.groupby('trial_labels'))
    design_spikes = [x.sort_values('trial_time').spikes.values for _,x in design_trials]
    design_spikes_array = np.stack(design_spikes)

    # Predicted PSTH
    pred_spikes = pd.DataFrame(best_fit.predict(actual_design_mat), columns = ['spikes'])
    pred_spikes['trial_labels'] = actual_design_mat.trial_labels
    pred_spikes['trial_time'] = actual_design_mat.trial_time
    pred_trials = list(pred_spikes.groupby('trial_labels'))
    pred_spikes = [x.sort_values('trial_time').spikes.values for _,x in pred_trials]
    pred_spikes_array = np.stack(pred_spikes)


    # Smoothen
    kern_len = 20
    kern = np.ones(kern_len) / kern_len

    smooth_design_spikes = np.apply_along_axis(
            lambda x: np.convolve(x, kern, mode = 'same'),
            1,
            design_spikes_array,
            )

    smooth_pred_spikes = np.apply_along_axis(
            lambda x: np.convolve(x, kern, mode = 'same'),
            1,
            pred_spikes_array,
            )

    fig, ax = plt.subplots(1,2, sharey = True, sharex = True, figsize = (7,3))
    ax[0].plot(time_bins, smooth_design_spikes.T, color = 'k', alpha = 0.3)
    ax[0].plot(time_bins, smooth_design_spikes.mean(0), color = 'r', linewidth = 2)
    ax[1].plot(time_bins, smooth_pred_spikes.T, color = 'k', alpha = 0.3)
    ax[1].plot(time_bins, smooth_pred_spikes.mean(0), color = 'r', linewidth = 2)
    ax[0].set_title('Actual')
    ax[1].set_title('Predicted')
    ax[0].set_ylabel('Firing rate (Hz)')
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    # Set ylim to be max of real data
    ax[0].set_ylim([0,smooth_design_spikes.max()])
    ax[1].set_ylim([0,smooth_design_spikes.max()])
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)
    plt.suptitle(f'Session {this_ind[0]}, Neuron {this_ind[2]}, Taste {this_ind[1]}')
    plt.savefig(os.path.join(plot_dir, 'example_nrns',
                             'psth_comp_'+ f'{num}_' + "_".join([str(x) for x in this_ind]) + '.png'),
                dpi = 300, bbox_inches = 'tight')
    plt.close()
    #plt.show()

############################################################
# Significant coupling filters
############################################################
# Throw out all rows which don't have significant differences in likelihood
# between actual and shuffled
fin_pval_frame = fin_pval_frame.merge(ll_pval_frame, on = ind_names)

# Only take fit_num with highest likelihood
max_ll_frame = fin_ll_frame[['fit_num','actual',*ind_names]]
max_inds = max_ll_frame.groupby(ind_names).actual.idxmax().reset_index().actual
max_vals = max_ll_frame.loc[max_inds].drop(columns = 'actual') 

fin_pval_frame = fin_pval_frame.merge(max_vals, on = ['fit_num',*ind_names])

# Extract history filters
hist_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('hist')]
hist_frame.drop(columns = ['trial_sh','circ_sh','rand_sh'], inplace = True)
hist_frame['ind_index'] = hist_frame[ind_names].astype(str).agg('_'.join, axis=1)

p_val_array = pd.pivot_table(
        hist_frame, index = 'ind_index', columns = 'param', values = 'p_val').values

# Kmeans clustering
kmeans = KMeans(n_clusters = 10, random_state = 0).fit(p_val_array)
p_val_array = p_val_array[kmeans.labels_.argsort()]

alpha = 0.05
plt.imshow(p_val_array < alpha, aspect = 'auto', interpolation = 'none')
plt.show()

hist_grouped = list(hist_frame.groupby(ind_names))


# Extract coupling filters
coupling_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('coup')]
coupling_frame.drop(columns = ['trial_sh','circ_sh','rand_sh'], inplace = True)

coupling_frame['lag'] = [int(x.split('_')[-1]) for x in coupling_frame.param]
coupling_frame['other_nrn'] = [int(x.split('_')[-1]) for x in coupling_frame.param]
