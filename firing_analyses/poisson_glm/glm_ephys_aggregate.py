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
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import pingouin as pg
import json

run_str = 'run_002'
save_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
plot_dir=  f'/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/plots/{run_str}'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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

p_val_frame_paths = sorted(glob(os.path.join(save_path, run_str, '*p_val_frame.csv')))
ll_frame_paths = sorted(glob(os.path.join(save_path, run_str, '*ll_frame.csv')))

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

thresh = np.round(np.log10(0.1),2)
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
n_shuffles_per_fit = 5

#hist_filter_len = 200
#stim_filter_len = 500
#coupling_filter_len = 200
#
#trial_start_offset = -2000
#trial_lims = np.array([1000,4000])
#stim_t = 2000
#
#bin_width = 10
#
## Reprocess filter lens
#hist_filter_len_bin = hist_filter_len // bin_width
#stim_filter_len_bin = stim_filter_len // bin_width
#coupling_filter_len_bin = coupling_filter_len // bin_width
#
## Define basis kwargs
#basis_kwargs = dict(
#    n_basis = 10,
#    basis = 'cos',
#    basis_spread = 'log',
#    )
#
## Number of fits on actual data (expensive)
#n_fits = 1
#n_max_tries = 20
############################################################
make_example_plots = True

stim_vec = np.zeros(spike_list[0].shape[-1])
stim_vec[stim_t] = 1

example_nrns_path = os.path.join(plot_dir, 'example_nrns')
if not os.path.exists(example_nrns_path):
    os.makedirs(example_nrns_path)

if make_example_plots:
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

        if len(fit_list) > 1:
            ll_names = ['actual','trial_sh','circ_sh','rand_sh']
            ll_outs = [[gt.calc_loglikelihood(actual_design_mat, res)\
                    for i in range(n_shuffles_per_fit)]\
                    for res in tqdm(fit_list)]
            ll_outs = np.array(ll_outs).reshape(-1,4)
            ll_frame = pd.DataFrame(ll_outs, columns=ll_names)
            ll_frame['fit_num'] = np.repeat(np.arange(len(fit_list)), n_shuffles_per_fit)

            # Grab best fit
            best_fit_ind = ll_frame.actual.idxmax()
        else:
            best_fit = fit_list[0]

        # PSTH
        time_bins = actual_design_mat.trial_time.unique()
        design_trials = list(actual_design_mat.groupby('trial_labels'))
        design_spikes = [x.sort_values('trial_time').spikes.values for _,x in design_trials]
        design_spikes_array = np.stack(design_spikes)

        # Predicted PSTH
        pred_spikes = pd.DataFrame(best_fit.predict(actual_design_mat[best_fit.params.index]), 
                                   columns = ['spikes'])
        pred_spikes.loc[pred_spikes.spikes > bin_width, 'spikes'] = bin_width
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

        fig, ax = plt.subplots(1,2, sharey = False, sharex = True, figsize = (7,3))
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
        plt.savefig(os.path.join(example_nrns_path,
                                 'psth_comp_'+ f'{num}_' + "_".join([str(x) for x in this_ind]) + '.png'),
                    dpi = 300, bbox_inches = 'tight')
        plt.close()
        #plt.show()

############################################################
# Generate PSTHs for all tastes
psth_plot_dir = os.path.join(plot_dir, 'example_psths')
if not os.path.exists(psth_plot_dir):
    os.makedirs(psth_plot_dir)

stim_vec = np.zeros(spike_list[0].shape[-1])
stim_vec[stim_t] = 1

if make_example_plots:
    for num, this_ind in tqdm(enumerate(taste_top_inds)):
        #this_ind = fin_inds[0]
        this_ind_str = '_'.join([str(x) for x in this_ind])
        this_session_dat = spike_list[this_ind[0]]
        this_nrn_ind = this_ind[1]
        other_nrn_inds = np.delete(np.arange(this_session_dat.shape[2]),
                this_nrn_ind)
        n_coupled_neurons = len(other_nrn_inds)

        design_spikes_list = []
        pred_spikes_list = []
        for this_taste_ind, this_taste_dat in enumerate(this_session_dat):
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

            if len(fit_list) > 0:
                ll_names = ['actual','trial_sh','circ_sh','rand_sh']
                ll_outs = [[gt.calc_loglikelihood(actual_design_mat, res)\
                        for i in range(n_shuffles_per_fit)]\
                        for res in tqdm(fit_list)]
                ll_outs = np.array(ll_outs).reshape(-1,4)
                ll_frame = pd.DataFrame(ll_outs, columns=ll_names)
                ll_frame['fit_num'] = np.repeat(np.arange(len(fit_list)), n_shuffles_per_fit)


            elif len(fit_list) == 1:
                best_fit = fit_list[0]
            else:
                # Grab best fit
                best_fit_ind = ll_frame.actual.idxmax()
                best_fit = fit_list[best_fit_ind]

            # PSTH
            time_bins = actual_design_mat.trial_time.unique()
            design_trials = list(actual_design_mat.groupby('trial_labels'))
            design_spikes = [x.sort_values('trial_time').spikes.values for _,x in design_trials]
            design_spikes_array = np.stack(design_spikes)
            design_spikes_list.append(design_spikes_array)

            # Predicted PSTH
            pred_spikes = pd.DataFrame(best_fit.predict(actual_design_mat[best_fit.params.index]), 
                                       columns = ['spikes'])
            # Cutoff at 1 spike per bin
            pred_spikes.loc[pred_spikes.spikes > bin_width, 'spikes'] = bin_width
            pred_spikes['trial_labels'] = actual_design_mat.trial_labels
            pred_spikes['trial_time'] = actual_design_mat.trial_time
            pred_trials = list(pred_spikes.groupby('trial_labels'))
            pred_spikes = [x.sort_values('trial_time').spikes.values for _,x in pred_trials]
            pred_spikes_array = np.stack(pred_spikes)
            pred_spikes_list.append(pred_spikes_array)

        design_spikes_stack = np.stack(design_spikes_list)
        pred_spikes_stack = np.stack(pred_spikes_list)

        mean_design_spikes = design_spikes_stack.mean(axis=1)
        mean_pred_spikes = pred_spikes_stack.mean(axis=1)

        # Smoothen
        kern_len = 20
        kern = np.ones(kern_len) / kern_len

        smooth_design_spikes = np.apply_along_axis(
                lambda x: np.convolve(x, kern, mode = 'same'),
                1,
                mean_design_spikes,
                )

        smooth_pred_spikes = np.apply_along_axis(
                lambda x: np.convolve(x, kern, mode = 'same'),
                1,
                mean_pred_spikes,
                )

        fig, ax = plt.subplots(1,2, sharey = True, sharex = True, figsize = (7,3))
        ax[0].plot(time_bins, smooth_design_spikes.T,  alpha = 0.7)
        ax[1].plot(time_bins, smooth_pred_spikes.T,  alpha = 0.7)
        ax[0].set_title('Actual')
        ax[1].set_title('Predicted')
        ax[0].set_ylabel('Firing rate (Hz)')
        ax[0].set_xlabel('Time (s)')
        ax[1].set_xlabel('Time (s)')
        # Set ylim to be max of real data
        #ax[0].set_ylim([0,smooth_design_spikes.max()])
        #ax[1].set_ylim([0,smooth_design_spikes.max()])
        plt.tight_layout()
        plt.subplots_adjust(top = 0.85)
        plt.suptitle(f'Session {this_ind[0]}, Neuron {this_ind[1]}')
        plt.savefig(os.path.join(psth_plot_dir,
                                 'psth_comp_'+ f'{num}_' + "_".join([str(x) for x in this_ind]) + '.png'),
                    dpi = 300, bbox_inches = 'tight')
        plt.close()
        #plt.show()

############################################################
# Process inferred filters 
############################################################
alpha = 0.01

# Length of basis is adjusted because models were fit on binned data
hist_cosine_basis = gt.cb.gen_raised_cosine_basis(
        hist_filter_len_bin,
        n_basis = basis_kwargs['n_basis'],
        spread = basis_kwargs['basis_spread'],
        )
stim_cosine_basis = gt.cb.gen_raised_cosine_basis(
        stim_filter_len_bin,
        n_basis = basis_kwargs['n_basis'],
        spread = basis_kwargs['basis_spread'],
        )
coup_cosine_basis = gt.cb.gen_raised_cosine_basis(
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

sig_alpha = 0.05
############################################################
# Extract history filters
hist_frame = fin_pval_frame.loc[fin_pval_frame.param.str.contains('hist')]
#hist_frame.drop(columns = ['trial_sh','circ_sh','rand_sh'], inplace = True)
hist_frame = hist_frame[['fit_num','param','p_val','values', *ind_names]]
hist_frame['lag'] = hist_frame.param.str.extract('(\d+)').astype(int)
hist_groups = [x[1] for x in list(hist_frame.groupby(ind_names))]
hist_groups = [x.sort_values('lag') for x in hist_groups]
hist_val_array = np.stack([x['values'].values for x in hist_groups])
hist_pval_array = np.stack([x['p_val'].values for x in hist_groups])

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

fig, ax = plt.subplots(2,1, sharex=True)
for i, dat in enumerate(pca_array.T):
    ax[0].plot(dat, label = f'PC {i}, {np.round(pca.explained_variance_ratio_[i],2)}', 
               linewidth = 5, alpha = 0.7)
ax[0].legend()
ax[0].set_ylabel('PC Magnitude')
ax[1].imshow(pca_array.T, aspect = 'auto', interpolation = 'none')
ax[1].set_ylabel('PC #')
ax[1].set_xlabel('Time (ms)')
fig.suptitle(f'PCA of history filters, \n Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}')
fig.savefig(os.path.join(plot_dir, 'hist_filter_pca.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)
#plt.show()

# Plot each filter in it's own subplot
plot_cutoff = 50
fig, ax = plt.subplots(len(pca_array.T), 1, sharex=True, sharey=True,
                       figsize = (3,10))
peak_markers = [3,6,11,19]
for i, (this_dat, this_ax) in enumerate(zip(pca_array.T[:,:plot_cutoff], ax)):
    this_ax.plot(this_dat)
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
stim_frame = stim_frame[['fit_num','param','p_val','values', *ind_names]]
stim_frame['lag'] = stim_frame.param.str.extract('(\d+)').astype(int)
stim_groups = [x[1] for x in list(stim_frame.groupby(ind_names))]
stim_groups = [x.sort_values('lag') for x in stim_groups]
stim_val_array = np.stack([x['values'].values for x in stim_groups])
stim_pval_array = np.stack([x['p_val'].values for x in stim_groups])

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
    ax[0].plot(dat, label = f'PC {i}, {np.round(pca.explained_variance_ratio_[i],2)}', 
               linewidth = 2, alpha = 0.7)
ax[0].legend()
ax[0].set_ylabel('PC Magnitude')
ax[1].imshow(pca_array.T, aspect = 'auto', interpolation = 'none')
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
    this_ax.plot(this_dat)
    this_ax.set_ylabel(f'PC {i} : {np.round(pca.explained_variance_ratio_[i],2)}')
ax[-1].set_xlabel('Time (ms)')
fig.suptitle(f'PCA of stimulus filters, \n Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}')
fig.savefig(os.path.join(plot_dir, 'stim_filter_pca2.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

# Plot each filter in it's own subplot
plot_cutoff = 50
fig, ax = plt.subplots(len(pca_array.T), 1, sharex=True, sharey=True,
                       figsize = (3,10))
peak_markers = [2,4,8,16]
for i, (this_dat, this_ax) in enumerate(zip(pca_array.T[:,:plot_cutoff], ax)):
    this_ax.plot(this_dat)
    this_ax.set_ylabel(f'PC {i} : {np.round(pca.explained_variance_ratio_[i],2)}')
    for this_peak in peak_markers:
        this_ax.axvline(this_peak, linestyle = '--', color = 'k', alpha = 0.5)
ax[-1].set_xlabel('Time (ms)')
pca_str = f'Total variance explained: {np.round(pca.explained_variance_ratio_.sum(),2)}'
marker_str = f'Peak markers: {peak_markers} ms'
fig.suptitle(f'PCA of Stim filters (zoomed) \n' + pca_str + '\n' + marker_str)
fig.savefig(os.path.join(plot_dir, 'stim_filter_pca2_zoom.png'), dpi = 300, bbox_inches = 'tight')
plt.close(fig)

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
coupling_pivoted_raw_inds = [np.where((x < alpha).sum(axis=1))[0] \
        for x in coupling_pivoted_pvals]
coupling_pivoted_frame_index = [x.index.values for x in coupling_pivoted_vals]
coupling_pivoted_sig_inds = [y[x] for x,y in zip(coupling_pivoted_raw_inds, coupling_pivoted_frame_index)]

########################################
# Coupling filter profiles
coupling_val_array = np.concatenate(coupling_pivoted_vals, axis = 0)

# Reconstruct coupling filters
coupling_recon = np.dot(coupling_val_array, coup_cosine_basis)

# plot principle components
pca = PCA(n_components = 5)
pca.fit(coupling_recon.T)
pca_array = pca.transform(coupling_recon.T)

fig, ax = plt.subplots(2,1, sharex=True)
for i, dat in enumerate(pca_array.T):
    ax[0].plot(dat, label = f'PC {i}, {np.round(pca.explained_variance_ratio_[i],2)}', 
               linewidth = 2, alpha = 0.7)
ax[0].legend()
ax[0].set_ylabel('PC Magnitude')
ax[1].imshow(pca_array.T, aspect = 'auto', interpolation = 'none')
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
    this_ax.plot(this_dat)
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
    this_ax.plot(this_dat)
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
# bla --> gc
bla_to_gc = tuple_frame.dropna().query('region == "gc" and input_region == "bla"').groupby(['session','input_neuron']).mean().reset_index()[['session','input_neuron']]
gc_to_bla = tuple_frame.dropna().query('region == "bla" and input_region == "gc"').groupby(['session','input_neuron']).mean().reset_index()[['session','input_neuron']]

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
