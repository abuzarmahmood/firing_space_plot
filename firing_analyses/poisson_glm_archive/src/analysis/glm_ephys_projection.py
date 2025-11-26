"""
Using fit models, breakdown the contribution of each
factor to the activity of each neuron
"""

import sys
from pprint import pprint
import matplotlib_venn as venn
import json
import pingouin as pg
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore, gaussian_kde
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu as mwu
from glob import glob
from itertools import product
base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm'
sys.path.append(base_path)
import utils.glm_ephys_process_utils as process_utils
import analysis.aggregate_utils as aggregate_utils
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
#from ephys_data import ephys_data
from tqdm import tqdm, trange
import os
from pandas import concat
from pandas import DataFrame as df
import numpy as np
import pylab as plt
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from functools import partial
import pickle
from scipy.signal import savgol_filter

def parallelize(func, iterator, n_jobs = 16):
    return Parallel(n_jobs = n_jobs)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))


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

def return_filter_contribution(
        this_ind, 
        fin_save_path,
        spike_list,
        stim_vec,
        params_dict,
        fin_pval_frame,
        fit_type = 'actual',
        # kern_len = 20, # For smoothing
        ):

    input_list=  []
    ll_pval_row = ll_pval_frame.loc[
        ll_pval_frame['dat_ind'].isin([list(this_ind)])]
    neuron_region = ll_pval_row['region'].values[0]

    this_design_mat = process_utils.return_design_mat(
            this_ind,
            fin_save_path,
            spike_list,
            stim_vec, 
            params_dict,
            force_process=True,
            )
    trial_cols = [x for x in this_design_mat.columns if 'trial' in x]

    # Get spiking data
    design_spike_dat = this_design_mat[['spikes', *trial_cols]]
    design_spike_pivot = design_spike_dat.pivot_table(
        index='trial_labels', columns='trial_time', values='spikes')

    trial_col_dat = this_design_mat[trial_cols]
    this_filter_values = fin_pval_frame.loc[
            fin_pval_frame['dat_ind'].isin([list(this_ind)])
            ][['param','values','dat_ind', 'fit_num', 'fit_type']]
    this_filter_values = this_filter_values.loc[
            this_filter_values.fit_type == fit_type]
    this_design_mat = this_design_mat[this_filter_values['param'].values]

    # ##############################
    # # Regen PSTHs
    # full_pred = np.exp(
    #         this_design_mat.values @ this_filter_values['values'].values)
    # # Cap at bin_width
    # full_pred[full_pred > bin_width] = bin_width

    # # Add trial labels and time
    # full_pred_w_time = pd.concat([
    #     pd.DataFrame(dict(pred = full_pred), index=this_design_mat.index),
    #     trial_col_dat], axis=1)

    # full_pred_pivot = full_pred_w_time.pivot_table(
    #     index='trial_labels', columns='trial_time', values='pred')

    # # Smoothen
    # #pred_smooth = np.apply_along_axis(
    # #    conv_1d, axis=1, arr=full_pred_pivot.values)
    # #spike_smooth = np.apply_along_axis(
    # #    conv_1d, axis=1, arr=design_spike_pivot.values)

    # ##############################

    param_key = 'coup'
    for input_region in ['bla', 'gc']:
        if param_key in ['coup', 'coupling']:
            param_str = param_key + '_' + input_region
        else:
            param_str = param_key

        # Parse param names for indexing below
        key_filter_values = this_filter_values[
            this_filter_values['param'].str.contains(param_key)]
        key_filter_values['input_nrn'] = key_filter_values['param'].str.split(
            '_').str[2].astype(int)
        key_filter_values['lag'] = key_filter_values['param'].str.split(
            '_').str[-1].astype(int)
        
        # Get inds of neurons from desired input region
        wanted_unit_region_frame = unit_region_frame[unit_region_frame.session == this_ind[0]]
        wanted_unit_region_frame = wanted_unit_region_frame[\
                wanted_unit_region_frame.region == input_region]
        key_filter_values = key_filter_values[key_filter_values.input_nrn.isin(\
                wanted_unit_region_frame.neuron.values)]

        # Index design mat by same parameters
        key_design_mat = this_design_mat.loc[:,key_filter_values['param'].values]
        # Perform prediction
        pred = key_design_mat.values @ key_filter_values['values'].values

        # Convert to PSTH
        pred_w_time = pd.concat([
            pd.DataFrame(dict(pred = pred), index=key_design_mat.index),
            trial_col_dat], axis=1) 
        pred_pivot = pred_w_time.pivot_table(
            index='trial_labels', columns='trial_time', values='pred')
        mean_pred = pred_pivot.mean(axis=0)
        exp_mean_pred = np.exp(mean_pred)
        # smooth_exp_mean_pred = np.convolve(exp_mean_pred, kern, mode='valid')
        # smooth_time = np.convolve(pred_pivot.columns, kern, mode='valid')

        input_dict = dict(
            this_ind = this_ind,
            neuron_region = neuron_region,
            input_region = input_region,
            param_key = param_key,
            exp_mean_pred = exp_mean_pred,
            )
        input_list.append(input_dict)
    del this_design_mat, design_spike_dat, design_spike_pivot, trial_col_dat
    # del full_pred, full_pred_w_time, full_pred_pivot
    del pred, pred_w_time, pred_pivot, mean_pred, exp_mean_pred
    return input_list

    ## Plot mean pred_pivot
    #fig,ax = plt.subplots(4,1, sharex=True)
    ## Raw mean pred
    #ax[0].plot(full_pred_pivot.columns, pred_smooth.mean(axis=0))
    #ax[0].plot(design_spike_pivot.columns, spike_smooth.mean(axis=0))
    #ax[0].set_title('Mean PSTHs')
    #ax[0].legend(['pred', 'spike'])
    #ax[1].plot(pred_pivot.columns, mean_pred)
    #ax[2].plot(pred_pivot.columns, exp_mean_pred)
    #ax[3].plot(smooth_time, smooth_exp_mean_pred)
    #fig.suptitle(f'{param_key} contribution, ind: {this_ind}')
    #plt.savefig(os.path.join(
    #    contribution_plot_dir, f'{param_str}_contribution_{this_ind}.png'))
    #plt.close()
    ##plt.show()

############################################################
save_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
input_run_ind = 6
run_str = f'run_{input_run_ind:03d}'
plot_dir = f'/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/plots/{run_str}'
fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03}')

json_path = os.path.join(save_path, run_str,'fit_params.json')
params_dict = json.load(open(json_path))

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Parameters
set_params_to_globals(save_path, run_str)

############################################################
############################################################
# Get Spikes
save_path = os.path.join(base_path, 'artifacts')
spike_list_path = os.path.join(save_path, 'spike_save')
############################################################
# Reconstitute data
spike_inds_paths = sorted(
    glob(os.path.join(spike_list_path, '*_spike_inds.npy')))
spike_inds_list = [np.load(x) for x in spike_inds_paths]
spike_list = [process_utils.gen_spike_train(x) for x in spike_inds_list]

# Load unit_region_frame
unit_region_frame = pd.read_csv(os.path.join(
    save_path, 'unit_region_frame.csv'), index_col=0)

# Load ind_frame
ind_frame = pd.read_csv(os.path.join(save_path, 'ind_frame.csv'), index_col=0)

# Sort inds by total number of neurons per session
# This is needed because larger sessions take a long time to fit
count_per_session = ind_frame.groupby(by='session').count().values[:, 0]
ind_frame['count'] = count_per_session[ind_frame['session'].values]
ind_frame = ind_frame.sort_values(by='count')
fin_inds = ind_frame.values[:, :-1]  # Drop Count

############################################################
# Process data
stim_vec = np.zeros(spike_list[0].shape[-1])
stim_vec[stim_t] = 1

############################################################
# Load Data
(unit_region_frame,
    fin_pval_frame, 
    fin_ll_frame, 
    pred_spikes_list, 
    design_spikes_list, 
    ind_frame,
    session_taste_inds,
    all_taste_inds,
    ) = aggregate_utils.return_data(save_path, run_str)

max_ll_frame = fin_ll_frame[['fit_num','actual',*ind_names]]
max_inds = max_ll_frame.groupby(ind_names).actual.idxmax().reset_index().actual
max_vals = max_ll_frame.loc[max_inds].drop(columns = 'actual') 

fin_pval_frame = fin_pval_frame.merge(max_vals, on = ['fit_num',*ind_names])
# fin_ll_frame = fin_ll_frame.merge(max_vals, on = ['fit_num',*ind_names])

############################################################
############################################################

# # How many neurons are significant
# # Can perform paired test as all were tested on same model
# grouped_ll_frame = list(fin_ll_frame.groupby(['session', 'taste', 'neuron']))
# grouped_ll_inds, grouped_ll_frame = zip(*grouped_ll_frame)
# sh_cols = [x for x in grouped_ll_frame[0].columns if 'sh' in x]
# 
# ll_pval_list = []
# ll_stat_list = []
# for i in trange(len(grouped_ll_frame)):
#     this_frame = grouped_ll_frame[i]
#     pval_dict = {}
#     stat_dict = {}
#     for this_col in sh_cols:
#         try:
#             this_pval = wilcoxon(this_frame[this_col], this_frame['actual'])
#             pval_dict[this_col] = this_pval.pvalue
#             stat_dict[this_col] = this_pval.statistic
#         except ValueError:
#             pval_dict[this_col] = np.nan
#             stat_dict[this_col] = np.nan
#     ll_pval_list.append(pval_dict)
#     ll_stat_list.append(stat_dict)
# 
# ll_pval_frame = np.log10(pd.DataFrame(ll_pval_list))
# grouped_ll_inds_frame = pd.DataFrame(
#     grouped_ll_inds, columns=['session', 'taste', 'neuron'])
# ll_pval_frame = pd.concat([grouped_ll_inds_frame, ll_pval_frame], axis=1)
# 
# ll_stat_frame = pd.DataFrame(ll_stat_list)
# ll_stat_frame = pd.concat([grouped_ll_inds_frame, ll_stat_frame], axis=1)
# 
# # Drop nan rows
# ll_pval_frame = ll_pval_frame.dropna()
# ll_stat_frame = ll_stat_frame.dropna()
# 
# # Sort by session, taste, neuron
# ll_pval_frame = ll_pval_frame.sort_values(by=['session', 'taste', 'neuron'])
# ll_stat_frame = ll_stat_frame.sort_values(by=['session', 'taste', 'neuron'])
# 
# wanted_cols = [x for x in ll_pval_frame.columns if 'sh' in x]
# thresh = np.round(np.log10(0.1), 2)
# 
# ############################################################
# # Toss out neurons that are not significant for all 3 shuffles
# sig_rows = np.all((ll_pval_frame[wanted_cols] < thresh).values, axis=-1)
# ll_pval_frame = ll_pval_frame[sig_rows]
# 
# # Merge ll_pval_frame with unit_region_frame
# ll_pval_frame = ll_pval_frame.merge(
#     unit_region_frame, on=['session', 'neuron'])
# 
# wanted_inds = ll_pval_frame[ind_names].values
# ############################################################
 
fin_pval_frame['dat_ind'] = fin_pval_frame[ind_names].values.tolist()
# ll_pval_frame['dat_ind'] = ll_pval_frame[ind_names].values.tolist()


############################################################
############################################################
# Recreate contributions to firing per ind
contribution_plot_dir = os.path.join(
    plot_dir, 'filter_contribution_plots')
if not os.path.isdir(contribution_plot_dir):
    os.mkdir(contribution_plot_dir)

kern = np.ones(kern_len) / kern_len
conv_1d = lambda m: np.convolve(m, kern, mode='valid')

# fit_type = 'actual'
peak_dat_list = []
for fit_type in ['actual','trial_shuffled']:
    pkl_save_path = os.path.join(fin_save_path, f'{fit_type}_filter_contributions.pkl')
    if not os.path.isfile(pkl_save_path):
        return_contribution_parallel = partial(return_filter_contribution,
                                               fin_save_path = fin_save_path,
                                               spike_list = spike_list,
                                               stim_vec = stim_vec,
                                               params_dict = params_dict,
                                               fin_pval_frame = fin_pval_frame,
                                               fit_type = fit_type,
                                               # kern_len = 20,
                                               )
        # return_contribution_parallel(wanted_inds[0])
        outs = parallelize(return_contribution_parallel, wanted_inds, n_jobs=16)
        # Dump as a pickle with fit_type in name
        outs = [x for x in outs if x is not None]
        outs = [x for y in outs for x in y]
        pickle.dump(outs, open(pkl_save_path, 'wb'))
    else:
        outs = pickle.load(open(pkl_save_path, 'rb'))

    time_vec = outs[0]['exp_mean_pred'].index.values

    input_frame = pd.DataFrame(outs)
    input_frame_groups = list(input_frame.groupby(['neuron_region', 'input_region']))

    fig,ax = plt.subplots(3,4, sharex=True, sharey='row', figsize = (15,10))
    fig2, ax2 = plt.subplots(4,1, sharex=True, sharey=True, figsize = (5,10))
    for i, ((nrn_region, input_region), dat) in enumerate(input_frame_groups): 
        cont_dat = np.stack([x.values for x in dat.exp_mean_pred])
        cont_dat_smooth = np.apply_along_axis(conv_1d, axis=1, arr=cont_dat)
        time_smooth = np.convolve(time_vec, kern, mode='valid')
        zscore_cont_dat = zscore(cont_dat_smooth,axis=-1)

        ## Sort by kmeans
        #kmeans = KMeans(n_clusters=4, random_state=0).fit(zscore_cont_dat)
        #sort_inds = np.argsort(kmeans.labels_)
        #zscore_cont_dat = zscore_cont_dat[sort_inds,:]
        
        # Sort by peak location
        peak_inds = np.argmax(zscore_cont_dat, axis=1)
        sort_inds = np.argsort(peak_inds)[::-1]
        sorted_zscore_cont_dat = zscore_cont_dat[sort_inds,:]

        kde = gaussian_kde(time_smooth[peak_inds])
        kde_dat = kde(time_smooth)

        peak_times = time_smooth[peak_inds]
        save_dict = dict(
                fit_type = fit_type,
                input_region = input_region,
                nrn_region = nrn_region,
                peak_times = peak_times,
                cont_dat = cont_dat,
                zscore_smooth_cont_dat = zscore_cont_dat, # Don't take sorted
                )
        peak_dat_list.append(save_dict)

        ax2[i].hist(peak_times, bins = 30, density=True, alpha = 0.5);
        ax2[i].plot(time_smooth, kde_dat)
        ax2[i].axvline(stim_t, color='r', linestyle='--')
        ax2[i].set_title(f'{input_region}-->{nrn_region}')

        ax[0,i].hist(time_smooth[peak_inds], bins = 30, density=True, alpha = 0.5);
        ax[0,i].plot(time_smooth, kde_dat)
        ax[0,i].axvline(stim_t, color='r', linestyle='--')
        ax[0,i].set_title(f'{input_region}-->{nrn_region}' + '\n' + 'Peak time distribution')
        ax[1,i].pcolormesh(
             time_smooth, np.arange(cont_dat.shape[0]),
             sorted_zscore_cont_dat, cmap='RdBu_r', vmin=-2, vmax=2)
        ax[1,i].axvline(stim_t, color='r', linestyle='--')
        ax[1,i].set_title('Zscored Contribution matrix')
        # Perform PCA
        pca = PCA(n_components=3)
        pca.fit(sorted_zscore_cont_dat.T)
        pca_dat = pca.transform(sorted_zscore_cont_dat.T)
        explained_variances = np.round(pca.explained_variance_ratio_,2)
        for j in range(len(pca_dat.T)):
            ax[2,i].plot(time_smooth, pca_dat[:,j], label=f'PC{j+1}',
                       linewidth=2, alpha=0.8)
        ax[2,i].axvline(stim_t, color='r', linestyle='--')
        ax[2,i].legend()
        ax[2,i].set_title('PCA' + '\n' + \
                f'Explained variances: {explained_variances}' + '\n' + \
                f'SUM: {np.round(explained_variances.sum(),2)}')
        ax[2,i].set_xlabel('Time (ms)')
        #plt.show()
    fig.suptitle(f'Projection contribution, fit type: {fit_type}')
    fig.tight_layout()
    fig.savefig(os.path.join(
        contribution_plot_dir, f'projection_contribution_agg_fit_type_{fit_type}.png'),
        bbox_inches='tight', dpi=300)
    plt.close(fig)

    # fig2_peaks = [
    #         [2400],
    #         [2300, 2900],
    #         [2600],
    #         [2300, 2800],
    #         ]
    fig_2_peaks = [2300, 2800]
    #for peaks, this_ax in zip(fig2_peaks, ax2):
    for this_ax in ax2:
        for this_peak in fig_2_peaks:
            this_ax.axvline(this_peak, color='k', linestyle='--', alpha = 0.7)
            this_ax.text(this_peak, 0, f'{this_peak}', rotation=90, fontsize=8)

    fig2.suptitle(f'Peak time distribution, fit type: {fit_type}')
    fig2.tight_layout()
    fig2.savefig(os.path.join(
        contribution_plot_dir, f'peak_time_distribution_fit_type_{fit_type}.png'),
        bbox_inches='tight', dpi=300)
    plt.close(fig2)
 
##############################
# Comparison of actual - shuffle peak time distribution
peak_dat_frame = pd.DataFrame(peak_dat_list)
peak_dat_frame_list = [x for x in peak_dat_frame.groupby('fit_type')]
peak_frame_inds = [x[0] for x in peak_dat_frame_list]
peak_frame_dat = [x[1] for x in peak_dat_frame_list]
ind_dat_str = [x[['input_region', 'nrn_region']].values for x in peak_frame_dat][0]
ind_dat_str = [f'{x[0]}->{x[1]}' for x in ind_dat_str]

hist_bins = np.linspace(1000, 4500, 30)
peak_dat_hists = [[np.histogram(x, bins=hist_bins)[0] for x in this_type.peak_times] \
        for this_type in peak_frame_dat]
peak_dat_hists = np.array(peak_dat_hists)

diff_str = " - ".join(peak_frame_inds)
peak_dat_hists_diff = peak_dat_hists[0] - peak_dat_hists[1]

dt = np.diff(hist_bins)[0]
fig, ax = plt.subplots(3, len(peak_dat_hists[0]), 
                       figsize=(15,10), sharex=True, sharey=True)
for i, this_type in enumerate(peak_dat_hists):
    for j, this_hist in enumerate(this_type):
        ax[i,j].bar(hist_bins[:-1], this_hist, width = dt)
        ax[i,j].axvline(2000, color='r', linestyle='--')
        # ax[i,j].set_title(f'{peak_frame_inds[i]}')
        if i == 0:
            ax[i,j].set_title(f'{ind_dat_str[j]}')
        #ax[i,j].set_xlabel('Peak time (ms)')
        if j == 0:
            ax[i,j].set_ylabel(peak_frame_inds[i])
for i, this_hist in enumerate(peak_dat_hists_diff):
    ax[2,i].bar(hist_bins[:-1], this_hist, width = dt)
    # Also plot smoothed overlay
    smoothed_hist = savgol_filter(this_hist, 5, 2)
    ax[2,i].plot(hist_bins[:-1], smoothed_hist, color='k', linewidth=2, alpha = 0.7)
    ax[2,i].axvline(2000, color='r', linestyle='--')
    ax[2,i].set_xlabel('Peak time (ms)')
    if i == 0:
        ax[2,i].set_ylabel(diff_str)
fig.suptitle('Peak time distribution')
fig.tight_layout()
fig.savefig(os.path.join(
    contribution_plot_dir, f'peak_time_distribution_diff_{diff_str}.png'),
    bbox_inches='tight', dpi=300)
plt.close(fig)
# plt.show()

for this_name, this_dat in zip(peak_dat_hists_diff, ind_dat_str):
    plt.plot(hist_bins[:-1],this_name, '-x', label=this_dat,
             linewidth = 2, alpha = 0.7)
plt.legend()
plt.axvline(2000, color='k', linestyle='--')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time (ms)')
plt.ylabel('Difference in peak time distribution')
plt.title('Difference in peak time distribution')
plt.savefig(os.path.join(
    contribution_plot_dir, f'peak_time_distribution_diff_{diff_str}_line.png'),
    bbox_inches='tight', dpi=300)
plt.close(fig)
# plt.show()

diff_coef = np.corrcoef(peak_dat_hists_diff)
plt.matshow(diff_coef)
plt.xticks(range(len(ind_dat_str)), ind_dat_str)
plt.yticks(range(len(ind_dat_str)), ind_dat_str)
plt.colorbar()
plt.title('Correlation between peak time distribution')
plt.savefig(os.path.join(
    contribution_plot_dir, f'peak_time_distribution_diff_{diff_str}_corr.png'),
    bbox_inches='tight', dpi=300)
plt.close(fig)

##############################
# Instead of differencing distributions, difference the contributions
# of the projections on a single neuron basis

# Pull out cont_dat
cont_dat_array = np.array([x.cont_dat for x in peak_frame_dat])
cont_diff_array = cont_dat_array[0] - cont_dat_array[1]
zscore_cont_diff = [zscore(x,axis=-1) for x in cont_diff_array]
zscore_cont_diff_smooth = [np.apply_along_axis(conv_1d, axis=1, arr=x) \
        for x in zscore_cont_diff]
mean_zscore_cont_diff_smooth = np.array([np.mean(x, axis=0) for \
        x in zscore_cont_diff_smooth])

zscore_smooth_cont_dat_array = np.array([x.zscore_smooth_cont_dat for x in peak_frame_dat])
actual_zscore_smooth_cont_dat = zscore_smooth_cont_dat_array[0]
mean_actual_zscore_smooth_cont_dat = np.array([np.mean(x, axis=0) for \
        x in actual_zscore_smooth_cont_dat])


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10,5))
for this_dat, this_name in zip(mean_actual_zscore_smooth_cont_dat, ind_dat_str):
    ax[0].plot(smooth_time,this_dat, label=this_name, linewidth=4, alpha=0.7)
    # Normalize baseline to 0
    baseline_inds = np.where(smooth_time < 1500)[0]
    this_dat = this_dat - np.mean(this_dat[baseline_inds])
    ax[1].plot(smooth_time,this_dat, label=this_name, linewidth=4, alpha=0.7)
ax[0].set_title('Mean Zscore Smooth Contribution')
ax[1].set_title('Mean Zscore Smooth Contribution (baseline subtracted)')
fig.suptitle('Mean Zscore Smooth Directional Contribution')
plt.legend()
plt.axvline(2000, color='k', linestyle='--')
#plt.show()
fig.savefig(os.path.join(
    contribution_plot_dir, f'zscore_smooth_cont_dat_mean.png'),
    bbox_inches='tight', dpi=300)
plt.close(fig)


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(10,5))
for this_dat, this_name in zip(mean_zscore_cont_diff_smooth, ind_dat_str):
    ax[0].plot(smooth_time,this_dat, label=this_name, linewidth=4, alpha=0.7)
    # Normalize baseline to 0
    baseline_inds = np.where(smooth_time < 1500)[0]
    this_dat = this_dat - np.mean(this_dat[baseline_inds])
    ax[1].plot(smooth_time,this_dat, label=this_name, linewidth=4, alpha=0.7)
ax[0].set_title('Mean Zscore Smooth Contribution Difference')
ax[1].set_title('Mean Zscore Smooth Contribution  Difference (baseline subtracted)')
fig.suptitle('Mean Zscore Smooth Directional Contribution Difference')
plt.legend()
plt.axvline(2000, color='k', linestyle='--')
#plt.show()
fig.savefig(os.path.join(
    contribution_plot_dir, f'zscore_smooth_cont_diff_mean.png'),
    bbox_inches='tight', dpi=300)
plt.close(fig)


# Also look at non-zscored smoothed contributions to make sure
# GC is actually high in the beginning, and not just relatively
