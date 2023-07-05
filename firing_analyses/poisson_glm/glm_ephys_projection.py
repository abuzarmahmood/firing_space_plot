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
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
#from ephys_data import ephys_data
import glm_tools as gt
from tqdm import tqdm, trange
import os
from pandas import concat
from pandas import DataFrame as df
import numpy as np
import pylab as plt
import pandas as pd
from joblib import Parallel, delayed, cpu_count

def parallelize(func, iterator, n_jobs = 16):
    return Parallel(n_jobs = n_jobs)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def gen_spike_train(spike_inds):
    spike_train = np.zeros(spike_inds.max(axis=1)+1)
    spike_train[tuple(spike_inds)] = 1
    return spike_train


############################################################
#run_str = 'run_004'
save_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
# Check if previous runs present
run_list = sorted(glob(os.path.join(save_path, 'run*')))
run_basenames = sorted([os.path.basename(x) for x in run_list])
print(f'Present runs : {run_basenames}')

#input_run_ind = int(input('Please specify current run (integer) :'))
input_run_ind = 2
run_str = f'run_{input_run_ind:03d}'
plot_dir = f'/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/plots/{run_str}'
fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03}')
json_path = os.path.join(fin_save_path, 'fit_params.json')
params_dict = json.load(open(json_path))
print('Run exists with following parameters :')
pprint(params_dict)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

############################################################
# Load params
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
# Get Spikes

save_path = os.path.join(base_path, 'artifacts')
spike_list_path = os.path.join(save_path, 'spike_save')
############################################################
# Reconstitute data
spike_inds_paths = sorted(
    glob(os.path.join(spike_list_path, '*_spike_inds.npy')))
spike_inds_list = [np.load(x) for x in spike_inds_paths]
spike_list = [gen_spike_train(x) for x in spike_inds_list]

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


def return_design_mat(this_ind):
    this_ind_str = '_'.join([f'{x:03}' for x in this_ind])
    pval_save_name = f'{this_ind_str}_p_val_frame.csv'
    pval_save_path = os.path.join(fin_save_path, pval_save_name)
    ll_save_name = f'{this_ind_str}_ll_frame.csv'
    ll_save_path = os.path.join(fin_save_path, ll_save_name)

    this_session_dat = spike_list[this_ind[0]]
    this_taste_dat = this_session_dat[this_ind[1]]
    this_nrn_ind = this_ind[2]
    other_nrn_inds = np.delete(np.arange(this_session_dat.shape[2]),
                               this_nrn_ind)
    n_coupled_neurons = len(other_nrn_inds)

    this_nrn_dat = this_taste_dat[:, this_nrn_ind]
    other_nrn_dat = this_taste_dat[:, other_nrn_inds]
    stim_dat = np.tile(stim_vec, (this_taste_dat.shape[0], 1))

    this_nrn_flat = np.concatenate(this_nrn_dat)
    other_nrn_flat = np.concatenate(np.moveaxis(other_nrn_dat, 1, -1)).T
    stim_flat = np.concatenate(stim_dat)

    #import importlib
    # importlib.reload(gt)

    # To convert to dataframe, make sure trials are not directly
    # concatenated as that would imply temporal continuity
    data_frame = gt.gen_data_frame(
        this_nrn_flat,
        other_nrn_flat,
        stim_flat,
        stim_filter_len=stim_filter_len,
        trial_start_offset=trial_start_offset,
    )

    # Replace coupling data names with actual indices
    coup_names = ['coup_{}'.format(x) for x in range(n_coupled_neurons)]
    replace_names = ['coup_{}'.format(x) for x in other_nrn_inds]
    replace_dict = dict(zip(coup_names, replace_names))
    data_frame = data_frame.rename(columns=replace_dict)

    # Bin data
    data_frame['time_bins'] = data_frame.trial_time // bin_width
    data_frame = data_frame.groupby(['trial_labels', 'time_bins']).sum()
    data_frame['trial_time'] = data_frame['trial_time'] // bin_width
    data_frame = data_frame.reset_index()

    # Create design mat
    actual_design_mat = gt.dataframe_to_design_mat(
        data_frame,
        hist_filter_len=hist_filter_len_bin,
        stim_filter_len=stim_filter_len_bin,
        coupling_filter_len=coupling_filter_len_bin,
        basis_kwargs=basis_kwargs,
    )

    # Cut to trial_lims
    # Note, this needs to be done after design matrix is created
    # so that overlap of history between trials is avoided
    trial_lims_vec = np.arange(*trial_lims)
    actual_design_mat = actual_design_mat.loc[actual_design_mat.trial_time.isin(
        trial_lims_vec)]

    return actual_design_mat


############################################################
# Load Data

unit_region_frame = pd.read_csv(
    os.path.join(save_path, 'unit_region_frame.csv'),
    index_col=0)
unit_region_frame.rename(columns={'unit': 'neuron'}, inplace=True)
ind_frame = pd.read_csv(os.path.join(save_path, 'ind_frame.csv'), index_col=0)

p_val_frame_paths = sorted(
    glob(os.path.join(save_path, run_str, '*p_val_frame.csv')))
ll_frame_paths = sorted(
    glob(os.path.join(save_path, run_str, '*ll_frame.csv')))

p_val_basenames = [os.path.basename(x) for x in p_val_frame_paths]
ll_basenames = [os.path.basename(x) for x in ll_frame_paths]

p_val_inds_str = [x.split('_p_val')[0] for x in p_val_basenames]
ll_inds_str = [x.split('_ll')[0] for x in ll_basenames]

p_val_inds = np.array([list(map(int, x.split('_'))) for x in p_val_inds_str])
ll_inds = np.array([list(map(int, x.split('_'))) for x in ll_inds_str])

ind_names = ['session', 'taste', 'neuron']

# Make sure inds match up
assert np.all(p_val_inds == ll_inds), "Mismatched inds"

p_val_frame_list = [pd.read_csv(x, index_col=0) for x in p_val_frame_paths]
ll_frame_list = [pd.read_csv(x, index_col=0) for x in ll_frame_paths]

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
fin_pval_frame = fin_pval_frame.sort_values(by=['session', 'taste', 'neuron'])
# Reset index
fin_pval_frame = fin_pval_frame.reset_index(drop=True)

fin_ll_frame = pd.concat(ll_frame_list_fin)
# Sort by inds
fin_ll_frame = fin_ll_frame.sort_values(by=['session', 'taste', 'neuron'])
# Merge fin_ll_frame and unit_region_frame
fin_ll_frame = pd.merge(fin_ll_frame, unit_region_frame,
                        on=['session', 'neuron'])


############################################################
############################################################

# How many neurons are significant
# Can perform paired test as all were tested on same model
grouped_ll_frame = list(fin_ll_frame.groupby(['session', 'taste', 'neuron']))
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
grouped_ll_inds_frame = pd.DataFrame(
    grouped_ll_inds, columns=['session', 'taste', 'neuron'])
ll_pval_frame = pd.concat([grouped_ll_inds_frame, ll_pval_frame], axis=1)

ll_stat_frame = pd.DataFrame(ll_stat_list)
ll_stat_frame = pd.concat([grouped_ll_inds_frame, ll_stat_frame], axis=1)

# Drop nan rows
ll_pval_frame = ll_pval_frame.dropna()
ll_stat_frame = ll_stat_frame.dropna()

# Sort by session, taste, neuron
ll_pval_frame = ll_pval_frame.sort_values(by=['session', 'taste', 'neuron'])
ll_stat_frame = ll_stat_frame.sort_values(by=['session', 'taste', 'neuron'])

wanted_cols = [x for x in ll_pval_frame.columns if 'sh' in x]
thresh = np.round(np.log10(0.1), 2)

############################################################
# Toss out neurons that are not significant for all 3 shuffles
sig_rows = np.all((ll_pval_frame[wanted_cols] < thresh).values, axis=-1)
ll_pval_frame = ll_pval_frame[sig_rows]

# Merge ll_pval_frame with unit_region_frame
ll_pval_frame = ll_pval_frame.merge(
    unit_region_frame, on=['session', 'neuron'])

wanted_inds = ll_pval_frame[ind_names].values
############################################################

fin_pval_frame['dat_ind'] = fin_pval_frame[ind_names].values.tolist()
ll_pval_frame['dat_ind'] = ll_pval_frame[ind_names].values.tolist()


############################################################
############################################################
# Recreate contributions to firing per ind
contribution_plot_dir = os.path.join(
    plot_dir, 'filter_contribution_plots')
if not os.path.isdir(contribution_plot_dir):
    os.mkdir(contribution_plot_dir)

#for this_ind in tqdm(wanted_inds):
def return_filter_contribution(this_ind):
    input_list=  []
    ll_pval_row = ll_pval_frame.loc[
        ll_pval_frame['dat_ind'].isin([list(this_ind)])]
    neuron_region = ll_pval_row['region'].values[0]

    #this_ind = wanted_inds[6]
    this_design_mat = return_design_mat(this_ind)
    trial_cols = [x for x in this_design_mat.columns if 'trial' in x]

    # Get spiking data
    design_spike_dat = this_design_mat[['spikes', *trial_cols]]
    design_spike_pivot = design_spike_dat.pivot_table(
        index='trial_labels', columns='trial_time', values='spikes')

    trial_col_dat = this_design_mat[trial_cols]
    this_filter_values = fin_pval_frame.loc[
            fin_pval_frame['dat_ind'].isin([list(this_ind)])
            ][['param','values','dat_ind']]
    this_design_mat = this_design_mat[this_filter_values['param'].values]

    ##############################
    # Regen PSTHs
    full_pred = np.exp(
            this_design_mat.values @ this_filter_values['values'].values)
    # Cap at bin_width
    full_pred[full_pred > bin_width] = bin_width

    # Add trial labels and time
    full_pred_w_time = pd.concat([
        pd.DataFrame(dict(pred = full_pred), index=this_design_mat.index),
        trial_col_dat], axis=1)

    full_pred_pivot = full_pred_w_time.pivot_table(
        index='trial_labels', columns='trial_time', values='pred')

    # Smoothen
    kern_len = 200
    kern = np.ones(kern_len) / kern_len
    conv_1d = lambda m: np.convolve(m, kern, mode='valid')
    #pred_smooth = np.apply_along_axis(
    #    conv_1d, axis=1, arr=full_pred_pivot.values)
    #spike_smooth = np.apply_along_axis(
    #    conv_1d, axis=1, arr=design_spike_pivot.values)

    ##############################

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
        key_design_mat = this_design_mat[key_filter_values['param'].values]
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
        smooth_exp_mean_pred = np.convolve(exp_mean_pred, kern, mode='valid')
        smooth_time = np.convolve(pred_pivot.columns, kern, mode='valid')

        input_dict = dict(
            this_ind = this_ind,
            neuron_region = neuron_region,
            input_region = input_region,
            param_key = param_key,
            exp_mean_pred = exp_mean_pred,
            )
        input_list.append(input_dict)
    del this_design_mat, design_spike_dat, design_spike_pivot, trial_col_dat
    del full_pred, full_pred_w_time, full_pred_pivot
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

outs = parallelize(return_filter_contribution, wanted_inds, n_jobs=16)
outs = [x for x in outs if x is not None]
outs = [x for y in outs for x in y]
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
    zscore_cont_dat = zscore_cont_dat[sort_inds,:]

    kde = gaussian_kde(time_smooth[peak_inds])
    kde_dat = kde(time_smooth)

    ax2[i].hist(time_smooth[peak_inds], bins = 30, density=True, alpha = 0.5);
    ax2[i].plot(time_smooth, kde_dat)
    ax2[i].axvline(stim_t, color='r', linestyle='--')
    ax2[i].set_title(f'{input_region}-->{nrn_region}')

    ax[0,i].hist(time_smooth[peak_inds], bins = 30, density=True, alpha = 0.5);
    ax[0,i].plot(time_smooth, kde_dat)
    ax[0,i].axvline(stim_t, color='r', linestyle='--')
    ax[0,i].set_title(f'{input_region}-->{nrn_region}' + '\n' + 'Peak time distribution')
    ax[1,i].pcolormesh(
         time_smooth, np.arange(cont_dat.shape[0]),
         zscore_cont_dat, cmap='RdBu_r', vmin=-2, vmax=2)
    ax[1,i].axvline(stim_t, color='r', linestyle='--')
    ax[1,i].set_title('Zscored Contribution matrix')
    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(zscore_cont_dat.T)
    pca_dat = pca.transform(zscore_cont_dat.T)
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
fig.tight_layout()
fig.savefig(os.path.join(
    contribution_plot_dir, 'projection_contribution_agg.png'),
    bbox_inches='tight', dpi=300)
plt.close(fig)

fig2_peaks = [
        [2400],
        [2300, 2900],
        [2600],
        [2300, 2800],
        ]
for peaks, this_ax in zip(fig2_peaks, ax2):
    for this_peak in peaks:
        this_ax.axvline(this_peak, color='k', linestyle='--', alpha = 0.7)
        this_ax.text(this_peak, 0.001, f'{this_peak}', rotation=90, fontsize=8)

fig2.suptitle('Peak time distribution')
fig2.tight_layout()
fig2.savefig(os.path.join(
    contribution_plot_dir, 'peak_time_distribution.png'),
    bbox_inches='tight', dpi=300)
plt.close(fig2)

