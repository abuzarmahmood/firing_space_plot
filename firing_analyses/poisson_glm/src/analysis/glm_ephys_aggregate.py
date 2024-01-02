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
import visualize as vz
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
from collections import Counter 
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

# plt.rcParams.update({'font.size': 5})
# # Set rcParams to default
# plt.rcParams.update(plt.rcParamsDefault)

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
input_run_ind = 7
save_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03d}')
# Check if previous runs present
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

# There might an issue with the new inds calculation
# DOUBLE CHECK!!

(unit_region_frame,
    fin_pval_frame, 
    fin_ll_frame, 
    pred_spikes_list, 
    design_spikes_list, 
    ind_frame,
    session_taste_inds,
    all_taste_inds,
    test_dat_list,
    data_inds,
    ) = aggregate_utils.return_data(save_path, run_str)

data_inds_frame = pd.DataFrame(data_inds,
                               columns = ind_frame.columns)

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
# Pred R^2
############################################################
# Compare pred R^2 to:
#   1) Trial shuffled R^2
#   2) Neural shuffled R^2 

# pivot all frames to have rows=trials, cols=time
pivot_test_spikes_list = [x.pivot(index = 'trial_labels', columns = 'trial_time')
        for x in tqdm(test_dat_list)]
# Fill nans with 0
pivot_test_spikes_list = [x.fillna(0) for x in pivot_test_spikes_list]
test_spikes_list = [x['spikes'] for x in pivot_test_spikes_list]
test_pred_spikes_list = [x['pred_spikes'] for x in pivot_test_spikes_list]

# Keep the most common shape size
shape_list = [x.shape for x in test_spikes_list]
shape_counter = Counter(shape_list)

# First, just calculate actual R^2
# 1) on matching data
# 2) on trial shuffled data
# 2) on circularly shuffled data (shuffle time bins in same session)
kern_len = 200
kern = np.ones(kern_len)/kern_len

pred_r2_list = []
circ_shuffle_r2_list = []
trial_shuffled_r2_list = []
trial_avg_r2_list = []
circ_shuffle_avg_r2_list = []
# Calculate r^2 compared to psth on single trials
psth_r2_list = []

pred_mae_list = []
psth_mae_list = []
pred_corr_list = []
psth_corr_list = []
pred_corr_time_list = []

test_psth_plot_dir = os.path.join(plot_dir, 'test_psth')
if not os.path.exists(test_psth_plot_dir):
    os.makedirs(test_psth_plot_dir)

make_plots = False

for ind in tqdm(range(len(test_spikes_list))):

    # ind = 0
    test_spikes = test_spikes_list[ind].values
    test_pred_spikes = test_pred_spikes_list[ind].values

    test_psth = np.apply_along_axis(
            lambda m: np.convolve(m, kern, mode = 'same'), 
            axis = -1, 
            arr = test_spikes)
    test_pred_psth = np.apply_along_axis(
            lambda m: np.convolve(m, kern, mode = 'same'),
            axis = -1,
            arr = test_pred_spikes)

    mean_test_psth = np.mean(test_psth, axis = 0)
    mean_test_pred_psth = np.mean(test_pred_psth, axis = 0)
    circ_sh_mean_test_pred_psth = np.mean(
            np.random.permutation(test_pred_psth.T).T, axis = 0)

    pred_r2 = r2_score(test_psth, test_pred_psth)
    trial_avg_pred_r2 = r2_score(mean_test_psth, mean_test_pred_psth)
    circ_sh_trial_avg_pred_r2 = r2_score(mean_test_psth,
                                         circ_sh_mean_test_pred_psth)
    trial_shuffled_r2 = r2_score(test_psth, 
                                 np.random.permutation(test_pred_psth.T).T)
    circ_shuffle_r2 = r2_score(test_psth, 
                               np.random.permutation(test_pred_psth.T).T
                               ) 
    tiled_psth = np.tile(mean_test_psth,#*test_psth.shape[0], 
                         (test_psth.shape[0], 1))
    # For this comparison, have to svaled mean_test_psth
    psth_r2 = r2_score(tiled_psth, test_psth)

    # Calculate correlations for:
    #   1) test_psth vs tiled_psth
    #   2) test_psth vs test_pred_psth
    pred_r = pearsonr(test_psth.flatten(), test_pred_psth.flatten())[0]
    psth_r = pearsonr(tiled_psth.flatten(), test_psth.flatten())[0]

    pred_r_time = [pearsonr(test_psth[:, x], test_pred_psth[:, x]) \
            for x in range(test_psth.shape[1])]
    pred_r_time = [x[0] for x in pred_r_time]
    pred_corr_time_list.append(pred_r_time)

    # plt.plot(pred_r_time);plt.show()

    pred_corr_list.append(pred_r)
    psth_corr_list.append(psth_r)

    # Calculate mean absolute error for test_psth vs tiled_psth and 
    # test_pred_psth
    pred_mae = np.mean(np.abs(test_psth - test_pred_psth))
    psth_mae = np.mean(np.abs(test_psth - tiled_psth))

    pred_mae_list.append(pred_mae)
    psth_mae_list.append(psth_mae)

    pred_r2_list.append(pred_r2)
    trial_avg_r2_list.append(trial_avg_pred_r2)
    circ_shuffle_avg_r2_list.append(circ_sh_trial_avg_pred_r2)
    trial_shuffled_r2_list.append(trial_shuffled_r2)
    circ_shuffle_r2_list.append(circ_shuffle_r2)
    psth_r2_list.append(psth_r2)

    # Plot all trials for test_psth vs test_pred_psth
    if trial_avg_pred_r2 > 0.5 and make_plots and pred_r > psth_r:
        fig, ax = vz.gen_square_subplots(test_psth.shape[0],
                                         sharex = True, sharey = True)
        for i in range(test_psth.shape[0]):
            ax.flatten()[i].plot(1000*test_psth[i,:], label = 'Actual')
            ax.flatten()[i].plot(1000*test_pred_psth[i,:], label = 'Pred')
            if i == 0:
                ax.flatten()[i].legend()
        fig.savefig(os.path.join(test_psth_plot_dir, f'test_psth_{ind}.png'))
        plt.close(fig)

        # Also plot mean psths for both test and test_pred
        fig, ax = plt.subplots()
        ax.plot(1000*mean_test_psth, label = 'Actual')
        ax.plot(1000*mean_test_pred_psth, label = 'Pred')
        ax.legend()
        fig.savefig(os.path.join(test_psth_plot_dir, f'mean_test_psth_{ind}.png'))
        plt.close(fig)

        # Plot scatters of tiled psth vs test psth vs test pred psth
        fig, ax = plt.subplots(2,1, sharex=True, sharey=True,
                               figsize = (5,5))
        ax[0].scatter(tiled_psth.flatten(), test_psth.flatten(), alpha = 0.1)
        ax[0].set_title('Tiled psth vs test psth')
        # ax[0].set_aspect('equal')
        ax[0].set_xlabel('Tiled psth')
        ax[0].set_ylabel('Test psth')
        ax[1].scatter(test_pred_psth.flatten(), test_psth.flatten(), alpha = 0.1)
        ax[1].set_title('Test pred psth vs test psth')
        # ax[1].set_aspect('equal')
        ax[1].set_xlabel('Test pred psth')
        ax[1].set_ylabel('Test psth')
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        fig.suptitle(f'Pred corr: {pred_r:.3f}, Psth corr: {psth_r:.3f}') 
        min_val = np.min([ax[0].get_xlim()[0], ax[0].get_ylim()[0]])
        max_val = np.max([ax[0].get_xlim()[1], ax[0].get_ylim()[1]])
        ax[0].set_xlim([min_val, max_val])
        ax[0].set_ylim([min_val, max_val])
        for this_ax in ax:
            this_ax.plot([min_val, max_val], [min_val, max_val],
                         color = 'r', linestyle = '--')
        plt.tight_layout()
        fig.savefig(os.path.join(test_psth_plot_dir,
                                 f'test_psth_scatter_{ind}.png'),
                    bbox_inches = 'tight') 
        plt.close(fig)
        #plt.show()

        fig,ax = plt.subplots(3,1)
        vmin = np.min([test_psth, test_pred_psth, tiled_psth])
        vmax = np.max([test_psth, test_pred_psth, tiled_psth])
        img_kwargs = dict(vmin = vmin, vmax = vmax, 
                          aspect = 'auto', cmap = 'jet')
        im = ax[0].imshow(tiled_psth, **img_kwargs) 
        ax[0].set_title('Tiled psth')
        # ax[0].plot(mean_test_pred_psth, label = 'Mean test pred')
        # ax[0].plot(test_psth.T, alpha = 0.1)#label = 'Test data')
        ax[1].imshow(test_psth, **img_kwargs) 
        ax[1].set_title('Test psth')
        ax[2].imshow(test_pred_psth, **img_kwargs) 
        ax[2].set_title('Test pred psth')
        for this_ax in ax:
            plt.colorbar(im, ax = this_ax)
        plt.tight_layout()
        fig.savefig(os.path.join(test_psth_plot_dir,
                                 f'test_psth_heatmap_{ind}.png'),
                    bbox_inches = 'tight')
        plt.close(fig)
        # plt.show()

# Convert lists to arrays for easier manipulation
pred_r2_list = np.array(pred_r2_list)
trial_avg_r2_list = np.array(trial_avg_r2_list)
circ_shuffle_avg_r2_list = np.array(circ_shuffle_avg_r2_list)
trial_shuffled_r2_list = np.array(trial_shuffled_r2_list)
circ_shuffle_r2_list = np.array(circ_shuffle_r2_list)
psth_r2_list = np.array(psth_r2_list)

pred_mae_list = np.array(pred_mae_list)
psth_mae_list = np.array(psth_mae_list)

pred_corr_list = np.array(pred_corr_list)
psth_corr_list = np.array(psth_corr_list)

pred_corr_time_frame = pd.concat(
        [pd.DataFrame(dict(
            t = np.arange(len(x)), val = x, ind = ind)
                    )
                for ind, x in enumerate(pred_corr_time_list)]
        )
pred_corr_time_frame.dropna(inplace=True)

# Bin into 50ms bins
pred_corr_time_frame['t_bin'] = np.floor(pred_corr_time_frame['t']/50)
pred_corr_time_frame = pred_corr_time_frame.groupby(['ind','t_bin']).mean().reset_index()

mean_corr_time = pred_corr_time_frame.groupby('t_bin').mean().reset_index()
sd_corr_time = pred_corr_time_frame.groupby('t_bin').std().reset_index()

fig, ax = plt.subplots(1,2, sharey=True, figsize = (8,4))
ax[0].plot(mean_corr_time.t, mean_corr_time.val)
ax[0].fill_between(mean_corr_time.t,
                   mean_corr_time.val - sd_corr_time.val,
                   mean_corr_time.val + sd_corr_time.val,
                   alpha = 0.5)
ax[1].hist(pred_corr_time_frame.val, bins = 50, orientation = 'horizontal',
           alpha = 0.5)
ax[1].axhline(pred_corr_time_frame.val.mean(), color = 'r', linestyle = '--',
              label = 'Mean')
ax[1].legend()
ax[0].set_ylabel('Mean +/- SD Correlation')
ax[0].xlabel('Time (ms)')
fig.suptitle('Correlation over time')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'corr_over_time.png'),
            bbox_inches = 'tight')
plt.close(fig)
#plt.show()

# sns.lineplot(data = pred_corr_time_frame.iloc[::100], x = 't', y = 'val',
#              errorbar = 'sd')
# plt.show()

##############################
# Sort by largest differences between pred_corr and psth_corr
corr_diff = pred_corr_list - psth_corr_list
pred_greater_bool = corr_diff > 0

# Write out pred_greater_bool along with data_inds_frame 
data_inds_frame['pred_corr_greater'] = pred_greater_bool
data_inds_frame.to_csv(os.path.join(fin_save_path, 'data_inds_frame.csv'))

pred_greater_inds = np.where(pred_greater_bool)[0]
corr_sorted_inds = np.argsort(corr_diff)[::-1]
corr_sorted_inds = [x for x in corr_sorted_inds if x in pred_greater_inds]

corr_sort_plots_dir = os.path.join(plot_dir, 'corr_sorted_psth')
if not os.path.isdir(corr_sort_plots_dir):
    os.mkdir(corr_sort_plots_dir)

# Plot single-trial activity for sorted data
for i, ind in enumerate(corr_sorted_inds):
    this_pred_corr = pred_corr_list[ind]
    this_psth_corr = psth_corr_list[ind]

    test_spikes = test_spikes_list[ind].values
    test_pred_spikes = test_pred_spikes_list[ind].values

    test_psth = np.apply_along_axis(
            lambda m: np.convolve(m, kern, mode = 'same'), 
            axis = -1, 
            arr = test_spikes)
    test_pred_psth = np.apply_along_axis(
            lambda m: np.convolve(m, kern, mode = 'same'),
            axis = -1,
            arr = test_pred_spikes)
    mean_test_psth = np.mean(test_psth, axis = 0)
    tiled_psth = np.tile(mean_test_psth,#*test_psth.shape[0], 
                         (test_psth.shape[0], 1))

    fig,ax = plt.subplots(3,1)
    vmin = np.min([test_psth, test_pred_psth, tiled_psth])
    vmax = np.max([test_psth, test_pred_psth, tiled_psth])
    img_kwargs = dict(
                        #vmin = vmin, vmax = vmax, 
                      aspect = 'auto', cmap = 'jet',
                      #interpolation = 'none'
                      )
    im = ax[0].imshow(tiled_psth, **img_kwargs) 
    ax[0].set_title('Tiled psth')
    # ax[0].plot(mean_test_pred_psth, label = 'Mean test pred')
    # ax[0].plot(test_psth.T, alpha = 0.1)#label = 'Test data')
    ax[1].imshow(test_psth, **img_kwargs) 
    ax[1].set_title('Test psth')
    ax[2].imshow(test_pred_psth, **img_kwargs) 
    ax[2].set_title('Test pred psth')
    # for this_ax in ax:
    #     plt.colorbar(im, ax = this_ax)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.8)
    fig.suptitle(f'Ind: {ind}, ' + '\n' +\
                 f'Pred corr: {this_pred_corr:.3f}, '
                 f'Psth corr: {this_psth_corr:.3f}')
    fig.savefig(os.path.join(corr_sort_plots_dir,
                             f'test_psth_heatmap_{i}.png'),
                bbox_inches = 'tight')
    plt.close(fig)
    # plt.show()


##############################
# Scatter of pred_mae vs psth_mae
fig, ax = plt.subplots(1,3, sharex=False)
ax[0].scatter(pred_mae_list, psth_mae_list, alpha = 0.1,
           s = 10, color = 'k')
ax[0].set_xlabel('Pred mae')
ax[0].set_ylabel('Psth mae')
ax[0].set_aspect('equal')
min_val = np.min([ax[0].get_xlim()[0], ax[0].get_ylim()[0]])
max_val = np.max([ax[0].get_xlim()[1], ax[0].get_ylim()[1]])
ax[0].plot([min_val, max_val], [min_val, max_val], color = 'r', linestyle = '--')
# Project onto orthogonal axis
joint_dat = np.vstack([pred_mae_list, psth_mae_list])
proj_dat = np.dot(joint_dat.T, np.array([1, -1]))
ax[1].hist(proj_dat, bins = 50)
ax[1].axvline(x = 0, color = 'r', linestyle = '--')
ax[1].set_xlabel('<-- PSTH Larger Error | Pred Larger Error -->')
pred_greater_bool = pred_mae_list > psth_mae_list
pred_greater_frac = np.round(np.mean(pred_greater_bool),2)
# Plot firing rates depending on pred_greater_bool
pred_greater_spikes = [test_spikes_list[i].values.mean() for i in \
        range(len(test_spikes_list)) if pred_greater_bool[i]]
pred_lesser_spikes = [test_spikes_list[i].values.mean() for i in \
        range(len(test_spikes_list)) if not pred_greater_bool[i]]
ax[2].hist(pred_greater_spikes, bins = 50, alpha = 0.5, label = 'Pred > PSTH',
           density = True)
ax[2].hist(pred_lesser_spikes, bins = 50, alpha = 0.5, label = 'Pred < PSTH',
           density = True)
ax[2].legend()
ax[2].set_yscale('log')
ax[2].set_xlabel('Mean firing rate')
fig.suptitle('Pred mae vs psth mae\n' + \
        f'Pred > PSTH : {pred_greater_frac}' + \
        '\n' + \
        f'Pred < PSTH : {np.round(1-pred_greater_frac,2)}')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'pred_mae_vs_psth_mae.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################
# Scatter of pred_corr vs psth_corr
fig, ax = plt.subplots(1,3, sharex=False, figsize = (10,5))
ax[0].scatter(pred_corr_list, psth_corr_list, alpha = 0.1,
           s = 10, color = 'k')
ax[0].set_xlabel('Pred Corr')
ax[0].set_ylabel('Psth Corr')
ax[0].set_aspect('equal')
min_val = np.min([[np.nanmin(pred_corr_list), np.nanmin(psth_corr_list)]])
max_val = np.max([[np.nanmax(pred_corr_list), np.nanmax(psth_corr_list)]])
ax[0].plot([min_val, max_val], [min_val, max_val], color = 'r', linestyle = '--')
# Project onto orthogonal axis
joint_dat = np.vstack([pred_corr_list, psth_corr_list])
proj_dat = np.dot(joint_dat.T, np.array([1, -1]))
ax[1].hist(proj_dat, bins = 50)
ax[1].axvline(x = 0, color = 'r', linestyle = '--')
ax[1].set_xlabel('<-- PSTH Larger Corr | Pred Larger Corr -->')
pred_greater_bool = pred_corr_list > psth_corr_list
pred_greater_frac = np.round(np.mean(pred_greater_bool),2)
# Plot firing rates depending on pred_greater_bool
pred_greater_spikes = [test_spikes_list[i].values.mean() for i in \
        range(len(test_spikes_list)) if pred_greater_bool[i]]
pred_lesser_spikes = [test_spikes_list[i].values.mean() for i in \
        range(len(test_spikes_list)) if not pred_greater_bool[i]]
ax[2].hist(pred_greater_spikes, 
           bins = 50, alpha = 0.5, label = 'Pred > PSTH',
           density = True)
ax[2].hist(pred_lesser_spikes, 
           bins = 50, alpha = 0.5, label = 'Pred < PSTH',
           density = True)
ax[2].set_yscale('log')
ax[2].legend()
ax[2].set_xlabel('Mean firing rate')
ax[2].set_ylabel('Count')
fig.suptitle('Pred corr vs psth corr\n' + \
        f'Pred > PSTH : {pred_greater_frac}' + \
        '\n' + \
        f'Pred < PSTH : {np.round(1-pred_greater_frac,2)}')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'pred_corr_vs_psth_corr.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################
# Plot histogram of trial_avg_r2_list
plot_dat = np.array(trial_avg_r2_list)
keep_bool = plot_dat > -1
plot_dat = plot_dat[keep_bool]
r2_thresh = 0.1
low_dat = plot_dat[plot_dat < r2_thresh]
high_dat = plot_dat[plot_dat >= r2_thresh]
# Calculate mean firing rates for low and high r^2 dat
mean_firing_rate = [x.values.mean(axis=None)*1000 for x in test_spikes_list]
keep_spikes = [x for i, x in enumerate(mean_firing_rate) if keep_bool[i]]
low_spikes = [x for i, x in enumerate(keep_spikes) \
        if plot_dat[i] < r2_thresh]
high_spikes = [x for i, x in enumerate(keep_spikes) \
        if plot_dat[i] >= r2_thresh]

bin_vec  = np.linspace(-1, 1, 50)
fig, ax = plt.subplots(3,1, figsize = (5,10))
ax[0].axvline(r2_thresh, c = 'k', linestyle = '--')
ax[0].hist(low_dat, bins = bin_vec, alpha = 0.5, label = 'R^2 < 0.1')
ax[0].hist(high_dat, bins = bin_vec, alpha = 0.5, label = 'R^2 >= 0.1')
ax[0].legend()
# Remove x-ticks
ax[0].set_xticks([])
ymax = ax[0].get_ylim()[1]
# Write fraction of each side
ax[0].text(-0.5, ymax*0.5, 
        f'{100*len(low_dat)/len(plot_dat):0.2f}%')
ax[0].text(0.2, ymax*0.5,
        f'{100*len(high_dat)/len(plot_dat):0.2f}%')
# Text for thresh line
ax[0].text(r2_thresh, ymax*0.1, f'{r2_thresh:0.2f}', rotation = 90)
ax[0].set_ylabel('Count')
ax[0].set_title('Histogram of Trial-Averaged R^2')
ax[1].hist(circ_shuffle_avg_r2_list, bins = bin_vec, alpha = 0.5,
           color = 'k', label = 'Circ Shuffle')
ax[1].legend()
# Flip y-axis
ax[1].set_ylim(ax[1].get_ylim()[::-1])
ax[1].set_xlabel('R^2')
ax[1].set_ylabel('Count')
# Plot ECDF of low and high firing rates
low_outs = np.histogram(low_spikes, bins = 50)
high_outs = np.histogram(high_spikes, bins = 50)
ax[2].plot(low_outs[1][1:], np.cumsum(low_outs[0])/np.sum(low_outs[0]),
         label = 'R^2 < 0.1', linewidth = 3)
ax[2].plot(high_outs[1][1:], np.cumsum(high_outs[0])/np.sum(high_outs[0]),
         label = 'R^2 >= 0.1', linewidth = 3)
ax[2].legend()
ax[2].set_xlabel('Mean FR (Hz)')
ax[2].set_ylabel('Count')
ax[2].set_title('ECDF of Mean FR')
# # Plot mean firing rates for low and high r^2 on twin axis
# ax2 = ax.twinx()
# ax2.plot([-0.5, 0.5], [np.mean(low_spikes), np.mean(high_spikes)],
#          '-o', c = 'k', label = 'Mean FR')
# ax2.errorbar([-0.5, 0.5], [np.mean(low_spikes), np.mean(high_spikes)],
#              yerr = [np.std(low_spikes), np.std(high_spikes)],
#              c = 'k', linestyle = '', capsize = 5)
# ax2.set_ylabel('Mean FR (Hz)')
# plt.tight_layout()
plt.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('R^2 of Trial Avg\n' + \
        f'Fraction of data >-1 : {np.mean(keep_bool):0.2f}')
fig.savefig(os.path.join(plot_dir, 'r2_trial_avg_hist.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################
# Compare single-trial r^2 to psth r^2
fig, ax = plt.subplots(figsize = (5,5))
ax.scatter(psth_r2_list, pred_r2_list, alpha = 0.5)
min_val = np.min([np.min(psth_r2_list), np.min(pred_r2_list)])
max_val = np.max([np.max(psth_r2_list), np.max(pred_r2_list)])
ax.plot([min_val, max_val], [min_val, max_val], c = 'k', linestyle = '--')
ax.set_xlim([min_val, 0])
ax.set_ylim([min_val, 0])
# Set x and y to log scales
ax.set_xscale('symlog')
ax.set_yscale('symlog')
ax.set_xlabel('PSTH R^2')
ax.set_ylabel('Single Trial R^2')
fig.suptitle('Comparison of R^2')
fig.savefig(os.path.join(plot_dir, 'psth_r2_comparison.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################
# Plot all 3 on histogram
high_bool = np.array(trial_avg_r2_list) >= r2_thresh

fig, ax = plt.subplots(figsize = (5,5))
ax.hist(pred_r2_list[high_bool], 
        bins = 50, alpha = 0.5, label = 'Actual')
ax.hist(trial_shuffled_r2_list[high_bool], 
        bins = 50, alpha = 0.5, label = 'Trial Shuffled')
ax.hist(circ_shuffle_r2_list[high_bool], 
        bins = 50, alpha = 0.5, label = 'Circ Shuffled')
ax.legend()
# ax.set_xlim([0, 1])
fig.suptitle('R^2 of Actual vs Shuffled')
fig.savefig(os.path.join(plot_dir, 'r2_hist.png'))
plt.close(fig)


##############################
# Plot scatter of actual vs trial shuffled R2
fig, ax = plt.subplots(1, 2, figsize = (10,5), sharex=True)
ax[0].scatter(
            trial_shuffled_r2_list[high_bool], 
            pred_r2_list[high_bool], 
           s = 2, c = 'k', alpha = 0.3)
# Project data onto x=-y
joint_dat = np.vstack([trial_shuffled_r2_list[high_bool],
                       pred_r2_list[high_bool]])
proj_dat = np.dot(joint_dat.T, np.array([1, -1]))
med_val = np.median(proj_dat)
min_val = np.min([np.min(pred_r2_list), np.min(trial_shuffled_r2_list)])
max_val = np.max([np.max(pred_r2_list), np.max(trial_shuffled_r2_list)])
# Plot x=y
# ax.plot([min_val, max_val],[min_val, max_val], 
#          color = 'red', linestyle = '--', linewidth = 2)
ax[0].plot([-1, 1],[-1, 1], 
        color = 'red', linestyle = '--', linewidth = 2)
ax[0].set_xlim([-1, 1])
ax[0].set_ylim([-1, 1])
ax[0].set_ylabel('Actual R^2')
ax[0].set_xlabel('Trial Shuffled R^2')
ax[0].set_aspect('equal')
ax[1].hist(proj_dat, bins = 100, color = 'grey')
ax[1].hist(proj_dat, bins = 100, color = 'k', histtype = 'step')
ax[1].annotate(
        '',
        xy = (med_val, ax[1].get_ylim()[0]),
        xytext = (med_val, 0.02*ax[1].get_ylim()[1]),
        arrowprops = dict(
            facecolor = 'white', 
            edgecolor = 'black',
            shrink = 0.05
            ),
        label = 'Median',
        )
#ax.legend()
# Remove all axes except bottom
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].set_yticks([])
ax[1].axvline(0, color = 'red', linestyle = '--', linewidth = 2)
fig.suptitle('Single Trial R^2 of Actual vs Trial Shuffled')
fig.savefig(os.path.join(plot_dir, 'r2_scatter.png'),
            bbox_inches = 'tight')
plt.close(fig)


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
    kern_len = 200
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

