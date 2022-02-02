"""
Compile correlation percentiles across sessions 
(and subaggregate on a per-animal basis)

Calculate correlations for all-to-all transitions

Refine results using
1) Trials where the model fits are more confident
2) Recordings with more discriminative and responsive neurons
3) Recordings with "stable" neurons
"""

import numpy as np
import re
import json
from glob import glob
import os
import pandas as pd
import pickle 
import sys
from scipy import stats
from scipy.stats import percentileofscore as p_of_s
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange 
import tables
import pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler as ss

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs/all_tastes')
from check_data import check_data 
import itertools as it

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def corr_percentile_single(a,b, shuffles = 1000):
    #shuffles = 1000
    #this_comp = comparison_list[1]
    #a,b = tau_array[0, this_comp[0]], tau_array[1,this_comp[1]]
    corr_val = stats.spearmanr(a,b)[0]
    shuffle_vals = [stats.spearmanr(a, 
                    np.random.permutation(b))[0] \
            for i in range(shuffles)]
    percentile_val = p_of_s(shuffle_vals, corr_val)
    return percentile_val, corr_val, shuffle_vals

def return_corr_percentile(tau_array, shuffles = 5000):
    #tau_array = tau_list[0]
    trans_list = np.arange(tau_array.shape[1])
    # **Note: The transitions in BLA and GC are not the same,
    #           therefore we must look at all permutations, not simply
    #           all combinations.
    comparison_list = list(it.product(trans_list, trans_list))
    #comparison_list = list(it.combinations_with_replacement(trans_list, 2))
    percentile_array = np.zeros((tau_array.shape[1], tau_array.shape[1]))
    corr_array = np.zeros((tau_array.shape[1], tau_array.shape[1]))
    shuffle_array = np.zeros((tau_array.shape[1], tau_array.shape[1], shuffles))
    for this_comp in tqdm(comparison_list):
        percentile_val, corr_val, shuffle_vals = \
                corr_percentile_single(tau_array[0, this_comp[0]],
                                        tau_array[1, this_comp[1]],
                                        shuffles = shuffles)
        percentile_array[this_comp] = percentile_val
        corr_array[this_comp] = corr_val
        shuffle_array[this_comp] = shuffle_vals
    return percentile_array, corr_array, shuffle_array

class params_from_path:
    def __init__(self, path):
        # Extract model params from basename
        self.path = path
        self.model_name = os.path.basename(self.path).split('.')[0]
        self.states = int(re.findall("\d+states",self.model_name)[0][:-6])
        self.time_lims = [int(x) for x in \
                re.findall("\d+_\d+time",self.model_name)[0][:-4].split('_')]
        self.bin_width = int(re.findall("\d+bin",self.model_name)[0][:-3])
        #self.fit_type = re.findall("type_.+",self.model_name)[0].split('_')[1]
        # Exctract data_dir from model_path
        self.data_dir = "/".join(self.path.split('/')[:-3])
        self.session_name = self.data_dir.split('/')[-1]
        self.animal_name = self.session_name.split('_')[0]
    def to_dict(self):
        return dict(zip(['states','time_lims','bin_width','session_name'],
            [self.states,self.time_lims,self.bin_width,self.session_name]))

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/inter_region'
wanted_names = ['rho_percentiles','mode_tau','rho_shuffles',
        'tau_corrs','tau_list'] 

# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/multi_region_frame.pkl'
inter_frame = pd.read_pickle(data_dir_pkl)
inter_frame['animal_name'] = [x.split('_')[0] for x in inter_frame['name']]

black_list = ['AM26_4Tastes_200828_112822', 'AM18','AM37', 'AM39']

# Pull out fit params for one file
data_dir = inter_frame.path.iloc[0]
this_info = check_data(data_dir)
this_info.run_all()
inter_region_paths = [path for  num,path in enumerate(this_info.pkl_file_paths) \
                if num in this_info.region_fit_inds]
state4_models = [path for path in inter_region_paths if '4state' in path]

# Check params for both fits add up
check_params_bool = params_from_path(state4_models[0]).to_dict() ==\
                    params_from_path(state4_models[1]).to_dict()
region_check = all([any([region_name in params_from_path(x).model_name \
                    for region_name in ['gc','bla']])\
                    for x in state4_models])
if not (check_params_bool and region_check):
    raise Exception('Fit params dont match up')
params = params_from_path(inter_region_paths[0]) 

session_num_list = []
dat_list = []
for num, data_dir in tqdm(enumerate(inter_frame.path)):

    if data_dir[-1] == '/':
        temp_name = data_dir[:-1]
    else:
        temp_name = data_dir
    basename = os.path.basename(temp_name)
    if (basename in black_list) or any([x in basename for x in black_list]):
        continue
    #num = 0
    #data_dir = inter_frame.path.iloc[num]
    dat = ephys_data(data_dir)
    #if inter_frame.name.iloc[num] in good_files_list:
    #    hf5_path = glob(os.path.join(data_dir,'*.h5'))[0]
        #hf5 = tables.open_file(hf5_path,'r')

    with tables.open_file(dat.hdf5_path,'r') as hf5:
        if save_path in hf5:
            this_dat = [hf5.get_node(save_path, this_name)[:] \
                    for this_name in wanted_names]
            dat_list.append(this_dat)
            session_num_list.append(num)

dat_list_zip = list(zip(*dat_list))
dat_list_zip = [np.stack(x) for x in dat_list_zip]
for this_var, this_dat in zip(wanted_names, dat_list_zip):
    globals()[this_var] = this_dat 

########################################
## All to all transition correlation
########################################

outs = parallelize(return_corr_percentile, tau_list)
percentile_array, corr_array, shuffle_array = list(zip(*outs))
percentile_array = np.stack(percentile_array)
corr_array = np.stack(corr_array)
shuffle_array = np.stack(shuffle_array)


# Find number of significant correlations
sig_perc = 90
sig_frac = np.mean(percentile_array >= sig_perc, axis=0)

# Find percentile of shuffles with respect to self
iters = list(np.ndindex(shuffle_array.shape[:-1]))
shuffle_perc_array = np.empty(shuffle_array.shape) 
for this_iter in iters:
    crit_val = stats.scoreatpercentile(shuffle_array[this_iter], sig_perc)
    shuffle_perc_array[this_iter] = shuffle_array[this_iter] >= crit_val

# Find distributions of random counts
random_sig_frac = np.mean(shuffle_perc_array, axis=0)

# 2 tailed, Bonferroni corrected p-value

#iters = list(np.ndindex(random_sig_frac.shape[:-1]))
#p_val_array = np.empty(random_sig_frac.shape[:-1]) 
#for this_iter in iters:
#
#comparisons = all_bin_percs.shape[1]
#abs_diff_perc = np.min(np.stack([100 - all_bin_percs, all_bin_percs]),axis=0)
#abs_diff_perc = ((abs_diff_perc*2)/100)*comparisons
#bonf_sig = abs_diff_perc <= 0.05

## Running into finite sample issues because of low dataset count
## i.e. because we only have 20 or so datasetes, when we calculate
## the percentile of actual significant counts to shuffle signficant
## counts, the support of the shuffle counts is small (e.g. 0-10
## out of 20), which creates problems for calculating the percentile
## to significant accuracy.
## Therefore, resample actual dataset and shuffle to optain larger
## number of possible counts
alpha = 0.05
resampled_counts = 10000
resample_inds = np.random.choice(
        np.arange(percentile_array.shape[0]), resampled_counts, replace=True)
resampled_percentile_array = percentile_array[resample_inds] 
resampled_sig_frac = np.mean(resampled_percentile_array >= sig_perc, axis=0)

resampled_shuffle_perc = shuffle_perc_array[resample_inds]
resampled_random_sig_frac = np.mean(resampled_shuffle_perc, axis=0)

iters = list(np.ndindex(random_sig_frac.shape[:-1]))
p_val_array = np.empty(random_sig_frac.shape[:-1]) 
for this_iter in iters:
    p_val_array[this_iter] = \
        p_of_s(resampled_random_sig_frac[this_iter], 
                resampled_sig_frac[this_iter])

#mask =  np.tri(p_val_array.shape[0], k=-1)
#p_val_array = np.ma.array(p_val_array, mask=mask)
#comparisons = (p_val_array.mask==False).sum()
comparisons = p_val_array.size
abs_diff_perc = np.min(np.stack([100 - p_val_array, p_val_array]),axis=0)
abs_diff_perc = ((abs_diff_perc*2)/100)
bonf_sig = abs_diff_perc <= (alpha/comparisons)
#bonf_sig = np.ma.array(bonf_sig, mask = mask)

def holm_bonferroni_test(p_vals, alpha = 0.05):
    sorted_vals = np.sort(p_vals)
    denoms = np.arange(len(sorted_vals),0, step = -1)
    corrected_alpha = alpha/denoms
    return sorted_vals < corrected_alpha

########################################
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
########################################

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/plots/multi_region'

########################################
#plot_sig_frac = np.ma.array(resampled_sig_frac, mask = mask).T
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cmap = plt.cm.get_cmap('viridis') # jet doesn't have white color
#cmap.set_bad('w') # default value is 'k'
#im = ax.imshow(plot_sig_frac, interpolation="nearest", cmap=cmap)
#iters = list(np.ndindex(bonf_sig.shape))
#for this_iter in iters:
#    if not bonf_sig.mask[this_iter]:
#        if bonf_sig.data[this_iter]:
#            text = ax.text(this_iter[0], this_iter[1], "*",
#                    ha="center", va="center", color="black",)
#            text.set_fontsize(20)
#tick_vals = np.arange(bonf_sig.shape[0])
#ax.set_xticks(tick_vals)
#ax.set_xticklabels(tick_vals)
#ax.set_yticks(tick_vals)
#ax.set_yticklabels(tick_vals)
#ax.set_xlabel("GC Transition")
#ax.set_ylabel("BLA Transition")
#for key,val in ax.spines.items():
#    val.set_visible(False)
#ax.set_xticks(tick_vals - 0.5, minor = True)
#ax.set_yticks(tick_vals - 0.5, minor = True)
#ax.grid(which="minor", color="w", linestyle='-', linewidth=5)
#ax.tick_params(which="minor", bottom=False, left=False)
#plt.colorbar(im)
#plt.suptitle('All-to-all transition correlation \n' + \
#        f"* = 2 Tailed Bonferroni Corrected p-value < alpha ({alpha})")
#fig.savefig(os.path.join(plot_dir, 'multi_transition_corr'))#, format='svg'))
#plt.close(fig)
##plt.show()

########################################

wanted_perc= 90
perc_mark_val = stats.scoreatpercentile(rho_shuffles[0,0],wanted_perc)

resample_num = 1000
bins = np.linspace(-0.5,0.5,50)
fig,ax = plt.subplots(tau_corrs.shape[1],1, sharex = True, sharey = True,
        figsize = (7,10))
for trans_num in range(tau_corrs.shape[1]):
    ax[trans_num].hist(rho_shuffles[:,trans_num,:1000].flatten(), 
            density = True, alpha = 0.5, bins = bins, label = 'Shuffle')
    ax[trans_num].hist(tau_corrs[:,trans_num],
            alpha = 0.5, bins = bins, label = 'Actual')
    ax[trans_num].hist(rho_shuffles[:,trans_num,:1000].flatten(), 
            density = True, alpha = 0.5, bins = bins, histtype = 'step') 
    ax[trans_num].hist(tau_corrs[:,trans_num],
            alpha = 0.5, bins = bins,  histtype = 'step')
    ax[trans_num].axvline(perc_mark_val, color = 'red', linestyle = 'dashed',
                        linewidth = 2, 
                        label = f'{wanted_perc}th Shuffle percentile')
    ax[trans_num].set_title(f'Transition {trans_num}')
ax[-1].set_xlabel('Rho Value')
ax[-1].legend()
            #np.random.choice(tau_corrs[:,trans_num], resample_num) +\
            #        (np.random.random(resample_num)-0.5)*0.05, 
plt.suptitle('Inter region Cor Hists')
#fig.savefig(os.path.join(plot_dir, 'agg_rho_comparison')) 
#plt.close(fig)
#plt.show()

########################################
## Create same plot with KDE
########################################
resample_num = 1000
bins = np.linspace(-0.5,0.5,50)
fig,ax = plt.subplots(tau_corrs.shape[1],1, sharex = True, sharey = True,
        figsize = (7,10))
for trans_num in range(tau_corrs.shape[1]):
    ax[trans_num].hist(rho_shuffles[:,trans_num,:1000].flatten(), 
            density = True, alpha = 0.5, bins = bins, label = 'Shuffle')
    count,_,_ = ax[trans_num].hist(tau_corrs[:,trans_num],
            density = True, alpha = 0.5, bins = bins, label = 'Actual')
    actual_dat = tau_corrs[:,trans_num]
    kernel = stats.gaussian_kde(actual_dat)
    kde_plot = kernel(bins)
    ax[trans_num].plot(bins, (kde_plot/np.max(kde_plot))*np.max(count))
    ax[trans_num].axvline(perc_mark_val, color = 'red', linestyle = 'dashed',
                        linewidth = 2, 
                        label = '{wanted_perc}th shuffle percentile')
    ax[trans_num].set_title(f'Transition {trans_num}')
ax[-1].set_xlabel('Rho Value')
ax[-1].legend()
            #np.random.choice(tau_corrs[:,trans_num], resample_num) +\
            #        (np.random.random(resample_num)-0.5)*0.05, 
plt.suptitle('Inter region Cor Hists')
fig.savefig(os.path.join(plot_dir, 'agg_rho_kde_comparison')) 
plt.close(fig)
#plt.show()

########################################
## Shuffled Rho histograms overlayed with actual data
########################################
bins = np.linspace(-0.5,0.5,50)
shuff_hist_list = np.stack([[np.histogram(x,bins = bins)[0] for x in y]\
                    for y in rho_shuffles.T])
mean_counts = np.mean(shuff_hist_list,axis=0)
std_counts = np.std(shuff_hist_list,axis=0)
#std_counts = np.percentile(shuff_hist_list,[25,75],axis=0)

resample_num = 1000
bins = np.linspace(-0.5,0.5,50)
fig,ax = plt.subplots(tau_corrs.shape[1],1, sharex = True, sharey = True,
        figsize = (7,10))
for trans_num in range(tau_corrs.shape[1]):
    ax[trans_num].errorbar(x = bins[:-1], y = mean_counts[trans_num],
                yerr = std_counts[trans_num], alpha = 0.5,
                label = 'Expected Shuffle Counts', linewidth = 2)
    #ax[trans_num].fill_between(x = bins[:-1],
    #        y1 = std_counts[0,trans_num], y2 = std_counts[1,trans_num],
    #        alpha = 0.5)
    ax[trans_num].hist(tau_corrs[:,trans_num],
            alpha = 0.5, bins = bins, label = 'Actual')
    ax[trans_num].axvline(perc_mark_val, color = 'red', linestyle = 'dashed',
                        linewidth = 2, 
                        label = f'{wanted_perc}th shuffle percentile')
    ax[trans_num].set_title(f'Transition {trans_num}')
ax[-1].set_xlabel('Rho Value')
ax[-1].legend()
            #np.random.choice(tau_corrs[:,trans_num], resample_num) +\
            #        (np.random.random(resample_num)-0.5)*0.05, 
plt.suptitle('Inter region Cor Hists')
fig.savefig(os.path.join(plot_dir, 'agg_rho_comparison2')) 
plt.close(fig)
#plt.show()

########################################
## Plot of percentiles with shuffled counts
########################################

hist_bins = np.linspace(0,100,11)

rho_shuffle_percs = np.array([[[p_of_s(x,z) for z in x] for x in y] \
                        for y in tqdm(rho_shuffles)])

# Hist counts you will get from random data
rho_shuff_perc_cut = rho_shuffle_percs[...,:1000]
random_hists_counts = np.array(
    [[np.histogram(x, hist_bins)[0] for x in y] for y in rho_shuff_perc_cut.T]) 
# Histogram of hist counts you will get from random data
# Essential distribution of counts from random data per bin
random_hists = np.array(
    [[np.histogram(x)[0] for x in y] for y in random_hists_counts.T])

mean_hist_counts = np.mean(random_hists_counts, axis = 0)
std_hist_counts = np.std(random_hists_counts, axis = 0)

mean_hist_frac = mean_hist_counts/tau_corrs.shape[0]
std_hist_frac = std_hist_counts/tau_corrs.shape[0]

########################################
## Shuffle # 1
########################################
# For each "set" of recordings and each transition
# Calculate the distribution of "number of recordings passing threshold"
# This will give us a distirbution to compare our observed frequencies with

#threshold = 90
#rho_shuffle_percs_sig = np.sum(rho_shuffle_percs >= threshold, axis=0)
#rho_percs_sig = np.sum(rho_percentiles >= threshold, axis=0)
#sig_count_perc = [p_of_s(x,y) for x,y in zip(rho_shuffle_percs_sig, rho_percs_sig)]

## Heck, we can do this for all bins and changepoints
rho_perc_counts = np.array(
        [np.histogram(x,hist_bins)[0] for x in rho_percentiles.T])
rho_perc_frac = rho_perc_counts/tau_corrs.shape[0]
all_bin_percs = np.array([[p_of_s(
                    random_hists_counts[:,ch_ind, bin_ind], 
                    rho_perc_counts[ch_ind, bin_ind])\
        for ch_ind in range(rho_perc_counts.shape[0])]\
        for bin_ind in range(rho_perc_counts.shape[1])] ).T

# 2 tailed, Bonferroni corrected p-value
comparisons = all_bin_percs.shape[1]
alpha = 0.05
abs_diff_perc = np.min(np.stack([100 - all_bin_percs, all_bin_percs]),axis=0)
abs_diff_perc = ((abs_diff_perc*2)/100)*comparisons
bonf_sig = abs_diff_perc <= 0.05

########################################
## Generate Plot 
########################################
# Calculate mean value of each changepoint
tau_scaled = (tau_list * params.bin_width) + params.time_lims[0]
mean_tau = np.vectorize(np.int)(np.mean(tau_scaled, axis = (0,1,3)))
std_tau = np.vectorize(np.int)(np.std(tau_scaled, axis = (0,1,3)))

cmap = plt.get_cmap('tab10')
bin_width = (np.unique(np.diff(hist_bins))[0])/2
x = hist_bins[:-1]+ bin_width
fig,ax = plt.subplots(rho_percentiles.shape[1], 
        sharey = True, sharex=True, figsize = (7,10))
hist_kwargs = dict(alpha = 1, bins = hist_bins)
step_hist_kwargs = dict(density = True, histtype = 'step', bins = hist_bins) 
for trans_num in range(rho_percentiles.shape[1]):
    #ax[trans_num].hist(rho_percentiles[:,trans_num], 
    #        **hist_kwargs, label = 'Actual', color = cmap(0)) 
    #ax[trans_num].hist(rho_percentiles[:,trans_num], 
    #        **step_hist_kwargs, color = cmap(0)) 
    ax[trans_num].bar(x, rho_perc_frac[trans_num], width = bin_width*2)
    ax[trans_num].errorbar(x = x,
            #y = mean_hist_counts[trans_num], 
            #yerr = std_hist_counts[trans_num],
            y = mean_hist_frac[trans_num], 
            yerr = std_hist_frac[trans_num],
            label = 'Shuffled', color = cmap(1), linewidth = 5)
    for num,(perc,sig) in \
            enumerate(zip(all_bin_percs[trans_num],bonf_sig[trans_num])):
        #ax[trans_num].text(x[num], np.max(rho_perc_counts) - 1, 
        ax[trans_num].text(x[num], np.max(rho_perc_frac)*0.9, 
                perc, ha = 'center', rotation = 45)
        if sig:
            #ax[trans_num].text(x[num], np.max(rho_perc_counts) - 1, 
            ax[trans_num].text(x[num], np.max(rho_perc_frac)*0.8, 
                '*', ha = 'center', fontweight = 'bold',
                fontsize = 'xx-large')
    #ax[trans_num].hist(rho_shuffle_percs[trans_num], 
    #        **hist_kwargs, label = 'Shuffle', color = cmap(1))
    #ax[trans_num].hist(rho_shuffle_percs[trans_num], 
    #        **step_hist_kwargs, color = cmap(1))
    #ax[trans_num,1].hist(mse_percentiles[:,trans_num])
    ax[trans_num].set_title(f'Transition {trans_num} :'\
            f'Time = {mean_tau[trans_num]}' + '+/-' + f'{std_tau[trans_num]}')
    ax[trans_num].set_ylabel('Frequency')
ax[-1].set_xlabel('Percentile')
ax[-1].legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
#ax[0,1].set_title('MSE Percentiles')
plt.suptitle('Aggregate transition hist, '\
        f'total count : {rho_percentiles.shape[0]}' + \
        "\n" + "Numbers = Percentiles of counts relative to shuffle" +\
        "\n" + "* = p<=0.05 (Bonferroni corrected 2 tailed)")
fig.savefig(os.path.join(plot_dir, 
    f'aggregate_percentiles_hist_bins{len(hist_bins)}'), 
        bbox_inches = 'tight')
plt.close(fig)
#plt.show()

# Plot count of recordings with percentiles above a threshold
# Compared with shuffles
# So basically above plot, only with last bin, concatenated for transitions
sig_hist_bins = [90,100]
random_hists_counts = np.array(
    [[np.histogram(x, sig_hist_bins)[0] for x in y] for y in rho_shuff_perc_cut.T]) 

mean_hist_counts = np.mean(random_hists_counts, axis = 0)
std_hist_counts = np.std(random_hists_counts, axis = 0)
mean_hist_frac = mean_hist_counts/tau_corrs.shape[0]
std_hist_frac = std_hist_counts/tau_corrs.shape[0]

rho_perc_counts = np.array(
        [np.histogram(x,sig_hist_bins)[0] for x in rho_percentiles.T])
rho_perc_frac = rho_perc_counts/tau_corrs.shape[0]
all_bin_percs = np.array([[p_of_s(
                    random_hists_counts[:,ch_ind, bin_ind], 
                    rho_perc_counts[ch_ind, bin_ind])\
        for ch_ind in range(rho_perc_counts.shape[0])]\
        for bin_ind in range(rho_perc_counts.shape[1])] ).T

# 2 tailed, Bonferroni corrected p-value
comparisons = all_bin_percs.shape[0]
alpha = 0.05
abs_diff_perc = np.min(np.stack([100 - all_bin_percs, all_bin_percs]),axis=0)
abs_diff_perc = ((abs_diff_perc*2)/100)*comparisons
bonf_sig = abs_diff_perc <= alpha 

x = np.arange(len(bonf_sig))
fig,ax = plt.subplots(figsize=(5,7))
#ax.bar(x, rho_perc_counts.flatten(), label = 'Actual', 
ax.bar(x, rho_perc_frac.flatten(), label = 'Actual', 
        color = cmap(0), edgecolor = None, alpha = 0.7, linewidth = 2)
        #color = cmap(0), edgecolor = cmap(0), alpha = 0.7, linewidth = 2)
#ax.errorbar(x, mean_hist_counts, std_hist_counts, 
ax.errorbar(x, mean_hist_frac, std_hist_frac.flatten(), 
        label = 'Shuffle', color = 'k', linewidth = 5, alpha = 0.7)
        #label = 'Shuffle', color = cmap(1), linewidth = 5, alpha = 0.7)
for num,(perc,sig) in enumerate(zip(abs_diff_perc,bonf_sig)):
    #ax.text(x[num], np.max(rho_perc_counts) - 1, 
    ax.text(x[num], np.max(rho_perc_frac)*0.9, 
            np.round(perc[0],3), ha = 'center', rotation = 45)
    if sig:
        #ax.text(x[num], np.max(rho_perc_counts) - 2, 
        ax.text(x[num], np.max(rho_perc_frac)*0.8, 
            '*', ha = 'center', fontweight = 'bold',
            fontsize = 'xx-large')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
ax.set_xlabel('Transition Number')
ax.set_ylabel('Frequency Number')
#ax[0,1].set_title('MSE Percentiles')
plt.suptitle(f'Aggregate transition hist : Bin = {sig_hist_bins}' + \
        "\n" + "Numbers = Bonf Corrected 2-tailed p-vals" + "\n" +\
        f'total count : {rho_percentiles.shape[0]}' +\
        f' ::: alpha = {alpha}')
fig.savefig(os.path.join(plot_dir, 
    f'sig_hist_bins_transition_max{np.diff(sig_hist_bins)[0]}'), 
    bbox_inches = 'tight')
plt.close(fig)
#plt.show()

