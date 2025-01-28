"""
Aggregate measures across recordings
"""
########################################
# Setup
########################################

########################################
# Import modules
########################################

import os
import sys
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
import pandas as pd
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare
import pylab as plt
import matplotlib as mpl
import argparse
import seaborn as sns
from pathlib import Path
from pprint import pprint as pp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz


############################################################
# Load Data
############################################################

#parser = argparse.ArgumentParser(\
#        description = 'Script to aggregate noise corr measures',
#parser.add_argument('file_list',  
#        help = 'dirs containing files to perform analysis on')
#args = parser.parse_args()
#file_list_path = args.file_list 

#file_list_path = '/media/bigdata/firing_space_plot/firing_analyses/spike_noise_corrs/spike_corr_files.txt'
# dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
expt_name = 'TRG_corrs'
dir_list_path = '/media/fastdata/Thomas_Data/all_data_dirs.txt'                    
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 

save_dir = os.path.join('/media/bigdata/firing_space_plot/firing_analyses/'
                'spike_noise_corrs/Plots/aggregate_analysis')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Check which h5 files
# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/spike_noise_corrs'
frame_name_list = [     
                'inter_region_frame',
                'shuffle_inter_region_frame',
                'intra_region_frame',
                'shuffle_intra_region_frame'
                   ]#,


present_bool_list = []
for this_file in tqdm(file_list):
    with tables.open_file(this_file, 'r') as hf5:
        if save_path not in hf5:
            present_bool_list.append(False)
        else:
            present_bool_list.append(True)

file_list = [x for x,y in zip(file_list,present_bool_list) if y]
dir_list = [x for x,y in zip(dir_list,present_bool_list) if y]
basenames = [os.path.basename(x) for x in dir_list]

#with open(file_list_path,'r') as this_file:
#    file_list = this_file.read().splitlines()
#
#file_list = [x for x in file_list if os.path.exists(x)]

overview_plot_dir = os.path.join(save_dir,'overview_plots')
if not os.path.exists(overview_plot_dir):
    os.makedirs(overview_plot_dir)


region_name_list = []
region_units_list = []
info_dict_list = []
firing_rate_list = []
for i, this_file in enumerate(tqdm(file_list)):
    dat = ephys_data(os.path.dirname(this_file))
    dat.get_info_dict()
    dat.get_region_units()
    dat.get_spikes()
    dat.firing_rate_params = dat.default_firing_params
    dat.get_firing_rates()
    region_name_list.append(dat.region_names)
    region_units_list.append([len(x) for x in dat.region_units])
    info_dict_list.append(dat.info_dict)
    firing_rate_list.append(dat.firing_list)
    fig, ax = vz.firing_overview(dat.all_normalized_firing)
    fig.suptitle(basenames[i])
    fig.savefig(os.path.join(overview_plot_dir,f'{basenames[i]}_firing_overview.png'),
                bbox_inches = 'tight')
    plt.close(fig)

# For each session, calc drift
all_pca_list = []
all_firing_long_list = []
for this_firing_rate in firing_rate_list:
    taste_pca_list = []
    taste_firing_long_list = []
    for this_taste in this_firing_rate:
        pca = PCA(n_components = 1, whiten = True)
        this_taste_long = this_taste.reshape(this_taste.shape[0],-1)
        this_taste_long = StandardScaler().fit_transform(this_taste_long.T).T
        pca_data = pca.fit_transform(this_taste_long)
        taste_firing_long_list.append(this_taste_long)
        taste_pca_list.append(pca_data)
    stack_pca = np.squeeze(np.stack(taste_pca_list))
    cat_firing_long = np.concatenate(taste_firing_long_list,axis = 0)
    all_pca_list.append(stack_pca)
    all_firing_long_list.append(cat_firing_long)

# Plot
fig, ax = plt.subplots(
        len(all_pca_list),2,
        figsize = (10, len(all_pca_list)),
        sharex =False, sharey =False
        )
for num,this_pca in enumerate(all_pca_list):
    ax[num,0].plot(this_pca.T)
    ax[num,0].set_title(basenames[num])
    ax[num,0].set_xlabel('Time')
    ax[num,0].set_ylabel('Drift')
    ax[num,1].imshow(all_firing_long_list[num], aspect = 'auto', interpolation = 'none')
    # ax[num,1].set_title(basenames[num])
    ax[num,1].set_xlabel('Time')
    ax[num,1].set_ylabel('Neuron')
# plt.tight_layout()
# plt.tight_layout()
fig.savefig(os.path.join(save_dir,f'{expt_name}_pca_drift.svg'),
            format = 'svg', bbox_inches = 'tight')
plt.close(fig)

sorted_region_name_list = np.array([np.array(names)[np.argsort(counts)[::-1]] \
        for names,counts in zip(region_name_list,region_units_list)])

# Load frames
save_path = '/ancillary_analysis/spike_noise_corrs'
frame_name_list = ['inter_region_frame', 'shuffle_inter_region_frame']
frame_list = [[pd.read_hdf(h5_path,
                           os.path.join(save_path,this_frame_name)) \
                                   for this_frame_name in frame_name_list] \
                                   for h5_path in tqdm(file_list)]

alpha = 0.05
# Add repeat numbers to the shuffle frame
actual_frame_list = []
shuffle_frame_list = []

for this_actual_frame,this_shuffle_frame in tqdm(frame_list):

    this_actual_frame['repeat_num'] = 0
    actual_frame_list.append(this_actual_frame)

    this_grouped_shuffle = list(this_shuffle_frame.groupby(\
            ['pair_ind','taste']))
    for num,this_frame in this_grouped_shuffle:
        this_frame['repeat_num'] = np.arange(this_frame.shape[0])

    this_shuffle_frame = \
            pd.concat([x[1] for x in this_grouped_shuffle])
    shuffle_frame_list.append(this_shuffle_frame)

# Convert alphas to boolean
for x,y in zip(actual_frame_list, shuffle_frame_list):
    x['sig_bool'] = x['p_vals'] < alpha
    y['sig_bool'] = y['p_vals'] < alpha

mean_actual_count = [x.groupby('taste')['sig_bool'].mean() for x in actual_frame_list]
mean_shuffle_count = [x.groupby(['taste','repeat_num'])['sig_bool'].mean() \
        for x in shuffle_frame_list]
# Add tiny jitter to smoothen then shuffle distributions
mean_shuffle_count = [x+((np.random.random(x.shape)-0.5)*0.001) \
        for x in mean_shuffle_count]

mean_actual_count = [x.reset_index(drop=False) for x in mean_actual_count]
mean_shuffle_count = [x.reset_index(drop=False) for x in mean_shuffle_count]

# Calculate critical value relative to shuffle distribution
# mean_shuffle_crit = [np.percentile(x, [5,95]) for x in mean_shuffle_count]
# mean_mean_shuffle_count = [x.mean() for x in mean_shuffle_count]
# std_mean_shuffle_count = [x.std() for x in mean_shuffle_count]

mean_shuffle_crit = [x.groupby('taste')['sig_bool'].agg(
    [
        lambda x: np.percentile(x,5, method = 'inverted_cdf'),
        lambda x: np.percentile(x,95, method = 'inverted_cdf')
        ]
    ) for x in mean_shuffle_count]
mean_shuffle_crit = [x.reset_index(drop=False) for x in mean_shuffle_crit]
# mean_mean_shuffle_count = [x.groupby('taste')['sig_bool'].mean() for x in mean_shuffle_count]
# mean_mean_shuffle_count = [x.reset_index(drop=False) for x in mean_mean_shuffle_count]
# std_mean_shuffle_count = [x.groupby('taste')['sig_bool'].std() for x in mean_shuffle_count]

mean_actual_count_list = []
for i in range(len(mean_actual_count)):
    this_df = mean_actual_count[i]
    this_df['session'] = i
    mean_actual_count_list.append(this_df)

mean_actual_count_df = pd.concat(mean_actual_count_list)

mean_shuffle_crit_list = []
for i in range(len(mean_shuffle_crit)):
    this_df = mean_shuffle_crit[i]
    this_df['session'] = i
    mean_shuffle_crit_list.append(this_df)

mean_shuffle_crit_df = pd.concat(mean_shuffle_crit_list)
mean_shuffle_crit_df.rename(
        columns = {'<lambda_0>':'low_crit',
                   '<lambda_1>':'high_crit'},
        inplace = True)
# If low_crit < 0, set to 0
mean_shuffle_crit_df['low_crit'] = np.maximum(mean_shuffle_crit_df['low_crit'],0)

# Merge dataframes
fin_count_df = mean_actual_count_df.merge(
        mean_shuffle_crit_df,
        on = ['taste','session'],
        how = 'left')

# Add taste names
taste_names = [
        x['taste_params']['tastes'] for x in info_dict_list
        ]
df_list = []
for session_ind in fin_count_df['session'].unique():
    this_df = fin_count_df[fin_count_df['session'] == session_ind]
    this_taste_names = taste_names[session_ind]
    taste_map = {num: name for num,name in enumerate(this_taste_names)}
    this_df['taste_name'] = this_df['taste'].map(taste_map) 
    df_list.append(this_df)
fin_count_df = pd.concat(df_list)

taste_rename_map = dict(
        MV = 'MV',
        S = 'S',
        SMV = 'SMV',
        EB = 'EB',
        CA = 'CA',
        CAEB = 'CAEB',
        SEB = 'SEB',
        CAMV = 'CAMV',
        Suc = 'S',
        W = 'W',
        Carv = 'Carv',
        CarvS = 'CarvS',
        )
fin_count_df['taste_name'] = fin_count_df['taste_name'].map(taste_rename_map)

taste_name_full_map = dict(
        MV = 'methyl valerate',
        S = 'sucrose',
        SMV = 'sucrose + methyl valerate',
        EB = 'ethyl butyrate',
        CA = 'citric acid',
        CAEB = 'citric acid + ethyl butyrate',
        SEB = 'sucrose + ethyl butyrate',
        CAMV = 'citric acid + methyl valerate',
        W = 'water',
        Carv = 'carvone',
        CarvS = 'carvone + sucrose',
        )

stim_type_map = dict(
        MV = 'Odor',
        S = 'Taste',
        SMV = 'Flavor',
        EB = 'Odor',
        CA = 'Taste',
        CAEB = 'Flavor',
        SEB = 'Flavor',
        CAMV = 'Flavor',
        W = 'Water',
        Carv = 'Odor',
        CarvS = 'Flavor',
        )

mixture_children = dict(
        SMV = ['S','MV'],
        CAEB = ['CA','EB'],
        SEB = ['S','EB'],
        CAMV = ['CA','MV'],
        CarvS = ['Carv','S'],
        )

fin_count_df['full_taste_name'] = fin_count_df['taste_name'].map(taste_name_full_map)
fin_count_df['stim_type'] = fin_count_df['taste_name'].map(stim_type_map)
fin_count_df['session_name'] = [basenames[x] for x in fin_count_df['session']]

# Plot
fig,ax = plt.subplots(
        1, len(np.unique(fin_count_df['session'])),
        figsize = (3*len(np.unique(fin_count_df['session'])),3),
        sharey = False, sharex=False)
for num,this_session in enumerate(np.unique(fin_count_df['session'])):
    this_df = fin_count_df[fin_count_df['session'] == this_session]
    this_ax = ax[num]
    this_ax.scatter(this_df['taste_name'],this_df['sig_bool'],label = 'Actual',
                    c = 'r', alpha = 0.7)
    err_mid = (this_df['low_crit'] + this_df['high_crit'])/2
    low_err = err_mid - this_df['low_crit']
    high_err = this_df['high_crit'] - err_mid
    # this_ax.errorbar(this_df['taste_name'],
    #                  err_mid.values,
    #                  low_err.values,
    #                  high_err.values,
    #                  )
    for taste_num in range(this_df.shape[0]):
        this_ax.plot(
                [this_df['taste_name'].values[taste_num]]*2, 
                [this_df['low_crit'].values[taste_num],
                     this_df['high_crit'].values[taste_num]],
                color = 'k', alpha = 0.7)
    this_ax.set_title(
            this_df.session_name.values[0], 
            rotation = 45,
            ha = 'left')
    this_ax.set_xlabel('Taste')
    # Rotate x labels
    for tick in this_ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_fontsize(10)
ax[0].set_ylabel('Fraction of significant correlations')
fig.suptitle('Significant Correlations by Taste')
plt.tight_layout()
# plt.show()
fig.savefig(os.path.join(save_dir,f'{expt_name}_sig_corrs_by_session.svg'),
            format = 'svg', bbox_inches = 'tight')
plt.close(fig)

# Make plots of mixture and children
# First get all mixture and children grouped
session_num_list = []
all_mix_list = []
all_child_list = []
for num,this_session in enumerate(np.unique(fin_count_df['session'])):
    this_df = fin_count_df[fin_count_df['session'] == this_session]
    # Find all mixtures
    mixtures = this_df[this_df['taste_name'].isin(mixture_children.keys())]
    # Find all children
    children_list = []
    for this_mixture in mixtures['taste_name']:
        this_children = mixture_children[this_mixture]
        this_child_df = this_df[this_df['taste_name'].isin(this_children)]
        children_list.append(this_child_df)
    mix_names = mixtures['taste_name']
    child_names = [x.taste_name for x in children_list]
    mix_vals = mixtures['sig_bool']
    child_vals = [x.sig_bool.values for x in children_list]
    mix_list = [{x:y} for x,y in zip(mix_names,mix_vals)]
    child_list = [dict(zip(this_names,this_vals)) for this_names,this_vals in zip(child_names,child_vals)]
    all_mix_list.append(mix_list)
    all_child_list.append(child_list)
    session_num_list.append([num]*len(mix_list))

# Flatten lists
all_session_nums = [item for sublist in session_num_list for item in sublist]
all_mix_list = [item for sublist in all_mix_list for item in sublist]
all_child_list = [item for sublist in all_child_list for item in sublist]

norm_child_list = []
for i in range(len(all_child_list)):
    this_mix = all_mix_list[i]
    this_child = all_child_list[i]
    this_mix_val = list(this_mix.values())[0]
    this_norm_child = {x: y / this_mix_val for x,y in this_child.items()}
    norm_child_list.append(this_norm_child)

# Get all normalized children
all_child_vals = [list(x.values()) for x in norm_child_list]
# Flatten
all_child_vals = [item for sublist in all_child_vals for item in sublist]
med_norm_child = np.median(all_child_vals)
log_med_norm_child = np.median(np.log(all_child_vals)) 

# Plot all children values
fig,ax = plt.subplots(figsize = (5,5))
for this_dict in norm_child_list:
    for key,val in this_dict.items():
        ax.scatter(key,val, color = 'black', alpha = 0.5)
ax.axhline(1, color = 'k', linestyle = '--')
ax.set_xlabel('Taste')
ax.set_ylabel('Normalized Fraction of significant correlations')
fig.suptitle('Normalized Children of Mixtures')
plt.tight_layout()
fig.savefig(os.path.join(save_dir,f'{expt_name}_norm_child_scatter.svg'),
            format = 'svg', bbox_inches = 'tight')
plt.close(fig)

# Bootstrap to get confidence intervals
n_boot = 1000
boot_vals = np.zeros(n_boot)
for i in range(n_boot):
    boot_vals[i] = np.median(np.random.choice(all_child_vals,len(all_child_vals)))
median_ci = np.percentile(boot_vals,[2.5,97.5])

fig,ax = plt.subplots(2,1,figsize = (5,5))
ax[0].hist(all_child_vals, bins = 30, alpha = 0.5, color = 'black',
           label = 'All Children')
ax[0].axvline(med_norm_child, color = 'red', linestyle = '--',
              label = 'Median')
ax[0].axvspan(median_ci[0],median_ci[1], color = 'red', alpha = 0.3,
              label = '95% CI of Median')
ax[0].legend()
ax[0].set_xlabel("""
 <- Mixture higher | Mixture lower ->
 Normalized Fraction of significant correlations
 """)
ax[0].set_ylabel('Count')
ax[1].hist(np.log(all_child_vals), bins = 30, alpha = 0.5, color = 'black')
ax[1].axvline(log_med_norm_child, color = 'red', linestyle = '--')
ax[1].set_xlabel("""
 <- Mixture higher | Mixture lower ->
Log Normalized Fraction of significant correlations')
"""
ax[1].set_ylabel('Count')
fig.suptitle('Normalized Children of Mixtures')
plt.tight_layout()
fig.savefig(os.path.join(save_dir,f'{expt_name}_norm_child_hist.svg'),
            format = 'svg', bbox_inches = 'tight')
plt.close(fig)

# Plot
fig,ax = plt.subplots(
        1,
        len(norm_child_list),
        figsize = (len(norm_child_list),3),
        sharey = True, sharex=False)
for num,this_dict in enumerate(norm_child_list):
    this_ax = ax[num]
    this_ax.scatter(this_dict.keys(),this_dict.values())
    this_ax.axhline(1, color = 'k', linestyle = '--')
    this_ax.set_title(
            f'Session {all_session_nums[num]}',
            rotation = 45,
            ha = 'left')
    this_ax.set_xlabel('Taste')
    # Rotate x labels
    for tick in this_ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_fontsize(10)
    this_session_num = all_session_nums[num]
    this_ax.set_title(
            basenames[this_session_num],
            rotation = 45,
            ha = 'left')
ax[0].set_ylabel('Normalized Fraction of significant correlations')
fig.suptitle('Normalized Children of Mixtures')
plt.tight_layout()
fig.savefig(os.path.join(save_dir,f'{expt_name}_sig_corrs_children_by_session.svg'),
            bbox_inches = 'tight')
plt.close(fig)

## Plot scatterplot of shuffle vs actual with error bars for shuffles
#max_val = np.max(mean_actual_count + mean_mean_shuffle_count)
#x = y = np.linspace(0,max_val,10)
#fig,ax = plt.subplots(figsize=(5,5))
#ax.errorbar(x = mean_mean_shuffle_count,
             #            y = mean_actual_count,
             #            xerr = std_mean_shuffle_count,
             #            marker = 'v', fmt = 'o')
#plt.plot(x,y, color = 'red', linestyle = '--')
#ax.set_xlabel('Shuffle')
#ax.set_ylabel('Actual')
#plt.show()

# Since mean shuffles are very stably around 0.05, there's no 
# point in creating a 2D plot
x = np.arange(len(mean_actual_count))
sorted_inds = np.argsort(mean_actual_count)
sorted_actual = np.array(mean_actual_count)[sorted_inds]
sorted_shuffle_mean = np.array(mean_mean_shuffle_count)[sorted_inds]
sorted_shuffle_std = np.array(std_mean_shuffle_count)[sorted_inds]
sorted_shuffle_crit = np.array(mean_shuffle_crit)[sorted_inds]
sorted_shuffle_error = np.abs(sorted_shuffle_crit - sorted_shuffle_mean[:,None])
# Don't plot lower error
#sorted_shuffle_error[:,0] = 0
sig_bool = sorted_actual >= sorted_shuffle_crit[:,1]

##############################
# Inter-region noise corrs
##############################
frame_name_list = ['intra_region_frame']
intra_frame_list = [[pd.read_hdf(h5_path,
                                 os.path.join(save_path,this_frame_name)) \
                                         for this_frame_name in frame_name_list] \
                                         for h5_path in tqdm(file_list)]
intra_frame_list = [x[0] for x in intra_frame_list]
for this_frame in intra_frame_list:
    this_frame['sig_bool'] = this_frame['p_vals'] < alpha
mean_sig_list = [x.groupby('label').mean('sig_bool')['sig_bool']\
        for x in intra_frame_list]
mean_sig_frame = pd.concat(mean_sig_list)
mean_sig_frame = pd.DataFrame(mean_sig_frame)
mean_sig_frame['label'] = mean_sig_frame.index
mean_sig_frame.reset_index(drop=True, inplace=True)

intra_stats = dict(mean = mean_sig_frame.mean().values[0],
                   sd = mean_sig_frame.std().values[0],
                   sem = mean_sig_frame.std().values[0]/ np.sqrt(mean_sig_frame.shape[0]))

fig,ax = plt.subplots(figsize = (5,5))
sns.swarmplot(data = mean_sig_frame, x = 'label', y = 'sig_bool',
              s = 10, alpha = 0.7, ax = ax)
plt.ylim(0,0.3)
plt.xticks([0,1],labels=['BLA','GC'])
plt.xlabel('Region Name')
plt.ylabel('Fraction of Significant Correlations')
plt.show()

fig.savefig(os.path.join(save_dir,'intra_region_sig_corrs'),dpi=300,
            format = 'svg')
plt.close(fig)


##############################
# Plots
##############################

# Set general font size
font_size = 15
plt.rcParams['font.size'] = str(font_size)
fig,ax = plt.subplots(figsize=(7,7))
cmap = mpl.colors.ListedColormap(['black', 'red'])
ax.errorbar(x = x, y = np.repeat(alpha, len(x)), 
            #yerr = sorted_shuffle_std,
            yerr = sorted_shuffle_error.T,
            alpha = 0.5, lw = 2, fmt = ' ', 
            label = 'Expected from chance (5-95th percentile)',
            color = 'k',
            zorder = -1,
            capsize = 5)
ax.scatter(x = x, y = sorted_actual, s = 30, 
           #label = ['False','True'],
           c = sig_bool *1,
           cmap = cmap, zorder=  1)
#ax.scatter(x=x, y = np.repeat(alpha, len(x)),
            #        color = 'k', marker = '_', s = 100)
#label = f'Alpha : {alpha}')
#ax.axhline(alpha, color = 'k', linestyle = '--', lw = 3,
            #        label = f'Alpha : {alpha}')
#ax.set_xlabel('Shuffle')
ax.axhline(intra_stats['mean'], color = 'red', 
           label = 'Mean Intra-region Fraction +/- SEM')
ax.axhline(intra_stats['mean'] + intra_stats['sem'], color = 'red', linestyle = '--')
ax.axhline(intra_stats['mean'] - intra_stats['sem'], color = 'red', linestyle = '--')
ax.set_ylabel('Fraction of Significant Correlations')
ax.set_xticks(np.arange(len(x),step=2),[])
# Set tick font size
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(font_size)
plt.xlabel('Recording Session')
plt.legend(loc='lower left')
#plt.show()
rig.savefig(os.path.join(save_dir,'actual_sig_corrs_with_errorbars_w_intra'),
            dpi=300, format = 'svg')
plt.close(fig)

# Plot pair plots of shuffle and actual
fig,ax = plt.subplots(figsize=(5,5))
for this_dat in zip(mean_mean_shuffle_count,mean_actual_count):
    #ax.scatter(['Shuffle','Actual'],this_dat, 
                #        color='black', s = 80, facecolors = 'white')
    ax.plot(['Shuffle','Actual'],
            this_dat, '-o', color = 'grey', mfc = 'white', ms = 10)
ax.axhline(alpha, linestyle = '--', color = 'black', linewidth = 3)
ax.set_xlabel('Comparison Type')
ax.set_ylabel('Fraction of significant correlations')
plt.tight_layout()
fig.savefig(os.path.join(save_dir,'actual_vs_shuffle_sig_interactions'),dpi=300)
plt.close(fig)
#plt.show()

#for this_dat,this_name in zip(frame_list,frame_name_list):
#    globals()[this_name] = this_dat

# Create function to pull out significant pairs
def gen_sig_mat(pd_frame, index_label):
    label_cond = pd_frame['label'] == index_label
    sig_cond = pd_frame['p_vals'] <= alpha
    sig_frame = pd_frame[label_cond & sig_cond]
    sig_pairs = sig_frame['pair']

    sig_hist_array = np.zeros([x+1 for x in pd_frame[label_cond]['pair'].max()])
    for this_pair in sig_pairs:
        sig_hist_array[this_pair] += 1
    return sig_hist_array, sig_frame

inter_sig_hist_list = [gen_sig_mat(this_frame.dropna(),'inter_region')[0] \
        for this_frame in frame_list]

# Get fraction of significant interactions
mean_interaction_array = np.array([[np.mean(this_array,axis=1-num) \
        for num in range(this_array.ndim) ]\
        for this_array in inter_sig_hist_list])

unique_region_names = np.unique(sorted_region_name_list)
region_interaction_list = [mean_interaction_array\
        [np.where(sorted_region_name_list == this_region)] \
        for this_region in unique_region_names]

# Flatten out sublists
region_interaction_list = [[item for sublist in this_region for item in sublist] \
        for this_region in region_interaction_list]

from scipy.stats import gaussian_kde

cmap = plt.get_cmap("tab10")
fig,ax = plt.subplots(2,1,sharex=True, sharey=True)
for num,(this_dat,this_name) in \
        enumerate(zip(region_interaction_list, unique_region_names)):
            # Dividing by 4 to account for the TOTAL number of interactions POSSIBLE
    fin_dat = np.array(this_dat)/4
    vals,bins,patches = \
            ax[num].hist(fin_dat, bins = 20, alpha = 0.5,color = cmap(0))
    ax[num].hist(fin_dat, bins = 20, histtype='step',color = cmap(0))
    x = np.linspace(bins[0],bins[-1],100)
    kde_vals = gaussian_kde(fin_dat, bw_method = 0.2)(x)
    ax[num].plot(x,kde_vals/np.max(kde_vals)*np.max(vals),
                 color = 'red', linewidth = 2, alpha = 0.7)
    ax[num].set_title(this_name + f' : n  = {len(this_dat)}')
plt.suptitle('Significant interactions per region')
fig.savefig(os.path.join(save_dir,'sig_interaction_hists'),dpi=300)
plt.close(fig)
#plt.show()
# Load frames

