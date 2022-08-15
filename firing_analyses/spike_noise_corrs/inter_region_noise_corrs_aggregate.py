"""
Aggregate measures across recordings
"""
########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
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

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

#parser = argparse.ArgumentParser(\
#        description = 'Script to aggregate noise corr measures',
#parser.add_argument('file_list',  
#        help = 'dirs containing files to perform analysis on')
#args = parser.parse_args()
#file_list_path = args.file_list 

#file_list_path = '/media/bigdata/firing_space_plot/firing_analyses/spike_noise_corrs/spike_corr_files.txt'
dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 

save_dir = os.path.join('/media/bigdata/firing_space_plot/firing_analyses/'
                'spike_noise_corrs/Plots/aggregate_analysis')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#with open(file_list_path,'r') as this_file:
#    file_list = this_file.read().splitlines()
#
#file_list = [x for x in file_list if os.path.exists(x)]


region_name_list = []
region_units_list = []
for this_file in tqdm(file_list):
    dat = ephys_data(os.path.dirname(this_file))
    dat.get_region_units()
    region_name_list.append(dat.region_names)
    region_units_list.append([len(x) for x in dat.region_units])

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

mean_actual_count = [x['sig_bool'].mean() for x in actual_frame_list]
mean_shuffle_count = [x.groupby('repeat_num').mean()['sig_bool'] \
                    for x in shuffle_frame_list]
# Add tiny jitter to smoothen then shuffle distributions
mean_shuffle_count = [x+((np.random.random(x.shape)-0.5)*0.001) \
                        for x in mean_shuffle_count]

# Calculate critical value relative to shuffle distribution
mean_shuffle_crit = [np.percentile(x, [5,95]) for x in mean_shuffle_count]
mean_mean_shuffle_count = [x.mean() for x in mean_shuffle_count]
std_mean_shuffle_count = [x.std() for x in mean_shuffle_count]

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

# ___       _               ____            _             
#|_ _|_ __ | |_ _ __ __ _  |  _ \ ___  __ _(_) ___  _ __  
# | || '_ \| __| '__/ _` | | |_) / _ \/ _` | |/ _ \| '_ \ 
# | || | | | |_| | | (_| | |  _ <  __/ (_| | | (_) | | | |
#|___|_| |_|\__|_|  \__,_| |_| \_\___|\__, |_|\___/|_| |_|
#                                     |___/               
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


# ___       _                 ____            _             
#|_ _|_ __ | |_ ___ _ __     |  _ \ ___  __ _(_) ___  _ __  
# | || '_ \| __/ _ \ '__|____| |_) / _ \/ _` | |/ _ \| '_ \ 
# | || | | | ||  __/ | |_____|  _ <  __/ (_| | | (_) | | | |
#|___|_| |_|\__\___|_|       |_| \_\___|\__, |_|\___/|_| |_|
#                                       |___/               

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

