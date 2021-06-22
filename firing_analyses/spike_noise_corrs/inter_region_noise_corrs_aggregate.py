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
import argparse

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

parser = argparse.ArgumentParser(\
        description = 'Script to aggregate noise corr measures',
parser.add_argument('file_list',  
        help = 'dirs containing files to perform analysis on')
args = parser.parse_args()
file_list_path = args.file_list 

#file_list_path = '/media/bigdata/firing_space_plot/'\
#        'firing_analyses/lfp_power_xcorr_file_list.txt'

save_dir = os.path.join('/media/bigdata/firing_space_plot/firing_analyses',
                            'aggregate_analysis')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(file_list_path,'r') as this_file:
    file_list = this_file.read().splitlines()

# Get region_names  and region_units for all files
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
mean_mean_shuffle_count = [x.mean() for x in mean_shuffle_count]

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
