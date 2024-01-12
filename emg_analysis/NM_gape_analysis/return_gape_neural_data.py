"""
# Run PCA on whole trials
Analyze differences in neural and emg activity conditioned on 
strength of gaping activity per trial
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
import numpy as np
from tqdm import tqdm
import tables
import pylab as plt
from glob import glob
import pandas as pd
import pingouin as pg
from scipy.stats import zscore
from sklearn.cluster import KMeans
from collections import Counter

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

plot_dir = '/media/bigdata/firing_space_plot/NM_gape_analysis/plots'

file_list_path = '/media/fastdata/NM_sorted_data/h5_file_list.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
dir_names = [os.path.dirname(x) for x in file_list]
basename_list = [os.path.basename(x) for x in dir_names]

def time_box_conv(array, kern_width):
    """
    Convolution with 1D box kern along last dim
    """
    box_kern = np.ones((kern_width))/kern_width
    inds = list(np.ndindex(array.shape[:-1]))
    conv_array = np.empty(array.shape)
    for this_ind in tqdm(inds):
        conv_array[this_ind] = np.convolve(array[this_ind], box_kern, mode='same') 
    return conv_array

############################################################
# / ___| __ _ _ __   ___  |  _ \  __ _| |_ __ _ 
#| |  _ / _` | '_ \ / _ \ | | | |/ _` | __/ _` |
#| |_| | (_| | |_) |  __/ | |_| | (_| | || (_| |
# \____|\__,_| .__/ \___| |____/ \__,_|\__\__,_|
#            |_|                                
############################################################

gape_path = '/ancillary_analysis/gapes'
# laser_conds x taste x trials x time

emg_bsa_path = '/ancillary_analysis/emg_BSA_results'
# laser_conds x taste x trials x time x freq 

laser_dl_path = '/ancillary_analysis/laser_combination_d_l'
# condition_num x (duration + onset) 

# Temporal parameters
time_lims = [0,7000]
real_time = np.arange(-2000, 5000)
cut_real_time = real_time[time_lims[0]:time_lims[1]]
stim_t = 2000 - time_lims[0]

# Taste names and inds
taste_inds = np.array([0,3]) # 0:Sucrose, 3:quinine
taste_names = ['suc','quin']
taste_dict = dict(zip(taste_names, [0,3]))

def _return_gape_data():

    condition_list = []
    gapes_list = []
    for this_path in file_list:
        # Gape related info
        with tables.open_file(this_path,'r') as h5:
            gape_laser_conditions = h5.get_node(laser_dl_path)[:] 
            gapes_array = h5.get_node(gape_path)[:] 
        condition_list.append(gape_laser_conditions)
        gapes_list.append(gapes_array)

    # Some values are really low (e.g. e-88), clean those out
    gapes_list = [(x>0.5)*1 for x in gapes_list]

    wanted_condition = [np.where(x.sum(axis=-1)==0)[0][0] for x in condition_list]
    off_gape_array = [x[i] for x,i in zip(gapes_list, wanted_condition)]

    ############################################################
    # ____                                             
    #|  _ \ _ __ ___ _ __  _ __ ___   ___ ___  ___ ___ 
    #| |_) | '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|
    #|  __/| | |  __/ |_) | | | (_) | (_|  __/\__ \__ \
    #|_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/
    #               |_|                                
    ############################################################

    off_gape_array = [x[taste_inds] for x in off_gape_array]
    #off_gape_array = [x[...,time_lims[0]:time_lims[1]] for x in off_gape_array]

    #############################################################
    ## Remove sessions where
    #1) Sucrose and Quinine have similar responses
    #2) Sucrose response > Quinine Response
    #############################################################
    # Check for which recordings distances are sufficiently separate
    inds = [np.array(list(np.ndindex(x.shape))) for x in off_gape_array]
    gape_frames = [pd.DataFrame(dict(
                    session = num,
                    taste = this_inds[:,0],
                    trials = this_inds[:,1],
                    time = this_inds[:,2],
                    vals = this_dat.flatten())) \
            for num, (this_inds,this_dat) in enumerate(zip(inds, off_gape_array))]
    fin_gape_frame = pd.concat(gape_frames)
    fin_gape_frame['real_time'] = cut_real_time[fin_gape_frame['time']]

    # Only include data in 2000ms post-stim
    fin_gape_frame = fin_gape_frame.loc[fin_gape_frame.real_time.\
            isin(np.arange(2000))]

    # Downsample for ANOVA
    binsize = 500
    #bincount = int(np.diff(time_lims)[0]/binsize)
    bincount = int(
            (fin_gape_frame.time.max() + \
                    1 - fin_gape_frame.time.min())/binsize)
    fin_gape_frame['time_bins'] = pd.cut(fin_gape_frame['time'], bincount, 
           labels = np.arange(bincount))
    fin_gape_frame['vals'] += np.random.random(fin_gape_frame['vals'].shape)*0.01

    group_cols = ['session','taste','trials','time_bins']
    bin_gape_frame = fin_gape_frame.groupby(group_cols).mean().reset_index()
    bin_gape_frame.dropna(inplace=True)

    # Perform ANOVA
    # Perform separately for each session
    group_bin_gape = [x[1] for x in list(bin_gape_frame.groupby('session'))]
    anova_list = [pg.anova(data = this_dat,
                dv = 'vals', between = ['taste','time_bins']) \
                        for this_dat in group_bin_gape]
    pval_list = [x['p-unc'] for x in anova_list]
    taste_bool = np.stack(pval_list)[:,0]<0.05

    # Also check that quinine is HIGHER than sucrose
    quin_bool = [x.groupby('taste')['vals'].mean().diff()[1]>0 for x in group_bin_gape]

    ########################################
    ## Clustering in gape responses to quinine 
    ########################################

    quin_gape_array = [x[1] for x in off_gape_array]
    gape_t_lims = [2750,4500]
    #gape_t_lims = [x+stim_t for x in gape_t_lims]

    cut_gape_array = [x[...,gape_t_lims[0]:gape_t_lims[1]] for x in quin_gape_array]
    mean_gape_val = [x.mean(axis=-1) for x in cut_gape_array]

    # Simply dividing into equally sized groups doesn't make sense
    groups = 2
    group_labels = np.arange(groups)
    cluster_inds = [KMeans(n_clusters=groups).fit(dat.reshape(-1,1)).labels_ \
                        for dat in mean_gape_val]
    cluster_sort_inds = [np.argsort(x) for x in cluster_inds]
    cluster_counts = [list(Counter(sorted(x)).values()) for x in cluster_inds]

    # Only take sessions with >=3 sessions per cluster
    cluster_bool = [all(np.array(x)>=3) for x in cluster_counts]

    ########################################
    ## Finalize sessions 
    ########################################

    fin_bool = np.logical_and(
            np.logical_and(
                taste_bool, 
                quin_bool),
            cluster_bool)
    fin_bool_inds = np.where(fin_bool)[0]

    fin_bool_inds = np.where(fin_bool)[0]
    fin_bin_gape = bin_gape_frame[bin_gape_frame['session'].isin(fin_bool_inds)] 
    off_gape_array = [off_gape_array[i] for i in fin_bool_inds]
    fin_basenames = [basename_list[i] for i in fin_bool_inds]

    quin_gape_array = [x[1] for x in off_gape_array]
    suc_gape_array = [x[0] for x in off_gape_array]

    cut_gape_array = [cut_gape_array[i] for i in fin_bool_inds]
    mean_gape_val = [x.mean(axis=-1) for x in cut_gape_array]

    cluster_inds = [cluster_inds[i] for i in fin_bool_inds]
    cluster_sort_inds = [np.argsort(x) for x in cluster_inds]
    cluster_counts = [list(Counter(sorted(x)).values()) for x in cluster_inds]

    # Sort clusters from low to high response
    mean_clustered_vals = []
    for ind in range(len(cut_gape_array)):
        this_dat = cut_gape_array[ind]
        this_inds = cluster_inds[ind]
        clustered_trials = [this_dat[this_inds==x] for x in group_labels]
        mean_clustered_vals.append([np.round(x.mean(axis=None),2) for x in clustered_trials])

    sorted_cluster_order = [np.argsort(x) for x in mean_clustered_vals]
    cluster_map = [dict(zip(group_labels, x)) for x in sorted_cluster_order]
    fin_cluster_inds = [np.array([this_map[x] for x in this_inds]) \
            for this_map, this_inds in zip(cluster_map, cluster_inds)]

    return off_gape_array, fin_cluster_inds, fin_bool_inds 
    ## Relevant variables
    #1) cut_gape_array
    #3) fin_cluster_inds

############################################################
#| \ | | ___ _   _ _ __ __ _| | |  _ \  __ _| |_ __ _ 
#|  \| |/ _ \ | | | '__/ _` | | | | | |/ _` | __/ _` |
#| |\  |  __/ |_| | | | (_| | | | |_| | (_| | || (_| |
#|_| \_|\___|\__,_|_|  \__,_|_| |____/ \__,_|\__\__,_|
############################################################

# Given the above clusters, extract neural data and cluster similarly
off_gape_array, fin_cluster_inds, fin_bool_inds = _return_gape_data()

def return_gape_data():
    return off_gape_array, fin_cluster_inds, fin_bool_inds

fin_file_list = [file_list[i] for i in fin_bool_inds]
fin_basenames = [os.path.basename(x) for x in fin_file_list]
fin_dirnames = [os.path.dirname(x) for x in fin_file_list]

def return_names():
    return fin_file_list, fin_basenames

def return_neural_data(kern_width = 250):
    off_spikes_list = []
    off_firing_list = []
    for this_session_ind in range(len(fin_file_list)):
        this_dir = fin_dirnames[this_session_ind]
        this_basename = fin_basenames[this_session_ind]
        dat = ephys_data(this_dir)
        dat.get_spikes()
        dat.check_laser()
        if dat.laser_exists:
            dat.separate_laser_spikes()
            off_spikes = dat.off_spikes
        else:
            off_spikes = dat.spikes
        off_firing = time_box_conv(off_spikes, kern_width)
        firing_time_bins = time_box_conv(
                np.arange(off_spikes.shape[-1]), kern_width)
        off_spikes_list.append(off_spikes)
        off_firing_list.append(off_firing)
    return off_spikes_list, off_firing_list, firing_time_bins
## Relevant variables
#1) quin_off_spikes
#2) quin_firing
