"""
For each session, perform ANOVA over time-bins for all tastes
1) Save results of ANOVA
2) Save indices and spike trains of "dynamic" neurons
"""

########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import os
import sys
import pymc3 as pm
import theano.tensor as tt
import json
import pandas as pd
from tqdm import tqdm
import pylab as plt
import tables
import shutil

import numpy as np
import pickle
import argparse
import pingouin as pg
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc')
from ephys_data import ephys_data
import visualize


def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

parser = argparse.ArgumentParser(description = 'Script to extract dynamic neurons')
parser.add_argument('dir_name',  help = 'Directory containing data files')
args = parser.parse_args()
data_dir = args.dir_name 

#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'

dat = ephys_data(data_dir)
dat.get_spikes()

# Bin firing into larger bins for anova
bin_width = 500
trial_bin_width = 10
time_lims = [1500,4000]
stim_t = 2000
plot_time_lims = np.array(time_lims) - stim_t

#bin_width_in_inds = bin_width // step_size
#fin_bin_count = len(bin_inds)//bin_width_in_inds
fin_bin_count = np.abs(np.diff(time_lims))[0]//bin_width

spike_array = np.array(dat.spikes)[...,time_lims[0]:time_lims[1]]
binned_spikes = np.sum(\
        np.reshape(spike_array,(*spike_array.shape[:-1],fin_bin_count,-1)),
        axis = -1)
binned_spikes = np.reshape(binned_spikes, 
        (binned_spikes.shape[0], -1, trial_bin_width, *binned_spikes.shape[2:]))

inds = np.array(list(np.ndindex(binned_spikes.shape))).T
firing_frame = pd.DataFrame({
    'taste' : inds[0],
    'trial_bins' : inds[1],
    'trials' : inds[2],
    'neurons' : inds[3],
    'bins' : inds[4],
    'count' : binned_spikes.flatten()})

neuron_group_list = firing_frame.groupby(['neurons'])
anova_list = [this_dat[1].anova( dv = 'count',
                        between = ['taste','trial_bins','bins']) \
                for this_dat in tqdm(neuron_group_list)]

anova_p_list = [x[['Source','p-unc']][:3] for x in anova_list]
time_bin_thresh = 0.05 
trial_bin_thresh = 1e-3
anova_trial_bin_p = [x['p-unc'][1] for x in anova_p_list]
anova_time_bin_p = [x['p-unc'][2] for x in anova_p_list]

# If there WAS NOT a significant effect of trials
trial_bin_bool = np.array([x>trial_bin_thresh for x in anova_trial_bin_p])
# If there WAS a significant effect of time
time_bin_bool = np.array([x<time_bin_thresh for x in anova_time_bin_p])

# Remove neurons with firing rate below threshold post-stim for all tastes
firing_rate_thresh = 2 #Hz
stim_spike_array = np.array(dat.spikes)[...,stim_t:time_lims[1]]
mean_firing = np.mean(stim_spike_array,axis=(1,-1)) * 1000

# If firing rate WAS high enough for at least one tastant
mean_firing_bool = mean_firing > firing_rate_thresh
all_firing_bool = np.sum(mean_firing_bool,axis=0) > 0 

# Neurons should show within trial dynamics but not across-trial
# variation
fin_bin_bool = np.array([trial and time and firing \
        for trial, time, firing \
        in zip(trial_bin_bool, time_bin_bool, all_firing_bool)])

##############################
#/ ___|  __ ___   _____ 
#\___ \ / _` \ \ / / _ \
# ___) | (_| |\ V /  __/
#|____/ \__,_| \_/ \___|
##############################                       
# Save good_nrn bool list and spike trains to HDF5
with tables.open_file(dat.hdf5_path,'r+') as h5:
    remove_node('/selected_changepoint_nrns', h5)
    h5.create_array('/','selected_changepoint_nrns', fin_bin_bool)

#######################
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
#######################
# Plots overlayed PSTHS and rasters for all tastes in
# a single plot
# Create overlay PSTH using 50ms bins
firing_bin_width = 50

plot_dir = os.path.join(\
        '/media/bigdata/firing_space_plot/firing_analyses/dynamic_neurons',
        'plots', os.path.basename(dat.data_dir[:-1]))
raster_plot_dir = os.path.join(plot_dir, 'rasters')
overlay_psth_plot_dir = os.path.join(plot_dir, 'overlay_psth')
conditions = ['bad_nrns','good_nrns']

if os.path.exists(raster_plot_dir):
    shutil.rmtree(plot_dir)
for this_cond in conditions:
    os.makedirs(os.path.join(raster_plot_dir,this_cond))
    os.makedirs(os.path.join(overlay_psth_plot_dir,this_cond))

firing_array = np.mean(np.reshape(spike_array, 
        (*spike_array.shape[:-1],-1,firing_bin_width)), axis = (1,-1))
tick_count = 6
t_inds = np.vectorize(np.int)\
        (np.linspace(plot_time_lims[0], plot_time_lims[1], tick_count))

for nrn_num, bool_val in enumerate(fin_bin_bool):
    this_spikes = spike_array[:,:,nrn_num]
    this_firing = firing_array[:,nrn_num]

    bools = np.array([trial_bin_bool[nrn_num], time_bin_bool[nrn_num],
            all_firing_bool[nrn_num]])*1
    bools_text = 'Trial {}, Time {}, Firing {}'.format(*bools)

    fig, ax = visualize.gen_square_subplots(len(spike_array), 
            sharex = True, sharey = True) 
    for num,(this_taste, this_ax) in enumerate(zip(this_spikes, ax.flatten())):
       this_ax = visualize.raster(this_ax, this_taste, marker = "|") 
       this_ax.axvline(stim_t - time_lims[0], 
               linewidth = 2, color = 'red', alpha = 0.5)
       this_ax.set_title(f'Taste {num}')
       plt.suptitle(os.path.basename(dat.data_dir[:-1]) + '\n' + f'Unit {nrn_num} : ' \
               + bools_text)
    for num in range(int(np.sqrt(len(spike_array)))):
        ax[num,0].set_ylabel('Trial')
        ax[-1,num].set_xlabel('Time post-stim (ms)')
        ax[-1,num].set_xticks(np.linspace(0,spike_array.shape[-1],tick_count))
        ax[-1,num].set_xticklabels(t_inds)
    plt.subplots_adjust(top=0.8)
    fig.savefig(os.path.join(raster_plot_dir, conditions[bool_val*1],
        f'firing_raster_{nrn_num}'))
    plt.close(fig)
    #plt.show()

    plt.plot(this_firing.T)
    plt.suptitle(os.path.basename(dat.data_dir[:-1]) + '\n' + f'Unit {nrn_num} : ' \
           + bools_text)
    plt.xlabel('Firing post-stim (ms)')
    plt.ylabel('Firing Rate')
    plt.xticks(np.linspace(0,firing_array.shape[-1], tick_count), t_inds)
    plt.subplots_adjust(top=0.8)
    plt.savefig(os.path.join(overlay_psth_plot_dir, conditions[bool_val*1],
        f'overlay_psth_{nrn_num}'))
    plt.close('all')
    #plt.show()
