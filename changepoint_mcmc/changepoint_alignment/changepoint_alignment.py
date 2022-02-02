"""
Code to align spiking activity to changepoints from a specified model
Calculate statistics on sharpening of activity due to alignment using:
    1) Change in activity in windows before and after transition
        - Controls :
            a) Distribution of changes in activity across bins in single trials
            b) Activity from trial-shuffled alignments
    2) Average template of neural activity surrounding transition
        - Tested using 2 Way ANOVA (group x time-bin)
    3) Correlation of activity during different states with palatability

Code should perform alignment analysis regardless of region of recording
This way, alignment by region can just be pulled out later
But which model to use should be specified
Make sure to store metadata
Plots and metadata kept in separate folder for entire analysis
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
import shutil
#import pymc3 as pm
import re
from glob import glob
from tqdm import tqdm, trange

import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import percentileofscore,mode,zscore
from scipy.ndimage import gaussian_filter1d as gfilt

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def get_state_firing(spike_array,tau_array):
    """
    spike_array : trials x nrns x bins
    tau_array : trials x switchpoints
    """
    states = tau_array.shape[-1] + 1
    # Get mean firing rate for each STATE using model
    state_inds = np.hstack([np.zeros((tau_array.shape[0],1)),
                            tau_array,
                            np.ones((tau_array.shape[0],1))*spike_array.shape[-1]])
    state_lims = np.array([state_inds[:,x:x+2] for x in range(states)])
    state_lims = np.vectorize(np.int)(state_lims)
    state_lims = np.swapaxes(state_lims,0,1)

    state_firing = \
            np.array([[np.mean(trial_dat[:,start:end],axis=-1) \
            for start, end in trial_lims] \
            for trial_dat, trial_lims in zip(spike_array,state_lims)])

    state_firing = np.nan_to_num(state_firing)
    return state_firing

def get_state_windows(spike_array, transition_times, window_radius, shuffle = False):
    """
    **NOTE** : This function does NOT HANDLE windows going past array limits
    spike_array : trials x nrns x bins
    transition_times : trials x switchpoints
    If shuffle = True, returns windows using trial shuffled transition times
    """
    states = transition_times.shape[-1] + 1
    # Get mean firing rate for each STATE using model
    if shuffle:
        fin_transition_times = np.array([np.random.permutation(x) \
                for x in transition_times.T]).T
    else:
        fin_transition_times = transition_times

    window_lims = np.array([[\
            (x-window_radius, x+window_radius) for x in transition]
            for transition in transition_times])
    window_lims = np.vectorize(np.int)(window_lims)
    aligned_windows = \
            np.zeros((*spike_array.shape[:2], 
                window_lims.shape[1], int(2*window_radius))) 
    inds = np.ndindex(aligned_windows.shape[:-1])
    for this_ind in inds:
        window_inds = window_lims[this_ind[0],this_ind[2]]
        this_trial = spike_array[this_ind[:2]]
        aligned_windows[this_ind] = this_trial[window_inds[0]:window_inds[1]]

    return aligned_windows

class params_from_path:
    def __init__(self, path):
        # Extract model params from basename
        self.path = path
        self.model_name = os.path.basename(self.path).split('.')[0]
        self.states = int(re.findall("\d+states",self.model_name)[0][:-6])
        self.time_lims = [int(x) for x in \
                re.findall("\d+_\d+time",self.model_name)[0][:-4].split('_')]
        self.bin_width = int(re.findall("\d+bin",self.model_name)[0][:-3])
        self.fit_type = re.findall("type_.+",self.model_name)[0].split('_')[1]
        # Exctract data_dir from model_path
        self.data_dir = "/".join(self.path.split('/')[:-3])
        self.session_name = self.data_dir.split('/')[-1]
        self.animal_name = self.session_name.split('_')[0]

def load_model(model_path):
    if os.path.exists(model_path):
        print('Trace loaded from cache')
        with open(model_path, 'rb') as buff:
            data = pickle.load(buff)
        #lambda_stack = data['lambda']
        tau_samples = data['tau']
        binned_dat = data['data']
        # Remove pickled data to conserve memory
        del data
        # Recreate samples
        return tau_samples, binned_dat
    else:
        raise Exception('Model path does not exist')

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

#parser = argparse.ArgumentParser(description = 'Script to analyze fit models')
#parser.add_argument('model_path',  help = 'Path to model pkl file')
#args = parser.parse_args()
#model_path = args.model_path 

model_path = '/media/bigdata/Abuzar_Data/bla_gc/AM35/AM35_4Tastes_201230_115322/'\
        'saved_models/vi_4_states/'\
        'actual_vi_4states_40000fit_2000_4000time_50bin_region_gc_type_reg.pkl'

if not os.path.exists(model_path):
    raise Exception('Model path does not exist')


##########
# PARAMS 
##########
params = params_from_path(model_path)

dat = ephys_data(params.data_dir)
dat.get_spikes()

########################################
# Create dirs and names
########################################
plot_super_dir = '/media/bigdata/firing_space_plot/'\
        'changepoint_mcmc/changepoint_alignment/plots'
plot_dir = os.path.join(plot_super_dir,
        params.animal_name, params.model_name,'analysis_plots')

if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir)

tau_samples, binned_dat = load_model(params.path)
binned_t_vec = np.arange(*params.time_lims)[::params.bin_width]
taste_label = np.sort(list(range(len(dat.spikes)))*dat.spikes[0].shape[0])

##################################################
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           
##################################################
# To be able to move between mean and mode more easily
#stat_tau = np.mean(tau_samples, axis=0)
int_tau = np.vectorize(np.int)(tau_samples)
stat_tau = np.squeeze(mode(int_tau,axis=0)[0])
scaled_stat_tau = (stat_tau/binned_dat.shape[-1])\
        *np.diff(params.time_lims)[0] + params.time_lims[0]

# Reshape tau to have a taste index
scaled_stat_tau = np.reshape(scaled_stat_tau, \
        (len(dat.spikes),-1, scaled_stat_tau.shape[-1]))

unbinned_spikes = np.array(dat.spikes)

window_radius = 200
shuffle_count = 100
aligned_windows = np.array([get_state_windows(this_spikes, this_tau, window_radius) \
        for this_spikes, this_tau in zip(unbinned_spikes, scaled_stat_tau)])
#shuffled_windows = np.array([[get_state_windows(this_spikes, this_tau, 
#                                            window_radius, shuffle = True) \
#        for this_spikes, this_tau in zip(unbinned_spikes, scaled_stat_tau)]\
#        for i in trange(shuffle_count)])

#bin_size = 50
#binned_aligned = np.reshape(aligned_windows, 
#        (*aligned_windows.shape[:-1],-1, bin_size)) 
#binned_aligned = np.sum(binned_aligned, axis = -1)
#binned_shuffled = np.reshape(shuffled_windows, 
#        (*shuffled_windows.shape[:-1],-1, bin_size)) 
#binned_shuffled = np.sum(binned_shuffled, axis = -1)
#
## For shuffles, each shuffle is a set of new trials which shouldn't
## have transitions which are as sharp, so we can concatenate all shuffled
## sets into one
#binned_shuffled = np.moveaxis(binned_shuffled, 0,1)
#binned_shuffled = np.reshape(binned_shuffled,
#        (binned_shuffled.shape[0],-1,*binned_shuffled.shape[3:]))
#
#binned_aligned = np.moveaxis(binned_aligned, 1,-1)
#binned_shuffled = np.moveaxis(binned_shuffled, 1,-1)
#
## Add noise so ANOVA doesn't conk out
#binned_aligned += np.random.random(binned_aligned.shape) * 1e-3
#binned_shuffled += np.random.random(binned_shuffled.shape) * 1e-3
#
#inds = np.array(list(np.ndindex(binned_aligned.shape[:3])))
#
#align_inds = np.array(list(np.ndindex(binned_aligned.shape[3:]))).T
#shuffle_inds = np.array(list(np.ndindex(binned_shuffled.shape[3:]))).T
#
#type_p_val = []
#interact_p_val = []
#
#def calculate_anova(this_ind):
#    align_frame = pd.DataFrame(
#        dict(zip(['bin','trial','firing','type'], 
#            [*align_inds, binned_aligned[tuple(this_ind)].flatten(), 'actual'])))
#    shuffle_frame = pd.DataFrame(
#        dict(zip(['bin','trial','firing','type'], 
#            [*shuffle_inds, binned_shuffled[tuple(this_ind)].flatten(),'shuffle'])))
#    fin_frame = pd.concat([align_frame, shuffle_frame])
#    anova_out = pg.anova(data = fin_frame, dv = 'firing', between = ['type','bin'])
#    return (anova_out['p-unc'][0],anova_out['p-unc'][2])
#
#outs = np.array(parallelize(calculate_anova,inds))

#binned_aligned_long = np.moveaxis(binned_aligned,-1,1)
#binned_aligned_long = np.reshape(binned_aligned_long, (-1, *binned_aligned_long.shape[2:]))
#binned_aligned_long = np.moveaxis(binned_aligned_long, 0,-1)
#binned_shuffled_long = np.moveaxis(binned_shuffled,-1,1)
#binned_shuffled_long = np.reshape(binned_shuffled_long, (-1, *binned_shuffled_long.shape[2:]))
#binned_shuffled_long = np.moveaxis(binned_shuffled_long, 0,-1)
#
#nrn_ind = 1
#fig,ax = plt.subplots(2,binned_aligned_long.shape[1])
#for transition in range(ax.shape[-1]):
#    ax[0,transition].imshow(binned_aligned_long[nrn_ind,transition].T, aspect='auto')
#    ax[1,transition].imshow(binned_shuffled_long[nrn_ind,transition].T, aspect = 'auto')
#plt.show()

aligned_windows_long = np.reshape(aligned_windows, (-1, *aligned_windows.shape[2:]))
aligned_windows_long = np.moveaxis(aligned_windows_long, 0,-2)

def subplot_generator(trans_num, taste_num):




for nrn_ind in trange(aligned_windows.shape[2]):

    plot_dat = aligned_windows[:,:,nrn_ind]
    # Take mean before filtering
    mean_dat = np.mean(plot_dat,axis=1)
    plot_dat = gfilt(plot_dat, sigma = 10) 
    plot_dat_long = np.reshape(plot_dat,(-1, *plot_dat.shape[2:]))
    #plot_dat_long = zscore(plot_dat_long,axis=-1)
    mean_dat = gfilt(mean_dat, sigma = 20) 
    mean_dat = mean_dat.swapaxes(0,1)
    mean_dat = np.reshape(mean_dat, (-1, mean_dat.shape[-1]), order = 'F')
    #mean_dat = zscore(mean_dat,axis=-1)
    marks = np.arange(0,plot_dat_long.shape[0], plot_dat.shape[1])

    trans_num = plot_dat.shape[2]
    taste_num = plot_dat.shape[0]
    fig = plt.figure(figsize = (5,10))
    upper_axes = [fig.add_subplot(2,trans_num,num+1) for num in range(trans_num)]
    lower_shape = (taste_num*2, trans_num)
    inds = np.array(list(np.ndindex(lower_shape)))
    wanted_inds = np.where(inds[:,0] >3)[0]
    lower_axes = [fig.add_subplot(lower_shape[0],lower_shape[1],this_ind+1) \
            for this_ind in wanted_inds]
    for this_trans in range(len(upper_axes)):
        upper_axes[this_trans].imshow(plot_dat_long[:,this_trans], 
                aspect='auto',cmap='viridis')
        upper_axes[this_trans].hlines(marks, 0, plot_dat.shape[-1], 
                linewidth = 2, color = 'red', linestyle = 'dashed')
        for this_plot in range(len(lower_axes)):
            lower_axes[this_plot].plot(mean_dat[this_plot].T)
    for this_ax in np.concatenate([[*upper_axes,*lower_axes]]):
        this_ax.axes.get_yaxis().set_visible(False)
        #this_ax.axes.get_xaxis().set_visible(False)
    fig.savefig(os.path.join(plot_dir,f'nrn_{nrn_ind}_aligned'))
    plt.close(fig)
    #plt.show()


    #fig,ax = plt.subplots(2,plot_dat_long.shape[1])
    #for num in range(plot_dat_long.shape[1]): 
    #    ax[0,num].imshow(plot_dat_long[:,num], aspect='auto',cmap='viridis')
    #    ax[1,num].plot(mean_dat[:,num].T)
    #    ax[0,num].hlines(marks, 0, plot_dat.shape[-1], 
    #            linewidth = 2, color = 'red', linestyle = 'dashed')
    #fig.savefig(os.path.join(plot_dir,f'nrn_{nrn_ind}_aligned'))
    #plt.close(fig)

#outs = [calculate_anova(this_ind) for this_ind in tqdm(inds)]

#for this_ind in tqdm(inds):
#    align_frame = pd.DataFrame(
#        dict(zip(['bin','trial','firing','type'], 
#            [*align_inds, binned_aligned[tuple(this_ind)].flatten(), 'actual'])))
#    shuffle_frame = pd.DataFrame(
#        dict(zip(['bin','trial','firing','type'], 
#            [*shuffle_inds, binned_shuffled[tuple(this_ind)].flatten(),'shuffle'])))
#    fin_frame = pd.concat([align_frame, shuffle_frame])
#    anova_out = pg.anova(data = fin_frame, dv = 'firing', between = ['type','bin'])
#    type_p_val.append(anova_out['p-unc'][0])
#    interact_p_val.append(anova_out['p-unc'][2])


#aligned_inds = np.array(list(np.ndindex(aligned_windows.shape)))
#shuffled_inds = np.array(list(np.ndindex(shuffled_windows.shape)))
#aligned_frame = pd.DataFrame(
#        dict(zip(['taste','trial','neuron','transition','time','spike'],
#            [*aligned_inds.T, aligned_windows.flatten()])))
#shuffled_frame = pd.DataFrame(
#        dict(zip(['shuffle_num','taste','trial','neuron','transition','time','spike'],
#            [*shuffled_inds.T, shuffled_windows.flatten()])))

## Zscore firing for later plotting
#state_firing = np.array([zscore(nrn) for nrn in state_firing.T]).T
#state_firing = np.nan_to_num(state_firing)

# Reshape state_firing to have separate axis for tastes
# Otherwise ANOVA will pull all tastes together
#taste_state_firing = np.reshape(state_firing,
#                        (len(dat.spikes),-1,*state_firing.shape[1:]))
#frame_inds = np.array(list(np.ndindex(taste_state_firing.shape)))
#
#mean_firing_frame = pd.DataFrame({\
#                    'taste' : frame_inds[:,0],
#                    'trial' : frame_inds[:,1],
#                    'state' : frame_inds[:,2],
#                    'neuron' : frame_inds[:,3],
#                    'firing' : taste_state_firing.flatten()})
#
