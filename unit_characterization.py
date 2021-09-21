"""
Script to calculate unit response characteristics per recording per region
1) Activity onset
2) Activity duration
3) Clusters of units based off 1 and 2
4) Duration of palatability responsiveness (dissimilarity metric)
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
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare, ttest_rel
import pylab as plt

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

dir_list_path = "/media/bigdata/Abuzar_Data/dir_list.txt"
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
data_dir = dir_list[0] 

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()

region_frame = pd.concat(\
        [pd.DataFrame({'region' : [region]*(len(nrns)),
                        'neuron' : nrns}) \
                for region, nrns in zip(dat.region_names, dat.region_units)])

spikes = np.array(dat.spikes)
#dat.firing_rate_params = dat.default_firing_params
#dat.firing_rate_params['window_size'] = 500
#dat.firing_rate_params['step_size'] = 10
#dat.firing_rate_params['type'] = 'baks'
#dat.firing_rate_params['baks_resolution'] = 0.025
##step_size = dat.firing_rate_params['step_size']
#step_size = dat.firing_rate_params['baks_resolution']*1000
#dat.get_firing_rates()

# Separate units by region
#sorted_units_list = [[spikes[:,:,x] for x in this_region_list]\
#        for this_region_list in dat.region_units]

#########################################
## Response period
#########################################
#stim_t = 2000
#stim_ind = int(2000//step_size)
#
#pre_stim_rates = dat.firing_array[...,:stim_ind] 
#post_stim_rates = dat.firing_array[...,stim_ind:]
#
#pre_stim_ci = np.percentile(pre_stim_rates, [2.5,97.5], axis = (-1,-2)) 
#pre_stim_ci_cast = [np.broadcast_to(np.expand_dims(x,axis=(-1,-2)),
#                        post_stim_rates.shape) \
#                    for x in pre_stim_ci]
#sig_array = (post_stim_rates < pre_stim_ci_cast[0])*1 + \
#                    (post_stim_rates > pre_stim_ci_cast[1])
#
#sig_window = 3
#this_conv = lambda x : np.convolve(x, np.ones(sig_window)/sig_window, mode = 'valid')
## Ensuring the selection criteria for the response is true
#sig_conv = np.apply_along_axis(this_conv, axis=-1, arr = sig_array) == 1
#
## Pick out response periods
## Set first time lim = 0 so we only have to look for first up and down
#sig_conv[...,0] = 0
#sig_conv[...,-1] = 0
#sig_conv = sig_conv*1
#sig_conv_diff = np.apply_along_axis(np.ediff1d, axis=-1, arr = sig_conv)
#ups = sig_conv_diff == 1
#downs = sig_conv_diff == -1
#
#trial_inds = np.array(list(np.ndindex(ups.shape[:-1])))
#first_up = [np.where(ups[tuple(ind)])[0][0] if sum(ups[tuple(ind)]) > 0 \
#        else np.nan for ind in trial_inds]
#first_down = [np.where(downs[tuple(ind)])[0][0] if sum(downs[tuple(ind)]) > 0 \
#        else np.nan for ind in trial_inds]
#
#response_frame = pd.DataFrame({
#                    'tastes' : trial_inds[:,0],
#                    'neurons' : trial_inds[:,1],
#                    'trials' : trial_inds[:,2],
#                    'up' : first_up,
#                    'down' : first_down})
#
#response_frame['duration'] = response_frame['down'] - response_frame['up']
#
#response_frame.groupby(by = ['neurons']).aggregate(np.mean)

########################################
## Responsive Neurons 
########################################
base_time_lims = [0,2000]
post_time_lims = [2000, 4000]

base_spikes = np.mean(spikes[..., base_time_lims[0]:base_time_lims[1]],axis=-1)
post_spikes = np.mean(spikes[..., post_time_lims[0]:post_time_lims[1]],axis=-1)
base_spikes = base_spikes.swapaxes(1,2)
post_spikes = post_spikes.swapaxes(1,2)

# Run pairwise t-tests across trials
inds = list(np.ndindex(base_spikes.shape[:-1]))
p_vals_array = np.empty(base_spikes.shape[:-1])
for this_ind in inds:
    p_vals_array[this_ind] = ttest_rel(base_spikes[this_ind],
                                        post_spikes[this_ind])[1]
alpha = 0.01
sig_array = p_vals_array < alpha
taste_ind, neuron_ind = np.where(sig_array)
########################################
## PSTH Repsonses
## (Spikes summed across trials)
########################################
psths = np.sum(spikes, axis=1)

#def isi_rate(array, smoothing_window):
#    # For interpolation coverage, force spiking at start and end of trial
#    array[...,0] = 1
#    array[...,-1] = 1
#    trial_inds = list(np.ndindex(array.shape[:-1]))
#    spike_inds = [np.where(array[x])[0] for x in trial_inds]
#    isi_s = [np.diff(inds) for inds in spike_inds]
#    filt_isi = [np.convolve(x, np.ones(smoothing_window), mode = 'same')\
#            if len(x) > smoothing_window else np.ones(len(x))*np.inf for x in isi_s]
#    filt_rate = [1/x for x in filt_isi]
#    pad_filt_rate = [np.pad(x,(1,1),'constant',constant_values = (0,0)) \
#            for x in filt_rate]
#    x_new = np.linspace(0,array.shape[-1]-1,int(array.shape[-1]/10))
#    fin_rate = [interp1d(this_inds, this_filt_rate[:-1])(x_new) \
#            for this_inds, this_filt_rate in zip(spike_inds, pad_filt_rate)]
#    fin_rate_array = np.empty((*array.shape[:-1],len(x_new)))
#    for num,this_ind in enumerate(trial_inds):
#        fin_rate_array[this_ind] = fin_rate[num]
#    return x_new, fin_rate_array
#
#x_new, psth_firing = isi_rate(psths, 51)
step_size = 0.01
baks_firing = dat._calc_baks_rate(step_size, 0.001, psths)

sig_responses = np.array([baks_firing[this_ind] for this_ind \
        in list(zip(*np.where(sig_array)))])

fig, ax = visualize.gen_square_subplots(len(sig_responses))
for this_ax, this_dat in zip(ax.flatten(), sig_responses):
    this_ax.plot(this_dat)
plt.show()

########################################
# Response period
########################################
stim_t = 2000
stim_ind = int(2//step_size)

pre_stim_rates = sig_responses[...,:stim_ind] 
post_stim_rates = sig_responses[...,stim_ind:]

pre_stim_ci = [np.percentile(this_pre, [2.5,97.5]) for this_pre in pre_stim_rates]
sig_array = np.array([(this_post < this_ci[0])*1 + (this_post > this_ci[1]) \
        for this_post, this_ci in zip(post_stim_rates, pre_stim_ci)])

fig, ax = plt.subplots(len(sig_array),1, sharex=True)
for num, this_ax in enumerate(ax):
    this_ax.plot(stats.zscore(sig_responses[num]))
    this_ax.plot(np.arange(stim_ind, stim_ind + sig_array.shape[-1]), sig_array[num])
plt.show()

sig_window = 5*3 # 3 windows of 50ms
this_conv = lambda x : np.convolve(x, np.ones(sig_window)/sig_window, mode = 'valid')
# Ensuring the selection criteria for the response is true
sig_conv = np.apply_along_axis(this_conv, axis=-1, arr = sig_array) 

fig, ax = plt.subplots(len(sig_array),1, sharex=True)
for num, this_ax in enumerate(ax):
    this_ax.plot(stats.zscore(sig_responses[num]))
    this_ax.plot(np.arange(stim_ind, stim_ind + sig_conv.shape[-1]), sig_conv[num])
plt.show()

# Pick out response periods
# Set first time lim = 0 so we only have to look for first up and down
sig_conv[...,0] = 0
sig_conv[...,-1] = 0

first_up = np.array([np.where(x)[0][0] for x in sig_conv])
first_down = np.array([np.where(x[up:] == 0)[0][0] for up,x in zip(first_up, sig_conv)])
durations = first_down-first_up

dframe = pd.DataFrame({
            'taste' : taste_ind,
            'neuron' : neuron_ind,
            'up' : first_up,
            'duration' : durations})

fin_frame = dframe.merge(region_frame, on = 'neuron')
fin_frame['session_name'] = os.path.basename(dat.data_dir) 
fin_frame['animal_num'] = fin_frame['session_name'].iloc[0].split('_')[0]
fin_frame['date'] = fin_frame['session_name'].iloc[0].split('_')[2]

#plt.scatter(durations, first_up)
#plt.show()

#this_unit = spikes[:,:,9]
#this_trial_spikes = this_unit[0,0]
#this_trial_firing = dat.firing_array[0,9,0] 
#
## ISI firing rate
#smoothing_window = 7
#spike_inds = np.where(this_trial_spikes)
#isi_s = np.diff(spike_inds).flatten()
##filt_isi = medfilt(isi_s.flatten(), smoothing_window)
#filt_isi = np.convolve(isi_s.flatten(), np.ones(smoothing_window), mode = 'same')
#
#def isi_rate(array, smoothing_window):
#    # For interpolation coverage, force spiking at start and end of trial
#    array[...,0] = 1
#    array[...,-1] = 1
#    trial_inds = list(np.ndindex(array.shape[:-1]))
#    spike_inds = [np.where(array[x])[0] for x in trial_inds]
#    isi_s = [np.diff(inds) for inds in spike_inds]
#    filt_isi = [np.convolve(x, np.ones(smoothing_window), mode = 'same')\
#            if len(x) > smoothing_window else np.ones(len(x))*np.inf for x in isi_s]
#    filt_rate = [1/x for x in filt_isi]
#    pad_filt_rate = [np.pad(x,(1,1),'constant',constant_values = (0,0)) \
#            for x in filt_rate]
#    x_new = np.linspace(0,array.shape[-1]-1,100)
#    fin_rate = [interp1d(this_inds, this_filt_rate[:-1])(x_new) \
#            for this_inds, this_filt_rate in zip(spike_inds, pad_filt_rate)]
#    fin_rate_array = np.empty((*array.shape[:-1],len(x_new)))
#    for num,this_ind in enumerate(trial_inds):
#        fin_rate_array[this_ind] = fin_rate[num]
#
#    return x_new, fin_rate_array
#
#x_new, isi_rate_array = isi_rate(spikes, 7)
#inds = (0,0,9)
#plt.plot(x_new, stats.zscore(isi_rate_array[inds]))
#plt.plot(np.linspace(0,7000,dat.firing_array.shape[-1]), 
#        stats.zscore(dat.firing_array[inds[0],inds[2],inds[1]]))
##spike_inds = np.where(spikes[inds])[0]
##plt.scatter(spike_inds, np.ones(len(spike_inds)), marker = '|')
#plt.plot(spikes[inds])
#plt.show()
#
#
#nrn_ind = 4
#visualize.firing_overview(isi_rate_array[:,:,nrn_ind,1:-1])
#visualize.firing_overview(dat.firing_array[:,nrn_ind,1:-1])
#plt.show()
#
#fig,ax = plt.subplots(2,1)
#ax[0].plot(np.mean(isi_rate_array[:,:,nrn_ind,1:-1],axis=1).T)
#ax[1].plot(np.mean(dat.firing_array[:,nrn_ind],axis=1).T)
#plt.show()
#
#plt.plot(spike_inds[0][:-1], stats.zscore(1/filt_isi))
#plt.plot(np.linspace(0,7000,len(this_trial_firing)), stats.zscore(this_trial_firing))
#plt.scatter(spike_inds, np.ones(len(spike_inds[0])), marker = '|')
#plt.show()


