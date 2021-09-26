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

def isi_rate(array, smoothing_fraction = 0.05):
    # For interpolation coverage, force spiking at start and end of trial
    array[...,0] = 1
    array[...,-1] = 1
    trial_inds = list(np.ndindex(array.shape[:-1]))
    spike_inds = [np.where(array[x])[0] for x in trial_inds]
    isi_s = [np.diff(inds) for inds in spike_inds]
    spike_counts = [len(x) for x in spike_inds]
    kern_len = [int(smoothing_fraction * count) for count in spike_counts]
    filt_isi = [np.convolve(x, np.ones(this_len), mode = 'same')\
            if len(x) > this_len else np.ones(len(x))*np.inf \
            for x, this_len in zip(isi_s, kern_len)]
    filt_rate = [1/x for x in filt_isi]
    pad_filt_rate = [np.pad(x,(1,1),'constant',constant_values = (0,0)) \
            for x in filt_rate]
    x_new = np.arange(array.shape[-1])
    fin_rate = [interp1d(this_inds, this_filt_rate[:-1])(x_new) \
            for this_inds, this_filt_rate in zip(spike_inds, pad_filt_rate)]
    fin_rate_array = np.empty((*array.shape[:-1],len(x_new)))
    for num,this_ind in enumerate(trial_inds):
        fin_rate_array[this_ind] = fin_rate[num]
    return x_new, fin_rate_array

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
data_dir = dir_list[-1]

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()

region_frame = pd.concat(\
        [pd.DataFrame({'region' : [region]*(len(nrns)),
                        'neuron' : nrns}) \
                for region, nrns in zip(dat.region_names, dat.region_units)])

spikes = np.array(dat.spikes).swapaxes(1,2).swapaxes(0,1)

#########################################
### Responsive Neurons 
#########################################
#base_time_lims = [0,2000]
#post_time_lims = [2000, 4000]
#
#base_spikes = np.mean(spikes[..., base_time_lims[0]:base_time_lims[1]],axis=-1)
#post_spikes = np.mean(spikes[..., post_time_lims[0]:post_time_lims[1]],axis=-1)
#
## Run pairwise t-tests across trials
#inds = list(np.ndindex(base_spikes.shape[:-1]))
#p_vals_array = np.empty(base_spikes.shape[:-1])
#for this_ind in inds:
#    p_vals_array[this_ind] = ttest_rel(base_spikes[this_ind],
#                                        post_spikes[this_ind])[1]
#alpha = 0.05
#sig_array = p_vals_array < alpha
#neuron_ind, taste_ind = np.where(sig_array)

########################################
## PSTH Repsonses
## (Spikes summed across trials)
########################################
psths = np.sum(spikes, axis=2)

step_size = 0.001
#firing_array = dat._calc_baks_rate(step_size, 0.001, psths)
x_new, firing_array = isi_rate(psths)

#sig_spikes = np.array([spikes[this_ind] for this_ind \
#        in list(zip(*np.where(sig_array)))])
#sig_sum_spikes = np.sum(sig_spikes,axis=1)
#sig_responses = np.array([firing_array[this_ind] for this_ind \
#        in list(zip(*np.where(sig_array)))])

########################################
# Response period
########################################
chop_ends = 500
cut_firing = firing_array[...,chop_ends:-chop_ends]
stim_t = 2000 - chop_ends
stim_ind = stim_t #int(2//step_size)

# Subtract 250ms to account for any bleeding from the firing rate calculation
pre_stim_rates = cut_firing[...,:(stim_ind-250)] 
post_stim_rates = cut_firing[...,stim_ind:]

inds = list(np.ndindex(pre_stim_rates.shape[:-1]))
pre_stim_long = np.array([pre_stim_rates[this_ind] for this_ind in inds])
post_stim_long = np.array([post_stim_rates[this_ind] for this_ind in inds])

ci = 99
pre_stim_ci = [np.percentile(this_pre, [(100-ci)/2, ci + ((100-ci)/2)]) \
        for this_pre in pre_stim_long]

ci_sig_array = np.array([(this_post < this_ci[0])*1 + (this_post > this_ci[1]) \
        for this_post, this_ci in zip(post_stim_long, pre_stim_ci)])

sig_window = 150
this_conv = lambda x : np.convolve(x, np.ones(sig_window)/sig_window, mode = 'same')
# Ensuring the selection criteria for the response is true
sig_conv = np.apply_along_axis(this_conv, axis=-1, arr = ci_sig_array) 

# Pick out response periods
# Set first time lim = 0 so we only have to look for first up and down
# Set last bin = 0 so we have a definitive end to all responses 
sig_conv[...,0] = 0
sig_conv[...,-1] = 0

all_ups = np.where(sig_conv==1)
all_downs = np.where(sig_conv==0)

inds_array = np.array(inds)
nrn_ind_frame = pd.DataFrame({
                'response' : np.arange(len(inds_array)),
                'neuron' : inds_array[:,0],
                'taste' : inds_array[:,1]})
nrn_ind_frame = nrn_ind_frame.merge(region_frame, on = 'neuron')

on_frame = pd.DataFrame({
                'response' : all_ups[0],
                'time'  : all_ups[1],
                'on' : 1})
off_frame = pd.DataFrame({
                'response' : all_downs[0],
                'time'  : all_downs[1],
                'off' : 1})
on_frame.sort_values(by = ['response','time'], inplace=True)
off_frame.sort_values(by = ['response','time'], inplace=True)
                
on_frame = on_frame.groupby('response').head(1).reset_index(drop=True)
# Only take off respones if on present
off_frame = off_frame.loc[off_frame.response.isin(on_frame.response)]
# Take first value after on timepoint
fin_frame = off_frame.merge(on_frame, how = 'outer', on = 'response',
        suffixes = ['_off','_on'])
fin_frame = fin_frame.loc[fin_frame.time_on < fin_frame.time_off]
fin_frame = fin_frame.groupby('response').head(1).reset_index(drop=True)
fin_frame = fin_frame.merge(nrn_ind_frame, how = 'inner', on = 'response')
fin_frame.drop(labels = ['off','on'], axis = 1, inplace=True)

# Max out values at 2000 post-stim
fin_frame[['time_on','time_off']] = fin_frame[['time_on','time_off']].clip(upper = 2000)
fin_frame['duration'] = fin_frame['time_off'] - fin_frame['time_on']
fin_frame = fin_frame[[
        'neuron','taste','response','region','time_on','time_off','duration']]

first_up = fin_frame.time_on
first_down = fin_frame.time_off
sig_conv = sig_conv[np.sort(fin_frame.response)]
ci_sig_array = ci_sig_array[np.sort(fin_frame.response)]

mean_nrn_frame = fin_frame.groupby('neuron').mean().reset_index()
mean_nrn_frame = mean_nrn_frame.merge(\
        nrn_ind_frame[['neuron','region']].drop_duplicates(),
        how = 'inner', on = 'neuron').drop(\
                columns = ['neuron','taste','response'])
mean_nrn_frame = mean_nrn_frame.round()

plot_dir = os.path.join(dat.data_dir, 'unit_response_plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

g = sns.relplot(x='duration', y='time_on', col='region', data = fin_frame)
plt.subplots_adjust(0.1,0.1, top = 0.8)
plt.suptitle(f'{dat.hdf5_name}' + '\n' 'All responses')
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'all_responses'))
#plt.show()

g = sns.relplot(x='duration', y='time_on', col='region',data = mean_nrn_frame)
plt.subplots_adjust(0.1,0.1, top = 0.8)
plt.suptitle(f'{dat.hdf5_name}' + '\n' 'Mean responses')
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'mean_responses'))
plt.show()

fin_frame.to_hdf(dat.hdf5_path, 
        key = os.path.join('ancillary_analysis', 'unit_response_characteristic'))

#first_up = np.array([np.where(x==1)[0][0] for x in sig_conv])
#all_downs = np.array([np.where(x == 0)[0] for up,x in zip(first_up, sig_conv)])
#first_down = np.array([this_downs[np.where(this_downs > this_up)[0][0]] \
#        for this_up, this_downs in zip(first_up, all_downs)])

#fig,ax = plt.subplots(1,2, sharex=True, sharey=True)
#ax[0].imshow(sig_conv, aspect='auto')
#ax[1].imshow(ci_sig_array, aspect='auto')
#for num, (up, down) in enumerate(zip(first_up, first_down)):
#    ax[0].scatter(up, num, color = 'red')
#    ax[0].scatter(down, num, color = 'red')
#    ax[1].scatter(up, num, color = 'red')
#    ax[1].scatter(down, num, color = 'red')
#plt.show()

#durations = first_down-first_up
#
#dframe = pd.DataFrame({
#            'taste' : taste_ind,
#            'neuron' : neuron_ind,
#            'up' : first_up,
#            'duration' : durations})
#
#fin_frame = dframe.merge(region_frame, on = 'neuron')
#fin_frame['session_name'] = os.path.basename(dat.data_dir) 
#fin_frame['animal_num'] = fin_frame['session_name'].iloc[0].split('_')[0]
#fin_frame['date'] = fin_frame['session_name'].iloc[0].split('_')[2]
#fin_frame.sort_values(['neuron','taste'], inplace=True)
#
#sort_inds = np.argsort(fin_frame['neuron'].to_numpy())
#
#plot_frame = fin_frame.query('region == "bla"')
#plt.scatter(plot_frame['duration'], plot_frame['up'])
#plt.show()

#fig, ax = plt.subplots(len(sig_conv),1, sharex=True, figsize = (5,15))

#fig,ax = visualize.gen_square_subplots(len(sig_conv))
#conv_time = np.arange(stim_ind, stim_ind + sig_conv.shape[-1])
#for num, this_ax in enumerate(ax.flatten()):
#    #num = sort_inds[in_num]
#    #zscore_y = stats.zscore(sig_responses[num])
#    zscore_y = sig_responses[num]
#    this_ax.plot(zscore_y)
#    #this_ax.plot(conv_time, sig_conv[num])
#    times = np.where(sig_conv[num] > 0)[0]+ stim_t
#    this_ax.scatter(times ,zscore_y[times], s = 5, color = 'orange')
#    this_ax.axvline(stim_ind, alpha = 0.5, color ='red')
#    this_ax.axhline(pre_stim_ci[num][0], 0, len(zscore_y), alpha = 0.5, color ='red')
#    this_ax.axhline(pre_stim_ci[num][1], 0, len(zscore_y), alpha = 0.5, color ='red')
#    #this_ax.set_ylabel(f'Nrn {fin_frame.neuron.loc[num]}' + '\n' +f'taste {fin_frame.taste.loc[num]}')
#plt.show()

