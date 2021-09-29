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
from scipy.stats import kruskal
import pylab as plt
from glob import glob
import json
from scipy.spatial import distance_matrix as distmat
from sklearn.preprocessing import LabelEncoder as LE
import pingouin as pg

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
    kern_len = np.array([int(smoothing_fraction * count) for count in spike_counts])
    # Enforce minimum length of 1
    kern_len[np.where(kern_len == 0)] = 1
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
#data_dir = dir_list[39]

#for data_dir in tqdm(dir_list):

data_dir = dir_list[0]

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()

region_frame = pd.concat(\
        [pd.DataFrame({'region' : [region]*(len(nrns)),
                        'neuron' : nrns}) \
                for region, nrns in zip(dat.region_names, dat.region_units)])

dat.check_laser()
if dat.laser_exists:
    dat.separate_laser_spikes()
    spikes = np.array(dat.off_spikes).swapaxes(1,2).swapaxes(0,1)
else:
    spikes = np.array(dat.spikes).swapaxes(1,2).swapaxes(0,1)

sum_spikes = np.sum(spikes,axis=(1,2,3))
# Put one spike at start and end to ensure there is atleast 1 ISI
spikes[...,0] = 1
spikes[...,-1] = 1

########################################
## PSTH Repsonses
## (Spikes summed across trials)
########################################
psths = np.sum(spikes, axis=2)

step_size = 0.001
x_new, firing_array = isi_rate(psths)

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
firing_long = np.array([cut_firing[this_ind] for this_ind in inds])
pre_stim_long = np.array([pre_stim_rates[this_ind] for this_ind in inds])
post_stim_long = np.array([post_stim_rates[this_ind] for this_ind in inds])

ci = 99
pre_stim_ci = [np.percentile(this_pre, [(100-ci)/2, ci + ((100-ci)/2)]) \
        for this_pre in pre_stim_long]

ci_sig_array = np.array([(this_post < this_ci[0])*1 + (this_post > this_ci[1]) \
        for this_post, this_ci in zip(post_stim_long, pre_stim_ci)])

sig_window = 150
this_conv = lambda x : np.convolve(
        x, np.ones(sig_window)/sig_window, mode = 'same')
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
post_upper_lim = 5000
fin_frame[['time_on','time_off']] = \
        fin_frame[['time_on','time_off']].clip(upper = post_upper_lim)
fin_frame['duration'] = fin_frame['time_off'] - fin_frame['time_on']
fin_frame = fin_frame[[
        'neuron','taste','response','region','time_on','time_off','duration']]

fin_frame.to_hdf(dat.hdf5_path, 
        key = os.path.join('ancillary_analysis', 'unit_response_characteristic'))

first_up = fin_frame.time_on
first_down = fin_frame.time_off
response_num = fin_frame.response

mean_nrn_frame = fin_frame.groupby('neuron').mean().reset_index()
mean_nrn_frame = mean_nrn_frame.merge(\
        nrn_ind_frame[['neuron','region']].drop_duplicates(),
        how = 'inner', on = 'neuron').drop(\
                columns = ['neuron','taste','response'])
mean_nrn_frame = mean_nrn_frame.round()

########################################
# ____  _       _       
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
#                       
########################################

plot_dir = os.path.join(dat.data_dir, 'unit_response_plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

g = sns.relplot(x='duration', y='time_on', col='region', data = fin_frame)
plt.subplots_adjust(0.1,0.1, top = 0.8)
plt.suptitle(f'{dat.hdf5_name}' + '\n' 'All responses')
plt.xlim([0,post_upper_lim])
plt.ylim([0,post_upper_lim])
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'all_responses'))
#plt.show()

g = sns.relplot(x='duration', y='time_on', col='region',data = mean_nrn_frame)
plt.subplots_adjust(0.1,0.1, top = 0.8)
plt.suptitle(f'{dat.hdf5_name}' + '\n' 'Mean responses')
plt.xlim([0,post_upper_lim])
plt.ylim([0,post_upper_lim])
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'mean_responses'))
#plt.show()


fig,ax = plt.subplots(1,2, sharex=True, sharey=True, figsize = (10,15))
ax[0].imshow(sig_conv[:,:post_upper_lim], aspect='auto', cmap = 'viridis')
ax[1].imshow(ci_sig_array[:,:post_upper_lim], aspect='auto')
for num, up, down in zip(response_num, first_up, first_down):
    ax[0].scatter(up, num, color = 'red', s=8)
    ax[0].scatter(down, num, color = 'red', s=8)
    ax[1].scatter(up, num, color = 'red', s=8)
    ax[1].scatter(down, num, color = 'red', s=8)
plt.suptitle(f'{dat.hdf5_name}' + '\n' 'All responses')
plt.subplots_adjust(0.1,0.1, top = 0.9)
fig.savefig(os.path.join(plot_dir,'all_responses_up_down'))
#plt.show()

fig,ax = visualize.gen_square_subplots(
        len(sig_conv), sharex=True, figsize = (20,20))
conv_time = np.arange(stim_ind, stim_ind + sig_conv.shape[-1])
for num in range(len(firing_long)):
    this_ax = ax.flatten()[num]
    zscore_y = firing_long[num, :(stim_t+post_upper_lim)]
    this_ax.plot(zscore_y)
    this_ax.axes.get_yaxis().set_visible(False)
    this_ax.axvline(stim_ind, alpha = 0.5, color ='red')
    this_ax.axhline(\
            pre_stim_ci[num][0], 0, len(zscore_y), alpha = 0.5, color ='red')
    this_ax.axhline(\
            pre_stim_ci[num][1], 0, len(zscore_y), alpha = 0.5, color ='red')
for num, up, down in zip(response_num, first_up, first_down):
    ax.flatten()[num].\
            axvspan(up + stim_t ,down + stim_t, color = 'yellow', alpha = 0.5)
plt.suptitle(f'{dat.hdf5_name}' + '\n' 'All responses' + '\n' + \
        f'Sig responses {len(response_num)}/{len(sig_conv)}')
plt.subplots_adjust(0.1,0.1, top = 0.9)
fig.savefig(os.path.join(plot_dir,'all_responses_traces'))
#plt.show()

plt.close('all')

############################################################
#|  _ \ __ _| | __ _| |_ __ _| |__ (_) (_) |_ _   _ 
#| |_) / _` | |/ _` | __/ _` | '_ \| | | | __| | | |
#|  __/ (_| | | (_| | || (_| | |_) | | | | |_| |_| |
#|_|   \__,_|_|\__,_|\__\__,_|_.__/|_|_|_|\__|\__, |
#                                             |___/ 
############################################################

json_path = glob(os.path.join(dat.data_dir,"*.info"))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)
tastant_names = info_dict['taste_params']['tastes']
pal_inds = info_dict['taste_params']['pal_rankings']
# If recordings don't match tastant order, GTFO: 
if not pal_inds == [3,4,2,1]: 
    continue

psth_stim_t = 2000
post_stim_psths = psths[...,psth_stim_t:]
epoch_lims = [[0,250],[250,1000],[1000,1750]]
epoch_stim_rates = np.array([np.mean(post_stim_psths[...,lim[0]:lim[1]],axis=-1)\
        for lim in epoch_lims])
inds = np.array(list(np.ndindex(epoch_stim_rates.shape[:-1])))
epoch_stim_rates_long = [epoch_stim_rates[tuple(this_ind)] for this_ind in inds]
dist_list = \
        [distmat(x[:,np.newaxis],x[:,np.newaxis]) for x in epoch_stim_rates_long] 
# Add a bit of noise to prevent nans
dist_list = [x + np.random.random(x.shape) * 1e-3 for x in dist_list]
# Normalize each dist to the max value
dist_list = [x/np.max(x,axis=None) for x in dist_list]
similar_dists = [[x[0,1],x[3,2]] for x in dist_list]
non_similar_dists = [x[2:][:,:2].flatten() for x in dist_list]

similar_frame = pd.DataFrame({
    'epoch' : np.repeat(inds[:,0],2),
    'neuron' : np.repeat(inds[:,1],2),
    'dist' : np.array(similar_dists).flatten(),
    'type' : 'similar'})
non_similar_frame = pd.DataFrame({
    'epoch' : np.repeat(inds[:,0],4),
    'neuron' : np.repeat(inds[:,1],4),
    'dist' : np.array(non_similar_dists).flatten(),
    'type' : 'non_similar'})
dists_frame = pd.concat([similar_frame, non_similar_frame])
dists_frame = dists_frame.merge(
        nrn_ind_frame[['neuron','region']].drop_duplicates(),
        how = 'inner', on = 'neuron')

dists_frame.region, region_encoding = pd.factorize(dists_frame.region)
dists_frame.type, type_encoding = pd.factorize(dists_frame.type)

mean_dists_frame = \
        dists_frame.groupby(['epoch','neuron','type']).mean().reset_index()
epoch_neuron, grouped_dat = zip(*list(mean_dists_frame.groupby(['epoch','neuron'])))
epoch_neuron = np.array(epoch_neuron)
diffs_list = [np.diff(x.dist)[0] for x in grouped_dat]
region_list  = [x.region.iloc[0] for x in grouped_dat]
diff_dists_frame = pd.DataFrame({
    'epoch' : epoch_neuron[:,0],
    'neuron' : epoch_neuron[:,1],
    'diff_dist' : diffs_list,
    'region' : region_list})

mean_dists_array = np.array([x[1].dist for x in mean_dists_frame.groupby('neuron')])
mean_diff_dists_array = \
        np.array([x[1].diff_dist for x in diff_dists_frame.groupby('neuron')])
zscore_diff_dists_array = stats.zscore(mean_diff_dists_array,axis=-1)
diff_region_list = [x[1].region.iloc[0] for x in diff_dists_frame.groupby('neuron')]

#g = sns.catplot(x = 'epoch', y = 'dist', hue = 'type', col = 'neuron',
#        kind = 'bar', ci = 'sd', col_wrap = 6, data = dists_frame)
##plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.show()

lut = dict(zip(np.unique(diff_region_list),"rbg"))
row_colors = [lut[x] for x in diff_region_list]
color_region = [(region_encoding[key],val) for key, val in lut.items()]

if len(dat.region_names) > 1:
    region_selections = [[0],[1],[0,1]]
else:
    region_selections = [[0]]

for this_selection in region_selections:
    region_inds = [num for num,val in enumerate(diff_region_list) \
            if val in this_selection]
    region_names = [region_encoding[x] for x in this_selection]

    if len(region_inds) < 2:
        open(os.path.join(plot_dir,
            f'{"_".join(region_names)} doesnt have enough nrns'),'w')
    else:
        sns.clustermap(mean_diff_dists_array[region_inds],
            row_colors = np.array(row_colors)[region_inds], col_cluster=False)
        plt.subplots_adjust(0.1,0.1, top = 0.9)
        plt.suptitle(f'Raw dists : {"_".join(region_names)}' + \
                '\n' + str(color_region))
        fig = plt.gcf()
        fig.savefig(os.path.join(plot_dir,f'raw_dists_{"_".join(region_names)}'))
        #plt.show()
        sns.clustermap(zscore_diff_dists_array[region_inds],
            row_colors = np.array(row_colors)[region_inds], col_cluster=False)
        plt.subplots_adjust(0.1,0.1, top = 0.9)
        plt.suptitle(f'Zscore dists : {"_".join(region_names)}' + \
                '\n' + str(color_region))
        fig = plt.gcf()
        fig.savefig(os.path.join(
            plot_dir,f'zscore_dists_{"_".join(region_names)}'))
        #plt.show()

plt.close('all')

#################################################################################
# ____  _               _           _             _     _ _ _ _         
#|  _ \(_)___  ___ _ __(_)_ __ ___ (_)_ __   __ _| |__ (_) (_) |_ _   _ 
#| | | | / __|/ __| '__| | '_ ` _ \| | '_ \ / _` | '_ \| | | | __| | | |
#| |_| | \__ \ (__| |  | | | | | | | | | | | (_| | |_) | | | | |_| |_| |
#|____/|_|___/\___|_|  |_|_| |_| |_|_|_| |_|\__,_|_.__/|_|_|_|\__|\__, |
#                                                                 |___/ 
#################################################################################

dat.firing_rate_params = dat.default_firing_params
dat.firing_rate_params['step_size'] = 50
dat.firing_rate_params['window_size'] = 500
dat.get_firing_rates()

firing_array = np.moveaxis(dat.firing_array.swapaxes(0,1),-1,1)

p_val_array = np.empty(firing_array.shape[:2])
inds = np.array(list(np.ndindex(firing_array.shape[:2])))
for this_ind in inds:
    p_val_array[this_ind] = kruskal(*firing_array[this_ind])[1]

stim_t = 2000
t_vec = np.linspace(0,spikes.shape[-1],firing_array.shape[1]) - stim_t

save_path = '/ancillary_analysis'
group_name = 'taste_discriminability'
fin_save_path = os.path.join(save_path, group_name)
with tables.open_file(dat.hdf5_path,'r+') as h5:
    if fin_save_path not in h5:
        h5.create_group(save_path, group_name)
    h5.create_array(fin_save_path, 'taste_discrim_kw', p_val_array)
    h5.create_array(fin_save_path, 'post_stim_time', t_vec)
    h5.flush()

#taste_discrim_frame = pd.DataFrame({
#        'neuron' : inds[:,0],
#        'time_bin' : inds[:,1],
#        'post_stim_t' : np.tile(t_vec, p_val_array.shape[0]),
#        'p_vals' : p_val_array.flatten()})
#
#taste_discrim_frame = \
#        taste_discrim_frame.merge(
#                nrn_ind_frame[['neuron','region']].drop_duplicates(),
#                how = 'inner', on = 'neuron')
#sns.relplot(x = 'post_stim_t', y = 'p_vals', hue = 'neuron', 
#        data = taste_discrim_frame, col = 'region', kind = 'line')
#plt.show()

alpha = 0.05
p_val_sig_array = p_val_array <= alpha

post_t_lims = [-500,2000]
wanted_inds = np.array([num for num,x in enumerate(t_vec) \
        if post_t_lims[0] < x <= post_t_lims[1]])
cut_p_val_sig_array = p_val_sig_array[...,wanted_inds]
sorted_sig_arrays = [cut_p_val_sig_array[units] for units in dat.region_units]
fig,ax = plt.subplots(2,len(dat.region_names), sharex=True)
for num,this_array in enumerate(sorted_sig_arrays):
    ax[0,num].pcolormesh(
            t_vec[wanted_inds], np.arange(this_array.shape[0]), this_array)
    ax[1,num].plot(t_vec[wanted_inds], np.mean(this_array,axis=0))
    ax[1,num].set_ylim([0,1])
    ax[0,num].set_title(dat.region_names[num])
plt.suptitle(dat.hdf5_name + '\n' + f'Taste ANOVA, alpha = {alpha}')
plt.subplots_adjust(0.1,0.1, top = 0.8)
fig.savefig(os.path.join(plot_dir),'discrim_region_plot')
#plt.show()
