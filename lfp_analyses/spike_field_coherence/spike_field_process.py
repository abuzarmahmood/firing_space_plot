## Import required modules
import os
import matplotlib.pyplot as plt
import tables
import numpy as np
from tqdm import tqdm, trange
import shutil
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import zscore
from pathlib import Path
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import xarray as xr
import seaborn as sns
import pingouin as pg

############################################################ 
## Initialization
############################################################ 
##################################################
## Define functions
##################################################

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

##################################################
## Read in data 
##################################################
dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
#file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 

############################################################
## explore firing rates as these affect inference of coherence
############################################################
def bin_spikes(spikes, bin_size = 100):
    temp_spikes = np.reshape(spikes, (*spikes.shape[:3],-1,bin_size))
    temp_spikes = np.sum(temp_spikes, axis = -1)
    return temp_spikes

# Get mean firing rate before and after stimulus
binned_spikes_list = []
region_names = ['bla','gc']
for this_dir in tqdm(dir_list):
    #this_dir = dir_list[0]
    dat = ephys_data(this_dir)
    dat.get_spikes()
    region_spikes = [dat.return_region_spikes(x) for x in region_names]
    #spikes = np.array(dat.spikes)

    # Bin into 100ms bins
    bin_size = 100
    binned_spikes = [bin_spikes(x, bin_size) for x in region_spikes]
    flat_binned_spikes = [np.reshape(x, (-1, *x.shape[2:])) for x in binned_spikes]
    #mean_bin_spikes = [np.mean(x,axis=(0,1)) for x in binned_spikes]
    #mean_rate_list.append(mean_bin_spikes)
    binned_spikes_list.append(flat_binned_spikes)

region_binned_spikes = list(zip(*binned_spikes_list))
region_binned_spikes = [np.concatenate(x, axis=1) for x in region_binned_spikes]

t_vec = np.arange(0, 7000, bin_size)
baseline_lims = [0, 1750]
stim_lims = [2000, 4000]
baseline_inds = np.logical_and(t_vec >= baseline_lims[0], t_vec < baseline_lims[1])
stim_inds = np.logical_and(t_vec >= stim_lims[0], t_vec < stim_lims[1])

mean_baseline_rates = [np.mean(x[...,baseline_inds], axis = -1) \
        for x in region_binned_spikes] 
mean_stim_rates = [np.mean(x[...,stim_inds], axis = -1) \
        for x in region_binned_spikes]

inds = [np.array(list(np.ndindex(mean_baseline_rates[i].shape))) \
        for i in range(len(mean_baseline_rates))]

rate_frames = [pd.DataFrame(
        dict(
            trial = inds[i][:,0],
            neuron = inds[i][:,1],
            baseline_rate = mean_baseline_rates[i].flatten(),
            stim_rate = mean_stim_rates[i].flatten(), 
            region = region_names[i] 
            )
        )
        for i in range(len(mean_baseline_rates))]
fin_rate_frame = pd.concat(rate_frames)

group_list = list(fin_rate_frame.groupby(['region','neuron']))
group_inds = [x[0] for x in group_list]
group_vals = [x[1] for x in group_list]

p_val_list = []
for i in trange(len(group_list)):
    p_val = pg.ttest(
            group_vals[i]['baseline_rate'], 
            group_vals[i]['stim_rate'], 
            paired = True)['p-val']
    p_val_list.append(p_val.values[0])

alpha = 0.001
sig_inds = [this_ind for this_ind, this_p in zip(group_inds, p_val_list) \
        if this_p < alpha]
region_sig_inds = [[val for region, val in sig_inds if region == this_region] \
        for this_region in region_names]
region_sig_inds = [np.array(x) for x in region_sig_inds]

wanted_groups = [this_dat for this_ind, this_dat in zip(group_inds, group_vals) \
        if this_ind in sig_inds]
wanted_dat_frame = pd.concat(wanted_groups)

mean_wanted_dat = wanted_dat_frame.groupby(['region','neuron']).mean()
mean_wanted_dat = mean_wanted_dat.reset_index()
mean_wanted_dat.drop(columns = ['trial', 'neuron'], inplace = True)

rate_frame = mean_wanted_dat.melt(
        id_vars = ['region'], 
        var_name = 'stim_type',
        value_name = 'rate')

# region_mean_rates = list(zip(*mean_rate_list))
# region_mean_rates = [np.concatenate(x) for x in region_mean_rates]

sns.violinplot(
        data = rate_frame,
        x = 'region',
        y = 'rate',
        hue = 'stim_type',
        alpha = 0.5,
        split=True)
plt.xlabel('Region')
plt.ylabel(f'Spikes / {bin_size}ms')
plt.gca().set_xticklabels([x.upper() for x in region_names])
plt.show()

fig, ax = plt.subplots(2,1, sharex=True) 
for ax_ind, this_ax in enumerate(ax):
    #this_ax.imshow(region_mean_rates[ax_ind], aspect = 'auto')
    raw_dat = region_mean_rates[ax_ind]
    sig_dat = raw_dat[region_sig_inds[ax_ind]]
    this_ax.imshow(zscore(sig_dat, axis=-1), aspect = 'auto')
    this_ax.set_title(region_names[ax_ind])
plt.show()

fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
for ax_ind, this_ax in enumerate(ax):
    raw_dat = region_mean_rates[ax_ind]
    sig_dat = raw_dat[region_sig_inds[ax_ind]]
    for this_dat in sig_dat:
        this_ax.plot(np.log10(this_dat), alpha = 0.1, color = 'k')
    this_ax.set_title(region_names[ax_ind])
plt.show()


############################################################
## Calculate spike-field coherence 
############################################################
for this_dir in tqdm(dir_list):
    #this_dir = dir_list[0]
    dat = ephys_data(this_dir)
    dat.stft_params['max_freq'] = 100
    dat.get_stft(recalculate = False, dat_type = ['phase'])
    dat.get_lfp_electrodes()

    #stft_array = dat.stft_array
    #stft_array = np.concatenate(np.swapaxes(stft_array, 1,2))
    phase_array = dat.phase_array
    phase_array = np.concatenate(np.swapaxes(phase_array, 1,2))
    time_vec = dat.time_vec
    freq_vec = dat.freq_vec
    region_electrodes = dat.lfp_region_electrodes

    # Get spikes
    dat.get_region_units()
    region_sort_order = np.argsort(dat.region_names)
    sorted_region_names = np.array(dat.region_names)[region_sort_order]
    region_spikes = [dat.return_region_spikes(x) for x in sorted_region_names]
    long_region_spikes = [np.concatenate(x) for x in region_spikes]

    # Find channel closest to mean phase
    select_region_electrodes = []
    for region in region_electrodes:
        #region = region_electrodes[0]
        this_region_phase = phase_array[:,region] 
        region_mean_phase = np.mean(this_region_phase,axis=1)
        phase_abs_diff = np.sum(
                np.abs(this_region_phase - region_mean_phase[:, np.newaxis]),
                axis = (0,2,3))
        select_ind = np.argmin(phase_abs_diff)
        select_region_electrodes.append(region[select_ind])

    min_err_phase = phase_array[:, np.array(select_region_electrodes)].swapaxes(0,1)
    # Sort by region
    min_err_phase = min_err_phase[region_sort_order]
    del phase_array

    ############################################################ 
    ## Phase Locking
    ############################################################ 
    # Phase locking of a region's spikes to LFP of the other region
    # This would suggest if the activity in a region is contributing to inputs in the other region

    phase_list = []
    for region_ind in range(len(sorted_region_names)):
        #region_ind = 0
        this_phase = np.swapaxes(min_err_phase[region_ind],1,-1)
        phase_region = sorted_region_names[region_ind]
        this_spikes = long_region_spikes[1-region_ind]
        spikes_region = sorted_region_names[1-region_ind]
        # Cut to same length
        this_spikes = np.swapaxes(this_spikes[...,:this_phase.shape[1]],0,1)
        #for nrn_num, this_nrn in tqdm(enumerate(this_spikes)):
        for nrn_num, this_nrn in enumerate(this_spikes):
            if this_nrn.sum(axis=None):
                inds = np.where(this_nrn)
                phases = this_phase[inds]
                phase_inds = np.array(list(np.ndindex(phases.shape)))
                phase_frame = pd.DataFrame(
                        dict(
                           spikes_region = spikes_region,
                           phase_region = phase_region,
                           nrn_num = nrn_num,
                           trials = inds[0][phase_inds[:,0]],
                           time = inds[1][phase_inds[:,0]],
                           freq = phase_inds[:,1],
                           phases = phases.flatten()
                            )
                        )
                phase_frame = phase_frame.set_index(
                        ['spikes_region','phase_region','nrn_num','trials','time','freq'])
                phase_list.append(phase_frame)
    fin_phase_frame = pd.concat(phase_list)

    hf5_save_path = '/stft/analyses/spike_phase_coherence/actual'
    fin_phase_frame.to_hdf(dat.hdf5_path, hf5_save_path) 

    ##############################
    # Also generate shuffled data
    ##############################
    n_shuffles = 100
    for shuffle_num in tqdm(range(n_shuffles)):
        shuffled_phase_list = []
        for region_ind in range(len(sorted_region_names)):
            #region_ind = 0
            this_phase = np.swapaxes(min_err_phase[region_ind],1,-1)
            # Shuffle trials
            this_phase = np.random.permutation(this_phase)
            phase_region = sorted_region_names[region_ind]
            this_spikes = long_region_spikes[1-region_ind]
            spikes_region = sorted_region_names[1-region_ind]
            # Cut to same length
            this_spikes = np.swapaxes(this_spikes[...,:this_phase.shape[1]],0,1)
            #for nrn_num, this_nrn in tqdm(enumerate(this_spikes)):
            for nrn_num, this_nrn in enumerate(this_spikes):
                if this_nrn.sum(axis=None):
                    inds = np.where(this_nrn)
                    phases = this_phase[inds]
                    phase_inds = np.array(list(np.ndindex(phases.shape)))
                    phase_frame = pd.DataFrame(
                            dict(
                               spikes_region = spikes_region,
                               phase_region = phase_region,
                               nrn_num = nrn_num,
                               trials = inds[0][phase_inds[:,0]],
                               time = inds[1][phase_inds[:,0]],
                               freq = phase_inds[:,1],
                               phases = phases.flatten()
                                )
                            )
                    phase_frame = phase_frame.set_index(
                            ['spikes_region',
                             'phase_region',
                             'nrn_num',
                             'trials',
                             'time',
                             'freq'])
                    shuffled_phase_list.append(phase_frame)
        fin_phase_frame = pd.concat(phase_list)

        hf5_save_path = '/stft/analyses/spike_phase_coherence/shuffled/' +\
                'sh_' + str(shuffle_num)
        fin_phase_frame.to_hdf(dat.hdf5_path, hf5_save_path) 
