#############################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
#############################

# Import modules

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel,delayed
import pingouin
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import zscore

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

# Get list of relevant files
# Ask user for all relevant files
# If file_list already exists then ask the user if they want to use it
data_folder = '/media/bigdata/firing_space_plot/don_grant_figs'
log_file_name = os.path.join(data_folder, 'file_list.txt')
if os.path.exists(log_file_name):
    old_list = open(log_file_name,'r').readlines()
    old_list = [x.rstrip() for x in old_list]
    old_basename_list = [os.path.basename(x) for x in old_list]
    old_bool = easygui.ynbox(msg = "Should this file list be used: \n\n Old list: \n\n{}"\
            .format("\n".join(old_basename_list), 
            title = "Save these files?"))
else:
    print('Sucks...cant do anything else right now :p')

plot_dir = os.path.join(data_folder, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

taste_stat_list = []
laser_stat_list = []
interaction_stat_list = []
region_list = []

# Bins for ANOVA
time_lims = (2000,4500)
bin_width = 250
fs = 1000

for file_num in trange(len(old_list)):

    # Extract data
    dat = \
        ephys_data(os.path.dirname(old_list[file_num]))
    dat.firing_rate_params = dict(zip(\
        ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
        ('conv',1,250,1,1e-3,1e-3)))

    dat.get_unit_descriptors()
    dat.get_spikes()
    dat.separate_laser_spikes()
    #dat.get_firing_rates()
    #dat.separate_laser_firing()

    dat.get_hdf5_name()
    file_str_parts = os.path.basename(dat.hdf5_name).split('_')
    file_iden = '_'.join([file_str_parts[0], file_str_parts[2]])

    on_spikes = dat.on_spikes
    off_spikes = dat.off_spikes
    #on_firing = dat.on_firing
    #off_firing = dat.off_firing

    # Bin firing into larger bins for anova

    binned_on_firing = np.sum(np.reshape(on_spikes[...,time_lims[0]:time_lims[1]],
                        (*on_spikes.shape[:3],-1,bin_width)),axis=-1)/ (bin_width/fs)
    binned_off_firing = np.sum(np.reshape(off_spikes[...,time_lims[0]:time_lims[1]],
                        (*off_spikes.shape[:3],-1,bin_width)),axis=-1)/ (bin_width/fs)
    
    # Instead of directly calculating binned firing rate from spikes
    # Average moving window firing rate into bins
    #binned_on_firing = np.mean(np.reshape(on_firing[...,time_lims[0]:time_lims[1]],
    #                    (*on_firing.shape[:3],-1,bin_width)),axis=-1)
    #binned_off_firing = np.mean(np.reshape(off_firing[...,time_lims[0]:time_lims[1]],
    #                    (*off_firing.shape[:3],-1,bin_width)),axis=-1)

    #on_firing_long = on_firing.reshape(\
    #        (-1, *on_firing.shape[-2:]))
    #binned_on_firing_long = binned_on_firing.reshape(\
    #        (-1, *binned_on_firing.shape[-2:]))
    #visualize.firing_overview(binned_on_firing_long.swapaxes(0,1));plt.show()

    t_vec = np.arange(on_spikes.shape[-1])
    binned_t_vec = np.median(
            np.reshape(t_vec[time_lims[0]:time_lims[1]],(-1,bin_width)),axis=-1)

    dim_inds = np.stack(list(np.ndindex(binned_on_firing.shape))).T
    all_electrode_numbers = [x['electrode_number'] for x in dat.unit_descriptors]

    firing_frame_list = [pd.DataFrame(\
            data = {'taste' : dim_inds[0],
                    'trial' : dim_inds[1],
                    'neuron' : dim_inds[2],
                    'bin' : dim_inds[3],
                    'time' : binned_t_vec[dim_inds[3]],
                    'laser' : num,
                    'firing' : this_firing.flatten()}) \
                            for num,this_firing in \
                            enumerate(
                                list((binned_off_firing,binned_on_firing)))]

    firing_frame = pd.concat(firing_frame_list)

    # Plot firing to make sure everything looks good
    # Plot all discriminative neurons
    #g = sns.FacetGrid(data = firing_frame.loc[firing_frame.neuron < 10],
    #            col = 'neuron', row = 'taste', hue = 'laser', sharey = 'col')
    #g.map(sns.pointplot, 'time', 'firing')
    #g.set_xticklabels(plt.xticks()[1],rotation = 45, horizontalalignment = 'right')
    #fig = plt.gcf()
    #plt.tight_layout()
    #fig.savefig(os.path.join(plot_dir,'{}_all_nrns'.\
    #        format(file_iden)))

    #visualize.firing_overview(dat.all_normalized_firing)


    # Perform ANOVA on each neuron individually
    # 3 Way ANOVA : Taste, laser, time
    anova_list = [
        firing_frame.loc[firing_frame.neuron == nrn,:]\
                .anova(dv = 'firing', \
                 between = ['taste','laser','bin'])\
                for nrn in tqdm(firing_frame.neuron.unique())]

    anova_p_list = [x[['Source','p-unc']][:4] for x in anova_list]
    p_val_tresh = 0.05 #/ len(firing_frame.bin.unique()) 
    sig_taste = [True if x['p-unc'][0]<p_val_tresh else False for x in anova_p_list]
    sig_laser = [True if x['p-unc'][1]<p_val_tresh else False for x in anova_p_list]
    port_a = np.arange(32)
    sig_region = [True if unit in port_a else False for unit in all_electrode_numbers] 
    sig_interaction = [True if x['p-unc'][3]<p_val_tresh else False for x in anova_p_list]

    gc_taste_laser = np.array((sig_taste,sig_laser,sig_region)).sum(axis=0) == 3


    #if sum(gc_taste_laser) > 0:
    #    g = sns.FacetGrid(\
    #            data = firing_frame.loc\
    #                [firing_frame.neuron.isin(np.where(gc_taste_laser)[0])],
    #            col = 'neuron', row = 'taste', hue = 'laser', sharey = 'col')
    #    g.map(sns.pointplot, 'time', 'firing')
    #    g.set_xticklabels(plt.xticks()[1],rotation = 45, horizontalalignment = 'right')
    #    fig = plt.gcf()
    #    plt.tight_layout()
    #    fig.savefig(os.path.join(plot_dir,'{}_mean_firing'.\
    #            format(file_iden)))

    taste_stat_list.append(sig_taste)
    laser_stat_list.append(sig_laser)
    interaction_stat_list.append(sig_interaction)
    region_list.append(sig_region)
plt.close('all')

# Doing it in 2 loops because I'm lazy :P

# Find all GC neurons with laser effects
gc_laser = [np.array(laser)*np.array(region) \
        for laser,region in zip(laser_stat_list,region_list)]
# Separate out taste_discriminative 
gc_laser_taste = [np.array(laser)*np.array(taste) \
        for laser,taste in zip(gc_laser,taste_stat_list)]
gc_interaction = [np.array(interaction)*np.array(region) \
        for interaction,region in zip(interaction_stat_list,region_list)]

# Find union of all these lists
fin_gc_nrns = [[any(x) for x in zip(*nrn)] \
        for nrn in zip(gc_laser, gc_laser_taste, gc_interaction)]

#########################
# ____  _       _       
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
#                       
#########################

# Loop through files again and pull out rasters and PSTHs for 
# GC Taste Responsive neurons which were affected by laser


lower_time_lim = 1500
for file_num in trange(len(old_list)):

    # Extract data
    dat = \
        ephys_data(os.path.dirname(old_list[file_num]))
    dat.firing_rate_params = dict(zip(\
        ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
        ('conv',1,250,1,1e-3,1e-3)))
    dat.get_hdf5_name()
    file_str_parts = os.path.basename(dat.hdf5_name).split('_')
    file_iden = '_'.join([file_str_parts[0], file_str_parts[2]])

    dat.get_unit_descriptors()
    dat.get_spikes()
    dat.separate_laser_spikes()

    dat.get_firing_rates()
    dat.separate_laser_firing()

    if sum(fin_gc_nrns[file_num])>0:

        on_spikes = dat.on_spikes[:,:,fin_gc_nrns[file_num]]
        off_spikes = dat.off_spikes[:,:,fin_gc_nrns[file_num]]
        on_firing = dat.on_firing[:,:,fin_gc_nrns[file_num]]
        off_firing = dat.off_firing[:,:,fin_gc_nrns[file_num]]

        t_vec = np.arange(off_firing.shape[-1])*dat.firing_rate_params['step_size']
        time_inds = (lower_time_lim < t_vec) * (t_vec < time_lims[1])
        t_vec = t_vec[time_inds]

        on_spikes = on_spikes[..., lower_time_lim:time_lims[1]]
        off_spikes = off_spikes[..., lower_time_lim:time_lims[1]]
        on_firing = on_firing[..., time_inds]
        off_firing = off_firing[..., time_inds]

        # Plot PSTHs as 2 columns
        for nrn_num in range(off_spikes.shape[2]):

            # Raster plots
            fig, ax = plt.subplots(off_firing.shape[0],1, sharex=True, sharey=True)
            for taste in range(off_firing.shape[0]):
                spike_inds = np.where(\
                        np.concatenate(\
                            (off_spikes[taste,:,nrn_num], on_spikes[taste,:,nrn_num])))
                ax[taste].scatter(np.arange(off_spikes.shape[-1])[spike_inds[1]], 
                        spike_inds[0], 
                        marker = '.', alpha = 0.5, color = 'k') 
            fig.set_size_inches(6,10)
            plt.suptitle(file_iden + '\n nrn # {}'.format(nrn_num))
            fig.savefig(os.path.join(\
                    plot_dir,'{}_nrn{}_raster'\
                    .format(file_iden,nrn_num)))

            # Firing rate for all trials
            fig, ax = plt.subplots(off_firing.shape[0],1, sharex=True, sharey=True)
            for taste in range(off_firing.shape[0]):
                concat_firng_rate = np.concatenate(\
                            (off_firing[taste,:,nrn_num], on_firing[taste,:,nrn_num]))
                plt.sca(ax[taste])
                visualize.imshow(concat_firng_rate)
            fig.set_size_inches(6,10)
            plt.suptitle(file_iden + '\n nrn # {}'.format(nrn_num))
            fig.savefig(os.path.join(\
                    plot_dir,'{}_nrn{}_all_firing'\
                    .format(file_iden,nrn_num)))

            # Firing rate
            fig, ax = plt.subplots(off_firing.shape[0],1, sharex=True, sharey=True)
            for taste in range(off_firing.shape[0]):
                mean_off = np.mean(off_firing[taste,:,nrn_num],axis=0)
                std_off = np.std(off_firing[taste,:,nrn_num],axis=0)
                mean_on = np.mean(on_firing[taste,:,nrn_num],axis=0)
                std_on = np.std(on_firing[taste,:,nrn_num],axis=0)
                ax[taste].fill_between(x=t_vec,
                                        y1 = mean_off - 2*std_off,
                                        y2 = mean_off + 2*std_off,
                                        alpha = 0.5)
                ax[taste].plot(t_vec, mean_off)
                ax[taste].fill_between(x=t_vec,
                                        y1 = mean_on - 2*std_on,
                                        y2 = mean_on + 2*std_on,
                                        alpha = 0.5)
                ax[taste].plot(t_vec, mean_on)
                #ax[taste,1].plot(np.mean(on_firing[taste,:,nrn_num],axis=0))
            fig.set_size_inches(6,10)
            plt.suptitle(file_iden + '\n nrn # {}'.format(nrn_num))
            fig.savefig(os.path.join(plot_dir,'{}_nrn{}_mean_firing'.\
                    format(file_iden,nrn_num)))

            # Collage of mean firing rate and raster
            def gauss_kern(size):
                x = np.arange(-size,size+1)
                kern = np.exp(-(x**2)/float(size))
                return kern / sum(kern)
            def gauss_filt(vector, size):
                kern = gauss_kern(size)
                return np.convolve(vector, kern, mode='same')

            taste_labels = ['NaCl','Sucrose','Citric Acid','Quinine']
            fig, ax = plt.subplots(off_firing.shape[0],2, 
                    sharex=False, sharey=False)
            ax = ax.flatten()
            line_inds = [0,1,4,5]
            raster_inds = [2,3,6,7]
            trial_tick_num = 3
            laser_time = [2000,4500]
            laser_trials = [0,15]
            stim_t = 2000
            plot_t_vec = t_vec - 2000
            for taste in range(off_firing.shape[0]):
                mean_off = np.mean(off_firing[taste,:,nrn_num],axis=0)
                mean_on = np.mean(on_firing[taste,:,nrn_num],axis=0)
                ax[line_inds[taste]].plot(plot_t_vec, gauss_filt(mean_off,250),
                        '--', linewidth = 3)
                ax[line_inds[taste]].plot(plot_t_vec, gauss_filt(mean_on, 250), 
                                color = 'lime', linewidth = 3)
                ax[line_inds[taste]].set_ylabel('Firing Rate (Hz)')
                ax[line_inds[taste]].set_title(taste_labels[taste])
                ## Sort trials weighted by time-bins with biggest difference
                ## AFTER stim_t in the mean firing rate
                #mean_firing_diff = np.abs(mean_off - mean_on)
                #mean_firing_diff[t_vec < stim_t] = 0
                this_off_spikes = off_spikes[taste,:,nrn_num]
                this_on_spikes = on_spikes[taste,:,nrn_num]
                #weighted_off_spikes = this_off_spikes[...,:-1]*mean_firing_diff
                #weighted_on_spikes = this_on_spikes[...,:-1]*mean_firing_diff
                ## Off spikes in increasing and on in decreasing order
                #off_trial_order = np.argsort(np.mean(weighted_off_spikes,axis=-1))
                #on_trial_order = np.argsort(np.mean(weighted_on_spikes,axis=-1))[::-1]
                #ordered_off_spikes = this_off_spikes[off_trial_order]
                #ordered_on_spikes = this_on_spikes[on_trial_order]
                spike_inds = np.where(\
                        np.concatenate(\
                            (this_off_spikes, this_on_spikes)))
                            #(ordered_off_spikes, ordered_on_spikes)))
                tick_inds = np.linspace(0,np.max(spike_inds[0]+1),trial_tick_num)
                ax[raster_inds[taste]].axvspan(laser_time[0]-stim_t, laser_time[1]-stim_t,
                        0, 0.5,
                        facecolor='lime', alpha =0.5)
                ax[raster_inds[taste]].scatter(
                        lower_time_lim - stim_t + \
                                np.arange(off_spikes.shape[-1])[spike_inds[1]], 
                        np.max(spike_inds[0]) - spike_inds[0], 
                        marker = '.', alpha = 1, color = 'k') 
                ax[raster_inds[taste]].set_yticks(list(map(int,tick_inds)))
                ax[raster_inds[taste]].set_xlabel('Time post-stimulus delivery (ms)')
                ax[raster_inds[taste]].set_ylabel('Trial #')
            for this_ax in ax:
                this_ax.spines['top'].set_visible(False)
                this_ax.spines['right'].set_visible(False)
            ax[0].get_shared_y_axes().join(*ax[line_inds])
            #for this_ax in ax[line_inds]:
            #    this_ax.autoscale()
            #    this_ax.set_xticklabels([])
            fig.set_size_inches(12,8)
            plt.suptitle(file_iden + '\n nrn # {}'.format(nrn_num))
            plt.tight_layout()
            plt.subplots_adjust(top = 0.85)
            fig.savefig(os.path.join(plot_dir,'firing_raster',
                    '{}_nrn{}_firing_raster'.\
                    format(file_iden,nrn_num)))
            
            plt.close('all')

