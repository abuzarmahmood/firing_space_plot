## Import required modules
import glob
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
import json
import pingouin
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from sklearn.decomposition import PCA as pca
from scipy.stats import zscore
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Create file to store data + analyses in
file_list_path = '/media/bigdata/firing_space_plot/don_grant_figs/laser_files.txt'
plot_dir = os.path.join(os.path.dirname(file_list_path),'plots2')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

file_list = open(file_list_path,'r').readlines()
file_list = [x.rstrip() for x in file_list]
basename_list = [os.path.basename(x) for x in file_list]
name_date_str_list = ['_'.join([x.split('_')[0],x.split('_')[2]]) \
        for x in basename_list]
dirname_list = [os.path.dirname(x) for x in file_list]
json_path_list = [glob.glob(dirname+'/*json')[-1] for dirname in dirname_list]

# Find all electrodes corresponding to BLA and GC using the jsons
json_list = [json.load(open(file,'r')) for file in json_path_list]
bla_electrodes = [x['regions']['bla'] for x in json_list]
gc_electrodes = [x['regions']['gc'] for x in json_list]

all_on_firing = []
all_off_firing = []
all_on_spikes = []
all_off_spikes = []
all_unit_descriptors = []
for file_num, file_name in tqdm(enumerate(file_list)):
    data = ephys_data(data_dir = dirname_list[file_num]) 
    data.firing_rate_params = dict(zip(\
        ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
        ('conv',25,250,1,25e-3,1e-3)))
    data.get_unit_descriptors()
    data.get_spikes()
    data.get_firing_rates()
    data.separate_laser_firing()
    data.separate_laser_spikes()
    all_on_firing.append(data.on_firing)
    all_off_firing.append(data.off_firing)
    all_on_spikes.append(data.on_spikes)
    all_off_spikes.append(data.off_spikes)
    all_unit_descriptors.append(data.unit_descriptors)

# Remove cases where size of firing array doesn't match with
# unit descriptors
# WTF WOULD THAT HAPPEN THOUGH???

all_unit_electrodes = [x["electrode_number"] for x in all_unit_descriptors]
unit_count_matches = [num for num, (firing, electrodes) in \
        enumerate(zip(all_on_firing, all_unit_electrodes)) \
        if firing.shape[2] == len(electrodes)]

# Also check that all files have waveform_count in unit_descriptor
# and remove ones which don't. This is to ensure we only use quality
# units in analysis
waveform_count_exists = [num for num,x in enumerate(all_unit_descriptors) \
        if 'waveform_count' in x.dtype.names]

matching_inds = list(set(waveform_count_exists).\
        intersection(set(unit_count_matches)))

fin_filenames = [name_date_str_list[x] for x in matching_inds]
fin_on_firing = [all_on_firing[x] for x in matching_inds]
fin_off_firing = [all_off_firing[x] for x in matching_inds]
fin_on_spikes = [all_on_spikes[x] for x in matching_inds]
fin_off_spikes = [all_off_spikes[x] for x in matching_inds]
fin_unit_electrodes = [all_unit_electrodes[x] for x in matching_inds]
fin_gc_electrodes = [gc_electrodes[x] for x in matching_inds]
fin_bla_electrodes = [bla_electrodes[x] for x in matching_inds]
fin_unit_descriptor = [all_unit_descriptors[x] for x in matching_inds]

fin_waveform_counts = [x['waveform_count'] for x in fin_unit_descriptor]
waveform_tresh = 1500

gc_unit_inds = [np.where(np.isin(unit_electrodes, this_gc_electrodes))[-1]\
        for unit_electrodes, this_gc_electrodes \
        in zip(fin_unit_electrodes, fin_gc_electrodes)]
bla_unit_inds = [np.where(np.isin(unit_electrodes, this_bla_electrodes))[-1]\
        for unit_electrodes, this_bla_electrodes \
        in zip(fin_unit_electrodes, fin_bla_electrodes)]
gc_nrn_count = [len(x) for x in gc_unit_inds]
bla_nrn_count = [len(x) for x in bla_unit_inds]

gc_waveform_counts = np.concatenate([counts[inds] for counts,inds in \
        zip(fin_waveform_counts, gc_unit_inds)])
bla_waveform_counts = np.concatenate([counts[inds] for counts,inds in \
        zip(fin_waveform_counts, bla_unit_inds)])
gc_on_list = [firing[:,:,inds] for firing,inds in zip(fin_on_firing, gc_unit_inds)]
gc_off_list = [firing[:,:,inds] for firing,inds in zip(fin_off_firing, gc_unit_inds)]
bla_on_list = [firing[:,:,inds] for firing,inds in zip(fin_on_firing, bla_unit_inds)]
bla_off_list = [firing[:,:,inds] for firing,inds in zip(fin_off_firing, bla_unit_inds)]
gc_on_spikes_list = [spikes[:,:,inds] for spikes,inds in zip(fin_on_spikes, gc_unit_inds)]
gc_off_spikes_list = [spikes[:,:,inds] \
        for spikes,inds in zip(fin_off_spikes, gc_unit_inds)]
bla_on_spikes_list = [spikes[:,:,inds] \
        for spikes,inds in zip(fin_on_spikes, bla_unit_inds)]
bla_off_spikes_list = [spikes[:,:,inds] \
        for spikes,inds in zip(fin_off_spikes, bla_unit_inds)]

gc_on_array = np.moveaxis(np.concatenate(gc_on_list, axis = 2),2,0)
gc_off_array = np.moveaxis(np.concatenate(gc_off_list, axis = 2),2,0)
bla_on_array = np.moveaxis(np.concatenate(bla_on_list, axis = 2),2,0)
bla_off_array = np.moveaxis(np.concatenate(bla_off_list, axis = 2),2,0)
gc_on_spikes_array = np.moveaxis(np.concatenate(gc_on_spikes_list, axis = 2),2,0)
gc_off_spikes_array = np.moveaxis(np.concatenate(gc_off_spikes_list, axis = 2),2,0)
bla_on_spikes_array = np.moveaxis(np.concatenate(bla_on_spikes_list, axis = 2),2,0)
bla_off_spikes_array = np.moveaxis(np.concatenate(bla_off_spikes_list, axis = 2),2,0)


time_lims = (2000,4500)
time_inds = np.arange(time_lims[0],time_lims[1])
step_size = data.firing_rate_params['step_size']
bin_lims = [x//step_size for x in time_lims]
bin_inds = np.arange(bin_lims[0],bin_lims[1])
gc_firing = np.stack([gc_on_array, gc_off_array])
bla_firing = np.stack([bla_on_array, bla_off_array])
#gc_firing = gc_firing[...,bin_lims[0]:bin_lims[1]]
#bla_firing = bla_firing[...,bin_lims[0]:bin_lims[1]]
gc_spikes = np.stack([gc_on_spikes_array, gc_off_spikes_array])
bla_spikes = np.stack([bla_on_spikes_array, bla_off_spikes_array])
#gc_spikes = gc_spikes[...,time_lims[0]:time_lims[1]]
#bla_spikes = bla_spikes[...,time_lims[0]:time_lims[1]]

# Remove neurons with waveform counts below threshold
gc_firing= gc_firing[:,gc_waveform_counts > waveform_tresh]
bla_firing= bla_firing[:,bla_waveform_counts > waveform_tresh]
gc_spikes= gc_spikes[:,gc_waveform_counts > waveform_tresh]
bla_spikes= bla_spikes[:,bla_waveform_counts > waveform_tresh]

# Use separate array for spikes fed to DataFrame
# Otherwise making the indices becomes too taxing
gc_frame_spikes = gc_spikes[...,time_inds]
bla_frame_spikes = bla_spikes[...,time_inds]

# Convert to pandas DataFrames
gc_dim_inds = np.stack(np.ndindex(gc_frame_spikes.shape)).T
bla_dim_inds = np.stack(np.ndindex(bla_frame_spikes.shape)).T

gc_firing_frame = pd.DataFrame(\
        data = {'laser' : gc_dim_inds[0],
                'neuron' : gc_dim_inds[1],
                'taste' : gc_dim_inds[2],
                'trial' : gc_dim_inds[3],
                'time' : gc_dim_inds[4],
                'firing' : gc_frame_spikes.flatten()}) \

bla_firing_frame = pd.DataFrame(\
        data = {'laser' : bla_dim_inds[0],
                'neuron' : bla_dim_inds[1],
                'taste' : bla_dim_inds[2],
                'trial' : bla_dim_inds[3],
                'time' : bla_dim_inds[4],
                'firing' : bla_frame_spikes.flatten()}) \

# Bin firing into larger bins for anova
bin_width = 250
#bin_width_in_inds = bin_width // step_size
#fin_bin_count = len(bin_inds)//bin_width_in_inds
fin_bin_count = len(time_inds)//bin_width

gc_bin_firing_frame = gc_firing_frame.copy()
gc_bin_firing_frame['bin'] = pd.cut(gc_firing_frame.time,
        bins = fin_bin_count, include_lowest = True, 
        labels = np.arange(fin_bin_count))

bla_bin_firing_frame = bla_firing_frame.copy()
bla_bin_firing_frame['bin'] = pd.cut(bla_firing_frame.time,
        bins = fin_bin_count, include_lowest = True, 
        labels = np.arange(fin_bin_count))

# Fix things up after binning
gc_bin_firing_frame.drop('time',inplace=True,axis=1)
gc_bin_firing_frame = \
gc_bin_firing_frame.groupby(['laser','neuron','taste','trial','bin'])\
                .mean().reset_index()

bla_bin_firing_frame.drop('time',inplace=True,axis=1)
bla_bin_firing_frame = \
bla_bin_firing_frame.groupby(['laser','neuron','taste','trial','bin'])\
                .mean().reset_index()

##################################################
# ____  _             _        _   _            
#/ ___|(_)_ __   __ _| | ___  | \ | |_ __ _ __  
#\___ \| | '_ \ / _` | |/ _ \ |  \| | '__| '_ \ 
# ___) | | | | | (_| | |  __/ | |\  | |  | | | |
#|____/|_|_| |_|\__, |_|\___| |_| \_|_|  |_| |_|
#               |___/                           
##################################################

# Plot firing to make sure everything looks good
# Plot all discriminative neurons
#g = sns.FacetGrid(data = gc_bin_firing_frame.loc[gc_bin_firing_frame.neuron < 5],
#            col = 'neuron', row = 'taste', hue = 'laser', sharey = False)
#g.map(sns.pointplot, 'bin', 'firing')
#plt.show()
#
#g = sns.FacetGrid(data = bla_bin_firing_frame.loc[bla_bin_firing_frame.neuron < 5],
#            col = 'neuron', row = 'taste', hue = 'laser', sharey = False)
#g.map(sns.pointplot, 'bin', 'firing')
#plt.show()

# Perform ANOVA on each neuron individually
# 3 Way ANOVA : Taste, laser, time
#gc_anova_list = [
#    gc_bin_firing_frame.loc[gc_bin_firing_frame.neuron == nrn,:]\
#            .anova(dv = 'firing', \
#             between = ['taste','laser','bin'])\
#            for nrn in tqdm(gc_bin_firing_frame.neuron.unique())]

gc_neuron_group_list = gc_bin_firing_frame.groupby(['neuron'])
gc_anova_list = [this_dat[1].anova( dv = 'firing',
                        between = ['taste','laser','bin']) \
                for this_dat in tqdm(gc_neuron_group_list)]

bla_anova_list = [
    bla_bin_firing_frame.loc[bla_bin_firing_frame.neuron == nrn,:]\
            .anova(dv = 'firing', \
             between = ['taste','laser','bin'])\
            for nrn in tqdm(bla_bin_firing_frame.neuron.unique())]

gc_anova_p_list = [x[['Source','p-unc']][:3] for x in gc_anova_list]
bla_anova_p_list = [x[['Source','p-unc']][:3] for x in bla_anova_list]

p_val_tresh = 0.05 
gc_sig_laser = [True if x['p-unc'][1]<p_val_tresh else False for x in gc_anova_p_list]
bla_sig_laser = [True if x['p-unc'][1]<p_val_tresh else False for x in bla_anova_p_list]
gc_sig_taste = [True if x['p-unc'][0]<p_val_tresh else False for x in gc_anova_p_list]
bla_sig_taste = [True if x['p-unc'][0]<p_val_tresh else False for x in bla_anova_p_list]

##################################################
# Find how many tastes changes per neuron
##################################################
# Perform an anova per neuron, per taste
# If there is a significant difference, find average of all trials and compare
# between laser and non-laser conditions

gc_taste_group_list = gc_bin_firing_frame.groupby(['neuron','taste'])
gc_taste_anova_list = [this_dat[1].anova(dv = 'firing',
                                between = ['laser','bin'])
            for this_dat in tqdm(gc_taste_group_list)]
gc_taste_anova_p_list = [x[['Source','p-unc']][:1] for x in gc_taste_anova_list]
gc_taste_sig_laser = [True if x['p-unc'][0]<p_val_tresh else False for x in gc_taste_anova_p_list]

##################################################
# Find how many neurons showed an increase, decrease or both
##################################################

# Generate venn diagrams to visualize taste discriminatoriness
# and laser affectedness

#taste_and_laser = np.mean(1*np.array(gc_sig_laser)*np.array(gc_sig_taste))
#taste_only = np.mean(gc_sig_taste) - taste_and_laser
#laser_only = np.mean(gc_sig_laser) - taste_and_laser
v3 = venn3([set(np.arange(len(gc_anova_p_list))),
            set(np.where(gc_sig_laser)[0]),
            set(np.where(gc_sig_taste)[0])],
            set_labels = ('All neurons','Laser','Taste'))
v3_c = venn3_circles([set(np.arange(len(gc_anova_p_list))),
            set(np.where(gc_sig_laser)[0]),
            set(np.where(gc_sig_taste)[0])],
            linestyle = 'dashed')
#v3.get_label_by_id('010').set_text('{}%'.format(laser_only))
#v3.get_label_by_id('001').set_text('{}%'.format(taste_only))
#v3.get_label_by_id('011').set_text('{}%'.format(taste_and_laser))
plt.show()

v3 = venn3([set(np.arange(len(bla_anova_p_list))),
            set(np.where(bla_sig_laser)[0]),
            set(np.where(bla_sig_taste)[0])],
            set_labels = ('All neurons','Laser','Taste'))
v3_c = venn3_circles([set(np.arange(len(bla_anova_p_list))),
            set(np.where(bla_sig_laser)[0]),
            set(np.where(bla_sig_taste)[0])],
            linestyle = 'dashed')
plt.show()

#################################################
## Plots of raster and firing rate for each laser affected neuron
#################################################

# Collage of mean firing rate and raster
#def gauss_kern(size):
#    x = np.arange(-size,size+1)
#    kern = np.exp(-(x**2)/float(size))
#    return kern / sum(kern)
#def gauss_filt(vector, size):
#    kern = gauss_kern(size)
#    return np.convolve(vector, kern, mode='same')

gc_laser_firing = gc_firing[:,gc_sig_laser].swapaxes(0,1)
gc_mean_laser_firing = np.mean(gc_laser_firing, axis = 3)
gc_laser_spikes = gc_spikes[:,gc_sig_laser].swapaxes(0,1).swapaxes(1,2)
gc_laser_spikes_cat = gc_laser_spikes.reshape(\
        (*gc_laser_spikes.shape[:2], -1, gc_laser_spikes.shape[-1]))

bla_laser_firing = bla_firing[:,bla_sig_laser].swapaxes(0,1)
bla_mean_laser_firing = np.mean(bla_laser_firing, axis = 3)
bla_laser_spikes = bla_spikes[:,bla_sig_laser].swapaxes(0,1).swapaxes(1,2)
bla_laser_spikes_cat = bla_laser_spikes.reshape(\
        (*bla_laser_spikes.shape[:2], -1, bla_laser_spikes.shape[-1]))

taste_labels = ['NaCl','Sucrose','Citric Acid','Quinine']
line_inds = [0,1,4,5]
raster_inds = [2,3,6,7]
trial_tick_num = 3
laser_time = [2000,4500]
laser_trials = [0,15]
stim_t = 2000
t_vec = np.linspace(0,7000, gc_laser_firing.shape[-1])
plot_t_vec = t_vec - 2000

firing_raster_plot_dir = os.path.join(plot_dir,'firing_raster')
if not os.path.exists(firing_raster_plot_dir):
    os.makedirs(firing_raster_plot_dir)


for nrn_num in range(gc_laser_firing.shape[0]):

    fig, ax = plt.subplots(gc_firing.shape[2],2, 
            sharex=False, sharey=False)
    ax = ax.flatten()
    for taste in range(gc_firing.shape[2]):
        ax[line_inds[taste]].plot(plot_t_vec,gc_mean_laser_firing[nrn_num,1,taste],
                '--', linewidth = 3)
        ax[line_inds[taste]].plot(plot_t_vec,gc_mean_laser_firing[nrn_num,0,taste], 
                        color = 'lime', linewidth = 3)
        ax[line_inds[taste]].set_ylabel('Firing Rate (Hz)')
        ax[line_inds[taste]].set_title(taste_labels[taste])
        this_spikes = gc_laser_spikes_cat[nrn_num, taste] 
        spike_inds = np.where(this_spikes)
        tick_inds = np.linspace(0,np.max(spike_inds[0]+1),trial_tick_num)
        ax[raster_inds[taste]].axvspan(
                laser_time[0] - stim_t,
                laser_time[1] - stim_t, 
                0, 0.5,
                facecolor='lime', alpha =0.5)
        ax[raster_inds[taste]].scatter(
                spike_inds[1] - stim_t, spike_inds[0],
                marker = '.', alpha = 1, color = 'k') 
        ax[raster_inds[taste]].set_yticks(list(map(int,tick_inds)))
        ax[raster_inds[taste]].set_xlabel('Time post-stimulus delivery (ms)')
        ax[raster_inds[taste]].set_ylabel('Trial #')
    for this_ax in ax:
        this_ax.spines['top'].set_visible(False)
        this_ax.spines['right'].set_visible(False)
        this_ax.set_xlim([-1000, 2500])
    ax[0].get_shared_y_axes().join(*ax[line_inds])
    #for this_ax in ax[line_inds]:
    #    this_ax.autoscale()
    #    this_ax.set_xticklabels([])
    fig.set_size_inches(12,8)
    #plt.suptitle(file_iden + '\n nrn # {}'.format(nrn_num))
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)

    fig.savefig(os.path.join(firing_raster_plot_dir,
            'gc_nrn{}_firing_raster'.\
            format(nrn_num)))

    plt.close('all')

for nrn_num in range(bla_laser_firing.shape[0]):

    fig, ax = plt.subplots(bla_firing.shape[2],2, 
            sharex=False, sharey=False)
    ax = ax.flatten()
    for taste in range(bla_firing.shape[2]):
        ax[line_inds[taste]].plot(plot_t_vec,bla_mean_laser_firing[nrn_num,1,taste],
                '--', linewidth = 3)
        ax[line_inds[taste]].plot(plot_t_vec,bla_mean_laser_firing[nrn_num,0,taste], 
                        color = 'lime', linewidth = 3)
        ax[line_inds[taste]].set_ylabel('Firing Rate (Hz)')
        ax[line_inds[taste]].set_title(taste_labels[taste])
        this_spikes = bla_laser_spikes_cat[nrn_num, taste] 
        spike_inds = np.where(this_spikes)
        tick_inds = np.linspace(0,np.max(spike_inds[0]+1),trial_tick_num)
        ax[raster_inds[taste]].axvspan(
                laser_time[0] - stim_t,
                laser_time[1] - stim_t, 
                0, 0.5,
                facecolor='lime', alpha =0.5)
        ax[raster_inds[taste]].scatter(
                spike_inds[1] - stim_t, spike_inds[0],
                marker = '.', alpha = 1, color = 'k') 
        ax[raster_inds[taste]].set_yticks(list(map(int,tick_inds)))
        ax[raster_inds[taste]].set_xlabel('Time post-stimulus delivery (ms)')
        ax[raster_inds[taste]].set_ylabel('Trial #')
    for this_ax in ax:
        this_ax.spines['top'].set_visible(False)
        this_ax.spines['right'].set_visible(False)
        this_ax.set_xlim([-1000, 2500])
    ax[0].get_shared_y_axes().join(*ax[line_inds])
    #for this_ax in ax[line_inds]:
    #    this_ax.autoscale()
    #    this_ax.set_xticklabels([])
    fig.set_size_inches(12,8)
    #plt.suptitle(file_iden + '\n nrn # {}'.format(nrn_num))
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)

    fig.savefig(os.path.join(firing_raster_plot_dir,
            'bla_nrn{}_firing_raster'.\
            format(nrn_num)))

    plt.close('all')

###########################################################
## Find which time bins show most differences for each region
###########################################################

# Iterate over every neuron and find where differences in firing
# occur. Do this for each taste separately
# Perform test only on neurons showing laser differences 
gc_test_firing = np.moveaxis(np.moveaxis(gc_firing, 0,2),-1,2)
bla_test_firing = np.moveaxis(np.moveaxis(bla_firing, 0,2),-1,2)

gc_test_firing = gc_test_firing[[x and y for x,y in zip(gc_sig_laser,gc_sig_taste)]]
bla_test_firing = bla_test_firing[[x and y for x,y in zip(bla_sig_laser,bla_sig_taste)]]

# Add noise to avoid singular results
gc_test_firing += np.random.random(gc_test_firing.shape)*1e-3
bla_test_firing += np.random.random(bla_test_firing.shape)*1e-3

gc_iter_inds = list(np.ndindex(*gc_test_firing.shape[:3]))
bla_iter_inds = list(np.ndindex(*bla_test_firing.shape[:3]))

from scipy.stats import anderson_ksamp

gc_p_vals = np.empty(gc_test_firing.shape[:3]) 
bla_p_vals = np.empty(bla_test_firing.shape[:3]) 

for this_iter in tqdm(gc_iter_inds):
    gc_p_vals[this_iter] = anderson_ksamp(
            [gc_test_firing[this_iter][0], gc_test_firing[this_iter][1]])[-1]
for this_iter in tqdm(bla_iter_inds):
    bla_p_vals[this_iter] = anderson_ksamp(
            [bla_test_firing[this_iter][0], bla_test_firing[this_iter][1]])[-1]

alpha = 0.05
fig,ax = plt.subplots(2)
ax[0].plot(np.mean(gc_p_vals < alpha, axis = (0,1)).T)
ax[1].plot(np.mean(bla_p_vals < alpha, axis = (0,1)).T)
plt.show()
