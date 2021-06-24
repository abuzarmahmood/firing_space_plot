"""
Comapare rasters and PSTHs for trial shuffled and simulated
data to actual data
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
import pylab as plt

import numpy as np
import pickle
import argparse

from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA as pca

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

params_file_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/fit_params.json'
parser = argparse.ArgumentParser(\
        description = 'Script to fit control changepoint models')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('states', type = int, help = 'Number of States to fit')
args = parser.parse_args()
data_dir = args.dir_name 

#data_dir = '/media/bigdata/Abuzar_Data/AM32/AM32_4Tastes_201124_164617'
#states = 4

dat = ephys_data(data_dir)

dat.firing_rate_params = dat.default_firing_params.copy()
#dat.firing_rate_params['type'] = 'baks'

dat.get_unit_descriptors()
dat.get_spikes()
#dat.get_firing_rates()
#dat.default_stft_params['max_freq'] = 50
taste_dat = np.array(dat.spikes)

##########
# PARAMS 
##########
states = int(args.states)

with open(params_file_path, 'r') as file:
    params_dict = json.load(file)

for key,val in params_dict.items():
    globals()[key] = val

bin_width = 10
##########
# Bin Data
##########
t_stim = 2000
t_vec = np.arange(taste_dat.shape[-1]) - t_stim
binned_t_vec = np.min(t_vec[time_lims[0]:time_lims[1]].\
                    reshape((-1,bin_width)),axis=-1)
this_dat_binned = \
        np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
        reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = np.vectorize(np.int)(this_dat_binned)
mean_dat_binned = np.mean(this_dat_binned,axis=1)

# Unroll along all time dimensions for scaling and PCA
mean_binned_long = np.reshape(np.moveaxis(mean_dat_binned,1,0),
                            (mean_shuffle_dat.shape[1],-1))
#visualize.imshow(mean_binned_long);plt.show()

# Save transform objects to transform single trials later
scaler_object = scaler().fit(mean_binned_long.T)
scaled_long_data = scaler_object.transform(mean_binned_long.T).T
#visualize.imshow(scaled_long_data);plt.show()

n_components = 3
pca_object = pca(n_components = n_components)\
        .fit(scaled_long_data.T)

# Use pca to convert data trial by trial
scaled_dat_binned = np.array([[scaler_object.transform(trial.T).T \
        for trial in this_taste] \
        for this_taste in this_dat_binned])
scaled_dat_pca = np.array([[pca_object.transform(trial.T).T \
        for trial in this_taste] \
        for this_taste in scaled_dat_binned])
#pca_taste = np.array([pca_object.transform(trial.T).T \
#        for trial in scaled_taste])

mean_dat_pca = np.mean(scaled_dat_pca,axis=1)


##################################################
## Create shuffled data
##################################################
# Unbinned shuffled dat for plotting
shuffle_taste_dat = np.array([np.random.permutation(neuron) \
            for neuron in np.swapaxes(taste_dat,2,0)])
shuffle_taste_dat = np.swapaxes(shuffle_taste_dat,0,2)

# Shuffle neurons across trials FOR SAME TASTE
shuffled_dat_binned = np.array([np.random.permutation(neuron) \
            for neuron in np.swapaxes(this_dat_binned,2,0)])
shuffled_dat_binned = np.swapaxes(shuffled_dat_binned,0,2)
mean_shuffled_binned = np.mean(shuffled_dat_binned,axis=1)

#mean_shuffle_dat = np.mean(shuffle_taste_dat,axis=1)

#mean_shuffle_long = np.reshape(np.moveaxis(mean_shuffled_binned,1,0),
#                            (mean_shuffled_binned.shape[1],-1))

#visualize.imshow(mean_shuffle_long);plt.show()

# Use pca to convert data trial by trial
scaled_dat_shuffled = np.array([[scaler_object.transform(trial.T).T \
        for trial in this_taste] \
        for this_taste in shuffled_dat_binned])
scaled_pca_shuffled = np.array([[pca_object.transform(trial.T).T \
        for trial in this_taste] \
        for this_taste in scaled_dat_shuffled])

mean_pca_shuffled = np.mean(scaled_pca_shuffled,axis=1)

#nrn = 1
#taste = 0
#plot_raster(this_dat_binned[taste,:,nrn])
#plt.figure()
#plot_raster(shuffled_dat_binned[taste,:,nrn])
#plt.show()
#
#trial = 0
#plot_raster(this_dat_binned[taste,trial])
#plt.figure()
#plot_raster(shuffled_dat_binned[taste,trial])
#plt.show()
#
#mean_firing = np.mean(this_dat_binned,axis=1)
#mean_shuffled_firing = np.mean(shuffled_dat_binned,axis=1)
#
#visualize.firing_overview(mean_firing)
#plt.figure()
#visualize.firing_overview(mean_shuffled_firing)
#plt.show()

##################################################
## Create simulated data 
##################################################
# Inhomogeneous poisson process using BAKS firing rates

spike_array = np.array(dat.spikes)
interp_mean_firing = np.mean(spike_array,axis=1)

# Simulate spikes
simulated_spike_array = np.array(\
        [np.random.random(interp_mean_firing.shape) < \
        interp_mean_firing \
        for trial in range(this_dat_binned.shape[1])])*1
simulated_spike_array = simulated_spike_array.swapaxes(0,1)
simulated_dat_binned = \
        np.sum(simulated_spike_array[...,time_lims[0]:time_lims[1]].\
        reshape(*simulated_spike_array.shape[:-1],-1,bin_width),axis=-1)
simulated_dat_binned = np.vectorize(np.int)(simulated_dat_binned)

# Use pca to convert data trial by trial
scaled_dat_simulated = np.array([[scaler_object.transform(trial.T).T \
        for trial in this_taste] \
        for this_taste in simulated_dat_binned])
scaled_pca_simulated = np.array([[pca_object.transform(trial.T).T \
        for trial in this_taste] \
        for this_taste in scaled_dat_simulated])

mean_pca_simulated = np.mean(scaled_pca_simulated,axis=1)

#nrn = 20
#taste = 0
#plot_raster(spike_array[taste,:,nrn])
#plt.figure()
#plot_raster(simulated_spike_array[taste,:,nrn])
#plt.show()
#
#trial = 15
#plot_raster(spike_array[taste,trial])
#plt.figure()
#plot_raster(simulated_spike_array[taste,trial])
#plt.show()
#
#nrn = 20
#taste = 0
#plot_raster(this_dat_binned[taste,:,nrn])
#plt.figure()
#plot_raster(simulated_dat_binned[taste,:,nrn])
#plt.show()
#
#trial = 15
#plot_raster(this_dat_binned[taste,trial])
#plt.figure()
#plot_raster(simulated_dat_binned[taste,trial])
#plt.show()
#
#mean_firing = np.mean(this_dat_binned,axis=1)
#mean_simulated_firing = np.mean(simulated_dat_binned,axis=1)
#
#visualize.firing_overview(mean_firing)
#plt.figure()
#visualize.firing_overview(mean_simulated_firing)
#plt.show()
#
#nrn = 4
#fig,ax = plt.subplots(1,2,sharey=True)
#ax[0].plot(mean_firing[:,nrn].T)
#ax[1].plot(mean_simulated_firing[:,nrn].T)
#plt.show()

########################################
# ____  _       _       
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
########################################
                      
# Plot PCA for single taste from actual data
# and shuffle and simulated data

from scipy.signal.windows import gaussian
smooth_kern = gaussian(scaled_dat_pca.shape[-1]//6,
                    scaled_dat_pca.shape[-1]//6/3)
taste = 0
name_list = ['Actual','Simulated']

fig, ax = plt.subplots(1,3)
for dat_num, dat_set in \
            enumerate([scaled_dat_pca,scaled_pca_simulated]):
    mean_dat = np.mean(dat_set,axis=1)[taste]
    sd_dat = np.std(dat_set,axis=1)[taste]
    for num, (this_mean,this_sd) in enumerate(zip(mean_dat,sd_dat)):
        x = np.arange(len(this_mean))
        ax[num].plot(binned_t_vec,
                np.convolve(this_mean,smooth_kern,mode='same'), 
                linewidth = 3,
                alpha = 0.9,
                label =name_list[dat_num])
        #ax[num].fill_between(x = x,
        #            y1 = this_mean - this_sd,
        #            y2 = this_mean + this_sd,
        #            alpha = 0.3)
plt.legend()
plt.show()

# Rasters from single trials
# of UNBINNED DATA
def plot_raster(ax,matrix):
    ax.scatter(*np.where(matrix)[::-1], marker = '|', c='k')

trial = 0
nrn = 0

fig,ax = plt.subplots(3,1)
for dat_num, dat_set in \
            enumerate([taste_dat,shuffle_taste_dat,simulated_spike_array]):
        this_trial = dat_set[taste,trial,:,time_lims[0]:time_lims[1]]
        plot_raster(ax[dat_num],this_trial)
plt.show()

# Rasters and PSTHs for single NEURONS
# of UNBINNED DATA
kern_size = 250
smooth_kern = gaussian(kern_size,50)/kern_size
plt.plot(smooth_kern);plt.show()

cmap = plt.get_cmap("tab10")

nrn_num = 0
fig,ax = plt.subplots(2,2,sharey='row',sharex=True,figsize=(5,5))
for dat_num, dat_set in \
            enumerate([taste_dat,simulated_spike_array]):
        this_psth = np.mean(\
                dat_set[taste,:,nrn_num,time_lims[0]:time_lims[1]],axis=0)
        this_psth /= (bin_width/1000)
        this_psth /= (kern_size/1000)
        ax[0,dat_num].plot(np.convolve(this_psth,smooth_kern,mode='same'),
                color = cmap(dat_num), alpha = 0.9)
        ax[0,dat_num].set_title(name_list[dat_num])
        ax[0,dat_num].spines['right'].set_visible(False)
        ax[0,dat_num].spines['top'].set_visible(False)
        ax[1,dat_num].set_xlabel('Time post-stimulus delivery (ms)')
        this_trial = dat_set[taste,:,nrn_num,time_lims[0]:time_lims[1]]
        plot_raster(ax[-1,dat_num],this_trial)
ax[0,0].set_ylabel('Firing Rate (Hz)')
ax[1,0].set_ylabel('Trial Number')
plt.tight_layout()
plt.show()
