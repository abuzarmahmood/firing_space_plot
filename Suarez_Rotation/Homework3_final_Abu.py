#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:49:37 2019

@author: victorsuarez
"""

# Built-in Python libraries
import os # functions for interacting w operating system
import glob
# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import tables
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
# =============================================================================
# #Load Data
# =============================================================================
#Get name of directory where the data files and hdf5 file sits, 
#and change to that directory for processing
dir_name = easygui.diropenbox(msg = 'Choose directory\
                              with where hdf5 file exists.')

#os.chdir(dir_name)

figs_dir = 'figs'

#Look for the hdf5 file in the directory
#hdf5_name = sorted(glob.glob('NM*.h5'))
hdf5_name = sorted(glob.glob(dir_name + '/' + 'NM*.h5'))
#hdf5_name = hdf5_name[0:4]

#for files in hdf5_name:

files = hdf5_name[0]
#Open the hdf5 file
hf5 = tables.open_file(files, 'r+')
#Pull in all spike trains into list of arrays
trains_dig_in = hf5.list_nodes('/spike_trains')
#laser_on = hf5.list_nodes('/on_laser')

# Find trials with laser on
on_laser = np.asarray([on.on_laser[:] for on in trains_dig_in])
# Convert to boolean for each trial
on_laser = np.sum(on_laser, axis = -1) > 0

spike_array = np.asarray([spikes.spike_array[:] \
        for spikes in trains_dig_in])
gape_array = hf5.root.ancillary_analysis.gapes
laser_combination_array = hf5.root.ancillary_analysis.laser_combination_d_l[:]

t_length = gape_array.shape[3] #Here
n_laser = on_laser.shape[2]
n_conditions = gape_array.shape[0] 
n_tastes = gape_array.shape[1] 
n_trials = gape_array.shape[2] 
n_neurons = spike_array.shape[2]

taste_names = ['dSucrose', 'cSucrose','dQuinine', 'cQuinine']
t_bin = 250
step_size = 25
n_bins = gape_array.shape[3]//step_size
bins = np.arange(0, spike_array.shape[3], step_size)

gape_array_window = gape_array[...,3000:4500]

for condition in range(n_conditions): 
    fig, axs = plt.subplots(n_tastes,sharex=True) 
    for taste, axs in zip(range(n_tastes),axs.flatten()): 
        for trial in range(n_trials):
            axs.imshow(gape_array_window[condition,taste,:,:], \
                    interpolation='nearest', aspect='auto')
    plt.xlabel('Time (ms)')
    plt.ylabel('Gape')
    plt.legend(taste_names, loc='upper right')
    fig.suptitle('Gapes per Trial per Tastant (Condition {})'.format(condition)) 
    #Creates a title for the plots kind of not needed
    plt.show()
#        plt.savefig('{}/{}_heatmap_condition{}'.format(figs_dir,files[0:4],condition))
    #plt.show()

#Plotted 4 tastes as heatmaps for one conditions
total_prob = np.sum(gape_array_window, axis=3)
#data matrix
prob_matrix = total_prob

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs

idx_list = make_array_identifiers(prob_matrix)

df_prob = pd.DataFrame({'condition' : idx_list[0],
                        'taste' : idx_list[1],
                        'trial' : idx_list[2],
                        'gape_prob' : prob_matrix.flatten()})

#Plotting
fig = plt.figure(3)
ax = sns.boxplot(x="taste", y="gape_prob", hue="condition", data=df_prob, palette="Set2")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["On", "Off"],bbox_to_anchor=(1, 0.5))
#plt.savefig('{}/{}_gaping_probability.png'.format(figs_dir,files[0:4]))
plt.show()

#create condition (change with respect to laser obs)
condition = 0 

# Find median gaping prob for each tastes
dQ_threshhold_median = np.median(total_prob[condition,2,:])
cQ_threshhold_median = np.median(total_prob[condition,3,:])

# Subset high gaping trials
Hg_dQ = np.where(total_prob[condition,2,:] > dQ_threshhold_median)[0]
Hg_cQ = np.where(total_prob[condition,3,:] > cQ_threshhold_median)[0]

# Subset low gaping trials
Lg_dQ = np.where(total_prob[condition,2,:] < dQ_threshhold_median)[0]
Lg_cQ = np.where(total_prob[condition,3,:] < cQ_threshhold_median)[0]

#Define array by trial, neuron, and time


## Separate trials by on laser and off laser ##
# Since the gape array is ALREADY separated by laser condition
# Spiking NEEDS to be indexed by that first
off_spiking = np.asarray([spike_array[taste,~on_laser[taste]] \
        for taste in range(spike_array.shape[0])])
on_spiking = np.asarray([spike_array[taste,on_laser[taste]] \
        for taste in range(spike_array.shape[0])])

# We're indexing trials NOT neurons
# Variable names count for people reading the code
#Hg_dQ_neurons = off_spiking[2,Hg_dQ,:,:]
#Hg_cQ_neurons = off_spiking[3,Hg_cQ,:,:]
#
#Lg_dQ_neurons = off_spiking[2,Lg_dQ,:,:]
#Lg_cQ_neurons = off_spiking[3,Lg_cQ,:,:]

Hg_dQ_trials = off_spiking[2,Hg_dQ,:,:]
Hg_cQ_trials = off_spiking[3,Hg_cQ,:,:]
Lg_dQ_trials = off_spiking[2,Lg_dQ,:,:]
Lg_cQ_trials = off_spiking[3,Lg_cQ,:,:]

# These varables are never used again
# Avoid having unused variables because it makes reading the code harder
#dQ_neurons_Hg = Hg_dQ_neurons.shape[1]
#cQ_neurons_Hg = Hg_cQ_neurons.shape[1]
#
#dQ_neurons_Lg = Lg_dQ_neurons.shape[1]
#cQ_neurons_Lg = Lg_cQ_neurons.shape[1]

# ============================================================= #

firing_rate_array = np.zeros(list(spike_array.shape[:3])+[n_bins])
for i in range(n_bins):
    firing_rate_array[:,:,:,i] = \
            np.sum(spike_array[:,:,:,i*step_size:i*step_size+t_bin],axis=-1)
firing_rate_array = firing_rate_array*1000/t_bin

## Example
# Pull 10 random trials from taste 3 and find mean and sd
random_trials = np.random.choice(range(firing_rate_array.shape[1]),10)
test_firing = firing_rate_array[3,random_trials,:,:]
# (trial x neuron x time_bin)
# Find mean firing and SD of firing for ALL neurons over given trials
mean_firing = np.mean(test_firing, axis = 0)
error_firing = np.std(test_firing, axis = 0)

# ============================================================= #

#Plot PSTH
#    #gives array of the same dimension but w/ 0 (dummy array to fill in)
counts_Hg_dQ = np.zeros((n_neurons, n_tastes, n_bins)) 
counts_Lg_dQ= np.zeros((n_neurons, n_tastes, n_bins))
counts_Hg_cQ= np.zeros((n_neurons, n_tastes, n_bins))
counts_Lg_cQ= np.zeros((n_neurons, n_tastes, n_bins))

# ============================================================= #
# Example firing rate calculation function
def calc_firing_rates(spike_array, step_size, t_bin, n_bins):
    """
    argument : spike_array :: 3D array (trial x neuron x time)
    argument : step_size
    argument : t_bin :: Window size
    argument : n_bins :: Number of time bins for every trial
    output : firing_array :: 3D array (neuron x trial x time_bin)
    """
    firing_array = np.zeros((spike_array.shape[1], n_bins))

    for neuron in range(spike_array.shape[1]):
        for i in range(n_bins):
            firing_array[neuron, i] = \
                    np.sum(spike_array[:,neuron,
                        i*step_size:i*step_size+t_bin])
    firing_array = firing_array*1000/t_bin/spike_array.shape[0]
    return firing_array

counts_Hg_dQ = calc_firing_rates(Hg_dQ_trials, step_size, t_bin, n_bins)
# ============================================================= #

#for condition in range(n_condition):
for neuron in range(n_neurons): #Starts for loop to go through all the neurons (16 of them)
    #for taste in range(n_tastes): #Starts with the first taste at 0 until done
    
        #pulls out only viable matrix (trialsXduration)
        laser_trials = np.squeeze(np.array(np.where(on_laser[taste,:,condition]==1))) 
        cond_trials = spike_array[taste,:,neuron,:]
        
        # All these lines of code are doing the same thing
        # The process can be turned into a function thats called multiple times
        # Although the output is the same, this will make the code significantly
        # cleaner which is important for readiblity and debugging
        # This is probably on the advanced end for you but INDISPENSABLE for
        # even a novice coder

        # Also, list comprehensions are quite useful but when they become
        # lengthy like this, it compromises readability. In that case it is
        # recommended just to write a for loop

        counts_Hg_dQ[neuron, taste] = \
        np.asarray([np.sum(cond_trials[laser_trials[np.array(Hg_dQ)],i*step_size:i*step_size+t_bin])/len(laser_trials) \
                             for i in range(n_bins)])*1000/t_bin
        counts_Lg_dQ[neuron, taste] = \
        np.asarray([np.sum(cond_trials[laser_trials[np.array(Lg_dQ)],i*step_size:i*step_size+t_bin])/len(laser_trials) \
                             for i in range(n_bins)])*1000/t_bin
        counts_Hg_cQ[neuron, taste] = \
        np.asarray([np.sum(cond_trials[laser_trials[np.array(Hg_cQ)],i*step_size:i*step_size+t_bin])/len(laser_trials) \
                             for i in range(n_bins)])*1000/t_bin
        counts_Lg_cQ[neuron, taste] = \
        np.asarray([np.sum(cond_trials[laser_trials[np.array(Lg_cQ)],i*step_size:i*step_size+t_bin])/len(laser_trials) \
                             for i in range(n_bins)])*1000/t_bin 

        counts_Hg_dQ_SEM =\
        (([np.std(np.sum(cond_trials[laser_trials[np.array(Hg_dQ)],i*step_size:i*step_size+t_bin],axis=-1))\
                         for i in range(n_bins)])/np.sqrt(laser_trials.shape[0]))*1000/t_bin
        counts_Lg_dQ_SEM =\
        (([np.std(np.sum(cond_trials[laser_trials[np.array(Lg_dQ)],i*step_size:i*step_size+t_bin],axis=-1))\
                         for i in range(n_bins)])/np.sqrt(laser_trials.shape[0]))*1000/t_bin
        counts_Hg_cQ_SEM =\
        (([np.std(np.sum(cond_trials[laser_trials[np.array(Hg_cQ)],i*step_size:i*step_size+t_bin],axis=-1))\
                         for i in range(n_bins)])/np.sqrt(laser_trials.shape[0]))*1000/t_bin
        counts_Lg_cQ_SEM = \
                (([np.std(np.sum(cond_trials[laser_trials[np.array(Lg_cQ)],i*step_size:i*step_size+t_bin],axis=-1))\
                         for i in range(n_bins)])/np.sqrt(laser_trials.shape[0]))*1000/t_bin

        counts_Hg_dQ[neuron, taste] = \
            np.asarray([np.sum(cond_trials[laser_trials[np.array(Hg_dQ)],\
            i*step_size:i*step_size+t_bin])/len(laser_trials) \
            for i in range(n_bins)])*1000/t_bin

        #if taste >= 2:
            #Initiate figure
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

        #Plot
        ax = axs[0]
        #ax.plot(bins, counts_Hg_dQ[neuron, taste, :],label = 'High Gape',color='midnightblue')
        #ax.plot(bins, counts_Lg_dQ[neuron, taste, :], label = 'Low Gape',color='darkred')
        ax.set_title('Dil. QHCl Gaping Prob')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.errorbar(bins, counts_Hg_dQ[neuron, taste, :], \
                yerr = counts_Hg_dQ_SEM,\
                label = 'High Gape',color='midnightblue')
        ax.errorbar(bins, counts_Lg_dQ[neuron, taste, :], \
                yerr = counts_Lg_dQ_SEM,\
                label = 'Low Gape',color='darkred')
        ax.legend()
    
        ax = axs[1]
        #ax.plot(bins, counts_Hg_cQ[neuron, taste, :],label = 'High Gape',color='midnightblue')
        #ax.plot(bins, counts_Lg_cQ[neuron, taste, :], label = 'Low Gape',color='darkred')
        ax.set_title('Conc. QHCl Gaping Prob')
        plt.suptitle('Taste: %s' %(taste))
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_xlabel('Time (ms)')
        ax.legend(bbox_to_anchor=(0.8, -0.3),ncol=2)
        ax.errorbar(bins, counts_Hg_cQ[neuron, taste, :], \
                yerr = counts_Hg_cQ_SEM,label = 'High Gape',color='midnightblue')
        ax.errorbar(bins, counts_Lg_cQ[neuron, taste, :], \
                yerr = counts_Lg_cQ_SEM,\
                label = 'Low Gape',color='darkred')
        ax.legend()
        #plt.savefig('{}/{}_psth_neuron{}_quinine{}.png'.format(figs_dir,files[0:4],neuron,taste))
        
