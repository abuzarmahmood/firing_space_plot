#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:16:14 2019

@author: victorsuarez
"""
#import Libraries
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

os.chdir(dir_name)

#Look for the hdf5 file in the directory
hdf5_name = sorted(glob.glob('NM*.h5'))
#hdf5_name = hdf5_name[0:4]
for files in hdf5_name:
    #Open the hdf5 file
    hf5 = tables.open_file(files, 'r+')
    #Pull in all spike trains into list of arrays
    trains_dig_in = hf5.list_nodes('/spike_trains')
    #laser_on = hf5.list_nodes('/on_laser')
    on_laser = np.asarray([on.on_laser[:] for on in trains_dig_in])
    spike_array = np.asarray([spikes.spike_array[:] for spikes in trains_dig_in])
    
    gape_array = hf5.root.ancillary_analysis.gapes
    
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
#                counts[condition,taste,trial] = ([np.sum(gape_array[condition, taste, trial, (i*t_bin):((i+1)*t_bin)]) for i in range(n_bins)])
#                axs.imshow(counts[condition, taste,:,:],interpolation = 'nearest', aspect ='auto')
                axs.imshow(gape_array_window[condition,taste,:,:], interpolation='nearest', aspect='auto')
    plt.xlabel('Time (ms)')
    plt.ylabel('Gape')
    plt.legend(taste_names, loc='upper right')
    fig.suptitle('Gapes per Trial per Tastant') #Creates a title for the plots kind of not needed
    #plt.savefig('neuron{}'.format(neuron+1))
    plt.show()
    
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
                            'taste' : 3 - idx_list[1],
                            'trial' : idx_list[2],
                            'gape_prob' : prob_matrix.flatten()})

    #get sizes
    #conditions, tastes, trials = prob_matrix.shape  
    #
    ##create labels for dataframe values
    #taste_labels = np.tile(np.stack(np.sort((trials*list(range(0,tastes))))),(conditions))
    #condition_labels = np.stack(np.sort(((tastes*trials)*list(range(0,conditions)))))
    #
    ##Panel data based on condition and taste
    #panel = pd.Panel(data=np.rollaxis(prob_matrix,2)).to_frame()
    #
    ##Reset index based on panel (after pulling out column for condition) 
    #decon_panel = panel.set_index(panel.index.labels[0]).reset_index()
    #df_prob = decon_panel.stack().reset_index()
    #
    ##Clear out predetermined label value within frame
    #df_prob = df_prob[~df_prob['level_1'].isin(['index'])]
    #
    ##Insert categorical labels
    #df_prob.insert(loc=0, column='taste', value=taste_labels)
    #df_prob.insert(loc=0, column='condition', value=condition_labels)
    #
    ##Drop extraneous label
    #df_prob = df_prob.drop(['level_0'], axis=1)
    #
    ##Rename columns for ease of understanding
    #df_prob.rename(columns = {'level_1':'trial'},inplace=True)
    #df_prob.rename(columns={df_prob.columns[3]: "gape_prob" }, inplace = True)
    #
    ##Convert to strings
    #all_columns = list(df_prob)
    #df_prob[all_columns[0:2]] = df_prob[all_columns[0:2]].astype(str)
    

    #Plotting
    fig = plt.figure(1)
    ax = sns.boxplot(x="taste", y="gape_prob", hue="condition", data=df_prob, palette="Set2")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["On", "Off"],bbox_to_anchor=(1, 0.5))
    plt.show()
    
    #create condition (change with respect to laser obs)
    condition = 1
    
    dQ_threshhold_median = np.median(total_prob[condition,2,:])
    cQ_threshhold_median = np.median(total_prob[condition,3,:])

    Hg_dQ = np.where(total_prob[condition,2,:] > dQ_threshhold_median)[0]
    Hg_cQ = np.where(total_prob[condition,3,:] > cQ_threshhold_median)[0]
    
    Lg_dQ = np.where(total_prob[condition,2,:] < dQ_threshhold_median)[0]
    Lg_cQ = np.where(total_prob[condition,3,:] < cQ_threshhold_median)[0]
    #Define array by trial, neuron, and time
    Hg_dQ_neurons = spike_array[2,Hg_dQ,:,:]
    Hg_cQ_neurons = spike_array[3,Hg_cQ,:,:]
    
    Lg_dQ_neurons = spike_array[2,Lg_dQ,:,:]
    Lg_cQ_neurons = spike_array[3,Lg_cQ,:,:]
    
    dQ_neurons_Hg = Hg_dQ_neurons.shape[1]
    cQ_neurons_Hg = Hg_cQ_neurons.shape[1]
    
    dQ_neurons_Lg = Lg_dQ_neurons.shape[1]
    cQ_neurons_Lg = Lg_cQ_neurons.shape[1]

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
    #gives array of the same dimension but w/ 0 (dummy array to fill in)
    counts_Hg_dQ = np.zeros((n_neurons, n_tastes, n_bins)) 
    counts_Lg_dQ= np.zeros((n_neurons, n_tastes, n_bins))
    counts_Hg_cQ= np.zeros((n_neurons, n_tastes, n_bins))
    counts_Lg_cQ= np.zeros((n_neurons, n_tastes, n_bins))
    
    #for condition in range(n_condition):
    for neuron in range(n_neurons): #Starts for loop to go through all the neurons (16 of them)
        for taste in range(n_tastes): #Starts with the first taste at 0 until done
        

            #pulls out only viable matrix (trialsXduration)
            laser_trials = np.squeeze(np.array(np.where(on_laser[taste,:,condition]==1))) 
            cond_trials = spike_array[taste,:,neuron,:]
            
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
    
            if taste >= 2:
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
                plt.legend()
            
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
                plt.legend()
