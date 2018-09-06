#  ____                 _ _              _____  _       
# |  _ \               | (_)            |  __ \(_)      
# | |_) | __ _ ___  ___| |_ _ __   ___  | |  | |___   __
# |  _ < / _` / __|/ _ \ | | '_ \ / _ \ | |  | | \ \ / /
# | |_) | (_| \__ \  __/ | | | | |  __/ | |__| | |\ V / 
# |____/ \__,_|___/\___|_|_|_| |_|\___| |_____/|_| \_/  ergence
#

# 
######################### Import dat ish #########################
import tables
#import easygui
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from scipy.spatial import distance_matrix as dist_mat
from scipy.stats.mstats import zscore
from scipy.stats import pearsonr

#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#
all_corrs = []
for file in range(1,7):
    data_dir = '/media/bigdata/jian_you_data/des_ic/file_%i/' % file
    os.chdir(data_dir)
    # Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
    file_list = os.listdir('./')
    hdf5_name = ''
    params_file = ''
    units_file = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files
        if files[-10:] == 'hmm_params':
            params_file = files
        if files[-9:] == 'hmm_units':
            units_file = files
    
    # Read the .hmm_params file
    f = open(params_file, 'r')
    params = []
    for line in f.readlines():
        params.append(line)
    f.close()
    
    # Assign the params to variables
    min_states = int(params[0])
    max_states = int(params[1])
    max_iterations = int(params[2])
    start_t = int(params[6])
    end_t = int(params[7])
    bin_size = int(params[8])
    
    # Read the chosen units
    f = open(units_file, 'r')
    chosen_units = []
    for line in f.readlines():
        chosen_units.append(int(line))
    chosen_units = np.array(chosen_units) - 1
    
    # Open up hdf5 file
    hf5 = tables.open_file(hdf5_name, 'r+')
    
    # Import all data and store relevant variables in lists
    # For alignment
    spikes = []
    off_spikes = []
    on_spikes = []
    all_off_trials = []
    all_on_trials = []
    
    for taste in range(4):
        exec('spikes.append(hf5.root.spike_trains.dig_in_%i.spike_array[:])' % taste)
    
        # Slice out the required portion of the spike array, and bin it
        spikes[taste] = spikes[taste][:, chosen_units, :]
        spikes[taste] = np.swapaxes(spikes[taste], 0, 1)
    
        exec('dig_in = hf5.root.spike_trains.dig_in_%i' % taste)
        laser_exists = []
        try:
            laser_exists = dig_in.laser_durations[:]
        except:
            pass
        on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
        off_trials = np.where(dig_in.laser_durations[:] == 0.0)[0]
    
        all_off_trials.append(off_trials + taste * len(off_trials) * 2)
        all_on_trials.append(on_trials + taste * len(on_trials) * 2)
    
        off_spikes.append(spikes[taste][:, off_trials, :])
        on_spikes.append(spikes[taste][:, on_trials, :])
    
    all_off_trials = np.concatenate(np.asarray(all_off_trials))
    all_on_trials = np.concatenate(np.asarray(all_on_trials))
    
    ################### Convert spikes to firing rates ##################
    step_size = 25
    window_size = 250
    tot_time = 7000
    firing_len = int((tot_time-window_size)/step_size)-1
    off_firing = []
    
    ## Moving Window
    for l in range(len(off_spikes)): # How TF do you get nan's from means?
        # [trials, nrns, time]
        this_off_firing = np.zeros((off_spikes[0].shape[0],off_spikes[0].shape[1],firing_len))
        for i in range(this_off_firing.shape[0]):
            for j in range(this_off_firing.shape[1]):
                for k in range(this_off_firing.shape[2]):
                    this_off_firing[i, j, k] = np.mean(off_spikes[l][i, j, step_size*k:step_size*k + window_size])
                    if np.isnan(this_off_firing[i, j, k]):
                        print('found nan')
                        break
        #this_off_firing = this_off_firing.reshape((this_off_firing.shape[1],this_off_firing.shape[0]*this_off_firing.shape[2]))
        off_firing.append(this_off_firing)
        
    ## Inter-spike interval
    # =============================================================================
    # off_firing = []
    # for l in range(len(off_spikes)):
    #     this_off_firing = np.zeros(off_spikes[0].shape)
    #     for i in range(this_off_firing.shape[0]):
    #         for j in range(this_off_firing.shape[1]):
    #             this_trial = off_spikes[l][m,n,:]
    #             spike_inds = np.where(off_spikes[l][i,j,:]>0)[0]
    #             f_rate = np.reciprocal(np.diff(spike_inds).astype(float))
    #             for k in range(len(f_rate)):
    #                 this_off_firing[i,j,spike_inds[k]:] = f_rate[k]
    #     off_firing.append(this_off_firing)  
    # =============================================================================
        
        
    # Normalize firing (over every trial of every neuron)
# =============================================================================
#     for l in range(len(off_firing)):
#         for m in range(off_firing[0].shape[0]):
#             for n in range(off_firing[0].shape[1]):
#                 min_val = np.min(off_firing[l][m,n,:])
#                 max_val = np.max(off_firing[l][m,n,:])
#                 off_firing[l][m,n,:] = (off_firing[l][m,n,:] - min_val)/(max_val-min_val)
#     
# =============================================================================
    # Normalized firing of every neuron over entire dataset
    off_firing_array = np.asarray(off_firing) #(taste x nrn x trial x time)
    for m in range(off_firing_array.shape[1]): # nrn
        min_val = np.min(off_firing_array[:,m,:,:])
        max_val = np.max(off_firing_array[:,m,:,:])
        for l in range(len(off_firing)): #taste
            for n in range(off_firing[0].shape[1]): # trial
                off_firing[l][m,n,:] = (off_firing[l][m,n,:] - min_val)/(max_val-min_val)


#   _____                    _       _   _             
#  / ____|                  | |     | | (_)            
# | |     ___  _ __ _ __ ___| | __ _| |_ _  ___  _ __  
# | |    / _ \| '__| '__/ _ \ |/ _` | __| |/ _ \| '_ \ 
# | |___| (_) | |  | | |  __/ | (_| | |_| | (_) | | | |
#  \_____\___/|_|  |_|  \___|_|\__,_|\__|_|\___/|_| |_|
#                                                      
  
    ##############################
    stim_t = 2000 # Where to start
    pre_stim_t = 200 # How much behind start time to take for baseline
    post_stim_t = 2000 # How much after start time to take for activity
    plt.figure()
    corrs = []
    for taste in range(4):
        data = off_firing[taste]
        
        mean_pre_stim = np.mean(data[:,:,int((stim_t - pre_stim_t)/bin_size):int(stim_t/bin_size)],axis = 2).T #(neurons x trials)
        pre_stim_dist = np.tril(dist_mat(mean_pre_stim,mean_pre_stim)) # Take out upper diagonal to prevent double counting
        
        stim_dists = np.zeros((data.shape[1],data.shape[1],data.shape[2]))
        for time in range(stim_dists.shape[2]):
            stim_dists[:,:,time] = dist_mat(data[:,:,time].T,data[:,:,time].T)
        sum_stim_dists = np.tril(np.sum(stim_dists,axis = 2))
        
        temp_corr = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
        corrs.append(temp_corr)
        
        exec('plt.subplot(22%i)' % (taste+1))
        plt.scatter(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
        plt.title('taste%i, rho = %.3f, p = %1.2e' % (taste,temp_corr[0],temp_corr[1]))
        # =============================================================================
        # sum_stim_dists2 = sum_stim_dists
        # sum_stim_dists2[sum_stim_dists == 0] = np.mean(sum_stim_dists) 
        # =============================================================================
    all_corrs.append(corrs)
# =============================================================================
#     plt.subplot(211)
#     plt.imshow(pre_stim_dist)
#     plt.subplot(212)
#     plt.imshow(sum_stim_dists)
#     
#     plt.figure()
#     plt.scatter(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
# 
# =============================================================================

