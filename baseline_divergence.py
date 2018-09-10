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
from scipy import signal

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import scale

#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#
corr_dat = pd.DataFrame()

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
    
    hf5.close()
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
    all_pre_stim_window = np.arange(100,1000,100) # How much behind start time to take for baseline
    post_stim_t = 2000 # How much after start time to take for activity
    stim_t = 2000 # When stimulus was given
    
    all_vars = []
    
    for j in range(len(all_pre_stim_window)):
        pre_stim_window = all_pre_stim_window[j]
        all_pre_stim_t = np.arange(2000,200,-pre_stim_window) # Where to start
        for i in range(len(all_pre_stim_t)):
            pre_stim_t = all_pre_stim_t[i]
            corrs = []
            stim_vars = []
            #fig = plt.figure()
            for taste in range(4):
                data = off_firing[taste]
                
                mean_pre_stim = np.mean(data[:,:,int((pre_stim_t - pre_stim_window)/step_size):int(pre_stim_t/step_size)],axis = 2).T #(neurons x trials)
                pre_stim_dist = np.tril(dist_mat(mean_pre_stim,mean_pre_stim)) # Take out upper diagonal to prevent double counting
                
                stim_dat = data[:,:,int(stim_t/step_size):int((stim_t+post_stim_t)/step_size)]
                stim_dists = np.zeros((stim_dat.shape[1],stim_dat.shape[1],stim_dat.shape[2]))
                stim_dist_var = np.zeros(stim_dat.shape[2])
                for time in range(stim_dists.shape[2]):
                    stim_dists[:,:,time] = dist_mat(stim_dat[:,:,time].T,stim_dat[:,:,time].T)
                    stim_dist_var_temp = np.tril(stim_dists[:,:,time])
                    stim_dist_var[time] = np.var(stim_dist_var_temp[stim_dist_var_temp.nonzero()].flatten())
                sum_stim_dists = np.tril(np.sum(stim_dists,axis = 2))
                
                temp_corr = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
                temp_corr_dat = pd.DataFrame(dict(file = file, taste = taste, 
                        baseline_end = pre_stim_t, rho = temp_corr[0],p = temp_corr[1],
                        index = [corr_dat.shape[0]], shuffle = False, pre_stim_window_size = pre_stim_window))
                
                for repeats in range(200): # Shuffle trials
                    temp_corr_sh = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),
                                                          np.random.permutation(sum_stim_dists[sum_stim_dists.nonzero()].flatten()))
                    temp_corr_sh_dat = pd.DataFrame(dict(file = file, taste = taste, 
                            baseline_end = pre_stim_t, rho = temp_corr_sh[0],p = temp_corr_sh[1],
                            index = [corr_dat.shape[0]], shuffle = True, pre_stim_window_size = pre_stim_window))
                    corr_dat = pd.concat([corr_dat,temp_corr_sh_dat])
                
                corr_dat = pd.concat([corr_dat,temp_corr_dat])
            print('file %i end_at %i window %i' % (file, pre_stim_t,pre_stim_window))
            
# =============================================================================
#             corrs.append(temp_corr)
#             stim_vars.append(stim_dist_var)
#             
#             exec('plt.subplot(22%i)' % (taste+1))
#             plt.scatter(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
#             #sb.regplot(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
#             plt.title('%i, rho = %.2f, p = %.2f' % (taste, temp_corr[0], temp_corr[1]))
#             plt.suptitle('%i' % pre_stim_t)
#         
#         for i in fig.axes:
#             i.get_xaxis().set_visible(False)
#             i.get_yaxis().set_visible(False)
#             
#         all_corrs.append(corrs)
#         all_vars.append(stim_vars)
# =============================================================================

g = sns.FacetGrid(corr_dat,col = 'taste', row = 'file', hue = 'shuffle',sharey = 'row')
#g.set(ylim=(0,None))
g.map(sns.regplot,'baseline_end','rho', 
      x_estimator=np.mean, x_ci = 'sd').add_legend()



# Variation in trajectories
all_vars_array = np.asarray(all_vars)
for animal in range(all_vars_array.shape[0]):
    plt.figure()
    plt.errorbar(x = range((np.mean(all_vars_array[animal,:,:],axis=0)).size),
              y = np.mean(all_vars_array[animal,:,:],axis=0),
              yerr = np.std(all_vars_array[animal,:,:],axis=0))
    
# Autocorrelation of baseline firing
pre_stim = data[:,:,:int(stim_t/step_size)] #(neurons x trials)
auto_corr = np.zeros(pre_stim.shape)
for trial in range(auto_corr.shape[1]):
    auto_corr[:,trial,:] = signal.correlate2d(pre_stim[:,trial,:], pre_stim[:,trial,:], boundary='symm', mode='same')
auto_corr_1d = scale(auto_corr[np.argmax(np.sum(auto_corr[:,0,:],axis=1)),:,:],axis=1)
plt.errorbar(x = range(auto_corr_1d.shape[1]),y = np.mean(auto_corr_1d,axis=0),yerr = np.std(auto_corr_1d,axis=0))

