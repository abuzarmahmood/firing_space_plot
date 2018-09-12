def stim_corr_shuffle(pre_stim_dist, sum_stim_dists, baseline_end, baseline_start, step_size, taste):
    temp_corr_sh = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),
                                          np.random.permutation(sum_stim_dists[sum_stim_dists.nonzero()].flatten()))
    temp_corr_sh_dat = pd.DataFrame(dict(file = file, taste = taste, 
            baseline_end = baseline_end*step_size, rho = temp_corr_sh[0],p = temp_corr_sh[1],
            index = [corr_dat.shape[0]], shuffle = True, pre_stim_window_size = (baseline_end - baseline_start)*step_size))
    return temp_corr_sh_dat

def baseline_stimulus_correlation(off_firing, baseline_window, stimulus_time,
                                  stimulus_window_size, step_size, shuffle_repeats):
    corr_dat = pd.DataFrame() 
    for taste in range(4):
        data = off_firing[taste]
        
        baseline_start = int(baseline_window[0]/step_size)
        baseline_end = int(baseline_window[1]/step_size)
        stim_start = int(stimulus_time/step_size)
        stim_end = int((stimulus_time + stimulus_window_size)/step_size)
        
        mean_pre_stim = np.mean(data[:,:,baseline_start:baseline_end],axis = 2).T #(neurons x trials)
        pre_stim_dist = np.tril(dist_mat(mean_pre_stim,mean_pre_stim)) # Take out upper diagonal to prevent double counting
        
        stim_dat = data[:,:,stim_start:stim_end]
        stim_dists = np.zeros((stim_dat.shape[1],stim_dat.shape[1],stim_dat.shape[2]))
        for time_bin in range(stim_dists.shape[2]):
            stim_dists[:,:,time_bin] = dist_mat(stim_dat[:,:,time_bin].T,stim_dat[:,:,time_bin].T)
        sum_stim_dists = np.tril(np.sum(stim_dists,axis = 2))
        
        temp_corr = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
        temp_corr_dat = pd.DataFrame(dict(file = file, taste = taste, 
                baseline_end = baseline_end*step_size, rho = temp_corr[0],p = temp_corr[1],
                index = [corr_dat.shape[0]], shuffle = False, pre_stim_window_size = (baseline_end - baseline_start)*step_size))
        corr_dat = pd.concat([corr_dat, temp_corr_dat])
        
        for repeat in range(shuffle_repeats):
            output = stim_corr_shuffle(pre_stim_dist, sum_stim_dists,baseline_end, baseline_start, step_size, taste)
            corr_dat = pd.concat([corr_dat,output])
        
    return corr_dat

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

import multiprocessing as mp
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
# =============================================================================
#     # Avg subtracted normalization
#     off_firing_array = np.asarray(off_firing) #(taste x nrn x trial x time)
#     for m in range(off_firing_array.shape[1]): # nrn
#         mean_val = np.mean(off_firing_array[:,m,:,:])
#         var_val = np.var(off_firing_array[:,m,:,:])
#         for l in range(len(off_firing)): #taste
#             for n in range(off_firing[0].shape[1]): # trial
#                 off_firing[l][m,n,:] = (off_firing[l][m,n,:] - mean_val)/var_val
# =============================================================================


#   _____                    _       _   _             
#  / ____|                  | |     | | (_)            
# | |     ___  _ __ _ __ ___| | __ _| |_ _  ___  _ __  
# | |    / _ \| '__| '__/ _ \ |/ _` | __| |/ _ \| '_ \ 
# | |___| (_) | |  | | |  __/ | (_| | |_| | (_) | | | |
#  \_____\___/|_|  |_|  \___|_|\__,_|\__|_|\___/|_| |_|
#                                                      
  
    ##############################
    baseline_window_sizes = np.arange(100,1000,100)
    baseline_window_end = 2000
    baseline_window_start = 200
    all_baseline_windows = []
    for i in range(len(baseline_window_sizes)):
        #temp_baseline_windows = np.arange(baseline_window_end,baseline_window_start-baseline_window_sizes[i],-baseline_window_sizes[i])
        temp_baseline_windows = np.arange(baseline_window_end, baseline_window_start, -100)
        temp_baseline_windows = temp_baseline_windows[(temp_baseline_windows - baseline_window_sizes[i]) >0]
        for j in range(0,len(temp_baseline_windows)):
            all_baseline_windows.append((temp_baseline_windows[j]- baseline_window_sizes[i],temp_baseline_windows[j]))
    
    stimulus_time = 2000
    stimulus_window_size = 2000
    step_size = 25
    
    shuffle_repeats = 100
    
    pool = mp.Pool(processes = mp.cpu_count())
    results = [pool.apply_async(baseline_stimulus_correlation, args = (off_firing, all_baseline_windows[i], stimulus_time,
                                      stimulus_window_size, step_size, shuffle_repeats)) for i in range(len(all_baseline_windows))]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    for i in range(len(output)):
        corr_dat = pd.concat([corr_dat,output[i]])
        
    print('file %i' % file)

#############################
for file in range(1,7):
    g = sns.FacetGrid(corr_dat.query('file==%i' % file),col = 'taste', 
                      row = 'pre_stim_window_size', hue = 'shuffle', sharey = 'all')
    #g.set(ylim=(0,None)
    g.map(sns.regplot,'baseline_end','rho', 
          x_estimator=np.mean, x_ci = 'sd').add_legend()
    g.fig.suptitle('FILE %i' % file)
    g.savefig('file%i_windows.png' % file)

######################################################
# Dimensionality reduction test
# Since the system changes dramatically around stimulus delivery
# If baseline and some time after stim delivery is dimension reduced
    # we should be able to see a jump in the trajectory
plt.imshow(np.mean(off_firing[0],axis=1), interpolation='nearest', aspect='auto')

f, axs = plt.subplots(8,1,figsize=(9,6))
for ax in range(len(axs)):
    axs[ax].imshow(off_firing[0][:,ax,:], interpolation='nearest', aspect='auto')

from sklearn.manifold import LocallyLinearEmbedding as LLE
from mpl_toolkits.mplot3d import Axes3D

for trial in range(8):
    off_f_red = LLE(n_neighbors = 50,n_components=3).fit_transform(np.transpose(off_firing[0][:,trial,:160]))
# =============================================================================
#     plt.figure()
#     plt.scatter(off_f_red[:,0],off_f_red[:,1],c =np.linspace(1,255,len(off_f_red[:,0])))
#     
# =============================================================================
    
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(off_f_red[:,0],off_f_red[:,1],off_f_red[:,2],c =range(len(off_f_red[:,0])))
    ax.scatter(off_f_red[0,0],off_f_red[0,1],off_f_red[0,2],c = 'red')
    ax.scatter(off_f_red[-1,0],off_f_red[-1,1],off_f_red[-1,2],c = 'red')
    ax.plot(off_f_red[:,0],off_f_red[:,1],off_f_red[:,2],linewidth=0.5)
    plt.colorbar(p)
######################################################
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
