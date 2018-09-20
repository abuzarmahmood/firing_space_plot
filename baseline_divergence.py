

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
from scipy.stats import multivariate_normal
import copy

import multiprocessing as mp

#########################################3
def stim_corr_shuffle(pre_stim_dist, sum_stim_dists, baseline_end, baseline_start, step_size, file, taste, laser, corr_dat):
    temp_corr_sh = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),
                                          np.random.permutation(sum_stim_dists[sum_stim_dists.nonzero()].flatten()))
    temp_corr_sh_dat = pd.DataFrame(dict(file = file, taste = taste, 
            baseline_end = baseline_end*step_size, rho = temp_corr_sh[0],p = temp_corr_sh[1],
            index = [corr_dat.shape[0]], shuffle = True, pre_stim_window_size = (baseline_end - baseline_start)*step_size,laser = laser))
    return temp_corr_sh_dat

def baseline_stimulus_correlation_acc(off_firing, baseline_window, stimulus_time,
                                  stimulus_window_size, step_size, shuffle_repeats, file, laser):
    corr_dat = pd.DataFrame() 
    all_pre_stim_dist = []
    all_stim_dist = []
    for taste in range(len(off_firing)):
        data = off_firing[taste]
        
        baseline_start = int(baseline_window[0]/step_size)
        baseline_end = int(baseline_window[1]/step_size)
        stim_start = int(stimulus_time/step_size)
        stim_end = int((stimulus_time + stimulus_window_size)/step_size)
        
        pre_dat = data[:,:,baseline_start:baseline_end]
        pre_dists = np.zeros((pre_dat.shape[1],pre_dat.shape[1],pre_dat.shape[2]))
        for time_bin in range(pre_dists.shape[2]):
            pre_dists[:,:,time_bin] = dist_mat(pre_dat[:,:,time_bin].T,pre_dat[:,:,time_bin].T)
        pre_stim_dist = np.tril(np.sum(pre_dists,axis = 2))
        all_pre_stim_dist.append(pre_stim_dist)
        
        stim_dat = data[:,:,stim_start:stim_end]
        stim_dists = np.zeros((stim_dat.shape[1],stim_dat.shape[1],stim_dat.shape[2]))
        for time_bin in range(stim_dists.shape[2]):
            stim_dists[:,:,time_bin] = dist_mat(stim_dat[:,:,time_bin].T,stim_dat[:,:,time_bin].T)
        sum_stim_dists = np.tril(np.sum(stim_dists,axis = 2))
        all_stim_dist.append(sum_stim_dists)
        
        temp_corr = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),sum_stim_dists[sum_stim_dists.nonzero()].flatten())
        temp_corr_dat = pd.DataFrame(dict(file = file, taste = taste, 
                baseline_end = baseline_end*step_size, rho = temp_corr[0],p = temp_corr[1],
                index = [corr_dat.shape[0]], shuffle = False, pre_stim_window_size = (baseline_end - baseline_start)*step_size,laser = laser))
        corr_dat = pd.concat([corr_dat, temp_corr_dat])
        
        for repeat in range(shuffle_repeats):
            output = stim_corr_shuffle(pre_stim_dist, sum_stim_dists,baseline_end, baseline_start, step_size, file, taste, laser, corr_dat)
            corr_dat = pd.concat([corr_dat,output])
        
    return corr_dat, all_pre_stim_dist, all_stim_dist

def baseline_stimulus_correlation_mean(off_firing, baseline_window, stimulus_time,
                                  stimulus_window_size, step_size, shuffle_repeats, file, laser):
    corr_dat = pd.DataFrame() 
    all_pre_stim_dist = []
    all_stim_dist = []
    for taste in range(len(off_firing)):
        data = off_firing[taste]
        
        baseline_start = int(baseline_window[0]/step_size)
        baseline_end = int(baseline_window[1]/step_size)
        stim_start = int(stimulus_time/step_size)
        stim_end = int((stimulus_time + stimulus_window_size)/step_size)
        
        mean_pre_stim = np.mean(data[:,:,baseline_start:baseline_end],axis = 2).T #(neurons x trials)
        pre_stim_dist = np.tril(dist_mat(mean_pre_stim,mean_pre_stim)) # Take out upper diagonal to prevent double counting
        all_pre_stim_dist.append(pre_stim_dist)
        
        mean_stim_dat = np.mean(data[:,:,stim_start:stim_end],axis=2).T
        stim_dist = np.tril(dist_mat(mean_stim_dat,mean_stim_dat))
        all_stim_dist.append(stim_dist)
        
        temp_corr = pearsonr(pre_stim_dist[pre_stim_dist.nonzero()].flatten(),stim_dist[stim_dist.nonzero()].flatten())
        temp_corr_dat = pd.DataFrame(dict(file = file, taste = taste, 
                baseline_end = baseline_end*step_size, rho = temp_corr[0],p = temp_corr[1],
                index = [corr_dat.shape[0]], shuffle = False, pre_stim_window_size = (baseline_end - baseline_start)*step_size,laser = laser))
        corr_dat = pd.concat([corr_dat, temp_corr_dat])
        
        for repeat in range(shuffle_repeats):
            output = stim_corr_shuffle(pre_stim_dist, stim_dist,baseline_end, baseline_start, step_size, file, taste, laser, corr_dat)
            corr_dat = pd.concat([corr_dat,output])
        
    return corr_dat, all_pre_stim_dist, all_stim_dist
#############################################
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
    #data_dir = '/media/bigdata/NM_2500/file_%i/' % file
    os.chdir(data_dir)
    # Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
    file_list = os.listdir('./')
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files
    
    # Open up hdf5 file
    hf5 = tables.open_file(hdf5_name, 'r+')
    # Pull out single units
    units_descriptors = hf5.root.unit_descriptor[:]
    chosen_units = np.zeros(units_descriptors.size)
    for i in range(units_descriptors.size):
        if units_descriptors[i][3] == 1:
            chosen_units[i] = 1
    chosen_units = np.nonzero(chosen_units)[0]
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
    on_firing = []
    
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
        off_firing.append(this_off_firing)
    
    for l in range(len(on_spikes)): # How TF do you get nan's from means?
        # [trials, nrns, time]
        this_on_firing = np.zeros((on_spikes[0].shape[0],on_spikes[0].shape[1],firing_len))
        for i in range(this_on_firing.shape[0]):
            for j in range(this_on_firing.shape[1]):
                for k in range(this_on_firing.shape[2]):
                    this_on_firing[i, j, k] = np.mean(on_spikes[l][i, j, step_size*k:step_size*k + window_size])
                    if np.isnan(this_on_firing[i, j, k]):
                        print('found nan')
                        break
        on_firing.append(this_on_firing)
            
    # Normalized firing of every neuron over entire dataset
    
    off_firing_array = np.asarray(off_firing) #(taste x nrn x trial x time)
    for m in range(off_firing_array.shape[1]): # nrn
        min_val = np.min(off_firing_array[:,m,:,:])
        max_val = np.max(off_firing_array[:,m,:,:])
        for l in range(len(off_firing)): #taste
            for n in range(off_firing[0].shape[1]): # trial
                off_firing[l][m,n,:] = (off_firing[l][m,n,:] - min_val)/(max_val-min_val)

    on_firing_array = np.asarray(on_firing) #(taste x nrn x trial x time)
    for m in range(on_firing_array.shape[1]): # nrn
        min_val = np.min(on_firing_array[:,m,:,:])
        max_val = np.max(on_firing_array[:,m,:,:])
        for l in range(len(on_firing)): #taste
            for n in range(on_firing[0].shape[1]): # trial
                on_firing[l][m,n,:] = (on_firing[l][m,n,:] - min_val)/(max_val-min_val)

# =============================================================================
#     # Avg subtracted normalization
#     off_firing_array = np.asarray(off_firing) #(taste x nrn x trial x time)
#     for m in range(off_firing_array.shape[1]): # nrn
#         this_nrn = off_firing_array[:,m,:,:]
#         this_nonzero_nrn = this_nrn[this_nrn.nonzero()]
#         mean_val = np.mean(this_nonzero_nrn)
#         var_val = np.var(this_nonzero_nrn)
#         for l in range(len(off_firing)): #taste
#             for n in range(off_firing[0].shape[1]): # trial
#                 off_firing[l][m,n,:] = (off_firing[l][m,n,:] - mean_val)/var_val
# #####
# =============================================================================
# Running stats on baseline 0 -2000ms
    mat2num = lambda x: x[x.nonzero()].flatten()
    
    corr_dat, all_pre_stim_dist, all_stim_dist =  baseline_stimulus_correlation_mean(
                                                                off_firing = off_firing, 
                                                                baseline_window = (0,2000), 
                                                                stimulus_time = 2000,
                                                                stimulus_window_size = 2000, 
                                                                step_size = 25, 
                                                                shuffle_repeats = 100, 
                                                                file = file, 
                                                                laser = False)
    
# =============================================================================
#     plt.figure()
#     for i in range(4):
#         plt.subplot(2,2,i+1)
#         sns.regplot(mat2num(all_pre_stim_dist[i]),mat2num(all_stim_dist[i]))
#         corr, p = pearsonr(mat2num(all_pre_stim_dist[i]),mat2num(all_stim_dist[i]))
#         plt.title('r^2 %.2f, p %.3f' % (corr**2,p) )
#         plt.suptitle('Off, file %i' % file)
# =============================================================================
        
    corr_dat, all_pre_stim_dist, all_stim_dist =  baseline_stimulus_correlation_mean(
                                                                off_firing = on_firing, 
                                                                baseline_window = (0,2000), 
                                                                stimulus_time = 2000,
                                                                stimulus_window_size = 2000, 
                                                                step_size = 25, 
                                                                shuffle_repeats = 100, 
                                                                file = file, 
                                                                laser = True)
# =============================================================================
#     plt.figure()
#     for i in range(4):
#         plt.subplot(2,2,i+1)
#         sns.regplot(mat2num(all_pre_stim_dist[i]),mat2num(all_stim_dist[i]))
#         corr, p = pearsonr(mat2num(all_pre_stim_dist[i]),mat2num(all_stim_dist[i]))
#         plt.title('r^2 %.2f, p %.3f' % (corr**2,p) )
#         plt.suptitle('On, file %i' % file)
# =============================================================================
        
#   _____                    _       _   _             
#  / ____|                  | |     | | (_)            
# | |     ___  _ __ _ __ ___| | __ _| |_ _  ___  _ __  
# | |    / _ \| '__| '__/ _ \ |/ _` | __| |/ _ \| '_ \ 
# | |___| (_) | |  | | |  __/ | (_| | |_| | (_) | | | |
#  \_____\___/|_|  |_|  \___|_|\__,_|\__|_|\___/|_| |_|
#                                                      
  
    ##############################
    # Create all windows for taking mean baseline
    baseline_window_sizes = np.arange(100,2000,300)
    baseline_window_end = 2000
    baseline_window_start = 0
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
    
    # Run correlation analysis
    # Off_firing
    pool = mp.Pool(processes = mp.cpu_count())
    results = [pool.apply_async(baseline_stimulus_correlation_mean, args = (off_firing, all_baseline_windows[i], stimulus_time,
                                      stimulus_window_size, step_size, shuffle_repeats, file, False)) for i in range(len(all_baseline_windows))]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    for i in range(len(output)):
        corr_dat = pd.concat([corr_dat,output[i][0]])
        
    print('file %i off' % file)
    
    # On_firing
    pool = mp.Pool(processes = mp.cpu_count())
    results = [pool.apply_async(baseline_stimulus_correlation_mean, args = (on_firing, all_baseline_windows[i], stimulus_time,
                                      stimulus_window_size, step_size, shuffle_repeats, file, True)) for i in range(len(all_baseline_windows))]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    for i in range(len(output)):
        corr_dat = pd.concat([corr_dat,output[i][0]])
        
    print('file %i on' % file)

#############################
mean_squared = lambda x: np.mean(x**2)
for file in [6]:#range(1,7):
    g = sns.FacetGrid(corr_dat.query('file == %i and shuffle == False' % file),col = 'taste', 
                      row = 'pre_stim_window_size', hue = 'laser', 
                      sharey = 'row')
    #g.set(ylim=(0,None)
    g.map(sns.regplot,'baseline_end','rho', 
          x_estimator = np.mean, x_ci = 'sd').add_legend()
    g.fig.suptitle('FILE %i' % file)
    #g.fig.suptitle('All files')
    g.savefig('file%i_JY.png' % file)
    plt.close('all')


# =============================================================================
# for file in range(1,6):
#     g = sns.lmplot(x='baseline_end',y='rho',col = 'taste',row = 'pre_stim_window_size',
#                    hue = 'laser', 
#                    data = corr_dat.query('file == %i and shuffle == False' % file),
#                    ylim = (0,1)).add_legend()
#     g.fig.suptitle('FILE %i' % file)
#     #g.fig.suptitle('All files')
#     g.savefig('file%i_JY.png' % file)
#     plt.close('all')
# =============================================================================

###########################   
for file in range(1,7):
    g = sns.lmplot(x='baseline_end',y='rho',col = 'taste',row = 'laser',
                   hue = 'shuffle',  x_estimator = np.mean, x_ci = 'sd',
                   data = corr_dat.query('file == %i and pre_stim_window_size == 1900' % file)).add_legend()
    g.fig.suptitle('FILE %i' % file)
    #g.fig.suptitle('All files')
    g.savefig('file%i_laser_comp.png' % file)
    plt.close('all')
######################################################
# Dimensionality reduction test
# Since the system changes dramatically around stimulus delivery
# If baseline and some time after stim delivery is dimension reduced
    # we should be able to see a jump in the trajectory
    
# Trials reduced individually
plt.imshow(np.mean(off_firing[0],axis=1), interpolation='nearest', aspect='auto')

f, axs = plt.subplots(8,1,figsize=(9,6))
for ax in range(len(axs)):
    axs[ax].imshow(off_firing[0][:,ax,:], interpolation='nearest', aspect='auto')

#############
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture

for trial in range(8):
    #off_f_red = LLE(n_neighbors = 50,n_components=3).fit_transform(np.transpose(off_firing[0][:,trial,:160]))
    off_f_red = PCA(n_components=2).fit_transform(np.transpose(off_firing[0][:,trial,:160]))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    colors  = np.concatenate((np.arange(80),np.arange(len(off_f_red[:,0]),len(off_f_red[:,0])+len(off_f_red[:,0])-80)))
    sizes = np.ones(len(off_f_red[:,0]))*10
    sizes[0], sizes[-1] = 80,80
    p = ax.scatter(off_f_red[:,0],off_f_red[:,1],off_f_red[:,2],c = colors, s = sizes)
    #ax.scatter(off_f_red[0,0],off_f_red[0,1],off_f_red[0,2],c = 'red')
    #ax.scatter(off_f_red[-1,0],off_f_red[-1,1],off_f_red[-1,2],c = 'red')
    ax.plot(off_f_red[:,0],off_f_red[:,1],off_f_red[:,2],linewidth=0.5)
    plt.colorbar(p)
    
# Trials reduced collectively
reduced_dims = 2
orig_data = off_firing[taste][:,:,:160]
reshaped_data = np.empty((orig_data.shape[0],orig_data.shape[1]*orig_data.shape[2]))
unit_len = orig_data.shape[2]
for i in range(orig_data.shape[1]):
    reshaped_data[:,(i*unit_len):(i+1)*unit_len] = orig_data[:,i,:]
    
off_f_red = PCA(n_components= reduced_dims).fit_transform(np.transpose(reshaped_data))
off_f_red_array = np.empty((reduced_dims,orig_data.shape[1],orig_data.shape[2]))
for i in range(orig_data.shape[1]):
    off_f_red_array[:,i,:] = off_f_red[(i*unit_len):(i+1)*unit_len,:].T
    
corr_dat, all_pre_stim_dist, all_stim_dist =  baseline_stimulus_correlation_acc(
                                                            off_firing = [off_f_red_array], 
                                                            baseline_window = (0,2000), 
                                                            stimulus_time = 2000,
                                                            stimulus_window_size = 2000, 
                                                            step_size = 25, 
                                                            shuffle_repeats = 100, 
                                                            file = file, 
                                                            laser = True)

x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y

fig = plt.figure()
for trial in range(15):
    #ax = fig.add_subplot(4,2,trial+1,projection='3d', aspect='auto')
    ax = fig.add_subplot(3,5,trial+1, aspect='auto',)
    colors  = np.concatenate((np.arange(80),np.arange(off_f_red_array.shape[2],(off_f_red_array.shape[2]*2)-80)))
    sizes = np.ones(off_f_red_array.shape[2])*10
    sizes[0], sizes[-1] = 80,80
    #p = ax.scatter(off_f_red_array[0,trial,:],off_f_red_array[1,trial,:],off_f_red_array[2,trial,:],c = colors, s = sizes)
    #p = ax.scatter(off_f_red_array[0,trial,:],off_f_red_array[1,trial,:],c = colors, s = sizes)
    pre_norm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(off_f_red_array[:,trial,range(80)].T)
    post_norm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(off_f_red_array[:,trial,range(80,160)].T)
    plt.xlim((-0.5,0.5))
    plt.ylim((-0.5,0.5))
    pre_plot = multivariate_normal(pre_norm.means_[0], pre_norm.covariances_[0])
    post_plot = multivariate_normal(post_norm.means_[0], post_norm.covariances_[0])
    plt.contour(x, y, pre_plot.pdf(pos),cmap = 'Blues')
    plt.contour(x, y, post_plot.pdf(pos),cmap = 'Greens')
    #ax.plot(off_f_red_array[0,trial,:],off_f_red_array[1,trial,:],off_f_red_array[2,trial,:],linewidth=0.5)
    #ax.plot(off_f_red_array[0,trial,:],off_f_red_array[1,trial,:],linewidth=0.5)
    #plt.colorbar(p)
    plt.title('Trial %i' % trial)
#plt.tight_layout()
plt.figure()
zscore_dist_mat(all_pre_stim_dist[taste])
plt.colorbar()
plt.figure()
zscore_dist_mat(all_stim_dist[taste])
plt.colorbar()

def zscore_dist_mat(x):
    x2 = copy.deepcopy(x)
    x2[x2.nonzero()] = zscore(x2[x2.nonzero()])
    plt.imshow(x2)

# =============================================================================
#     plt.figure()
#     plt.scatter(off_f_red[:,0],off_f_red[:,1],c =np.linspace(1,255,len(off_f_red[:,0])))
#     
# =============================================================================
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


### Baseline similarity across tastes ###
"""
Calculate how different are baselines across different tastes
Look at ?
    : Trajectories
        : Collect all trajectories from baseline and stimulus and reduce dimensions
        : You should see 2 blobs which are different
    : Distances of baselines from eachother compared to stimuli
        : Calculate distances between baseline and stimulus vectors
        : Baseline vectors should be closer that stimulus vectors
"""
all_trajectories = np.empty((off_firing[0].shape[0],off_firing[0].shape[1]*4,off_firing[0].shape[2]))
for taste in range(off_firing_array.shape[0]):
    all_trajectories[:,(taste*15):((taste+1)*15),:] = off_firing_array[taste,:,:,:]
    

# Dimensionality Reduction
dat_long = np.empty((all_trajectories.shape[0],all_trajectories.shape[1]*all_trajectories.shape[2]))

for trial in range(all_trajectories.shape[1]):
    dat_long[:,(trial*all_trajectories.shape[2]):((trial+1)*all_trajectories.shape[2])] = all_trajectories[:,trial,:]

reduced_dat = PCA(n_components=3).fit_transform(np.transpose(dat_long)).T

reduced_dat_array = np.empty((3,all_trajectories.shape[1],all_trajectories.shape[2]))
for trial in range(reduced_dat_array.shape[1]):
    reduced_dat_array[:,trial,:] = reduced_dat[:,(trial*all_trajectories.shape[2]):((trial+1)*all_trajectories.shape[2])]

baseline_trajs = reduced_dat_array[:,:,range(80)]
stimulus_trajs = reduced_dat_array[:,:,range(80,160)]

all_baseline = np.empty((baseline_trajs.shape[0],baseline_trajs.shape[1]*baseline_trajs.shape[2]))
all_stimulus = np.empty((baseline_trajs.shape[0],baseline_trajs.shape[1]*baseline_trajs.shape[2]))
for trial in range(baseline_trajs.shape[1]):
    all_baseline[:,(trial*baseline_trajs.shape[2]):((trial+1)*baseline_trajs.shape[2])] = baseline_trajs[:,trial,:]
    all_stimulus[:,(trial*stimulus_trajs.shape[2]):((trial+1)*stimulus_trajs.shape[2])] = stimulus_trajs[:,trial,:]

# All data points
plot_data = np.concatenate((all_baseline,all_stimulus),axis=1)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
colors  = np.concatenate((np.ones((1,all_baseline.shape[1])),np.multiply(np.ones((1,all_stimulus.shape[1])),2)),axis=1)
base = ax.scatter(all_baseline[0,:],all_baseline[1,:],all_baseline[2,:],c=range(len(all_baseline[0,:])),label='baseline')
stim = ax.scatter(all_stimulus[0,:],all_stimulus[1,:],all_stimulus[2,:],c='blue',label='stimulus')
plt.colorbar(base)

# Mean trials
mean_baseline_trajs = np.mean(baseline_trajs,axis=2)
mean_stimulus_trajs = np.mean(stimulus_trajs,axis=2)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
colors  = np.concatenate((np.ones((1,all_baseline.shape[1])),np.multiply(np.ones((1,all_stimulus.shape[1])),2)),axis=1)
base = ax.scatter(mean_baseline_trajs[0,:],mean_baseline_trajs[1,:],mean_baseline_trajs[2,:],c=range(len(mean_baseline_trajs[0,:])),label='baseline',cmap='hsv')
ax.plot(mean_baseline_trajs[0,:],mean_baseline_trajs[1,:],mean_baseline_trajs[2,:],linewidth=0.5)
stim = ax.scatter(mean_stimulus_trajs[0,:],mean_stimulus_trajs[1,:],mean_stimulus_trajs[2,:],c='black',label='stimulus')
plt.colorbar(base)

# Distance
distance_data = np.concatenate((all_trajectories[:,:,range(80)],all_trajectories[:,:,range(80,160)]),axis=1)
distance_array = np.empty((distance_data.shape[1],distance_data.shape[1],distance_data.shape[2]))
for time in range(distance_array.shape[2]):
    distance_array[:,:,time] = dist_mat(distance_data[:,:,time].T,distance_data[:,:,time].T)
fin_acc_dists = np.mean(distance_array,axis=2)

mean_distance_data = np.mean(distance_data,axis=2)
fin_mean_dists = dist_mat(mean_distance_data.T,mean_distance_data.T)