import os
import numpy as np
import tables
import copy
import pandas as pd
from scipy.spatial import distance_matrix as dist_mat
from scipy.stats import pearsonr
import multiprocessing as mp
#  ______       _                      _____        _         
# |  ____|     | |                    |  __ \      | |        
# | |__   _ __ | |__  _   _ ___ ______| |  | | __ _| |_ __ _  
# |  __| | '_ \| '_ \| | | / __|______| |  | |/ _` | __/ _` | 
# | |____| |_) | | | | |_| \__ \      | |__| | (_| | || (_| | 
# |______| .__/|_| |_|\__, |___/      |_____/ \__,_|\__\__,_| 
#        | |           __/ |                                  
#        |_|          |___/ 

"""
Make a class to streamline data analysis from multiple files
Class has a container for data from different files and functions for analysis
Functions in class can autoload data from specified files according to specified paramters
E.g. whether to take single units, fast spiking etc (as this data is already in the hdf5 file)
"""

class ephys_data():
    def __init__(self, data_dir, file_id = None, unit_type = None):
        """
        data_dirs : where to look for hdf5 file
            : get_data() loads data from this directory
        """
        self.data_dir =         data_dir
        self.unit_descriptors = None
        self.chosen_units =     None
        self.file_id =          file_id
        
        self.firing_rate_params = {
            'step_size' :   None,
            'window_size' : None,
            'total_time' :  None
                }
        
        self.correlation_params = {
            'stimulus_start_time' :     None,
            'stimulus_end_time' :       None,
            'data_binning_step_size' :  self.firing_rate_params['step_size'],
            'baseline_start_time' :     None,
            'baseline_end_time' :       None,
            'baseline_window_sizes' :   None,
            'shuffle_repeats' :         None,
            'accumulated' :             None
                }
# =============================================================================
#         if unit_type == None:
#             user_input = input('fast_spiking, regular_spiking, single_unit:\n')
#             self.unit_type = (int(user_input[0]),int(user_input[1]),int(user_input[2]))
#         else:
#             self.unit_type = unit_type
# =============================================================================
    
        
    def get_data(self):
        file_list = os.listdir(self.data_dir)
        hdf5_name = ''
        for files in file_list:
            if files[-2:] == 'h5':
                hdf5_name = files
                
        hf5 = tables.open_file(self.data_dir + hdf5_name, 'r+')
        
        self.units_descriptors = hf5.root.unit_descriptor[:]
        chosen_units = np.zeros(self.units_descriptors.size)
        for i in range(self.units_descriptors.size):
            if self.units_descriptors[i][3] == 1:
                chosen_units[i] = 1
        self.chosen_units = np.nonzero(chosen_units)[0]
# =============================================================================
#         chosen_units_log = np.zeros((self.unit_type.size,self.units_descriptors.size))
#         for condition in range(self.unit_type.size):
#             for unit in range(self.units_descriptors.size):
#                 if self.units_descriptors[unit][condition+1] == 1:
#                     chosen_units_log[i] = 1
#             self.chosen_units = np.nonzero(chosen_units)[0]
# =============================================================================
        self.spikes = []
        self.off_spikes = []
        self.on_spikes = []
        self.all_off_trials = []
        self.all_on_trials = []
        
        all_off_trials = []
        all_on_trials = []
        
        
        for taste in range(4):
            exec('self.spikes.append(hf5.root.spike_trains.dig_in_%i.spike_array[:])' % taste)
        
            # Slice out the required portion of the spike array, and bin it
            self.spikes[taste] = self.spikes[taste][:, self.chosen_units, :]
            self.spikes[taste] = np.swapaxes(self.spikes[taste], 0, 1)
        
            dig_in = hf5.get_node('/spike_trains/dig_in_%i/' % taste)
            on_trials = np.where(dig_in.laser_durations[:] > 0.0)[0]
            off_trials = np.where(dig_in.laser_durations[:] == 0.0)[0]
        
            all_off_trials.append(off_trials + (taste * len(off_trials) * 2))
            all_on_trials.append(on_trials + (taste * len(on_trials) * 2))
        
            self.off_spikes.append(self.spikes[taste][:, off_trials, :])
            self.on_spikes.append(self.spikes[taste][:, on_trials, :])
        
        
        self.all_off_trials = np.concatenate(np.asarray(all_off_trials))
        self.all_on_trials = np.concatenate(np.asarray(all_on_trials))
        


    def get_firing_rates(self):
        #[assert(self.firing_rate_params.values()[i] is not []) for i in self.firing_rate_params.values()]
        step_size = self.firing_rate_params['step_size']
        window_size = self.firing_rate_params['window_size']
        tot_time = self.firing_rate_params['total_time']
        firing_len = int((tot_time-window_size)/step_size)-1 # How many time-steps after binning
        off_spikes = self.off_spikes # list contraining arrays of dims [nrns, trials, time]
        on_spikes = self.on_spikes # list contraining arrays of dims [nrns, trials, time]
        off_firing = []
        on_firing = []
        normal_off_firing = []
        normal_on_firing = []
        
        ## Moving window calculation of firing rates
        for l in range(len(off_spikes)): # taste
            this_off_firing = np.zeros((off_spikes[0].shape[0],off_spikes[0].shape[1],firing_len))
            for i in range(this_off_firing.shape[0]): # nrns
                for j in range(this_off_firing.shape[1]): # trials
                    for k in range(this_off_firing.shape[2]): # time
                        this_off_firing[i, j, k] = np.mean(off_spikes[l][i, j, step_size*k:step_size*k + window_size])
                        if np.isnan(this_off_firing[i, j, k]):
                            print('found nan')
                            break
            off_firing.append(this_off_firing)
        
        for l in range(len(on_spikes)):
            this_on_firing = np.zeros((on_spikes[0].shape[0],on_spikes[0].shape[1],firing_len))
            for i in range(this_on_firing.shape[0]):
                for j in range(this_on_firing.shape[1]):
                    for k in range(this_on_firing.shape[2]):
                        this_on_firing[i, j, k] = np.mean(on_spikes[l][i, j, step_size*k:step_size*k + window_size])
                        if np.isnan(this_on_firing[i, j, k]):
                            print('found nan')
                            break
            on_firing.append(this_on_firing)
        
        self.off_firing = off_firing
        self.on_firing = on_firing
        
        normal_off_firing = copy.deepcopy(off_firing)
        normal_on_firing = copy.deepcopy(on_firing)
        
        # Normalized firing of every neuron over entire dataset
        off_firing_array = np.asarray(normal_off_firing) #(taste x nrn x trial x time)
        for m in range(off_firing_array.shape[1]): # nrn
            min_val = np.min(off_firing_array[:,m,:,:]) # Find min and max vals in entire dataset
            max_val = np.max(off_firing_array[:,m,:,:])
            for l in range(len(normal_off_firing)): #taste
                for n in range(normal_off_firing[0].shape[1]): # trial
                    normal_off_firing[l][m,n,:] = (normal_off_firing[l][m,n,:] - min_val)/(max_val-min_val)
    
        on_firing_array = np.asarray(normal_on_firing) #(taste x nrn x trial x time)
        for m in range(on_firing_array.shape[1]): # nrn
            min_val = np.min(on_firing_array[:,m,:,:])
            max_val = np.max(on_firing_array[:,m,:,:])
            for l in range(len(normal_on_firing)): #taste
                for n in range(normal_on_firing[0].shape[1]): # trial
                    normal_on_firing[l][m,n,:] = (normal_on_firing[l][m,n,:] - min_val)/(max_val-min_val)
            
        self.normal_off_firing = normal_off_firing
        self.normal_on_firing = normal_on_firing
        
        all_off_firing_array = np.asarray(self.normal_off_firing)
        all_on_firing_array = np.asarray(self.normal_on_firing)
        new_shape = (all_off_firing_array.shape[1],
                     all_off_firing_array.shape[2]*all_off_firing_array.shape[0],
                     all_off_firing_array.shape[3])
        new_all_off_firing_array = np.empty(new_shape)
        new_all_on_firing_array = np.empty(new_shape)
        
        for taste in range(all_off_firing_array.shape[0]):
            new_all_off_firing_array[:, taste*all_off_firing_array.shape[2]:(taste+1)*all_off_firing_array.shape[2],:] = all_off_firing_array[taste,:,:,:]                                  
            new_all_on_firing_array[:, taste*all_on_firing_array.shape[2]:(taste+1)*all_on_firing_array.shape[2],:] = all_on_firing_array[taste,:,:,:]
                                     
        self.all_off_firing = new_all_off_firing_array
        self.all_on_firing = new_all_on_firing_array
        
# =============================================================================
#     # WILL PROBABLY GET DEPRECATED    
#     def get_baseline_windows(self):
#         baseline_window_sizes = self.correlation_params['baseline_window_sizes']
#         baseline_start_time = self.correlation_params['baseline_start_time']
#         baseline_end_time = self.correlation_params['baseline_end_time']
#         all_baseline_windows = []
#         for i in range(len(baseline_window_sizes)):
#             #temp_baseline_windows = np.arange(baseline_window_end,baseline_window_start-baseline_window_sizes[i],-baseline_window_sizes[i])
#             temp_baseline_windows = np.arange(baseline_end_time, baseline_start_time, -100)
#             temp_baseline_windows = temp_baseline_windows[(temp_baseline_windows - baseline_window_sizes[i]) >0]
#             for j in range(0,len(temp_baseline_windows)):
#                 all_baseline_windows.append((temp_baseline_windows[j]- baseline_window_sizes[i],temp_baseline_windows[j]))
#         self.all_baseline_windows = all_baseline_windows
# =============================================================================
        

    def firing_correlation(self, 
                           firing_array, 
                           baseline_window, 
                           stimulus_window,
                           data_step_size = 25, 
                           shuffle_repeats = 100, 
                           accumulated = False):
    
        """
        General function, not bound by object parameters
        Calculates correlations in 2 windows of a firin_array (defined below) 
            according to either accumulated distance or distance of mean points
        PARAMS
        :firing_array: (nrn x trial x time) array of firing rates
        :baseline_window: Tuple of time in ms of what window to take for BASELINE firing
        :stimulus_window: Tuple of time in ms of what window to take for STIMULUS firing
        :data_step_size: Resolution at which the data was binned (if at all)
        :shuffle repeats: How many shuffle repeats to perform for analysis control
        :accumulated:   If True -> will calculate temporally integrated pair-wise distances between all points
                        If False -> will calculate distance between mean of all points  
        """
        # Calculate indices for slicing data
        baseline_start_ind = int(baseline_window[0]/data_step_size)
        baseline_end_ind = int(baseline_window[1]/data_step_size)
        stim_start_ind = int(stimulus_window[0]/data_step_size)
        stim_end_ind = int(stimulus_window[1]/data_step_size)
        
        pre_dat = firing_array[:,:,baseline_start_ind:baseline_end_ind]
        stim_dat = firing_array[:,:,stim_start_ind:stim_end_ind]
        
        if accumulated:
            # Calculate accumulated pair-wise distances for baseline data
            pre_dists = np.zeros((pre_dat.shape[1],pre_dat.shape[1],pre_dat.shape[2]))
            for time_bin in range(pre_dists.shape[2]):
                pre_dists[:,:,time_bin] = dist_mat(pre_dat[:,:,time_bin].T,pre_dat[:,:,time_bin].T)
            sum_pre_dist = np.sum(pre_dists,axis = 2)
            
            # Calculate accumulated pair-wise distances for post-stimulus data
            stim_dists = np.zeros((stim_dat.shape[1],stim_dat.shape[1],stim_dat.shape[2]))
            for time_bin in range(stim_dists.shape[2]):
                stim_dists[:,:,time_bin] = dist_mat(stim_dat[:,:,time_bin].T,stim_dat[:,:,time_bin].T)
            sum_stim_dist = np.sum(stim_dists,axis = 2)
            
            # Remove lower triangle in correlation to not double count points
            indices = np.mask_indices(stim_dat.shape[1], np.triu, 1)
            rho, p = pearsonr(sum_pre_dist[indices], sum_stim_dist[indices])
            
            pre_mat, stim_mat = sum_pre_dist, sum_stim_dist
    
        else:
            # Calculate accumulate pair-wise distances for baseline data
            mean_pre = np.mean(pre_dat,axis = 2)
            mean_pre_dist = dist_mat(mean_pre.T, mean_pre.T)
            
            # Calculate accumulate pair-wise distances for post-stimulus data
            mean_stim = np.mean(stim_dat, axis = 2)
            mean_stim_dist = dist_mat(mean_stim.T, mean_stim.T)
            
            indices = np.mask_indices(stim_dat.shape[1], np.triu, 1)
            rho, p = pearsonr(mean_pre_dist[indices], mean_stim_dist[indices])
            
            pre_mat, stim_mat = mean_pre_dist, mean_stim_dist
        
        
        rho_sh_vec = np.empty(shuffle_repeats)
        p_sh_vec = np.empty(shuffle_repeats)
        for repeat in range(shuffle_repeats):
            rho_sh_vec[repeat], p_sh_vec[repeat] = pearsonr(np.random.permutation(pre_mat), stim_mat)
        
        return rho, p, rho_sh_vec, p_sh_vec, pre_mat, stim_mat
    
    def firing_list_correlation(self, 
                               firing_array_list, # a list containing arrays which can be fed to firing correlation 
                               baseline_window, 
                               stimulus_window,
                               data_step_size = 25, 
                               shuffle_repeats = 100, 
                               accumulated = False):
        """
        Decorator function, not bound by object parameters
        Runs firing_correlation on a list of firing arrays (different tastes)
        """
        rho_list = []
        p_list = []
        rho_sh_vec_list = []
        p_sh_vec_list = []
        pre_mat_list = []
        stim_mat_list = []
        
        for array in firing_array_list:
            outputs = self.firing_correlation(
                                   array, 
                                   baseline_window, 
                                   stimulus_window,
                                   data_step_size, 
                                   shuffle_repeats, 
                                   accumulated)
            rho_list.append(outputs[0])
            p_list.append(outputs[1])
            rho_sh_vec_list.append(outputs[2])
            p_sh_vec_list.append(outputs[3])
            pre_mat_list.append(outputs[4])
            stim_mat_list.append(outputs[5])
        
        return rho_list, p_list, rho_sh_vec_list, p_sh_vec_list, pre_mat_list, stim_mat_list
            
    def get_correlations(self):
        """
        Object function bound to parameters specified
        Runs firing_list_correlation on on and off firing lists
        """
        off_outputs = self.firing_list_correlation( 
                                        firing_array_list = self.off_firing, # a list containing arrays which can be fed to firing correlation 
                                        baseline_window = (self.correlation_params['baseline_start_time'],
                                                           self.correlation_params['baseline_end_time']), 
                                        stimulus_window = (self.correlation_params['stimulus_start_time'],
                                                           self.correlation_params['stimulus_end_time']),
                                        data_step_size = self.firing_rate_params['step_size'], 
                                        shuffle_repeats = self.correlation_params['shuffle_repeats'], 
                                        accumulated = self.correlation_params['accumulated'])
        
        self.off_corr = {
            'rho' :         off_outputs[0],
            'p' :           off_outputs[1],
            'rho_shuffle' : off_outputs[2],
            'p_shuffle' :   off_outputs[3],
            'pre_dists' :   off_outputs[4],
            'stim_dists' :  off_outputs[5],
                }
        
        on_outputs = self.firing_list_correlation( 
                                        firing_array_list = self.on_firing, # a list containing arrays which can be fed to firing correlation 
                                        baseline_window = (self.correlation_params['baseline_start_time'],
                                                           self.correlation_params['baseline_end_time']), 
                                        stimulus_window = (self.correlation_params['stimulus_start_time'],
                                                           self.correlation_params['stimulus_end_time']),
                                        data_step_size = self.firing_rate_params['step_size'], 
                                        shuffle_repeats = self.correlation_params['shuffle_repeats'], 
                                        accumulated = self.correlation_params['accumulated'])
        
        self.on_corr = {
            'rho' :         on_outputs[0],
            'p' :           on_outputs[1],
            'rho_shuffle' : on_outputs[2],
            'p_shuffle' :   on_outputs[3],
            'pre_dists' :   on_outputs[4],
            'stim_dists' :  on_outputs[5],
                }
        
# =============================================================================
#   # WILL PROBABLY GET DEPRECATED
#     def get_correlations(self):
#         # Run correlation analysis over all defined baseline windows
#         corr_dat = pd.DataFrame()
#         data = {False:self.normal_off_firing,True:self.normal_on_firing}
#         for key, firing in data.items():
#             
#             pool = mp.Pool(processes = mp.cpu_count())
#             results = [pool.apply_async(self.baseline_stimulus_correlation,
#                                         kwds = dict(
#                                         firing = firing, 
#                                         baseline_start_time = self.all_baseline_windows[i][0],
#                                         baseline_end_time = self.all_baseline_windows[i][1], 
#                                         stimulus_start_time = self.correlation_params['stimulus_start_time'],
#                                         stimulus_end_time = self.correlation_params['stimulus_end_time'], 
#                                         step_size = self.correlation_params['data_binning_step_size'], 
#                                         shuffle_repeats = self.correlation_params['shuffle_repeats'], 
#                                         file = self.file_id, 
#                                         laser = key)) for i in range(len(self.all_baseline_windows))]
# # =============================================================================
# #                                         args = (firing, 
# #                                                 self.all_baseline_windows[i][0],
# #                                                 self.all_baseline_windows[i][1], 
# #                                                 self.correlation_params['stimulus_start_time'],
# #                                                 self.correlation_params['stimulus_end_time'], 
# #                                                 self.correlation_params['data_binning_step_size'], 
# #                                                 self.correlation_params['shuffle_repeats'], 
# #                                                 self.file_id, 
# #                                                 key)) for i in range(len(self.all_baseline_windows))]
# # =============================================================================
#             output = [p.get() for p in results]
#             pool.close()
#             pool.join()
#             
#             for i in range(len(output)):
#                 corr_dat = pd.concat([corr_dat,output[i][0]])
#                 
#         self.corr_dat = corr_dat
# =============================================================================