import os
import numpy as np
import tables
import copy
import pandas as pd
from scipy.spatial import distance_matrix as dist_mat
from scipy.stats import pearsonr
import multiprocessing as mp
import pylab as plt
from scipy.special import gamma
from scipy.stats import zscore

#from BAKS import BAKS

this_dir = os.getcwd()
os.chdir(this_dir)

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
    def __init__(self, data_dir, use_chosen_units, file_id = None, unit_type = None):
        
        """
        data_dirs : where to look for hdf5 file
            : get_data() loads data from this directory
        """
        self.data_dir =         data_dir
        self.units_descriptors = None
        self.chosen_units =     None
        self.use_chosen_units = use_chosen_units
        self.file_id =          file_id
        self.data_frame = pd.DataFrame()
        self.palatability_ranks = None
        
        self.firing_rate_params = {
            'step_size' :   None,
            'window_size' : None,
            'total_time' :  None,
            'calc_type' : None,      # Either 'baks' or 'conv'
            'baks_len' : None
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
        """
        Extract spike arrays from specified HD5 files
        """
        # Find file and open it
        file_list = os.listdir(self.data_dir)
        hdf5_name = ''
        for files in file_list:
            if files[-2:] == 'h5':
                hdf5_name = files
                
        hf5 = tables.open_file(self.data_dir + '/' + hdf5_name, 'r+')
        
        # Lists for spikes and trials from different tastes
        self.spikes = []
        self.dig_in_order = []
        self.off_spikes = []
        self.on_spikes = []
        self.all_off_trials = []
        self.all_on_trials = []
        
        # If use_chosen_units == True, pick out single units
        if self.use_chosen_units:
            self.units_descriptors = hf5.root.unit_descriptor[:]
            chosen_units = np.zeros(self.units_descriptors.size)
            for i in range(self.units_descriptors.size):
                if self.units_descriptors[i][3] == 1:
                    chosen_units[i] = 1
            self.chosen_units = np.nonzero(chosen_units)[0]
        else:
            self.units_descriptors = hf5.root.unit_descriptor[:]
            self.chosen_units = np.ones(self.units_descriptors.size)
        
        # Iterate through tastes and extract spikes from laser on and off conditions
        # If no array for laser durations, put everything in laser off
        
        dig_in_gen = hf5.root.spike_trains._f_iter_nodes()
        for taste in range(len(hf5.root.spike_trains._f_list_nodes())):
            
            this_dig_in = next(dig_in_gen)
            if 'dig_in' in this_dig_in.__str__():
                self.dig_in_order.append(this_dig_in.__str__())
                self.spikes.append(this_dig_in.spike_array[:])
                
                # Swap axes to make it (neurons x trials x time)
                self.spikes[taste] = np.swapaxes(self.spikes[taste], 0, 1)
                
                # If use_chosen_spikes specified
                # Slice out the required portion of the spike array
                if self.use_chosen_units:
                    self.spikes[taste] = self.spikes[taste][self.chosen_units, :, :]
                
                if this_dig_in.__contains__('laser_durations'):
                    on_trials = np.where(this_dig_in.laser_durations[:] > 0.0)[0]
                    off_trials = np.where(this_dig_in.laser_durations[:] == 0.0)[0]
                
                    self.all_off_trials.append(off_trials + (taste * len(off_trials) * 2))
                    self.all_on_trials.append(on_trials + (taste * len(on_trials) * 2))
                
                    self.off_spikes.append(self.spikes[taste][:, off_trials, :])
                    self.on_spikes.append(self.spikes[taste][:, on_trials, :])
                    
                else:
                    off_trials = np.arange(0,self.spikes[taste].shape[1])
                    self.all_off_trials.append(off_trials + (taste * len(off_trials)))
                    self.off_spikes.append(self.spikes[taste][:, off_trials, :])
                    self.on_spikes.append(None)
                
        if len(self.all_off_trials) > 0: self.all_off_trials = np.concatenate(np.asarray(self.all_off_trials))
        if len(self.all_on_trials) > 0: 
            self.all_on_trials = np.concatenate(np.asarray(self.all_on_trials))
            self.laser_exists = True
        else: 
            self.laser_exists = False
        
        hf5.close()
    
    
    @staticmethod
    def _calc_firing_rates(step_size, window_size, total_time, spike_array):
        """
        spike_array :: params :: 4D array with time as last dimension
        """
        bin_inds = (0,window_size)
        total_bins = int((total_time - window_size + 1) / step_size) + 1
        bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
                for step in np.arange(total_bins)*step_size ]

        firing_rate = np.empty((spike_array.shape[0],spike_array.shape[1],total_bins))
        for bin_inds in bin_list:
            firing_rate[:,:,bin_inds[0]//step_size] = \
                    np.sum(spike_array[:,:,bin_inds[0]:bin_inds[1]], axis=-1)

        return firing_rate

    def get_firing_rates(self):
        """
        Converts spikes to firing rates
        """
        
        off_spikes = self.off_spikes
        if self.laser_exists:
            on_spikes = self.on_spikes
        off_firing = []
        on_firing = []
        normal_off_firing = []
        normal_on_firing = []
        
        ### OFF Firing ###
        if self.firing_rate_params['calc_type'] == 'conv':
            
            step_size = self.firing_rate_params['step_size']
            window_size = self.firing_rate_params['window_size']
            tot_time = self.firing_rate_params['total_time']
            firing_len = int((tot_time-window_size)/step_size)-1

            self.off_firing = [self._calc_firing_rates(step_size = step_size,
                                                window_size = window_size,
                                                total_time = tot_time,
                                                spike_array = spikes)
                                for spikes in self.off_spikes ]

            # How many time-steps after binning
            
            
            #if 'step' in calc_type:
            ## Step-wise moving window calculation of firing rates
#            for l in range(len(off_spikes)): # taste
#                this_off_firing = np.zeros((off_spikes[0].shape[0],off_spikes[0].shape[1],firing_len))
#                for i in range(this_off_firing.shape[0]): # nrns
#                    for j in range(this_off_firing.shape[1]): # trials
#                        for k in range(this_off_firing.shape[2]): # time
#                            this_off_firing[i, j, k] = np.mean(off_spikes[l][i, j, step_size*k:step_size*k + window_size])
#                            if np.isnan(this_off_firing[i, j, k]):
#                                print('found nan')
#                                break
#                off_firing.append(this_off_firing)
#            
#            self.off_firing = off_firing
            
            if self.laser_exists:
                self.on_firing= [self._calc_firing_rates(step_size = step_size,
                                                    window_size = window_size,
                                                    total_time = tot_time,
                                                    spike_array = spikes)
                                    for spikes in self.on_spikes]
                
#                for l in range(len(on_spikes)):
#                    this_on_firing = np.zeros((on_spikes[0].shape[0],on_spikes[0].shape[1],firing_len))
#                    for i in range(this_on_firing.shape[0]):
#                        for j in range(this_on_firing.shape[1]):
#                            for k in range(this_on_firing.shape[2]):
#                                this_on_firing[i, j, k] = np.mean(on_spikes[l][i, j, step_size*k:step_size*k + window_size])
#                                if np.isnan(this_on_firing[i, j, k]):
#                                    print('found nan')
#                                    break
#                    on_firing.append(this_on_firing)
#                
#                self.on_firing = on_firing
                
        elif self.firing_rate_params['calc_type'] == 'baks':
            
            #if 'baks' in calc_type:
            tot_time = self.firing_rate_params['total_time']
            firing_len = self.firing_rate_params['baks_len']
            
            for l in range(len(off_spikes)): # taste
                this_off_firing = np.zeros((off_spikes[0].shape[0],off_spikes[0].shape[1],firing_len))
                for i in range(this_off_firing.shape[0]): # nrns
                    for j in range(this_off_firing.shape[1]): # trials
                            this_off_firing[i, j, :] = BAKS(np.where(off_spikes[l][i,j,:])[0]/1000, np.linspace(0,tot_time/1000,firing_len))
                            if sum(np.isnan(this_off_firing[i, j, :])):
                                print('found nan')
                                break
                off_firing.append(this_off_firing)
            self.off_firing = off_firing
            
            if self.laser_exists:
                
                for l in range(len(on_spikes)):
                    this_on_firing = np.zeros((on_spikes[0].shape[0],on_spikes[0].shape[1],firing_len))
                    for i in range(this_on_firing.shape[0]):
                        for j in range(this_on_firing.shape[1]):
                            this_on_firing[i, j, :] = BAKS(np.where(on_spikes[l][i,j,:])[0]/1000, np.linspace(0,tot_time/1000,firing_len))
                            if sum(np.isnan(this_on_firing[i, j, :])):
                                print('found nan')
                                break
                    on_firing.append(this_on_firing)
                
                self.on_firing = on_firing
                
        else:
            print('Please specify either "conv" or "baks" for "calc_type"')
        
        
        #(taste x nrn x trial x time)
        if self.laser_exists:
            all_firing_array = np.concatenate(
                    (np.asarray(self.off_firing),np.asarray(self.on_firing)), axis = 2)
        else:
            all_firing_array = np.asarray(off_firing)
        self.all_firing_array = all_firing_array
        
        
    def get_normalized_firing(self):
        """
        Converts spikes to firing rates
        """
        # =============================================================================
        # Normalize firing
        # =============================================================================
        all_firing_array = self.all_firing_array
        normal_off_firing = copy.deepcopy(self.off_firing)
        
        # Normalized firing of every neuron over entire dataset
        for m in range(all_firing_array.shape[1]): # nrn
            min_val = np.min(all_firing_array[:,m,:,:]) # Find min and max vals in entire dataset
            max_val = np.max(all_firing_array[:,m,:,:])
            if not (max_val == 0):
                for l in range(len(normal_off_firing)): #taste
                    for n in range(normal_off_firing[0].shape[1]): # trial
                        normal_off_firing[l][m,n,:] = (normal_off_firing[l][m,n,:] - min_val)/(max_val-min_val)
            else:
                for l in range(len(normal_off_firing)): #taste
                    normal_off_firing[l][m,:,:] = 0
                
        self.normal_off_firing = normal_off_firing
        all_off_firing_array = np.asarray(self.normal_off_firing)
        new_shape = (all_off_firing_array.shape[1],
                     all_off_firing_array.shape[2]*all_off_firing_array.shape[0],
                     all_off_firing_array.shape[3])
        
        new_all_off_firing_array = np.empty(new_shape)
        
        for taste in range(all_off_firing_array.shape[0]):
                new_all_off_firing_array[:, taste*all_off_firing_array.shape[2]:(taste+1)*all_off_firing_array.shape[2],:] = all_off_firing_array[taste,:,:,:] 
        
        self.all_normal_off_firing = new_all_off_firing_array
        
        ### ON FIRING ###
        
        # If on_firing exists, then calculate on firing
        if self.laser_exists:
            
# =============================================================================
#             for l in range(len(on_spikes)):
#                 this_on_firing = np.zeros((on_spikes[0].shape[0],on_spikes[0].shape[1],firing_len))
#                 for i in range(this_on_firing.shape[0]):
#                     for j in range(this_on_firing.shape[1]):
#                         for k in range(this_on_firing.shape[2]):
#                             this_on_firing[i, j, k] = np.mean(on_spikes[l][i, j, step_size*k:step_size*k + window_size])
#                             if np.isnan(this_on_firing[i, j, k]):
#                                 print('found nan')
#                                 break
#                 on_firing.append(this_on_firing)
#             
#             self.on_firing = on_firing
# =============================================================================
            
            normal_on_firing = copy.deepcopy(self.on_firing)
            
            for m in range(all_firing_array.shape[1]): # nrn
                min_val = np.min(all_firing_array[:,m,:,:])
                max_val = np.max(all_firing_array[:,m,:,:])
                if not (max_val == 0):
                    for l in range(len(normal_on_firing)): #taste
                        for n in range(normal_on_firing[0].shape[1]): # trial
                            normal_on_firing[l][m,n,:] = (normal_on_firing[l][m,n,:] - min_val)/(max_val-min_val)
                else:
                    for l in range(len(normal_on_firing)): #taste
                        normal_on_firing[l][m,:,:] = 0
                    
            self.normal_on_firing = normal_on_firing
            all_on_firing_array = np.asarray(self.normal_on_firing)
            new_all_on_firing_array = np.empty(new_shape)
    
            for taste in range(all_off_firing_array.shape[0]):
                new_all_on_firing_array[:, taste*all_on_firing_array.shape[2]:(taste+1)*all_on_firing_array.shape[2],:] = all_on_firing_array[taste,:,:,:]
                            
            self.all_normal_on_firing = new_all_on_firing_array
        
    def firing_rate_comparison(self, num_trials):
        """
        Compares firing rate calculation using moving widnow and baks for a random number of trials
        """
        # Pick out random number of trials
        spikes = np.asarray(self.spikes)
        spikes_long = np.reshape(spikes,(np.prod(list(spikes.shape[:-1])),spikes.shape[-1]))
        chosen_spikes = spikes_long[np.random.randint(0,spikes_long.shape[0],num_trials),:]
        
        step_size = self.firing_rate_params['step_size']
        window_size = self.firing_rate_params['window_size']
        tot_time = self.firing_rate_params['total_time']
        firing_len = int((tot_time-window_size)/step_size)-1 # How many time-steps after binning
        
        baks_rate = np.zeros((num_trials,self.firing_rate_params['baks_len']))
        for trial in range(baks_rate.shape[0]):
            baks_rate[trial,:] = self.BAKS(np.where(chosen_spikes[trial,:])[0]/1000, np.linspace(0,tot_time/1000,self.firing_rate_params['baks_len']))
        
        conv_rate = np.zeros((num_trials,firing_len))
        for trial in range(conv_rate.shape[0]):
            for time in range(conv_rate.shape[1]): # time
                conv_rate[trial,time] = np.mean(chosen_spikes[trial, step_size*time:step_size*time + window_size])
        
        square_len = np.int(np.ceil(np.sqrt(num_trials)))
        
        fig, ax = plt.subplots(square_len,square_len)
        
        nd_idx_objs = []
        for dim in range(ax.ndim):
            this_shape = np.ones(len(ax.shape))
            this_shape[dim] = ax.shape[dim]
            nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(ax.shape[dim]),this_shape.astype('int')), ax.shape).flatten())

        for trial in range(num_trials):
            plt.sca(ax[nd_idx_objs[0][trial],nd_idx_objs[1][trial]])
            raster(chosen_spikes[trial,:])
            plt.plot(np.linspace(0,tot_time,firing_len),conv_rate[trial,:]/np.max(conv_rate[trial,:]))
            plt.plot(np.linspace(0,tot_time,self.firing_rate_params['baks_len']),baks_rate[trial,:]/np.max(baks_rate[trial,:]))
        
    def firing_overview(self,dat_set, zscore_bool = False, subtract = False, cmap = 'jet'):
        # dat_set takes string values: 'on', 'off'
        data = eval('self.all_normal_%s_firing' % dat_set)
        
        # Subtract the average component of firing to
        # (hopefully) leave the taste discriminative components
        if subtract:
            data = data - np.mean(data,axis = 1)[:,np.newaxis,:]
        
        # Zscore the firing of every neuron for visualization
        if zscore_bool:
            data = zscore(data, axis = 1)

        num_nrns = data.shape[0]
        t_vec = np.linspace(start=0, stop = \
                self.firing_rate_params['total_time'], num = data.shape[-1])

        # Plot firing rates
        square_len = np.int(np.ceil(np.sqrt(num_nrns)))
        fig, ax = plt.subplots(square_len,square_len)
        
        nd_idx_objs = []
        for dim in range(ax.ndim):
            this_shape = np.ones(len(ax.shape))
            this_shape[dim] = ax.shape[dim]
            nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(ax.shape[dim]),this_shape.astype('int')), ax.shape).flatten())
        
        for nrn in range(num_nrns):
            plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
            plt.gca().set_title(nrn)
            plt.gca().pcolormesh(t_vec, 
                    np.arange(data.shape[1]), 
                    data[nrn,:,:],
                    cmap = cmap)
            #self.imshow(data[nrn,:,:])
        plt.show()
            
            
        
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
            rho_sh_vec[repeat], p_sh_vec[repeat] = pearsonr(np.random.permutation(pre_mat[indices]), stim_mat[indices])
        
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
    
    def imshow(self,x):
        """
        Decorator function for more viewable firing rate heatmaps
        """
        plt.imshow(x,interpolation='nearest',aspect='auto')
        
# =============================================================================
#     def get_hmm_probs(self):
#         """
#         Extract spike arrays from specified HD5 files
#         **Chosen units currently only selects single units
#         """
#         
#         file_list = os.listdir(self.data_dir)
#         hdf5_name = ''
#         for files in file_list:
#             if files[-2:] == 'h5':
#                 hdf5_name = files
#                 
#         hf5 = tables.open_file(self.data_dir + hdf5_name, 'r+')
#         
#         self.off_var_probs
#         self.on_var_probs
#         self.off_maps_probs
#         self.on_map_probs
#         
# =============================================================================

    def get_dataframe(self):
        
        for taste in range(4):
            off_corr_dat = pd.DataFrame(dict(
                    file = self.file_id, 
                    taste = taste, 
                    baseline_end = self.correlation_params['baseline_end_time'], 
                    rho = self.off_corr['rho'][taste],
                    p = self.off_corr['p'][taste],
                    index = [0], 
                    shuffle = False, 
                    baseline_window_size = self.correlation_params['baseline_end_time'] - self.correlation_params['baseline_start_time'],
                    stimulus_end = self.correlation_params['stimulus_end_time'],
                    stim_window_size = self.correlation_params['stimulus_end_time'] - self.correlation_params['stimulus_start_time'],
                    laser = False))
            
            off_corr_sh_dat = pd.DataFrame(dict(
                    file = self.file_id, 
                    taste = taste, 
                    baseline_end = self.correlation_params['baseline_end_time'], 
                    rho = self.off_corr['rho_shuffle'][taste],
                    p = self.off_corr['p_shuffle'][taste],
                    index = range(self.off_corr['p_shuffle'][taste].size), 
                    shuffle = True, 
                    baseline_window_size = self.correlation_params['baseline_end_time'] - self.correlation_params['baseline_start_time'],
                    stimulus_end = self.correlation_params['stimulus_end_time'],
                    stim_window_size = self.correlation_params['stimulus_end_time'] - self.correlation_params['stimulus_start_time'],
                    laser = False))            
            
            on_corr_dat = pd.DataFrame(dict(
                    file = self.file_id, 
                    taste = taste, 
                    baseline_end = self.correlation_params['baseline_end_time'], 
                    rho = self.on_corr['rho'][taste],
                    p = self.on_corr['p'][taste],
                    index = [0], 
                    shuffle = False, 
                    baseline_window_size = self.correlation_params['baseline_end_time'] - self.correlation_params['baseline_start_time'],
                    stimulus_end = self.correlation_params['stimulus_end_time'],
                    stim_window_size = self.correlation_params['stimulus_end_time'] - self.correlation_params['stimulus_start_time'],
                    laser = True))
            
            on_corr_sh_dat = pd.DataFrame(dict(
                    file = self.file_id, 
                    taste = taste, 
                    baseline_end = self.correlation_params['baseline_end_time'], 
                    rho = self.on_corr['rho_shuffle'][taste],
                    p = self.on_corr['p_shuffle'][taste],
                    index = range(self.on_corr['p_shuffle'][taste].size),
                    shuffle = True, 
                    baseline_window_size = self.correlation_params['baseline_end_time'] - self.correlation_params['baseline_start_time'],
                    stimulus_end = self.correlation_params['stimulus_end_time'],
                    stim_window_size = self.correlation_params['stimulus_end_time'] - self.correlation_params['stimulus_start_time'],
                    laser = True))   
            
            self.data_frame = pd.concat([self.data_frame, off_corr_dat, on_corr_dat, off_corr_sh_dat, on_corr_sh_dat])
        self.data_frame = self.data_frame.drop_duplicates()
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
