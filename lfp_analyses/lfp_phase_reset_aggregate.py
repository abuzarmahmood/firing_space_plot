"""
Requirement:
    lfp_coherence_setup.py already run
"""

## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed, cpu_count
import multiprocessing as mp
import shutil
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from visualize import firing_overview, imshow
from scipy.stats import zscore
from shutil import rmtree
from sklearn.linear_model import LinearRegression

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/fastdata/lfp_analyses'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

middle_channels = np.arange(8,24)

# Pull out all terminal groups (leafs) under stft
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    freq_vec = hf5.get_node('/stft','freq_vec')[:]
    time_vec = hf5.get_node('/stft','time_vec')[:]

    phase_node_path_list = [x.__str__().split(" ")[0] \
            for x in hf5.root.stft._f_walknodes() \
            if 'phase_array' in x.__str__()]
    print('Extracting phase info')
    #phase_array_list = [x[:] for x in tqdm(phase_node_list)]
    # Extract all nodes with phase array
    node_path_list = [os.path.dirname(x) for x in phase_node_path_list]
    # Pull parsed_lfp_channel from each array
    parsed_channel_list  = [hf5.get_node(path,'parsed_lfp_channels')[:] \
            for path in node_path_list]


# Define variables to be maintained across files
initial_dir = dir() + ['initial_dir']

for this_node_num in tqdm(range(len(phase_node_path_list))):
    with tables.open_file(data_hdf5_path,'r') as hf5:
        phase_array = hf5.get_node(node_path_list[this_node_num],
                                'phase_array')[:].swapaxes(0,1)

    phase_array_long = np.reshape(phase_array,
            (phase_array.shape[0],-1,*phase_array.shape[3:]))

    # Unwrap the phase array
    phase_array_long = np.unwrap(phase_array_long)

    middle_channels_bool = np.isin(parsed_channel_list[this_node_num],
            middle_channels)

    split_phase_long = [phase_array_long[~middle_channels_bool],\
                        phase_array_long[middle_channels_bool]]

    # For each trial, regress phase in a window that
    # starts at (t_stim - padding_window) and is (fit_window) long
    # to predict phase at time of taste delivery

    def parallelize_over_array(array, index_list, function):
        """
        Applies a function over the specified indices of
        an array
        """
        out_list = Parallel(n_jobs = cpu_count())\
                (delayed(function)(array[index]) for index in tqdm(index_list))
        return out_list


    def calc_linear_prediction(x, trial, t_stim, padding_window, prev_cycles):
        # Find start point for fit data
        start_ind = \
                np.argmin((trial - (trial[t_stim - padding_window] - \
                            prev_cycles*2*np.pi))**2)
        # Fit model from start point up to t_stim - padding_window
        this_fit = LinearRegression().\
                fit(x[start_ind:(t_stim-padding_window),np.newaxis],
                        trial[x[start_ind:(t_stim-padding_window)],np.newaxis])
        # Generate preduction for entire x
        this_prediction = this_fit.predict(x[:,np.newaxis])
        return this_prediction

    def parallel_linear_prediction(trial):
        return calc_linear_prediction(x, 
                                        trial, 
                                        t_stim, 
                                        padding_window, 
                                        prev_cycles)

    # Convert list to array
    def convert_to_array(iterator, iter_inds):
        temp_array  =\
                np.empty(
                    tuple((*(np.max(np.array(iter_inds),axis=0) + 1),
                            *iterator[0].shape)),
                        dtype=np.dtype(iterator[0].flatten()[0]))
        for iter_num, this_iter in tqdm(enumerate(iter_inds)):
            temp_array[this_iter] = iterator[iter_num]
        return temp_array


    index_lists = [list(np.ndindex(array.shape[:-1])) \
            for array in split_phase_long]

    t_stim = 2000
    padding_window = 250 # How many ms of padding before stimulus delivery
    #fit_window = 1000
    prev_cycles = 5 # How many pervious cycles to use for regression
    post_stim_winow = 500

    # Take cycles equal to number of prev_cycles starting padding_window before
    # stimulus delivery
    # This is to try to avoid bleedover from stimulus delivery biasing
    # the fit of the regression
    # Pad end of x with a window to include post-stimulus time
    x = np.arange(t_stim+post_stim_winow)

    pred_split_phase = [parallelize_over_array(array, 
                                                indices, 
                                                parallel_linear_prediction)
                for array, indices in zip(split_phase_long, index_lists)]

    pred_split_phase_arrays = \
            [convert_to_array(this_list, this_inds).squeeze() \
            for this_list, this_inds in zip(pred_split_phase, index_lists)]

    # Visual comparison of predicted and actual phases
    wrapped_split_phase = [x % (2*np.pi) for x in split_phase_long ]
    wrapped_pred_split_phase = [x % (2*np.pi) for x in pred_split_phase_arrays ]

    region, channel, freq = 0,0,1 
    # Sort by predicted phase at stimulus delivery for visualization
    sort_order = np.argsort(\
            wrapped_pred_split_phase[region][channel,:,freq,t_stim])
    fig,ax = plt.subplots(2,1)
    plt.sca(ax[0])
    imshow(wrapped_split_phase[region][channel,:,freq][...,x])
    plt.sca(ax[1])
    imshow(wrapped_pred_split_phase[region][channel,:,freq])
    ax[0].set_title('Actual phase')
    ax[1].set_title('Predicted phase')
    plt.show()

    # Test phase-reset plot
    time_inds = np.arange(1750,2500,100)
    actual_data = wrapped_split_phase[region][channel,:,freq, time_inds].T
    predicted_data = wrapped_pred_split_phase[region][channel,:,freq, time_inds].T
    color_mat = np.broadcast_to(np.arange(actual_data.shape[1])[np.newaxis,:],
                                actual_data.shape)
    fig, ax = plt.subplots()
    ax.scatter(predicted_data.flatten(), 
                actual_data.flatten(), 
                c=color_mat.flatten(), alpha = 1)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


    # Polar plot
    # Add arrows to show the consistency of the phase reset
    cmap = 'viridis'
    mean_actual_vector = np.mean(np.exp(-1.j * actual_data),axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = 'polar')
    ax.grid(False)
    im = ax.scatter(actual_data.flatten(), 
                predicted_data.flatten(),
                c=color_mat.flatten(), alpha = 0.5, cmap = 'viridis')
    cbar = plt.colorbar(im, ticks = range(len(time_inds)))
    cbar.ax.set_yticklabels = time_inds
    norm = matplotlib.colors.Normalize(np.min(color_mat),np.max(color_mat))
    cmap_object= matplotlib.cm.ScalarMappable(cmap = 'viridis', norm = norm)
    for num, arrow in enumerate(mean_actual_vector):
        plt.arrow(0,0,np.angle(arrow),np.abs(arrow)*2*np.pi,
                width = 0.1, lw = 0.5, 
                facecolor = cmap_object.to_rgba(color_mat[0,num]),
                edgecolor = 'k')
    #plt.xlabel('Predicted')
    #plt.ylabel('Actual')
    plt.show()

    # Bar plot showing Predicted vs Actual

    #animal_date_list = node_path_list[this_node_num].split('/')[2:]
    #this_plot_dir = os.path.join(
    #            data_folder,*animal_date_list)


#####################################
#/ ___|  ___ _ __ __ _| |_ ___| |__  
#\___ \ / __| '__/ _` | __/ __| '_ \ 
# ___) | (__| | | (_| | || (__| | | |
#|____/ \___|_|  \__,_|\__\___|_| |_|
#####################################
                                    
    #phase_plot_dir = os.path.join(this_plot_dir,'phase_plots')
    #if os.path.exists(phase_plot_dir):
    #    rmtree(phase_plot_dir)
    #os.makedirs(phase_plot_dir)

    ## Generate phase plots for all trials and channels
    #def save_firing_overview(num,array):
    #    firing_overview(array,t_vec = time_vec, subplot_labels = freq_vec)
    #    fig = plt.gcf()
    #    fig.set_size_inches(8,10)
    #    fig.suptitle("_".join(animal_date_list) + '_Channel{}'.format(num))
    #    fig.savefig(os.path.join(phase_plot_dir,'channel{}'.format(num)))
    #    plt.close(fig)

    #Parallel(n_jobs = mp.cpu_count()-2)(delayed(save_firing_overview)\
    #        (num,array) for num,array in enumerate(phase_array_long.swapaxes(2,1)))

    #for channel_num ,channel in enumerate(phase_array_long.swapaxes(2,1)):
    #    firing_overview(channel,t_vec = time_vec, subplot_labels = freq_vec)
    #    fig = plt.gcf()
    #    fig.set_size_inches(8,10)
    #    fig.suptitle("_".join(animal_date_list) + '_Channel{}'.format(channel_num))
    #    fig.savefig(os.path.join(phase_plot_dir,'channel{}'.format(channel_num)))
    #    plt.close(fig)

    # Calculate phase consistency for each channel across all trials
    #phase_consistency = np.abs(np.mean(np.exp(phase_array_long*-1.j),axis=1))

    ## Plot phase consistency for every channel for all frequencies
    #firing_overview(phase_consistency,y_values_vec=freq_vec,cmap_lims = 'shared')
    #fig = plt.gcf()
    #fig.set_size_inches(10,10)
    #fig.suptitle("_".join(animal_date_list) + '_Phase_Consistency') 
    #fig.savefig(os.path.join(this_plot_dir,'phase_consistency_plots.png'))
    #plt.close(fig)

    #del phase_array, phase_array_long

##################################################
# Generate phase reset plots for both regions
##################################################
#
## Signal projection using IFFT
## Get stft from file
#stft_list = [hf5.get_node(os.path.join(this_path,'stft_array')) \
#        for this_path in node_path_list] 
## Pull out min err channels
#channel_list = [hf5.get_node(os.path.join(this_path,'relative_region_channel_nums'))[:] \
#        for this_path in node_path_list] 
#selected_stft_list = [stft[:,channels] for stft,channels in zip(stft_list,channel_list)]
## Pull out mean power in pre-delivery time
#window_len = 0.5 # sec
#padding = 0.25 # sec
#stim_delivery_t = 2 # sec
#pre_stim_stft = [ stft[...,(time_vec > (stim_delivery_t - window_len - padding)) \
#        * (time_vec < (stim_delivery_t - padding))] for stft in selected_stft_list]
#pre_stim_power = [np.abs(x) for x in pre_stim_stft]
#
#########################################
## Reconstructing test signal
#Fs = 1000 
#signal_window = 500 
#window_overlap = 499
#max_freq = 25
#time_range_tuple = (1,5)
#
#session_num = 0
#trial_ind = (0,0,0)
#test_trial_stft = selected_stft_list[session_num][trial_ind]
#
## Temp stft to find size of origin stft
## Add to setup to save dimensions
#f,t,temp_stft = scipy.signal.stft(
#            scipy.signal.detrend(actual_test_lfp), 
#            fs=Fs, 
#            window='hanning', 
#            nperseg=signal_window, 
#            noverlap=signal_window-(signal_window-window_overlap)) 
#
## Define function to return ISTFT on single frequency band data
#
#temp_trial_stft = np.zeros((temp_stft.shape[0],test_trial_stft.shape[-1]),
#        dtype = np.dtype('complex'))
#temp_trial_stft[:test_trial_stft.shape[0]] = test_trial_stft 
#test_istft = scipy.signal.istft(
#                temp_trial_stft, 
#                fs=Fs, 
#                window='hanning', 
#                nperseg=signal_window, 
#                noverlap=signal_window-(signal_window-window_overlap))
#
#dat = ephys_data('/media/bigdata/Abuzar_Data/AM17/AM17_extracted/'\
#        'AM17_4Tastes_191125_084206')
#dat.get_lfps()
#actual_selected_lfp = dat.lfp_array[:,channel_list[0]]
#actual_test_lfp = actual_selected_lfp[trial_ind]
#
## Check overlay of ISTFT with original LFP
#plt.plot(time_vec[1:],test_istft[1],linewidth = 5);
#plt.plot(np.arange(len(actual_test_lfp))/Fs,actual_test_lfp);plt.show()
#
#test_pre = pre_stim_power[session_num][trial_ind]
#
#########################################
##test_array = final_phases_long[0] 
### Reshape so mean period can be calculated
##tmp_array = test_array.swapaxes(1,2)
##test_long = np.reshape(tmp_array,(tmp_array.shape[0],tmp_array.shape[1],-1))
##period_arrays = np.where(np.abs(np.diff(test_long,axis=-1)) > 6)
##freq_period_arrays = [period_arrays[2][period_arrays[1] == freq]\
##        for freq in np.sort(np.unique(period_arrays[1]))]
##freq_periods = [np.diff(x) for x in freq_period_arrays]
##freq_periods_mean = [np.mean(x) for x in freq_periods]
##freq_periods_std = [np.std(x) for x in freq_periods]
##plt.errorbar(np.arange(len(freq_periods)),freq_periods_mean, freq_periods_std)
##plt.show()
##plt.plot(freq_periods_mean,'-x');plt.show()
#
## Unroll phase for each band
## Perform linear regression on pre-stimulus phase
## Check deviation at stimulus delivery
#
#session = 0
#trial_ind = (0,0,0)
#band_num = 1
#test_trial = final_phases[session_num][trial_ind]
#this_band = test_trial[band_num]
#
#from sklearn.linear_model import LinearRegression as LR
#reg = LR().fit(time_vec.reshape(-1,1),np.unwrap(this_band).reshape(-1,1))
#
#plt.plot(time_vec, np.unwrap(this_band))
#plt.plot(time_vec, np.squeeze(reg.predict(time_vec.reshape(-1,1))))
#plt.show()
