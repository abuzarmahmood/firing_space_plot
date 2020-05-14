## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
from scipy.signal import spectrogram
import glob
import json
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
from sklearn.utils import resample
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import zscore

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

##################################################
## Define functions
##################################################

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[...,time_vec<stim_time],axis=-1)[...,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    #mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    #array -= mean_baseline
    return array

def calc_coherence(stft_a, stft_b, trial_axis = 0):
    """
    inputs : arrays of shape (trials x freq x time)
    """
    cross_spec = np.mean(stft_a * np.conj(stft_b),axis=trial_axis)
    a_power_spectrum = np.mean(np.abs(stft_a)**2,axis=trial_axis)
    b_power_spectrum = np.mean(np.abs(stft_b)**2,axis=trial_axis)
    coherence = np.abs(cross_spec)/np.sqrt(a_power_spectrum*b_power_spectrum)
    return coherence

##################################################
## Read in data 
##################################################

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/fastdata/lfp_analyses'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Read all filenames and extract basenames
log_file_name = os.path.join(data_folder, 'file_list.txt')

if os.path.exists(log_file_name):
    file_list = open(log_file_name,'r').readlines()
    file_list = [x.rstrip() for x in file_list]
    basename_list = [os.path.basename(x) for x in file_list]

# Pull out all terminal groups (leafs) under stft
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    freq_vec = hf5.get_node('/stft','freq_vec')[:]
    time_vec = hf5.get_node('/stft','time_vec')[:]

    phase_node_path_list = \
            [x.__str__().split(" ")[0] for x in hf5.root.stft._f_walknodes() \
            if 'phase_array' in x.__str__()]
    phase_node_path_list.sort()

    # Extract all nodes with phase array
    node_path_list = [os.path.dirname(x) for x in phase_node_path_list]

    # Ask user to select files to perform anaysis on
    selected_files = easygui.multchoicebox(\
            msg = 'Please select files to run analysis on',
            choices = ['{}) '.format(num)+x[6:] \
                    for num,x in enumerate(node_path_list)])
    selected_file_inds = [int(x.split(')')[0]) for x in selected_files]

    phase_node_path_list = [phase_node_path_list[i] for i in selected_file_inds]
    node_path_list = [node_path_list[i] for i in selected_file_inds]

    # Pull parsed_lfp_channel from each array
    parsed_channel_list  = \
            [hf5.get_node(path,'parsed_lfp_channels')[:] \
            for path in node_path_list]

# Each node present will have run through lfp_coherence_setup
# Therefore, filename for each node will be present in the file-list
# Extract filename for each node from the file-list
# This will be used to extract laser trials for each file
animal_name_date_list = [x.split('/')[2:4] for x in node_path_list]

# Match strings from animal_name_date_list with filenames in the log file
matched_filename_inds = [[num for num, filename in enumerate(file_list) \
            if (name_date_str[0] in filename) and (name_date_str[1] in filename)] \
            for name_date_str in animal_name_date_list]
matched_filename_list = [file_list[i[0]] for i in matched_filename_inds]

# Read json files corresponding to each chosen file
# And extract electrodes for each region
dirname_list = [os.path.dirname(x) for x in matched_filename_list]
json_path_list = [glob.glob(dirname+'/*json')[-1] for dirname in dirname_list]
json_list = [json.load(open(file,'r')) for file in json_path_list]
region_list = [x['regions'] for x in json_list]

# Confirm with user that correct regions in each file
def str_continuous_nums(vec):
    """
    Returns representation of vector as string showing continuous intervals
    """
    sorted_vec = np.sort(vec)
    breakpoints = np.where(np.diff(sorted_vec)>1)[0]
    all_interval_markers = np.sort([0,*breakpoints, *(breakpoints+1), len(vec)-1])
    all_interval_lims = sorted_vec[all_interval_markers]
    interval_str_list = ['{}-{}'.format(x,y) \
            for x,y in np.reshape(all_interval_lims, (-1,2))]
    return ", ".join(interval_str_list) 

region_num_str_list = [[" :: ".join([key, str_continuous_nums(val)]) \
                        for key,val in x.items()] for x in region_list]
filename_region_str_list = [" ::: ".join(\
                ["_".join(filename), " , ".join(region)])\
                for filename, region in 
                zip(animal_name_date_list,region_num_str_list)]

# Check with user if electrode numbers look ok
electrode_bool = easygui.ynbox(\
        msg = "Do these eletrode numbers look good? \n\n{}".\
        format("\n".join(filename_region_str_list)))
if not electrode_bool:
    exit()

# Define variables to be maintained across files
initial_dir = dir() + ['initial_dir']

#  ____      _                                  
# / ___|___ | |__   ___ _ __ ___ _ __   ___ ___ 
#| |   / _ \| '_ \ / _ \ '__/ _ \ '_ \ / __/ _ \
#| |__| (_) | | | |  __/ | |  __/ | | | (_|  __/
# \____\___/|_| |_|\___|_|  \___|_| |_|\___\___|
#
#Refer to:
#    http://math.bu.edu/people/mak/sfn-2013/sfn_tutorial.pdf
#    http://math.bu.edu/people/mak/sfn/tutorial.pdf

print('Calculating phase coherences')
for this_node_num in tqdm(range(len(phase_node_path_list))):

    ######################################## 
    ## Coherence from Phase Difference
    ######################################## 
    with tables.open_file(data_hdf5_path,'r') as hf5:
        phase_array = hf5.get_node(node_path_list[this_node_num],
                                'phase_array')[:].swapaxes(0,1)
    parsed_channels = parsed_channel_list[this_node_num]
    
    # Split phase array by channels
    
    # The indices of the array belonging to each region
    channel_split_inds = [[num for num, x in enumerate(parsed_channels) \
            if x in vals] for key, vals in region_list[this_node_num].items()]
    # The actual channel numbers
    channel_num_split = [[parsed_channels[i] for i in region] \
            for region in channel_split_inds]
    phase_array_split = [phase_array[np.array(inds)] for inds in channel_split_inds]
    del phase_array

    # Check whether file had lasers
    # And if so, extract laser trials
    data = ephys_data(data_dir = dirname_list[this_node_num]) 
    data.check_laser()

    #data.firing_rate_params = dict(zip(\
    #    ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
    #    ('conv',25,250,1,25e-3,1e-3)))

    #middle_channels_bool = np.array(\
    #        [True if channel in middle_channels else False \
    #        for channel in parsed_channels ])
    #phase_array_split = [phase_array[middle_channels_bool], 
    #                    phase_array[~middle_channels_bool]]
    #channel_num_split = \
    #                [parsed_channels[middle_channels_bool], 
    #                parsed_channels[~middle_channels_bool]]
    #relative_channel_num_split = \
    #                [np.arange(len(parsed_channels))[middle_channels_bool], 
    #                np.arange(len(parsed_channels))[~middle_channels_bool]]

    # Find channel with phase closest to mean phase across all channels
    mean_phase_across_channels = [np.mean(x,axis=0) for x in phase_array_split]
    mean_error = [np.mean(np.abs(this_phase_array - \
            np.broadcast_to(this_mean_phase,this_phase_array.shape)),
                axis=tuple(np.arange(len(this_phase_array.shape))[1:])) \
            for this_phase_array, this_mean_phase in \
            zip(phase_array_split,mean_phase_across_channels)]

    # These are the relative numbers for the selected channels
    # To be used with indexing in arrays where BOTH REGIONS ARE COMBINED
    min_err_channel_nums = [this_channel_vec[np.argmin(this_error)] \
                            for this_channel_vec,this_error in \
                            zip(channel_split_inds,mean_error)] 
                            #zip(relative_channel_num_split,mean_error)] 

    # Pick channel with lowest error
    min_err_phase = np.array(\
                [this_phase_array[np.argmin(this_error)] \
            for this_phase_array,this_error in zip(phase_array_split,mean_error)])

    # For visualization purposes reshape both phase arrays
    # to have trials along a single axis
    phase_diff = np.exp(1.j*(np.squeeze(np.diff(min_err_phase, axis = 0))))

    # If laser exists, break down trials by on and off conditions
    if data.laser_exists:
        laser_inds = [x>0 for x in data.laser_durations] 
        laser_on_phase = np.array([phase[inds] \
                for phase,inds in zip(phase_diff, laser_inds)])
        laser_off_phase = np.array([phase[~inds] \
                for phase,inds in zip(phase_diff, laser_inds)])
        phase_diff_array = np.array([laser_off_phase, laser_on_phase])
    else:
        # shape ::: ( laser condition x taste x trials x freqs x time )
        phase_diff_array = phase_diff[np.newaxis]


    # Plot phase diff as histogram time-series for each band to confirm coherence
    # Reshape phase_diff array and extract angle
    phase_diff_reshape = np.angle(
            np.reshape(phase_diff_array, 
            (phase_diff_array.shape[0],
            np.prod(np.array(phase_diff_array.shape)[[1,2]]),
            *phase_diff_array.shape[3:]))).swapaxes(1,2)
    phase_bin_nums = 30
    phase_bins = np.linspace(-np.pi, np.pi, phase_bin_nums)
    time_bin_width = 10
    time_bins = np.arange(phase_diff_reshape.shape[-1], step = time_bin_width)

    time_inds = np.stack(np.ndindex(phase_diff_reshape.shape[-2:]))[:,-1]

    phase_diff_hists = np.array(\
            [[ np.histogram2d(freq.flatten(), time_inds,
                            bins = (phase_bins, time_bins))[0] \
            for freq in tqdm(cond)] for cond in phase_diff_reshape])

    #phase_diff_hists = np.array([[np.histogram(freq,phase_bins)[0] \
    #        for freq in time_bin] \
    #        for time_bin in phase_diff_reshape.T]).swapaxes(0,1) 

    # Average Coherence across trials
    mean_taste_coherence = np.abs(np.mean(phase_diff_array,axis=2))
    mean_coherence = np.mean(mean_taste_coherence,axis=1)

    ######################################## 
    ## Coherence from STFT
    ######################################## 
    # As a secondary measure, calculate coherence from STFT
    # to make sure calculations are correct

    with tables.open_file(data_hdf5_path,'r') as hf5:
        stft_array  = hf5.get_node(node_path_list[this_node_num],'stft_array')[:] 
    # Extract relevant channels and discard rest
    selected_stft_array = stft_array[:,min_err_channel_nums]

    if data.laser_exists:
        on_stft_array = np.array([stft[:,inds] \
                    for stft,inds in zip(selected_stft_array, laser_inds)])
        on_stft_coherence = calc_coherence(on_stft_array[:,0],
                    on_stft_array[:,1], trial_axis = 1)

        off_stft_array = np.array([stft[:,~inds] \
                    for stft,inds in zip(selected_stft_array, laser_inds)])
        off_stft_coherence = calc_coherence(off_stft_array[:,0],
                    off_stft_array[:,1], trial_axis = 1)
        stft_coherence = np.array([off_stft_coherence, on_stft_coherence])

    else:
        stft_coherence = calc_coherence(selected_stft_array[:,0],
                    selected_stft_array[:,1], trial_axis = 1)
        stft_coherence = stft_coherence[np.newaxis]

    ######################################## 
    ## Write out data 
    ######################################## 
    # Write out final phase channels and channel numbers 
    with tables.open_file(data_hdf5_path,'r+') as hf5:
        # region_phase_channels are the phases of the chosen channels
        # relative_region_channel_nums are the indices of the channels used
        remove_node(os.path.join(
                node_path_list[this_node_num],'region_phase_channels'),hf5)
        remove_node(os.path.join(
                node_path_list[this_node_num],'relative_region_channel_nums'),hf5)
        remove_node(os.path.join(
                node_path_list[this_node_num],'phase_difference_array'),hf5)
        remove_node(os.path.join(
                node_path_list[this_node_num],'mean_coherence_array'),hf5)

        hf5.create_array(node_path_list[this_node_num], 'region_phase_channels', 
             np.array(min_err_phase))
        hf5.create_array(node_path_list[this_node_num], 'relative_region_channel_nums', 
             np.array(min_err_channel_nums))
        hf5.create_array(node_path_list[this_node_num], 
                            'phase_difference_array', phase_diff_array)
        hf5.create_array(node_path_list[this_node_num], 
                            'mean_coherence_array', mean_coherence)

    # Phase consistency for BLA and GC
    phase_vectors = np.exp(min_err_phase*1.j) 
    taste_phase_consistency = np.abs(np.mean(phase_vectors,axis = 2)) 
    mean_phase_consistency = np.mean(taste_phase_consistency,axis=1) 

    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    #                       

    # Plot 1
    # a) Spectrograms of chosen channels
    # b) Mean Phase consistency 
    # c) Mean Coherence

    # Pull out stft amplitude from file
    #with tables.open_file(data_hdf5_path,'r+') as hf5:
    #    amplitude_array  = hf5.get_node(\
    #            node_path_list[this_node_num],'amplitude_array')[:] 

    #amplitude_array_split = [amplitude_array[:,x] \
    #            for x in channel_split_inds]
    #median_channel_amplitude_array = np.array([np.median(x,axis=1) \
    #            for x in amplitude_array_split])

    # Extract relevant channels
    #min_err_chan_amp = np.array([np.squeeze(amplitude_array[:,chan]) \
    #            for chan in min_err_channel_nums])

    ## If lase is present, separate out trials by laser
    #if data.laser_exists:
    #    on_amplitude = np.array([min_err_chan_amp[:,taste,inds] \
    #                    for taste,inds in enumerate(laser_inds)])
    #    off_amplitude = np.array([min_err_chan_amp[:,taste,~inds] \
    #                    for taste,inds in enumerate(laser_inds)])
    #    # shape ::: laser x taste x region x trial x freq x time
    #    min_err_amp_array = np.array([off_amplitude, on_amplitude])
    #else:
    #    min_err_amp_array = min_err_chan_amp[np.newaxis]

    ## shape ::: laser x region x freq x time
    #min_err_spectrograms = np.median(min_err_amp_array, axis = (1,3)) 
    #norm_spectrograms = normalize_timeseries(min_err_spectrograms,time_vec,2) 

    # Just use STFT extracted above for amplitude rather than separating
    # by laser again
    laser_spectrogram_array = np.abs(np.array([off_stft_array, on_stft_array]))
    median_spectrogram_array = np.median(laser_spectrogram_array, axis = (1,3))
    # shape ::: laser x region x freq x time
    norm_spectrograms = normalize_timeseries(median_spectrogram_array,time_vec,2)

    # Calculate difference between laser conditions
    if data.laser_exists:
        #diff_spectrogram_array = np.squeeze(np.diff(median_spectrogram_array, axis = 0))
        #norm_median_diff_spec = normalize_timeseries(diff_spectrogram_array,
        #                                time_vec, stim_time = 2)
        norm_median_diff_spec = np.squeeze(np.diff(norm_spectrograms,axis=0)) 

    this_plot_dir = os.path.join(data_folder,*animal_name_date_list[this_node_num])
    plot_titles = list(product(['off','on'], 
                    list(zip(min_err_channel_nums, 
                        region_list[this_node_num].keys()))))

    # Plot 1
    # a) Normalized BLA Spectrogram(s)
    # b) Normalized GC Spectrogram(s)
    # If laser exists, difference between on and off conditions
    vmin,vmax = np.min(norm_spectrograms),np.max(norm_spectrograms)
    fig,ax = plt.subplots(norm_spectrograms.shape[0] + data.laser_exists * 1,
                                norm_spectrograms.shape[1])
    iter_inds = np.stack(np.ndindex(norm_spectrograms.shape[:2]))
    for ind in iter_inds:
        ax[tuple(ind)].contourf(time_vec, freq_vec, 
                norm_spectrograms[tuple(ind)], 
                cmap = 'jet', levels = 30, vmin = vmin, vmax = vmax)
    for num,this_ax in enumerate(ax.flatten()):
        this_ax.set_title(plot_titles[num])
    if data.laser_exists:
        ax[-1,0].contourf(time_vec, freq_vec, norm_median_diff_spec[0], 
                cmap = 'jet', levels = 30)
        ax[-1,1].contourf(time_vec, freq_vec, norm_median_diff_spec[1], 
                cmap = 'jet', levels = 30)
        ax[-1,0].set_title(list(region_list[this_node_num].keys())[0] + '_diff')
        ax[-1,1].set_title(list(region_list[this_node_num].keys())[1] + '_diff')
    fig.suptitle("_".join(animal_name_date_list[this_node_num]) + '\nSpectrograms')
    fig.set_size_inches(8,10)
    fig.savefig(os.path.join(this_plot_dir,'mean_spectrograms'))

    #ax.pcolormesh(time_vec, freq_vec, norm_spectrograms[0], cmap = 'jet')
    #ax0.set_title('Channel {}\nSpectrogram'.format(min_err_channel_nums[0]))
    #ax1.pcolormesh(time_vec, freq_vec, norm_spectrograms[1], cmap = 'jet')
    #ax1.set_title('Channel {}\nSpectrogram'.format(min_err_channel_nums[1]))
    #ax3.pcolormesh(time_vec, freq_vec, mean_phase_consistency[0], 
    #                                    cmap = 'jet',vmin=0,vmax=1)
    #ax3.set_title('Phase consistency'.format(min_err_channel_nums[0]))
    #ax4.pcolormesh(time_vec, freq_vec, mean_phase_consistency[1], 
    #                                    cmap = 'jet',vmin=0,vmax=1)
    #ax4.set_title('Phase consistency'.format(min_err_channel_nums[1]))
     
    # Plot 2
    # a) Phase coherence by taste and mean phase coherence
    fig, ax = plt.subplots(mean_taste_coherence.shape[1] + 1,
            mean_taste_coherence.shape[0])
    for this_ax in range(len(ax)):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, mean_taste_coherence[this_ax], 
    ax5 = plt.subplot(4,1,3)
    ax5.pcolormesh(time_vec, freq_vec, mean_coherence, 
                                        cmap = 'jet',vmin=0,vmax=1)
    ax5.set_title('Phase coherence'.format(min_err_channel_nums[0]))
    ax6 = plt.subplot(4,1,4)
    ax6.pcolormesh(time_vec, freq_vec, 
            normalize_timeseries(mean_coherence, time_vec, 2),
            cmap = 'jet')
    ax6.set_title('Normalized Phase coherence'.format(min_err_channel_nums[0]))
                                        cmap = 'jet', vmin=0,vmax=1)
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'phase_coherence_taste'))

    # Plot 3
    # a) Normalized Phase coherence by taste
    fig, ax = plt.subplots(4,1)
    for this_ax in range(len(ax)):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, 
                normalize_timeseries(mean_taste_coherence[this_ax], time_vec, 2),
                cmap = 'jet')
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'normalized_phase_coherence_taste'))

    # Plot 4
    # Histograms of phase difference by bands
    firing_overview(phase_diff_hists.swapaxes(1,2),
            time_vec,phase_bins[1:],subplot_labels = freq_vec)
    fig = plt.gcf()
    fig.set_size_inches(10,8)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'phase_diff_histograms'))

    # Plot 5
    # b) Phase consistency by taste 
    # And mean phase consistency
    fig, ax = plt.subplots(4,1)
    for this_ax in range(len(ax)):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, taste_phase_consistency[0][this_ax], 
                                            cmap = 'jet')
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'phase_consistency_RG0'))

    # Plot 6
    # b) Phase consistency by taste 
    fig, ax = plt.subplots(4,1)
    for this_ax in range(len(ax)):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, taste_phase_consistency[1][this_ax], 
                                            cmap = 'jet')
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'phase_consistency_RG1'))

    # Plot 7
    # a-d) Coherence per taste using STFT
    # e) Average STFT coherence
    fig, ax = plt.subplots(5,1)
    for this_ax in range(len(ax)-1):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, 
                normalize_timeseries(stft_coherence[this_ax],time_vec,2), 
                                            cmap = 'jet')
                #stft_coherence[this_ax], cmap = 'jet')
    plt.sca(ax[-1])
    plt.pcolormesh(time_vec, freq_vec, 
                normalize_timeseries(np.mean(stft_coherence,axis=0),time_vec,2), 
                                            cmap = 'jet')
                #np.mean(stft_coherence,axis=0), cmap = 'jet')
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'STFT_Coherence'))

    plt.close('all')

    ######################################## 
    ## Delete all variables related to single file
    ######################################## 
    for item in dir():
        if item not in initial_dir:
            del globals()[item]

