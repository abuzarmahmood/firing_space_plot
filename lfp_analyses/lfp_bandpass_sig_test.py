## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
from scipy.signal import spectrogram
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import imshow, firing_overview
from operator import itemgetter

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/fastdata/lfp_analyses'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Pull out bandpassed lfp and hilbert phase
# Also extract channels used for STFT phase analyses
# ** Future updates can reimplement picking channel for bandpass lfp **
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    band_freqs = hf5.get_node('/bandpass_lfp','frequency_bands')[:]
    lfp_node_path_list = \
            [x.__str__().split(" ")[0] \
            for x in hf5.root.bandpass_lfp._f_walknodes() \
            if 'bandpassed_lfp_array' in x.__str__()]
    lfp_node_path_list.sort()

    #lfp_node_list = [x for x in hf5.root.bandpass_lfp._f_walknodes() \
    #        if 'bandpassed_lfp_array' in x.__str__()]
    #print('Extracting phase info')
    #lfp_array_list = [x[:] for x in tqdm(lfp_node_list)]
    # Extract all nodes with phase array
    #node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) \
    #                            for x in lfp_node_list]
    # For each bandpass lfp extracted, find the channels used for STFT 
    #stft_path_list = [os.path.join('/stft/',*x.split("/")[2:]) \
    #                            for x in node_path_list]
    
    # Check which nodes in STFT have 'relative_region_channel_nums'
    channel_path_list = \
            [x.__str__().split(" ")[0] \
            for x in hf5.root.stft._f_walknodes() \
            if 'relative_region_channel_nums' in x.__str__()]
    channel_path_list.sort()
    relative_channel_nums = [hf5.get_node(os.path.dirname(this_path),
                            'relative_region_channel_nums')[:] \
                            for this_path in channel_path_list]

# Check which files have both bandpassed lfp and stft
lfp_name_date_str = [x.split('/')[2:4] for x in lfp_node_path_list]
channel_name_date_str = [x.split('/')[2:4] for x in channel_path_list]
common_files = [file for file in lfp_name_date_str if file in channel_name_date_str]
lfp_inds = [num for num, file in enumerate(lfp_name_date_str) \
        if file in common_files]
channel_inds = [num for num, file in enumerate(channel_name_date_str) \
        if file in common_files]

fin_lfp_node_path_list = [lfp_node_path_list[i] for i in lfp_inds] 
fin_lfp_name_date_str = [lfp_name_date_str[i] for i in lfp_inds] 
fin_channel_nums = [relative_channel_nums[i] for i in channel_inds]


# Print which files will be processed
# Find union of files
union_files = list(set(map(tuple,lfp_name_date_str+channel_name_date_str)))
union_files.sort()
union_process_inds = [1 if file in map(tuple,common_files) else 0 \
        for file in union_files]
print('The following files will be processed')
print("\n".join(list(map(str,zip(union_process_inds,union_files)))))

# Run through files
# Calculate phase difference histograms and
# Plot phases of random subset of trials
for this_node_num in tqdm(range(len(fin_lfp_node_path_list))):

    this_plot_dir = os.path.join(
                data_folder,*fin_lfp_name_date_str[this_node_num])

    with tables.open_file(data_hdf5_path,'r') as hf5:
        #bandpass_lfp_array = hf5.get_node(
        #        os.path.dirname(fin_lfp_node_path_list[this_node_num]),
        #        'bandpassed_lfp_array')[:][:,:,fin_channel_nums[this_node_num]]
        phase_array = hf5.get_node(
                os.path.dirname(fin_lfp_node_path_list[this_node_num]),
                'phase_array')[:][:,:,fin_channel_nums[this_node_num]]
        #amplitude_array = hf5.get_node(
        #        os.path.dirname(fin_lfp_node_path_list[this_node_num]),
        #        'amplitude_array')[:][:,:,fin_channel_nums[this_node_num]]

    # Test that phase extraction was good
    #phase_array_long = phase_array.swapaxes(1,2).\
    #        reshape((phase_array.shape[0],
    #                phase_array.shape[2],
    #                -1,
    #                phase_array.shape[-1]))
    #firing_overview(phase_array_long[:,0])
    #firing_overview(phase_array_long[:,1])
    #plt.show()

    # Calculate phase difference
    #phase_diff_array = \
    #        np.angle(np.exp(-1.j*np.diff(phase_array,axis=2))).squeeze()
    phase_diff_array = np.diff(phase_array,axis=2).squeeze()
    phase_diff_reshape = phase_diff_array.reshape(\
            (phase_array.shape[0],-1,phase_diff_array.shape[-1]))
    phase_coherence_array = np.abs(np.mean(np.exp(-1.j*phase_diff_reshape),axis=1))
    phase_bin_nums = 20
    phase_bins = np.linspace(-np.pi, np.pi, phase_bin_nums)
    phase_diff_hists = np.array([[np.histogram(time_bin,phase_bins)[0] \
            for time_bin in freq] \
            for freq in phase_diff_reshape.swapaxes(1,2)]).swapaxes(1,2) 

    firing_overview(phase_diff_hists);plt.show()
    imshow(phase_coherence_array);plt.show()

    fig, ax = plt.subplots(phase_coherence_array.shape[0])
    for data,this_ax in zip(phase_coherence_array,ax):
        this_ax.plot(data)
    plt.show()

    # ** STILL TOO HARD TO LOOK AT...SCRAPPED!!!**
    # For each band, plot phase for a given number of trials
    #num_trials = 4
    #choices = list(product(range(phase_array.shape[1]),
    #                            range(phase_array.shape[3])))
    #random_trial_choices = np.random.choice(range(len(choices)),num_trials) 
    #random_trial_inds = [choices[i] for i in random_trial_choices]
    #selected_trials = np.array([phase_array[:,ind[0],:,ind[1]] \
    #        for ind in random_trial_inds])

    #fig, ax = plt.subplots(*selected_trials.shape[:2])
    #inds = list(np.ndindex(ax.shape))
    #for this_ind, this_ax in zip(inds,ax.flatten()):
    #    this_ax.imshow(selected_trials[this_ind],
    #            interpolation='nearest', aspect='auto')
    #plt.show()


# Extract only relevant channels from each file
#relevant_band_ind = 2
#time_range = np.arange(0,5000)
#selected_lfp_channels = [session[:,:,channels] \
#        for session,channels in zip(lfp_array_list,relative_channel_nums)]
#
#selected_lfp_band = [session[relevant_band_ind].swapaxes(0,1) \
#                                    for session in selected_lfp_channels]
#selected_lfp_band_long = [np.reshape(x,(x.shape[0],-1,x.shape[-1])) \
#                                    for x in selected_lfp_band]
#selected_lfp_band_long = [x[...,time_range] for x in selected_lfp_band_long]
#
## Calculate phase consistency for each as a check that channels
## from the same region weren't selected
#selected_lfp_phase = [np.angle(hilbert(x)) for x in selected_lfp_band_long]
#selected_lfp_phase_consistency = [np.abs(np.mean(np.exp(-1.j*x),axis=1)) \
#        for x in selected_lfp_phase]
#
## For each session, plot n number of random trials frm both regions overlayed
#num_trials = 10
#trial_inds = np.random.choice(
#        np.arange(selected_lfp_band_long[0].shape[1]),num_trials,replace=False)
#
#for this_node_num in tqdm(range(len(lfp_node_list))):
#    this_plot_dir = os.path.join(
#                data_folder,*node_path_list[this_node_num].split('/')[2:])
#
#    fig, ax = plt.subplots(num_trials+1,2)
#    ax[0,0].set_title('Bandpassed LFP Trace')
#    ax[0,1].set_title('Corresponding Phase')
#    for trial_count, trial_num in enumerate(trial_inds):
#        ax[trial_count,0].plot(
#                selected_lfp_band_long[this_node_num][:,trial_num].T)
#        ax[trial_count,1].plot(
#                np.angle(
#                    hilbert(selected_lfp_band_long[this_node_num][:,trial_num]).T))
#    ax[-1,0].plot(selected_lfp_phase_consistency[this_node_num].T)
#    ax[-1,0].set_title('Phase consistency per channel')
#    fig.set_size_inches(15,15)
#    fig.suptitle('Frequency Band : {}'.format(band_freqs[relevant_band_ind]))
#    #plt.show()
#    fig.savefig(os.path.join(this_plot_dir,'random_trials_phase'))
#    plt.close(fig)
#
