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

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),os.path.basename(path_to_node))

def normalize_spectrogram(array, time_vec, stim_time):
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array -= mean_baseline
    return array

# Define middle channels in board
middle_channels = np.arange(8,24)

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Pull out all terminal groups (leafs) under stft
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    freq_vec = hf5.get_node('/stft','freq_vec')[:]
    time_vec = hf5.get_node('/stft','time_vec')[:]

    phase_node_list = [x for x in hf5.root.stft._f_walknodes() if 'phase_array' in x.__str__()]
    print('Extracting phase info')
    phase_array_list = [x[:] for x in tqdm(phase_node_list)]
    # Extract all nodes with phase array
    node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) for x in phase_node_list]
    # Pull parsed_lfp_channel from each array
    parsed_channel_list  = [hf5.get_node(path,'parsed_lfp_channels')[:] for path in node_path_list]

#  ____      _                                  
# / ___|___ | |__   ___ _ __ ___ _ __   ___ ___ 
#| |   / _ \| '_ \ / _ \ '__/ _ \ '_ \ / __/ _ \
#| |__| (_) | | | |  __/ | |  __/ | | | (_|  __/
# \____\___/|_| |_|\___|_|  \___|_| |_|\___\___|
#                                               
#
#Refer to:
#    http://math.bu.edu/people/mak/sfn-2013/sfn_tutorial.pdf
#    http://math.bu.edu/people/mak/sfn/tutorial.pdf

print('Calculating phase coherences')
for this_node_num in tqdm(range(len(phase_node_list))):
    phase_array = phase_array_list[this_node_num].swapaxes(0,1)
    parsed_channels = parsed_channel_list[this_node_num]
    middle_channels_bool = np.array([True if channel in middle_channels else False \
            for channel in parsed_channels ])
    phase_array_split = [phase_array[middle_channels_bool], phase_array[~middle_channels_bool]]
    channel_num_split = \
            [parsed_channels[middle_channels_bool], parsed_channels[~middle_channels_bool]]
    relative_channel_num_split = \
            [np.arange(len(parsed_channels))[middle_channels_bool], 
                    np.arange(len(parsed_channels))[~middle_channels_bool]]
    del phase_array

    # Find channel closest to mean phase
    mean_phase_across_channels = [np.mean(x,axis=0) for x in phase_array_split]
    mean_error = [np.mean(np.abs(this_phase_array - \
            np.broadcast_to(this_mean_phase,this_phase_array.shape)),
                axis=tuple(np.arange(len(this_phase_array.shape))[1:])) \
            for this_phase_array, this_mean_phase in \
            zip(phase_array_split,mean_phase_across_channels)]

    # Pick channel with lowest error
    min_err_phase = [this_phase_array[np.argmin(this_error)] \
            for this_phase_array,this_error in zip(phase_array_split,mean_error)]

    # These are the relative numbers for the channels
    min_err_channel_nums = [this_channel_vec[np.argmin(this_error)] \
            for this_channel_vec,this_error in zip(relative_channel_num_split,mean_error)] 

    # For visualization purposes reshape both phase arrays
    # to have trials along a single axis
    phase_diff = np.exp(-1.j*min_err_phase[0]) - np.exp(-1.j*min_err_phase[1])
    # Average across trials
    mean_taste_coherence = np.abs(np.mean(phase_diff,axis=1))
    mean_coherence = np.mean(mean_taste_coherence,axis=0)

    # Write out final phase channels and channel numbers 
    with tables.open_file(data_hdf5_path,'r+') as hf5:
        remove_node(os.path.join(node_path_list[this_node_num],'region_phase_channels'),hf5)
        remove_node(os.path.join(node_path_list[this_node_num],'relative_region_channel_nums'),hf5)
        #remove_node(os.path.join(node_path_list[this_node_num],'phase_difference_array'),hf5)
        #remove_node(os.path.join(node_path_list[this_node_num],'mean_coherence_array'),hf5)

        hf5.create_array(node_path_list[this_node_num], 'region_phase_channels', 
             np.array(min_err_phase))
        hf5.create_array(node_path_list[this_node_num], 'relative_region_channel_nums', 
             np.array(min_err_channel_nums))
        #hf5.create_array(node_path_list[this_node_num], 'phase_difference_array', phase_diff)
        #hf5.create_array(node_path_list[this_node_num], 'mean_coherence_array', mean_coherence)

    # Phase consistency for BLA and GC
    phase_vectors = [np.exp(phase*1.j) for phase in min_err_phase]
    mean_phase_consistency = [np.abs(np.mean(phase_vec, axis = (0,1))) \
            for phase_vec in phase_vectors]
    taste_phase_consistency = [np.abs(np.mean(phase_vec, axis = (1))) \
            for phase_vec in phase_vectors]

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
    with tables.open_file(data_hdf5_path,'r+') as hf5:
        amplitude_array  = hf5.get_node(node_path_list[this_node_num],'amplitude_array')[:] 
    # Extract relevant channels
    min_err_chan_amp = [np.squeeze(amplitude_array[:,chan]) for chan in min_err_channel_nums]
    min_err_spectrograms = [np.mean(dat, axis = (0,1)) for dat in min_err_chan_amp]
    norm_spectrograms = [normalize_spectrogram(dat,time_vec,2) for dat in min_err_spectrograms]

    this_plot_dir = os.path.join(
                data_folder,*node_path_list[this_node_num].split('/')[2:])

    fig = plt.figure()
    ax0 = plt.subplot(3,2,1)
    ax1 = plt.subplot(3,2,2)
    ax0.pcolormesh(time_vec, freq_vec, norm_spectrograms[0], cmap = 'jet')
    ax0.set_title('Channel {}\nSpectrogram'.format(min_err_channel_nums[0]))
    ax1.pcolormesh(time_vec, freq_vec, norm_spectrograms[1], cmap = 'jet')
    ax1.set_title('Channel {}\nSpectrogram'.format(min_err_channel_nums[1]))
    ax3 = plt.subplot(3,2,3)
    ax4 = plt.subplot(3,2,4)
    ax3.pcolormesh(time_vec, freq_vec, mean_phase_consistency[0], cmap = 'jet',vmin=0,vmax=1)
    ax3.set_title('Phase consistency'.format(min_err_channel_nums[0]))
    ax4.pcolormesh(time_vec, freq_vec, mean_phase_consistency[1], cmap = 'jet',vmin=0,vmax=1)
    ax4.set_title('Phase consistency'.format(min_err_channel_nums[1]))
    ax5 = plt.subplot(3,1,3)
    ax5.pcolormesh(time_vec, freq_vec, mean_coherence, cmap = 'jet',vmin=0,vmax=1)
    ax5.set_title('Phase coherence'.format(min_err_channel_nums[0]))
    fig.set_size_inches(8,10)
    fig.savefig(os.path.join(this_plot_dir),'mean_phase_coherence')
     
    # Plot 2
    # a) Phase coherence by taste
    fig, ax = plt.subplots(4,1)
    for this_ax in range(len(ax)):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, mean_taste_coherence[this_ax], cmap = 'jet')
    fig.set_size_inches(8,10)
    fig.savefig(os.path.join(this_plot_dir),'phase_coherence_taste'))

    # Plot 2
    # b) Phase consistency by taste 
    fig, ax = plt.subplots(4,1)
    for this_ax in range(len(ax)):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, taste_phase_consistency[0][this_ax], cmap = 'jet')
    fig.set_size_inches(8,10)
    fig.savefig(os.path.join(this_plot_dir),'phase_consistency_RG0')

    fig, ax = plt.subplots(4,1)
    for this_ax in range(len(ax)):
        plt.sca(ax[this_ax])
        plt.pcolormesh(time_vec, freq_vec, taste_phase_consistency[1][this_ax], cmap = 'jet')
    fig.set_size_inches(8,10)
    fig.savefig(os.path.join(this_plot_dir),'phase_consistency_RG1')

    plt.close('all')

#    _                                    _       
#   / \   __ _  __ _ _ __ ___  __ _  __ _| |_ ___ 
#  / _ \ / _` |/ _` | '__/ _ \/ _` |/ _` | __/ _ \
# / ___ \ (_| | (_| | | |  __/ (_| | (_| | ||  __/
#/_/   \_\__, |\__, |_|  \___|\__, |\__,_|\__\___|
#        |___/ |___/          |___/               

# Run through all directories and pull out final channel phases
hf5 = tables.open_file(data_hdf5_path,'r')
final_phases = [hf5.get_node(os.path.join(this_path,'region_phase_channels')) \
        for this_path in node_path_list] 
phase_diffs = [np.exp(-1.j*x[0]) - np.exp(-1.j*x[1]) for x in tqdm(final_phases)]
coherence_array = np.array([np.abs(np.mean(x,axis=(0,1))) for x in phase_diffs])
mean_aggregate_coherence = np.mean(coherence_array,axis=0)

# Show mean and std of coherence for bands
# Remove first freq band from array
fin_coherence_array = coherence_array[:,1:].swapaxes(0,1)
coherence_list = np.split(fin_coherence_array,6,axis=0)
freq_list = np.split(freq_vec[1:],6)
coherence_means = [np.mean(x,axis=(0,1)) for x in coherence_list]
coherence_std = [np.std(x,axis=(0,1)) for x in coherence_list]

##################################################
# Save outputs are plots
##################################################
agg_plot_dir = os.path.join(data_folder,'aggregate_analysis')
if not os.path.exists(agg_plot_dir):
    os.makedirs(agg_plot_dir)

fig,ax = plt.subplots()
ax.pcolormesh(time_vec, freq_vec, mean_aggregate_coherence, 
        cmap = 'jet',vmin = 0, vmax=1)
fig.set_size_inches(8,10)
fig.savefig(os.path.join(agg_plot_dir,'mean_phase_coherence'))


fig, ax = plt.subplots(len(coherence_means),sharey=True,sharex=True)
for this_ax, this_mean, this_std, this_freq in \
        zip(ax, coherence_means, coherence_std, freq_list):
    this_ax.fill_between(time_vec, this_mean - 2*this_std, this_mean+ 2*this_std)
    this_ax.plot(time_vec, this_mean, color='r')
    this_ax.set_title('Frequencies : {}'.format(this_freq))
fig.set_size_inches(8,10)
fig.savefig(os.path.join(agg_plot_dir,'bandwise_phase_coherence'))

plt.close('all')

