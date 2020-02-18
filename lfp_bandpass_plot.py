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

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Pull out all terminal groups (leafs) under stft
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    band_freqs = hf5.get_node('/bandpass_lfp','frequency_bands')[:]
    lfp_node_list = [x for x in hf5.root.bandpass_lfp._f_walknodes() \
            if 'bandpassed_lfp_array' in x.__str__()]
    print('Extracting phase info')
    lfp_array_list = [x[:] for x in tqdm(lfp_node_list)]
    # Extract all nodes with phase array
    node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) for x in lfp_node_list]
    # For each bandpass lfp extracted, find the channels used for STFT 
    stft_path_list = [os.path.join('/stft/',*x.split("/")[2:]) for x in node_path_list]
    relative_channel_nums = [hf5.get_node(this_path,'relative_region_channel_nums')[:] \
            for this_path in stft_path_list]

# Extract only relevant channels from each file
relevant_band_ind = 2
time_range = np.arange(0,5000)
selected_lfp_channels = [session[:,:,channels] \
        for session,channels in zip(lfp_array_list,relative_channel_nums)]

selected_lfp_band = [session[relevant_band_ind].swapaxes(0,1) for session in selected_lfp_channels]
selected_lfp_band_long = [np.reshape(x,(x.shape[0],-1,x.shape[-1])) for x in selected_lfp_band]
selected_lfp_band_long = [x[...,time_range] for x in selected_lfp_band_long]

# Calculate phase consistency for each as a check that channels
# from the same region weren't selected
selected_lfp_phase = [np.angle(hilbert(x)) for x in selected_lfp_band_long]
selected_lfp_phase_consistency = [np.abs(np.mean(np.exp(-1.j*x),axis=1)) \
        for x in selected_lfp_phase]

# For each session, plot n number of random trials frm both regions overlayed
num_trials = 10
trial_inds = np.random.choice(
        np.arange(selected_lfp_band_long[0].shape[1]),num_trials,replace=False)

for this_node_num in tqdm(range(len(lfp_node_list))):
    this_plot_dir = os.path.join(
                data_folder,*node_path_list[this_node_num].split('/')[2:])

    fig, ax = plt.subplots(num_trials+1,2)
    ax[0,0].set_title('Bandpassed LFP Trace')
    ax[0,1].set_title('Corresponding Phase')
    for trial_count, trial_num in enumerate(trial_inds):
        ax[trial_count,0].plot(selected_lfp_band_long[this_node_num][:,trial_num].T)
        ax[trial_count,1].plot(
                np.angle(
                    hilbert(selected_lfp_band_long[this_node_num][:,trial_num]).T))
    ax[-1,0].plot(selected_lfp_phase_consistency[this_node_num].T)
    ax[-1,0].set_title('Phase consistency per channel')
    fig.set_size_inches(15,15)
    fig.suptitle('Frequency Band : {}'.format(band_freqs[relevant_band_ind]))
    #plt.show()
    fig.savefig(os.path.join(this_plot_dir,'random_trials_phase'))
    plt.close(fig)

