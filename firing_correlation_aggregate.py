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
from scipy.signal import fftconvolve

########################################################
# Define functions and relevant variables to extract data
########################################################

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),os.path.basename(path_to_node))

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Define middle channels in board
middle_channels = np.arange(8,24)

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

with tables.open_file(data_hdf5_path,'r') as hf5:
    firing_node_list  = [x for x in hf5.root.firing._f_walknodes() \
            if 'normalized_firing_array' in x.__str__()]
    print('Extracting firing info')
    firing_array_list = [x[:] for x in tqdm(firing_node_list )]
    # Extract all node names with firing array
    node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) for x in firing_node_list]
    # Pull electrode numbers from each array
    electrode_num_list  = [hf5.get_node(path,'unit_electrode')[:] for path in node_path_list]
    spike_array_list =  [hf5.get_node(path,'spike_array')[:] for path in node_path_list]

print('Calculating firing correlations')
#for this_node_num in tqdm(range(len(phase_node_list))):

this_node_num = 0

firing_array = firing_array_list[this_node_num].swapaxes(0,1)
# Should be a better wap to reshape spike_array
spike_array = spike_array_list[this_node_num].swapaxes(0,2).swapaxes(1,2)
electrode_nums = electrode_num_list[this_node_num]
middle_channels_bool = np.array([True if channel in middle_channels else False \
        for channel in electrode_nums ])
firing_array_split = [firing_array[middle_channels_bool], firing_array[~middle_channels_bool]]
spike_array_split = [spike_array[middle_channels_bool], spike_array[~middle_channels_bool]]
channel_num_split = \
        [electrode_nums[middle_channels_bool], electrode_nums[~middle_channels_bool]]
relative_channel_num_split = \
        [np.arange(len(electrode_nums))[middle_channels_bool], 
                np.arange(len(electrode_nums))[~middle_channels_bool]]
del firing_array, spike_array


#  ____                    _       _   _             
# / ___|___  _ __ _ __ ___| | __ _| |_(_) ___  _ __  
#| |   / _ \| '__| '__/ _ \ |/ _` | __| |/ _ \| '_ \ 
#| |__| (_) | |  | | |  __/ | (_| | |_| | (_) | | | |
# \____\___/|_|  |_|  \___|_|\__,_|\__|_|\___/|_| |_|
#                                                    

# 1) Correlation of spike trains
#   a) Convolve spike trains
#   b) Convolve shuffled spike trains

## Have this stored somewhere
Fs = 1000
time_vec = np.arange(0,7)/Fs

# Generate test cases for correlations
# Pair of spike trains with varying correlations
# 1/3 : No correlation
# 2/3 : Very strong correlation
# 3/3 : Half as strong as previous section
# Poisson approximation
smooth_kern_sd = 50e-3 # seconds; For plotting
gaussian_func = lambda x,sd : (1/(sd*np.sqrt(2*np.pi)))*\
        np.exp(-0.5*((x/sd)**2))
smooth_kern = gaussian_func(np.arange(-3*smooth_kern_sd,3*smooth_kern_sd,step=dt),smooth_kern_sd)
dt = 1e-3
t_epoch = 20 # 10 second epochs
max_rate = 100
noise_ratio = 0.5

epochs = [np.concatenate([
    np.random.random(int(t_epoch/dt)) * max_rate,
        (np.sin(2*np.pi*np.arange(t_epoch, step = dt))**2 + \
                noise_ratio*np.random.random(int(t_epoch/dt)))*max_rate,
        ((np.sin(2*np.pi*np.arange(t_epoch, step = dt)))**2 + \
                noise_ratio*np.random.random(int(t_epoch/dt)))*max_rate/2]) for x in range(2)]
spike_probs = [np.random.random(epochs[0].shape) for trials in range(2)]
spike_train_list = [((x)<epoch*dt)*1 for x,epoch in zip(spike_probs,epochs)]
firing_rate_list = [np.convolve(x,smooth_kern, 'same') for x in spike_train_list]
fig,ax = plt.subplots(3,1,sharex=True)
ax[0].plot(np.arange(len(epochs[0]))*dt,epochs[0])
ax[0].plot(np.arange(len(epochs[0]))*dt,epochs[1])
ax[1].scatter(np.where(spike_train_list[0])[0]*dt,np.ones(sum(spike_train_list[0])),alpha=0.1)
ax[1].scatter(np.where(spike_train_list[1])[0]*dt,np.ones(sum(spike_train_list[1]))*2,alpha=0.1)
ax[2].plot(np.arange(len(epochs[0]))*dt, firing_rate_list[0])
ax[2].plot(np.arange(len(epochs[0]))*dt, firing_rate_list[1])
plt.show()

        
# Reshape spike_arrays to be total_steps x window_size arrays
def step_break_vector(vector, window_size, step_size):
    total_time = vector.shape[-1] 
    bin_inds = (0,window_size)
    total_bins = int((total_time - window_size + 1) / step_size) + 1
    bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
            for step in np.arange(total_bins)*step_size ]
    vector_steps = np.empty((total_bins,window_size))
    for bin_num,bin_inds in enumerate(bin_list):
        vector_steps[bin_num, :] = \
                vector[bin_inds[0]:bin_inds[1]]
    return vector_steps

#test = step_break_vector(spike_train_list[0], int(window_size/dt), int(step_size/dt))

#plt.imshow(test,interpolation='gaussian',aspect='auto',cmap='jet');plt.show()


# Moving window spike correlations
window_size = 5 #sec
step_size = 5  #sec
# Convert array and convolve
step_break_arrays = [step_break_vector(x,int(window_size/dt), int(step_size/dt))\
        for x in spike_train_list]

plt.subplot(121)
plt.imshow(step_break_arrays[0],interpolation='gaussian',aspect='auto',cmap='jet')
plt.subplot(122)
plt.imshow(step_break_arrays[1],interpolation='gaussian',aspect='auto',cmap='jet')
plt.show()

# Perform convolution
out = fftconvolve(step_break_arrays[0],step_break_arrays[1],mode = 'same',axes=-1)
plt.pcolormesh(np.linspace(-window_size/2,window_size/2,out.shape[-1]),
        np.linspace(0,len(epochs[0])*dt,out.shape[0]),out)
plt.xlabel('Time lag (sec)')
plt.ylabel('Window center (sec)')
plt.show()

plt.imshow(out,interpolation='gaussian',aspect='auto',cmap='jet', origin='lower');plt.show()

# 2) Correlation of firing rates
## Zscore each window of firing for both neurons separately so
## spike count differences can be ruled out
window_size = 1 #sec
step_size = 0.1  #sec
# Convert array and convolve
firing_break_arrays = [step_break_vector(x,int(window_size/dt), int(step_size/dt))\
        for x in firing_rate_list]

plt.subplot(121)
plt.imshow(firing_break_arrays[0],interpolation='gaussian',aspect='auto',cmap='jet')
plt.subplot(122)
plt.imshow(firing_break_arrays[1],interpolation='gaussian',aspect='auto',cmap='jet')
plt.show()

zscore_firing_break_arrays = [zscore(x,axis=-1) for x in firing_break_arrays]
plt.subplot(121)
plt.imshow(zscore_firing_break_arrays[0],interpolation='gaussian',aspect='auto',cmap='jet')
plt.subplot(122)
plt.imshow(zscore_firing_break_arrays[1],interpolation='gaussian',aspect='auto',cmap='jet')
plt.show()

from scipy.signal import savgol_filter as sk_ft
# Perform convolution
out = fftconvolve(zscore_firing_break_arrays[0],zscore_firing_break_arrays[1],mode = 'same',axes=-1)
# Test histogram
percentiles = np.arange(0,100,5)
abs_out = np.abs(out)
bins_edges = np.percentile(abs_out,percentiles,axis=None)
window_centers = np.linspace(0,len(epochs[0])*dt,abs_out.shape[0])
time_hist = np.array([np.histogram(x,bins_edges)[0] for x in abs_out])
fig, ax = plt.subplots(1,3)
ax[0].pcolormesh(np.linspace(-window_size/2,window_size/2,out.shape[-1]),
        window_centers,out, cmap='jet')
ax[0].set_xlabel('Time lag (sec)')
ax[0].set_ylabel('Window center (sec)')
ax[1].plot(np.mean(abs_out,axis=-1),np.linspace(0,len(epochs[0])*dt,out.shape[0]))
ax[1].plot(sk_ft(np.mean(abs_out,axis=-1),21,1),np.linspace(0,len(epochs[0])*dt,out.shape[0]))
ax[2].pcolormesh(percentiles,
        window_centers ,time_hist, cmap='jet')
ax[2].set_xlabel('Percentile Coherence')
ax[2].set_ylabel('Window center (sec)')
ax[2].set_title('Coherence Percentile count')
plt.show()

