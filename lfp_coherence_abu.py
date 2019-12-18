## Import required modules
import os
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import lspopt
import tables
import easygui
import scipy
from scipy.signal import spectrogram
from lspopt import spectrogram_lspopt
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from sklearn.utils import resample
from itertools import product
from scipy.stats import zscore
from joblib import Parallel, delayed
import multiprocessing as mp


os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extract data
dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM11/AM11_extracted/AM11_4Tastes_191031_083633')
    #ephys_data('/media/bigdata/Abuzar_Data/AM12/AM12_extracted/AM12_4Tastes_191106_085215')
    #ephys_data('/media/bigdata/Abuzar_Data/AM17/AM17_extracted/AM17_4Tastes_191126_084934')
    #ephys_data('/media/bigdata/brads_data/Brad_LFP_ITI_analyses/BS26/BS26_180204')

dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()

middle_channels = np.arange(8,24)
region_label = [1 if any(x[0] == middle_channels) else 0 for x in dat.unit_descriptors]
dat.firing_overview(dat.all_normalized_firing,subplot_labels = region_label);plt.show()

# ____                  _                                       
#/ ___| _ __   ___  ___| |_ _ __ ___   __ _ _ __ __ _ _ __ ___  
#\___ \| '_ \ / _ \/ __| __| '__/ _ \ / _` | '__/ _` | '_ ` _ \ 
# ___) | |_) |  __/ (__| |_| | | (_) | (_| | | | (_| | | | | | |
#|____/| .__/ \___|\___|\__|_|  \___/ \__, |_|  \__,_|_| |_| |_|
#      |_|                            |___/                     

# Extract channel numbers for lfp
with tables.open_file(dat.hdf5_name,'r') as hf5:
    parsed_lfp_channels = hf5.root.Parsed_LFP_channels[:]

middle_channels_bool = np.array([True if channel in middle_channels else False \
        for channel in parsed_lfp_channels ])

# Calculate clims
mean_val = np.mean(dat.all_lfp_array, axis = None)
sd_val = np.std(dat.all_lfp_array, axis = None)
dat.firing_overview(dat.all_lfp_array, min_val = mean_val - 2*sd_val,
                    max_val = mean_val + 2*sd_val, cmap = 'viridis');plt.show()

# Mean LFP spectrogram 
region_a = dat.lfp_array[:,middle_channels_bool,:,:] 
region_b = dat.lfp_array[:,~middle_channels_bool,:,:]

band_freqs = [(1,4),
                (4,7),
                (7,12),
                (12,25)]


# ____ _____ _____ _____ 
#/ ___|_   _|  ___|_   _|
#\___ \ | | | |_    | |  
# ___) || | |  _|   | |  
#|____/ |_| |_|     |_|  
                        
# Use STFT to find instantaneous phase for each region

# Resolution has to be increased for phase of higher frequencies
Fs = 1000 
signal_window = 500 
window_overlap = 499
max_freq = 25
time_range_tuple = (1,5)

def calc_stft(trial, max_freq,time_range_tuple,
        Fs,signal_window,window_overlap):
    """
    trial : 1D array
    max_freq : where to lob off the transform
    time_range_tuple : (start,end) in seconds
    """
    f,t,this_stft = scipy.signal.stft(
                scipy.signal.detrend(trial), 
                fs=Fs, 
                window='hanning', 
                nperseg=signal_window, 
                noverlap=signal_window-(signal_window-window_overlap)) 
    this_stft =  this_stft[np.where(f<max_freq)[0]]
    this_stft = this_stft[:,np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))[0]]
    return this_stft
                        

# Region A

region_a_iters = list(product(*list(map(np.arange,region_a.shape[:3]))))

# Test run for calc_stft
this_iter = region_a_iters[0]
test_stft = calc_stft(region_a[this_iter],
                    max_freq,
                    time_range_tuple,
                    Fs,
                    signal_window,window_overlap)
dat.imshow(np.abs(test_stft));plt.show()

region_a_stft = Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(calc_stft)(region_a[this_iter],max_freq,time_range_tuple,
                            Fs,signal_window,window_overlap)\
        for this_iter in tqdm(region_a_iters))

region_a_stft_array =\
        np.empty(tuple((*region_a.shape[:3],*test_stft.shape)),
                dtype=np.dtype(region_a_stft[0][0,0]))
for iter_num, this_iter in tqdm(enumerate(region_a_iters)):
    region_a_stft_array[this_iter] = region_a_stft[iter_num]

# Delete varialbes to free up memory
del region_a_stft

# Region B

region_b_iters = list(product(*list(map(np.arange,region_b.shape[:3]))))
# Test run for calc_stft
this_iter = region_b_iters[0]
test_stft = calc_stft(region_b[this_iter],
                    max_freq,
                    time_range_tuple,
                    Fs,
                    signal_window,window_overlap)
dat.imshow(np.abs(test_stft));plt.show()

region_b_stft = Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(calc_stft)(region_b[this_iter],max_freq,time_range_tuple,
            Fs,signal_window,window_overlap)\
        for this_iter in tqdm(region_b_iters))

region_b_stft_array =\
        np.empty(tuple((*region_b.shape[:3],*test_stft.shape)),
                dtype=np.dtype(region_b_stft[0][0,0]))
for iter_num, this_iter in tqdm(enumerate(region_b_iters)):
    region_b_stft_array[this_iter] = region_b_stft[iter_num]

# Remove original list
del region_b_stft
gc.collect()

# Pick channel with phase most consistently closest to mean
region_a_phases = np.angle(region_a_stft_array)
mean_phase_across_channels = np.mean(region_a_phases,axis=1)
mean_error = [np.mean(np.abs(region_a_phases[:,channel] - mean_phase_across_channels),axis=None)\
        for channel in range(region_a_phases.shape[1])]
# Pick channel with lowest error
region_a_phases_final = region_a_phases[:,np.argmin(mean_error)]

# Remove temporary variables to free up memory
del mean_phase_across_channels, mean_error

# Do same for region b
region_b_phases = np.angle(region_b_stft_array)
mean_phase_across_channels = np.mean(region_b_phases,axis=1)
mean_error = [np.mean(np.abs(region_b_phases[:,channel] - mean_phase_across_channels),axis=None)\
        for channel in range(region_b_phases.shape[1])]
# Pick channel with lowest error
region_b_phases_final = region_b_phases[:,np.argmin(mean_error)]

# Remove temporary variables to free up memory
del mean_phase_across_channels, mean_error
gc.collect()

# To have appropriate f and t
f,t,this_stft = scipy.signal.stft(
            scipy.signal.detrend(region_b[region_b_iters[0]]), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap)) 
valid_f = f[f<max_freq]
valid_t = t[(t>=time_range_tuple[0])*(t<time_range_tuple[1])]

# Check spectrum and phase-resetting
stim_time = 2
this_phase_array = region_a_stft_array
mean_power = np.mean(np.abs(this_phase_array),axis=(0,1,2))
mean_normalized_power = mean_power /\
                np.mean(mean_power[:,valid_t<stim_time],axis=1)[:,np.newaxis]
#mean_normalized_power -=\
#                np.mean(mean_normalized_power[:,valid_t<stim_time],axis=1)[:,np.newaxis]
plt.pcolormesh(valid_t, valid_f, 10*np.log10(mean_normalized_power), cmap='jet')
plt.show()

all_phases =\
        np.angle(this_phase_array).reshape(np.prod(this_phase_array.shape[:3]),
                len(f[f<25]),-1).swapaxes(0,1)
# Plot phases
plt.plot(all_phases[2,0].T,'x',c='orange');plt.show()
dat.imshow(all_phases[7]);plt.colorbar();plt.show()

del this_phase_array, all_phases
gc.collect()

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

# For visualization purposes reshape both phase arrays
# to have trials along a single axis
phase_diff = np.exp(-1.j*region_b_phases_final) - np.exp(-1.j*region_a_phases_final)
# Average across trials
mean_phase_diff = np.mean(phase_diff,axis=(0,1))

def normalize_spectrogram(array, time_vec, stim_time):
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array -= mean_baseline
    return array

normalized_region_a_spec = normalize_spectrogram(
                        np.mean(np.abs(region_a_stft_array),axis=(0,1,2)),
                        valid_t, 2)
normalized_region_b_spec = normalize_spectrogram(
                        np.mean(np.abs(region_b_stft_array),axis=(0,1,2)),
                        valid_t, 2)


# Calculate phase differences across every combination of channels
# in each region and average them all together
summed_differences = np.zeros(
        tuple((region_a_phases.shape[0],*region_a_phases.shape[2:])),
        dtype = 'complex64')
for channel_a in trange(region_a_phases.shape[1]):
    for channel_b in range(region_b_phases.shape[1]):
        a_dat = region_a_phases[:,channel_a]
        b_dat = region_b_phases[:,channel_b]
        summed_differences += np.exp(-1.j*a_dat) - np.exp(-1.j*b_dat)

# Plot
plt.subplot(411)
plt.pcolormesh(valid_t,valid_f,normalized_region_a_spec,
        cmap = 'jet')
plt.subplot(412)
plt.pcolormesh(valid_t,valid_f,normalized_region_b_spec,
        cmap = 'jet')
plt.subplot(413)
plt.pcolormesh(valid_t,valid_f,np.abs(mean_phase_diff),
        cmap='jet')
plt.subplot(414)
plt.pcolormesh(valid_t,valid_f,np.abs(np.mean(summed_differences,axis=(0,1))),
        cmap='jet')
plt.show()

# Per taste coherence
taste_coherence = np.abs(np.mean(summed_differences,axis=1))
dat.firing_overview(taste_coherence,time_step=1);plt.show()


# ____  _                          _               _    _             
#|  _ \| |__   __ _ ___  ___      | |    ___   ___| | _(_)_ __   __ _ 
#| |_) | '_ \ / _` / __|/ _ \_____| |   / _ \ / __| |/ / | '_ \ / _` |
#|  __/| | | | (_| \__ \  __/_____| |__| (_) | (__|   <| | | | | (_| |
#|_|   |_| |_|\__,_|___/\___|     |_____\___/ \___|_|\_\_|_| |_|\__, |
#                                                               |___/ 
# Cross frequency phase-locking of neurons
phases_final = region_b_phases_final.swapaxes(-1,-2)
spike_inds = np.where(np.asarray(dat.spikes))
nrn_phases = []
for nrn in tqdm(np.sort(np.unique(spike_inds[2]))):
    relevant_spikes = spike_inds[2] == nrn
    relevant_spike_inds = [x[relevant_spikes] for x in spike_inds]
    relevant_spike_inds = [relevant_spike_inds[0],
                            relevant_spike_inds[1],
                            relevant_spike_inds[3]]
    relevant_spike_inds = list(zip(*relevant_spike_inds))
    nrn_phases.append(
            np.asarray([phases_final[this_ind] \
            for this_ind in relevant_spike_inds]))

# Generate histograms
phase_bins = np.linspace(-np.pi,np.pi,101)
# (nrn x freqs x phase_bins)
phase_hists= np.asarray([[np.histogram(this_phases[:,freq],phase_bins,density=True)[0] \
        for freq in range(1,this_phases.shape[1])]\
        for this_phases in nrn_phases])

# For some reason every other frequency is flipped in phase
zscore_phase_hists = zscore(phase_hists,axis=1)
direction = np.arange(0,zscore_phase_hists.shape[1],2)
corrected_zscore_phase_hists = zscore_phase_hists
corrected_zscore_phase_hists[:,direction] *= -1
dat.firing_overview(corrected_zscore_phase_hists,cmap='jet');plt.show()

plt.imshow(corrected_zscore_phase_hists[0],
        cmap='magma',interpolation='gaussian',
        aspect = 'auto');plt.show()
