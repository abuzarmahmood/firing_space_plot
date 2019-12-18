## Import required modules
import os
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
    ephys_data('/media/bigdata/brads_data/Brad_LFP_ITI_analyses/BS26/BS26_180204')
    #ephys_data('/media/bigdata/Abuzar_Data/AM12/AM12_extracted/AM12_4Tastes_191106_085215')
    #ephys_data('/media/bigdata/Abuzar_Data/AM17/AM17_extracted/AM17_4Tastes_191126_084934')

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
#region_a = dat.lfp_array[:,middle_channels_bool,:,:] 
#region_b = dat.lfp_array[:,~middle_channels_bool,:,:]

region_b = dat.lfp_array

band_freqs = [(1,4),
                (4,7),
                (7,12),
                (12,25)]

#  ____      _                                  
# / ___|___ | |__   ___ _ __ ___ _ __   ___ ___ 
#| |   / _ \| '_ \ / _ \ '__/ _ \ '_ \ / __/ _ \
#| |__| (_) | | | |  __/ | |  __/ | | | (_|  __/
# \____\___/|_| |_|\___|_|  \___|_| |_|\___\___|
#                                               

#Refer to:
#    http://math.bu.edu/people/mak/sfn-2013/sfn_tutorial.pdf
#    http://math.bu.edu/people/mak/sfn/tutorial.pdf


##################
## Using STFT (Uses too much memory)
###################

# Resolution has to be increased for phase of higher frequencies
Fs = 1000 
signal_window = 1000 
window_overlap = 999

def calc_stft(trial, max_freq,Fs,signal_window,window_overlap):
    """
    trial : 1D array
    max_freq : where to lob off the transform
    """
    f,t,this_stft = scipy.signal.stft(
                scipy.signal.detrend(trial), 
                fs=Fs, 
                window='hanning', 
                nperseg=signal_window, 
                noverlap=signal_window-(signal_window-window_overlap)) 
    return this_stft[f<max_freq]


# Region A
#
#region_a_iters = list(product(*list(map(np.arange,region_a.shape[:3]))))
#
## Test run for calc_stft
#this_iter = region_a_iters[0]
#test_stft = calc_stft(region_a[this_iter],25,Fs,signal_window,window_overlap)
#dat.imshow(np.abs(test_stft));plt.show()
#
#region_a_stft = Parallel(n_jobs = mp.cpu_count()-2)\
#        (delayed(calc_stft)(region_a[this_iter],25,Fs,signal_window,window_overlap)\
#        for this_iter in tqdm(region_a_iters))
#
#region_a_stft_array =\
#        np.empty(tuple((*region_a.shape[:3],*test_stft.shape)),
#                dtype=np.dtype(region_a_stft[0][0,0]))
#for iter_num, this_iter in tqdm(enumerate(region_a_iters)):
#    region_a_stft_array[this_iter] = region_a_stft[iter_num]

# Region B

region_b_iters = list(product(*list(map(np.arange,region_b.shape[:3]))))
this_iter = region_b_iters[0]
test_stft = calc_stft(region_b[this_iter],25,Fs,signal_window,window_overlap)
region_b_stft = Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(calc_stft)(region_b[this_iter],25,Fs,signal_window,window_overlap)\
        for this_iter in tqdm(region_b_iters))

region_b_stft_array =\
        np.empty(tuple((*region_b.shape[:3],*test_stft.shape)),
                dtype=np.dtype(region_b_stft[0][0,0]))
for iter_num, this_iter in tqdm(enumerate(region_b_iters)):
    region_b_stft_array[this_iter] = region_b_stft[iter_num]
# Remove original list
del region_b_stft

# Pick channel with phase most consistently closest to mean
region_b_phases = np.angle(region_b_stft_array)
mean_phase_across_channels = np.mean(region_b_phases,axis=1)
mean_error = [np.mean(np.abs(region_b_phases[:,channel] - mean_phase_across_channels),axis=None)\
        for channel in range(region_b_phases.shape[1])]
# Pick channel with lowest error
region_b_phases_final = region_b_phases[:,np.argmin(mean_error)]
del region_b_phases, mean_phase_across_channels, mean_error
# Plot and confirm resetting
all_phases = region_b_phases_final.reshape(tuple((-1,*region_b_phases_final.shape[2:])))
dat.firing_overview(all_phases.swapaxes(0,1));plt.show()
dat.imshow(region_b_phases_final[0,0]);plt.show()

# To have appropriate f and t
f,t,this_stft = scipy.signal.stft(
            scipy.signal.detrend(region_b[region_b_iters[0]]), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap)) 
valid_f = f[f<25]
#
## Check spectrum and phase-resetting
#stim_time = 2
#this_phase_array = region_a_stft_array
#mean_power = np.mean(np.abs(this_phase_array),axis=(0,1,2))
#mean_normalized_power = mean_power /\
#                np.mean(mean_power[:,t<stim_time],axis=1)[:,np.newaxis]
#mean_normalized_power -=\
#                np.mean(mean_normalized_power[:,t<stim_time],axis=1)[:,np.newaxis]
#plt.pcolormesh(t, valid_f, mean_normalized_power, cmap='jet')
#plt.show()
#
#all_phases =\
#np.angle(this_phase_array).reshape(len(region_a_iters),len(f[f<25]),-1).swapaxes(0,1)
## Plot phases
#plt.plot(all_phases[7,0].T,'x',c='orange');plt.show()
#dat.imshow(all_phases[7]);plt.colorbar();plt.show()

# ____  _                          _               _    _             
#|  _ \| |__   __ _ ___  ___      | |    ___   ___| | _(_)_ __   __ _ 
#| |_) | '_ \ / _` / __|/ _ \_____| |   / _ \ / __| |/ / | '_ \ / _` |
#|  __/| | | | (_| \__ \  __/_____| |__| (_) | (__|   <| | | | | (_| |
#|_|   |_| |_|\__,_|___/\___|     |_____\___/ \___|_|\_\_|_| |_|\__, |
#                                                               |___/ 
# Cross frequency phase-locking of neurons
region_b_phases_final = region_b_phases_final.swapaxes(-1,-2)
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
            np.asarray([region_b_phases_final[this_ind] \
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
