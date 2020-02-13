

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
from sklearn.utils import resample
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data


# Generate test signal
dt = 1e-3
tmax = 20
trial_num = 20
freq = 7
noise = 1.5
phase_vec = np.arange(tmax, step = dt)/(2*np.pi)
phase_offset_func = lambda x,mid : -(x**2 - mid*x) 
x = np.arange(0,tmax, step = dt)
phase_offset = phase_offset_func(x, tmax//2) 
phase_jitter = np.random.random(trial_num) + 2*np.pi

# Single test case
y1 = np.sin(2*np.pi*freq*x + phase_vec) + np.random.random(x.shape)*noise
y2 = np.sin(2*np.pi*freq*x + phase_vec + phase_offset) + np.random.random(x.shape)*noise
plt.subplot(211)
plt.plot(x,phase_offset)
plt.subplot(212)
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()

def img_plot(array):
    plt.imshow(array, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

# Generate multiple trials
x_mat = np.broadcast_to(x, (trial_num, len(x)))
jitter_mat = np.broadcast_to(phase_jitter[:,np.newaxis],x_mat.shape)
phase_mat = np.broadcast_to(phase_vec[np.newaxis, :],x_mat.shape)
offset_mat = np.broadcast_to(phase_offset[np.newaxis, :],x_mat.shape)
y1 = np.sin(2*np.pi*freq*x_mat[0] + phase_mat + jitter_mat) + np.random.random(x.shape)*noise
y2 = np.sin(2*np.pi*freq*x_mat[0] + phase_mat + jitter_mat + offset_mat) \
        + np.random.random(x.shape)*noise

plt.subplot(211)
plt.imshow(y1, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.subplot(212)
plt.imshow(y2, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.show()

# _____ _ _ _            
#|  ___(_) | |_ ___ _ __ 
#| |_  | | | __/ _ \ '__|
#|  _| | | | ||  __/ |   
#|_|   |_|_|\__\___|_|   
#                        
#

# Resolution has to be increased for phase of higher frequencies
Fs = 1000 
signal_window = 500 
window_overlap = 499
max_freq = 25
time_range_tuple = (0,5)

# Define function to parse out only wanted frequencies in STFT
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
    return f[f<max_freq],t[np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))],this_stft

y1_filt = np.array([calc_stft(x,
                        max_freq,
                        (np.min(x),np.max(x)),
                        Fs,
                        signal_window,
                        window_overlap) for x in tqdm(y1)])

y2_filt = np.array([calc_stft(x,
                        max_freq,
                        (np.min(x),np.max(x)),
                        Fs,
                        signal_window,
                        window_overlap) for x in tqdm(y2)])

y1_phase = np.asarray([np.angle(x[-1]) for x in y1_filt])
y2_phase = np.asarray([np.angle(x[-1]) for x in y2_filt])

plt.subplot(211)
plt.imshow(y1_phase, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.subplot(212)
plt.imshow(y2_phase, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.show()

