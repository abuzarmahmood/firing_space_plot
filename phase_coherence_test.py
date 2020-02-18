

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
from scipy.stats import zscore


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
phase_jitter = np.random.random(trial_num) * 2*np.pi

# Single test case
#y1 = np.sin(2*np.pi*freq*x + phase_vec) + np.random.random(x.shape)*noise
#y2 = np.sin(2*np.pi*freq*x + phase_vec + phase_offset) + np.random.random(x.shape)*noise
#plt.subplot(211)
#plt.plot(x,phase_offset)
#plt.subplot(212)
#plt.plot(x,y1)
#plt.plot(x,y2)
#plt.show()

def img_plot(array):
    plt.imshow(array, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

## Generate multiple trials
#x_mat = np.broadcast_to(x, (trial_num, len(x)))
##jitter_mat = np.broadcast_to(phase_jitter[:,np.newaxis],x_mat.shape)
#jitter_mat = np.zeros(x_mat.shape)
#phase_mat = np.broadcast_to(phase_vec[np.newaxis, :],x_mat.shape)
##offset_mat = np.broadcast_to(phase_offset[np.newaxis, :],x_mat.shape)
#offset_mat = np.zeros(x_mat.shape)
#y1 = np.sin(2*np.pi*freq*x_mat[0] + phase_mat + jitter_mat) + np.random.random(x.shape)*noise
#y2 = np.sin(2*np.pi*freq*x_mat[0] + phase_mat + jitter_mat + offset_mat) \
#        + np.random.random(x.shape)*noise
#
#plt.subplot(211)
#plt.imshow(y1, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
#plt.subplot(212)
#plt.imshow(y2, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
#plt.show()
#
# Different test case
# First half of trials simply copies of the same signal
# Second half of trials have random offsets
# Add noise to both signals
x_mat = np.broadcast_to(x, (trial_num, len(x)))
jitter_mat = np.broadcast_to(phase_jitter[:,np.newaxis],x_mat.shape)
x_jitter_mat = x_mat + jitter_mat
fin_x1_mat = np.concatenate((x_mat, x_jitter_mat),axis=-1)
offset_mat = np.random.random(x_mat.shape) * 2 * np.pi 
x_jitter_offset_mat = x_jitter_mat + offset_mat
fin_x2_mat = np.concatenate((x_mat,x_jitter_offset_mat),axis = -1)

y1 = np.sin(2*np.pi*freq*fin_x1_mat)
y1 += np.random.random(y1.shape)*noise
y2 = np.sin(2*np.pi*freq*fin_x2_mat)
y2 += np.random.random(y1.shape)*noise

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(np.mean(y1,axis=0))
ax[1].plot(np.mean(y2,axis=0))
plt.show()

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].imshow(y1, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
ax[1].imshow(y2, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
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
    #this_stft = this_stft[:,np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))[0]]
    #return f[f<max_freq],t[np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))],this_stft
    return f[f<max_freq],t,this_stft

y1_stft_tuple = np.array([calc_stft(
                        x,
                        max_freq,
                        (np.min(x),np.max(x)),
                        Fs,
                        signal_window,
                        window_overlap) for x in tqdm(y1)])


y2_stft_tuple = np.array([calc_stft(
                        x,
                        max_freq,
                        (np.min(x),np.max(x)),
                        Fs,
                        signal_window,
                        window_overlap) for x in tqdm(y2)])

y1_stft = [x[-1] for x in y1_stft_tuple] 
y2_stft = [x[-1] for x in y2_stft_tuple] 

# Chop to consistent length
y1_tmin = np.min([x.shape[1] for x in y1_stft])
y2_tmin = np.min([x.shape[1] for x in y2_stft])
fin_t = np.min((y1_tmin, y2_tmin))
y1_stft = np.array([x[...,:fin_t] for x in y1_stft] )
y2_stft = np.array([x[...,:fin_t] for x in y2_stft] )

y1_amplitude = np.abs(y1_stft) #np.asarray([np.abs(x) for x in y1_stft])
y2_amplitude = np.abs(y2_stft) #np.asarray([np.abs(x) for x in y2_stft])

y1_mean_power = np.mean(y1_amplitude,axis=0)
y2_mean_power = np.mean(y2_amplitude,axis=0)

plt.subplot(211)
plt.imshow(y1_mean_power, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.subplot(212)
plt.imshow(y2_mean_power, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.show()

y1_phase = np.angle(y1_stft) #np.asarray([np.angle(x) for x in y1_stft])
y2_phase = np.angle(y2_stft) #np.asarray([np.angle(x) for x in y2_stft])

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].imshow(y1_phase[0], interpolation = 'nearest', aspect = 'auto', origin = 'lower')
ax[1].imshow(y2_phase[0], interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.show()

# Calculate phase diff
all_phase_array = np.stack((y1_phase,y2_phase))

# Calculate difference, exponentiate, then take mean
#phase_diff_array = np.squeeze(np.diff(all_phase_array, axis = 0))
#mean_coherence = np.abs(np.mean(np.exp(-1.j*phase_diff_array),axis=0))

# Exponentiate, take difference, then take mean
# *** BAD IDEA ***
#phase_diff_array = np.squeeze(np.diff(np.exp(-1.j*all_phase_array), axis = 0))
#mean_coherence = np.abs(np.mean(phase_diff_array,axis=0))

# Exponentiate, divide, then take mean
imaginary_phase = np.exp(-1.j*all_phase_array)
phase_diff_array = np.divide(imaginary_phase[0],imaginary_phase[1])
mean_coherence = np.abs(np.mean(phase_diff_array,axis=0))

freq_vec = y1_stft_tuple[0][0]
time_vec = y1_stft_tuple[0][1][:fin_t]
#img_plot(zscore(mean_coherence,axis=-1));plt.colorbar();plt.show()
img_plot(mean_coherence);plt.colorbar();plt.show()

plt.plot(mean_coherence.T);plt.show()

#  ____      _                                     __                     
# / ___|___ | |__   ___ _ __ ___ _ __   ___ ___   / _|_ __ ___  _ __ ___  
#| |   / _ \| '_ \ / _ \ '__/ _ \ '_ \ / __/ _ \ | |_| '__/ _ \| '_ ` _ \ 
#| |__| (_) | | | |  __/ | |  __/ | | | (_|  __/ |  _| | | (_) | | | | | |
# \____\___/|_| |_|\___|_|  \___|_| |_|\___\___| |_| |_|  \___/|_| |_| |_|
#                                                                         
# ____ _____ _____ _____ 
#/ ___|_   _|  ___|_   _|
#\___ \ | | | |_    | |  
# ___) || | |  _|   | |  
#|____/ |_| |_|     |_|  
                        
"""
Refer to 
http://math.bu.edu/people/mak/sfn-2013/sfn_tutorial.pdf 
Slide 29,30
"""
def calc_coherence(stft_a, stft_b):
    """
    inputs : arrays of shape (trials x freq x time)
    """
    cross_spec = np.mean(stft_a * np.conj(stft_b),axis=0)
    a_power_spectrum = np.mean(np.abs(stft_a)**2,axis=0)
    b_power_spectrum = np.mean(np.abs(stft_b)**2,axis=0)
    coherence = np.abs(cross_spec)/np.sqrt(a_power_spectrum*b_power_spectrum)
    return coherence

#cross_spectrum = np.mean(y1_stft * np.conj(y2_stft),axis=0)
#y1_power_spectrum = np.real(np.mean(y1_stft * np.conj(y1_stft),axis=0))
#y2_power_spectrum = np.real(np.mean(y2_stft * np.conj(y2_stft),axis=0))
#y1_power_spectrum = np.mean(np.abs(y1_stft)**2,axis=0)
#y2_power_spectrum = np.mean(np.abs(y2_stft)**2,axis=0)

#coherence = np.abs(cross_spectrum) / np.sqrt(y1_power_spectrum * y2_power_spectrum)

coherence = calc_coherence(y1_stft,y2_stft)
img_plot(coherence);plt.show()
