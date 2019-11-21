# Import stuff!
import matplotlib.pyplot as plt
import tables
import os
import numpy as np
import glob
from skimage import exposure
from scipy.signal import hilbert
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import savgol_filter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

## Load data
#dir_list = ['/media/bigdata/Abuzar_Data/run_this_file']
dir_list = ['/media/bigdata/firing_space_plot']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.npy',recursive=True)

raw_emg = np.load(file_list[0])
filtered_emg = np.load(file_list[1])

raw_emg_long = raw_emg.reshape(\
        (raw_emg.shape[0], np.prod(raw_emg.shape[1:3]),raw_emg.shape[-1]))

filtered_emg_long = filtered_emg.reshape(\
        (np.prod(filtered_emg.shape[:2]),filtered_emg.shape[-1]))

filt_filt_emg = butter_bandpass_filter(
        data = raw_emg_long,
        lowcut = 5,
        highcut = 10,
        fs = 1000)

analytic_signal = np.abs(hilbert(filtered_emg_long)) 
smooth_analytic = savgol_filter(analytic_signal,51,2)

def dat_imshow(x):
    plt.imshow(x,interpolation='nearest',aspect='auto')

