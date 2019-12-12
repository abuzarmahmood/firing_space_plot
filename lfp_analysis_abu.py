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


os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extract data
dat = \
ephys_data('/media/bigdata/Abuzar_Data/AM17/AM17_extracted/AM17_4Tastes_191126_084934')

dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()
dat.firing_overview(dat.all_normalized_firing);plt.show()

# _     _____ ____    ____  _          __  __ 
#| |   |  ___|  _ \  / ___|| |_ _   _ / _|/ _|
#| |   | |_  | |_) | \___ \| __| | | | |_| |_ 
#| |___|  _| |  __/   ___) | |_| |_| |  _|  _|
#|_____|_|   |_|     |____/ \__|\__,_|_| |_|  
                                             
# Extract channel numbers for lfp
with tables.open_file(dat.hdf5_name,'r') as hf5:
    parsed_lfp_channels = hf5.root.Parsed_LFP_channels[:]

middle_channels = np.arange(8,24)

# Calculate clims
mean_val = np.mean(dat.all_lfp_array, axis = None)
sd_val = np.std(dat.all_lfp_array, axis = None)
dat.firing_overview(dat.all_lfp_array, min_val = mean_val - 2*sd_val,
                    max_val = mean_val + 2*sd_val, cmap = 'viridis');plt.show()

# Mean LFP spectrogram 
middle_channels_bool = np.array([True if channel in middle_channels else False \
        for channel in parsed_lfp_channels ])
region_a = dat.lfp_array[:,middle_channels_bool,:,:] 
region_b = dat.lfp_array[:,~middle_channels_bool,:,:]
region_a_mean = np.mean(region_a,axis=(0,1,2))
region_b_mean = np.mean(region_b,axis=(0,1,2))
region_a_std = np.std(region_a,axis=(0,1,2))
region_b_std = np.std(region_b,axis=(0,1,2))

# Timeseries plot
plt.fill_between(x = range(len(region_a_mean)), 
        y1 = region_a_mean - region_a_std, 
        y2 = region_a_mean + region_a_std, alpha = 0.5)
plt.fill_between(x = range(len(region_b_mean)), 
        y1 = region_b_mean - region_b_std, 
        y2 = region_b_mean + region_b_std, alpha = 0.5)
plt.plot(region_a_mean);plt.plot(region_b_mean)
plt.show()

# Mean spectrogram of all responses
Fs = 1000 
signal_window = 2000 
window_overlap = 1950

f,t,region_a_spectrograms= scipy.signal.spectrogram(
            scipy.signal.detrend(region_a), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap), 
            mode='psd')

f,t,region_b_spectrograms= scipy.signal.spectrogram(
            scipy.signal.detrend(region_b), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap), 
            mode='psd')

region_a_mean_spec = np.mean(region_a_spectrograms,axis=(1,2))
region_b_mean_spec = np.mean(region_b_spectrograms,axis=(1,2))
combined_mean_spec = np.stack((region_a_mean_spec,region_b_mean_spec))

fig,ax = plt.subplots(combined_mean_spec.shape[1]+1,2)
vmean = np.mean(combined_mean_spec,axis=None)
vstd = np.std(combined_mean_spec,axis=None)
vmin = None 
vmax = None 
for taste in range(combined_mean_spec.shape[1]):
    ax[taste,0].pcolormesh(t, f[f<15], combined_mean_spec[0][taste][f<15,:], 
            cmap='jet',vmin = vmin, vmax = vmax)
    ax[taste,1].pcolormesh(t, f[f<15], combined_mean_spec[1][taste][f<15,:], 
            cmap='jet',vmin = vmin, vmax = vmax)
ax[-1,0].pcolormesh(t, f[f<15], np.mean(combined_mean_spec[0], axis = 0)[f<15,:],
            cmap='jet',vmin = vmin, vmax = vmax)
ax[-1,1].pcolormesh(t, f[f<15], np.mean(combined_mean_spec[1], axis = 0)[f<15,:],
            cmap='jet',vmin = vmin, vmax = vmax)

plt.show()

# Background normalized 
# Fold change of average of power before 2000ms
normalized_region_a_mean_spec = region_a_mean_spec /  \
        np.mean(region_a_mean_spec[:,:,t<2],axis=2)[:,:,np.newaxis]
normalized_region_b_mean_spec = region_b_mean_spec / \
        np.mean(region_b_mean_spec[:,:,t<2],axis=2)[:,:,np.newaxis]

ax = plt.subplot(211)
ax.pcolormesh(t, f[f<25], np.mean(normalized_region_a_mean_spec, axis=0)[f<25,:], cmap='viridis')
plt.title('Region A')
ax = plt.subplot(212)
ax.pcolormesh(t, f[f<25], np.mean(normalized_region_b_mean_spec,axis = 0)[f<25,:], cmap='viridis')
plt.title('Region B')
plt.show()


# ____  _                    ____  _          __  __ 
#|  _ \| |__   __ _ ___  ___/ ___|| |_ _   _ / _|/ _|
#| |_) | '_ \ / _` / __|/ _ \___ \| __| | | | |_| |_ 
#|  __/| | | | (_| \__ \  __/___) | |_| |_| |  _|  _|
#|_|   |_| |_|\__,_|___/\___|____/ \__|\__,_|_| |_|  
                                                    
#define bandpass filter parameters to parse out frequencies
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

band_freqs = [(1,4),
                (4,7),
                (7,12),
                (12,25)]

bandpassed_lfp = np.asarray([
                    butter_bandpass_filter(
                        data = dat.lfp_array,
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])

# Plot mean bandpassed lfps for every region
bandpassed_region_a = bandpassed_lfp[:,:,middle_channels_bool] 
bandpassed_region_b = bandpassed_lfp[:,:,~middle_channels_bool] 

bandpassed_region_a_mean = np.mean(bandpassed_region_a,axis=(1,2,3))
bandpassed_region_b_mean = np.mean(bandpassed_region_b,axis=(1,2,3))
bandpassed_region_a_std = np.std(bandpassed_region_a,axis=(1,2,3))
bandpassed_region_b_std = np.std(bandpassed_region_b,axis=(1,2,3))


# Mean plots for each band
fig,ax = plt.subplots(combined_mean_spec.shape[1],2)
for band in range(bandpassed_region_a.shape[0]):
    ax[band,0].fill_between(\
            x = range(bandpassed_region_a_mean.shape[-1]),
            y1 = bandpassed_region_a_mean[band] -\
                bandpassed_region_a_std[band],
            y2 = bandpassed_region_a_mean[band] +\
                bandpassed_region_a_std[band],
            alpha = 0.5)
    ax[band,0].plot(\
            range(bandpassed_region_a_mean.shape[-1]),
            bandpassed_region_a_mean[band])
    ax[band,1].fill_between(\
            x = range(bandpassed_region_b_mean.shape[-1]),
            y1 = bandpassed_region_b_mean[band] -\
                bandpassed_region_b_std[band],
            y2 = bandpassed_region_b_mean[band] +\
                bandpassed_region_b_std[band],
            alpha = 0.5)
    ax[band,1].plot(\
            range(bandpassed_region_b_mean.shape[-1]),
            bandpassed_region_b_mean[band])
plt.show()



# ____                 _       _     
#/ ___|  ___ _ __ __ _| |_ ___| |__  
#\___ \ / __| '__/ _` | __/ __| '_ \ 
# ___) | (__| | | (_| | || (__| | | |
#|____/ \___|_|  \__,_|\__\___|_| |_|
#                                    

# Filter visualization plot
this_band = band_freqs[-1]
b, a = butter_bandpass(this_band[0], this_band[1], fs, order = 2)
w, h = scipy.signal.freqz(b, a, worN=2000)
plt.plot((fs * 0.5 / np.pi) * w, abs(h), '-x') 
plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
plt.vlines(this_band[0],0,1)
plt.vlines(this_band[1],0,1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.xlim([0,2*this_band[1]])
plt.grid(True)
plt.legend(loc='best')
plt.show()

# Empirical test
dt = 1e-3
t = np.arange(0,7,dt)
freq_list = np.arange(2,12,1.5)
sin_array = np.array([(2**freq)*np.sin(2*np.pi*freq*t) for freq in freq_list])
sum_sin = np.sum(sin_array,axis=0) + \
   np.random.normal(size=len(t))*0.5
plt.subplot(211)
dat.imshow(sin_array)
plt.subplot(212)
plt.plot(sum_sin)
plt.show()

Fs = 1/dt 
signal_window = 2000 
window_overlap = 1950
f,t,Sxx= scipy.signal.spectrogram(
            scipy.signal.detrend(sum_sin), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap), 
            mode='psd')
plt.pcolormesh(t, f[f<freq_list[-1]*2], Sxx[f<freq_list[-1]*2,:], cmap='jet')
plt.show()

recovered_sin = np.array([butter_bandpass_filter(sum_sin, 
                                                freq-1, 
                                                freq+1,
                                                Fs,
                                                order=3) \
                    for freq in freq_list])
fig,ax = plt.subplots(len(freq_list))
for i in range(len(freq_list)):
    ax[i].plot(sin_array[i,:])
    ax[i].plot(recovered_sin[i,:])
plt.show()


# Spectrogram comparison

# Synethetic data
from scipy.signal import chirp

t = np.linspace(0, 100, 50001)
w = chirp(t, f0=20, f1=1, t1=100, method='linear')
t_down = t[np.arange(0,len(w),10)]
w_down = w[np.arange(0,len(w),10)]
plt.plot(t, w);plt.plot(t_down,w_down);plt.show()

f, t, Sxx = spectrogram(w, 10, nperseg = 1000)
ax = plt.subplot(211)
ax.pcolormesh(t,f[f<0.5],Sxx[f<0.5,:])

f, t, Sxx = spectrogram(w_down, 1,  nperseg = 100)
ax = plt.subplot(212)
ax.pcolormesh(t,f[f<0.5],Sxx[f<0.5,:])

plt.show()

# Actual data
test_dat = dat.all_lfp_array[0,0,:]

ax = plt.subplot(411)
ax.plot(test_dat)

fs = 1000

f, t, Sxx = spectrogram(test_dat, fs, nperseg = 100)

ax = plt.subplot(412)
ax.pcolormesh(t, f[f<50], Sxx[f<50,:])
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')

f, t, Sxx = spectrogram_lspopt(test_dat, fs, c_parameter=20.0)

ax = plt.subplot(413)
ax.pcolormesh(t, f[f<50], Sxx[f<50,:])
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')


# From LFP_Spectrogram_Stone

#create timing variables
Fs = 1000 
signal_window = 900 
window_overlap = 850

f, t_spec, x_spec = scipy.signal.spectrogram(
        scipy.signal.detrend(test_dat), 
        fs=Fs, 
        window='hanning', 
        nperseg=signal_window, 
        noverlap=signal_window-(signal_window-window_overlap), 
        mode='psd')

ax = plt.subplot(414)
ax.pcolormesh(t_spec, f[f<12], x_spec[f<12,:])
plt.show()
