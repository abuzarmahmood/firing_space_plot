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
    ephys_data('/media/bigdata/Abuzar_Data/AM12/AM12_extracted/AM12_4Tastes_191106_085215')
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
f_bool = (f>3) * (f<25)
for taste in range(combined_mean_spec.shape[1]):
    ax[taste,0].pcolormesh(t, f[f_bool], combined_mean_spec[0][taste][f_bool,:], 
            cmap='jet',vmin = vmin, vmax = vmax)
    ax[taste,1].pcolormesh(t, f[f_bool], combined_mean_spec[1][taste][f_bool,:], 
            cmap='jet',vmin = vmin, vmax = vmax)
ax[-1,0].pcolormesh(t, f[f_bool], np.mean(combined_mean_spec[0], axis = 0)[f_bool,:],
            cmap='jet',vmin = vmin, vmax = vmax)
ax[-1,1].pcolormesh(t, f[f_bool], np.mean(combined_mean_spec[1], axis = 0)[f_bool,:],
            cmap='jet',vmin = vmin, vmax = vmax)
plt.show()

# Background normalized and subtracted 
# Fold change of average of power before 2000ms
normalized_region_a_mean_spec = region_a_mean_spec /  \
        np.mean(region_a_mean_spec[:,:,t<2],axis=2)[:,:,np.newaxis]
normalized_region_a_mean_spec -= \
        np.mean(normalized_region_a_mean_spec[:,:,t<2],axis=2)[:,:,np.newaxis]
normalized_region_b_mean_spec = region_b_mean_spec / \
        np.mean(region_b_mean_spec[:,:,t<2],axis=2)[:,:,np.newaxis]
normalized_region_b_mean_spec -= \
        np.mean(normalized_region_b_mean_spec[:,:,t<2],axis=2)[:,:,np.newaxis]

ax = plt.subplot(211)
ax.pcolormesh(t, f[f_bool], np.mean(normalized_region_a_mean_spec,
    axis=0)[f_bool,:], cmap='jet')
plt.title('Region A')
ax = plt.subplot(212)
ax.pcolormesh(t, f[f_bool], np.mean(normalized_region_b_mean_spec,
    axis = 0)[f_bool,:], cmap='jet')
plt.title('Region B')
plt.show()

# ____                  _ ____               
#| __ )  __ _ _ __   __| |  _ \ __ _ ___ ___ 
#|  _ \ / _` | '_ \ / _` | |_) / _` / __/ __|
#| |_) | (_| | | | | (_| |  __/ (_| \__ \__ \
#|____/ \__,_|_| |_|\__,_|_|   \__,_|___/___/
                                            

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

#band_freqs = [(x,x+1) for x in range(1,50)]


# (band x taste x channel x trial x time)
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

def error_plot(array,ax):
    """
    Array with dims (... x time)
    """
    array = array.reshape(-1,array.shape[-1])
    mean = np.mean(array, axis= 0) 
    std = np.std(array, axis= 0)
    ax.fill_between(\
            x = range(array.shape[-1]),
            y1 = mean + std,
            y2 = mean - std,
            alpha = 0.5)
    ax.plot(range(array.shape[-1]),mean)

# Mean plots for each band
fig,ax = plt.subplots(combined_mean_spec.shape[1],2, sharex='all',sharey='row')
for band in range(bandpassed_region_a.shape[0]):
    error_plot(bandpassed_region_a[band], ax[band,0])    
    error_plot(bandpassed_region_b[band], ax[band,1])
plt.show()

# Average band, taste plots for every region
fig, ax = plt.subplots(bandpassed_region_a.shape[0],
        bandpassed_region_a.shape[1], sharex='all',sharey='row')
for band in range(bandpassed_region_a.shape[0]):
    for taste in range(bandpassed_region_a.shape[1]):
        error_plot(bandpassed_region_a[band,taste],
                ax[band,taste])
fig, ax = plt.subplots(bandpassed_region_b.shape[0],
        bandpassed_region_b.shape[1], sharex='all',sharey='row')
for band in range(bandpassed_region_b.shape[0]):
    for taste in range(bandpassed_region_b.shape[1]):
        error_plot(bandpassed_region_b[band,taste],
                ax[band,taste])
plt.show()

# ____  _                    ____  _          __  __ 
#|  _ \| |__   __ _ ___  ___/ ___|| |_ _   _ / _|/ _|
#| |_) | '_ \ / _` / __|/ _ \___ \| __| | | | |_| |_ 
#|  __/| | | | (_| \__ \  __/___) | |_| |_| |  _|  _|
#|_|   |_| |_|\__,_|___/\___|____/ \__|\__,_|_| |_|  
                                                    
# Hilbert transform on all bandpassed signals
hilbert_bandpass_lfp = hilbert(bandpassed_lfp) 

# Extract phase for all signals
lfp_phase = np.angle(hilbert_bandpass_lfp)
lfp_phase_a = lfp_phase[:,:,middle_channels_bool]
lfp_phase_b = lfp_phase[:,:,~middle_channels_bool]


# Extract amplitude envelope for all signals
lfp_amplitude = np.abs(hilbert_bandpass_lfp)
lfp_amplitude_a = lfp_amplitude[:,:,middle_channels_bool]
lfp_amplitude_b = lfp_amplitude[:,:,~middle_channels_bool]

# Plot test examples of amplitude and phase to make sure things are working
random_trial = tuple((np.random.choice(range(hilbert_bandpass_lfp.shape[i])) \
               for i in range(len(hilbert_bandpass_lfp.shape)-1))) 
ax1 = plt.subplot(211)
ax1.plot(hilbert_bandpass_lfp[random_trial])
ax1.plot(lfp_amplitude[random_trial],c='orange')
ax1.plot(-lfp_amplitude[random_trial],c='orange')
ax2 = plt.subplot(212, sharex = ax1)
ax2.plot(lfp_phase[random_trial])
plt.show()

# Mean amplitude plots for each band
fig,ax = plt.subplots(combined_mean_spec.shape[1],2, sharex='all',sharey='row')
for band in range(bandpassed_region_a.shape[0]):
    error_plot(lfp_amplitude_a[band], ax[band,0])    
    error_plot(lfp_amplitude_b[band], ax[band,1])
plt.show()

# Average ampltide for band, taste plots for every region
fig, ax = plt.subplots(lfp_amplitude_a.shape[0],
        lfp_amplitude_a.shape[1], sharex='all',sharey='row')
for band in range(lfp_amplitude_a.shape[0]):
    for taste in range(lfp_amplitude_a.shape[1]):
        error_plot(lfp_amplitude_a[band,taste],
                ax[band,taste])
fig, ax = plt.subplots(lfp_amplitude_b.shape[0],
        lfp_amplitude_b.shape[1], sharex='all',sharey='row')
for band in range(lfp_amplitude_b.shape[0]):
    for taste in range(lfp_amplitude_b.shape[1]):
        error_plot(lfp_amplitude_b[band,taste],
                ax[band,taste])
plt.show()

# Average phase consistency
def phase_consistency_plot(array,
                            ax = None, 
                            bootsample_fraction = 0.2, 
                            boot_iters = 100):
    """
    Array with dims (... x time)
    Phase will be average in all dimensions supplied except time
    """
    phase_vectors = np.exp(array.reshape(-1,array.shape[-1])*1.j)
    mean = np.abs(np.mean(phase_vectors, axis= 0)) 
    bootsample = np.array(\
            [np.abs(np.mean(\
                resample(phase_vectors,
                    n_samples = np.int(phase_vectors.shape[0]*bootsample_fraction)),
                axis=0)) \
            for x in range(boot_iters)])
    std = np.std(bootsample, axis= 0) 
    if ax is not None:
        ax.plot(mean)
        ax.fill_between(\
                x = range(array.shape[-1]),
                y1 = mean + std,
                y2 = mean - std,
                alpha = 0.5)
        return mean, std, ax
    else:
        return mean, std

# Average phase consistency for every band in both regions
fig,ax = plt.subplots(combined_mean_spec.shape[1],2, sharex='all',sharey='all')
for band in range(bandpassed_region_a.shape[0]):
    phase_consistency_plot(lfp_phase_a[band], ax[band,0], 
            bootsample_fraction = 0.3, boot_iters = 20)    
    phase_consistency_plot(lfp_phase_b[band], ax[band,1], 
            bootsample_fraction = 0.3, boot_iters = 20)
plt.show()

# Does phase consistency in BLA get washed out by averaging

# Taste dependent phase consistency
fig, ax = plt.subplots(lfp_phase_a.shape[0],
        lfp_phase_a.shape[1], sharex='all',sharey='all')
for band in range(lfp_phase_a.shape[0]):
    for taste in range(lfp_phase_a.shape[1]):
        phase_consistency_plot(lfp_phase_a[band,taste],
                ax[band,taste])
fig, ax = plt.subplots(lfp_phase_b.shape[0],
        lfp_phase_b.shape[1], sharex='all',sharey='all')
for band in range(lfp_phase_b.shape[0]):
    for taste in range(lfp_phase_b.shape[1]):
        phase_consistency_plot(lfp_phase_b[band,taste],
                ax[band,taste])
plt.show()


# Look at single trials
band = 0
this_dat = \
    bandpassed_region_a[band].reshape(-1,bandpassed_region_a.shape[-1])[:,1000:3000]
mean = np.mean(this_dat, axis = None)
std = np.std(this_dat, axis = None)
dat.imshow(this_dat)
plt.clim(mean-std,mean+std)
plt.show()

fig, ax = plt.subplots(1,len(lfp_phase_a))
for band,this_ax in enumerate(ax):
    this_dat = lfp_phase_a[band,:,1].reshape(-1,lfp_phase_a.shape[-1])[:,1000:3000]
    plt.sca(this_ax)
    dat.imshow(this_dat)
plt.show()

# Within channel phase consistency for BLA
this_dat = lfp_phase_b
iters = list(product(*list(map(np.arange,this_dat.shape[:3]))))
phase_consistency_array =\
    np.zeros(tuple((*this_dat.shape[:3],this_dat.shape[-1])))
for this_iter in tqdm(iters):
    phase_consistency_array[this_iter], a = \
            phase_consistency_plot(this_dat[this_iter])

# Heatmap instead of taking mean
fig, ax = plt.subplots(phase_consistency_array.shape[0],
         phase_consistency_array.shape[1], sharex='all',sharey='all')
for band in range(phase_consistency_array.shape[0]):
    for taste in range(phase_consistency_array.shape[1]):
        im = ax[band,taste].imshow(phase_consistency_array[band,taste],
                vmin = 0, vmax = 1, interpolation='nearest',aspect='auto')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(im,cbar_ax)
plt.show()

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


region_b_iters = list(product(*list(map(np.arange,region_b.shape[:3]))))

# Test run for calc_stft
this_iter = region_b_iters[0]
test_stft = calc_stft(region_b[this_iter],25,Fs,signal_window,window_overlap)
dat.imshow(np.abs(test_stft));plt.show()

region_b_stft = Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(calc_stft)(region_b[this_iter],25,Fs,signal_window,window_overlap)\
        for this_iter in tqdm(region_b_iters))

region_b_stft_array =\
        np.empty(tuple((*region_b.shape[:3],*test_stft.shape)),
                dtype=np.dtype(region_b_stft[0][0,0]))
for iter_num, this_iter in tqdm(enumerate(region_b_iters)):
    region_b_stft_array[this_iter] = region_b_stft[iter_num]

# To have appropriate f and t
f,t,this_stft = scipy.signal.stft(
            scipy.signal.detrend(region_b[region_b_iters[0]]), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap)) 
valid_f = f[f<25]

# Check spectrum and phase-locking
stim_time = 2
mean_power = np.mean(np.abs(region_b_stft_array),axis=(0,1,2))
mean_normalized_power = mean_power /\
                np.mean(mean_power[:,t<stim_time],axis=1)[:,np.newaxis]
mean_normalized_power -=\
                np.mean(mean_normalized_power[:,t<stim_time],axis=1)[:,np.newaxis]
plt.pcolormesh(t, valid_f, mean_normalized_power, cmap='jet')
plt.show()

all_phases =\
np.angle(region_b_stft_array).reshape(len(region_b_iters),len(f[f<25]),-1).swapaxes(0,1)
# Plot phases
plt.plot(all_phases[7,0].T,'x',c='orange');plt.show()
dat.imshow(all_phases[7]);plt.colorbar();plt.show()

# Check that the stft is working as expected by visualizing power and phase
# Check that stft agrees with hilbert

# Plot test examples of amplitude and phase to make sure things are working
this_dat = region_b

random_trial = tuple((np.random.choice(range(this_dat.shape[i])) \
               for i in range(len(this_dat.shape)-1))) 

f,t,region_b_stft = scipy.signal.stft(
            scipy.signal.detrend(region_b[random_trial]), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap)) 

fig, ax = plt.subplots(len(band_freqs),2, sharex='col')
for band_num, band in enumerate(band_freqs):
    ax[band_num,0].plot(np.arange(lfp_amplitude_b.shape[-1]),
            zscore(lfp_amplitude_b[tuple((band_num,*random_trial))]))
    ax[band_num,0].plot(t*Fs,
            zscore(np.mean(np.abs(region_b_stft[(f>band[0])&(f<band[1])]),axis=0)))
    ax[band_num,1].plot(np.arange(bandpassed_region_b.shape[-1]),
            np.angle(hilbert(
                butter_bandpass_filter(
                            data = this_dat[random_trial],
                            lowcut = band[0]-0.5,
                            highcut = band[0]+0.5,
                            fs = Fs)
                )))
    ax[band_num,1].plot(t*Fs,
            np.angle(region_b_stft[(np.argmin(np.abs(f-band[0])))]))
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
from scipy.signal import chirp

dt = 1e-3
t = np.arange(0,7,dt)
freq_list = np.arange(2,12,1.5)
sin_array = np.array([(2**freq)*np.sin(2*np.pi*freq*t) for freq in freq_list])
sum_sin = np.sum(sin_array,axis=0) + \
   np.random.normal(size=len(t))*0.5
sum_sin = chirp(t,f0=min(freq_list),f1=max(freq_list),t1=max(t),method='linear')
# Pad the wave in the beginning
sum_sin = np.concatenate((np.ones(int(2/dt)),sum_sin))
t = np.arange(0,9,dt)
plt.subplot(211)
plt.imshow(sin_array,interpolation='nearest',aspect='auto')
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
plt.pcolormesh(t, f[f<freq_list[-1]*2], 10*np.log10(Sxx[f<freq_list[-1]*2,:]), cmap='jet')
plt.show()

recovered_sin = np.array([butter_bandpass_filter(sum_sin, 
                                                freq-0.5, 
                                                freq+0.5,
                                                Fs,
                                                order=3) \
                    for freq in freq_list])
fig,ax = plt.subplots(len(freq_list))
for i in range(len(freq_list)):
    #ax[i].plot(sin_array[i,:])
    ax[i].plot(recovered_sin[i,:])
plt.show()


# Spectrogram comparison

# Synethetic data

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
