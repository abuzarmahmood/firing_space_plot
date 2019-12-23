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
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA as pca
from sklearn.mixture import GaussianMixture as gmm
from matplotlib.colors import LogNorm


os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extract data
dat = ephys_data('/media/bigdata/brads_data/Brad_LFP_ITI_analyses/BS23')

dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()
dat.get_unit_descriptors()

mean_val = np.mean(dat.all_lfp_array, axis = None)
sd_val = np.std(dat.all_lfp_array, axis = None)
dat.firing_overview(dat.all_lfp_array, min_val = mean_val - 2*sd_val,
                    max_val = mean_val + 2*sd_val, cmap = 'viridis',
                    time_step = 1);plt.show()

dat.firing_overview(dat.all_normalized_firing);plt.show()

# Extract whole session LFPs
with tables.open_file(dat.hdf5_name,'r') as hf5:
    whole_session_lfp_node = hf5.list_nodes('/Whole_session_raw_LFP') 
    whole_lfp = whole_session_lfp_node[0][:]

#Downsample and plot
down_ratio = 10
down_lfp = whole_lfp[:,np.arange(0,whole_lfp.shape[-1],down_ratio)]
dat.imshow(down_lfp);plt.show()
plt.plot(down_lfp.T);plt.show()

# Bandpass filter whole_lfp

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
                (12,25),
                (25,50)]

bandpassed_lfp = np.asarray([
                    butter_bandpass_filter(
                        data = whole_lfp, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])

down_bandpassed_lfp = bandpassed_lfp[:,:,np.arange(0,bandpassed_lfp.shape[-1],down_ratio)]
fig,ax = plt.subplots(2,1,sharex='all')
ax[0].plot(down_lfp.T)
ax[1].plot(down_bandpassed_lfp[0].T);plt.show()

# Extract ITIs from whole session lfp
Fs = 1000 
delivery_times = pd.read_hdf(dat.hdf5_name,'/Whole_session_spikes/delivery_times')
delivery_times['taste'] = delivery_times.index
delivery_times = pd.melt(delivery_times,id_vars = 'taste',var_name ='trial',value_name='time')
delivery_times.sort_values(by='time',inplace=True)
# Delivery times are in 30kHz samples, convert to ms
delivery_times['time'] = delivery_times['time'] // 30

time_before_delivery = 10 #seconds
padding = 1 #second before taste delivery won't be extracted

# (trials x channels x time)
iti_array = np.asarray([whole_lfp[:,(x-(time_before_delivery*Fs)):(x-(padding*Fs))]\
        for x in delivery_times.time])

dat.imshow(np.mean(iti_array,axis=1));plt.colorbar();plt.show()
plt.plot(np.mean(iti_array,axis=1).T);plt.show()

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs

# Spectrogram for ITIs 
signal_window = 2000 
window_overlap = 1950

f,t,iti_spectrograms= scipy.signal.spectrogram(
            scipy.signal.detrend(iti_array), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap), 
            mode='psd')


# Average spectrogram across channels
fmax = 25
mean_iti_spectrograms = np.mean(iti_spectrograms,axis=1)
#mean_iti_specs_long = mean_iti_spectrograms.swapaxes(0,1).\
#        reshape(mean_iti_spectrograms.shape[1],-1)[f<fmax]
#plt.scatter(np.broadcast_to(f[f<fmax,np.newaxis],mean_iti_specs_long.shape).flatten(),
#        mean_iti_specs_long.flatten(),alpha = 0.2)
#plt.show()


# Average power across bands
mean_band_spectrograms = np.asarray(\
        [np.mean(mean_iti_spectrograms[:,(f>band[0])*(f<band[1]),:],axis=1) \
        for band in band_freqs])

# Check for outliers
zscore_trials_power = np.asarray([\
        zscore(band,axis = None) for band in mean_band_spectrograms])
mean_trials_power = np.mean(zscore_trials_power,axis=-1)
plt.plot(mean_trials_power.T,'x');plt.show()

# Plot trials in PCA space

#reduced_trials = pca(n_components = 2).fit_transform(mean_trials_power.T)

###
### Indexes of trials being removed are not consistent
###

#tol = 1e-9
#clf = gmm(n_components = 2,covariance_type = 'full').fit(reduced_trials)
#lowest_ll = []
#lowest_ll.append(np.min(clf.score_samples(reduced_trials)))
#removed_trials = []
#removed_trials.append(np.argmin(clf.score_samples(reduced_trials)))
#remaining_trials = np.where(np.arange(reduced_trials.shape[0]) != removed_trials)[0]
#clf = gmm(n_components = 2,covariance_type = 'full').\
#        fit(reduced_trials[remaining_trials])
#lowest_ll.append(np.min(clf.score_samples(reduced_trials)[remaining_trials]))
#removed_trials.append(np.argmin(clf.score_samples(reduced_trials)[remaining_trials]))
#
#
#while np.abs(np.diff(lowest_ll[-2:]))>tol:
#    remaining_trials = [trial \
#                    for trial in np.arange(reduced_trials.shape[0]) \
#                    if trial not in removed_trials] 
#    clf = gmm(n_components = 2,covariance_type = 'full').\
#            fit(reduced_trials[remaining_trials])
#    lowest_ll.append(np.min(clf.score_samples(reduced_trials)[remaining_trials]))
#    removed_trials.append(np.argmin(clf.score_samples(reduced_trials)[remaining_trials]))
#    print(len(removed_trials))
#
## display predicted scores by the model as a contour plot
#x = np.linspace(-1,7)
#y = np.linspace(-1,1)
#X, Y = np.meshgrid(x, y)
#XX = np.array([X.ravel(), Y.ravel()]).T
#Z = -clf.score_samples(XX)
#Z = Z.reshape(X.shape)
#CS = plt.contour(X, Y, Z,levels=100) 
#plt.scatter(reduced_trials[:, 0], reduced_trials[:, 1], .8)
#plt.show()
#
#plt.plot(np.arange(reduced_trials.shape[0]),clf.score_samples(reduced_trials))
#plt.show()
#
#bad_trials = np.where(clf.score_samples(reduced_trials) < -0.5)[0]

dat.firing_overview(mean_band_spectrograms, 
        time_step = signal_window - window_overlap);plt.show()

fig,ax = plt.subplots(1,len(mean_band_spectrograms),sharey=True)
for ax_num,this_ax in enumerate(ax):
    prio_mean = np.mean(mean_band_spectrograms[ax_num],axis=None)
    prio_std= np.std(mean_band_spectrograms[ax_num],axis=None)
    this_dat = zscore(np.ma.masked_greater(mean_band_spectrograms[ax_num],
                                    prio_mean + prio_std),axis=None)
    mean = np.mean(this_dat,axis=None)
    std= np.std(this_dat,axis=None)
    im = this_ax.pcolormesh(this_dat,
            vmin = mean-std, vmax = mean+std, cmap = 'viridis')
fig.subplots_adjust(bottom = 0.2)
cbar_ax = fig.add_axes([0.15,0.1,0.7,0.02])
plt.colorbar(im, cax = cbar_ax,orientation = 'horizontal', pad = 0.2)
plt.show()



# Average spectrogram for (taste x band)
taste_band_spectrograms = np.asarray([\
        mean_band_spectrograms[:,delivery_times.taste == taste,:] \
        for taste in np.sort(delivery_times.taste.unique())])

fig,ax = plt.subplots(taste_band_spectrograms.shape[0],
                        taste_band_spectrograms.shape[1])
for taste in range(taste_band_spectrograms.shape[0]):
    for band in range(taste_band_spectrograms.shape[1]):
        plt.sca(ax[taste,band])
        dat.imshow(taste_band_spectrograms[taste,band])
plt.show()

# Convert power from spectrogram to DataFrame

# Find average power across ITI interval
taste_band_avg_power = np.mean(taste_band_spectrograms,axis=-1)
nd_idx_objs = make_array_identifiers(taste_band_avg_power)
taste_band_df = pd.DataFrame({\
                        'taste' : nd_idx_objs[0],
                        'band' : nd_idx_objs[1],
                        'trial' : nd_idx_objs[2],
                        'power' : taste_band_avg_power.flatten()})

# Cluster trials into bins for anova
trial_bin_num = 5
taste_band_df['trial_bin'] = pd.cut(taste_band_df.trial,
        bins = trial_bin_num, include_lowest = True, labels = range(trial_bin_num))

# Plot dataframe to visualize
g = sns.FacetGrid(data = \
            taste_band_df, col = 'band', hue = 'taste', sharey=False)
g.map(sns.pointplot, 'trial_bin','power')
plt.legend()
plt.show()


# Perform 2-way ANOVA to look at differences in taste and trial_bin
taste_trial_anova =\
    [taste_band_df.loc[taste_band_df.band == band_num].anova(dv = 'power', \
            between= ['trial_bin','taste'])[['Source','p-unc','np2']] \
            for band_num in np.sort(taste_band_df.band.unique())]

#
