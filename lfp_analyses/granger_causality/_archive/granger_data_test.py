# Perform granger causality test on the data
# Steps:
# 1) Load data
# 2) Preprocess data:
#     a) Remove trials with artifacts
#     b) Detrend single-trial data
#     c) Remove temporal mean from single-trial data
#     d) Dvidide by temporal standard deviation
#     e) Subtract mean across trials from each trial (for each timepoint)
#     f) Divide by standard deviation across trials (for each timepoint)
# 3) Perform Augmented Dickey-Fuller test on each channel to check for stationarity
# 4) Perform Granger Causality test on each channel pair
# 5) Test for good fitting by checking that residuals are white noise
# 6) Calculate significance of granger causality by shuffling trials
# 7) Plot results
#
# References:
# Ding, Mingzhou, et al. “Short-Window Spectral Analysis of Cortical Event-Related Potentials by Adaptive Multivariate Autoregressive Modeling: Data Preprocessing, Model Validation, and Variability Assessment.” Biological Cybernetics, vol. 83, no. 1, June 2000, pp. 35–45, https://doi.org/10.1007/s004229900137.


############################################################
## Imports
############################################################

import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
sys.path.append(ephys_data_dir)
from ephys_data import ephys_data
import numpy as np
import pylab as plt
from scipy.signal import detrend
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm, trange
from joblib import Parallel, delayed
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity

def parallelize(func, arg_list, n_jobs=10):
    return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in tqdm(arg_list))

############################################################
## Additional preprocessing functions 
############################################################
class lfp_preprocessing():
    """
    Class to handle proprocessing steps related to Granger Causality
    """
    def __init__(self, lfp_data):
        self.lfp_data = lfp_data

    def detrend_data(self):
        """
        Detrend data
        """
        self.detrended_data = detrend(self.lfp_data,axis=-1)

    def remove_temporal_mean(self):
        """
        Remove temporal mean from each trial
        """
        if not hasattr(self,'detrended_data'):
            self.detrend_data()
        self.mean_removed_data = self.detrended_data - \
                np.mean(self.detrended_data,axis=-1,keepdims=True)

    def divide_by_temporal_std(self):
        """
        Divide by temporal standard deviation
        """
        if not hasattr(self,'mean_removed_data'):
            self.remove_temporal_mean()
        self.std_divided_data = self.mean_removed_data / \
                np.std(self.mean_removed_data,axis=-1,keepdims=True)

    def subtract_mean_across_trials(self):
        """
        Subtract mean across trials from each trial (for each timepoint)
        """
        if not hasattr(self,'std_divided_data'):
            self.divide_by_temporal_std()
        self.mean_across_trials_subtracted_data = \
                self.std_divided_data - \
                np.mean(self.std_divided_data,axis=1,keepdims=True)

    def divide_by_std_across_trials(self):
        """
        Divide by standard deviation across trials (for each timepoint)
        """
        if not hasattr(self,'mean_across_trials_subtracted_data'):
            self.subtract_mean_across_trials()
        self.std_across_trials_divided_data = \
                self.mean_across_trials_subtracted_data / \
                np.std(self.mean_across_trials_subtracted_data,
                       axis=1,keepdims=True)

    def return_preprocessed_data(self):
        """
        Return preprocessed data
        """
        if not hasattr(self,'std_across_trials_divided_data'):
            self.divide_by_std_across_trials()
        return self.std_across_trials_divided_data

def run_adfuller_test(preprocessed_data, alpha = 0.05, wanted_fraction = 0.95):
    """
    alpha = threshold for single test (will be Bonferroni corrected internally)
    wanted_fraction = minimum fraction of tests that should be significant (stationary)
    """
    inds = list(np.ndindex(preprocessed_data.shape[:-1]))
    return_adfuller_pval = lambda this_ind: adfuller(preprocessed_data[this_ind])[1]
    pval_list = np.array(parallelize(return_adfuller_pval, inds, n_jobs=30))
    alpha = 0.05
    threshold = alpha/len(pval_list)
    wanted_fraction = 0.95
    if np.sum(pval_list < threshold) > wanted_fraction * len(pval_list):
        print('Data is stationary')
    else:
        raise ValueError('Data is not stationary')

def calc_granger(time_series,
                 sampling_frequency=1000,
                 time_window_duration=0.3,
                 time_window_step=0.05,
                 ):
    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency, # in Hz
        time_halfbandwidth_product=1,
        start_time=0, 
        time_window_duration=time_window_duration,
        time_window_step=time_window_step,
    )
    c = Connectivity.from_multitaper(m)
    granger = c.pairwise_spectral_granger_prediction()
    return granger, c

class granger_handler():
    """
    Class to handle granger causality and calculate
    significance from shuffled data
    """
    def __init__(self, 
                 good_lfp_data,
                 #preprocessed_data,
                 sampling_frequency=1000,
                 n_shuffles=500,
                 wanted_window = [1500,4000],
                 alpha = 0.05,
                 multitaper_time_window_duration=0.3,
                 multitaper_time_window_step=0.05,
                 ):
        """
        preprocessed_data = (n_channels, n_trials, n_timepoints)
        sampling_frequency = in Hz
        n_shuffles = number of shuffles to perform
        wanted_window = window to calculate granger causality in
        alpha = significance level
        multitaper_time_window_duration = duration of time window for multitaper
        multitaper_time_window_step = step of time window for multitaper
        """
        #self.preprocessed_data = preprocessed_data
        #self.input_data = preprocessed_data.T[wanted_window[0]:wanted_window[1]]
        self.good_lfp_data = good_lfp_data
        self.sampling_frequency = sampling_frequency
        self.n_shuffles = n_shuffles
        self.wanted_window = wanted_window
        self.alpha = alpha
        self.multitaper_time_window_duration = multitaper_time_window_duration
        self.multitaper_time_window_step = multitaper_time_window_step

    def preprocess_and_check_stationarity(self):
        self.preprocessed_data = \
                lfp_preprocessing(self.good_lfp_data).return_preprocessed_data()
        run_adfuller_test(self.preprocessed_data)
        self.input_data = \
                self.preprocessed_data.T[self.wanted_window[0]:self.wanted_window[1]]

    def calc_granger_actual(self):
        """
        Calculate actual granger causality
        """
        if not hasattr(self,'input_data'):
            self.preprocess_and_check_stationarity()
        self.granger_actual, self.c_actual = \
                calc_granger(self.input_data,
                            sampling_frequency=self.sampling_frequency,
                          time_window_duration=self.multitaper_time_window_duration,
                          time_window_step=self.multitaper_time_window_step,
                          )

    def calc_granger_shuffle(self):
        """
        Calculate shuffled granger causality
        """
        if not hasattr(self,'input_data'):
            self.preprocess_and_check_stationarity()
        temp_series = [np.stack([np.random.permutation(x) \
                                for x in self.input_data.T]).T \
                                        for i in trange(self.n_shuffles)]
        def temp_calc_granger(x):
            return calc_granger(x,
                            sampling_frequency=self.sampling_frequency,
                          time_window_duration=self.multitaper_time_window_duration,
                          time_window_step=self.multitaper_time_window_step,
                          )[0]
        self.shuffle_outs = np.array(
                parallelize(temp_calc_granger, temp_series, n_jobs=30))

    def calc_shuffle_threshold(self):
        if not hasattr(self,'shuffle_outs'):
            self.calc_granger_shuffle()
        self.n_comparisons = self.shuffle_outs.shape[0] * self.shuffle_outs.shape[1]
        corrected_alpha = self.alpha / self.n_comparisons
        self.wanted_percentile = 100 - (corrected_alpha * 100)
        self.percentile_granger = np.percentile(
               self.shuffle_outs, self.wanted_percentile, axis=0)

    def get_granger_sig_mask(self):
        if not hasattr(self,'percentile_granger'):
            self.calc_shuffle_threshold()
        if not hasattr(self,'granger_actual'):
            self.calc_granger_actual()
        self.masked_granger = np.ma.masked_where(
                self.granger_actual < self.percentile_granger, self.granger_actual)
        self.mask_array = np.ma.getmask(self.masked_granger)


############################################################
## Load Data 
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path,'r') as f:
    dir_list = f.read().splitlines()
dir_name = dir_list[0]

dat = ephys_data(dir_name)
# Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
#region_lfps, region_names = dat.return_region_lfps()
lfp_channel_inds, region_lfps, region_names = \
        dat.return_representative_lfp_channels()

flat_region_lfps = np.reshape(region_lfps,(region_lfps.shape[0], -1,region_lfps.shape[-1]))

############################################################
## Preprocessing 
############################################################

# 1) Remove trials with artifacts
#good_lfp_trials_bool = dat.lfp_processing.return_good_lfp_trial_inds(dat.all_lfp_array)
good_lfp_data = dat.lfp_processing.return_good_lfp_trials(flat_region_lfps)

# Plot single trials data and mean
fig, ax = plt.subplots(len(good_lfp_data), 2,
                       sharex=True, sharey='row')
for num, region_dat in enumerate(good_lfp_data):
    ax[0,num].imshow(region_dat,
                     aspect='auto', interpolation='nearest')
    mean_dat = np.mean(region_dat,axis=0)
    std_dat = np.std(region_dat,axis=0)
    ax[1,num].fill_between(np.arange(mean_dat.shape[0]),
                           mean_dat-std_dat, mean_dat+std_dat,
                           alpha=0.5)
    ax[1,num].plot(mean_dat)
    ax[0,num].set_ylabel(region_names[num])
    ax[1,num].set_title('Single Trials')
    ax[0,num].set_title('Mean')
ax[-1,0].set_xlabel('Time (ms)')
ax[-1,1].set_xlabel('Time (ms)')
plt.show()


# Further preprocessing 
preprocessed_data = lfp_preprocessing(good_lfp_data).return_preprocessed_data()

# Perform Augmented Dickey-Fuller test on each channel to check for stationarity
run_adfuller_test(preprocessed_data)

############################################################
## Compute Granger Causality 
############################################################

# Wanted shape : (n_timpoints, n_trials, n_channels)
input_data = preprocessed_data.T
sampling_frequency = 1000

wanted_window = [1500,4000]
input_data = input_data[wanted_window[0]:wanted_window[1]]

granger, c = calc_granger(input_data)

n_shuffles = 500
temp_series = [np.stack([np.random.permutation(x) for x in input_data.T]).T \
                                for i in trange(n_shuffles)]

# Calc shuffled granger
# Pass data as iterator, parallel doesn't like indexing itself
temp_calc_granger = lambda x: calc_granger(x)[0]
shuffle_outs = np.array(parallelize(temp_calc_granger, temp_series, n_jobs=30))

# Calculate given percentile across all shuffled values 
n_comparisons = shuffle_outs.shape[0] * shuffle_outs.shape[1]
alpha = 0.05
corrected_alpha = alpha / n_comparisons
wanted_percentile = 100 - (corrected_alpha * 100)
percentile_granger = np.percentile(shuffle_outs, wanted_percentile, axis=0)

# Create masked array for plotting
masked_granger = np.ma.masked_where(granger < percentile_granger, granger)
mask_array = np.ma.getmask(masked_granger)


#this_granger = granger_handler(preprocessed_data)
#this_granger.get_granger_sig_mask()
#granger = this_granger.granger_actual
#c = this_granger.c_actual
#masked_granger = this_granger.masked_granger
#mask_array = this_granger.mask_array
#wanted_window = this_granger.wanted_window


alpha_mask = np.empty(mask_array.shape)
alpha_mask[~mask_array] = 1
alpha_mask[mask_array] = 0.3

# Plot values of granger
# Show values less than shuffle in red
t_vec = np.linspace(wanted_window[0], wanted_window[1], masked_granger.shape[0])
stim_t = 2000
t_vec = t_vec - stim_t

wanted_freqs = [0,60]
freq_inds = np.where((c.frequencies > wanted_freqs[0]) & \
        (c.frequencies < wanted_freqs[1]))[0]

cmap = plt.cm.viridis
cmap.set_bad(color='red')

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
#im = ax[0].pcolormesh(
im = ax[0].contourf(
        #t_vec, c.frequencies[freq_inds], masked_granger[:, freq_inds, 0, 1].T,
        t_vec, c.frequencies[freq_inds], granger[:, freq_inds, 0, 1].T,
    cmap=cmap,
    levels = 15,
    #shading="gouraud",
    #vmin=vmin, vmax=vmax
)
#im = ax[0].imshow(
#        #t_vec, c.frequencies[freq_inds], masked_granger[:, freq_inds, 0, 1].T,
#    granger[:, freq_inds, 0, 1].T,
#    cmap='viridis', aspect="auto",
#    origin = 'lower', alpha= alpha_mask[:, freq_inds, 0,1].T,
#    #vmin=vmin, vmax=vmax
#)
ax[0].set_title("x1 -> x2")
ax[0].set_ylabel("Frequency")
ax[0].axvline(0, color='red', linestyle='--')
plt.colorbar(im, ax = ax[0])
#im = ax[1].pcolormesh(
im = ax[1].contourf(
        #t_vec, c.frequencies[freq_inds], masked_granger[:, freq_inds, 1, 0].T,
        t_vec, c.frequencies[freq_inds], granger[:, freq_inds, 1, 0].T,
    cmap=cmap, 
    levels = 15,
    #shading="auto",
    #vmin=vmin, vmax=vmax
)
#im = ax[1].imshow(
#        #t_vec, c.frequencies[freq_inds], masked_granger[:, freq_inds, 1, 0].T,
#        granger[:, freq_inds, 1, 0].T,
#    cmap='viridis', aspect="auto",
#    origin = 'lower', alpha= alpha_mask[:, freq_inds, 1,0].T,
#    #vmin=vmin, vmax=vmax
#)
ax[1].set_title("x2 -> x1")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Frequency")
plt.colorbar(im, ax = ax[1])
ax[1].axvline(0, color='red', linestyle='--')
plt.show()

## Plot values of granger summed across all frequencies
#sum_granger = np.nansum(granger, axis=1)
#sum_shuffle_granger = np.nansum(shuffle_outs, axis=2)
#percentile_thresh = 95
#percentile_sum_shuffle = np.percentile(sum_shuffle_granger, percentile_thresh, axis=0)
#fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
#inds = list(np.ndindex(ax.shape))
#for num, this_ind in enumerate(inds):
#    ax[this_ind].plot(t_vec, sum_granger.T[this_ind])
#    ax[this_ind].plot(t_vec, percentile_sum_shuffle.T[this_ind], color = 'red')
#    #ax[this_ind].set_title(region_names[num])
#    ax[this_ind].set_ylabel('Trial')
#    ax[this_ind].set_xlabel('Trial')
#plt.show()
