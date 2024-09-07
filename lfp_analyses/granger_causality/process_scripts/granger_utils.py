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
#
# References:
# Ding, Mingzhou, et al. “Short-Window Spectral Analysis of Cortical Event-Related Potentials by Adaptive Multivariate Autoregressive Modeling: Data Preprocessing, Model Validation, and Variability Assessment.” Biological Cybernetics, vol. 83, no. 1, June 2000, pp. 35–45, https://doi.org/10.1007/s004229900137.


############################################################
# Imports
############################################################

import numpy as np
from scipy.signal import detrend
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm, trange
from joblib import Parallel, delayed
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity
from scipy.stats import percentileofscore 


def parallelize(func, arg_list, n_jobs=10):
    return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in tqdm(arg_list))

############################################################
# Additional preprocessing functions
############################################################


class lfp_preprocessing():
    """
    Class to handle proprocessing steps related to Granger Causality
    """

    def __init__(self, lfp_data):
        """
        Initialize class with lfp data
        lfp_data : (n_channels, n_trials, n_timepoints)
        """
        self.lfp_data = lfp_data

    def detrend_data(self):
        """
        Detrend data
        """
        self.detrended_data = detrend(self.lfp_data, axis=-1)

    def remove_temporal_mean(self):
        """
        Remove temporal mean from each trial
        """
        if not hasattr(self, 'detrended_data'):
            self.detrend_data()
        self.mean_removed_data = self.detrended_data - \
            np.mean(self.detrended_data, axis=-1, keepdims=True)

    def divide_by_temporal_std(self):
        """
        Divide by temporal standard deviation
        """
        if not hasattr(self, 'mean_removed_data'):
            self.remove_temporal_mean()
        self.std_divided_data = self.mean_removed_data / \
            np.std(self.mean_removed_data, axis=-1, keepdims=True)

    def subtract_mean_across_trials(self):
        """
        Subtract mean across trials from each trial (for each timepoint)
        """
        if not hasattr(self, 'std_divided_data'):
            self.divide_by_temporal_std()
        self.mean_across_trials_subtracted_data = \
            self.std_divided_data - \
            np.mean(self.std_divided_data, axis=1, keepdims=True)

    def divide_by_std_across_trials(self):
        """
        Divide by standard deviation across trials (for each timepoint)
        """
        if not hasattr(self, 'mean_across_trials_subtracted_data'):
            self.subtract_mean_across_trials()
        self.std_across_trials_divided_data = \
            self.mean_across_trials_subtracted_data / \
            np.std(self.mean_across_trials_subtracted_data,
                   axis=1, keepdims=True)

    def return_preprocessed_data(self):
        """
        Return preprocessed data
        """
        if not hasattr(self, 'std_across_trials_divided_data'):
            self.divide_by_std_across_trials()
        return self.std_across_trials_divided_data


def run_adfuller_test(
        preprocessed_data, 
        alpha=0.05, 
        wanted_fraction=0.95,
        warning_only=False
        ):
    """
    alpha = threshold for single test (will be Bonferroni corrected internally)
    wanted_fraction = minimum fraction of tests that should be significant (stationary)
    warning_only = if True, will only print a warning if data is not stationary
    """
    inds = list(np.ndindex(preprocessed_data.shape[:-1]))

    def return_adfuller_pval(this_ind): return adfuller(
        preprocessed_data[this_ind])[1]
    pval_list = np.array(parallelize(return_adfuller_pval, inds, n_jobs=30))
    threshold = alpha/len(pval_list)
    if np.sum(pval_list < threshold) > wanted_fraction * len(pval_list):
        print('Data is stationary')
    else:
        if not warning_only:
            raise ValueError('Data is not stationary')
        else:
            print('Warning: Data is NOT stationary')


def calc_granger(time_series,
                 time_halfbandwidth_product=1,
                 sampling_frequency=1000,
                 time_window_duration=0.3,
                 time_window_step=0.05,
                 ):
    """
    Calculate granger causality
    time_series = time x trials x channels
    """
    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,  # in Hz
        time_halfbandwidth_product=time_halfbandwidth_product,
        start_time=0,
        time_window_duration=time_window_duration, # in seconds
        time_window_step=time_window_step, # in seconds
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
                 # preprocessed_data,
                 sampling_frequency=1000,
                 n_shuffles=500,
                 wanted_window=[1500, 4000],
                 alpha=0.05,
                 multitaper_time_halfbandwidth_product=1,
                 multitaper_time_window_duration=0.3,
                 multitaper_time_window_step=0.05,
                 preprocess=True,
                 warning_only=False
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
        self.preprocess_flag = preprocess
        self.good_lfp_data = good_lfp_data
        self.sampling_frequency = sampling_frequency
        self.n_shuffles = n_shuffles
        self.wanted_window = wanted_window
        self.alpha = alpha
        self.multitaper_time_halfbandwidth_product = \
                multitaper_time_halfbandwidth_product
        self.multitaper_time_window_duration = multitaper_time_window_duration
        self.multitaper_time_window_step = multitaper_time_window_step
        self.warning_only = warning_only

    def calc_granger(self, x):
        granger, c = calc_granger(
            x,
            sampling_frequency=self.sampling_frequency,
            time_halfbandwidth_product=self.multitaper_time_halfbandwidth_product,
            time_window_duration=self.multitaper_time_window_duration,
            time_window_step=self.multitaper_time_window_step,
                )
        return granger, c

    def preprocess_data(self):
        if self.preprocess_flag:
            self.preprocessed_data = \
                lfp_preprocessing(self.good_lfp_data).return_preprocessed_data()
        else:
            self.preprocessed_data = self.good_lfp_data

    def check_stationarity(self):
        if not hasattr(self, 'preprocessed_data'):
            self.preprocess_data()
        run_adfuller_test(self.preprocessed_data, warning_only = self.warning_only)

    def preprocess_and_check_stationarity(self):
        self.preprocess_data()
        self.check_stationarity()
        # Transform data to (n_timepoints, n_trials, n_channels)
        self.input_data = \
            self.preprocessed_data.T[self.wanted_window[0]:self.wanted_window[1]]

    def calc_granger_actual(self):
        """
        Calculate bootstrapped actual granger causality
        to allow for estimation of error
        """
        if not hasattr(self, 'input_data'):
            self.preprocess_and_check_stationarity()
        # input_data shape = (n_timepoints, n_trials, n_channels)
        # Calculate as many bootstrapped samples as n_shuffles
        trial_inds = np.random.randint(
                0, self.input_data.shape[1],
                (self.n_shuffles, self.input_data.shape[1]))
        temp_dat = [self.input_data[:, trial_inds[i]]
                    for i in trange(self.n_shuffles)]
        outs_temp = parallelize(self.calc_granger, temp_dat, n_jobs=30)
        time_vec = outs_temp[0][1].time
        freq_vec = outs_temp[0][1].frequencies
        outs_temp = [x[0] for x in outs_temp]
        self.granger_actual = np.array(outs_temp)
        self.time_vec = time_vec
        self.freq_vec = freq_vec
        #self.granger_actual, self.c_actual = \
        #    self.calc_granger(self.input_data)

    def calc_granger_shuffle(self):
        """
        Calculate shuffled granger causality
        """
        if not hasattr(self, 'input_data'):
            self.preprocess_and_check_stationarity()
        temp_series = [np.stack([np.random.permutation(x)
                                for x in self.input_data.T]).T
                       for i in trange(self.n_shuffles)]

        outs_temp = parallelize(self.calc_granger, temp_series, n_jobs=30)
        outs_temp = [x[0] for x in outs_temp]
        self.shuffle_outs = np.array(outs_temp)

    def calc_shuffle_threshold(self):
        if not hasattr(self, 'shuffle_outs'):
            self.calc_granger_shuffle()
        self.n_comparisons = self.shuffle_outs.shape[0] * \
            self.shuffle_outs.shape[1]
        corrected_alpha = self.alpha / self.n_comparisons
        self.wanted_percentile = 100 - (corrected_alpha * 100)
        self.percentile_granger = np.percentile(
            self.shuffle_outs, self.wanted_percentile, axis=0)

    def get_granger_sig_mask(self):
        """
        Mask is True when granger causality is NOT SIGNIFICANT
        """
        if not hasattr(self, 'percentile_granger'):
            self.calc_shuffle_threshold()
        if not hasattr(self, 'granger_actual'):
            self.calc_granger_actual()
        mean_granger_actual = np.nanmean(self.granger_actual, axis=0)
        self.masked_granger = np.ma.masked_where(
            mean_granger_actual < self.percentile_granger, mean_granger_actual)
        self.mask_array = np.ma.getmask(self.masked_granger)

    def get_granger_summed_sig(self):
        """
        Return summed significance across all frequencies
        """
        mean_granger_actual = np.nanmean(self.granger_actual, axis= (0,2))
        mean_granger_shuffle = np.nanmean(self.shuffle_outs, axis= (2))
        n_comparisons = mean_granger_actual.shape[0] 
        corrected_alpha = self.alpha / n_comparisons

        data_inds = [
                tuple([0,1]),
                tuple([1,0])
                ]

        # Calculate p-values for each direction
        p_val_list = []
        sig_list = []
        freq_summed_actual_list = []
        freq_summed_shuffle_list = []
        for data_ind in data_inds:
            this_actual = mean_granger_actual[..., data_ind[0], data_ind[1]]
            this_shuffle = mean_granger_shuffle[..., data_ind[0], data_ind[1]]
            this_percentile = np.array(
                    [
                        percentileofscore(this_shuffle[x], this_actual[x]) \
                                for x in range(this_actual.shape[0])
                                ]
                    )
            this_p_val = 1 - this_percentile / 100
            this_sig = this_p_val < corrected_alpha
            p_val_list.append(this_p_val)
            sig_list.append(this_sig)
            freq_summed_actual_list.append(this_actual)
            freq_summed_shuffle_list.append(this_shuffle)

        self.freq_summed_pvals = p_val_list
        self.freq_summed_sig = sig_list
        self.freq_summed_actual_list = freq_summed_actual_list
        self.freq_summed_shuffle_list = freq_summed_shuffle_list

            # mean_shuffle = np.nanmean(this_shuffle, axis=0)
            # std_shuffle = np.nanstd(this_shuffle, axis=0)

            # stim_time_vec = self.time_vec - 0.5
            # plt.plot(stim_time_vec, mean_shuffle, label='shuffle')
            # plt.fill_between(
            #         stim_time_vec,
            #         mean_shuffle - std_shuffle,
            #         mean_shuffle + std_shuffle,
            #         alpha=0.5
            #         )
            # plt.plot(stim_time_vec, this_actual, label='actual')
            # plt.legend()
            # plt.show()


    #def calc_granger_single_trial(self):
    #    """
    #    Calculate granger causality for single trials
    #    """
    #    if not hasattr(self, 'input_data'):
    #        self.preprocess_and_check_stationarity()

    #    single_trial_dat = self.input_data.copy()
    #    single_trial_dat = np.moveaxis(single_trial_dat, 0, 1)
    #    single_trial_dat = single_trial_dat[:, :, np.newaxis, :]

    #    outs = parallelize(self.calc_granger, single_trial_dat, n_jobs=30)
    #    granger_single_trial, c_single_trial = zip(*outs)
    #    self.granger_single_trial = np.array(granger_single_trial)
    #    self.c_single_trial = np.array(c_single_trial)

    #def calc_granger_shuffle_single_trial(self):
    #    """
    #    Calculate shuffled granger causality for single trials
    #    Will have to make an "AVERAGE" shuffle since we can't shuffle
    #    trials for a single metric. Instead, we'll make a dataset
    #    with mismatched trials
    #    """
    #    if not hasattr(self, 'input_data'):
    #        self.preprocess_and_check_stationarity()
    #    temp_series = [np.stack([np.random.permutation(x)
    #                            for x in self.input_data.T]).T
    #                   for i in trange(self.n_shuffles)]

    #    outs_temp = parallelize(self.calc_granger, temp_series, n_jobs=30)
    #    outs_temp = [x[0] for x in outs_temp]
    #    self.shuffle_outs = np.array(outs_temp)

    #def calc_shuffle_threshold_single_trial(self):
    #    if not hasattr(self, 'shuffle_outs'):
    #        self.calc_granger_shuffle()
    #    self.n_comparisons = self.shuffle_outs.shape[0] * \
    #        self.shuffle_outs.shape[1]
    #    corrected_alpha = self.alpha / self.n_comparisons
    #    self.wanted_percentile = 100 - (corrected_alpha * 100)
    #    self.percentile_granger = np.percentile(
    #        self.shuffle_outs, self.wanted_percentile, axis=0)

    #def get_granger_sig_mask_single_trial(self):
    #    if not hasattr(self, 'percentile_granger'):
    #        self.calc_shuffle_threshold()
    #    if not hasattr(self, 'granger_actual'):
    #        self.calc_granger_actual()
    #    self.masked_granger = np.ma.masked_where(
    #        self.granger_actual < self.percentile_granger, self.granger_actual)
    #    self.mask_array = np.ma.getmask(self.masked_granger)

