
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
import itertools
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.decomposition import PCA as pca

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

##################################################
## Define functions
##################################################

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    #mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    #array -= mean_baseline
    return array

def firing_overview(data, t_vec = None, y_values_vec = None,
                    interpolation = 'nearest',
                    cmap = 'jet',
                    min_val = None, max_val=None, 
                    subplot_labels = None):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron
    """
    if data.shape[-1] != len(t_vec) or t_vec is None:
        raise Exception('Time dimension in data needs to be'\
            'equal to length of t_vec')
    num_nrns = data.shape[0]

    if min_val is None:
        min_val = [np.min(x,axis=None) for x in data]
    if max_val is None:
        max_val = [np.max(x,axis=None) for x in data]
    if t_vec is None:
        t_vec = np.arange(data.shape[-1])
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])
    if subplot_labels is None:
        subplot_labels = np.zeros(num_nrns)

    # Plot firing rates
    square_len = np.int(np.ceil(np.sqrt(num_nrns)))
    fig, ax = plt.subplots(square_len,square_len, sharex='all',sharey='all')
    
    nd_idx_objs = []
    for dim in range(ax.ndim):
        this_shape = np.ones(len(ax.shape))
        this_shape[dim] = ax.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to( 
                    np.reshape(
                        np.arange(ax.shape[dim]),
                        this_shape.astype('int')), ax.shape).flatten())
    
    for nrn in range(num_nrns):
        plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
        plt.gca().set_title('{}:{}'.format(int(subplot_labels[nrn]),nrn))
        plt.gca().pcolormesh(t_vec, y_values_vec,
                data[nrn,:,:],cmap=cmap,
                vmin = min_val[nrn], vmax = max_val[nrn])
    return ax

def kl_divergence(vec_a,vec_b):
    """
    Both vectors are 1D arrays
    Vectors will be renormalized
    Order of divergence is D(A||B)
    """
    dat1 = vec_a/np.sum(vec_a)
    dat2 = vec_b/np.sum(vec_b)
    kl_div = np.sum(dat1*np.log(dat1/dat2))
    return kl_div

# Calculate pairwise comparisons of tastes, on single time-bins using KL Divergence
def calc_taste_discrim(this_nrn, taste_labels, symbols = 5, time_bin_count = 40):
    """
    this_nrn : (trials,time)
    """
    # Chop firing rates in quartile and time bins for each taste
    quartiles = np.linspace(0,100,symbols+1)
    quart_vals = np.percentile(this_nrn.flatten(),quartiles)
    time_bins = list(map(int,np.floor(np.linspace(0,this_nrn.shape[-1],time_bin_count+1))))

    cpd = np.empty((len(np.unique(taste_labels)),symbols,time_bin_count))
    for taste in np.sort(np.unique(taste_labels)):
        for time_bin_num in range(1,len(time_bins)):
            cpd[taste,:,time_bin_num-1] = \
                    np.histogram(this_nrn[taste_labels == taste,\
                        time_bins[time_bin_num-1]:time_bins[time_bin_num]],bins = quart_vals)[0]

    # Add some noise to CPD to avoid zero errors
    cpd += np.random.random(cpd.shape)*1e-9

    # Normalize CPD within each bin
    norm_cpd = cpd / np.sum(cpd,axis=(1))[:,np.newaxis]

    # List all possible combinations of pairwise taste comparisons
    taste_comparisons = list(
            itertools.combinations(
                range(len(np.unique(taste_labels))),2))

    kld_array = np.empty((len(taste_comparisons),time_bin_count))
    for pair_num, pair in enumerate(taste_comparisons):
        for time_bin in range(cpd.shape[-1]):
            kld_array[pair_num,time_bin] = \
                kl_divergence(cpd[pair[0],:,time_bin], cpd[pair[1],:,time_bin])

    return kld_array

# Wrapper function to generate both shuffles
def shuffle_taste_discrim(this_nrn, taste_labels, shuffle_count = 100,
        symbols = 5, time_bin_count = 40): 
    """
    this_nrn = (trials x time) # All trials concatenated
    """
    random_gen = [np.random.permutation(this_nrn) for repeat in range(shuffle_count)]
    random_klds = np.array([calc_taste_discrim(this_shuffle,taste_labels,
                                        symbols = symbols, time_bin_count = time_bin_count) \
            for this_shuffle in random_gen])
    return random_klds

##################################################
## Read in data 
##################################################


# Define middle channels in board
middle_channels = np.arange(8,24)

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Pull out all terminal groups (leafs) under stft
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    freq_vec = hf5.get_node('/stft','freq_vec')[:]
    time_vec = hf5.get_node('/stft','time_vec')[:]

    amplitude_node_list = [x for x in hf5.root.stft._f_walknodes() \
            if 'amplitude_array' in x.__str__()]
    print('Extracting phase info')
    amplitude_array_list = [x[:] for x in tqdm(amplitude_node_list)]
    # Extract all nodes with phase array
    node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) for x in amplitude_node_list]
    # Pull channels selected for coherence analyses
    relative_channel_nums = [hf5.get_node(this_path,'relative_region_channel_nums')[:] \
            for this_path in node_path_list]
    # Pull parsed_lfp_channel from each array
    parsed_channel_list  = [hf5.get_node(path,'parsed_lfp_channels')[:] for path in node_path_list]

for this_node_num in tqdm(range(len(phase_node_list))):
    parsed_channels = parsed_channel_list[this_node_num]
    middle_channels_bool = np.array([True if channel in middle_channels else False \
            for channel in parsed_channels])
    relative_channel_num_split = \
            [np.arange(len(parsed_channels))[middle_channels_bool], 
                    np.arange(len(parsed_channels))[~middle_channels_bool]]
    amplitude_array = amplitude_array_list[this_node_num][:]
    amplitude_array_split = [amplitude_array[:,x] for x in relative_channel_num_split]
    mean_channel_amplitude_array = np.array([np.mean(x,axis=1) for x in amplitude_array_split])
    mean_channel_amplitude_array_long = np.reshape(mean_channel_amplitude_array,
            (-1,np.prod(mean_channel_amplitude_array.shape[1:3]),
                *mean_channel_amplitude_array.shape[3:]))
    # zscore array along trials for every timepoint
    zscore_amplitude_array_long = np.array([[zscore(freq,axis=1) for freq in region]\
            for region in mean_channel_amplitude_array_long.swapaxes(1,2)])

    # Find principal components of z-scored amplitude
    mean_channel_amplitude_very_long = np.reshape(mean_channel_amplitude_array_long.swapaxes(1,2),
            (*np.array(mean_channel_amplitude_array_long.shape)[[0,2]],-1))
    mean_zscore_ampltiude_very_long = zscore(mean_channel_amplitude_very_long,axis=-1)
    # Remove timepoints with large values
    relevant_times = np.where(mean_zscore_ampltiude_very_long < 4)
    # Find intersection of times across both regions
    relevant_times_per_region = [np.unique(relevant_times[-1][relevant_times[0] == region])\
            for region in np.sort(np.unique(relevant_times[0]))]
    fin_relevant_times = np.intersect1d(*relevant_times_per_region)
    mean_zscore_ampltiude_very_long = \
            mean_zscore_ampltiude_very_long\
            [...,]

    pca_object = pca(n_components = 10).fit(mean_zscore_ampltiude_very_long[0].T)
    explained_variance_threshold = 0.8
    needed_components = np.sum(
                            np.cumsum(
                            pca_object.explained_variance_ratio_) < explained_variance_threshold)+1
    pca_object = pca(n_components = needed_components)\
            .fit(mean_zscore_ampltiude_very_long[0].T)

    #this_amplitude_array = amplitude_array_list[this_node_num]\
    #        [:,relative_channel_nums[this_node_num]]
    #this_mean_channel_amplitude = np.mean(this_amplitude_array,axis=2

        
    #zscore_amplitude_array_long = zscore(mean_channel_amplitude_array_long,axis=1)
    stim_time = 2
    max_times = np.argmax(zscore_amplitude_array_long[...,time_vec>stim_time], axis=-1)
    trials_per_taste = mean_channel_amplitude_array.shape[2]
    trial_order = np.zeros(max_times.shape)
    for region_num,region in enumerate(max_times):
        for freq_num,freq in enumerate(region):
            for taste in range(mean_channel_amplitude_array.shape[1]):
                trial_order[region_num,freq_num,
                        (taste*trials_per_taste):((taste+1)*trials_per_taste)] = \
                            taste*trials_per_taste + \
                            np.argsort(freq[(taste*trials_per_taste):((taste+1)*trials_per_taste)])

    sorted_zscore_amplitude = np.array(
            [[freq[(len(freq_order) - 1) - freq_order] \
                    for freq,freq_order in zip(region,region_order)]\
                    for region,region_order in \
                    zip(zscore_amplitude_array_long,trial_order.astype(int))])

    # For each region and frequency band, calculate pairwise KL-Divergence between tastes

    # For each region and frequency band, calculate pairwise KL-Divergence between tastes
    taste_labels = np.sort([0,1,2,3]*30)
    taste_discrim_array = np.array([[\
            calc_taste_discrim(
                                freq,
                                taste_labels,
                                symbols = 10,
                                time_bin_count = 70) \
            for freq in session]\
            for session in mean_channel_amplitude_array_long.swapaxes(1,2)]) 

    shuffle_taste_discrim_array = np.array([
            Parallel(n_jobs = mp.cpu_count()-2)\
            (delayed(shuffle_taste_discrim)\
                                        (freq,
                                        taste_labels,
                                        symbols = 10,
                                        time_bin_count = 70) \
            for freq in tqdm(session))\
            for session in mean_channel_amplitude_array_long.swapaxes(1,2)]) 
    shuffle_taste_discrim_array = np.rollaxis(shuffle_taste_discrim_array,2,5)

    # Calculate the probability of each time point belonging to it's respective
    # shuffled distribution
    # Generate ECDFs
    iters = np.ndindex(taste_discrim_array.shape) 
    discrim_p_val_array = np.zeros(taste_discrim_array.shape)
    for this_iter in tqdm(iters):
        discrim_p_val_array[this_iter] = \
                ECDF(shuffle_taste_discrim_array[this_iter])(taste_discrim_array[this_iter])
    # Convert ECDF values into symmetric distance from the middle
    # fin_p_val = 0.5 - np.abs( x - 0.5)
    fin_discrim_p_val_array = 0.5 - np.abs( discrim_p_val_array - 0.5)
    # Convert p_vals to categories for visualization
    p_val_bin_edges = [0,0.01,0.05,0.1,1]
    discrim_p_val_strata = np.zeros(taste_discrim_array.shape)
    comparison_func = lambda low, high, x : low < x < high
    v_comparison_func = np.vectorize(comparison_func)
    for bin_num in range(len(p_val_bin_edges)-1):
        discrim_p_val_strata[\
                v_comparison_func(
                    p_val_bin_edges[bin_num],
                    p_val_bin_edges[bin_num+1],
                    fin_discrim_p_val_array)] = bin_num
    


    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    #                       

    # Plot 1 + 2 
    # Power in each band over all trials for both regions separately
    # Sort trials by time of max power per taste, per region
    # Cycle through max times for every band and region and sort trials within tastes
    firing_overview(
                sorted_zscore_amplitude[0],
                t_vec = time_vec,subplot_labels = freq_vec)
    fig = plt.gcf()
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'RG0_Freq_power_trials'))

    firing_overview(
                sorted_zscore_amplitude[1],
                t_vec = time_vec,subplot_labels = freq_vec)
    fig = plt.gcf()
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'RG1_Freq_power_trials'))

    # Plot 3
    # a-d) Mean spectrogram for each taste
    # e) Mean spectrogram across all tastes
    # Pre-determine color limits
    mean_channel_trial_amplitude_array = np.mean(mean_channel_amplitude_array,axis=2)
    min_val,max_val = mean_channel_trial_amplitude_array.min(),\
                        mean_channel_trial_amplitude_array.max()
    fig, ax = plt.subplots(5,2,sharex='all',sharey='all')
    for region_num, region in enumerate(mean_channel_trial_amplitude_array):
        for taste_num,taste in enumerate(region):
            ax[taste_num,region_num].pcolormesh(
                    #time_vec,freq_vec,normalize_timeseries(taste,time_vec,2),
                    time_vec,freq_vec,taste,
                    cmap = 'jet', vmin= min_val, vmax = max_val)
    ax[-1,0].pcolormesh(time_vec,freq_vec,
        #normalize_timeseries(np.mean(mean_channel_trial_amplitude_array,axis=(1))[0],time_vec,2),
            np.mean(mean_channel_trial_amplitude_array,axis=(1))[0],
            cmap = 'jet', vmin= min_val, vmax = max_val)
    ax[-1,1].pcolormesh(time_vec,freq_vec,
        #normalize_timeseries(np.mean(mean_channel_trial_amplitude_array,axis=(1))[1],time_vec,2),
            np.mean(mean_channel_trial_amplitude_array,axis=(1))[1],
            cmap = 'jet', vmin= min_val, vmax = max_val)
    ax[-1,0].set_title('Average spectrum')
    ax[-1,1].set_title('Average spectrum')
    ax[0,0].set_title('Region 0')
    ax[0,1].set_title('Region 1')
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'Average_Spectra'))

    # Plot 4 + 5
    # Significance of taste differences in LFP by frequency 
    time_bins = np.array(list(map(int,np.floor(np.linspace(0,time_vec.shape[-1],70)))))
    firing_overview(
                discrim_p_val_strata[0],
                t_vec = time_bins,subplot_labels = freq_vec, cmap='viridis')
    fig = plt.gcf()
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'RG0_significant_taste_comparisons'))

    firing_overview(
                discrim_p_val_strata[1],
                t_vec = time_bins,subplot_labels = freq_vec)
    fig = plt.gcf()
    fig.set_size_inches(8,10)
    fig.suptitle("_".join(animal_date_list))
    fig.savefig(os.path.join(this_plot_dir,'RG1_significant_taste_comparisons'))

