## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed, cpu_count
import multiprocessing as mp
import shutil
from sklearn.utils import resample
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize
from scipy.stats import zscore
from scipy.stats import mannwhitneyu, ttest_ind, ttest_1samp

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    #mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    #array -= mean_baseline
    return array


##################################################
## Read in data 
##################################################

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Pull out all terminal groups (leafs) under stft
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    freq_vec = hf5.get_node('/stft','freq_vec')[:]
    time_vec = hf5.get_node('/stft','time_vec')[:]

    coherence_node_path_list = \
        [x.__str__().split(" ")[0] for x in hf5.root.stft._f_walknodes() \
        if 'mean_coherence_array' in x.__str__()]
    coherence_node_path_list.sort()

    #coherence_array_list = [x[:] for x in tqdm(coherence_node_list)]
    # Extract all nodes with coherence array
    node_path_list = [os.path.dirname(x) for x in coherence_node_path_list]

    # Ask user to select files to perform anaysis on
    selected_files = easygui.multchoicebox(\
            msg = 'Please select files to run analysis on',
            choices = ['{}) '.format(num)+x[6:] \
                    for num,x in enumerate(node_path_list)])
    selected_file_inds = [int(x.split(')')[0]) for x in selected_files]

    coherence_node_path_list = [coherence_node_path_list[i] \
                    for i in selected_file_inds]
    node_path_list = [node_path_list[i] for i in selected_file_inds]

    # Pull parsed_lfp_channel from each array
    parsed_channel_list  = \
            [hf5.get_node(path,'parsed_lfp_channels')[:] \
            for path in node_path_list]

# Define variables to be maintained across files
initial_dir = dir() + ['initial_dir']

# _____         _   
#|_   _|__  ___| |_ 
#  | |/ _ \/ __| __|
#  | |  __/\__ \ |_ 
#  |_|\___||___/\__|
#                   

# 2 Tests:
# 1) Test on change in mean value of coherence across a single trial
#   - Will need to calculate error on coherence by resampling trial pairs
# 2) Test on change in variance? (Needed?)

for this_node_num in tqdm(range(len(phase_node_path_list))):
    with tables.open_file(data_hdf5_path,'r') as hf5:
        region_phase_channels = hf5.get_node(node_path_list[this_node_num],
                                'region_phase_channels')[:]
        phase_diff = hf5.get_node(node_path_list[this_node_num],
                                'phase_difference_array')[:]


    #phase_diff_reshape = np.angle(
    #        np.reshape(phase_diff, 
    #        (np.prod(phase_diff.shape[:2]),*phase_diff.shape[2:])))
    #phase_bin_nums = 30
    #phase_bins = np.linspace(-np.pi, np.pi, phase_bin_nums)
    #phase_diff_hists = np.array([[np.histogram(freq,phase_bins)[0] \
    #        for freq in time_bin] \
    #        for time_bin in phase_diff_reshape.T]).swapaxes(0,1) 

    # Estimate error in coherence calculation by bootstrapping
    bootstrap_samples = 500
    coherence_boot_array = np.zeros(\
            (bootstrap_samples,*phase_diff.shape[2:])) 

    for repeat in trange(bootstrap_samples):
        this_phase_diff = phase_diff[:,\
        np.random.choice(range(phase_diff.shape[1]),
                                phase_diff.shape[1], replace = True)]
        coherence_boot_array[repeat] = \
                np.abs(np.mean(this_phase_diff,axis=(0,1)))




    # Test plot : Plot mean coherence with std for all bands
    fig, ax = visualize.gen_square_subplots(coherence_boot_array.shape[1])
            #sharey=True,sharex=True)
    for ax_num, this_ax in enumerate(ax.flatten()[:coherence_boot_array.shape[1]]):
        this_coherence = coherence_boot_array[:,ax_num]
        mean_val = np.mean(this_coherence,axis=0)
        std_val = np.std(this_coherence,axis=0)
        this_ax.plot(mean_val)
        this_ax.fill_between(x = np.arange(coherence_boot_array.shape[-1]),
                y1 = mean_val - 2*std_val,
                y2 = mean_val + 2*std_val, alpha = 0.5)
    plt.show()


    ####################################### 
    # Difference from baseline
    ####################################### 
    # Pool baseline coherence and conduct tests on non-overlapping bins
    t = np.arange(coherence_boot_array.shape[-1])
    baseline_t = 2000 #ms
    baseline_range = (1250,1750)
    baseline_inds = np.where((t>baseline_range[0])*(t<baseline_range[1]))[0]
    bin_size = 100 #ms
    bin_starts = np.arange(0, max(t) , bin_size)  
    bin_inds = [(x,x+bin_size) for x in bin_starts]
    t_binned = np.arange(0,max(t),bin_size)

    #mean_coherence_array = np.mean(coherence_boot_array,axis=0)
    baseline_sub_coherence = coherence_boot_array - \
        np.mean(coherence_boot_array[..., t < baseline_t],axis=-1)\
        [...,np.newaxis]

    #baseline_sampling_dist = np.array([\
    #        np.mean(\
    #            resample(baseline_sub_coherence[:,baseline_inds].T, 
    #            n_samples = bin_size, replace = True),
    #        axis = 0) \
    #                for repeat in trange(bootstrap_samples)])

    ci_interval = 0.95
    lower_bound, higher_bound = \
            np.percentile(baseline_sub_coherence[...,baseline_inds],
                    100*(1-ci_interval)/2, axis=(0,-1)),\
            np.percentile(baseline_sub_coherence[...,baseline_inds],
                    100*(1+ci_interval)/2, axis=(0,-1))

    # Find bootstrapped deviation from mean baseline for baseline and
    # use to test differences
    # 1) Generate sampling distribution of baseline mean from baseline_sub array
    #       with count per sample equal to bin size
    # 2) Find p-value of mean-bin values using sampling distribution

    #baseline_boot_array = coherence_boot_array[..., baseline_inds]\
    #        .reshape((coherence_boot_array.shape[1],-1))

    #ci_interval = 0.95
    #lower_bound, higher_bound = \
    #        np.percentile(baseline_boot_array,100*(1-ci_interval)/2, axis=-1),\
    #        np.percentile(baseline_boot_array,100*(1+ci_interval)/2, axis=-1)

    #coherence_mean_bins = np.mean(\
    #        np.reshape(coherence_boot_array,
    #        (*coherence_boot_array.shape[:2],-1,bin_size)),
    #        axis = (0,-1)).T.swapaxes(0,1)

    freq_label_list = ["{}-{}".format(int(freq), int(freq+2)) \
            for freq in freq_vec]

    # Test plot : Plot mean coherence with std for all bands
    fig, ax = visualize.gen_square_subplots(baseline_sub_coherence.shape[1])
    for ax_num, this_ax in enumerate(ax.flatten()\
            [:baseline_sub_coherence.shape[1]]):
        this_coherence = baseline_sub_coherence[:,ax_num]
        mean_val = np.mean(this_coherence,axis=0)
        std_val = np.std(this_coherence,axis=0)
        this_ax.plot(mean_val)
        this_ax.fill_between(x = np.arange(this_coherence.shape[-1]),
                y1 = mean_val - 2*std_val,
                y2 = mean_val + 2*std_val, alpha = 0.5)
        this_ax.hlines((lower_bound[ax_num],higher_bound[ax_num]),
                0, baseline_sub_coherence.shape[-1], color = 'r')
        this_ax.set_title(freq_label_list[ax_num])
    plt.suptitle('Baseline 95% CI\n'\
            + "_".join(node_path_list[this_node_num].split('/')[-2:]))
    plt.show()

    #fig, ax = visualize.gen_square_subplots(coherence_mean_bins.shape[0])
    #for ax_num, this_ax in enumerate(ax.flatten()\
    #        [:coherence_mean_bins.shape[0]]):
    #    this_coherence = coherence_mean_bins[ax_num]
    #    mean_val = np.mean(this_coherence,axis=-1)
    #    std_val = np.std(this_coherence,axis=-1)
    #    this_ax.plot(mean_val)
    #    this_ax.fill_between(x = np.arange(this_coherence.shape[0]),
    #            y1 = mean_val - 2*std_val,
    #            y2 = mean_val + 2*std_val, alpha = 0.5)
    #plt.show()

    #from scipy.stats import percentileofscore
    bin_num = 1000
    percentile_mat = np.zeros((baseline_sub_coherence.shape[1],
                        bin_num))
    score_mat = np.zeros((baseline_sub_coherence.shape[1],
                        bin_num))

    for band_num in trange(baseline_sub_coherence.shape[1]):
        baseline_dist = np.histogram(baseline_sub_coherence\
                [:,band_num,baseline_inds].flatten(),bin_num)
        percentile_mat[band_num] = baseline_dist[1][1:]
        score_mat[band_num] = np.cumsum(baseline_dist[0])/np.sum(baseline_dist[0])

    def find_percentile_of(num, values, score):
        return score[np.argmin((values - num)**2)]

    baseline_sub_mean_coherence = np.mean(baseline_sub_coherence,axis=0)
    p_val_mat = np.zeros((baseline_sub_mean_coherence.shape[:2]))
    for this_iter in tqdm(np.ndindex(p_val_mat.shape)):
        p_val_mat[this_iter] = find_percentile_of(\
                baseline_sub_mean_coherence[this_iter],
                percentile_mat[this_iter[0]], score_mat[this_iter[0]])

    #p_val_mat = np.vectorize(find_percentile_of)\
    #        (baseline_sub_mean_coherence, percentile_val, score)

    #p_val_mat = np.zeros((baseline_sub_mean_coherence.shape[:2]))
    #for band_num in trange(p_val_mat.shape[0]):
    #    for bin_num in range(p_val_mat.shape[1]):
    #        p_val_mat[band_num, bin_num] = percentileofscore(\
    #                baseline_sub_coherence[:,band_num,baseline_inds].flatten(),
    #                baseline_sub_mean_coherence[band_num, bin_num])
    
    alpha = 0.05
    visualize.imshow(1*((p_val_mat>(1-(alpha/2))) \
            + (p_val_mat < (alpha/2))));plt.colorbar()
    this_ax = plt.gca()
    this_ax.set_yticks(range(len(freq_label_list)))
    this_ax.set_yticklabels(freq_label_list)
    this_ax.set_title('Signigicant difference from baseline\n'\
            + "_".join(node_path_list[this_node_num].split('/')[-2:]))
    plt.show()

    ####################################### 
    # Difference from shuffle
    ####################################### 
    # Mismatch trials between both regions
    region_phase_channels_long = np.reshape(region_phase_channels,
        (region_phase_channels.shape[0],-1,*region_phase_channels.shape[3:]))
    region0_long, region1_long = region_phase_channels_long[0],\
                                region_phase_channels_long[1]

    # To make sure nothing went wrong in the reshaping
    #coherence_array = np.abs(
    #    np.mean(
    #        np.exp(
    #            -1.j*np.diff(
    #                region_phase_channels_long,axis=0).squeeze()),axis=0))

    mismatch_coherence_array = np.zeros(\
            (bootstrap_samples,*phase_diff.shape[2:])) 

    def calc_mismatch_coherence(region0_long, region1_long):
        this_region0 = resample(region0_long)
        this_region1 = resample(region1_long)
        this_phase_diff = np.exp(-1.j*(this_region0-this_region1))
        coherence = np.abs(np.mean(\
                this_phase_diff,axis=0)).squeeze()
        return coherence

    mismatch_coherence_array = np.array( Parallel(n_jobs = cpu_count() - 2)\
            (delayed(calc_mismatch_coherence)(region0_long, region1_long) \
            for x in trange(bootstrap_samples)))

    #for repeat in trange(bootstrap_samples):
    #    this_region0 = resample(region0_long)
    #    this_region1 = resample(region1_long)
    #    this_phase_diff = np.exp(-1.j*(this_region0-this_region1))
    #    mismatch_coherence_array[repeat] = np.abs(np.mean(\
    #            this_phase_diff,axis=0)).squeeze()

    # Test plot : Plot mean coherence with std for all bands
    fig, ax = visualize.gen_square_subplots(coherence_boot_array.shape[1],
            sharey=True,sharex=True)
    for ax_num, this_ax in enumerate(ax.flatten()[:coherence_boot_array.shape[1]]):
        this_coherence = coherence_boot_array[:,ax_num]
        this_mismatch_coherence = mismatch_coherence_array[:,ax_num]
        mean_val = np.mean(this_coherence,axis=0)
        std_val = np.std(this_coherence,axis=0)
        mean_shuffle_val = np.mean(this_mismatch_coherence,axis=0)
        std_shuffle_val  = np.std(this_mismatch_coherence,axis=0)
        this_ax.plot(mean_val)
        this_ax.fill_between(x = np.arange(coherence_boot_array.shape[-1]),
                y1 = mean_val - 2*std_val,
                y2 = mean_val + 2*std_val, alpha = 0.5)
        this_ax.plot(mean_shuffle_val)
        this_ax.fill_between(x = np.arange(coherence_boot_array.shape[-1]),
                y1 = mean_shuffle_val - 2*std_shuffle_val,
                y2 = mean_shuffle_val + 2*std_shuffle_val, alpha = 0.5)
        this_ax.set_title(freq_label_list[ax_num])
    plt.suptitle('Shuffle comparison\n'\
            + "_".join(node_path_list[this_node_num].split('/')[-2:]))
    plt.show()

    # Generate null distribution of change in coherence
    # by taking bins from both pre- and post-stimulus delivery
    #mean_coherence_boot_array = np.mean(coherence_boot_array,axis=0)
    #bootstrap_samples = 1000
    #binned_coherence_array = np.mean(np.reshape(mean_coherence_boot_array,
    #            (mean_coherence_boot_array.shape[0],-1, bin_size)),axis=-1)
    #baseline_array = binned_coherence_array[:,:(baseline_t//bin_size)]
    #stimulus_array = binned_coherence_array[:,(baseline_t//bin_size):]
    #samples_per_epoch = np.min((baseline_array.shape[-1],
    #                            stimulus_array.shape[-1]))
    #null_dist_array = np.zeros((baseline_array.shape[0],
    #                            bootstrap_samples))
    ## Distribution of means of resampled distances
    #for repeat in trange(bootstrap_samples):
    #    null_dist_array[:,repeat] = np.mean(
    #            resample(baseline_array.T, n_samples = 1) - \
    #            resample(stimulus_array.T, n_samples = 1),
    #            axis=0)

# Test plot
# Plot baseline subtracted coherence along with lines indicating
# 95% confidence interval of the null distribution
#coherence_boot_array = coherence_boot_array - \
#        np.mean(\
#            coherence_boot_array[...,t < baseline_t],axis=-1)[...,np.newaxis]
#binned_coherence_boot_array = np.mean(np.reshape(coherence_boot_array,
#    (*coherence_boot_array.shape[:2],-1,bin_size)),axis=-1)
## Find confidence intervals for null distribution
#ci_interval = 0.95
#lower_bound, higher_bound = \
#        np.percentile(null_dist_array,(1-ci_interval)/2, axis=-1),\
#        np.percentile(null_dist_array,(1+ci_interval)/2, axis=-1)
#
#fig, ax = plt.subplots(1,coherence_boot_array.shape[1],sharey=True)
#for ax_num, this_ax in enumerate(ax):
#    this_coherence = binned_coherence_boot_array[:,ax_num]
#    mean_val = np.mean(this_coherence,axis=0)
#    std_val = np.std(this_coherence,axis=0)
#    this_ax.plot(mean_val)
#    this_ax.fill_between(\
#            x = np.arange(binned_coherence_boot_array.shape[-1]),
#            y1 = mean_val - 2*std_val,
#            y2 = mean_val + 2*std_val, alpha = 0.5)
#    this_ax.hlines((-lower_bound[ax_num],higher_bound[ax_num]),
#            0, binned_coherence_boot_array.shape[-1], color = 'r')
#    this_ax.set_aspect('auto')
#plt.show()

