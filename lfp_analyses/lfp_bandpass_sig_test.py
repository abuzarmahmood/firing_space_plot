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
from joblib import Parallel, delayed, cpu_count
import shutil
from sklearn.utils import resample
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import imshow, firing_overview, gen_square_subplots

##################################################
## Define functions
##################################################

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

##################################################
## Read in data 
##################################################

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/fastdata/lfp_analyses'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

# Pull out bandpassed lfp and hilbert phase
# Also extract channels used for STFT phase analyses
# ** Future updates can reimplement picking channel for bandpass lfp **
with tables.open_file(data_hdf5_path,'r') as hf5:

    # Extract frequency values and time vector for the STFT
    band_freqs = hf5.get_node('/bandpass_lfp','frequency_bands')[:]
    lfp_node_path_list = \
            [x.__str__().split(" ")[0] \
            for x in hf5.root.bandpass_lfp._f_walknodes() \
            if 'bandpassed_lfp_array' in x.__str__()]
    lfp_node_path_list.sort()

    # Check which nodes in STFT have 'relative_region_channel_nums'
    channel_path_list = \
            [x.__str__().split(" ")[0] \
            for x in hf5.root.stft._f_walknodes() \
            if 'relative_region_channel_nums' in x.__str__()]
    channel_path_list.sort()
    relative_channel_nums = [hf5.get_node(os.path.dirname(this_path),
                            'relative_region_channel_nums')[:] \
                            for this_path in channel_path_list]

# Check which files have both bandpassed lfp and stft
# This is needed so the channels used for STFT coherence can also be used here
lfp_name_date_str = [x.split('/')[2:4] for x in lfp_node_path_list]
channel_name_date_str = [x.split('/')[2:4] for x in channel_path_list]
common_files = [file for file in lfp_name_date_str if file in channel_name_date_str]
lfp_inds = [num for num, file in enumerate(lfp_name_date_str) \
        if file in common_files]
channel_inds = [num for num, file in enumerate(channel_name_date_str) \
        if file in common_files]

fin_lfp_node_path_list = [lfp_node_path_list[i] for i in lfp_inds] 
fin_lfp_name_date_str = [lfp_name_date_str[i] for i in lfp_inds] 
fin_channel_nums = [relative_channel_nums[i] for i in channel_inds]


# Print which files will be processed
# Find union of files
union_files = list(set(map(tuple,lfp_name_date_str+channel_name_date_str)))
union_files.sort()
union_process_inds = [1 if file in map(tuple,common_files) else 0 \
        for file in union_files]
print('The following files will be processed')
print("\n".join(list(map(str,zip(union_process_inds,union_files)))))

# Run through files
# Calculate phase difference histograms and
# Plot phases of random subset of trials
for this_node_num in tqdm(range(len(fin_lfp_node_path_list))):

    this_plot_dir = os.path.join(
                data_folder,*fin_lfp_name_date_str[this_node_num])

    with tables.open_file(data_hdf5_path,'r') as hf5:
        phase_array = hf5.get_node(
                os.path.dirname(fin_lfp_node_path_list[this_node_num]),
                'phase_array')[:][:,:,fin_channel_nums[this_node_num]]

    # Bootstrap coherence values and calculate deviation from baseline
    # as was done for STFT phase coherence

    # Calculate phase difference
    # shape ::: bands x taste x region x trial x time
    phase_diff = np.diff(phase_array,axis=2).squeeze()
    phase_diff_reshape = phase_diff.reshape(\
            (phase_array.shape[0],-1,phase_diff.shape[-1]))
    phase_coherence_array = \
            np.abs(np.mean(np.exp(-1.j*phase_diff_reshape),axis=1))
    phase_bin_nums = 20
    phase_bins = np.linspace(-np.pi, np.pi, phase_bin_nums)
    phase_diff_hists = np.array([[np.histogram(time_bin,phase_bins)[0] \
            for time_bin in freq] \
            for freq in phase_diff_reshape.swapaxes(1,2)]).swapaxes(1,2) 

    # Test plots
    #firing_overview(phase_diff_hists);plt.show()
    #imshow(phase_coherence_array);plt.show()

    #fig, ax = plt.subplots(phase_coherence_array.shape[0])
    #for data,this_ax in zip(phase_coherence_array,ax):
    #    this_ax.plot(data)
    #plt.show()

    ##################################################
    # Estimate error in coherence calculation by bootstrapping
    ##################################################
    bootstrap_samples = 500
    coherence_boot_array = np.zeros(\
            (bootstrap_samples,
                phase_diff_reshape.shape[0], 
                phase_diff_reshape.shape[-1])) 

    def generate_bootstrap_coherence(phase_diff_reshape):
        this_phase_diff = phase_diff_reshape[:,\
        np.random.choice(range(phase_diff_reshape.shape[1]),
                                phase_diff_reshape.shape[1], replace = True)]
        return np.abs(np.mean(np.exp(-1.j*this_phase_diff),axis=(1)))

    coherence_boot_array = np.array(\
        Parallel(n_jobs = cpu_count() - 2)\
        (delayed(generate_bootstrap_coherence)(phase_diff_reshape)\
                for repeat in trange(bootstrap_samples)))

    #for repeat in trange(bootstrap_samples):
    #    this_phase_diff = phase_diff_reshape[:,\
    #    np.random.choice(range(phase_diff_reshape.shape[1]),
    #                            phase_diff_reshape.shape[1], replace = True)]
    #    coherence_boot_array[repeat] = \
    #            np.abs(np.mean(np.exp(-1.j*this_phase_diff),axis=(1)))

    ####################################### 
    # Shuffled coherence 
    ####################################### 
    phase_array_long = phase_array.swapaxes(1,2)
    phase_array_long = phase_array_long.reshape(
            (*phase_array_long.shape[:2],-1,phase_array_long.shape[-1]))
    
    region0_long, region1_long = phase_array_long[:,0].swapaxes(0,1),\
                                phase_array_long[:,1].swapaxes(0,1)

    # shape ::: samples x bands x time
    #mismatch_coherence_array = np.zeros(\
    #        (bootstrap_samples, phase_diff.shape[0], phase_diff.shape[-1])) 

    def calc_mismatch_coherence(region0_long, region1_long):
        # Just have to resample one region
        this_region0 = region0_long
        this_region1 = resample(region1_long)
        this_phase_diff = np.exp(-1.j*(this_region0-this_region1))
        coherence = np.abs(np.mean(\
                this_phase_diff,axis=0)).squeeze()
        return coherence

    mismatch_coherence_array = np.array( Parallel(n_jobs = cpu_count() - 2)\
            (delayed(calc_mismatch_coherence)(region0_long, region1_long) \
            for x in trange(bootstrap_samples)))

    ####################################### 
    # Difference from baseline
    ####################################### 
    # Pool baseline coherence and conduct tests on non-overlapping bins
    t = np.arange(coherence_boot_array.shape[-1])
    baseline_t = 2000 #ms
    baseline_range = (1000,1500)
    baseline_inds = np.where((t>baseline_range[0])*(t<baseline_range[1]))[0]
    bin_size = 100 #ms
    bin_starts = np.arange(0, max(t) , bin_size)  
    bin_inds = [(x,x+bin_size) for x in bin_starts]
    t_binned = np.arange(0,max(t),bin_size)

    ci_interval = 0.95
    lower_bound, higher_bound = \
            np.percentile(coherence_boot_array[...,baseline_inds],
                    100*(1-ci_interval)/2, axis=(0,-1)),\
            np.percentile(coherence_boot_array[...,baseline_inds],
                    100*(1+ci_interval)/2, axis=(0,-1))

    # Find bootstrapped deviation from mean baseline for baseline and
    # use to test differences
    # 1) Generate sampling distribution of baseline mean from baseline_sub array
    #       with count per sample equal to bin size
    # 2) Find p-value of mean-bin values using sampling distribution

    freq_label_list = ["{}-{}".format(int(freq[0]), int(freq[1])) \
            for freq in band_freqs]

    bin_num = 1000
    percentile_mat = np.zeros((coherence_boot_array.shape[1],
                        bin_num))
    score_mat = np.zeros((coherence_boot_array.shape[1],
                        bin_num))

    for band_num in trange(coherence_boot_array.shape[1]):
        baseline_dist = np.histogram(coherence_boot_array\
                [:,band_num,baseline_inds].flatten(),bin_num)
        percentile_mat[band_num] = baseline_dist[1][1:]
        score_mat[band_num] = np.cumsum(baseline_dist[0])/np.sum(baseline_dist[0])

    def find_percentile_of(num, values, score):
        return score[np.argmin((values - num)**2)]

    mean_coherence_array = np.mean(coherence_boot_array,axis=0)
    p_val_mat = np.zeros((mean_coherence_array.shape[:2]))
    for this_iter in tqdm(np.ndindex(p_val_mat.shape)):
        p_val_mat[this_iter] = find_percentile_of(\
                mean_coherence_array[this_iter],
                percentile_mat[this_iter[0]], score_mat[this_iter[0]])

    ######################################## 
    # Write p_val_mat to HDF5
    ######################################## 
    with tables.open_file(data_hdf5_path,'r+') as hf5:

        remove_node(os.path.join(
                os.path.dirname(fin_lfp_node_path_list[this_node_num]),
                        'baseline_deviation_ecdf'),hf5)
        hf5.create_array(
                os.path.dirname(fin_lfp_node_path_list[this_node_num]), 
                'baseline_deviation_ecdf', p_val_mat) 

    ########################################
    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    ########################################
                           

    # Plot 1
    # Bootstrapped coherence with baseline 95% CI
    # Mark deviations with different color

    # Also add shuffle for comparison

    # Time limits for plotting
    t_lims = [1000,4500]
    stim_t = 2000
    norm = matplotlib.colors.Normalize(0,1)
    cmap_object= matplotlib.cm.ScalarMappable(cmap = 'viridis', norm = norm)
    alpha = 0.05
    min_time = 100
    sig_pval_mat = 1*((p_val_mat>(1-(alpha/2))) \
            + (p_val_mat < (alpha/2)))
    t_vec = np.arange(p_val_mat.shape[-1])

    fig, ax = gen_square_subplots(coherence_boot_array.shape[1])
    for ax_num, this_ax in enumerate(ax.flatten()\
            [:coherence_boot_array.shape[1]]):
        diff_vals = np.diff(sig_pval_mat[ax_num])
        change_inds = np.where(diff_vals!=0)[0]
        if diff_vals[change_inds][0] == -1:
            change_inds = np.concatenate([np.array(0)[np.newaxis],change_inds])    
        if diff_vals[change_inds][-1] == 1:
            change_inds = np.concatenate([change_inds, (max(t_vec)-1)[np.newaxis]])    
        change_inds = change_inds.reshape((-1,2))
        fin_change_inds = change_inds[np.diff(change_inds,axis=-1).flatten() > min_time]
        fin_change_inds -= stim_t
        # Remove any which are after end of plotting period
        upper_change_lim = np.sum(fin_change_inds < t_lims[1] - stim_t,axis=-1)==2
        lower_change_lim = np.sum(fin_change_inds > t_lims[0] - stim_t,axis=-1)==2
        fin_inds = np.where(upper_change_lim * lower_change_lim)[0]
        fin_change_inds = fin_change_inds[fin_inds]
        this_coherence = coherence_boot_array[:,ax_num]
        mean_val = np.mean(this_coherence,axis=0)
        std_val = np.std(this_coherence,axis=0)
        this_ax.plot(t_vec[t_lims[0]:t_lims[1]]-stim_t,mean_val[t_lims[0]:t_lims[1]])
        this_ax.fill_between(\
                x = t_vec[t_lims[0]:t_lims[1]] - stim_t,
                y1 = mean_val[t_lims[0]:t_lims[1]] - 2*std_val[t_lims[0]:t_lims[1]],
                y2 = mean_val[t_lims[0]:t_lims[1]] + 2*std_val[t_lims[0]:t_lims[1]], 
                alpha = 0.5)
        this_ax.hlines((lower_bound[ax_num],higher_bound[ax_num]),
                t_lims[0] - stim_t, t_lims[1]-stim_t, color = 'r', alpha = 0.5)
        for interval  in fin_change_inds:
            this_ax.axvspan(interval[0],interval[1],facecolor='y',alpha = 0.5)
        this_mismatch_coherence = mismatch_coherence_array[:,ax_num]
        mean_shuffle_val = np.mean(this_mismatch_coherence,axis=0)
        std_shuffle_val  = np.std(this_mismatch_coherence,axis=0)
        this_ax.plot(t_vec[t_lims[0]:t_lims[1]]-stim_t, 
                mean_shuffle_val[t_lims[0]:t_lims[1]])
        this_ax.fill_between(x = t_vec[t_lims[0]:t_lims[1]]-stim_t,
                y1 = mean_shuffle_val[t_lims[0]:t_lims[1]] - \
                        2*std_shuffle_val[t_lims[0]:t_lims[1]],
                y2 = mean_shuffle_val[t_lims[0]:t_lims[1]] + \
                        2*std_shuffle_val[t_lims[0]:t_lims[1]], alpha = 0.5)
        this_ax.set_title(freq_label_list[ax_num])
        this_ax.set_ylabel('Coherence (mean +/- 95% CI)')
        this_ax.set_xlabel('Time post-stimulus delivery (ms)')
    plt.suptitle('Baseline 95% CI (Bandpass) \n'\
            + "_".join(fin_lfp_node_path_list[this_node_num].split('/')[2:4]) + \
            '\nalpha = {}, minimum significant window  = {}'.format(alpha,min_time))
    fig.set_size_inches(16,8)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85, wspace = 0.15)
    fig.savefig(os.path.join(this_plot_dir,
        '_'.join(fin_lfp_name_date_str[this_node_num])+'_bandpass_coherence_baseline_CI'))
    plt.close(fig)

    # Plot 3
    # Comparison of coherence with shuffle
    #fig, ax = visualize.gen_square_subplots(coherence_boot_array.shape[1],
    #        sharey=True,sharex=True)
    #for ax_num, this_ax in enumerate(ax.flatten()[:coherence_boot_array.shape[1]]):
    #    this_coherence = coherence_boot_array[:,ax_num]
    #    this_mismatch_coherence = mismatch_coherence_array[:,ax_num]
    #    mean_val = np.mean(this_coherence,axis=0)
    #    std_val = np.std(this_coherence,axis=0)
    #    mean_shuffle_val = np.mean(this_mismatch_coherence,axis=0)
    #    std_shuffle_val  = np.std(this_mismatch_coherence,axis=0)
    #    this_ax.plot(mean_val)
    #    this_ax.fill_between(x = np.arange(coherence_boot_array.shape[-1]),
    #            y1 = mean_val - 2*std_val,
    #            y2 = mean_val + 2*std_val, alpha = 0.5)
    #    this_ax.plot(mean_shuffle_val)
    #    this_ax.fill_between(x = np.arange(coherence_boot_array.shape[-1]),
    #            y1 = mean_shuffle_val - 2*std_shuffle_val,
    #            y2 = mean_shuffle_val + 2*std_shuffle_val, alpha = 0.5)
    #    this_ax.set_title(freq_label_list[ax_num])
    #plt.suptitle('Shuffle comparison\n'\
    #        + "_".join(node_path_list[this_node_num].split('/')[-2:]))
    #fig.set_size_inches(16,8)
    #plt.show()

    #fig.savefig(os.path.join(this_plot_dir,'coherence_shuffle_comparison'))
    #plt.close(fig)

##################################################
#    _                                    _       
#   / \   __ _  __ _ _ __ ___  __ _  __ _| |_ ___ 
#  / _ \ / _` |/ _` | '__/ _ \/ _` |/ _` | __/ _ \
# / ___ \ (_| | (_| | | |  __/ (_| | (_| | ||  __/
#/_/   \_\__, |\__, |_|  \___|\__, |\__,_|\__\___|
#        |___/ |___/          |___/               
##################################################


##################################################
## Aggregate all significant bins across animals
##################################################
agg_plot_dir = os.path.join(data_folder,'aggregate_analysis')
if not os.path.exists(agg_plot_dir):
    os.makedirs(agg_plot_dir)

baseline_deviation_ecdf= []
with tables.open_file(data_hdf5_path,'r') as hf5:
    for this_node_num in tqdm(range(len(fin_lfp_node_path_list))):
        baseline_deviation_ecdf.append(\
                hf5.get_node(\
                    os.path.dirname(fin_lfp_node_path_list[this_node_num]),
                                'baseline_deviation_ecdf')[:])

t_range = np.arange(5000)
baseline_deviation_ecdf = np.array(baseline_deviation_ecdf)[...,t_range]

alpha = 0.05
increase_array = 1*(baseline_deviation_ecdf>(1-(alpha/2)))
decrease_array = 1*(baseline_deviation_ecdf < (alpha/2))
mean_increase_array = np.mean(increase_array,axis=0)
mean_decrease_array = np.mean(decrease_array,axis=0)
significance_array = 1*((increase_array + decrease_array)>0)
mean_signifince_array = np.mean(significance_array,axis=0)

freq_label_list = ["{}-{}".format(int(freq[0]), int(freq[1])) \
        for freq in band_freqs]

# Plot 1
# Significant coherence aggreageted over all sessions
fig, ax = plt.subplots(3,1)
plt.sca(ax[0])
imshow(mean_increase_array);plt.colorbar()
ax[0].set_yticks(range(len(freq_label_list)))
ax[0].set_yticklabels(freq_label_list)
ax[0].set_title('Increased coherence')
plt.sca(ax[1])
imshow(mean_decrease_array);plt.colorbar()
ax[1].set_yticks(range(len(freq_label_list)))
ax[1].set_yticklabels(freq_label_list)
ax[1].set_title('Decreased coherence')
plt.sca(ax[2])
imshow(mean_signifince_array);plt.colorbar()
ax[2].set_yticks(range(len(freq_label_list)))
ax[2].set_yticklabels(freq_label_list)
ax[2].set_title('All significant deviation')
fig.set_size_inches(10,8)
plt.suptitle('Aggregate significant deviations in coherence')
plt.xlabel('Time (ms)')
ax[1].set_ylabel('Frequency band (Hz)')
plt.suptitle('Bandpass LFP')
fig.savefig(os.path.join(agg_plot_dir,'bandpass_aggregate_coherence_significance'))
plt.close(fig)

# Plot 2
# Significant coherence as line plots 
y_ticks = np.arange(0,1.2,0.2)
t = np.arange(significance_array.shape[-1])
mean_sig = np.mean(significance_array,axis=0)
fig, ax = plt.subplots(mean_signifince_array.shape[0], 1, sharey = True)

for ax_num, this_ax in enumerate(ax):
    this_ax.plot(t,mean_sig[ax_num])#, c='r', linewidth=3)
    this_ax.set_title(freq_label_list[ax_num])
    this_ax.set_yticks(y_ticks)
    plt.suptitle('Bandpass LFP \n Aggregate Coherence Significance by band')

ax[int(np.floor(len(ax)/2))].set_ylabel('Fraction of significant bins')
ax[-1].set_xlabel('Time (ms)')
fig.set_size_inches(12,8)
fig.savefig(os.path.join(agg_plot_dir,'bandpass_aggregate_coherence_sig_band'))
plt.close(fig)

# Plot 3
# Scaled Significant coherence as line plots 
epochs = np.array([[2000,2250],[2250,2800],[2800,3250]])
color_vec = ['pink','orange','blue']
t = np.arange(significance_array.shape[-1])
mean_sig = np.mean(significance_array,axis=0)
fig, ax = plt.subplots(mean_signifince_array.shape[0], 1, sharex=True)
for ax_num, this_ax in enumerate(ax):
    this_ax.plot(t[t_lims[0]:t_lims[1]] - stim_t,
            mean_sig[ax_num][t_lims[0]:t_lims[1]], c='k', linewidth=3)
    this_ax.set_title(freq_label_list[ax_num])
    for num,interval in enumerate(epochs):
        this_ax.axvspan(interval[0] - stim_t,interval[1] - stim_t,
                facecolor=color_vec[num],alpha = 0.5)
    plt.suptitle('Bandpass LFP \n Aggregate Coherence Significance by band')
ax[int(np.floor(len(ax)/2))].set_ylabel('Fraction of significant bins')
ax[-1].set_xlabel('Time post-stimulus delivery  (ms)')
fig.set_size_inches(8,12)
fig.savefig(os.path.join(agg_plot_dir,
    'bandpass_aggregate_coherence_sig_band_scaled'))
plt.close(fig)

# Plot 4
# Smooth Scaled Significant coherence as line plots 
def gauss_kern(size):
    x = np.arange(-size,size+1)
    kern = np.exp(-(x**2)/float(size))
    return kern / sum(kern)
def gauss_filt(vector, size):
    kern = gauss_kern(size)
    return np.convolve(vector, kern, mode='same')
from scipy.signal import savgol_filter
epochs = np.array([[2000,2250],[2250,2800],[2800,3250]])
color_vec = ['pink','orange','blue']
t = np.arange(significance_array.shape[-1])
mean_sig = np.mean(significance_array,axis=0)
#smooth_mean_sig = np.array([gauss_filt(x,50) for x in mean_sig])
smooth_mean_sig = savgol_filter(mean_sig, 51, 1) 
fig, ax = plt.subplots(mean_signifince_array.shape[0], 1, sharex=True)
for ax_num, this_ax in enumerate(ax):
    this_ax.plot(t[t_lims[0]:t_lims[1]] - stim_t,
            smooth_mean_sig[ax_num][t_lims[0]:t_lims[1]], c='k', linewidth=3)
    this_ax.set_title(freq_label_list[ax_num] + ' Hz')
    for num,interval in enumerate(epochs):
        this_ax.axvspan(interval[0] - stim_t, interval[1] - stim_t,
                facecolor=color_vec[num],alpha = 0.5)
    plt.suptitle('Bandpass LFP \n Aggregate Coherence Significance by band')
ax[int(np.floor(len(ax)/2))].set_ylabel('Fraction of significant bins')
ax[-1].set_xlabel('Time post-stimulus delivery (ms)')
fig.set_size_inches(8,12)
fig.savefig(os.path.join(agg_plot_dir,
    'bandpass_aggregate_coherence_sig_band_scaled_smooth'))
plt.close(fig)
