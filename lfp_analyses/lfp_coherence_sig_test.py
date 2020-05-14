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

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

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

for this_node_num in tqdm(range(len(node_path_list))):

    animal_date_list = node_path_list[this_node_num].split('/')[2:]
    this_plot_dir = os.path.join(
                data_folder,*animal_date_list)

    with tables.open_file(data_hdf5_path,'r') as hf5:
        region_phase_channels = hf5.get_node(node_path_list[this_node_num],
                                'region_phase_channels')[:]
        phase_diff = hf5.get_node(node_path_list[this_node_num],
                                'phase_difference_array')[:]


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

    ####################################### 
    # Difference from baseline
    ####################################### 
    # Pool baseline coherence and conduct tests on non-overlapping bins
    ## ** time_vec is already defined **
    t = time_vec #np.arange(coherence_boot_array.shape[-1])
    baseline_t = 2000 #ms
    baseline_range = (1250,1750)
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

    diff_freq = np.unique(np.diff(freq_vec))
    freq_label_list = ["{}-{}".format(int(freq), int(freq+diff_freq)) \
            for freq in freq_vec]

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
                node_path_list[this_node_num],
                        'baseline_deviation_ecdf'),hf5)
        hf5.create_array(node_path_list[this_node_num], 
                'baseline_deviation_ecdf', p_val_mat) 

    ####################################### 
    # Comparison with shuffle
    ####################################### 
    # Mismatch trials between both regions
    region_phase_channels_long = np.reshape(region_phase_channels,
        (region_phase_channels.shape[0],-1,*region_phase_channels.shape[3:]))
    region0_long, region1_long = region_phase_channels_long[0],\
                                region_phase_channels_long[1]

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

    ########################################
    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    ########################################
                           

    # Plot 1
    # Bootstrapped coherence with baseline 95% CI
    fig, ax = visualize.gen_square_subplots(coherence_boot_array.shape[1])
    for ax_num, this_ax in enumerate(ax.flatten()\
            [:coherence_boot_array.shape[1]]):
        this_coherence = coherence_boot_array[:,ax_num]
        mean_val = np.mean(this_coherence,axis=0)
        std_val = np.std(this_coherence,axis=0)
        this_ax.plot(mean_val)
        this_ax.fill_between(x = np.arange(this_coherence.shape[-1]),
                y1 = mean_val - 2*std_val,
                y2 = mean_val + 2*std_val, alpha = 0.5)
        this_ax.hlines((lower_bound[ax_num],higher_bound[ax_num]),
                0, coherence_boot_array.shape[-1], color = 'r')
        this_ax.set_title(freq_label_list[ax_num])
    plt.suptitle('Baseline 95% CI\n'\
            + "_".join(node_path_list[this_node_num].split('/')[-2:]))
    fig.set_size_inches(16,8)
    fig.savefig(os.path.join(this_plot_dir,'coherence_baseline_CI'))
    plt.close(fig)

    # Plot 2
    # Marking significant deviations from baseline
    alpha = 0.05
    fig = plt.figure()
    visualize.imshow(1*((p_val_mat>(1-(alpha/2))) \
            + (p_val_mat < (alpha/2))));plt.colorbar()
    this_ax = plt.gca()
    this_ax.set_yticks(range(len(freq_label_list)))
    this_ax.set_yticklabels(freq_label_list)
    this_ax.set_title('Signigicant difference from baseline (alpha = {})\n'\
            .format(alpha)\
            + "_".join(node_path_list[this_node_num].split('/')[-2:]))
    plt.sca(this_ax)
    fig.set_size_inches(16,8)
    fig.savefig(os.path.join(this_plot_dir,'significant_coherence_baseline'))
    plt.close(fig)

    # Plot 3
    # Comparison of coherence with shuffle
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
    fig.set_size_inches(16,8)
    fig.savefig(os.path.join(this_plot_dir,'coherence_shuffle_comparison'))
    plt.close(fig)

    ######################################## 
    ## Delete all variables related to single file
    ######################################## 
    for item in dir():
        if item not in initial_dir:
            del globals()[item]

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
    for this_node_num in tqdm(range(len(node_path_list))):
        baseline_deviation_ecdf.append(\
                hf5.get_node(node_path_list[this_node_num],
                                'baseline_deviation_ecdf')[:])
baseline_deviation_ecdf = np.array(baseline_deviation_ecdf)

alpha = 0.05
increase_array = 1*(baseline_deviation_ecdf>(1-(alpha/2)))
decrease_array = 1*(baseline_deviation_ecdf < (alpha/2))
mean_increase_array = np.mean(increase_array,axis=0)
mean_decrease_array = np.mean(decrease_array,axis=0)
significance_array = 1*((increase_array + decrease_array)>0)
mean_signifince_array = np.mean(significance_array,axis=0)

freq_label_list = ["{}-{}".format(int(freq), int(freq+2)) \
        for freq in freq_vec]

# Plot 1
# Significant coherence aggreageted over all sessions
fig, ax = plt.subplots(3,1)
plt.sca(ax[0])
visualize.imshow(mean_increase_array);plt.colorbar()
ax[0].set_yticks(range(len(freq_label_list)))
ax[0].set_yticklabels(freq_label_list)
ax[0].set_title('Increased coherence')
plt.sca(ax[1])
visualize.imshow(mean_decrease_array);plt.colorbar()
ax[1].set_yticks(range(len(freq_label_list)))
ax[1].set_yticklabels(freq_label_list)
ax[1].set_title('Decreased coherence')
plt.sca(ax[2])
visualize.imshow(mean_signifince_array);plt.colorbar()
ax[2].set_yticks(range(len(freq_label_list)))
ax[2].set_yticklabels(freq_label_list)
ax[2].set_title('All significant deviation')
fig.set_size_inches(10,8)
plt.suptitle('Aggregate significant deviations in coherence')
plt.xlabel('Time (ms)')
ax[1].set_ylabel('Frequency band (Hz)')
fig.savefig(os.path.join(agg_plot_dir,'aggregate_coherence_significance'))
plt.close(fig)

# Plot 2
# Significant coherence for Alpha
y_ticks = np.arange(0,1.2,0.2)
mu_inds = np.where(1*(6<=freq_vec) * 1*(freq_vec<=12))[0]
fig, ax = plt.subplots(len(mu_inds), 1, sharey = True)
t = np.arange(significance_array.shape[-1])
for ax_num, this_ax in enumerate(ax):
    mean_sig = np.mean(significance_array[:,mu_inds[ax_num]],axis=0)
    this_ax.plot(mean_sig)#, c='r', linewidth=3)
    this_ax.set_title(freq_label_list[mu_inds[ax_num]])
    this_ax.set_yticks(y_ticks)
    #std_sig = np.std(significance_array[:,mu_inds[ax_num]],axis=0)
    #this_ax.fill_between(x = t,
    #            y1 = mean_sig + 2*std_sig, 
    #            y2 = mean_sig - 2*std_sig, alpha = 0.5)
    plt.suptitle('Aggregate Coherence Significance by band')
ax[int(np.floor(len(ax)/2))].set_ylabel('Fraction of significant bins')
ax[-1].set_xlabel('Time (ms)')
fig.set_size_inches(12,8)
fig.savefig(os.path.join(agg_plot_dir,'aggregate_coherence_sig_band'))
plt.close(fig)
