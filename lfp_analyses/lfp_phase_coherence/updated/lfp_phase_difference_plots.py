## Import required modules
import os
import matplotlib.pyplot as plt
import tables
import numpy as np
from tqdm import tqdm, trange
import shutil
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import zscore
from pathlib import Path
from joblib import Parallel, delayed, cpu_count

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

##################################################
## Define functions
##################################################

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

##################################################
## Read in data 
##################################################
dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
#file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 

for this_dir in tqdm(dir_list):

    this_dir = dir_list[1]
    #this_dir = dir_list[0]
    dat = ephys_data(this_dir)
    dat.stft_params['max_freq'] = 100
    dat.get_stft(recalculate = False, dat_type = ['phase'])
    dat.get_lfp_electrodes()

    #stft_array = dat.stft_array
    #stft_array = np.concatenate(np.swapaxes(stft_array, 1,2))
    phase_array = dat.phase_array
    phase_array = np.concatenate(np.swapaxes(phase_array, 1,2))
    time_vec = dat.time_vec
    freq_vec = dat.freq_vec
    region_electrodes = dat.lfp_region_electrodes

    #  ____      _                                  
    # / ___|___ | |__   ___ _ __ ___ _ __   ___ ___ 
    #| |   / _ \| '_ \ / _ \ '__/ _ \ '_ \ / __/ _ \
    #| |__| (_) | | | |  __/ | |  __/ | | | (_|  __/
    # \____\___/|_| |_|\___|_|  \___|_| |_|\___\___|
    #
    #Refer to:
    #    http://math.bu.edu/people/mak/sfn-2013/sfn_tutorial.pdf
    #    http://math.bu.edu/people/mak/sfn/tutorial.pdf

    # Find channel closest to mean phase
    select_region_electrodes = []
    for region in region_electrodes:
        #region = region_electrodes[0]
        this_region_phase = phase_array[:,region] 
        region_mean_phase = np.mean(this_region_phase,axis=1)
        phase_abs_diff = np.sum(
                np.abs(this_region_phase - region_mean_phase[:, np.newaxis]),
                axis = (0,2,3))
        select_ind = np.argmin(phase_abs_diff)
        select_region_electrodes.append(region[select_ind])

    min_err_phase = phase_array[:, np.array(select_region_electrodes)].swapaxes(0,1)
    phase_diff = np.exp(1.j*(min_err_phase[0] - min_err_phase[1]))

    # Downsample phase difference to make calculation easier
    downsample_mult = 10
    phase_diff_down = phase_diff[...,::downsample_mult]

    bins = 30
    freq_bins = np.round(np.linspace(-np.pi, np.pi, bins), 2)
    phase_diff_hists = np.zeros((bins, *phase_diff_down.shape[1:]))
    iters = list(np.ndindex(phase_diff_down.shape[1:]))
    for this_iter in tqdm(iters):
        phase_diff_hists[:,this_iter[0], this_iter[1]] = np.histogram(
                np.angle(phase_diff_down[:,this_iter[0], this_iter[1]]), 
                bins = bins)[0]

    from scipy.signal import savgol_filter as savgol

    img_kwargs = dict(interpolation= 'spline16', aspect = 'auto', cmap = 'viridis')
    freq_ind = 3
    this_phase_diff = phase_diff_down[:,freq_ind]
    var_diff = np.abs(np.mean(this_phase_diff, axis=0))
    img_dat=  phase_diff_hists[:,freq_ind]
    mode_diff = np.argmax(img_dat,axis=0)
    filt_mode = savgol(mode_diff, 21, 2)
    #fig,ax = plt.subplots(1,2, sharey=True)
    fig = plt.figure()
    ax = [fig.add_subplot(2,2,3), fig.add_subplot(2,2,4), fig.add_subplot(2,2,1)]
    ax[0].imshow(img_dat, **img_kwargs);
    ax[0].plot(np.arange(img_dat.shape[1]), filt_mode, 
            color = 'yellow', alpha = 0.7, linewidth = 1)
    wanted_ticks = np.array([int(x) for x in ax[0].get_yticks() if x < bins])
    ax[0].set_yticks(ticks = wanted_ticks)
    ax[0].set_yticklabels(labels = freq_bins[wanted_ticks])    
    ax[0].axhline(np.argmin((freq_bins**2)), color = 'yellow', linestyle = '--')
    ax[1].hist(filt_mode, orientation = 'horizontal', density = True)
    ax[1].axhline(np.argmin((freq_bins**2)), 
            color = 'k', linestyle = '--')
    ax[2].plot(np.arange(img_dat.shape[1]), var_diff)
    #plt.colorbar();
    plt.show()

    img_kwargs = dict(interpolation= 'nearest', aspect = 'auto', cmap = 'hsv')
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(min_err_phase[0,:,3], **img_kwargs)
    ax[1].imshow(min_err_phase[1,:,3], **img_kwargs);
    plt.show()
