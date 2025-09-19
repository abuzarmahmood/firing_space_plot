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
from scipy.signal import savgol_filter as savgol

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

plot_dir = '/media/bigdata/firing_space_plot/lfp_analyses/lfp_phase_coherence/plots'

for this_dir in tqdm(dir_list):

    this_dir = dir_list[7]
    basename = os.path.basename(this_dir)
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
    region_names = dat.region_names

    sorted_region_order = np.argsort(region_names)
    sorted_region_names = [region_names[x] for x in sorted_region_order]
    sorted_region_electrodes = [region_electrodes[x] for x in sorted_region_order]

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
    for region in sorted_region_electrodes:
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
    down_t_vec = time_vec[::downsample_mult]
    stim_t = 2
    down_t_vec = (down_t_vec - stim_t) * 1000

    bins = 30
    phase_bins = np.round(np.linspace(-np.pi, np.pi, bins), 2)
    phase_diff_hists = np.zeros((bins-1, *phase_diff_down.shape[1:]))
    iters = list(np.ndindex(phase_diff_down.shape[1:]))
    for this_iter in tqdm(iters):
        phase_diff_hists[:,this_iter[0], this_iter[1]] = np.histogram(
                np.angle(phase_diff_down[:,this_iter[0], this_iter[1]]), 
                bins = phase_bins)[0]

    #wanted_freq_lims = [8,12] 
    #freq_ind = [np.argmin(np.abs(freq_vec - x)) for x in wanted_freq_lims]#3
    #this_phase_diff = phase_diff_down[:,freq_ind[0]:freq_ind[1]+1]

    #wanted_freq = 10
    freq_ind = np.argmin(np.abs(freq_vec - wanted_freq))
    this_phase_diff = phase_diff_down[:,freq_ind]
    this_coh = np.abs(np.mean(this_phase_diff,axis=0))

    #var_diff = np.abs(np.mean(this_phase_diff, axis=0))
    img_dat=  phase_diff_hists[:,freq_ind]

    img_kwargs = dict(interpolation= 'spline16', aspect = 'auto', cmap = 'viridis')
    #this_dir = dir_list[0]
    mode_diff_inds = np.argmax(img_dat,axis=0)
    mode_diff = phase_bins[mode_diff_inds]
    filt_mode = savgol(mode_diff, 11, 2)
    #fig,ax = plt.subplots(1,2, sharey=True)
    #fig = plt.figure()
    fig,ax = plt.subplots(2,2, 
            sharey='row', sharex = 'col', figsize = (7,6)) 
    ax = ax.flatten()
    #ax = [fig.add_subplot(2,2,3), fig.add_subplot(2,2,4), fig.add_subplot(2,2,1)]
    #ax[0].imshow(img_dat, **img_kwargs);
    ax[0].pcolormesh(down_t_vec, phase_bins[1:], img_dat, 
            shading = 'gouraud', cmap = 'viridis') 
    #mean_phase = (img_dat * phase_bins[1:, np.newaxis]).mean(axis=0)
    mean_phase = np.angle(this_phase_diff.mean(axis=0))
    ax[0].plot(down_t_vec, mean_phase, 
            color = 'yellow', alpha = 0.7, linewidth = 1)
    #wanted_ticks = np.array([int(x) for x in ax[0].get_yticks() if x < bins])
    #ax[0].set_yticks(ticks = wanted_ticks)
    #ax[0].set_yticklabels(labels = phase_bins[wanted_ticks])    
    ax[0].axhline(0, color = 'yellow', linestyle = '--')
    ax[1].hist(filt_mode, bins = phase_bins, 
            orientation = 'horizontal', density = True)
    ax[1].axhline(0, color = 'k', linestyle = '--')
    ax[2].plot(down_t_vec, this_coh)
    ax[2].set_title('Phase Coherence')
    ax[0].set_title('Phase Difference : BLA - GC')
    ax[0].set_ylabel(' <-- GC Leads...BLA Leads -->')
    ax[0].set_xlim([0,2000])
    fig.suptitle(f'{basename} ::: {freq_vec[freq_ind]} Hz')
    #ax[2].plot(np.arange(img_dat.shape[1]), var_diff)
    #plt.colorbar();
    plt.subplots_adjust(top = 0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 
        f'phase_diff_{basename}_{int(freq_vec[freq_ind])}Hz'),
        dpi = 300)
    plt.close(fig)
    #plt.show()

    #img_kwargs = dict(interpolation= 'nearest', aspect = 'auto', cmap = 'hsv')
    #fig,ax = plt.subplots(2,1, sharex=True,sharey=True)
    #ax[0].imshow(min_err_phase[0,:,freq_ind], **img_kwargs)
    #ax[1].imshow(min_err_phase[1,:,freq_ind], **img_kwargs);
    #plt.show()
