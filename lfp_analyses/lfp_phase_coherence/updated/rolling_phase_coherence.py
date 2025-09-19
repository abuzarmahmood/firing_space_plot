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
import xarray as xr

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
    phase_diff_rad = np.angle(phase_diff)

    img_kwargs = dict(interpolation= 'nearest', aspect = 'auto', cmap = 'twilight')
    fig,ax = plt.subplots(2,1)
    im = ax[0].imshow(phase_diff_rad[:,3], **img_kwargs)
    plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(phase_diff_rad[:,3,::10], **img_kwargs)
    ax[1].colorbar()
    plt.colorbar(im, ax=ax[1])
    plt.colorbar()
    plt.show()

    test_diff = phase_diff[:,3]
    coh = np.abs(test_diff.mean(axis=0))
    plt.plot(coh);plt.show()

    # See if the phase difference vectors are correlated temporally
    # Magnitude of mean phase-difference vectors in a given window
    # Downsample phase_difference to make calculation easier
    down_factor = 10
    phase_diff_down = phase_diff[...,::down_factor]
    window_len = 250//down_factor

    phase_diff_xr = xr.DataArray(
            phase_diff_down,
            coords = [np.arange(len(phase_diff_down)),
                        freq_vec,
                        time_vec[::down_factor]],
            dims = ['trials','freqs','time']
            )

    calc_coh = lambda x,axis: np.abs(np.mean(x, axis=axis))
    temp_phase_coh = phase_diff_xr.rolling(time=window_len, center=True).reduce(calc_coh)

    img_kwargs = dict(interpolation= 'nearest', aspect = 'auto', cmap = 'twilight')
    fig,ax = plt.subplots(3,1)
    im = ax[0].imshow(phase_diff_rad[:,3,::10], **img_kwargs)
    #plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(temp_phase_coh[:,3], **img_kwargs)
    #plt.colorbar(im, ax=ax[1])
    ax[2].plot(temp_phase_coh[:,3].time.values,
            temp_phase_coh[:,3].mean(dim='trials').values)
    plt.show()
