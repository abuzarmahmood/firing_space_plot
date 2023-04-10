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
import pandas as pd
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

    # Get spikes
    dat.get_region_units()
    region_sort_order = np.argsort(dat.region_names)
    sorted_region_names = np.array(dat.region_names)[region_sort_order]
    region_spikes = [dat.return_region_spikes(x) for x in sorted_region_names]
    long_region_spikes = [np.concatenate(x) for x in region_spikes]

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
    # Sort by region
    min_err_phase = min_err_phase[region_sort_order]
    del phase_array

    # ____  _                      _               _    _             
    #|  _ \| |__   __ _ ___  ___  | |    ___   ___| | _(_)_ __   __ _ 
    #| |_) | '_ \ / _` / __|/ _ \ | |   / _ \ / __| |/ / | '_ \ / _` |
    #|  __/| | | | (_| \__ \  __/ | |__| (_) | (__|   <| | | | | (_| |
    #|_|   |_| |_|\__,_|___/\___| |_____\___/ \___|_|\_\_|_| |_|\__, |
    #                                                           |___/ 
    # Phase locking of a region's spikes to LFP of the other region
    # This would suggest if the activity in a region is contributing to inputs in the other region

    phase_list = []
    for region_ind in range(len(sorted_region_names)):
        #region_ind = 0
        this_phase = np.swapaxes(min_err_phase[region_ind],1,-1)
        phase_region = sorted_region_names[region_ind]
        this_spikes = long_region_spikes[1-region_ind]
        spikes_region = sorted_region_names[1-region_ind]
        # Cut to same length
        this_spikes = np.swapaxes(this_spikes[...,:this_phase.shape[1]],0,1)
        #for nrn_num, this_nrn in tqdm(enumerate(this_spikes)):
        for nrn_num, this_nrn in enumerate(this_spikes):
            if this_nrn.sum(axis=None):
                inds = np.where(this_nrn)
                phases = this_phase[inds]
                phase_inds = np.array(list(np.ndindex(phases.shape)))
                phase_frame = pd.DataFrame(
                        dict(
                           spikes_region = spikes_region,
                           phase_region = phase_region,
                           nrn_num = nrn_num,
                           trials = inds[0][phase_inds[:,0]],
                           time = inds[1][phase_inds[:,0]],
                           freq = phase_inds[:,1],
                           phases = phases.flatten()
                            )
                        )
                phase_frame = phase_frame.set_index(
                        ['spikes_region','phase_region','nrn_num','trials','time','freq'])
                phase_list.append(phase_frame)
    fin_phase_frame = pd.concat(phase_list)

    hf5_save_path = '/stft/analyses/spike_phase_coherence'
    fin_phase_frame.to_hdf(dat.hdf5_path, hf5_save_path) 
