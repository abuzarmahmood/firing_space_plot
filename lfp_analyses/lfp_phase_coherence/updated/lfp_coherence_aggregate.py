
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

def calc_coherence(stft_a, stft_b, trial_axis = 0):
    """
    inputs : arrays of shape (trials x freq x time)
    """
    cross_spec = np.mean(stft_a * np.conj(stft_b),axis=trial_axis)
    a_power_spectrum = np.mean(np.abs(stft_a)**2,axis=trial_axis)
    b_power_spectrum = np.mean(np.abs(stft_b)**2,axis=trial_axis)
    coherence = np.abs(cross_spec)/np.sqrt(a_power_spectrum*b_power_spectrum)
    return coherence

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
    dat.get_stft(recalculate = True, dat_type = ['phase'])
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


    intra_region_inds = [np.random.choice(x,2, replace=False) \
                            for x in region_electrodes]
    intra_region_phase = [phase_array[:,x].swapaxes(0,1) for x in intra_region_inds]
    intra_region_diff = [np.exp(1.j*(x[0]-x[1])) for x in intra_region_phase]

    min_err_phase = phase_array[:, np.array(select_region_electrodes)].swapaxes(0,1)
    phase_diff = np.exp(1.j*(min_err_phase[0] - min_err_phase[1]))
    
    bootstrap_samples = 500

    #coherence_boot_array = np.zeros(\
    #        (bootstrap_samples,*phase_diff.shape[1:])) 

    #for repeat in trange(bootstrap_samples):
    #    this_phase_diff = phase_diff[
    #                            np.random.choice(range(phase_diff.shape[0]),
    #                            phase_diff.shape[0], 
    #                            replace = True)]
    #    coherence_boot_array[repeat] = \
    #            np.abs(np.mean(this_phase_diff,axis=(0)))
    def gen_intra_diff_resamples(x):
        this_inds =  np.random.choice(range(phase_diff.shape[0]),
                                phase_diff.shape[0], 
                                replace = True)
        this_diff_list = np.stack([x[this_inds] for x in intra_region_diff])
        return np.abs(np.mean(this_diff_list,axis=(1)))

    def gen_diff_resamples(x):
        this_phase_diff = phase_diff[
                                np.random.choice(range(phase_diff.shape[0]),
                                phase_diff.shape[0], 
                                replace = True)]
        return np.abs(np.mean(this_phase_diff,axis=(0)))

    def gen_diff_shuffle_resamples(x):
        this_inds =  np.random.choice(range(phase_diff.shape[0]),
                                phase_diff.shape[0], 
                                replace = True)
        this_shuff_diff = np.exp(1.j*(min_err_phase[0][this_inds] - min_err_phase[1]))
        return np.abs(np.mean(this_shuff_diff,axis=(0)))

    stft_save_path = '/stft/analyses/phase_coherence'

    # Write out with calculation so array can be deleted to release memory
    diff_coherence_boot_array = np.stack(
            #[gen_diff_resamples(x) for x in tqdm(np.arange(bootstrap_samples))])
            parallelize(gen_diff_resamples, np.arange(bootstrap_samples)))
    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        remove_node(os.path.join(
                stft_save_path,'diff_phase_coherence_array'),hf5)
        hf5.create_array(stft_save_path, 'diff_phase_coherence_array', 
             diff_coherence_boot_array, createparents = True)
    del diff_coherence_boot_array

    diff_intra_coherence_boot = np.stack(
            #[gen_intra_diff_resamples(x) for x in tqdm(np.arange(bootstrap_samples))])
            parallelize(gen_intra_diff_resamples, np.arange(bootstrap_samples)))
    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        remove_node(os.path.join(
                stft_save_path,'diff_intra_phase_coherence_array'),hf5)
        hf5.create_array(stft_save_path, 'diff_intra_phase_coherence_array', 
             diff_intra_coherence_boot, createparents = True)
    del diff_intra_coherence_boot

    diff_coherence_shuffle_array = np.stack(
            #[gen_diff_shuffle_resamples(x) for x in tqdm(np.arange(bootstrap_samples))])
            parallelize(gen_diff_shuffle_resamples, np.arange(bootstrap_samples)))
    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        remove_node(os.path.join(
                stft_save_path,'diff_shuffle_phase_coherence_array'),hf5)
        hf5.create_array(stft_save_path, 'diff_shuffle_phase_coherence_array', 
             diff_coherence_shuffle_array, createparents = True)
    del diff_coherence_shuffle_array


    # Not using boot_stft_coherence
    ############################################################## 
    #selected_stft_array = stft_array[:,np.array(select_region_electrodes)]

    ## Resample to get confidence intervals and store percentiles for every timebin
    ##boot_stft_coherence = np.empty((resamples, *selected_stft_array.shape[2:]))
    ##for i in trange(resamples):

    #def gen_stft_resamples(x):
    #    trial_num = selected_stft_array.shape[0]
    #    inds = np.random.choice(np.arange(trial_num), trial_num)
    #    temp_stft_array = selected_stft_array[inds]
    #    #boot_stft_coherence[i] = calc_coherence(temp_stft_array[:,0],
    #    return calc_coherence(temp_stft_array[:,0],
    #                temp_stft_array[:,1], trial_axis = 0)

    #boot_stft_coherence = np.stack(
    #        parallelize(gen_stft_resamples, np.arange(bootstrap_samples)))

    ######################################## 
    ## Write out data 
    ######################################## 
    #stft_save_path = '/stft/analyses/phase_coherence'

    ## Write out final phase channels and channel numbers 
    #with tables.open_file(dat.hdf5_path,'r+') as hf5:
    #    #remove_node(os.path.join(
    #    #        stft_save_path,'stft_phase_coherence_array'),hf5)
    #    remove_node(os.path.join(
    #            stft_save_path,'diff_phase_coherence_array'),hf5)
    #    remove_node(os.path.join(
    #            stft_save_path,'diff_shuffle_phase_coherence_array'),hf5)
    #    remove_node(os.path.join(
    #            stft_save_path,'diff_intra_phase_coherence_array'),hf5)
    #    #hf5.create_array(stft_save_path, 'stft_phase_coherence_array', 
    #    #     boot_stft_coherence, createparents = True)
    #    hf5.create_array(stft_save_path, 'diff_phase_coherence_array', 
    #         diff_coherence_boot_array, createparents = True)
    #    hf5.create_array(stft_save_path, 'diff_shuffle_phase_coherence_array', 
    #         diff_coherence_shuffle_array, createparents = True)
    #    hf5.create_array(stft_save_path, 'diff_intra_phase_coherence_array', 
    #         diff_intra_coherence_boot, createparents = True)
