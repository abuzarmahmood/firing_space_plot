## Import required modules
import os
import matplotlib.pyplot as plt
import tables
import numpy as np
from tqdm import tqdm, trange
import shutil
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
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

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

##################################################
## Read in data 
##################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 

coherence_save_path = '/stft/analyses/phase_coherence/phase_coherence_array'

# Write out final phase channels and channel numbers 
coherence_list = []

for this_file in tqdm(file_list):
    with tables.open_file(this_file,'r') as hf5:
        coherence_list.append(hf5.get_node(coherence_save_path)[:])

coherence_array = np.stack(coherence_list)

# Get time and freq_vecs
with tables.open_file(this_file,'r') as hf5:
    time_vec = hf5.get_node('/stft/time_vec')[:]
    freq_vec = hf5.get_node('/stft/freq_vec')[:]

####################################### 
# Difference from baseline
####################################### 
# Pool baseline coherence and conduct tests on non-overlapping bins
## ** time_vec is already defined **
baseline_t = 2 #ms
baseline_range = np.array((1250,1750))/1000
baseline_inds = np.where(
        np.logical_and(time_vec>baseline_range[0], time_vec<baseline_range[1]))[0]

baseline_dat = coherence_array[...,baseline_inds].swapaxes(1,2)

ci_interval = [2.5, 97.5]
ci_array = np.empty((*baseline_dat.shape[:2] , len(ci_interval)))
inds = list(np.ndindex(ci_array.shape[:-1]))
for this_ind in inds:
    ci_array[this_ind] = np.percentile(baseline_dat[this_ind], ci_interval)

# Find when mean value deviates from baseline bounds
mean_coherence_array = coherence_array.mean(axis=1)
dev_array = np.logical_or(
        mean_coherence_array < ci_array[...,0][...,np.newaxis],
        mean_coherence_array > ci_array[...,1][...,np.newaxis])
