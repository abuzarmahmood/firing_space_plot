######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
import glob
os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import raster
from hinton import hinton
os.chdir('/media/bigdata/pomegranate_hmm')
from blech_hmm_abu import *
from scipy.signal import convolve

# =============================================================================
# =============================================================================
dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    

 
file  = 0

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])

data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
# Manually confirm chosen units are being selected
data.get_data()
all_spikes_array = np.asarray(data.off_spikes)

time_inds = range(2000,4000)
bin_size = 1
all_spikes_array = all_spikes_array[:,:,:,time_inds]

# Bin spikes (might decrease info for fast spiking neurons)
binned_spikes = np.zeros((all_spikes_array.shape[0],all_spikes_array.shape[1], 
                          all_spikes_array.shape[2], int((time_inds[-1]+1 - time_inds[0])/bin_size)))
for i in range(all_spikes_array.shape[0]): # Loop over tastes
    for j in range(all_spikes_array.shape[1]): # Loop over neurons
        for k in range(all_spikes_array.shape[2]): # Loop over trials
            for l in range(all_spikes_array.shape[3]): # Loop over time
                if (np.sum(all_spikes_array[i, j, k, l*bin_size:(l+1)*bin_size]) > 0):
                    binned_spikes[i,j,k,l] = 1

# Remove multiple spikes in same time bin (for categorical HMM)
for i in range(binned_spikes.shape[0]): # Loop over tastes
    for j in range(binned_spikes.shape[2]): # Loop over trials
        for k in range(binned_spikes.shape[3]): # Loop over time
            n_firing_units = np.where(binned_spikes[i,:,j,k] > 0)[0]
            if len(n_firing_units)>0:
                binned_spikes[i,:,j,k] = 0
                binned_spikes[i,np.random.choice(n_firing_units),j,k] = 1

# Convert bernoulli trials to categorical data        
cat_binned_spikes = np.zeros((binned_spikes.shape[0],binned_spikes.shape[2],binned_spikes.shape[3]))
for i in range(cat_binned_spikes.shape[0]): # Loop over tastes
    for j in range(cat_binned_spikes.shape[1]): # Loop over trials
        for k in range(cat_binned_spikes.shape[2]): # Loop over time
            firing_unit = np.where(binned_spikes[i,:,j,k] > 0)[0]
            if firing_unit.size > 0:
                cat_binned_spikes[i,j,k] = firing_unit + 1
                
# =============================================================================
# Run HMM
# =============================================================================
min_states = 3
max_states = 3
threshold = 1e-4
seeds=10
n_cpu = 30
binned_spikes = cat_binned_spikes[0,:,:]
off_trials = np.arange(10)
edge_inertia, dist_inertia = 0,0

n_states = 3

model_json, \
log_prob, \
AIC, \
BIC, \
state_emissions, \
state_transitions, \
posterior_proba = multinomial_hmm_implement(n_states, 
                                            threshold, 
                                            seeds, 
                                            n_cpu, 
                                            binned_spikes, 
                                            off_trials, 
                                            edge_inertia, 
                                            dist_inertia)

trial = 4
raster(data=binned_spikes[trial,:],expected_latent_state=posterior_proba[trial,:,:].T)