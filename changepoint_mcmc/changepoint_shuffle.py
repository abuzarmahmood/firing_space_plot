"""
Fitting trial-shuffled and simulated datasets
"""

########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import os
import sys
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from scipy.stats import percentileofscore
import pickle
import argparse

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc')
from ephys_data import ephys_data
import visualize
from poisson_all_tastes_changepoint_model \
        import create_changepoint_model, run_inference

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('states', type = int, help = 'Number of States to fit')
args = parser.parse_args()
data_dir = args.dir_name 

#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'
#states = 4

#plot_super_dir = os.path.join(data_dir,'changepoint_plots')
#if not os.path.exists(plot_super_dir):
#        os.makedirs(plot_super_dir)

dat = ephys_data(data_dir)

dat.firing_rate_params = dat.default_firing_params.copy()
#dat.firing_rate_params['type'] = 'baks'

dat.get_unit_descriptors()
dat.get_spikes()
#dat.get_firing_rates()
#dat.default_stft_params['max_freq'] = 50
taste_dat = np.array(dat.spikes)

##########
# PARAMS 
##########
time_lims = [1500,4000]
bin_width = 10
#states = 4
states = int(args.states)
fit = 40000
samples = 20000

# Create dirs and names
model_save_dir = os.path.join(data_dir,'saved_models',f'vi_{states}_states')
if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
model_name = f'vi_{states}_states_{fit}fit_'\
        f'time{time_lims[0]}_{time_lims[1]}_bin{bin_width}'
shuffle_model_name = 'shuffle_' + model_name
simulate_model_name = 'simulate_' + model_name
model_name_list = [shuffle_model_name,simulate_model_name]

#for this_model_name in model_name_list:
#    plot_dir = os.path.join(plot_super_dir,this_model)
#    if not os.path.exists(plot_dir):
#            os.makedirs(plot_dir)


##########
# Bin Data
##########
t_vec = np.arange(taste_dat.shape[-1])
binned_t_vec = np.min(t_vec[time_lims[0]:time_lims[1]].\
                    reshape((-1,bin_width)),axis=-1)
this_dat_binned = \
        np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
        reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = np.vectorize(np.int)(this_dat_binned)

##################################################
## Create shuffled data
##################################################
# Shuffle neurons across trials FOR SAME TASTE
shuffled_dat_binned = np.array([np.random.permutation(neuron) \
            for neuron in np.swapaxes(this_dat_binned,2,0)])
shuffled_dat_binned = np.swapaxes(shuffled_dat_binned,0,2)

#def plot_raster(matrix):
#    plt.scatter(*np.where(matrix)[::-1], marker = '|')
#
#nrn = 1
#taste = 0
#plot_raster(this_dat_binned[taste,:,nrn])
#plt.figure()
#plot_raster(shuffled_dat_binned[taste,:,nrn])
#plt.show()
#
#trial = 0
#plot_raster(this_dat_binned[taste,trial])
#plt.figure()
#plot_raster(shuffled_dat_binned[taste,trial])
#plt.show()
#
#mean_firing = np.mean(this_dat_binned,axis=1)
#mean_shuffled_firing = np.mean(shuffled_dat_binned,axis=1)
#
#visualize.firing_overview(mean_firing)
#plt.figure()
#visualize.firing_overview(mean_shuffled_firing)
#plt.show()

##################################################
## Create simulated data 
##################################################
# Inhomogeneous poisson process using BAKS firing rates
# Lower resolution BAKS used for speed
# Interpolate to dt = 1ms, generate spikes and bin as done for actual data

#mean_firing = np.mean(dat.firing_array, axis = 2)
#new_x = np.linspace(0,1,dat.spikes[0].shape[-1])
#current_x = np.linspace(0,1,mean_firing.shape[-1])
#interp_mean_firing = np.empty((*mean_firing.shape[:-1],len(new_x)))
#for taste_num, taste in enumerate(mean_firing):
#    for nrn_num, nrn in enumerate(taste):
#        interp_mean_firing[taste_num,nrn_num] = np.interp(new_x,
#                                                            current_x, nrn)

spike_array = np.array(dat.spikes)
interp_mean_firing = np.mean(spike_array,axis=2)

# Simulate spikes
simulated_spike_array = np.array(\
        [np.random.random(interp_mean_firing.shape) < \
        interp_mean_firing*dat.firing_rate_params['baks_dt'] \
        for trial in range(this_dat_binned.shape[1])])*1
simulated_spike_array = simulated_spike_array.swapaxes(0,1)
simulated_dat_binned = \
        np.sum(simulated_spike_array[...,time_lims[0]:time_lims[1]].\
        reshape(*simulated_spike_array.shape[:-1],-1,bin_width),axis=-1)
simulated_dat_binned = np.vectorize(np.int)(simulated_dat_binned)


#nrn = 20
#taste = 0
#plot_raster(spike_array[taste,:,nrn])
#plt.figure()
#plot_raster(simulated_spike_array[taste,:,nrn])
#plt.show()
#
#trial = 15
#plot_raster(spike_array[taste,trial])
#plt.figure()
#plot_raster(simulated_spike_array[taste,trial])
#plt.show()

#nrn = 20
#taste = 0
#plot_raster(this_dat_binned[taste,:,nrn])
#plt.figure()
#plot_raster(simulated_dat_binned[taste,:,nrn])
#plt.show()
#
#trial = 15
#plot_raster(this_dat_binned[taste,trial])
#plt.figure()
#plot_raster(simulated_dat_binned[taste,trial])
#plt.show()


########################################
# ___        __                              
#|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
# | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
# | || | | |  _|  __/ | |  __/ | | | (_|  __/
#|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
########################################

model_dump_path_list =[\
        os.path.join(model_save_dir,f'dump_{this_model_name}.pkl')\
        for this_model_name in model_name_list]

if not all([os.path.exists(x) for x in model_dump_path_list]):
    model_kwargs = {'states':states,'fit':fit,'samples':samples}
    model_list = [\
            create_changepoint_model(\
                    spike_array = this_data, **model_kwargs) \
                    for this_data in \
                    [shuffled_dat_binned, simulated_dat_binned]]
                
    
    for this_model,this_model_name in zip(model_list, model_name_list):
        run_inference(this_model, fit, samples, model_save_dir, this_model_name)
