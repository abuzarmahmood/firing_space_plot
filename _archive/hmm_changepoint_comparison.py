"""
Compare changepoint prediction with Narendra's Variational Catergorical HMM
as run on JY's files

Test:
    1) How well to transitions correlate
    2) Magnitude of firing rate changes predicted by both models
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
import pymc3 as pm
import theano.tensor as tt
import json
from glob import glob

import tables
import numpy as np
import pylab as plt
import pickle
import argparse

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc')
from ephys_data import ephys_data
import visualize
import poisson_all_tastes_changepoint_model as changepoint 

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

data_dirs = ['/media/NotVeryBig/for_you_to_play/file3',
                '/media/NotVeryBig/for_you_to_play/file4']


##########
# PARAMS 
##########
params_file_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/fit_params.json'

states = 4#int(args.states)

with open(params_file_path, 'r') as file:
    params_dict = json.load(file)

for key,val in params_dict.items():
    globals()[key] = val

# Create dirs and names
conditions = ['on','off']
model_save_dirs = [changepoint.get_model_save_dir(x, states) for x in data_dirs]
model_names = [changepoint.get_model_name(states,fit,time_lims,bin_width,this_str) \
                for this_str in  conditions]
model_dump_paths = [[changepoint.get_model_dump_path(this_model_name,this_save_dir) \
         for this_model_name in model_names] for this_save_dir in model_save_dirs]

for this_save_dir in model_save_dirs:
    if not os.path.exists(this_save_dir):
            os.makedirs(this_save_dir)

##########
# Get and Bin Data
##########

import itertools as it

iters = list(it.product(range(len(model_dump_paths)), conditions))

#condition = 'on'
#data_num = 0

for data_num, condition in iters:

    this_model_name = [x for x in model_names if condition in x][0]
    this_model_dump_path = [x for x in model_dump_paths[data_num] \
                        if condition in x][0]
    this_model_save_dir = model_save_dirs[data_num]

    #temp_data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201228_124547'
    #dat = ephys_data(temp_data_dir)

    dat = ephys_data(data_dirs[data_num])
    dat.get_unit_descriptors()
    dat.get_spikes()
    dat.separate_laser_spikes()

    # Pull out unit numbers used to fit HMM
    hmm_units = open(os.path.join(dat.data_dir,'blech.hmm_units'),'r').\
            readlines()
    hmm_units = np.array([int(x[:-1]) for x in hmm_units])

    this_spikes = getattr(dat,condition + '_spikes')
    this_spikes = this_spikes[:,:,hmm_units]
    this_spikes = this_spikes[...,time_lims[0]:time_lims[1]]

    this_dat_binned = \
            np.sum(this_spikes.\
            reshape(*this_spikes.shape[:-1],-1,bin_width),axis=-1)
    this_dat_binned = np.vectorize(np.int)(this_dat_binned)

    ########################################
    # ___        __                              
    #|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
    # | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
    # | || | | |  _|  __/ | |  __/ | | | (_|  __/
    #|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
    ########################################
    if not os.path.exists(this_model_dump_path):
        model = changepoint.create_changepoint_model(
                    spike_array = this_dat_binned,
                    states = states,
                    fit = fit,
                    samples = samples)
        
        # If the unnecessarily detailed model name exists
        # It will be loaded without running the inference
        # Otherwise model will be fit and saved

        changepoint.run_inference(model, fit, samples, this_spikes,
                this_model_save_dir, this_model_name)

    else:
        print('Model already exists')

    ########################################
    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    #########################################                       
    # Load outputs from inference
    def load_tau(model_path):
        if os.path.exists(model_path):
            print('Trace loaded from cache')
            with open(model_path, 'rb') as buff:
                data = pickle.load(buff)
            tau_samples = data['tau']
            # Remove pickled data to conserve memory
            del data
        return tau_samples

    from scipy import stats

    tau_samples = load_tau(this_model_dump_path)
    tau_samples_scaled = tau_samples*params_dict['bin_width']
    tau_samples_scaled = np.swapaxes(tau_samples_scaled,0,1)
    taste_tau_samples = np.reshape(tau_samples_scaled, 
            (this_spikes.shape[0],-1, *tau_samples_scaled.shape[1:]))
    int_taste_tau = np.vectorize(int)(taste_tau_samples)
    mode_tau = np.squeeze(stats.mode(int_taste_tau,axis=2)[0])

    laser_durations_array = np.array(dat.laser_durations)

    #taste = 0
    for taste in range(len(this_spikes)):
        # Plot changepoint distributions in same directory as HMM plots 
        plot_dir = os.path.join(dat.data_dir, 'variational_HMM_plots' ,\
                f'dig_in_{taste}', 'Categorical', 
                f'laser_{condition}', f'states_{states}')  

        this_plot_spikes = this_spikes[taste]
        this_tau = mode_tau[taste]

        # Pull out HMM posterior probabilities to incorporate into plot
        hmm_prob_path = os.path.join('/spike_trains',f'dig_in_{taste}',
                'categorical_vb_hmm_results',f'laser_{condition}',
                f'states_{states}','posterior_proba_VB')

        with tables.open_file(dat.hdf5_path,'r') as h5:
            this_hmm_probs = h5.get_node(hmm_prob_path)[:]

        #trial_num = 0
        if condition == 'on':
            actual_trial_nums = np.where(laser_durations_array[taste]>0)[0] 
        else:
            actual_trial_nums = np.where(laser_durations_array[taste]==0)[0] 

        for trial_num in range(this_spikes.shape[1]):
            this_ax = visualize.raster(None, this_plot_spikes[trial_num], 
                                        marker = "|")
            this_ax.vlines(this_tau[trial_num], 
                        ymin = 0 - 0.5 , ymax = this_plot_spikes.shape[1] - 0.5,
                        color = 'red', alpha = 0.5, linewidth = 2)
            fig = plt.gcf()
            fig.savefig(os.path.join(\
                    plot_dir, f'Trial_{actual_trial_nums[trial_num]+1}_change'))
            plt.close(fig)
            #plt.show()

            bins = 100
            fig,ax = plt.subplots(3,1, figsize = (5,10), sharex = True)
            ax[0] = visualize.raster(ax[0], this_plot_spikes[trial_num], 
                                            marker = "|")
            ax[0].plot(this_hmm_probs[:,trial_num].T * this_plot_spikes.shape[1])
            ax[1] = visualize.raster(ax[1], this_plot_spikes[trial_num], 
                                            marker = "|")
            ax[1].vlines(this_tau[trial_num], 
                        ymin = 0 - 0.5 , ymax = this_plot_spikes.shape[1] - 0.5,
                        color = 'red', alpha = 0.5, linewidth = 2)
            for transition_num in range(taste_tau_samples.shape[-1]):
                ax[2].hist(taste_tau_samples[taste,trial_num,:,transition_num], 
                                                bins = bins)
                ax[2].set_xlim(np.array(time_lims) - time_lims[0])
            fig.savefig(os.path.join(\
                    plot_dir, f'Trial_{actual_trial_nums[trial_num]+1}_all'))
            plt.close(fig)
            #plt.show()
