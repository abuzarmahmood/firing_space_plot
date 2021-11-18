"""
PyMC3 Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
- Changepoint distributions are shared across all tastes
- Each taste has it's own emission matrix
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
import random
import numpy as np
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

params_file_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/fit_params.json'

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('states', type = int, help = 'Number of States to fit')
args = parser.parse_args()
data_dir = args.dir_name 
states = int(args.states)

# data_dir = '/media/bigdata/Abuzar_Data/AM34/AM34_4Tastes_201217_114556/'
# states = 4


dat = ephys_data(data_dir)

#dat.firing_rate_params = dat.default_firing_params

dat.get_unit_descriptors()
dat.get_spikes()
#dat.get_firing_rates()
dat.default_stft_params['max_freq'] = 50
taste_dat = np.array(dat.spikes)

##########
# PARAMS 
##########

with open(params_file_path, 'r') as file:
    params_dict = json.load(file)

for key,val in params_dict.items():
    globals()[key] = val

#time_lims = [2000,4000]
#bin_width = 50
#fit = 40000
#samples = 20000


##########
# Bin Data
##########
this_dat_binned = \
        np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
        reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = np.vectorize(np.int)(this_dat_binned)

##########
# Split Data, Set Model, Run Inference
##########
data_size = [len(a) for a in this_dat_binned]
split_size = 2
iter_size = 10
#split_ind = [] #array of indices for each split
dat_binned_list = []
split_ind = np.array_split(np.random.permutation(np.arange(this_dat_binned.shape[2])),
        split_size,axis=-1)

#remaining_ind =list(np.arange(data_size[0]))
#for i in range(split_size-1):
#        sample_ind = random.sample(remaining_ind,int(data_size[0]/split_size))
#        split_ind.append(sample_ind)
#        remaining_ind = np.setdiff1d(sample_ind,remaining_ind)
#split_ind.append(remaining_ind)

for iter_num in range(iter_size):
    for i in range(split_size):
        # Create model dirs and names
        model_save_dir = changepoint.get_model_save_dir(data_dir, states)
        model_name = changepoint.get_model_name(states,fit,time_lims,bin_width,'split_'+str(i)+'_'+str(iter_num))
        model_dump_path = changepoint.get_model_dump_path(model_name,model_save_dir)


        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        this_dat_binned_split = this_dat_binned[:,:,split_ind[i]]
        unbinned_dat_split = taste_dat[:,:,split_ind[i]]
        dat_binned_list.append(this_dat_binned_split)

        #Create changepoint model
        if os.path.exists(model_dump_path): #REMOVE ME LATER
            os.remove(model_dump_path)


        if not os.path.exists(model_dump_path):
            model = changepoint.create_changepoint_model(
                spike_array = this_dat_binned_split,
                states = states,
                fit = fit,
                samples = samples)

        # If the unnecessarily detailed model name exists
        # It will be loaded without running the inference
        # Otherwise model will be fit and saved
        changepoint.run_inference(model,fit,samples,unbinned_dat_split,model_save_dir, model_name)

