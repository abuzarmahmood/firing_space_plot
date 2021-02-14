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

#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'
#states = 4


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
states = int(args.states)

with open(params_file_path, 'r') as file:
    params_dict = json.load(file)

for key,val in params_dict.items():
    globals()[key] = val

#time_lims = [1500,4000]
#bin_width = 10
#fit = 40000
#samples = 20000

# Create dirs and names
model_save_dir = changepoint.get_model_save_dir(data_dir, states)
#model_save_dir = os.path.join(data_dir,'saved_models',f'vi_{states}_states')
#model_name = f'vi_{states}_states_{fit}fit_'\
#        f'time{time_lims[0]}_{time_lims[1]}_bin{bin_width}'
model_name = changepoint.get_model_name(states,fit,time_lims,bin_width,'actual')
#model_dump_path = os.path.join(model_save_dir,f'dump_{model_name}.pkl')
model_dump_path = changepoint.get_model_dump_path(model_name,model_save_dir)

if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

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

########################################
# ___        __                              
#|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
# | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
# | || | | |  _|  __/ | |  __/ | | | (_|  __/
#|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
########################################
if not os.path.exists(model_dump_path):
    model = changepoint.create_changepoint_model(
                spike_array = this_dat_binned,
                states = states,
                fit = fit,
                samples = samples)
    
# If the unnecessarily detailed model name exists
# It will be loaded without running the inference
# Otherwise model will be fit and saved

changepoint.run_inference(model, fit, samples, model_save_dir, model_name)
