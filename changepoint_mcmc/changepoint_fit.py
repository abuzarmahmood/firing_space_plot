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
import tables

import numpy as np
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

params_file_path = '/media/bigdata/firing_space_plot'\
        '/changepoint_mcmc/fit_params.json'

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('states', type = int, help = 'Number of States to fit')
parser.add_argument("--multiregion", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Fit separate models to each region")
parser.add_argument("--good", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Use only good neurons")
parser.add_argument("--simulate", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Generate simulates AND shuffled fits")

args = parser.parse_args()
data_dir = args.dir_name 
good_nrn_bool = args.good
simulate_bool = args.simulate
multi_region_bool = args.multiregion

#data_dir = '/media/fastdata/KM28_4tastes_laser_200408_145209'
#good_nrn_bool = False
#simulate_bool = False
#multi_region_bool = False
#states = 3

# If multi_region, for now prevent good_nrn indexing (not implemented),
# and simulated fits (not needed)
if multi_region_bool:
    good_nrn_bool = False
    simulate_bool = False

dat = ephys_data(data_dir)

#dat.firing_rate_params = dat.default_firing_params

dat.get_unit_descriptors()
dat.get_spikes()
dat.check_laser()
#dat.get_firing_rates()
dat.get_region_units()

if multi_region_bool and not len(dat.region_names)>1:
    raise Exception("Cannot fit separate models to recordings with single region")

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

if dat.laser_exists:

    # Nothing for now, these may be added later
    good_nrn_bool = False
    simulate_bool = False

    dat.separate_laser_spikes()
    taste_dat = [np.array(dat.off_spikes), np.array(dat.on_spikes)]
    laser_names = ['off','on']

else:
    taste_dat = dat.spikes
    laser_names = ['none']

for this_laser_name, this_taste_dat in zip(laser_names, taste_dat):

    this_taste_dat = np.array(this_taste_dat)
    # Convert to list so more esy to separate by neuron
    taste_dat_list = [this_taste_dat[:,:,these_units] \
            for these_units in dat.region_units]

    if good_nrn_bool:
        # Not used if multiregion or laser
        print("Attempting to use good neurons")

        with tables.open_file(dat.hdf5_path,'r') as h5:
            good_nrn_bool_list = h5.get_node('/', 'selected_changepoint_nrns')[:]

        good_nrn_inds = np.where(good_nrn_bool_list)[0]
        this_taste_dat = this_taste_dat[:,:,good_nrn_inds]
        taste_dat_list = [this_taste_dat]


    # Create dirs and names
    model_save_dir = changepoint.get_model_save_dir(data_dir, states)
    if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

    if good_nrn_bool:
        suffix = '_type_good'
    else:
        suffix = '_type_reg'

    suffix_list = [f'_region_{name}{suffix}' for name in dat.region_names]

    model_name_list = [changepoint.get_model_name(\
            states,fit,time_lims,bin_width, 'actual', this_laser_name) + suffix \
            for suffix in suffix_list]

    model_dump_path_list = \
            [changepoint.get_model_dump_path(model_name,model_save_dir) \
            for model_name in model_name_list]

    ##########
    # Bin Data
    ##########
    this_dat_binned = \
            np.sum(this_taste_dat[...,time_lims[0]:time_lims[1]].\
            reshape(*this_taste_dat.shape[:-1],-1,bin_width),axis=-1)
    this_dat_binned = np.vectorize(np.int)(this_dat_binned)

    # Separate out final fit data by region
    if good_nrn_bool:
        this_dat_binned_list = [this_dat_binned]
    else:
        this_dat_binned_list = [this_dat_binned[:,:,these_units] \
                for these_units in dat.region_units]

    ########################################
    # ___        __                              
    #|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
    # | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
    # | || | | |  _|  __/ | |  __/ | | | (_|  __/
    #|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
    ########################################
    for num in range(len(this_dat_binned_list)):
        if not os.path.exists(model_dump_path_list[num]):
            model = changepoint.create_changepoint_model(
                        spike_array = this_dat_binned_list[num],
                        states = states,
                        fit = fit,
                        samples = samples,
                        changepoint_prior = changepoint_prior)
            
            # If the unnecessarily detailed model name exists
            # It will be loaded without running the inference
            # Otherwise model will be fit and saved

            changepoint.run_inference(model = model, 
                                        fit = fit, 
                                        samples = samples, 
                                        unbinned_array = taste_dat_list[num], 
                                        model_save_dir = model_save_dir, 
                                        model_name = model_name_list[num])

################################################################################
################################################################################

################################################
# ____  _                 _       _           _ 
#/ ___|(_)_ __ ___  _   _| | __ _| |_ ___  __| |
#\___ \| | '_ ` _ \| | | | |/ _` | __/ _ \/ _` |
# ___) | | | | | | | |_| | | (_| | ||  __/ (_| |
#|____/|_|_| |_| |_|\__,_|_|\__,_|\__\___|\__,_|
################################################                                               

if simulate_bool:

    print("Attempting simulated and shuffled fits")
    model_name_list = [changepoint.get_model_name(\
                            states,fit,time_lims,bin_width,this_type) \
                            for this_type in ['shuffle','simulate']]
    model_name_list = [x+suffix for x in model_name_list]

    model_dump_path_list =[\
            changepoint.get_model_dump_path(this_model_name,model_save_dir)
            for this_model_name in model_name_list]

    ##################################################
    ## Create shuffled data
    ##################################################
    # Shuffle neurons across trials FOR SAME TASTE
    shuffled_dat = np.array([np.random.permutation(neuron) \
                for neuron in np.swapaxes(taste_dat,2,0)])
    shuffled_dat = np.swapaxes(shuffled_dat,0,2)

    ##########
    # Bin Data
    ##########
    shuffled_dat_binned = \
            np.sum(shuffled_dat[...,time_lims[0]:time_lims[1]].\
            reshape(*shuffled_dat.shape[:-1],-1,bin_width),axis=-1)
    shuffled_dat_binned = np.vectorize(np.int)(shuffled_dat_binned)

    ##################################################
    ## Create simulated data 
    ##################################################
    # Inhomogeneous poisson process using mean firing rates

    mean_firing = np.mean(taste_dat,axis=1)

    # Simulate spikes
    simulated_spike_array = np.array(\
            [np.random.random(mean_firing.shape) < mean_firing \
            for trial in range(shuffled_dat_binned.shape[1])])*1
    simulated_spike_array = simulated_spike_array.swapaxes(0,1)
    simulated_dat_binned = \
            np.sum(simulated_spike_array[...,time_lims[0]:time_lims[1]].\
            reshape(*simulated_spike_array.shape[:-1],-1,bin_width),axis=-1)
    simulated_dat_binned = np.vectorize(np.int)(simulated_dat_binned)

    ########################################
    # ___        __                              
    #|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
    # | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
    # | || | | |  _|  __/ | |  __/ | | | (_|  __/
    #|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
    ########################################
    #if not all([os.path.exists(x) for x in model_dump_path_list]):

    model_kwargs = {'states':states,'fit':fit,
            'samples':samples, 'changepoint_prior' : changepoint_prior}
    if not os.path.exists(model_dump_path_list[0]):
        shuffle_model = changepoint.create_changepoint_model(\
                        spike_array = shuffled_dat_binned, **model_kwargs)
        changepoint.run_inference(model = shuffle_model, 
                                    fit = fit, 
                                    samples = samples, 
                                    unbinned_array = shuffled_dat, 
                                    model_save_dir = model_save_dir, 
                                    model_name = model_name_list[0])

    if not os.path.exists(model_dump_path_list[1]):
        simulate_model = changepoint.create_changepoint_model(\
                        spike_array = simulated_dat_binned, **model_kwargs)
        changepoint.run_inference(model = simulate_model, 
                                    fit = fit, 
                                    samples = samples, 
                                    unbinned_array = simulated_spike_array,
                                    model_save_dir = model_save_dir, 
                                    model_name = model_name_list[1])
