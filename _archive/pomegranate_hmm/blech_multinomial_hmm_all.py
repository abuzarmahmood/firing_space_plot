#***Duration and Lag on plot names are hardcoded - EDIT!!! ***

import matplotlib
matplotlib.use('Agg')

# Import stuff!
import numpy as np
import tables
import easygui
import sys
import os
from blech_hmm import *
import pylab as plt
import re

# The output models from pomegranate are in JSON format - pytables however doesn't allow strings
# to be saved to file. So we will map the string to bytes and save that to the file.
# Here are a couple of convenience functions for that.

def recordStringInHDF5(hf5, group, nodename, s):
    '''creates an Array object in an HDF5 file that represents a unicode string'''
    bytes = np.fromstring(s.encode('utf-8'), np.uint8)
    atom = tables.UInt8Atom()
    array = hf5.create_array(group, nodename, atom = atom, obj = bytes, shape=(len(bytes),))
    return array

def retrieveStringFromHDF5(node):
    return str(node.read(), 'utf-8')

# Read blech.dir, and cd to that directory
f = open('blech.dir', 'r')
dir_name = []
for line in f.readlines():
    dir_name.append(line)
f.close()
os.chdir(dir_name[0][:-1])

# Pull out the NSLOTS - number of CPUs allotted
#n_cpu = int(os.getenv('NSLOTS'))
if len(sys.argv) >= 2:
    n_cpu = int(sys.argv[1])
else:
    n_cpu = int(easygui.multenterbox('How many CPUs you want??',fields = ['# of CPUs'])[0])

# Get the names of all files in the current directory, and find the .params and hdf5 (.h5) file
file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
units_file = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
    if files[-10:] == 'hmm_params':
        params_file = files
    if files[-9:] == 'hmm_units':
        units_file = files

# Read the .hmm_params file
f = open(params_file, 'r')
params = []
for line in f.readlines():
    params.append(line)
f.close()

# Assign the params to variables
min_states = int(params[0])
max_states = int(params[1])
threshold = float(params[2])
seeds = int(params[3])
edge_inertia = float(params[4])
dist_inertia = float(params[5])
taste = int(params[6])
pre_stim = int(params[7])
bin_size = int(params[8])
pre_stim_hmm = int(params[9])
post_stim_hmm = int(params[10])

# Read the chosen units
f = open(units_file, 'r')
chosen_units = []
for line in f.readlines():
    chosen_units.append(int(line))
chosen_units = np.array(chosen_units)-1

# Open up hdf5 file
hf5 = tables.open_file(hdf5_name, 'r+')

# list all tastes
dig_in = hf5.list_nodes('/spike_trains')
dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]
dig_in = list(filter(None, dig_in))

## Number of trials per taste for indexing when concatenating spike trains
taste_n = [dig_in[i].spike_array[:].shape[0] for i in range(len(dig_in))]
if np.std(taste_n) == 0:
    taste_n = taste_n[0]
else:
    taste_n = int(easygui.multenterbox('How many trails per taste??',fields = ['# of trials'])[0])

# Get the spike array for each individual taste and merge all:
spikes = np.concatenate(tuple(dig_in[i].spike_array[:] for i in range(len(dig_in))), axis = 0)

# Slice out the required portion of the spike array, and bin it
spikes = spikes[:, chosen_units, pre_stim - pre_stim_hmm:pre_stim + post_stim_hmm]
binned_spikes = np.zeros((spikes.shape[0], int((pre_stim_hmm + post_stim_hmm)/bin_size)))
time = []
for i in range(spikes.shape[0]):
    time = []
    for k in range(0, spikes.shape[2], bin_size):
        time.append(k - pre_stim_hmm)
        n_firing_units = np.where(np.sum(spikes[i, :, k:k+bin_size], axis = 1) > 0)[0]
        if n_firing_units.size:
            n_firing_units = n_firing_units + 1
        else:
            n_firing_units = [0]
        binned_spikes[i, int(k/bin_size)] = np.random.choice(n_firing_units)

# ################
# ## All Trials ##
# ################
#
# # Implement a Multinomial HMM for no. of states defined by min_states and max_states - run all the trials through the HMM
# hmm_results = []
# all_trials = np.arange(binned_spikes.shape[0])
#
# for n_states in range(min_states, max_states + 1):
#     # Run the Multinomial HMM - skip it if it doesn't converge
#     try:
#         result = multinomial_hmm_implement(n_states, threshold, seeds, n_cpu, binned_spikes, all_trials, edge_inertia, dist_inertia)
#         hmm_results.append((n_states, result))
#     except:
#         continue
#
# Delete the multinomial_hmm_results node under /spike_trains/ if it exists
try:
	hf5.remove_node('/spike_trains/multinomial_hmm_results', recursive = True)
except:
	pass

# Then create the multinomial_hmm_results group
hf5.create_group('/spike_trains','multinomial_hmm_results')
# hf5.create_group('/spike_trains/multinomial_hmm_results','all_trials')
#
hf5.flush()
#
# Delete the Multinomial folder within HMM_plots if it exists for this taste
try:
    os.system("rm -r ./HMM_plots/Multinomial")
except:
    pass

# Make a folder for plots of Multinomial HMM analysis
os.mkdir("HMM_plots/Multinomial")
# os.mkdir("HMM_plots/Multinomial/all_trials")
#
# # Go through the HMM results, and make plots for each state and each trial
# for result in hmm_results:
#
#     # Make a directory for this number of states
#     os.mkdir("HMM_plots/Multinomial/all_trials/states_%i" % result[0])
#
#     # Make a group under multinomial_hmm_results for this number of states
#     hf5.create_group('/spike_trains/multinomial_hmm_results/all_trials','states_%i' % result[0])
#     # Write the emission and transition probabilties to this group
#     emission_labels = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'emission_labels', np.array(list(result[1][4][0].keys())))
#     emission_matrix = []
#     for i in range(len(result[1][4])):
#         emission_matrix.append(list(result[1][4][i].values()))
#     emission_probs = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'emission_probs', np.array(emission_matrix))
#     transition_probs = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'transition_probs', result[1][5])
#     posterior_proba = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'posterior_proba', result[1][6])
#
#     # Also write the json model string to file
#     #model_json = hf5.create_array('/spike_trains/dig_in_%i/poisson_hmm_results/laser/states_%i' % (taste, result[0]), 'model_json', result[1][0])
#     model_json = recordStringInHDF5(hf5, '/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'model_json', result[1][0])
#
#     # Write the log-likelihood, AIC/BIC score, and time vector to the hdf5 file too
#     log_prob = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'log_likelihood', np.array(result[1][1]))
#     aic = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'aic', np.array(result[1][2]))
#     bic = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'bic', np.array(result[1][3]))
#     time_vect = hf5.create_array('/spike_trains/multinomial_hmm_results/all_trials/states_%i' % result[0], 'time', time)
#     hf5.flush()
#
#     # Go through the trials in binned_spikes and plot the trial-wise posterior probabilities with the unit rasters
#     # First make a dictionary of colors for the rasters
#     raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
#     for i in range(binned_spikes.shape[0]):
#         fig = plt.figure()
#         for j in range(posterior_proba.shape[2]):
#             plt.plot(time, len(chosen_units)*posterior_proba[i, :, j])
#         for unit in range(len(chosen_units)):
#             # Determine the type of unit we are looking at - the color of the raster will depend on that
#             if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
#                 unit_type = 'regular_spiking'
#             elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
#                 unit_type = 'fast_spiking'
#             else:
#                 unit_type = 'multi_unit'
#             for j in range(spikes.shape[2]):
#                 if spikes[i, unit, j] > 0:
#                     plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
#         plt.xlabel('Time post stimulus (ms)')
#         plt.ylabel('Probability of HMM states')
#         plt.title('Trial %i' % (i+1) + '\n' + 'RSU: red, FS: blue, Multi: black')
#         fig.savefig('HMM_plots/Multinomial/all_trials/states_%i/Trial_%i.png' % (result[0], (i+1)))
#         plt.close("all")

################
## Off Trials ##
################

# Check if the laser_array exists - if it does, perform a 2nd round of HMM training on
# 1.) just the non-laser trials, 2.) just the laser trials

on_trials = np.concatenate(tuple(np.where(dig_in[i].laser_durations[:] > 0)[0] + taste_n*i for i in range(len(dig_in))))
off_trials = np.concatenate(tuple(np.where(dig_in[i].laser_durations[:] == 0)[0] + taste_n*i for i in range(len(dig_in))))

# Implement a Multinomial HMM for no. of states defined by min_states and max_states, on just the laser off trials
hmm_results = []
for n_states in range(min_states, max_states + 1):
    # Run Multinomial HMM - skip if it doesn't converge
    try:
        result = multinomial_hmm_implement(n_states, threshold, seeds, n_cpu, binned_spikes, off_trials, edge_inertia, dist_inertia)
        hmm_results.append((n_states, result))
    except:
        continue

# Delete the laser_off node under /spike_trains/dig_in_(taste)/multinomial_hmm_results/ if it exists
## On and off trials already deleted when multinomial_hmm_results node was removed for all trials - Abu ##

#try:
#    hf5.remove_group('/spike_trains/multinomial_hmm_results/laser_off', recursive = True)
#except:
#    pass

# Then create the laser_off node under the multinomial_hmm_results group
exec("hf5.create_group('/spike_trains/multinomial_hmm_results', 'laser_off')")
hf5.flush()

# Delete the laser_off folder within HMM_plots/Multinomial if it exists for this taste
try:
    os.system("rm -r ./HMM_plots/Multinomial/laser_off")
except:
    pass

# Make a folder for plots of Multinomial HMM analysis on laser off trials
os.mkdir("HMM_plots/Multinomial/laser_off")

# Go through the HMM results, and make plots for each state and each trial
for result in hmm_results:

    # Make a directory for this number of states
    os.mkdir("HMM_plots/Multinomial/laser_off/states_%i" % result[0])

    # Make a group under multinomial_hmm_results for this number of states
    hf5.create_group('/spike_trains/multinomial_hmm_results/laser_off', 'states_%i' % (result[0]))
    # Write the emission and transition probabilties to this group
    emission_labels = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'emission_labels', np.array(list(result[1][4][0].keys())))
    emission_matrix = []
    for i in range(len(result[1][4])):
        emission_matrix.append(list(result[1][4][i].values()))
    emission_probs = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'emission_probs', np.array(emission_matrix))
    transition_probs = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'transition_probs', result[1][5])
    posterior_proba = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'posterior_proba', result[1][6])

    # Also write the json model string to file
    #model_json = hf5.create_array('/spike_trains/dig_in_%i/poisson_hmm_results/laser/states_%i' % (taste, result[0]), 'model_json', result[1][0])
    model_json = recordStringInHDF5(hf5, '/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'model_json', result[1][0])

    # Write the log-likelihood and AIC/BIC score to the hdf5 file too
    log_prob = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'log_likelihood', np.array(result[1][1]))
    aic = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'aic', np.array(result[1][2]))
    bic = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'bic', np.array(result[1][3]))
    time_vect = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_off/states_%i' % result[0], 'time', time)
    hf5.flush()

    # Go through the trials in binned_spikes and plot the trial-wise posterior probabilities and raster plots
    # First make a dictionary of colors for the rasters
    raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
    for i in range(binned_spikes.shape[0]):
        if i in on_trials:
            label = 'laser_on_'
        else:
            label = 'laser_off_'
        fig = plt.figure()
        for j in range(posterior_proba.shape[2]):
            plt.plot(time, len(chosen_units)*posterior_proba[i, :, j])
        for unit in range(len(chosen_units)):
            # Determine the type of unit we are looking at - the color of the raster will depend on that
            if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
                unit_type = 'regular_spiking'
            elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
                unit_type = 'fast_spiking'
            else:
                unit_type = 'multi_unit'
            for j in range(spikes.shape[2]):
                if spikes[i, unit, j] > 0:
                    plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
        plt.xlabel('Time post stimulus (ms)')
        plt.ylabel('Probability of HMM states')

        #plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, dig_in.laser_durations, dig_in.laser_onset_lag) + '\n' + 'RSU: red, FS: blue, Multi: black')
        #Harcoded duration and lag here but fix later
        plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, 2500, 0) + '\n' + 'RSU: red, FS: blue, Multi: black')
        fig.savefig('HMM_plots/Multinomial/laser_off/states_%i/%sTrial_%i.png' % (result[0], label, (i+1)))
        plt.close("all")

###############
## On Trials ##
###############

# Implement a Multinomial HMM for no. of states defined by min_states and max_states, on just the laser on trials now
hmm_results = []
for n_states in range(min_states, max_states + 1):
    # Run Multinomial HMM - skip if it doesn't converge
    try:
        result = multinomial_hmm_implement(n_states, threshold, seeds, n_cpu, binned_spikes, on_trials, edge_inertia, dist_inertia)
        hmm_results.append((n_states, result))
    except:
        continue

# Delete the laser_on node under /spike_trains/dig_in_(taste)/multinomial_hmm_results/ if it exists
## On and off trials already deleted when multinomial_hmm_results node was removed for all trials - Abu ##

#try:
#    exec("hf5.remove_node('/spike_trains/multinomial_hmm_results/laser_on', recursive = True)")
#except:
#    pass

# Then create the laser_on node under the multinomial_hmm_results group
hf5.create_group('/spike_trains/multinomial_hmm_results', 'laser_on')
hf5.flush()

# Delete the laser_on folder within HMM_plots/Multinomial if it exists for this taste
try:
    os.system("rm -r ./HMM_plots/Multinomial/laser_on")
except:
    pass

# Make a folder for plots of Multinomial HMM analysis on laser on trials
os.mkdir("HMM_plots/Multinomial/laser_on")

# Go through the HMM results, and make plots for each state and each trial
for result in hmm_results:

    # Make a directory for this number of states
    os.mkdir("HMM_plots/Multinomial/laser_on/states_%i" %result[0])

    # Make a group under multinomial_hmm_results for this number of states
    hf5.create_group('/spike_trains/multinomial_hmm_results/laser_on', 'states_%i' % (result[0]))
    # Write the emission and transition probabilties to this group
    emission_labels = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' %result[0], 'emission_labels', np.array(list(result[1][4][0].keys())))
    emission_matrix = []
    for i in range(len(result[1][4])):
        emission_matrix.append(list(result[1][4][i].values()))
    emission_probs = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'emission_probs', np.array(emission_matrix))
    transition_probs = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'transition_probs', result[1][5])
    posterior_proba = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'posterior_proba', result[1][6])

    # Also write the json model string to file
    #model_json = hf5.create_array('/spike_trains/dig_in_%i/poisson_hmm_results/laser/states_%i' % (taste, result[0]), 'model_json', result[1][0])
    model_json = recordStringInHDF5(hf5, '/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'model_json', result[1][0])

    # Write the log-likelihood and AIC/BIC score to the hdf5 file too
    log_prob = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'log_likelihood', np.array(result[1][1]))
    aic = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'aic', np.array(result[1][2]))
    bic = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'bic', np.array(result[1][3]))
    time_vect = hf5.create_array('/spike_trains/multinomial_hmm_results/laser_on/states_%i' % result[0], 'time', time)
    hf5.flush()

    # Go through the trials in binned_spikes and plot the trial-wise posterior probabilities and raster plots
    # First make a dictionary of colors for the rasters
    raster_colors = {'regular_spiking': 'red', 'fast_spiking': 'blue', 'multi_unit': 'black'}
    for i in range(binned_spikes.shape[0]):
        if i in on_trials:
            label = 'laser_on_'
        else:
            label = 'laser_off_'
        fig = plt.figure()
        for j in range(posterior_proba.shape[2]):
            plt.plot(time, len(chosen_units)*posterior_proba[i, :, j])
        for unit in range(len(chosen_units)):
            # Determine the type of unit we are looking at - the color of the raster will depend on that
            if hf5.root.unit_descriptor[chosen_units[unit]]['regular_spiking'] == 1:
                unit_type = 'regular_spiking'
            elif hf5.root.unit_descriptor[chosen_units[unit]]['fast_spiking'] == 1:
                unit_type = 'fast_spiking'
            else:
                unit_type = 'multi_unit'
            for j in range(spikes.shape[2]):
                if spikes[i, unit, j] > 0:
                    plt.vlines(j - pre_stim_hmm, unit, unit + 0.5, color = raster_colors[unit_type], linewidth = 0.5)
        plt.xlabel('Time post stimulus (ms)')
        plt.ylabel('Probability of HMM states')
        #Harcoded duration and lag here but fix later
        plt.title('Trial %i, Dur: %ims, Lag:%ims' % (i+1, 2500,0) + '\n' + 'RSU: red, FS: blue, Multi: black')
        fig.savefig('HMM_plots/Multinomial/laser_on/states_%i/%sTrial_%i.png' % (result[0], label, (i+1)))
        plt.close("all")

hf5.close()
