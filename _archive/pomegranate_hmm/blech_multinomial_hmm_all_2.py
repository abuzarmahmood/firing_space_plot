# Import stuff!
import numpy as np
import os
import pylab as plt

os.chdir('/media/bigdata/pomegranate_hmm')
from blech_hmm import *

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp
import glob

# Find the number of cpus or define them
n_cpu = mp.cpu_count();

# Open file and get spikes
dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

file  = 0

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data.get_data()

all_spikes_array = np.asarray(data.off_spikes)


# For now, enter paramters manually
# Assign the params to variables
min_states = 3
max_states = 7
threshold = 1e-6
seeds = 100
edge_inertia = 0
dist_inertia = 0

# Only need to bin data and remove multiple spikes from single bins... HMM code turns it into categorical itself
taste = 0

################
## Off Trials ##
################

off_trials = np.asarray(range(binned_spikes.shape[2]))

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