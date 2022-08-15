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
dir_list = ['/media/bigdata/brads_data/BS28_4Tastes_180801_112138']#['/media/bigdata/jian_you_data/des_ic']
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
bin_size = 10
all_spikes_array = all_spikes_array[:,:,:,time_inds]

# =============================================================================
# # Calculate probability of neuronal firing in sliding windows for every neuron
# # To make sure they're obeying binomial statistics
# long_spikes_array = all_spikes_array[0,:,:,:]
# for taste in range(1,all_spikes_array.shape[0]):
#     long_spikes_array = np.concatenate((long_spikes_array,all_spikes_array[taste,:,:,:]),axis=1)
# v_long_spikes_array = long_spikes_array[:,0,:]
# for trial in range(1,long_spikes_array.shape[1]):
#     v_long_spikes_array = np.concatenate((v_long_spikes_array,long_spikes_array[:,trial,:]),axis=-1)
#     
# bin_width = 1000
# box_kern = np.ones((bin_width))/bin_width
# 
# firing_p = np.zeros((v_long_spikes_array.shape[0],v_long_spikes_array.shape[1]+bin_width-1))
# for nrn in range(v_long_spikes_array.shape[0]):
#     firing_p[nrn,:] = convolve(v_long_spikes_array[nrn,:],box_kern)
# =============================================================================

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
# # Unroll cat_binned_spikes so all tastes can be fed into HMM
# all_cat_spikes = cat_binned_spikes[0,:,:]
# for taste in range(1,cat_binned_spikes.shape[0]):
#     all_cat_spikes = np.concatenate((all_cat_spikes,cat_binned_spikes[taste,:,:]))
# =============================================================================
                
seed_num = 100
k_fold = 3

trial_labels = np.asarray([0]*30)

# Fit HMM and cross validate
min_states = 2
max_states = 2

for taste in range(4):
    
    data = cat_binned_spikes[taste,:,:]
    
    for state_val in tqdm(range(max_states-min_states+1)):
        
        n_states = range(min_states,max_states+1)[state_val]
        
        plot_dir = data_dir + '/' + 'hmm_plots' + '/' + 'taste_%i_states_%i' % (taste,n_states) #'/media/bigdata/pomegranate_hmm/plots'
        if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
    
        all_models, all_log_probs, all_accuracies =  multinomial_hmm_cross_validated_implement(
                                                    n_states = n_states, 
                                                    threshold = 1e-6, 
                                                    seeds = seed_num, 
                                                    k_fold = k_fold,
                                                    n_cpu = mp.cpu_count(), 
                                                    binned_spikes = data, 
                                                    trial_labels = trial_labels,
                                                    edge_inertia = 0, 
                                                    dist_inertia = 0)
    
        # =============================================================================
        # plt.subplot(211)
        # plt.errorbar(x = np.arange(min_states,max_states+1), y = np.mean(accuracies_array,axis=1),
        #              yerr = np.std(accuracies_array,axis=1))
        # plt.subplot(212)
        # plt.errorbar(x = np.arange(min_states,max_states+1), y = np.mean(probs_array,axis=1),
        #              yerr = np.std(probs_array,axis=1))
        # 
        # fig, ax = plt.subplots()
        # ax.scatter(all_log_probs, all_accuracies)
        # for i, txt in enumerate(range(len(all_models))):
        #     ax.annotate(txt, (all_log_probs[i], all_accuracies[i]))
        # =============================================================================
        
        model, log_prob, accuracy = all_models[np.argmax(all_log_probs)]
        
        # Set up things to return the parameters of the model - the state emission dicts and transition matrix 
        state_emissions = []
        state_transitions = model.dense_transition_matrix() # This definitely does not give log probability
        state_transitions = state_transitions[np.arange(n_states),:]
        state_transitions =state_transitions[:,np.arange(n_states)]
        
        for i in range(n_states):
            state_emissions.append(model.states[i].distribution.parameters[0])
        
        state_emissions_array = np.empty((len(state_emissions[0]),len(state_emissions)))
        for nrn in range(state_emissions_array.shape[0]):
            for state in range(state_emissions_array.shape[1]):
                state_emissions_array[nrn,state] = state_emissions[state][nrn]
        
        # Get the posterior probability sequence to return
        posterior_proba = np.zeros((data.shape[0], data.shape[1], n_states))
        for i in range(data.shape[0]):
            c, d = model.forward_backward(data[i, :])
            posterior_proba[i, :, :] = np.exp(d)
        
        fig = plt.figure(); hinton(state_transitions); plt.savefig(plot_dir + '/' + 'transitions_pom_%ist.png' % (n_states));plt.close(fig)
        fig = plt.figure(); hinton(state_emissions_array.T); plt.savefig(plot_dir + '/' + 'emissions_pom_%ist.png' % (n_states)); plt.close(fig)
        mean_probs = np.empty((len(np.unique(trial_labels)),n_states,data.shape[1]))
        
        mean_probs = np.mean(posterior_proba,axis=0).T
        
        fig = plt.figure()
        plt.plot(mean_probs.T)
        plt.savefig(plot_dir + '/' + 'mean_probs_pom_%ist.png' % (n_states))
        plt.close(fig)
        
        # Output results
        for i in range(data.shape[0]):
            fig = plt.figure()
            raster(data[i,:],expected_latent_state=posterior_proba[i,:,:].T)
            plt.savefig(plot_dir + '/' + '%i_pom_%ist.png' % (i,n_states))
            plt.close(fig)
