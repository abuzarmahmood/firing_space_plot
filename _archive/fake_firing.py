#  ______    _          _____        _        
# |  ____|  | |        |  __ \      | |       
# | |__ __ _| | _____  | |  | | __ _| |_ __ _ 
# |  __/ _` | |/ / _ \ | |  | |/ _` | __/ _` |
# | | | (_| |   <  __/ | |__| | (_| | || (_| |
# |_|  \__,_|_|\_\___| |_____/ \__,_|\__\__,_|
#

# Import stuff
import numpy as np
import pylab as plt
import copy
import tables
import csv
import seaborn as sns
import pandas as pd
    
#############
# BERNOULLI #
#############                                       
# Bernoulli trials with arbitrary numbers of states and transitions (upto constraints)
# Firing probability random for each neuron (upto a ceiling)
# Transitions random with some min durations
# Jitter between individual neuron state transtiions
    
def fake_ber_firing(nrns,
                    trials,
                    length,
                    state_order,
                    palatability_state,
                    ceil_p,
                    jitter_t,
                    jitter_p,
                    jitter_p_type,
                    min_duration):
    """
    nrns = number of neurons (emissions)
    trials = number of trials
    length = number of bins for data
    num_states = number of distinct states generated
    state_order = LIST: fixed order for states
    palatability_state = integer indicating which state to be taken as palatability
            Used to change firing rate to produce palatability correlations
    ceil_p = maximum firing probability
    jitter_t = max amount of jitter between neurons for a state transition (IN BINS)
    jitter_p = fraction of jitter in probability between trials of each neuron
    jitter_p_type = abs : random noise with max value jitter_p
                    scaled: random noise scaled by mean firing probability of that neuron
    min_duration =    time between start and first transition
                      time between final state and end
                      time between intermediate state transitions
    """
    
    # Returns data array, transition times, emission probabilities
    all_data = []
    all_t = [] # Transition times for 4 tastes -> generated randomly
    mean_p = [] # Firing probabilities for 4 tastes -> scaled by palatability
    num_states = len(np.unique(state_order))   
    # Emission probabilities of neurons for every state
    p = np.random.rand(num_states, nrns)*ceil_p # states x neuron
    taste_scaling_value = np.random.rand(nrns)*2
    for taste in range(4):
        this_p = copy.deepcopy(p)
        for nrn in range(this_p.shape[1]):
            this_p[palatability_state,nrn] =  this_p[palatability_state,nrn]*np.linspace(1,taste_scaling_value[nrn],4)[taste]
        mean_p.append(this_p)
        
    all_p = np.empty((4,trials,nrns,len(state_order)))
    for taste in range(4):
        for trial in range(trials):
            for neuron in range(nrns):
                this_trial_p = mean_p[taste]
                if jitter_p_type in 'scaled':
                    this_trial_p_jitter = this_trial_p*jitter_p*np.random.uniform(-1,1,(this_trial_p.shape))
                elif jitter_p_type in 'absolute':
                    this_trial_p_jitter = jitter_p*np.random.uniform(-1,1,(this_trial_p.shape))
                
            fin_this_trial_p = this_trial_p + this_trial_p_jitter
            all_p[taste,trial,:] = np.swapaxes(fin_this_trial_p,0,1)
    all_p = np.abs(all_p)
            
    # Transition times for every transition over every trial
    for taste in range(4):
        t = np.zeros((trials, len(state_order)-1)) # trials x num of transitions (2) 
        for trial in range(t.shape[0]):
            first_trans, last_trans, middle_trans = [1,1,1]
            while (first_trans or last_trans or middle_trans):
                t[trial,:] = (np.random.rand(1,t.shape[1]) * length)
                
                first_trans = (t[trial,0] < min_duration) # Make sure first transition is after min_duration
                last_trans = (t[trial,-1] + min_duration > length) # Make sure last transition is min_duration before the end
                middle_trans = np.sum(t[trial,1:] - t[trial,0:-1] < min_duration)  # Make sure there is a distance of min_duration between all intermediate transitions
           
        print(taste)
        
        t = np.repeat(t[:, :, np.newaxis], nrns, axis=2) # trials x num of transitions x neurons
        t = t + np.random.uniform(-1,1,t.shape)*jitter_t # Add jitter to individual neuron transitions
        t = t.astype('int')
        all_t.append(t)
    
    # For every trial, for every neuron, walk through time
    # If time has passed a transition, update transition count and use transitions count
    # to index from the appropriate state in state order
    
    
    for taste in range(4):
        data = np.zeros((nrns, trials, length)) # neurons x trials x time
        for trial in range(data.shape[1]):
            for neuron in range(data.shape[0]):
                
                trans_count = 0 # To keep track of transition
                
                for time in range(data.shape[2]):
                    try:
                        if time < all_t[taste][trial,trans_count, neuron]:
                            data[neuron, trial, time] = np.random.binomial(1, all_p[taste, trial, neuron, trans_count])
                        else:
                            trans_count += 1
                            data[neuron, trial, time] = np.random.binomial(1, all_p[taste, trial, neuron, trans_count])
                    except: # Lazy programming -_-
                        if trans_count >= all_t[taste].shape[1]:
                            data[neuron, trial, time] = np.random.binomial(1, all_p[taste, trial, neuron, trans_count])
        all_data.append(data)
        
    return all_data, all_t, mean_p, all_p, taste_scaling_value

###############
# CATEGORICAL #
###############
# Approximation to categorical data since I'm lazy
# Take data from bernoulli trials and convert to categorical
def fake_cat_firing(nrns,
                    trials,
                    length,
                    state_order,
                    palatability_state,
                    ceil_p,
                    jitter_t,
                    jitter_p,
                    jitter_p_type,
                    min_duration):
    """
    Converts data from fake_ber_firing into a categorical format
    PARAMS:
    : nrns = number of neurons (emissions)
    : trials = number of trials
    : length = number of bins for data
    : state_order = LIST: fixed order for states
    : ceil_p = maximum firing probability
    : jitter_t = max amount of jitter between neurons for a state transition
    : min_duration =    time between start and first transition
                           time between final state and end
                           time between intermediate state transitions
    """
    
    # Returns data array, transition times, emission probabilities
    ber_spikes, t, p, all_p, taste_scaling = fake_ber_firing(nrns,
                                trials,
                                length,
                                state_order,
                                palatability_state,
                                ceil_p,
                                jitter_t,
                                jitter_p,
                                jitter_p_type,
                                min_duration)
    
    ber_spikes = [np.swapaxes(this_ber_spikes,0,1) for this_ber_spikes in ber_spikes]
    
    # Remove multiple spikes in same time bin (for categorical HMM)
    for taste in range(len(ber_spikes)): # Loop over tastes
        for i in range(trials): # Loop over trials
            for k in range(length): # Loop over time
                n_firing_units = np.where(ber_spikes[taste][i,:,k] > 0)[0]
                if len(n_firing_units)>0:
                    ber_spikes[taste][i,:,k] = 0
                    ber_spikes[taste][i,np.random.choice(n_firing_units),k] = 1
    
    # Convert bernoulli trials to categorical data  
    all_cat_spikes = []
    for taste in range(len(ber_spikes)):
        cat_binned_spikes = np.zeros((ber_spikes[0].shape[0],ber_spikes[0].shape[2]))
        for i in range(cat_binned_spikes.shape[0]):
            for j in range(cat_binned_spikes.shape[1]):
                firing_unit = np.where(ber_spikes[taste][i,:,j] > 0)[0]
                if firing_unit.size > 0:
                    cat_binned_spikes[i,j] = firing_unit + 1
        all_cat_spikes.append(cat_binned_spikes)
        
    return all_cat_spikes, t, p, all_p, taste_scaling
    

def make_fake_file(filename, 
                   nrns,
                    trials,
                    length,
                    state_order,
                    palatability_state,
                    ceil_p,
                    jitter_t,
                    jitter_p,
                    jitter_p_type,
                    min_duration,
                    data_type = 'cat'):
    """
    Creates an HDF5 with fake bernoulli data
    """
    if data_type is 'ber':
        data, t, p, all_p, scaling = fake_ber_firing(
                            nrns,
                            trials,
                            length,
                            state_order,
                            palatability_state,
                            ceil_p,
                            jitter_t,
                            jitter_p,
                            jitter_p_type,
                            min_duration)
    elif data_type is 'cat':
        data, t, p, all_p, scaling = fake_cat_firing(
                            nrns,
                            trials,
                            length,
                            state_order,
                            palatability_state,
                            ceil_p,
                            jitter_t,
                            jitter_p,
                            jitter_p_type,
                            min_duration)        
    
    params = {
            'nrns' : nrns,
            'trials' : trials,
            'length' : length,
            'state_order' : state_order,
            'palatability_state' : palatability_state,
            'ceil_p' : ceil_p,
            'jitter_t' : jitter_t,
            'jitter_p' : jitter_p,
            'jitter_p_type' : jitter_p_type,
            'min_duration' : min_duration
            #'scaling' : scaling
            }
    
    with open(filename + 'params.csv','w') as f:
        w = csv.writer(f)
        w.writerows(params.items())
        
    hf5 = tables.open_file(filename + '.h5', mode = 'w', title = 'Fake Data')
    
    hf5.create_array('/', 'scaling', scaling, createparents=True)
# =============================================================================
#     hf5.create_array('/', 'spike_array', data, createparents=True)
#     hf5.create_array('/', 'transition_times', t, createparents=True)
#     hf5.create_array('/', 'probability_values', p, createparents=True)
# =============================================================================
    for taste in range(len(data)):
        hf5.create_array('/spike_trains/dig_in_%i' % (taste), 'spike_array', data[taste], createparents=True)
        hf5.create_array('/spike_trains/dig_in_%i' % (taste), 'transition_times', t[taste], createparents=True)
        hf5.create_array('/spike_trains/dig_in_%i' % (taste), 'probability_values', p[taste], createparents=True)
    
    
    hf5.flush()
    hf5.close()
    
    return data, t, p, all_p, scaling

def prob_plot(all_p):
    """
    4D array: Taste x Trials x Nrns x State
    Generates swarmplots with trials as dots, different states colored and different nrns
    along the x axis
    Different tastes are placed in different subplots
    """
    count = 0
    all_p_frame = pd.DataFrame()
    for taste in range(all_p.shape[0]):
        for trial in range(all_p.shape[1]):
            for nrn in range(all_p.shape[2]):
                for state in range(all_p.shape[3]):
                    this_point = {'taste' : taste,
                                  'trial' : trial,
                                  'neuron' : nrn,
                                  'state' : state,
                                  'firing_p' : all_p[taste,trial,nrn,state]}
                    all_p_frame = pd.concat([all_p_frame, pd.DataFrame(this_point,index=[count])])
                    count += 1
    
# =============================================================================
#     fig, ax = plt.subplots(nrows = all_p.shape[0], ncols = 1, sharey = True)
#     for taste in range(all_p.shape[0]):
#         #plt.subplot(all_p.shape[0],1,taste+1)
#         plt.sca(ax[taste])
#         sns.boxplot(data = all_p_frame.query('taste == %i' % taste), x = 'neuron', y = 'firing_p',hue = 'state',dodge=True)
#         
# =============================================================================
    count = 0
    nrows = np.int(np.floor(np.sqrt(all_p.shape[2])))
    ncols = np.int(np.ceil(all_p.shape[2]/nrows))
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, sharey = True)
    for rows in range(nrows):
        for cols in range(ncols):
            if count >= all_p.shape[2]:
                break
            plt.sca(ax[rows,cols])
            #ax = plt.gca()
            #ax.legend_ = None
            #plt.draw()
            #ax[rows,cols].legend().set_visible(False)
            this_plot = sns.boxplot(data = all_p_frame.query('neuron == %i' % count), 
                        x = 'state', y = 'firing_p',hue = 'taste',
                        dodge=True)
            this_plot.legend_ = None
            plt.draw()
            count += 1
            
    return fig
# Raster plot
def raster(data,trans_times=None,expected_latent_state=None,line_length = 0.5):
    #If bernoulli data, take three 2D arrays: 
        # data : neurons x time
        # trans_times : num_transition x neurons
        # expected_latent_state: states x time
    # If categorical data, take one 1D array
        # data : time (where each element indicates which neuron fired)
    # Red lines indicate mean transition times
    # Yellow ticks indicate individual neuron transition times
    
    # Bernoulli data
    if data.ndim > 1:       
        # Plot spikes 
        for unit in range(data.shape[0]):
            for time in range(data.shape[1]):
                if data[unit, time] > 0:
                    plt.vlines(time, unit, unit + line_length, linewidth = 0.5)
        # Plot state probability
        if expected_latent_state is not None:
            plt.plot(expected_latent_state.T*data.shape[0])
            
        # Plot mean transition times         
        if trans_times is not None:
            mean_trans = np.mean(trans_times, axis = 1)
            for transition in range(len(mean_trans)):
                plt.vlines(mean_trans[transition], 0, data.shape[0],colors = 'r', linewidth = 1)
        
        # Plot individual neuron transition times
        if trans_times is not None:
            for unit in range(data.shape[0]):
                for transition in range(trans_times.shape[0]):
                    plt.vlines(trans_times[transition,unit], unit, unit + line_length, linewidth = 3, color = 'y')
    
    # Categorical Data
    else:
        # Plot spikes
       for time in range(data.shape[0]):
           if data[time] > 0:
               plt.vlines(time, data[time], data[time] + line_length, linewidth = 0.5)
               
        # Plot state probabilities
       if expected_latent_state is not None:
            plt.plot(expected_latent_state.T*np.unique(data).size)
            
        # Plot mean transition times
       if trans_times is not None:
            mean_trans = np.mean(trans_times, axis = 1)
            for transition in range(len(mean_trans)):
                plt.vlines(mean_trans[transition], 0, np.unique(data).size,colors = 'r', linewidth = 1)
        
        # Plot individual neuron transition times
       if trans_times is not None:
            for unit in range(trans_times.shape[1]):
                for transition in range(trans_times.shape[0]):
                    plt.vlines(trans_times[transition,unit], unit, unit + line_length, linewidth = 3, color = 'y')
        
            
    plt.xlabel('Time post stimulus (ms)')
    plt.ylabel('Neuron')

# Dot Raster plot
# Same as rater but with different shape of points
def dot_raster(data,trans_times=None,expected_latent_state=None,markersize = 5,line_length=0.5):
    #If bernoulli data, take three 2D arrays: 
        # data : neurons x time
        # trans_times : num_transition x neurons
        # expected_latent_state: states x time
    # If categorical data, take one 1D array
        # data : time (where each element indicates which neuron fired)
    # Red lines indicate mean transition times
    # Yellow ticks indicate individual neuron transition times
    
    # Bernoulli data
    if data.ndim > 1:       
        # Plot spikes 
        for unit in range(data.shape[0]):
            for time in range(data.shape[1]):
                if data[unit, time] > 0:
                    plt.plot(time, unit, 'ko',markersize=markersize)
        
        # Plot state probability
        if expected_latent_state is not None:
            plt.plot(expected_latent_state.T*data.shape[0])
            
        # Plot mean transition times         
        if trans_times is not None:
            mean_trans = np.mean(trans_times, axis = 1)
            for transition in range(len(mean_trans)):
                plt.vlines(mean_trans[transition], 0, data.shape[0],colors = 'r', linewidth = 1)
        
        # Plot individual neuron transition times
        if trans_times is not None:
            for unit in range(data.shape[0]):
                for transition in range(trans_times.shape[0]):
                    plt.vlines(trans_times[transition,unit], unit, unit + line_length, linewidth = 3, color = 'y')
    
    # Categorical Data
    # Plot spikes
    else:
        for time in range(data.shape[0]):
           if data[time] > 0:
               plt.plot(time, data[time], 'ko',markersize=markersize)
               
        # Plot state probabilities
        if expected_latent_state is not None:
            plt.plot(expected_latent_state.T*np.unique(data).size)
            
        # Plot mean transition times
        if trans_times is not None:
            mean_trans = np.mean(trans_times, axis = 1)
            for transition in range(len(mean_trans)):
                plt.vlines(mean_trans[transition], 0, np.unique(data).size,colors = 'r', linewidth = 1)
        
        # Plot individual neuron transition times
        if trans_times is not None:
            for unit in range(trans_times.shape[1]):
                for transition in range(trans_times.shape[0]):
                    plt.vlines(trans_times[transition,unit], unit, unit + line_length, linewidth = 3, color = 'y')
        
            
    plt.xlabel('Time post stimulus (ms)')
    plt.ylabel('Neuron')
      