# Import stuff
from pomegranate import *
import numpy as np
import multiprocessing as mp
import math
from scipy.spatial import distance_matrix as dist_mat
from scipy.spatial.distance import cdist

def multinomial_hmm_generate(n_states, 
                             threshold, 
                             binned_spikes, 
                             seed,
                             edge_inertia, 
                             dist_inertia):
    """
    Simply fits a pomegranate multinomial HMM and returns the model object as output
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Make a pomegranate HiddenMarkovModel object
    model = HiddenMarkovModel('%i' % seed) 
    states = []
    # Make a pomegranate Discrete distribution object with emissions = range(n_units + 1) - 1 for each state
    n_units = int(np.max(binned_spikes))
    for i in range(n_states):
        dist_dict = {}
        prob_list = np.random.random(n_units + 1)
        prob_list = prob_list/np.sum(prob_list)
        for unit in range(n_units + 1):
            dist_dict[unit] = prob_list[unit]    
        states.append(State(DiscreteDistribution(dist_dict), name = 'State%i' % (i+1)))

    model.add_states(states)
    # Add transitions from model.start to each state (equal probabilties)
    for state in states:
        model.add_transition(model.start, state, float(1.0/len(states)))

    # Add transitions between the states - 0.95-0.999 is the probability of not transitioning in every state
    for i in range(n_states):
        not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
        for j in range(n_states):
            if i==j:
                model.add_transition(states[i], states[j], not_transitioning_prob)
            else:
                model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))

    # Bake the model
    model.bake()

    # Train the model only on the trials indicated by off_trials
    model.fit(binned_spikes, algorithm = 'baum-welch', stop_threshold = threshold, edge_inertia = edge_inertia, distribution_inertia = dist_inertia, verbose = False)

    return model

# This implements a wrong idea about cross-validation
# There's no point in averaging the likelihood of DIFFERENT models and taking the LAST ONE *facepalm*
# =============================================================================
# def multinomial_hmm_cross_validated(n_states, 
#                              threshold, 
#                              binned_spikes, 
#                              seed,
#                              k_fold,
#                              edge_inertia, 
#                              dist_inertia):
#     """
#     Creates a multinomial hmm model and calculates sum of log probabilities according
#     to the level of cross validation specified
#     """
#     all_test_trials = []
#     all_train_trials = []
#     cross_validated_prob = []
#     data = np.random.permutation(binned_spikes) # Will permute along 1st axis
#     for k in range(1,k_fold+1):
#         test_trials = np.arange(np.int(data.shape[0]*(1-(k/k_fold))),np.int(data.shape[0]*(1-((k-1)/k_fold))))
#         train_trials = np.asarray([x for x in np.arange(data.shape[0]) if x not in test_trials])
#         all_test_trials.append(test_trials)
#         all_train_trials.append(train_trials)
#         test_data = data[test_trials,:]
#         train_data = data[train_trials,:]
# 
#         model = multinomial_hmm_generate(
#                                 n_states = n_states, 
#                                 threshold = threshold, 
#                                 binned_spikes = train_data, 
#                                 seed = seed,
#                                 edge_inertia = edge_inertia, 
#                                 dist_inertia = dist_inertia)
#     
#     cross_validated_prob.append(np.sum([model.log_probability(test_data[i, :]) for i in range(len(test_trials))]))
#     
#     return model, sum(cross_validated_prob), all_test_trials, all_train_trials
# =============================================================================

def hmm_classification_accuracy(data,
                            trial_labels,
                            model):
    """
    Given an HMM over multiple tastes, calculates classification accuracy using a Naive Bayes classifier
    """
    trial_labels = np.asarray(trial_labels)
    n_states = model.state_count()-2
    posterior_proba = np.zeros((data.shape[0], data.shape[1], n_states))
    for i in range(data.shape[0]):
        c, d = model.forward_backward(data[i, :])
        posterior_proba[i, :, :] = np.exp(d)
    
    mean_probs = np.empty((len(np.unique(trial_labels)),n_states,data.shape[1]))
    for taste in range(len(np.unique(trial_labels))):
        mean_probs[taste,:,:] = np.mean(posterior_proba[trial_labels==np.unique(trial_labels)[taste],:,:],axis=0).T
        
    taste_dists = np.empty((len(np.unique(trial_labels)),data.shape[0]))
    for trial in range(data.shape[0]):
        for taste in range(mean_probs.shape[0]):
            taste_dists[taste,trial] = np.sum(np.diag(cdist(posterior_proba[trial,:,:],mean_probs[taste].T)))
            
    accuracy = np.mean(np.unique(trial_labels)[np.argmin(taste_dists,axis=0)] == trial_labels)
    
    return accuracy

def multinomial_hmm_cross_validated(n_states, 
                             threshold, 
                             binned_spikes, 
                         trial_labels,
                             seed,
                             k_fold,
                             edge_inertia, 
                             dist_inertia):
    """
    Creates a multinomial hmm model and calculates sum of log probabilities according
    to the level of cross validation specified
    """
    test_trials = []
    train_trials = []
    test_trials = np.random.choice(np.arange(binned_spikes.shape[0]),np.int(binned_spikes.shape[0]/k_fold))
    train_trials = np.asarray([x for x in np.arange(binned_spikes.shape[0]) if x not in test_trials])
    test_data = binned_spikes[test_trials,:]
    train_data = binned_spikes[train_trials,:]

    model = multinomial_hmm_generate(
                            n_states = n_states, 
                            threshold = threshold, 
                            binned_spikes = train_data, 
                            seed = seed,
                            edge_inertia = edge_inertia, 
                            dist_inertia = dist_inertia)
    
    cross_validated_prob = np.sum([model.log_probability(test_data[i, :]) for i in range(len(test_trials))])
    
    accuracy = hmm_classification_accuracy(
                                        data = binned_spikes[test_trials],
                                        trial_labels = trial_labels[test_trials],
                                        model = model)
    
    return model, cross_validated_prob, accuracy
    
def multinomial_hmm_cross_validated_implement(
                                            n_states, 
                                            threshold,
                                            binned_spikes,
                                            trial_labels,
                                            seeds, 
                                            k_fold,
                                            n_cpu, 
                                            edge_inertia, 
                                            dist_inertia):
                                                
    # Create a pool of asynchronous n_cpu processes running multinomial_hmm() - no. of processes equal to seeds
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(multinomial_hmm_cross_validated, args = (
                                                            n_states, 
                                                            threshold, 
                                                            binned_spikes, 
                                                            trial_labels,
                                                            seed,
                                                            k_fold, 
                                                            edge_inertia, 
                                                            dist_inertia)) for seed in range(seeds)]
    output = [p.get() for p in results]

    # Find the process that ended up with the highest log likelihood, and return it as the solution. If several processes ended up with the highest log likelihood, just pick the earliest one
    log_probs = [output[i][1] for i in range(len(output))]
    accuracies = [output[i][2] for i in range(len(output))]
    #maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    
    return output, log_probs, accuracies

def multinomial_hmm(n_states, threshold, binned_spikes, seed, off_trials, edge_inertia, dist_inertia):

    # Seed the random number generator
    np.random.seed(seed)

    # Make a pomegranate HiddenMarkovModel object
    model = HiddenMarkovModel('%i' % seed) 
    states = []
    # Make a pomegranate Discrete distribution object with emissions = range(n_units + 1) - 1 for each state
    n_units = int(np.max(binned_spikes))
    for i in range(n_states):
        dist_dict = {}
        prob_list = np.random.random(n_units + 1)
        prob_list = prob_list/np.sum(prob_list)
        for unit in range(n_units + 1):
            dist_dict[unit] = prob_list[unit]    
        states.append(State(DiscreteDistribution(dist_dict), name = 'State%i' % (i+1)))

    model.add_states(states)
    # Add transitions from model.start to each state (equal probabilties)
    for state in states:
        model.add_transition(model.start, state, float(1.0/len(states)))

    # Add transitions between the states - 0.95-0.999 is the probability of not transitioning in every state
    for i in range(n_states):
        not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
        for j in range(n_states):
            if i==j:
                model.add_transition(states[i], states[j], not_transitioning_prob)
            else:
                model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))

    # Bake the model
    model.bake()

    # Train the model only on the trials indicated by off_trials
    model.fit(binned_spikes[off_trials, :], algorithm = 'baum-welch', stop_threshold = threshold, edge_inertia = edge_inertia, distribution_inertia = dist_inertia, verbose = False)
    log_prob = [model.log_probability(binned_spikes[i, :]) for i in off_trials]
    log_prob = np.sum(log_prob)

    # Set up things to return the parameters of the model - the state emission dicts and transition matrix 
    state_emissions = []
    state_transitions = np.exp(model.dense_transition_matrix())
    for i in range(n_states):
        state_emissions.append(model.states[i].distribution.parameters[0])
    
    # Get the posterior probability sequence to return
    posterior_proba = np.zeros((binned_spikes.shape[0], binned_spikes.shape[1], n_states))
    for i in range(binned_spikes.shape[0]):
        c, d = model.forward_backward(binned_spikes[i, :])
        posterior_proba[i, :, :] = np.exp(d)

    # Get the json representation of the model - will be needed if we need to reload the model anytime
    model_json = model.to_json()
    
    return model_json, log_prob, 2*((n_states)**2 + n_states*(n_units + 1)) - 2*log_prob, (np.log(len(off_trials)*binned_spikes.shape[1]))*((n_states)**2 + n_states*(n_units + 1)) - 2*log_prob, state_emissions, state_transitions, posterior_proba    

def multinomial_hmm_implement(n_states, threshold, seeds, n_cpu, binned_spikes, off_trials, edge_inertia, dist_inertia):

    # Create a pool of asynchronous n_cpu processes running multinomial_hmm() - no. of processes equal to seeds
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(multinomial_hmm, args = (n_states, threshold, binned_spikes, seed, off_trials, edge_inertia, dist_inertia,)) for seed in range(seeds)]
    output = [p.get() for p in results]

    # Find the process that ended up with the highest log likelihood, and return it as the solution. If several processes ended up with the highest log likelihood, just pick the earliest one
    log_probs = [output[i][1] for i in range(len(output))]
    maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    return output[maximum_pos]

# =============================================================================
# Gaussian Shit...tread carefully
# =============================================================================

def mutlivariate_gaussian_hmm_implement(n_states, threshold, seeds, n_cpu, firing_rates):

    # Create a pool of asynchronous n_cpu processes running multinomial_hmm() - no. of processes equal to seeds
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(mutlivariate_gaussian_hmm, args = (n_states, threshold, firing_rates, seed)) for seed in range(seeds)]
    output = [p.get() for p in results]

    # Find the process that ended up with the highest log likelihood, and return it as the solution. If several processes ended up with the highest log likelihood, just pick the earliest one
    log_probs = [output[i][1] for i in range(len(output))]
    maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    return output[maximum_pos]

def mutlivariate_gaussian_hmm(n_states, threshold, firing_rates, seed):
    """
    firing_rates : trials x time x neurons
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Make a pomegranate HiddenMarkovModel object
    model = HiddenMarkovModel('%i' % seed) 
    states = []
    # Make a pomegranate Multivariate Gaussian distribution object with emissions = range(n_units + 1) - 1 for each state
    n_units = firing_rates.shape[-1]
    for i in range(n_states):
        # Random initial conditions
        means = np.random.random((1,n_units))[0]
        dat_points = n_units**2
        covar_mat = np.cov(np.random.random((n_units,dat_points)))
        states.append(State(MultivariateGaussianDistribution(means,covar_mat), name = 'State%i' % (i+1)))

    model.add_states(states)
    # Add transitions from model.start to each state (equal probabilties)
    for state in states:
        model.add_transition(model.start, state, float(1.0/len(states)))

    # Add transitions between the states - 0.95-0.999 is the probability of not transitioning in every state
    for i in range(n_states):
        not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
        for j in range(n_states):
            if i==j:
                model.add_transition(states[i], states[j], not_transitioning_prob)
            else:
                model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))

    # Bake the model
    model.bake()

    # Train the model only on the trials indicated by off_trials
    model.fit(firing_rates, algorithm = 'baum-welch', stop_threshold = threshold, verbose = False)
    log_prob = [model.log_probability(firing_rates[i,:,:]) for i in range(firing_rates.shape[0])]
    log_prob = np.sum(log_prob)


    # Extract estimated emission probabilities
    state_emissions = []
    for i in range(n_states):
        state_emissions.append(model.states[i].distribution.parameters[0])
    state_emissions = np.asarray(state_emissions)
        
    # Extract estimated covariances
    state_covars = []
    for i in range(n_states):
        state_covars.append(model.states[i].distribution.parameters[1])
    state_covars = np.asarray(state_covars)
    
    # Set up things to return the parameters of the model - the state emission dicts and transition matrix 
    
    state_transitions = np.exp(model.dense_transition_matrix())[:n_states,:n_states]

    # Get the posterior probability sequence to return
    posterior_proba = np.zeros((firing_rates.shape[0], firing_rates.shape[1], n_states))
    for i in range(firing_rates.shape[0]):
        c, d = model.forward_backward(firing_rates[i, :, :])
        posterior_proba[i, :, :] = np.exp(d)
    
    return log_prob, state_emissions, state_covars, state_transitions, posterior_proba

# =============================================================================
# Independent Bernoulli emissions
# =============================================================================
# Breaks with trials lengths ~1000
    
def independent_bernoulli_hmm_implement(n_states, threshold, seeds, n_cpu, spikes, max_iters = 2e3):

    # Create a pool of asynchronous n_cpu processes running multinomial_hmm() - no. of processes equal to seeds
    pool = mp.Pool(processes = n_cpu)
    results = [pool.apply_async(independent_bernoulli_hmm, args = (n_states, threshold, spikes, seed, max_iters)) for seed in range(seeds)]
    output = [p.get() for p in results]

    # Find the process that ended up with the highest log likelihood, and return it as the solution. If several processes ended up with the highest log likelihood, just pick the earliest one
    log_probs = [output[i][1] for i in range(len(output))]
    maximum_pos = np.where(log_probs == np.max(log_probs))[0][0]
    return output[maximum_pos]

def independent_bernoulli_hmm(n_states, threshold, spikes, seed, max_iters = 2e3):
    """
    firing_rates : trials x time x neurons
    """
    # Seed the random number generator
    np.random.seed(seed)

    # Make a pomegranate HiddenMarkovModel object
    model = HiddenMarkovModel('%i' % seed) 
    states = []
    # Make a pomegranate Multivariate Independent Benoulli distribution object with emissions = range(n_units + 1) - 1 for each state
    
    for i in range(n_states):
        dists = [BernoulliDistribution(np.random.rand()) for x in range(spikes.shape[-1])]
        states.append(State(IndependentComponentsDistribution(dists), name = 'State%i' % (i+1)))
    
    model.add_states(states)
    # Add transitions from model.start to each state (equal probabilties)
    for state in states:
        model.add_transition(model.start, state, float(1.0/len(states)))
    
    # Add transitions between the states - 0.95-0.999 is the probability of not transitioning in every state
    for i in range(n_states):
        not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
        for j in range(n_states):
            if i==j:
                model.add_transition(states[i], states[j], not_transitioning_prob)
            else:
                model.add_transition(states[i], states[j], float((1.0 - not_transitioning_prob)/(n_states - 1)))
    
    # Bake the model
    model.bake()

    # Train the model only on the trials indicated by off_trials
    model.fit(spikes, algorithm = 'baum-welch', stop_threshold = threshold, verbose = True, max_iterations = max_iters)
    log_prob = [model.log_probability(spikes[i,:,:]) for i in range(spikes.shape[0])]
    log_prob = np.sum(log_prob)


    # Extract estimated emission probabilities
    state_emissions = np.zeros((n_states,spikes.shape[-1]))
    for i in range(n_states):
        for j in range(spikes.shape[-1]):
            state_emissions[i,j] = model.states[i].distribution.parameters[0][j].parameters[0]
        
    # Set up things to return the parameters of the model - the state emission dicts and transition matrix 
    
    state_transitions = np.exp(model.dense_transition_matrix())[:n_states,:n_states]

    # Get the posterior probability sequence to return
    posterior_proba = np.zeros((spikes.shape[0], spikes.shape[1], n_states))
    for i in range(spikes.shape[0]):
        c, d = model.forward_backward(spikes[i, :, :])
        posterior_proba[i, :, :] = np.exp(d)
    
    return log_prob, state_emissions, state_transitions, posterior_proba
