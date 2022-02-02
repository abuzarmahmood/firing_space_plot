"""
Test poisson HMM with spiking data
"""
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   

from pomegranate import *
import numpy as np
import pylab as plt
from scipy.stats import zscore
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from visualize import *
os.chdir('/media/bigdata/firing_space_plot/_old')
from fake_firing import fake_ber_firing

# Define functions

def plot_image(array):
    plt.imshow(array, interpolation='nearest',aspect='auto',origin='lower',cmap='viridis')

def raster(array):
    """
    Array is actually 2D
    """
    spike_times = np.where(array)
    plt.scatter(spike_times[1],spike_times[0], alpha = 0.1)

def random_Poisson_HMM(seed, n_states, dims, lambdas = None):
    """
    lambdas :: List of initial starting guess for lambdas
    """
    np.random.seed(seed)
    model = HiddenMarkovModel('{}'.format(seed)) 
    # Allowance to have a wide range of lambdas
    if lambdas is None:
        lambdas = [np.random.random(dims)*10 for state in range(n_states)] 
    states = [State(
                IndependentComponentsDistribution(
                    [PoissonDistribution(x) for x in lambdas[state]]),\
            name = 'State{}'.format(state+1)) for state in range(n_states)]
    model.add_states(states)

    # Add transitions from model.start to each state (equal probabilties)
    for state in states:
        model.add_transition(model.start, state, float(1.0/len(states)))

    # Generate initial transition matrix
    initial_trans_mat = np.eye(n_states)
    initial_trans_mat += np.random.random(initial_trans_mat.shape)*0.05
    initial_trans_mat /= np.sum(initial_trans_mat,axis=1)

    for (i,j),prob in np.ndenumerate(initial_trans_mat):
        model.add_transition(states[i], states[j],prob)

    # Bake the model
    model.bake()

    return model

def extract_emission_matrix(model):
    emission_mat = np.array(
                [[param.parameters[0] for param in state.distribution.parameters[0]] \
                        for state in model.states[:-2]])
    return emission_mat

def calculate_total_ll(model,data):
    total_ll = np.sum([model.log_probability(x) for x in data])
    return total_ll

def calculate_BIC(total_ll, n_states, dims, data):
    """
    See Fontanini Nature Neuroscience 2016 Methods
    BIC = -2LL + M( M+N-1 )ln T
    LL = Model log likelihood
    M = num states
    N = num_neurons (or dims)
    T = total observations (trials x bins/trial)
    """
    BIC = -2*total_ll + n_states*(n_states + data.shape[1] - 1)*np.log(np.prod(data.shape[1:]))
    return BIC

def train_random_Poisson_HMM(   data, 
                                n_states, 
                                dims, 
                                seed,
                                lambdas = None,
                                threshold = 1e-9,
                                max_iters = 1e3):
    train_model = random_Poisson_HMM(   seed = seed, 
                                        n_states = n_states, 
                                        dims = dims,
                                        lambdas = lambdas)
    train_model.fit(train_dat,  algorithm = 'baum-welch', 
                                stop_threshold = threshold, 
                                max_iterations = max_iters,
                                verbose = False)
    return train_model

def parallel_train_HMM(data, dims, param_tuple, lambdas = None):
    """
    param_tuple : (n_states,seed)
    """
    train_model = train_random_Poisson_HMM(
                                    data = data,
                                    n_states = param_tuple[0],
                                    dims = dims,
                                    seed = param_tuple[1],
                                    lambdas = lambdas)
    return train_model

# Generate fake data 
nrns = 10
trials = 30
length = 7000
state_order = [0,1,2,3,4]
palatability_state = 3
ceil_p = 0.005
jitter_t = length*0.05
jitter_p = 0.05
jitter_p_type = 'scaled'
min_duration = 1000


all_data, all_t, mean_p, all_p, taste_scaling_value = \
                            fake_ber_firing(nrns,
                                                trials,
                                                length,
                                                state_order,
                                                palatability_state,
                                                ceil_p,
                                                jitter_t,
                                                jitter_p,
                                                jitter_p_type,
                                                min_duration)
all_data = np.array(all_data)
all_data_long = np.reshape(np.swapaxes(all_data,0,1),
                        (all_data.shape[1],-1,all_data.shape[-1]))

# Test plot
raster(all_data_long[0]);plt.show()

#Extract data for single taste 
taste_data = all_data[0]
# Bin data to generate counts
bin_size = 50
binned_taste_data = np.sum(np.reshape(taste_data,
        (*taste_data.shape[:-1],-1,bin_size)), axis=-1)

# Test plot
firing_overview(binned_taste_data);plt.show()
firing_overview(binned_taste_data.swapaxes(0,1));plt.show()

# Required shape of data : (samples, time, variabiles)

train_dat = np.moveaxis(binned_taste_data, 0,-1) 
dims = train_dat.shape[-1]
## Test fit
train_model = train_random_Poisson_HMM(
                                data = train_dat,
                                n_states = 5,
                                dims = dims,
                                seed = 2)

state_probs = np.array([train_model.predict_proba(x) for x in train_dat])
viterbi_path = np.array([train_model.predict(x,'viterbi') for x in train_dat])
fin_emission_matrix = extract_emission_matrix(train_model)

# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(np.array(mean_p[0]).T)
plt.subplot(122)
plot_image(fin_emission_matrix.T)
plt.show()

trial_num = 1
fig,ax = plt.subplots(2,1,sharex=True)
plt.sca(ax[0])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
plt.plot(state_probs[trial_num]*dims)
plt.sca(ax[1])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
plt.plot(dims*viterbi_path[trial_num]/np.max(viterbi_path,axis=None),color='red')
plt.show()


# Fit HMM with range of states to determine state number
state_range = range(2,8)
repeats_per_state = 10
seed_vec = np.arange(len(state_range)*repeats_per_state)
# Generate vector with state_range and seed to give to Parallel
param_vec = list(zip(list(state_range)*repeats_per_state,seed_vec))

model_list = Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(parallel_train_HMM)(train_dat,dims,x) for x in tqdm(param_vec))

# For every model in the list, extract the BIC
all_log_likelihoods = np.array([calculate_total_ll(x,train_dat) for x in model_list])
all_BIC = np.array([calculate_BIC(   total_ll = this_ll, 
                            n_states = param_tuple[0], 
                            dims = dims, 
                            data = train_dat) \
            for this_ll, param_tuple in zip(all_log_likelihoods,param_vec)])

# Plot results
state_num_vec = np.array([x[0] for x in param_vec])

jitter = 0.5
plt.subplot(211)
plt.scatter(state_num_vec + np.random.random(state_num_vec.shape)*jitter, 
        zscore(all_log_likelihoods)+np.random.random(all_log_likelihoods.shape)*jitter)
plt.title('Log-Likelihood')
plt.subplot(212)
plt.scatter(state_num_vec + np.random.random(state_num_vec.shape)*jitter, 
        zscore(all_BIC)+np.random.random(all_BIC.shape)*jitter);
plt.title('BIC')
plt.show()
