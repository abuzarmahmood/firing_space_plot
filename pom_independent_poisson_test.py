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

# _____                 _   _                 
#|  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
#| |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#|  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#|_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                                             

def plot_image(array):
    plt.imshow(array, interpolation='nearest',aspect='auto',origin='lower',cmap='viridis')


## Define function to generate a Poisson HMM

## Data entered as 3D array (trials x neurons x time)

# Make a pomegranate Independent Poisson distribution object 
# with emissions = range(n_units + 1) - 1 for each state
def random_Poisson_HMM(seed, n_states, dims):
    np.random.seed(seed)
    model = HiddenMarkovModel('{}'.format(seed)) 
    lambdas = [np.random.random(dims) for state in range(n_states)] 
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
                                threshold = 1e-9,
                                max_iters = 1e3):
    train_model = random_Poisson_HMM(   seed = seed, 
                                        n_states = n_states, 
                                        dims = dims)
    train_model.fit(train_dat,  algorithm = 'baum-welch', 
                                stop_threshold = threshold, 
                                max_iterations = max_iters,
                                verbose = False)
    return train_model

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

# Generate data
n_states = 3
dims = 20
sample_len = 500
num_samples = 30

model = random_Poisson_HMM(seed = 1, n_states = n_states, dims = dims)
lambdas = extract_emission_matrix(model)
train_dat = np.asarray([np.asarray(model.sample(sample_len)) for x in range(num_samples)])

# Train the model on the samples collected
# Parameters for training

#threshold = 1e-9
#train_model = random_Poisson_HMM(seed = 2, n_states = n_states, dims = dims)
#train_original_lambdas = extract_emission_matrix(train_model)
#train_model.fit(train_dat, algorithm = 'baum-welch', stop_threshold = threshold, verbose = True)
train_model = train_random_Poisson_HMM(
                                data = train_dat,
                                n_states = n_states,
                                dims = dims,
                                seed = 2)

# Extract relevant info from trained model
#train_original_lambdas = extract_emission_matrix(original_model)
state_probs = np.array([train_model.predict_proba(x) for x in train_dat])
viterbi_path = np.array([train_model.predict(x,'viterbi') for x in train_dat])
fin_emission_matrix = extract_emission_matrix(train_model)

# Align state emissions
from scipy.spatial import distance_matrix as distmat
dists = distmat(lambdas, fin_emission_matrix)
emissions_order = np.argmin(dists,axis=1)

# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(np.array(lambdas).T)
plt.subplot(122)
plot_image(fin_emission_matrix[emissions_order].T)
#plt.subplot(133)
#plot_image(train_original_lambdas.T)
plt.show()

trial_num = 6
fig,ax = plt.subplots(2,1,sharex=True)
plt.sca(ax[0])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
plt.plot(state_probs[trial_num]*dims)
plt.sca(ax[1])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
plt.plot(dims*viterbi_path[trial_num]/np.max(viterbi_path,axis=None))
plt.show()

##################################################
## Find best number of states in training data empirically
##################################################
state_range = range(1,10)
repeats_per_state = 10
seed_vec = np.arange(len(state_range)*repeats_per_state)
# Generate vector with state_range and seed to give to Parallel
param_vec = list(zip(list(state_range)*repeats_per_state,seed_vec))

def parallel_train_HMM(data, dims, param_tuple):
    """
    param_tuple : (n_states,seed)
    """
    train_model = train_random_Poisson_HMM(
                                    data = data,
                                    n_states = param_tuple[0],
                                    dims = dims,
                                    seed = param_tuple[1])
    return train_model

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

# ____                 _       _                       _    
#/ ___|  ___ _ __ __ _| |_ ___| |____      _____  _ __| | __
#\___ \ / __| '__/ _` | __/ __| '_ \ \ /\ / / _ \| '__| |/ /
# ___) | (__| | | (_| | || (__| | | \ V  V / (_) | |  |   < 
#|____/ \___|_|  \__,_|\__\___|_| |_|\_/\_/ \___/|_|  |_|\_\
                                                           
dims = 20
lambdas = np.random.random(dims)
exp_dist_list = [PoissonDistribution(x) for x in lambdas]
ind_exp_dist = IndependentComponentsDistribution(exp_dist_list)

sample_num = 1000
samples = ind_exp_dist.sample(sample_num)

plot_image(samples.T);plt.show()

# =============================================================================
# Independent Poisson HMM 
# =============================================================================

seed = 0
np.random.seed(seed)
n_states = 3

#model = random_Poisson_HMM(seed, n_states, dims)

# Make a pomegranate Independent Poisson distribution object 
# with emissions = range(n_units + 1) - 1 for each state
model = HiddenMarkovModel('{}'.format(seed)) 
lambdas = [np.random.random(dims) for state in range(n_states)] 
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

##############################

# Generate samples to plot
samples = np.asarray(model.sample(1000))
state_probs = model.predict_proba(samples)
plot_image(zscore(samples.T,axis=-1))
plt.plot(state_probs * dims);plt.show()

##############################

# Train a new train_model on the collected data
# Make a pomegranate Independent Poisson distribution object 
# with emissions = range(n_units + 1) - 1 for each state
train_model = HiddenMarkovModel('{}'.format(seed)) 
states = [State(
            IndependentComponentsDistribution(
                [PoissonDistribution(x) for x in np.random.random(dims)]),\
        name = 'State{}'.format(state+1)) for state in range(n_states)]
train_model.add_states(states)
# Add transitions from train_model.start to each state (equal probabilties)
for state in states:
    train_model.add_transition(train_model.start, state, float(1.0/len(states)))

# Generate initial transition matrix
initial_trans_mat = np.eye(n_states)
initial_trans_mat += np.random.random(initial_trans_mat.shape)*0.05
initial_trans_mat /= np.sum(initial_trans_mat,axis=1)

for (i,j),prob in np.ndenumerate(initial_trans_mat):
    train_model.add_transition(states[i], states[j],prob)

# Bake the train_model
train_model.bake()

#train_model = random_Poisson_HMM(seed, n_states, dims)

##############################

# Train the model on the samples collected
sample_len = 500
num_samples = 20
threshold = 1e-9
train_dat = np.asarray([np.asarray(model.sample(sample_len)) for x in range(num_samples)])
train_model.fit(train_dat, algorithm = 'baum-welch', stop_threshold = threshold, verbose = True)
state_probs = np.array([train_model.predict_proba(x) for x in train_dat])
transition_mat, emission_mat= train_model.forward_backward(train_dat)

state_emissions = np.zeros((n_states,train_dat.shape[-1]))
for (i,j),temp in np.ndenumerate(state_emissions):
    state_emissions[i,j] = train_model.states[i].distribution.parameters[0][j].parameters[0]

#for i in range(n_states):
#    for j in range(train_dat.shape[-1]):
#        state_emissions[i,j] = train_model.states[i].distribution.parameters[0][j].parameters[0]

# Align state emissions
from scipy.spatial import distance_matrix as distmat
dists = distmat(np.array(lambdas), state_emissions)
emissions_order = np.argmin(dists,axis=1)

viterbi_path = np.array([train_model.predict(x,'viterbi') for x in train_dat])
    
# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(np.array(lambdas).T)
plt.subplot(122)
plot_image(state_emissions[emissions_order].T)
#plot_image(state_emissions.T)
plt.show()

trial_num = 6
fig,ax = plt.subplots(2,1,sharex=True)
plt.sca(ax[0])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
#plot_image(train_dat[trial_num].T)
plt.plot(state_probs[trial_num]*dims)
plt.sca(ax[1])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
plt.plot(dims*viterbi_path[trial_num]/np.max(viterbi_path,axis=None))
#plt.plot(posterior_proba[trial_num])
#plot_image(emission_mat)
#plt.colorbar()
plt.show()
