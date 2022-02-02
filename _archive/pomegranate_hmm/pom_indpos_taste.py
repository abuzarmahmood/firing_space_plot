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
from ephys_data import ephys_data
from visualize import *
os.chdir('/media/bigdata/firing_space_plot')
from pom_independent_poisson_test import random_Poisson_HMM,\
                                        extract_emission_matrix,\
                                        calculate_total_ll,\
                                        calculate_BIC,\
                                        train_random_Poisson_HMM,\
                                        parallel_train_HMM

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

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM12/AM12_4Tastes_191105_083246')
    #ephys_data('/media/bigdata/Abuzar_Data/AM11/AM11_4Tastes_191030_114043_copy/')
dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))
dat.extract_and_process()
firing_overview(dat.all_normalized_firing);plt.show()

# Split spikes by region

