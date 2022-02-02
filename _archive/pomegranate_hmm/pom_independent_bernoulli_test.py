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
from sklearn.cluster import KMeans as kmeans

# _____                 _   _                 
#|  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
#| |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#|  _|| |_| | | | | (__| |_| | (_) | | | \__ \
#|_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                                             

def raster(array):
    """
    Array is actually 2D
    """
    spike_times = np.where(array)
    plt.scatter(spike_times[1],spike_times[0], alpha = 0.1)

def plot_image(array):
    plt.imshow(array, interpolation='nearest',aspect='auto',origin='lower',cmap='viridis')


## Define function to generate a Poisson HMM

## Data entered as 3D array (trials x neurons x time)

# Make a pomegranate Independent Poisson distribution object 
# with emissions = range(n_units + 1) - 1 for each state

def random_Bernoulli_HMM(seed, n_states, dims, lambdas = None):

    """
    lambdas :: List of initial starting guess for lambdas
    """
    np.random.seed(seed)
    model = HiddenMarkovModel('{}'.format(seed)) 
    # Allowance to have a wide range of lambdas
    if lambdas is None:
        lambdas = [np.random.random(dims) for state in range(n_states)] 
    elif lambdas is not None:
        requirement = ((len(lambdas) == n_states) \
                and np.array(lambdas).shape[1] == dims) 
        if not requirement:
            raise Excpetion('Lambdas must be list of len(n_states)'\
                    'with each element of size dims')

    states = [State(
                IndependentComponentsDistribution(
                    [BernoulliDistribution(x) for x in lambdas[state]]),\
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

def train_random_Bernoulli_HMM( data, 
                                n_states, 
                                dims, 
                                seed,
                                lambdas = None,
                                threshold = 1e-9,
                                max_iters = 1e3):
    train_model = random_Bernoulli_HMM(   seed = seed, 
                                        n_states = n_states, 
                                        dims = dims,
                                        lambdas = lambdas)
    train_model.fit(train_dat,  algorithm = 'baum-welch', 
                                stop_threshold = threshold, 
                                max_iterations = max_iters,
                                verbose = False)
    return train_model

def extract_emission_matrix(model):
    emission_mat = np.array(
                [[param.parameters[0] \
                        for param in state.distribution.parameters[0]] \
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
    BIC = -2*total_ll + \
            n_states*(n_states + data.shape[1] - 1)\
            *np.log(np.prod(data.shape[1:]))
    return BIC

def calc_firing_rates(step_size, window_size, spike_array):

    total_time = spike_array.shape[-1]

    bin_inds = (0,window_size)
    total_bins = int((total_time - window_size + 1) / step_size) + 1
    bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
            for step in np.arange(total_bins)*step_size ]

    firing_rate = np.empty((\
                spike_array.shape[0],spike_array.shape[1],total_bins))
    for bin_inds in bin_list:
        firing_rate[...,bin_inds[0]//step_size] = \
                np.mean(spike_array[...,bin_inds[0]:bin_inds[1]], axis=-1)

    return firing_rate

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

############################################################
### Data from HMM
############################################################

# Generate data
n_states = 3
dims = 20
sample_len = 500
num_samples = 30
lambdas = [np.random.random(dims)*0.05 for x in range(n_states)]
model = random_Bernoulli_HMM(seed = 1, 
                            n_states = n_states, 
                            dims = dims,
                            lambdas = lambdas)
#lambdas = extract_emission_matrix(model)
train_dat = np.asarray([np.asarray(model.sample(sample_len)) for x in range(num_samples)])

# Estimate labda using moving window average and k-mean clustering

step_size = 1
window_size = 50
firing_rates = calc_firing_rates(step_size,window_size,
                        np.swapaxes(train_dat,1,-1))
#firing_overview(firing_rates.swapaxes(0,1));plt.show()

# Cluster firing rates to estimate lambdas
firing_rates_long = np.reshape(np.swapaxes(firing_rates,0,1),
        (firing_rates.shape[1],-1))
kmeans_obj = kmeans(n_clusters = n_states).fit(firing_rates_long.T) 
cluster_preds = kmeans_obj.predict(firing_rates_long.T)
cluster_means = np.array([np.mean(
                        firing_rates_long.T[cluster_preds == x],axis=0)\
                         for x in np.unique(cluster_preds)])

lambda_est = cluster_means
lambda_est = np.broadcast_to(\
        np.mean(firing_rates_long,axis=-1)[np.newaxis,:],
        (n_states,dims))
# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(np.array(lambdas).T)
plt.subplot(122)
plot_image(lambda_est.T)
plt.show()

bin_size = 10
binned_dat = np.sum(np.reshape(train_dat,
        (train_dat.shape[0],-1,bin_size,train_dat.shape[-1])),axis=2) > 0
binned_dat = binned_dat*1

train_model = train_random_Bernoulli_HMM(
                                data = binned_dat,
                                n_states = n_states,
                                dims = dims,
                                seed = 2,
                                lambdas = lambda_est,
                                max_iters = 1e4)

state_probs = np.array([train_model.predict_proba(x) for x in binned_dat])
viterbi_path = np.array([train_model.predict(x,'viterbi') for x in binned_dat])
fin_emission_matrix = extract_emission_matrix(train_model)

# Align state emissions
from scipy.spatial import distance_matrix as distmat
dists = distmat(lambdas, fin_emission_matrix)
emissions_order = np.argmin(dists,axis=1)

# Compare estimated emissions with ground truth
plt.subplot(131)
plot_image(np.array(lambdas).T)
plt.subplot(132)
plot_image(fin_emission_matrix[emissions_order].T)
plt.subplot(133)
plot_image(fin_emission_matrix.T)
plt.show()

trial_num = 2
fig,ax = plt.subplots(2,1,sharex=True)
plt.sca(ax[0])
plot_image(binned_dat[trial_num].T)
plt.plot(state_probs[trial_num]*dims)
plt.sca(ax[1])
plot_image(binned_dat[trial_num].T)
plt.plot(dims*viterbi_path[trial_num]/np.max(viterbi_path,axis=None),
        color='red')
plt.show()

############################################################
### Bernoulli Data - Single Taste 
############################################################

# Generate fake data 
nrns = 10
trials = 30
length = 7000
state_order = [0,1,2,3,4]
palatability_state = 3
ceil_p = 0.005
jitter_t = 0#length*0.05
jitter_p = 0#0.05
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

# Estimate labda using moving window average and k-mean clustering

step_size = 25
window_size = 250
firing_rates = calc_firing_rates(step_size,window_size,taste_data)
firing_overview(firing_rates);plt.show()

# Cluster firing rates to estimate lambdas
firing_rates_long = np.reshape(firing_rates,
        (firing_rates.shape[0],-1))
kmeans_obj = kmeans(n_clusters = len(state_order)).fit(firing_rates_long.T) 
cluster_preds = kmeans_obj.predict(firing_rates_long.T)
cluster_means = np.array([np.mean(
                        firing_rates_long.T[cluster_preds == x],axis=0)\
                         for x in np.unique(cluster_preds)])

lambda_est = cluster_means
lambda_est = np.broadcast_to(\
        np.mean(firing_rates_long,axis=-1)[np.newaxis,:],
        (len(state_order),nrns))
# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(mean_p[0].T)
plt.subplot(122)
plot_image(lambda_est.T)
plt.show()


# Required shape of data : (samples, time, variabiles)
# Bin data to generate counts
bin_size = 50
binned_taste_data = np.sum(np.reshape(taste_data,
        (*taste_data.shape[:-1],-1,bin_size)), axis=-1) > 0
binned_taste_data = binned_taste_data * 1

lambda_est_binned = np.broadcast_to(\
        np.mean(binned_taste_data,axis=(1,2))[np.newaxis,:],
        (len(state_order),nrns))

# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(lambda_est.T)
plt.subplot(122)
plot_image(lambda_est_binned.T)
plt.show()

# Test plot
raster(binned_taste_data[1]);plt.show()

train_dat = np.moveaxis(binned_taste_data,0,-1)

# Fit data with bernoulli HMM
train_model = train_random_Bernoulli_HMM(
                                data = train_dat,
                                n_states = len(state_order),
                                dims = taste_data.shape[0],
                                seed = 2,
                                lambdas = lambda_est_binned,
                                max_iters = 1e4)

state_probs = np.array([train_model.predict_proba(x) for x in train_dat])
viterbi_path = np.array([train_model.predict(x,'viterbi') for x in train_dat])
fin_emission_matrix = extract_emission_matrix(train_model)

# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(mean_p[0].T)
plt.subplot(122)
plot_image(fin_emission_matrix.T)
plt.show()

#trial_num = -1
#fig,ax = plt.subplots(2,1,sharex=True)
#plt.sca(ax[0])
#plot_image(zscore(train_dat[trial_num].T,axis=-1))
#plt.plot(state_probs[trial_num]*dims)
#plt.sca(ax[1])
#plot_image(zscore(train_dat[trial_num].T,axis=-1))
#plt.plot(dims*viterbi_path[trial_num]/np.max(viterbi_path,axis=None),
#        color='red')
#plt.show()

############################################################
### Bernoulli Data - 4 Tastes 
############################################################

# Required shape of data : (samples, time, variabiles)
# Bin data to generate counts
bin_size = 50
binned_taste_data = np.sum(np.reshape(all_data_long,
        (*all_data_long.shape[:-1],-1,bin_size)), axis=-1) > 0
binned_taste_data = binned_taste_data * 1

lambda_est_binned = np.broadcast_to(\
        np.mean(binned_taste_data,axis=(1,2))[np.newaxis,:],
        (9,nrns))

train_dat = np.moveaxis(binned_taste_data,1,-1).swapaxes(0,-1)
# Fit data with bernoulli HMM
train_model = train_random_Bernoulli_HMM(
                                data = train_dat,
                                n_states = 9,
                                dims = taste_data.shape[0],
                                seed = 2,
                                lambdas = lambda_est_binned,
                                max_iters = 1e4)

state_probs = np.array([train_model.predict_proba(x) for x in train_dat])
viterbi_path = np.array([train_model.predict(x,'viterbi') for x in train_dat])
fin_emission_matrix = extract_emission_matrix(train_model)

# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(mean_p[0].T)
plt.subplot(122)
plot_image(fin_emission_matrix.T)
plt.show()

trial_num = -1
fig,ax = plt.subplots(2,1,sharex=True)
plt.sca(ax[0])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
plt.plot(state_probs[trial_num]*dims)
plt.sca(ax[1])
plot_image(zscore(train_dat[trial_num].T,axis=-1))
plt.plot(dims*viterbi_path[trial_num]/np.max(viterbi_path,axis=None),
        color='red')
plt.show()

