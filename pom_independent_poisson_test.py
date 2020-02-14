
from pomegranate import *
import numpy as np
import pylab as plt
from scipy.stats import zscore

def plot_image(array):
    plt.imshow(array, interpolation='nearest',aspect='auto',origin='lower',cmap='viridis')

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
n_states = 2

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

#posterior_proba = np.zeros((train_dat.shape[0], train_dat.shape[1], n_states))
#for i in range(train_dat.shape[0]):
#    c, d = model.forward_backward(train_dat[i, :, :])
#    posterior_proba[i, :, :] = np.exp(d)
    
viterbi_path = np.array([train_model.predict(x,'viterbi') for x in train_dat])
    
# Compare estimated emissions with ground truth
plt.subplot(121)
plot_image(np.array(lambdas).T)
plt.subplot(122)
plot_image(state_emissions[emissions_order].T)
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
