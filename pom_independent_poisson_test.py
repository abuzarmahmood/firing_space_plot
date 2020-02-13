
from pomegranate import *
import numpy as np
import pylab as plt
from scipy.stats import zscore

def plot_image(array):
    plt.imshow(array, interpolation='nearest',aspect='auto',origin='lower',cmap='viridis')

dims = 20
lambdas = np.random.random(dims)
exp_dist_list = [ExponentialDistribution(x) for x in lambdas]
ind_exp_dist = IndependentComponentsDistribution(exp_dist_list)

sample_num = 1000
samples = ind_exp_dist.sample(sample_num)

plot_image(samples.T);plt.show()

# =============================================================================
# Independent Exponential HMM 
# =============================================================================

seed = 0
np.random.seed(seed)
n_states = 2

# Make a pomegranate Independent Exponential distribution object 
# with emissions = range(n_units + 1) - 1 for each state
model = HiddenMarkovModel('{}'.format(seed)) 
states = [State(
            IndependentComponentsDistribution(
                [ExponentialDistribution(x) for x in np.random.random(dims)]),\
        name = 'State{}'.format(state+1)) for state in range(n_states)]
model.add_states(states)
# Add transitions from model.start to each state (equal probabilties)
for state in states:
    model.add_transition(model.start, state, float(1.0/len(states)))

# Add transitions between the states - 
# 0.95-0.999 is the probability of not transitioning in every state
for i in range(n_states):
    not_transitioning_prob = (0.999-0.95)*np.random.random() + 0.95
    for j in range(n_states):
        if i==j:
            model.add_transition(states[i], states[j], not_transitioning_prob)
        else:
            model.add_transition(states[i], states[j], 
                    float((1.0 - not_transitioning_prob)/(n_states - 1)))

# Bake the model
model.bake()

# Generate samples to plot
model_samples = np.asarray(model.sample(1000))
state_probs = model.predict_proba(model_samples)
plot_image(zscore(model_samples.T,axis=-1))
plt.plot(state_probs * dims);plt.show()

# Train the model on the samples collected
sample_len = 500
num_samples = 20
threshold = 1e-9
train_dat = np.asarray([np.asarray(model.sample(sample_len)) for x in range(num_samples)])
model.fit(train_dat, algorithm = 'baum-welch', stop_threshold = threshold, verbose = True)
state_probs = np.array([model.predict_proba(x) for x in train_dat])
transition_mat, emission_mat= model.forward_backward(train_dat)

plt.subplot(121)
plot_image(emission_mat)
plt.subplot(122)
plot_image(transition_mat)
plt.show()

trial_num = 0
plt.subplot(121)
plot_image(zscore(train_dat[trial_num].T,axis=-1))
#plot_image(train_dat[trial_num].T)
plt.plot(state_probs[trial_num]*dims)
plt.subplot(122)
plot_image(emission_mat)
plt.colorbar()
plt.show()
