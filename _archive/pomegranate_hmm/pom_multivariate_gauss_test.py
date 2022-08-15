#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 22:05:41 2019

@author: abuzarmahmood
"""
from pomegranate import *
import numpy as np
import pylab as plt

dims = 20
means = np.random.random((1,dims))[0]
covar_mat = np.cov(np.random.random((dims,dims*2)))

test_dist = MultivariateGaussianDistribution(means,covar_mat)
samples = test_dist.sample(1000)

# =============================================================================
# Firring multivariate guassian hmm
# =============================================================================
seed = 0
np.random.seed(seed)
n_states = 2

# Make a pomegranate HiddenMarkovModel object
model = HiddenMarkovModel('%i' % seed) 
states = []
# Make a pomegranate Multivariate Gaussian distribution object with emissions = range(n_units + 1) - 1 for each state

for i in range(n_states):
    means = np.random.random((1,dims))[0]
    dat_points = dims**2
    covar_mat = np.cov(np.random.random((dims,dat_points)))
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

# Generate samples to plot
model_samples = np.asarray(model.sample(1000))
plt.imshow(model_samples.T,interpolation='nearest',aspect='auto')

plt.plot(model_samples[:,0],model_samples[:,1])

# Generate samples to use for fitting
sample_len = 100
num_samples = 20
train_dat = np.asarray([np.asarray(model.sample(sample_len)) for x in range(num_samples)])

train_dat += 25
plt.plot(train_dat[0,:,0],train_dat[0,:,1])
threshold = 1e-9

# Train the model only on the trials indicated by off_trials
model.fit(train_dat, algorithm = 'baum-welch', stop_threshold = threshold, verbose = True)
new_samples = np.asarray([np.asarray(model.sample(sample_len)) for x in range(num_samples)])
plt.plot(train_dat[0,:,0],train_dat[0,:,1])
