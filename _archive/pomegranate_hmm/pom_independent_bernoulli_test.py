#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:39:52 2019

@author: abuzarmahmood
"""

from pomegranate import *
import numpy as np
import pylab as plt
os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

dims = 3
dists = [BernoulliDistribution(np.random.rand()) for x in range(dims)]
mult_dist = IndependentComponentsDistribution(dists)

samples = mult_dist.sample(1000)

# =============================================================================
# Independent Bernoulli HMM
# =============================================================================

seed = 0
np.random.seed(seed)
n_states = 3

# Make a pomegranate HiddenMarkovModel object
model = HiddenMarkovModel('%i' % seed) 
states = []
# Make a pomegranate Multivariate Independent Benoulli distribution object with emissions = range(n_units + 1) - 1 for each state

for i in range(n_states):
    dists = [BernoulliDistribution(np.random.rand()**2) for x in range(dims)]
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

# Generate samples to plot
model_samples = np.asarray(model.sample(1000))
raster(model_samples.T)

# Generate samples to use for fitting
sample_len = 300
num_samples = 20
train_dat = np.asarray([np.asarray(model.sample(sample_len)) for x in range(num_samples)])

# Train the model on generated data
threshold = 1e-6
model.fit(train_dat, algorithm = 'baum-welch', stop_threshold = threshold, verbose = True)

posterior_proba = np.zeros((train_dat.shape[0], train_dat.shape[1], n_states))
for i in range(train_dat.shape[0]):
    c, d = model.forward_backward(train_dat[i, :, :])
    posterior_proba[i, :, :] = np.exp(d)
    
state_emissions = np.zeros((n_states,train_dat.shape[-1]))
for i in range(n_states):
    for j in range(train_dat.shape[-1]):
        state_emissions[i,j] = model.states[i].distribution.parameters[0][j].parameters[0]

trial = 7
raster(data=train_dat[trial,:,:].T,expected_latent_state=posterior_proba[trial,:,:].T)