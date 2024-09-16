import pickle
import scipy.stats as stats
import pymc as pm
import pytensor.tensor as tt
import numpy as np
import pylab as plt
import arviz as az
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange

def gen_lambda(nrns, states, rate_scaler = 1):
    lambda_multipliers = np.random.random(states) * rate_scaler
    true_lambda = np.random.random((nrns,states))
    true_lambda = true_lambda * lambda_multipliers[np.newaxis,:]
    return true_lambda

def return_trial_dat(true_lambda, states, nrns, length):
    true_tau = np.cumsum(np.random.random(states))
    true_tau /= np.max(true_tau)
    true_tau *= length
    true_tau = np.vectorize(int)(true_tau)
    state_inds = np.concatenate([np.zeros((1)),true_tau])
    state_inds = np.vectorize(int)(state_inds)
    true_tau = true_tau[:-1]

    true_r = np.zeros((nrns,length))
    for num, val in enumerate(true_lambda.T):
        true_r[:,state_inds[num]:state_inds[num+1]] = val[:,np.newaxis]
    
    data_array = np.random.poisson(true_r)
    return true_r, data_array, true_tau

def return_poisson_data(nrns, states, rate_scaler, trial_count, length):
    true_lambda = gen_lambda(nrns, states, rate_scaler)
    trial_list = [return_trial_dat(true_lambda, states, nrns, length) for i in range(trial_count)]
    r_list, spike_list, tau_list = list(zip(*trial_list))
    r_array, spike_array, tau_array = np.stack(r_list), np.stack(spike_list), np.stack(tau_list)
    return r_array, spike_array, tau_array

def return_poisson_data_switch(nrns, states, rate_scaler, trial_count, length, n_trial_states = 2):
    trial_fracs = np.cumsum(np.random.random(n_trial_states))
    trial_fracs = trial_fracs / trial_fracs.max()
    trial_inds = np.vectorize(int)(trial_fracs * trial_count)
    trial_inds = np.concatenate([[0], trial_inds])
    trial_sections = [(trial_inds[i], trial_inds[i+1]) for i in range(len(trial_inds)-1)]
    r_array_list = []
    spike_array_list = []
    tau_array_list = []
    for this_section in trial_sections:
        r_array, spike_array, tau_array = return_poisson_data(nrns, states, rate_scaler, trial_count, length)
        r_array_list.append(r_array[this_section[0]:this_section[1]])
        spike_array_list.append(spike_array[this_section[0]:this_section[1]])
        tau_array_list.append(tau_array[this_section[0]:this_section[1]])
    fin_r_array = np.concatenate(r_array_list, axis = 0)
    fin_spike_array = np.concatenate(spike_array_list, axis = 0)
    fin_tau_array = np.concatenate(tau_array_list, axis = 0)
    return fin_r_array, fin_spike_array, fin_tau_array, trial_sections
