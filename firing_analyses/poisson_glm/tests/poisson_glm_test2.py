"""
Simulate and infer models with different filters

1) Simple history filter
2) Stimulus and history filter
3) Stimulus, history, and coupling filter (2 neurons)
4) Stimulus, history, and coupling filter (n neurons)
"""

import numpy as np
import pylab as plt
from scipy.stats import zscore
import pandas as pd

# Generate data
def gen_raised_cosine_basis(n, dt, tau, n_basis):
    """
    Generate raised cosine basis functions

    args:
        n: number of time bins
        dt: time bin size
        tau: time constant of basis functions
        n_basis: number of basis functions

    returns:
        basis: n_basis x n matrix of basis functions
    """
    raise NotImplementedError

def generate_random_filter(length):
    """
    Generate a random filter using a random walk

    args:
        length: length of filter

    returns:
        filter: length x 1 vector
    """
    vals = np.random.randn(length)
    filter = np.cumsum(vals)
    # Scale between -3 and 0
    filter = filter - np.min(filter)
    filter = filter / np.max(filter)
    return filter * -3 

def calc_sta(data, filter_len):
    """
    Calculate the spike triggered average

    args:
        data: n x 1 vector of data
        filter_len: length of filter

    returns:
        sta: filter_len x 1 vector
    """
    inds = np.where(data)[0]
    cut_inds = inds[inds > filter_len]
    sta = np.stack([data[i-filter_len:i] for i in cut_inds]) 
    return np.mean(sta, axis = 0)


# Fit a poisson GLM with a history filter
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
import statsmodels.formula.api as smf
from pandas import DataFrame as df
from pandas import concat

def gen_history_design_matrix(data, filter_len):
    """
    Generate a design matrix for a history filter

    args:
        data: n x 1 vector of data
        filter_len: length of filter

    returns:
        X: n x filter_len matrix
    """
    index = np.arange(filter_len, len(data))
    hist_dat = np.stack([data[i-filter_len:i] \
            for i in index])
    X = df(hist_dat, index = index)
    X.columns = ['hist_lag' + str(i) for i \
            in np.arange(filter_len,0, step = -1)] 
    return X

def gen_stim_desigm_matrix(stim_vec, filter_len):
    """
    Generate a design matrix for a stimulus filter

    args:
        stim_vec: n x 1 vector of stimulus
        filter_len: length of filter

    returns:
        X: n x filter_len matrix
    """
    index = np.arange(filter_len, len(stim_vec))
    stim_dat = np.stack([stim_vec[i-filter_len:i] \
            for i in index])
    X = df(stim_dat, index = index)
    X.columns = [f'stim_lag{i:03d}' for i in \
            np.arange(filter_len,0, step = -1)] 
    return X

def fit_history_glm(data, filter_len, n_iter = 1000, lr = 0.01):
    """
    Fit a poisson GLM with a history filter

    args:
        data: n x 1 vector of data
        filter_len: length of filter
        n_iter: number of iterations
        lr: learning rate

    returns:
        filter: filter_len x 1 vector
    """
    design_mat=  gen_history_design_matrix(data, filter_len)
    glmdata = design_mat.copy()
    glmdata['spikes'] = data[filter_len:]
    lag_columns = [x for x in glmdata.columns if 'hist_lag' in x]
    formula = 'spikes ~ ' + ' + '.join(lag_columns) 
    model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    res = model.fit()
    return res

def generate_history_data(filter, n):
    prob = np.zeros(n)
    spikes = np.zeros(n)
    spikes[0] = 1
    for i in range(n-len(filter)):
        if spikes[i]:
            prob[i:i+len(filter)] += filter
        spike_prob = np.min([np.exp(prob[i]), 1])
        spikes[i+1] = np.random.binomial(1, spike_prob)
    return spikes, prob

def generate_stim_filter(filter_len = 500):
    x = np.arange(filter_len)
    filter = 10*(np.exp(-0.005*x) - np.exp(-0.5*x))#*np.sin(0.1*x) 
    return filter

def generate_stim_history_data(hist_filter, n, stim_loc = 0.1,
                               stim_filter_len = 500):
    prob = np.zeros(n)
    stim_ind = int(stim_loc*n)
    stim_filter = generate_stim_filter(filter_len = stim_filter_len)
    stim_vec = np.zeros(n)
    stim_vec[stim_ind] = 1 
    stim_prob = np.zeros(n)
    hist_prob = np.zeros(n)
    spikes = np.zeros(n)
    spikes[0] = 1
    for i in range(n-len(hist_filter)):
        if spikes[i]:
            hist_prob[i:i+len(hist_filter)] += hist_filter
        if stim_vec[i]:
            stim_prob[i:i+len(stim_filter)] += stim_filter
        prob[i] = np.exp(hist_prob[i] + stim_prob[i])
        spike_prob = np.min([prob[i], 1])
        spikes[i+1] = np.random.binomial(1, spike_prob)
    return spikes, prob, stim_vec

def fit_history_glm(
        spike_data, 
        stim_data,
        hist_filter_len, 
        stim_filter_len,
        n_iter = 1000, 
        lr = 0.01):
    """
    Fit a poisson GLM with a history filter

    args:
        data: n x 1 vector of data
        hist_filter_len: length of history filter
        stim_filter_len: length of stimulus filter
        n_iter: number of iterations
        lr: learning rate

    returns:
        filter: filter_len x 1 vector
    """
    hist_design_mat=  gen_history_design_matrix(spike_data, hist_filter_len)
    stim_design_mat = gen_stim_desigm_matrix(stim_data, stim_filter_len)


    # Data must be cut to largest filter
    spike_df = df(dict(spikes = spike_data), 
                  index = np.arange(len(spike_data)))
    glmdata = pd.concat(
            [
                hist_design_mat, 
                 stim_design_mat, 
                 spike_df
                 ], 
            axis = 1)
    glmdata = glmdata.dropna().sort_index()

    dv_cols = [x for x in glmdata.columns if 'lag' in x]
    #formula = 'spikes ~ ' + ' + '.join(dv_cols) 
    #model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    model = sm.GLM(
            glmdata['spikes'], 
            glmdata[dv_cols], 
            family = sm.families.Poisson())
    res = model.fit()
    return res


############################################################
## History filter
############################################################
# Need to look specifically at exponential filters for
# visual comparison because things are weird in log space
hist_filter_len = 80
hist_filter = generate_random_filter(hist_filter_len)
#plt.plot(np.exp(hist_filter))
#plt.show()
data,prob = generate_history_data(hist_filter, 10000)
sta = calc_sta(data, filter_len)
res = fit_history_glm(data, filter_len)
lag_params =  res.params[1:]
lag_params.index = [int(x.split('lag')[1]) for x in lag_params.index]
fig, ax = plt.subplots(4, 1, figsize = (10, 5), sharey = True)
ax[0].plot(data)
ax[0].set_title(f'Mean firing rate: {np.mean(data)*1000}')
ax[1].plot(sta[::-1])
ax[1].set_title('STA (reversed)')
ax[2].plot(np.exp(hist_filter))
ax[2].set_title('True filter (exp)')
ax[3].plot(np.exp(lag_params))
ax[3].set_title('Estimated filter (exp)')
plt.show()

############################################################
## Stim + History Filter 
############################################################
stim_filter = generate_stim_filter()
plt.plot(stim_filter); plt.show()

stim_filter_len = 500
spike_data, prob, stim_data = \
        generate_stim_history_data(
                hist_filter, 
                n, 
                stim_filter_len = stim_filter_len)

stim_design_mat = gen_stim_desigm_matrix(stim_vec, stim_filter_len)
plt.imshow(
        stim_design_mat, 
        aspect = 'auto', interpolation = 'none', cmap = 'gray',
        origin = 'lower',
        extent = [0, stim_filter_len, stim_filter_len, n])
plt.xlabel('Stimulus lag')
plt.ylabel('Time')
plt.show()

hist_params = res.params[[x for x in res.params.index if 'hist' in x]]
stim_params = res.params[[x for x in res.params.index if 'stim' in x]]

fig, ax = plt.subplots(7, 1, figsize = (5, 10), sharey = False)
ax[0].plot(spike_data, label = 'spikes', linewidth = 0.5)
ax[0].plot(stim_data, label = 'stim', linewidth = 3)
ax[0].legend()
ax[0].set_title(f'Mean firing rate: {np.mean(data)*1000}')
ax[1].plot(prob)
ax[1].set_title('Probability')
ax[2].plot(sta[::-1])
ax[2].set_title('STA (reversed)')
ax[3].plot(np.exp(hist_filter))
ax[3].set_title('True filter (exp)')
ax[4].plot(np.exp(hist_params)[::-1])
ax[4].set_title('Estimated filter (exp)')
ax[5].plot(np.exp(stim_filter))
ax[5].set_title('Stim filter (exp)')
ax[6].plot(np.exp(stim_params))
ax[6].set_title('Estimated stim filter (exp)')
plt.tight_layout()
plt.show()
