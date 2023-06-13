"""
Functions for fitting GLMs to data.
"""

import numpy as np
import pylab as plt
from scipy.stats import zscore
import pandas as pd
import sys
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
import makeRaisedCosBasis as cb
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
import statsmodels.formula.api as smf
from pandas import DataFrame as df
from pandas import concat

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
    X.columns = [f'hist_lag{i:03d}' for i \
            in np.arange(filter_len,0, step = -1)] 
    return X.iloc[:,::-1]

def gen_cosine_history_design(
        data, 
        filter_len, 
        n_basis = 10, 
        spread = 'log'):
    """
    Generate a design matrix for a history filter using
    cosine basis functions

    args:
        data: n x 1 vector of data
        filter_len: length of filter
        n_basis: number of basis functions
        spread: spread of basis functions (linear vs log)

    returns:
        X: n_basis x filter_len matrix
    """
    cos_basis = cb.gen_raised_cosine_basis(
            filter_len,
            n_basis,
            spread = spread)
    hist_mat = gen_history_design_matrix(data, filter_len)
    cos_mat = np.matmul(hist_mat, cos_basis.T)
    cos_mat.columns = [f'hist_lag{i:03d}' for i \
            in np.arange(n_basis)]
    return cos_mat

def gen_stim_design_matrix(stim_vec, filter_len):
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
    return X.iloc[:,::-1]

def gen_cosine_stim_design(
        stim_vec, 
        filter_len, 
        n_basis = 10, 
        spread = 'log'):
    """
    Generate a design matrix for a history filter using
    cosine basis functions

    args:
        data: n x 1 vector of data
        filter_len: length of filter
        n_basis: number of basis functions
        spread: spread of basis functions (linear vs log)

    returns:
        X: n_basis x filter_len matrix
    """
    cos_basis = cb.gen_raised_cosine_basis(
            filter_len,
            n_basis,
            spread = spread)
    stim_mat = gen_stim_design_matrix(stim_vec, filter_len)
    cos_mat = np.matmul(stim_mat, cos_basis.T)
    cos_mat.columns = [f'stim_lag{i:03d}' for i \
            in np.arange(n_basis)]
    return cos_mat


def fit_history_glm(
        data, 
        filter_len, 
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        ):
    """
    Fit a poisson GLM with a history filter

    args:
        data: n x 1 vector of data
        filter_len: length of filter
        basis: basis function to use for history filter ('full' or 'cos')

    returns:
        filter: filter_len x 1 vector
    """
    if basis == 'cos':
        design_mat = gen_cosine_history_design(data, filter_len,
                                               n_basis = n_basis,
                                               spread = basis_spread)
    elif basis == 'full':
        design_mat=  gen_history_design_matrix(data, filter_len)
    else:
        raise ValueError('Invalid basis function')
    glmdata = design_mat.copy()
    glmdata['spikes'] = data[filter_len:]
    lag_columns = [x for x in glmdata.columns if 'hist_lag' in x]
    formula = 'spikes ~ ' + ' + '.join(lag_columns) 
    model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    res = model.fit()
    return res

def process_glm_res(
        res, 
        filter_len, 
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        param_key = 'hist',):
    lag_params =  res.params
    lag_params = lag_params[[x for x in lag_params.index if param_key in x]]
    if basis == 'linear':
        lag_params.index = [int(x.split(param_key)[1]) for x in lag_params.index]
    elif basis == 'cos':
        cos_basis = cb.gen_raised_cosine_basis(
                            filter_len,
                            n_basis = n_basis,
                            spread = basis_spread,)
        lag_params = lag_params[None,:].dot(cos_basis).flatten()
    return lag_params

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

def gen_single_coupled_prob(input_spikes, coupling_filter, n):
    prob = np.zeros(n)
    for i in range(n-len(coupling_filter)):
        if input_spikes[i]:
            prob[i:i+len(coupling_filter)] += coupling_filter
    return prob

def gen_single_coupled_data(input_spikes, coupling_filter, n):
    prob = gen_single_coupled_prob(input_spikes, coupling_filter, n)
    spike_prob = prob.copy()
    spike_prob[spike_prob > 1] = 1
    output_spikes = np.random.binomial(1, np.exp(spike_prob))
    return spikes, spike_prob

def generate_coupling_data(hist_filter_list, coupling_filter_list, n):
    assert len(hist_filter_list) == len(coupling_filter_list), \
            'hist_filter_list and coupling_filter_list must be the same length'
    spike_outs = [generate_history_data(hist_filter, n)[0] \
            for hist_filter in hist_filter_list]
    coupling_probs = np.stack(
            [gen_single_coupled_prob(spike_out, coupling_filter, n) \
            for spike_out, coupling_filter \
            in zip(spike_outs, coupling_filter_list)]
            )
    sum_coupling_probs = np.sum(coupling_probs, axis = 0)
    spike_prob = sum_coupling_probs.copy()
    spike_prob[spike_prob > 1] = 1
    output_spikes = np.random.binomial(1, np.exp(spike_prob))
    return output_spikes, spike_prob, coupling_probs, np.stack(spike_outs)

def generate_history_coupling_data(
        hist_filter,
        coupling_hist_filter_list, 
        coupling_filter_list, 
        n):
    assert len(coupling_hist_filter_list) == len(coupling_filter_list), \
            'coupling_hist_filter_list and coupling_filter_list must be the same length'
    spike_outs = [generate_history_data(hist_filter, n)[0] \
            for hist_filter in coupling_hist_filter_list]
    coupling_probs = np.stack(
            [gen_single_coupled_prob(spike_out, coupling_filter, n) \
            for spike_out, coupling_filter \
            in zip(spike_outs, coupling_filter_list)]
            )
    sum_coupling_probs = np.sum(coupling_probs, axis = 0)
    _, history_prob = generate_history_data(hist_filter, n)

    spike_prob = sum_coupling_probs + history_prob 
    spike_prob[spike_prob > 1] = 1
    output_spikes = np.random.binomial(1, np.exp(spike_prob))
    return output_spikes, spike_prob, coupling_probs, np.stack(spike_outs)

def generate_stim_history_coupling_data(
        hist_filter,
        coupling_hist_filter_list, 
        coupling_filter_list, 
        stim_filter,
        stim_count = 10,
        n = 10000,
        ):
    assert len(coupling_hist_filter_list) == len(coupling_filter_list), \
            'coupling_hist_filter_list and coupling_filter_list must be the same length'
    spike_outs = [generate_history_data(hist_filter, n)[0] \
            for hist_filter in coupling_hist_filter_list]
    coupling_probs = np.stack(
            [gen_single_coupled_prob(spike_out, coupling_filter, n) \
            for spike_out, coupling_filter \
            in zip(spike_outs, coupling_filter_list)]
            )
    sum_coupling_probs = np.sum(coupling_probs, axis = 0)
    _, history_prob = generate_history_data(hist_filter, n)
    _, stim_prob, stim_vec = generate_stim_data(stim_filter, stim_count, n)

    spike_prob = sum_coupling_probs + history_prob + stim_prob 
    spike_prob[spike_prob > 1] = 1
    output_spikes = np.random.binomial(1, np.exp(spike_prob))
    return output_spikes, spike_prob, coupling_probs, np.stack(spike_outs), stim_vec


def gen_single_coupling_design(spikes, filter_len, id = None):
    if id is not None:
        id_str = str(id)
    else:
        id_str = ''
    index = np.arange(filter_len, len(spikes))
    coupling_dat = np.stack([spikes[i-filter_len:i] \
            for i in index])
    X = df(coupling_dat, index = index)
    X.columns = [f'coupling_lag_{id_str}_{i:03d}' for i in \
            np.arange(filter_len,0, step = -1)] 
    return X.iloc[:,::-1]

def gen_coupling_design_mat(coupled_spikes, filter_len, stack = False):
    coupling_design = [gen_single_coupling_design(this_spikes, filter_len, id = i) \
            for i, this_spikes in enumerate(coupled_spikes)]
    if stack:
        coupling_design = pd.concat(coupling_design, axis = 1)
    return coupling_design

def gen_cosine_coupling_design(
        coupled_spikes,
        filter_len, 
        n_basis = 10, 
        spread = 'log'):

    cos_basis = cb.gen_raised_cosine_basis(
            filter_len,
            n_basis,
            spread = spread)
    coupled_mat_list = gen_coupling_design_mat(
            coupled_spikes, 
            filter_len, 
            stack=False)
    cos_mat_list = [np.matmul(coupled_mat, cos_basis.T) \
            for coupled_mat in coupled_mat_list]
    for i in range(len(cos_mat_list)):
        cos_mat_list[i] = pd.DataFrame(cos_mat_list[i], 
                index = coupled_mat_list[i].index)
        cos_mat_list[i].columns = [f'coupling_lag_{i}_{j:03d}' for j in \
                np.arange(n_basis)]
    cos_mat = pd.concat(cos_mat_list, axis = 1)
    return cos_mat

def generate_stim_data(stim_filter , stim_count = 10, n = 10000):
    assert len(stim_filter)*stim_count < n, 'stim_filter_len*stim_count must be less than n'
    prob = np.zeros(n)
    stim_ind = np.arange(stim_count) * int(n/stim_count) 
    #stim_filter = generate_stim_filter(filter_len = stim_filter_len)
    stim_vec = np.zeros(n) 
    stim_vec[stim_ind] = 1 
    stim_prob = np.zeros(n)
    for i in np.where(stim_vec)[0]:
        stim_prob[i:i+len(stim_filter)] = stim_filter
    spikes = np.zeros(n)
    for i in range(n-1):
        prob[i] = np.exp(stim_prob[i])
        spike_prob = np.min([prob[i], 1])
        spikes[i+1] = np.random.binomial(1, spike_prob)
    return spikes, stim_prob, stim_vec

def generate_stim_filter(filter_len = 100):
    x = np.arange(filter_len)
    #filter = (1*(np.exp(-0.05*x) - np.exp(-0.5*x)))+1e-3#*np.sin(0.1*x) 
    filter = np.exp(-0.05*x) 
    return np.log(filter)

def generate_stim_history_data(
        hist_filter, 
        stim_filter,
        n, 
        stim_count = 10,
        stim_filter_len = 500):

    _, stim_prob, stim_vec = generate_stim_data(
            stim_filter_len = len(stim_filter), n = n, stim_count = stim_count)
    prob = np.zeros(n)
    hist_prob = np.zeros(n)
    spikes = np.zeros(n)
    spikes[0] = 1
    min_filter = np.min([len(hist_filter), len(stim_filter)])
    for i in range(n-min_filter):
        if spikes[i]:
            hist_prob[i:i+len(hist_filter)] += hist_filter
        prob[i+1] = hist_prob[i+1]
        spikes[i+1] = np.random.binomial(1, np.min([np.exp(prob[i+1]), 1]))

    stim_inds = np.where(stim_vec)[0]
    for i in stim_inds:
        prob[i:i+len(stim_filter)] = stim_filter

    spike_prob = np.exp(prob)
    spike_prob[spike_prob > 1] = 1
    spikes = np.random.binomial(1, spike_prob)
    return spikes, prob, stim_vec

def fit_stim_history_coupled_glm(
        spike_data, 
        coupled_spikes,
        stim_data,
        hist_filter_len = 10,
        coupling_filter_len = 10,
        stim_filter_len = 500,
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        ):
    if basis == 'cos':
        coupling_design_mat = gen_cosine_coupling_design(
                                        coupled_spikes, 
                                        coupling_filter_len,
                                        n_basis = n_basis,
                                        spread = basis_spread)
        history_design_mat = gen_cosine_history_design(
                                            spike_data, 
                                            hist_filter_len,
                                               n_basis = n_basis,
                                               spread = basis_spread)
        stim_design_mat = gen_cosine_stim_design(
                                        stim_data, 
                                        stim_filter_len,
                                        n_basis = n_basis,
                                        spread = basis_spread)
    elif basis == 'full':
        coupling_design_mat = gen_coupling_design_mat(
               coupled_spikes, 
               coupling_filter_len,
               stack=True)
        history_design_mat =  gen_history_design_matrix(data, hist_filter_len)
        stim_design_mat = gen_stim_desigm_matrix(stim_data, stim_filter_len)
    else:
        raise ValueError('Invalid basis function')

    # Concatenate design matrices
    design_mat = pd.concat(
            [history_design_mat, coupling_design_mat, stim_design_mat], 
            axis=1)

    # Data must be cut to largest filter
    spike_df = df(dict(spikes = spike_data), 
                  index = np.arange(len(spike_data)))
    glmdata = pd.concat(
            [
                 design_mat, 
                 spike_df
                 ], 
            axis = 1)
    glmdata = glmdata.dropna().sort_index()
    #glmdata['intercept'] = 1

    dv_cols = [x for x in glmdata.columns if 'lag' in x]
    #dv_cols.append('intercept')
    # Formula api already adds intercept
    formula = 'spikes ~ ' + ' + '.join(dv_cols) 
    model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    #model = sm.GLM(
    #        glmdata['spikes'], 
    #        glmdata[dv_cols], 
    #        family = sm.families.Poisson())
    res = model.fit()
    pred = res.predict(glmdata[dv_cols])
    return res, pred

def fit_history_coupled_glm(
        spike_data, 
        coupled_spikes,
        hist_filter_len = 10,
        coupling_filter_len = 10,
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        ):
    if basis == 'cos':
        coupling_design_mat = gen_cosine_coupling_design(
                                        coupled_spikes, 
                                        coupling_filter_len,
                                        n_basis = n_basis,
                                        spread = basis_spread)
        history_design_mat = gen_cosine_history_design(
                                            spike_data, 
                                            hist_filter_len,
                                               n_basis = n_basis,
                                               spread = basis_spread)
    elif basis == 'full':
        coupling_design_mat = gen_coupling_design_mat(
               coupled_spikes, 
               coupling_filter_len,
               stack=True)
        history_design_mat =  gen_history_design_matrix(data, hist_filter_len)
    else:
        raise ValueError('Invalid basis function')

    # Concatenate design matrices
    design_mat = pd.concat([history_design_mat, coupling_design_mat], axis=1)

    # Data must be cut to largest filter
    spike_df = df(dict(spikes = spike_data), 
                  index = np.arange(len(spike_data)))
    glmdata = pd.concat(
            [
                 design_mat, 
                 spike_df
                 ], 
            axis = 1)
    glmdata = glmdata.dropna().sort_index()
    #glmdata['intercept'] = 1

    dv_cols = [x for x in glmdata.columns if 'lag' in x]
    #dv_cols.append('intercept')
    # Formula api already adds intercept
    formula = 'spikes ~ ' + ' + '.join(dv_cols) 
    model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    #model = sm.GLM(
    #        glmdata['spikes'], 
    #        glmdata[dv_cols], 
    #        family = sm.families.Poisson())
    res = model.fit()
    pred = res.predict(glmdata[dv_cols])
    return res, pred

def fit_coupled_glm(
        spike_data, 
        coupled_spikes,
        coupling_filter_len = 10,
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        ):
    if basis == 'cos':
        coupling_design_mat = gen_cosine_coupling_design(
                                        coupled_spikes, 
                                        coupling_filter_len,
                                        n_basis = n_basis,
                                        spread = basis_spread)
    elif basis == 'full':
        coupling_design_mat = gen_coupling_design_mat(
               coupled_spikes, 
               coupling_filter_len,
               stack=True)
    else:
        raise ValueError('Invalid basis function')

    # Data must be cut to largest filter
    spike_df = df(dict(spikes = spike_data), 
                  index = np.arange(len(spike_data)))
    glmdata = pd.concat(
            [
                 coupling_design_mat, 
                 spike_df
                 ], 
            axis = 1)
    glmdata = glmdata.dropna().sort_index()
    #glmdata['intercept'] = 1

    dv_cols = [x for x in glmdata.columns if 'lag' in x]
    #dv_cols.append('intercept')
    # Formula api already adds intercept
    formula = 'spikes ~ ' + ' + '.join(dv_cols) 
    model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    #model = sm.GLM(
    #        glmdata['spikes'], 
    #        glmdata[dv_cols], 
    #        family = sm.families.Poisson())
    res = model.fit()
    pred = res.predict(glmdata[dv_cols])
    return res, pred

def fit_stim_glm(
        spike_data, 
        stim_data,
        stim_filter_len,
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        ):
    """
    Fit a poisson GLM with a stimulus filter

    args:
        data: n x 1 vector of data
        stim_filter_len: length of stimulus filter

    returns:
        filter: filter_len x 1 vector
    """
    if basis == 'cos':
        stim_design_mat = gen_cosine_stim_design(
                                        stim_data, 
                                        stim_filter_len,
                                        n_basis = n_basis,
                                        spread = basis_spread)
    elif basis == 'full':
        stim_design_mat = gen_stim_desigm_matrix(stim_data, stim_filter_len)
    else:
        raise ValueError('Invalid basis function')

    # Data must be cut to largest filter
    spike_df = df(dict(spikes = spike_data), 
                  index = np.arange(len(spike_data)))
    glmdata = pd.concat(
            [
                 stim_design_mat, 
                 spike_df
                 ], 
            axis = 1)
    glmdata = glmdata.dropna().sort_index()
    #glmdata['intercept'] = 1

    dv_cols = [x for x in glmdata.columns if 'lag' in x]
    #dv_cols.append('intercept')
    # Formula api already adds intercept
    formula = 'spikes ~ ' + ' + '.join(dv_cols) 
    model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    #model = sm.GLM(
    #        glmdata['spikes'], 
    #        glmdata[dv_cols], 
    #        family = sm.families.Poisson())
    res = model.fit()
    pred = res.predict(glmdata[dv_cols])
    return res, pred

def fit_stim_history_glm(
        spike_data, 
        stim_data,
        hist_filter_len, 
        stim_filter_len,
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        ):
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
    if basis == 'linear':
        hist_design_mat=  gen_history_design_matrix(spike_data, hist_filter_len)
        stim_design_mat = gen_stim_design_matrix(stim_data, stim_filter_len)
    elif basis == 'cos':
        stim_design_mat = gen_cosine_stim_design(
                                        stim_data, 
                                        stim_filter_len,
                                        n_basis = n_basis,
                                        spread = basis_spread)
        hist_design_mat = gen_cosine_history_design(
                                        spike_data, 
                                        hist_filter_len,
                                       n_basis = n_basis,
                                       spread = basis_spread)


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

    #stim_cols = [x for x in dv_cols if 'stim' in x]
    #stim_mat_data = glmdata[stim_cols]

    #fig,ax = plt.subplots(2,1, sharex=True)
    #ax[0].imshow(stim_mat_data.T, aspect = 'auto')
    #ax[1].plot(stim_mat_data.index, stim_data[stim_mat_data.index.values])
    #plt.show()

    # Formula api breaks with too many columns
    if glmdata.shape[1] < 100:
        formula = 'spikes ~ ' + ' + '.join(dv_cols) 
        model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    else:
        glmdata['intercept'] = 1
        dv_cols.append('intercept')
        model = sm.GLM(
                glmdata['spikes'], 
                glmdata[dv_cols], 
                family = sm.families.Poisson())
    res = model.fit()
    pred = res.predict(glmdata[dv_cols])
    return res, pred

