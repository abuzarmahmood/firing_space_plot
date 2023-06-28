"""
Functions for fitting GLMs to data.
"""

import numpy as np
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
from sklearn.model_selection import train_test_split

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
    cos_mat = pd.DataFrame(np.matmul(hist_mat.values, cos_basis.T))
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
        res = None, 
        filter_values = None,
        filter_len = 200, 
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        param_key = 'hist',):
    if res is not None:
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
    else:
        lag_params = filter_values
        if basis == 'linear':
            pass
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
        coupled_spike_inds,
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
        cos_mat_list[i].columns = [f'coupling_lag_{coupled_spike_inds[i]}_{j:03d}' for j in \
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

def gen_stim_history_coupled_design(
        spike_data, 
        coupled_spikes,
        coupled_spike_inds,
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
                                        coupled_spike_inds,
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
    return glmdata

def fit_stim_history_coupled_glm(
        spike_data = None, 
        coupled_spikes = None,
        stim_data = None,
        hist_filter_len = 10,
        coupling_filter_len = 10,
        stim_filter_len = 500,
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        regularized = True,
        alpha = 0,
        glmdata = None
        ):

    if glmdata is None:
        glmdata = gen_stim_history_coupled_design(
                spike_data, 
                coupled_spikes,
                stim_data,
                hist_filter_len = 10,
                coupling_filter_len = 10,
                stim_filter_len = 500,
                n_basis = 10,
                basis = 'cos',
                basis_spread = 'log',
                )

    dv_cols = [x for x in glmdata.columns if 'lag' in x]
    dv_cols.append('intercept')
    #if regularized == True:
    #    res = model.fit_regularized(alpha = alpha)
    #else:
    #    res = model.fit()
    #if glmdata.shape[1] < 100:
    #    formula = 'spikes ~ ' + ' + '.join(dv_cols) 
    #    model = smf.glm(formula = formula, data = glmdata, family = Poisson())
    #else:
    #    glmdata['intercept'] = 1
    #    dv_cols.append('intercept')
    #    model = sm.glm(
    #            glmdata['spikes'], 
    #            glmdata[dv_cols], 
    #            family = sm.families.poisson())
    model = sm.GLM(
            glmdata['spikes'], 
            glmdata[dv_cols], 
            family = sm.families.Poisson())
    res = model.fit()
    pred = res.predict(glmdata[dv_cols])
    return res, pred, glmdata

def fit_history_coupled_glm(
        spike_data, 
        coupled_spikes,
        hist_filter_len = 10,
        coupling_filter_len = 10,
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        regularized = True,
        alpha = 0,
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
    if regularized == True:
        res = model.fit_regularized(alpha = alpha)
    else:
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


############################################################
# Convenience Functions
############################################################

def gen_data_frame(
        spike_data, 
        coupled_spikes,
        stim_vec,
        stim_filter_len,
        trial_start_offset = 0,
        ):
    stacked_data = np.concatenate([
        spike_data[None,:], coupled_spikes, stim_vec[None,:]], 
                            axis=0)
    labels = ['spikes',*[f'coup_{i}' for i in range(len(coupled_spikes))], 'stim']
    data_frame = pd.DataFrame(
            data = stacked_data.T,
            columns = labels)
    trial_starts = np.where(stim_vec[:-stim_filter_len])[0]
    trial_starts = trial_starts + trial_start_offset
    dat_len = len(spike_data)
    trial_labels = np.zeros(dat_len)
    trial_time = np.zeros(dat_len)
    counter = 0
    for i in range(len(trial_starts)):
        if i != len(trial_starts)-1:
            trial_labels[trial_starts[i]:trial_starts[i+1]] = counter
            counter +=1
            trial_time[trial_starts[i]:trial_starts[i+1]] = \
                    np.arange(0 , trial_starts[i+1] - trial_starts[i])
        else:
            trial_labels[trial_starts[i]:dat_len] = counter
            trial_time[trial_starts[i]:dat_len] = \
                    np.arange(0, dat_len - trial_starts[i])

    data_frame['trial_labels'] = trial_labels
    data_frame['trial_time'] = trial_time
    data_frame = data_frame.astype('int')
    return data_frame

def gen_trial_shuffle(data_frame, dv = 'spikes'):
    """
    Mismatch trials between dv and iv
    """
    spike_dat = data_frame[dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x != dv]]
    unique_trials = iv_dat['trial_labels'].unique()
    trial_map = dict(zip(unique_trials, np.random.permutation(unique_trials)))
    iv_dat['trial_labels'] = [trial_map[x] for x in iv_dat['trial_labels']]
    iv_dat = iv_dat.sort_values(by = ['trial_labels', 'trial_time'])
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat], axis=1)
    return out_frame

def gen_circular_shuffle(data_frame, dv = 'spikes'):
    """
    Shuffle timebins across trials (i.e. maintain the position of time bins but
                                    change trial indices)
    """
    spike_dat = data_frame[dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x != dv]]
    time_grouped_dat = [x[1] for x in list(iv_dat.groupby('trial_time'))]
    for this_dat in time_grouped_dat:
        this_dat['trial_labels'] = np.random.permutation(this_dat['trial_labels'])
    iv_dat = pd.concat(time_grouped_dat)
    iv_dat = iv_dat.sort_values(by = ['trial_labels', 'trial_time'])
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat], axis=1)
    return out_frame

def gen_random_shuffle(data_frame, dv = 'spikes'):
    """
    Randomly shuffled IV and DV separately
    """
    trial_cols = ['trial_labels','trial_time']
    spike_dat = data_frame[dv]
    trial_dat = data_frame[trial_cols]
    rm_cols = trial_cols + [dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x not in rm_cols]]
    iv_dat = iv_dat.sample(frac = 1, replace=False)
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat, trial_dat.reset_index(drop=True)], axis=1)
    return out_frame

def dataframe_to_design_mat(
        data_frame,
        hist_filter_len,
        coupling_filter_len,
        stim_filter_len,
        basis_kwargs,
        ):
    """
    Split data into training and testing sets
    This NEEDS to be done at the design matrix level because
    temporal structure no longer matters then
    """
    coup_cols = [x for x in data_frame.columns if 'coup' in x]
    coup_inds = [int(x.split('_')[-1]) for x in coup_cols]
    glmdata = gen_stim_history_coupled_design(
                    spike_data = data_frame['spikes'].values, 
                    coupled_spikes = data_frame[coup_cols].values.T,
                    coupled_spike_inds = coup_inds,
                    stim_data = data_frame['stim'].values,
                    hist_filter_len = hist_filter_len,
                    coupling_filter_len = coupling_filter_len,
                    stim_filter_len = stim_filter_len,
                    n_basis = basis_kwargs['n_basis'],
                    basis = basis_kwargs['basis'],
                    basis_spread = basis_kwargs['basis_spread'],
                    )
    # Re-add trial_labels and trial_time
    trial_cols = ['trial_labels','trial_time']
    glmdata = pd.concat([glmdata, data_frame[trial_cols]], axis=1)
    glmdata = glmdata.dropna()
    glmdata.reset_index(inplace=True, drop=True)

    # Drop trials which are short
    trial_list = [x[1] for x in list(glmdata.groupby('trial_labels'))]
    trial_lens = [len(x) for x in trial_list]
    med_len = np.median(trial_lens)
    unwanted_trials = [i for i, this_len in enumerate(trial_lens) \
            if this_len != med_len]
    remaining_lens = [x for i,x in enumerate(trial_lens) \
            if i not in unwanted_trials]
    assert all([[x==y for x in remaining_lens] for y in remaining_lens]), \
            'Trial lengths are not equal'
    glmdata = glmdata.loc[~glmdata.trial_labels.isin(unwanted_trials)]
    glmdata.reset_index(inplace=True, drop=True)
    glmdata['intercept'] = 1
    return glmdata
    

def return_train_test_split(data_frame, test_size = 0.2, random_state = None):
    if random_state is None:
        random_state = np.random.randint(0,100)
    train_dat, test_dat = train_test_split(
            data_frame, test_size=test_size, random_state=random_state)
    return train_dat.sort_index(), test_dat.sort_index() 

from scipy.special import gammaln
def poisson_ll(lam, k):
    """
    Poisson log likelihood
    """
    lam += 1e-10 # To ensure there is no log(0)
    assert len(lam) == len(k), 'lam and k must be same length'
    assert all(lam > 0), 'lam must be non-negative'
    assert all(k >= 0), 'k must be non-negative'
    return np.sum(k*np.log(lam) - lam - gammaln(k+1))

def gen_actual_fit(
        data_frame,
        hist_filter_len = 10,
        coupling_filter_len = 10,
        stim_filter_len = 500,
        basis_kwargs = {},
        actual_design_mat = None,
        ):

    # If not given, generate an actual_design_mat
    # Otherwise use the given one
    if actual_design_mat is None:
        actual_input_dat = data_frame.copy()
        actual_design_mat = dataframe_to_design_mat(actual_input_dat)

    # Generate train test splits
    actual_train_dat, actual_test_dat = return_train_test_split(actual_design_mat)

    # Fit model to actual data
    res,pred,actual_train_dat = fit_stim_history_coupled_glm(
            glmdata = actual_train_dat,
            hist_filter_len = hist_filter_len,
            coupling_filter_len = coupling_filter_len,
            stim_filter_len= stim_filter_len,
            regularized=False,
            **basis_kwargs
            )
    return res, actual_design_mat

def calc_loglikelihood(actual_design_mat, res):
    # Generate shuffles and repeat testing
    # Note: No need to refit as we're simply showing that destroying different
    # parts of the predictors destroys the model's ability to predict actual data
    # i.e. model has learned TRIAL-SPECIFIC features
    trial_sh_design_mat = gen_trial_shuffle(actual_design_mat)
    circ_sh_design_mat = gen_circular_shuffle(actual_design_mat)
    rand_sh_design_mat = gen_random_shuffle(actual_design_mat)

    # Get train-test splits
    actual_train_dat, actual_test_dat = return_train_test_split(actual_design_mat)
    trial_sh_train_dat, trial_sh_test_dat = return_train_test_split(trial_sh_design_mat)
    circ_sh_train_dat, circ_sh_test_dat = return_train_test_split(circ_sh_design_mat)
    rand_sh_train_dat, rand_sh_test_dat = return_train_test_split(rand_sh_design_mat)

    # Calculate log-likelihoods
    actual_test_pred = res.predict(actual_test_dat[res.params.index])
    actual_test_ll = poisson_ll(actual_test_pred, actual_test_dat['spikes'].values)
    actual_test_ll = np.round(actual_test_ll, 2)

    trial_sh_test_pred = res.predict(trial_sh_test_dat[res.params.index])
    trial_sh_test_ll = poisson_ll(trial_sh_test_pred, actual_test_dat['spikes'].values)
    trial_sh_test_ll = np.round(trial_sh_test_ll, 2)

    circ_sh_test_pred = res.predict(circ_sh_test_dat[res.params.index])
    circ_sh_test_ll = poisson_ll(circ_sh_test_pred, actual_test_dat['spikes'].values)
    circ_sh_test_ll = np.round(circ_sh_test_ll, 2)

    rand_sh_test_pred = res.predict(rand_sh_test_dat[res.params.index])
    rand_sh_test_ll = poisson_ll(rand_sh_test_pred, actual_test_dat['spikes'].values)
    rand_sh_test_ll = np.round(rand_sh_test_ll, 2)

    return actual_test_ll, trial_sh_test_ll, circ_sh_test_ll, rand_sh_test_ll
