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

##############################
# Simple Fits 
##############################

# Fit a poisson GLM with a history filter
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

##############################
# Complex Fits 
##############################

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

##############################
# Complex Fits 
##############################
def gen_actual_fit(
		data_frame,
		hist_filter_len = 10,
		coupling_filter_len = 10,
		stim_filter_len = 500,
		basis_kwargs = {},
		actual_design_mat = None,
		):
    """
    Generate fit using dataframe
    """

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

