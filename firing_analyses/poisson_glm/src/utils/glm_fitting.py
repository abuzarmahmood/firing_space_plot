"""
Functions for fitting GLMs to data.
"""

import numpy as np
from scipy.stats import zscore
import pandas as pd
import sys
##############################
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
import utils.makeRaisedCosBasis as cb
from utils import utils
##############################
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
        # spike_data = None, 
        # coupled_spikes = None,
        # stim_data = None,
        # hist_filter_len = 10,
        # coupling_filter_len = 10,
        # stim_filter_len = 500,
        # n_basis = 10,
        # basis = 'cos',
        # basis_spread = 'log',
        # regularized = True,
        # alpha = 0,
        glmdata = None
        ):

    if glmdata is None:
        raise Exception('glmdata must be provided')
        # glmdata = gen_stim_history_coupled_design(
        #         spike_data, 
        #         coupled_spikes,
        #         stim_data,
        #         hist_filter_len = 10,
        #         coupling_filter_len = 10,
        #         stim_filter_len = 500,
        #         n_basis = 10,
        #         basis = 'cos',
        #         basis_spread = 'log',
        #         )

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
    
    # # Check corr of dv against iv
    # corr_dat = glmdata[[*dv_cols, 'spikes']].corr()
    # high_corr_dat = corr_dat.spikes.loc[corr_dat.spikes > 0.95]
    # high_corr_dat = high_corr_dat.drop('spikes')
    # high_corr_vars = high_corr_dat.index.values
    
    # # Remove high corr vars
    # dv_cols = [x for x in dv_cols if x not in high_corr_vars]

    model = sm.GLM(
            glmdata['spikes'], 
            glmdata[dv_cols], 
            family = sm.families.Poisson())

    # u, s, vt = np.linalg.svd(model.exog, 0)
    # np.mean(s>0)
    # # print(s)

    res = model.fit(method="lbfgs")
    # res = model.fit_regularized(L1_wt = 0, alpha = 0.1)
    pred = res.predict(glmdata[dv_cols])

    # plt.plot(glmdata.spikes.values)
    # plt.plot(pred.values, '-x', alpha = 0.5) 
    # plt.show()

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
def perform_fit(
		data_frame,
		hist_filter_len = 10,
		coupling_filter_len = 10,
		stim_filter_len = 500,
		basis_kwargs = {},
		design_mat = None,
                fit_type = 'actual',
		):
    """
    Generate fit using dataframe

    Inputs:
        data_frame: pandas dataframe with columns
                    Not used if actual_design_mat is given
        hist_filter_len: length of history filter (in bins)
        coupling_filter_len: length of coupling filter (in bins)
        stim_filter_len: length of stimulus filter (in bins)
        basis_kwargs: keyword arguments for basis functions
        actual_design_mat: design matrix to use for fitting
        fit_type: type of fit to use
            'actual': fit to actual data
            'trial_shuffled': fit to trial shuffled data
            'circle_shuffled': fit to circularly shuffled data
            'ranom_shuffled': fit to randomly shuffled data

    Returns:
        res: result of fit
        actual_design_mat: design matrix used for fitting
        held_out_ll: held out log likelihood

    """

    # If not given, generate an actual_design_mat
    # Otherwise use the given one
    if design_mat is None:
            input_dat = data_frame.copy()
            design_mat = utils.dataframe_to_design_mat(
                    input_dat,
                    hist_filter_len,
                    coupling_filter_len,
                    stim_filter_len,
                    basis_kwargs,
                    )

    if fit_type == 'actual':
        pass
    elif fit_type == 'trial_shuffled':
        design_mat = utils.gen_trial_shuffle(design_mat)
    elif fit_type == 'circle_shuffled':
        design_mat = utils.gen_circular_shuffle(design_mat)
    elif fit_type == 'random_shuffled':
        design_mat = utils.gen_random_shuffle(design_mat)


    # Generate train test splits
    train_dat, test_dat = \
            utils.return_train_test_split(
                    design_mat,
                    test_size = 0.25,
                    )

    # Fit model to actual data
    res, pred, train_dat = fit_stim_history_coupled_glm(
                    glmdata = train_dat,
                    hist_filter_len = hist_filter_len,
                    coupling_filter_len = coupling_filter_len,
                    stim_filter_len= stim_filter_len,
                    regularized=False,
                    **basis_kwargs
                    )

    # Calculate log likelihood on test data
    test_pred = res.predict(test_dat[res.params.index])
    test_ll = utils.poisson_ll(test_pred, test_dat['spikes'].values)
    test_ll = np.round(test_ll, 2)

    return res, design_mat, test_ll

def perform_fit_actual_and_trial_shuffled_fit(
		# hist_filter_len = 10,
		# coupling_filter_len = 10,
		# stim_filter_len = 500,
		# basis_kwargs = {},
		design_mat = None,
		):
    """
    Since there can be significant variability in the test set,
    compare the actual fit to a trial shuffled fit on the
    same data set

    Inputs:
        data_frame: pandas dataframe with columns
                    Not used if actual_design_mat is given
        hist_filter_len: length of history filter (in bins)
        coupling_filter_len: length of coupling filter (in bins)
        stim_filter_len: length of stimulus filter (in bins)
        basis_kwargs: keyword arguments for basis functions
        actual_design_mat: design matrix to use for fitting
        fit_type: type of fit to use
            'actual': fit to actual data
            'trial_shuffled': fit to trial shuffled data
            'circle_shuffled': fit to circularly shuffled data
            'ranom_shuffled': fit to randomly shuffled data

    Returns:
        res: result of fit
        actual_design_mat: design matrix used for fitting
        held_out_ll: held out log likelihood

    """
    # Check for nan values
    if design_mat.isnull().values.any():
        print('Nan values in design matrix...dropping')
        design_mat = design_mat.dropna()

    # Generate train test splits
    train_dat, test_dat = \
            utils.return_train_test_split(
                    design_mat,
                    test_size = 0.25,
                    )

    actual_train_dat = train_dat.copy()
    trial_shuffled_train_dat = utils.gen_trial_shuffle(train_dat)

    # Check similarity of trial shuffled data
    # np.mean(
    #         actual_train_dat.values == trial_shuffled_train_dat.values, 
    #         axis = 0)

    # Fit model to actual data
    out_list = []
    train_dat_list = [actual_train_dat, trial_shuffled_train_dat]
    train_dat_names = ['actual', 'trial_shuffled']
    for this_train_dat in train_dat_list:
        #res, pred, train_dat = fit_stim_history_coupled_glm(
        outs = fit_stim_history_coupled_glm(
                        glmdata = this_train_dat,
                        # hist_filter_len = hist_filter_len,
                        # coupling_filter_len = coupling_filter_len,
                        # stim_filter_len= stim_filter_len,
                        # regularized=False,
                        # **basis_kwargs
                        )
        out_list.append(outs)

    # Calculate log likelihood on test data
    # Iterate over actual fit and trial shuffled fit
    res_list = [out[0] for out in out_list]
    test_dat_list = []
    ll_list = []
    for this_res, res_name in zip(res_list, train_dat_names): 
        test_pred = this_res.predict(test_dat[this_res.params.index])
        test_ll = utils.poisson_ll(test_pred, test_dat['spikes'].values)
        test_ll = np.round(test_ll, 2)
        ll_list.append(test_ll)
        test_dat['pred_spikes'] = test_pred
        test_dat_list.append(test_dat)

    zipped_outs = list(zip(*out_list)) # res, pred, train_dat

    # Return res, train_dat, ll_list
    return zipped_outs[0], zipped_outs[2], ll_list, test_dat_list 
