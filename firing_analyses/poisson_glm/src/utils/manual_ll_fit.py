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
from scipy.sparse import csc_matrix
from scipy.optimize import minimize

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

def neg_log_lik_lnp(theta, X, y):
  """Return -loglike for the Poisson GLM model.

  Args:
    theta (1D array): Parameter vector.
    X (2D array): Full design matrix.
    y (1D array): Data values.

  Returns:
    number: Negative log likelihood.

  """
  # Compute the Poisson log likelihood
  rate = np.exp(X @ theta)
  log_lik = y @ np.log(rate) - sum(rate)

  return - log_lik

def gen_actual_fit_sparse(
        hist_filter_len = 10,
        coupling_filter_len = 10,
        stim_filter_len = 500,
        basis_kwargs = {},
        actual_design_mat = None,
        ):

    # Generate train test splits
    actual_train_dat, actual_test_dat = return_train_test_split(actual_design_mat)
    del actual_test_dat
    # Fit model to actual data
    dv_cols = [x for x in actual_train_dat.columns if 'lag' in x]
    dv_cols.append('intercept')
    x0 = np.random.normal(0, .2, len(dv_cols))
    y = actual_train_dat['spikes'].values
    X = csc_matrix(actual_train_dat[dv_cols].values)
    del actual_train_dat
    man_res = minimize(neg_log_lik_lnp, x0, args = (X,y))
    res = fit_handler(dv_cols, man_res.x)

    return res, actual_design_mat

