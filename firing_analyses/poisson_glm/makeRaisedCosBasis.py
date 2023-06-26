import numpy as np
from scipy.stats import zscore
import pandas as pd

# Generate data
def raisedCosFun(x, ctr, dCtr):
    """
    Raised cosine function

    args:
        x: input
        ctr: center
        dCtr: width (allows assymetry)

    returns:
        y: output
    """
    if not isinstance(dCtr, list):
        dCtr = np.array([dCtr,dCtr])
    left_x = x[x < ctr]
    right_x = x[x >= ctr]
    y_left = 0.5 * (1 + np.cos(np.pi * (left_x - ctr) / dCtr[0]))
    y_right = 0.5 * (1 + np.cos(np.pi * (right_x - ctr) / dCtr[1]))
    y = np.concatenate([y_left, y_right])
    y[x < (ctr - dCtr[0])] = 0
    y[x > (ctr + dCtr[1])] = 0
    return y

def gen_raised_cosine_basis(n, n_basis, spread = 'linear'):
    """
    Generate raised cosine basis functions

    args:
        n: number of time bins
        dt: time bin size
        n_basis: number of basis functions

    returns:
        basis: n_basis x n matrix of basis functions
    """
    if spread == 'linear':
        ctrs = np.linspace(0, n-2, n_basis)[1:-1]
        ctrs = np.concatenate([[0], ctrs, [n]])
        dctrs = np.diff(ctrs)[0]
        basis_funcs = np.stack([raisedCosFun(np.arange(n), ctr, dctrs) for ctr in ctrs])


    if spread == 'log':
        ctrs = np.logspace(0, np.log10(n-1), n_basis)[1:-1]
        ctrs = np.concatenate([[0], ctrs, [n]])
        dctrs = np.diff(ctrs)
        # make assymetric functions such that touching sides have same width
        dctrs = np.concatenate([[0.1], dctrs, [0.1]])
        dctrs = np.stack([dctrs[:-1], dctrs[1:]]).T
        basis_funcs = np.stack(
                [
                    raisedCosFun(
                        np.arange(n), 
                        this_ctr, 
                        list(this_dctrs)) for this_ctr, this_dctrs in zip(ctrs, dctrs)])


    return basis_funcs

if __name__ == '__main__':
    import pylab as plt
    linear_basis_funcs = gen_raised_cosine_basis(1000, 10, spread = 'linear')
    log_basis_funcs = gen_raised_cosine_basis(1000, 10, spread = 'log')

    fig, ax = plt.subplots(2, 1, sharex = True, sharey = True)
    ax[0].plot(linear_basis_funcs.T,color = 'k')
    ax[0].plot(linear_basis_funcs.sum(axis = 0), color = 'r', linewidth = 3)
    ax[0].set_title('Linearly spaced basis functions')
    ax[1].plot(log_basis_funcs.T,color = 'k')
    ax[1].plot(log_basis_funcs.sum(axis = 0), color = 'r', linewidth = 3)
    ax[1].set_title('Logarithmically spaced basis functions')
    plt.show()

