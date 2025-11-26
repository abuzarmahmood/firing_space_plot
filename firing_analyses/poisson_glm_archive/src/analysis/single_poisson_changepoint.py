"""
Functionality to infer single poisson changepoint
for a single timeseries
"""

import numpy as np
import matplotlib.pyplot as plt

##############################
# Generate random data

def gen_data(
        n = None,
        tau = None,
        lam1 = None,
        lam2 = None
        ):
    """
    Generate random data from two poisson distributions

    Parameters
    ----------
    n : int
        Number of data points
    tau : int
        Changepoint
    lam1 : float
        Poisson parameter for the first distribution
    lam2 : float
        Poisson parameter for the second distribution
    """
    
    if n is None:
        n = 100
    if tau is None:
        tau = np.random.randint(0, n)
    if lam1 is None:
        lam1 = np.random.randint(1, 10)
    if lam2 is None:
        lam2 = np.random.randint(1, 10)

    data = np.zeros(n)
    for i in range(n):
        if i < tau:
            data[i] = np.random.poisson(lam1)
        else:
            data[i] = np.random.poisson(lam2)
    return data, tau, lam1, lam2

def plot_data(data, tau, lam1, lam2, ax=None):
    """
    Plot data

    Parameters
    ----------
    data : array
        Data
    tau : int
        Changepoint
    lam1 : float
        Poisson parameter for the first distribution
    lam2 : float
        Poisson parameter for the second distribution
    """
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = ax.get_figure()
    ax.plot(data)
    ax.axvline(tau, color='red')
    ax.hlines(lam1, 0, tau, color='green')
    ax.hlines(lam2, tau, len(data), color='green')
    ax.set_title(f'Changepoint: {tau}, Lam1: {lam1}, Lam2: {lam2}')
    return fig, ax

##############################
# Calculate likelihood

def poisson_ll(data, lam):
    """
    Calculate poisson log likelihood

    Parameters
    ----------
    data : array
        Data
    lam : float
        Poisson parameter
    """
    
    ll = -len(data) * lam + np.sum(data * np.log(lam))
    return ll

def calc_cp_likelihood(data, tau):
    """
    Calculate likelihood of changepoint

    Parameters
    ----------
    data : array
        Data
    tau : int
        Changepoint
    """
    
    lam1 = np.mean(data[:tau])
    lam2 = np.mean(data[tau:])
    ll = poisson_ll(data[:tau], lam1) + poisson_ll(data[tau:], lam2)
    return ll

def infer_cp(data):
    """
    Infer changepoint

    Parameters
    ----------
    data : array
        Data
    """
    
    n = len(data)
    max_ll = -np.inf
    cp = None
    for tau in range(1, n):
        ll = calc_cp_likelihood(data, tau)
        if ll > max_ll:
            max_ll = ll
            cp = tau
    return cp

##############################

if __name__ == '__main__':
    n_repeats = 10

    data_list = []
    cp_list = []
    for i in range(n_repeats):
        outs = gen_data()
        cp = infer_cp(outs[0])
        data_list.append(outs)
        cp_list.append(cp)

    fig, ax = plt.subplots(n_repeats, 1, figsize=(10, 10),
                           sharex=True)
    for i in range(n_repeats):
        plot_data(*data_list[i], ax=ax[i])
        ax[i].axvline(cp_list[i], color='k', 
                      linestyle='--', alpha=0.5,
                      linewidth=2)
    plt.show()

