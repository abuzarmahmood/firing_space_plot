"""
Approximate Bayesian computation (ABC) for firing rate estimation
"""

import numpy as np
import pylab as plt
from tqdm import tqdm, trange
from scipy.stats import poisson

############################################################
# Generate data 
############################################################
n_batches = 500
n_trials = 100
dt = 0.01
x = np.arange(0, 10, dt) 
y = np.abs(np.cumsum(np.random.randn(n_batches, len(x), n_trials), axis=0))

kern_len = 25
kern = np.ones(kern_len) / kern_len
y = np.apply_along_axis(lambda m: np.convolve(m, kern, mode='valid'), axis=1, arr=y)
y *= 50

y = y - np.min(y, axis=1)[:, None]
x_conv = np.convolve(x, kern, mode='valid')

# spikes = np.random.rand(*y.shape) < y * dt
spikes = poisson.rvs(y * dt)

plt.scatter(*np.where(spikes[0]))
plt.show()

plt.imshow(y[0], aspect='auto')
plt.colorbar()
plt.show()

plt.plot(y[0], alpha=0.1)
plt.show()

ind = [1,10]
plt.plot(y[ind[0], :, ind[1]])
plt.plot(spikes[ind[0], :, ind[1]])
plt.show()

############################################################
# Estimate firing rates 
############################################################
test_ind = [0, 2]
test_x = spikes[test_ind[0], :, test_ind[1]]
test_y = y[test_ind[0], :, test_ind[1]]

plt.plot(x_conv, test_x)
plt.plot(x_conv, test_y)
plt.show()

def poisson_ll(x, lam):
    """
    Poisson log-likelihood

    Inputs:
    - x: array of spike counts
    - lam: array of firing rates

    Outputs:
    - ll: log-likelihood
    """
    # Can ignore the factorial term because it's a constant
    ll = np.sum(x * np.log(lam) - lam)
    # ll = poisson.logpmf(x, lam).sum()
    return ll

# Run all y against X
y_reshape = np.moveaxis(y, -1, 1) + 1e-6
y_inds = list(np.ndindex(y_reshape.shape[:-1])) 

ll_list = []
for this_ind in tqdm(y_inds):
    this_y = y_reshape[this_ind]
    ll_list.append(poisson_ll(test_x, this_y))

ll_list = np.array(ll_list)
max_inds = np.argsort(ll_list)[::-1]
max_y_inds = [y_inds[i] for i in max_inds]

n_max = 1
max_y = np.stack([y_reshape[i] for i in max_y_inds])[:n_max]

mean_max_y = np.mean(max_y, axis=0)
std_max_y = np.std(max_y, axis=0)
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(x_conv, test_x)
for this_y in max_y:
    ax[0].plot(x_conv, this_y, alpha=0.1, color='k')
ax[0].plot(x_conv, mean_max_y, color='r', alpha=0.5, label='Inferred firing rate')
ax[0].plot(x_conv, test_y, color='g', alpha=0.5, label='True firing rate')
ax[0].legend()
ax[1].plot(x_conv, mean_max_y)
ax[1].fill_between(x_conv, mean_max_y - std_max_y, mean_max_y + std_max_y, alpha=0.5)
plt.show()

############################################################
# Rolling window ABC

window_size = 100
n_samples = 10000
max_n = 10

summed_rate = np.zeros_like(test_y)
count = 0

window_starts = np.arange(0, len(test_x) - window_size)
for start in tqdm(window_starts):
    this_x = test_x[start:start+window_size]
    # Generate random samples
    sample_y = np.cumsum(np.random.randn(n_samples, window_size), axis=1)
    sample_y = np.abs(sample_y)
    sample_ll = [poisson_ll(this_x, this_y) for this_y in sample_y]
    max_n_inds = np.argsort(sample_ll)[::-1][:max_n]
    max_n_y = sample_y[max_n_inds]
    summed_rate[start:start+window_size] += np.mean(max_n_y, axis=0)
    count += 1

summed_rate /= count

plt.plot(x_conv, test_x)
# plt.plot(x_conv, test_y, label='True firing rate')
plt.plot(x_conv, summed_rate, label='Inferred firing rate')
plt.legend()
plt.show()

plt.plot(this_x)
plt.plot(max_n_y.T, alpha=0.1)
plt.plot(np.mean(max_n_y, axis=0), color='r')
plt.show()
