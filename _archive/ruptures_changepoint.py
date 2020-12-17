#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:50:19 2019

@author: abuzarmahmood
"""

import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt

from sklearn.decomposition import PCA as pca
from sklearn.decomposition import FastICA as ica

# creation of data
n, dim = 500, 3  # number of samples, dimension
n_bkps, sigma = 3, 5  # number of change points, noise standart deviation
signal, bkps = rpt.pw_constant(n, dim, n_bkps, noise_std=sigma)

# change point detection
model = 'normal'#"l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Window(width=40, model=model).fit(signal)
my_bkps = algo.predict(n_bkps=2)
#my_bkps = algo.predict(pen=np.log(n)*dim*sigma**2)
#my_bkps = algo.predict(epsilon=3*n*sigma**2)

# show results
rpt.show.display(signal, bkps, my_bkps, figsize=(10, 6))
plt.show()

# Perform PCA on signal and check changepoint detection
pca_obj = pca(n_components = 1).fit(signal)
reduced_signal = pca_obj.transform(signal)

algo = rpt.Window(width=40, model=model).fit(reduced_signal)
my_bkps = algo.predict(n_bkps=3)
rpt.show.display(reduced_signal, bkps, my_bkps, figsize=(10, 6))
plt.show()

# Compute ICA
ica_obj = ica(n_components=2)
S_ = ica_obj.fit_transform(signal)  # Reconstruct signals

plt.subplot(211)
plt.plot(signal)
plt.subplot(212)
plt.plot(S_)


algo = rpt.Window(width=40, model=model).fit(S_)
my_bkps = algo.predict(n_bkps=3)
rpt.show.display(S_, bkps, my_bkps, figsize=(10, 6))
plt.show()
