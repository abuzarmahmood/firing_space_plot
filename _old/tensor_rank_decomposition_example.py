#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:07:43 2019

@author: abuzarmahmood
"""

import tensortools as tt
import numpy as np
import matplotlib.pyplot as plt

# Make synthetic dataset.
I, J, K, R = 25, 25, 25, 4  # dimensions and rank
X = tt.randn_ktensor((I, J, K), rank=R).full()
X += np.random.randn(I, J, K)  # add noise

# Fit CP tensor decomposition (two times).
U = tt.cp_als(X, rank=R, verbose=True)
V = tt.cp_als(X, rank=R, verbose=True)

# Compare the low-dimensional factors from the two fits.
fig, _, _ = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)

# Align the two fits and print a similarity score.
sim = tt.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
print(sim)

# Plot the results again to see alignment.
fig, ax, po = tt.plot_factors(U.factors)
tt.plot_factors(V.factors, fig=fig)

# Show plots.
plt.show()

# =============================================================================
# =============================================================================
# More structured data to visualize decomposition
from sklearn.preprocessing import normalize as normalize

tensor_size = (3,100,25)
mode1 = np.sin(2*np.pi*np.arange(tensor_size[1])/(tensor_size[1]))
mode2 = np.cos(2*np.pi*np.arange(tensor_size[1])/(tensor_size[1]))
mode3 = np.arange(tensor_size[1])**2
mode1mod = np.arange(tensor_size[2])
mode2mod = np.log(np.arange(1,tensor_size[2]+1)**2)
mode3mod = np.sin(2*np.pi*np.arange(tensor_size[2])/(tensor_size[2]))

modes = np.concatenate([mode1[np.newaxis,:],mode2[np.newaxis,:],mode3[np.newaxis,:]],axis=0)
modes_norm = normalize(modes)

modulations = np.concatenate([mode1mod[np.newaxis,:],mode2mod[np.newaxis,:],mode3mod[np.newaxis,:]],axis=0)
modulations_norm = normalize(modulations)

X = np.empty(tensor_size)
for i in range(tensor_size[0]):
    for j in range(tensor_size[1]):
        for k in range(tensor_size[2]):
            X[i,j,k] = modes_norm[i,j]*modulations_norm[i,k] #+ np.random.rand()*0.05

plt.subplot(311)
plt.plot(X[0,:,-1].T)
plt.subplot(312)
plt.plot(X[1,:,-1].T)
plt.subplot(313)
plt.plot(X[2,:,-1].T)

fig,ax = plt.subplots(3,2)
for i in range(3):
    ax[i,0].plot(modes_norm[i,:])
    ax[i,1].plot(modulations_norm[i,:])
# Fit CP tensor decomposition (two times).
U = tt.cp_als(X, rank=3, verbose=True)
V = tt.cp_als(X, rank=3, verbose=True)

# Compare the low-dimensional factors from the two fits.
fig, _, _ = tt.plot_factors(U.factors)
fig2, _, _ = tt.plot_factors(V.factors)