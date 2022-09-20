"""
Find loadings for pre-defined temporal functions which approximate epochal durations
If projection of data into template_space is similar to templates, then
population follows the same pattern
"""

import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/projects/pytau')
from ephys_data import ephys_data
import visualize as vz
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
#from joblib import Parallel, cpu_count, delayed
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import mode
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from  matplotlib import colors
import itertools as it
import pymc3 as pm
from scipy.ndimage import gaussian_filter1d as gauss_filt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from functools import reduce
from numpy import dot
from numpy.linalg import norm

############################################################
## Simulation
############################################################
# Time-lims : 0-2000ms
# 4 States
states = 4
epoch_lims = [
        [0,200],
        [200,850],
        [850,1450],
        [1450,2000]
        ]
epoch_lens = np.array([np.abs(np.diff(x)[0]) for x in epoch_lims])
basis_funcs = np.stack([np.zeros(2000) for i in range(4)] )
for this_func, this_lims in zip(basis_funcs, epoch_lims):
    this_func[this_lims[0]:this_lims[1]] = 1
basis_funcs = basis_funcs / epoch_lens[:,np.newaxis]

vz.imshow(basis_funcs);plt.show()

nrns = 10
trials = 15
sim_w = np.random.random(size = (nrns,states))*100

# Similarity of projection vectors
sim_w_norm = norm(sim_w,axis=0)
norm_mat = sim_w_norm[:,np.newaxis].dot(sim_w_norm[np.newaxis,:])
w_similarity = sim_w.T.dot(sim_w)/norm_mat
plt.matshow(w_similarity);plt.colorbar();plt.show()

firing = np.matmul(sim_w, basis_funcs)
vz.imshow(firing);plt.show()

long_firing = np.tile(firing, (1,trials))
long_firing = long_firing + np.random.randn(*long_firing.shape)*0.1
long_firing = np.abs(long_firing)
vz.imshow(long_firing);plt.show()

long_template = np.tile(basis_funcs, (1,trials))

## Inference
def estimate_weights(firing, template):
    estim_weight = firing.dot(np.linalg.pinv(template))
    return estim_weight

estim_weights = estimate_weights(long_firing, long_template)


fig,ax = plt.subplots(1,2)
ax[0].matshow(sim_w)
ax[1].matshow(estim_weights)
ax[0].set_title('Orig Weights')
ax[1].set_title('Inferred Weights')
plt.show()

proj = np.linalg.pinv(estim_weights).dot(long_firing) 
template_norm = norm(long_template,axis=-1)[:,np.newaxis]
proj_norm = norm(proj,axis=-1)[:,np.newaxis]
proj_sim = dot(long_template, proj.T) / (template_norm.dot(proj_norm.T))

similarity_norm = np.round(norm(np.diag(proj_sim)),4)
similarity_met = np.round(similarity_norm / np.sqrt(2),4)

plt.matshow(proj_sim)
plt.colorbar()
plt.suptitle('Similarity between projetion and reconstruction' + '\n' + \
        f'Similarity : {similarity_met}')
plt.show()

img_kwargs = dict(interpolation = 'nearest', aspect = 'auto')
fig,ax = plt.subplots(2,1, sharex=True, sharey=True)
ax[0].imshow(long_template, **img_kwargs)
ax[1].imshow(proj, **img_kwargs)
plt.show()

############################################################
# Project actual data
file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in file_list]

normal_firing_list = []
#ind = 0
for ind in trange(len(file_list[:1])):
    dat = ephys_data(file_list[ind])
    dat.get_spikes()
    dat.firing_rate_params = dat.default_firing_params
    dat.get_firing_rates()
    this_firing = dat.firing_array
    this_normal_firing = dat.firing_array.copy()
    # Zscore
    mean_vals = this_normal_firing.mean(axis=(2,3))
    std_vals = this_normal_firing.std(axis=(2,3))
    this_normal_firing = this_normal_firing - np.expand_dims(mean_vals, (2,3)) 
    this_normal_firing = this_normal_firing / np.expand_dims(std_vals, (2,3)) 
    normal_firing_list.append(this_normal_firing)

time_lims = [2000,4000]
step_size = dat.firing_rate_params['step_size']
wanted_inds = np.arange(time_lims[0], time_lims[1]) / step_size 
wanted_inds = np.unique(np.vectorize(np.int)(wanted_inds))

this_dat = normal_firing_list[0][0]
this_dat = this_dat[..., wanted_inds]

this_basis = basis_funcs[:,::step_size]
this_basis = this_basis / this_basis.sum(axis=-1)[:,np.newaxis]
vz.imshow(this_basis);plt.show()

long_dat = np.reshape(this_dat, (len(this_dat),-1))
long_basis = np.tile(this_basis, (1, this_dat.shape[1]))

estim_weights = estimate_weights(long_dat, long_basis)

vz.firing_overview(this_dat)

fig,ax = plt.subplots(1,2)
im = ax[0].matshow(estim_weights)
fig.colorbar(im, ax=ax[0])
im = ax[1].matshow(np.abs(zscore(estim_weights,axis=-1)))
fig.colorbar(im, ax=ax[1])
plt.show()

############################################################
## Reconstruct activity
# Low Dim Projetion
proj = np.linalg.pinv(estim_weights).dot(long_dat) 
vz.imshow(proj);plt.show()

template_norm = norm(long_basis,axis=-1)[:,np.newaxis]
proj_norm = norm(proj,axis=-1)[:,np.newaxis]
proj_sim = dot(long_basis, proj.T) / (template_norm.dot(proj_norm.T))

similarity_norm = np.round(norm(np.diag(proj_sim)),4)
similarity_met = np.round(similarity_norm / np.sqrt(2),4)

plt.matshow(proj_sim)
plt.colorbar()
plt.suptitle('Similarity between projetion and reconstruction' + '\n' + \
        f'Similarity : {similarity_met}')
plt.show()

#recon = reduce(np.dot,
#        [estim_weights,
#        np.linalg.pinv(estim_weights),
#        long_dat]
#        )
#
#vmin = np.min([long_dat.min(), proj.min()])
#vmax = np.max([long_dat.max(), proj.max()])
#
#img_kwargs = dict(interpolation = 'nearest', aspect = 'auto',
#        vmin = vmin, vmax = vmax)
#fig,ax = plt.subplots(2,1, sharex=True, sharey=True)
#ax[0].imshow(long_dat, **img_kwargs)
#ax[1].imshow(recon, **img_kwargs)
#plt.show()
#
### Per trial projections
#proj_trials = np.stack(
#        [
#            np.linalg.pinv(estim_weights).dot(x) \
#                    for x in this_dat.swapaxes(0,1)
#            ]
#        ).swapaxes(0,1)
#
#vz.firing_overview(this_dat)
#vz.firing_overview(proj_trials)
#plt.show()
