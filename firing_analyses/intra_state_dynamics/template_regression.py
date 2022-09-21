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
from scipy.ndimage import gaussian_filter1d as gauss_filt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from functools import reduce
from numpy import dot
from numpy.linalg import norm

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'intra_state_dynamics/plots'

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
basis_funcs = basis_funcs / norm(basis_funcs,axis=-1)[:,np.newaxis] 

vz.imshow(basis_funcs);plt.show()

nrns = 10
trials = 15
sim_w = np.random.random(size = (nrns,states))

# Similarity of projection vectors
sim_w_norm = norm(sim_w,axis=0)
norm_mat = sim_w_norm[:,np.newaxis].dot(sim_w_norm[np.newaxis,:])
w_similarity = sim_w.T.dot(sim_w)/norm_mat
plt.matshow(w_similarity);plt.colorbar();plt.show()

firing = np.matmul(sim_w, basis_funcs)*10
#vz.imshow(firing);plt.show()
firing_array = np.tile(firing[:,np.newaxis], (1,trials,1))
firing_array = firing_array + np.random.randn(*firing_array.shape)*0.1
mean_firing = np.mean(firing_array, axis=1)
img_kwargs = dict(interpolation = 'nearest', aspect = 'auto')
#plt.imshow(mean_firing, **img_kwargs);plt.show()
#plt.plot(mean_firing.T);plt.show()
#long_firing = np.tile(firing, (1,trials))
long_firing = np.reshape(firing_array, (len(firing_array),-1)) 
long_firing = np.abs(long_firing)
vz.imshow(long_firing);plt.colorbar();plt.show()

long_spikes = np.random.random(long_firing.shape)*long_firing.max(axis=None) < long_firing 
vz.imshow(long_spikes);plt.show()
long_firing = long_spikes.copy()

long_template = np.tile(basis_funcs, (1,trials))

## Inference
def estimate_weights(firing, template):
    estim_weight = firing.dot(np.linalg.pinv(template))
    return estim_weight

#estim_weights = estimate_weights(long_firing, long_template)
estim_weights = estimate_weights(long_spikes, long_template)

fig,ax = plt.subplots(1,2)
ax[0].matshow(sim_w, **img_kwargs)
ax[1].matshow(estim_weights, **img_kwargs)
ax[0].set_title('Orig Weights')
ax[1].set_title('Inferred Weights')
plt.show()

trial_proj = np.stack(
        [np.linalg.pinv(estim_weights).dot(x) for x in \
                firing_array.swapaxes(0,1)]
        ).swapaxes(0,1)
mean_trial_proj = trial_proj.mean(axis=1)
plt.plot(mean_trial_proj.T);plt.show()

proj = np.linalg.pinv(estim_weights).dot(long_firing) 
template_norm = norm(long_template,axis=-1)[:,np.newaxis]
proj_norm = norm(proj,axis=-1)[:,np.newaxis]
proj_sim = dot(long_template, proj.T) / (template_norm.dot(proj_norm.T))

similarity_norm = np.round(norm(np.diag(proj_sim)),4)
similarity_met = np.round(similarity_norm / np.sqrt(states),4)

plt.matshow(proj_sim)
plt.colorbar()
plt.suptitle('Similarity between projetion and reconstruction' + '\n' + \
        f'Similarity : {similarity_met}')
plt.show()

fig,ax = plt.subplots(2,1, sharex=True, sharey=True)
ax[0].imshow(long_template, **img_kwargs)
ax[1].imshow(proj, **img_kwargs)
plt.show()

## Select neurons by specific epoch projection
weight_ratio = estim_weights / estim_weights.sum(axis=-1)[:,np.newaxis]
weight_frame = pd.DataFrame(weight_ratio, columns = [str(x) for x in range(states)])

epoch = 1
wanted_nrn = weight_frame.sort_values(str(epoch), ascending=False).index[0]
plt.plot(mean_firing[wanted_nrn]);plt.show()

############################################################
# Project actual data
############################################################
file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in file_list]

#normal_firing_list = []
spike_list = []
#ind = 3
for ind in trange(len(file_list)):
    dat = ephys_data(file_list[ind])
    dat.get_spikes()
    #dat.firing_rate_params = dat.default_firing_params
    #dat.get_firing_rates()
    #this_firing = dat.firing_array
    #this_normal_firing = dat.firing_array.copy()
    # Zscore
    #mean_vals = this_normal_firing.mean(axis=(2,3))
    #std_vals = this_normal_firing.std(axis=(2,3))
    #this_normal_firing = this_normal_firing - np.expand_dims(mean_vals, (2,3)) 
    #this_normal_firing = this_normal_firing / np.expand_dims(std_vals, (2,3)) 
    #normal_firing_list.append(this_normal_firing)
    spike_list.append(np.array(dat.spikes))

flat_spikes = [x for y in spike_list for x in y]
nrn_count = [x.shape[1] for x in flat_spikes]
flat_basenames = [[basenames[s_num] + f'_{num}' \
                        for num in range(spike_list[s_num].shape[0])] \
                        for s_num in range(len(basenames))]
flat_basenames = [x for y in flat_basenames for x in y]

time_lims = [2000,4000]
#step_size = dat.firing_rate_params['step_size']
#wanted_inds = np.arange(time_lims[0], time_lims[1]) / step_size 
#wanted_inds = np.unique(np.vectorize(np.int)(wanted_inds))

#this_dat = normal_firing_list[0][0]
#this_dat = this_dat[..., wanted_inds]
all_similarity_met = []
all_similarity_raw = []
all_scaled_weights = []
for i in trange(len(flat_spikes)):
    #this_dat = spike_list[0][0,...,time_lims[0]:time_lims[1]]
    this_dat = flat_spikes[i][...,time_lims[0]:time_lims[1]] 
    this_dat = gauss_filt(this_dat, 50, axis=-1)
    mean_firing = this_dat.mean(axis=0)
    std_firing = this_dat.std(axis=0)
    #vz.imshow(mean_firing);plt.show()
    #vz.firing_overview(this_dat.swapaxes(0,1));plt.show()

    #this_basis = basis_funcs[:,::step_size]
    this_basis = basis_funcs
    this_basis = this_basis / this_basis.sum(axis=-1)[:,np.newaxis]
    #vz.imshow(this_basis);plt.show()

    long_dat = np.reshape(this_dat.swapaxes(0,1), (this_dat.shape[1],-1))
    long_basis = np.tile(this_basis, (1, this_dat.shape[0]))

    estim_weights = estimate_weights(long_dat, long_basis)
    min_weights = estim_weights.min(axis=-1)
    max_weights = estim_weights.max(axis=-1)
    range_weights = max_weights - min_weights
    scaled_weights = (estim_weights - min_weights[:,np.newaxis])/range_weights[:,np.newaxis]
    all_scaled_weights.append(scaled_weights)


    #vz.firing_overview(this_dat)

    #fig,ax = plt.subplots(1,2)
    #im = ax[0].matshow(estim_weights)
    #fig.colorbar(im, ax=ax[0])
    #im = ax[1].matshow(scaled_weights)
    #fig.colorbar(im, ax=ax[1])
    #plt.show()

    ############################################################
    ## Reconstruct activity
    # Low Dim Projetion
    proj = np.linalg.pinv(estim_weights).dot(long_dat) 

    #fig,ax = plt.subplots(2,1)
    #ax[0].imshow(long_dat, **img_kwargs)
    #ax[1].imshow(proj, **img_kwargs)
    #plt.show()

    template_norm = norm(long_basis,axis=-1)[:,np.newaxis]
    proj_norm = norm(proj,axis=-1)[:,np.newaxis]
    proj_sim = dot(long_basis, proj.T) / (template_norm.dot(proj_norm.T))

    similarity_norm = np.round(norm(np.diag(proj_sim)),4)
    similarity_met = np.round(similarity_norm / np.sqrt(states),4)
    all_similarity_raw.append(np.diag(proj_sim))
    all_similarity_met.append(similarity_met)

    #plt.matshow(proj_sim)
    #plt.colorbar()
    #plt.suptitle('Similarity between projetion and reconstruction' + '\n' + \
    #        f'Similarity : {similarity_met}')
    #plt.show()

## Select neurons by specific epoch projection
weight_ratio = estim_weights / estim_weights.sum(axis=-1)[:,np.newaxis]
weight_frame = pd.DataFrame(weight_ratio, columns = [str(x) for x in range(states)])

fig,ax = plt.subplots(states,1,sharex=True)
for epoch, this_ax in enumerate(ax):
    #epoch = 1
    wanted_nrn = weight_frame.sort_values(str(epoch), 
            ascending=False).index[0]
    #fig,ax = plt.subplots()
    this_ax.plot(mean_firing[wanted_nrn])
    #this_ax.fill_between(
    #        x = range(len(mean_firing.T)),
    #        y1 = mean_firing[wanted_nrn] + std_firing[wanted_nrn],
    #        y2 = mean_firing[wanted_nrn] - std_firing[wanted_nrn],
    #        alpha = 0.5
    #        )
    this_ax.axvspan(epoch_lims[epoch][0], epoch_lims[epoch][1], 
            color = 'y', alpha = 0.5)
#plt.show()
fig.savefig(os.path.join(plot_dir,'example_single_neurons'))
plt.close(fig)

raw_sim_array = np.stack(all_similarity_raw)

sim_frame = pd.DataFrame(
        dict(
            nrn_count = nrn_count,
            similarity = all_similarity,
            basenames = flat_basenames
            ))

fig, ax = plt.subplots()
sns.regplot(data = sim_frame,
        x = 'nrn_count',
        y = 'similarity',
        ax=ax
        )
ax.scatter(*sim_frame.sort_values('similarity').iloc[-1][['nrn_count','similarity']], color = 'k', s = 100)
ax.scatter(*sim_frame.sort_values('similarity').iloc[0][['nrn_count','similarity']], color = 'k', s = 100)
plt.show()

max_ind = sim_frame.sort_values('similarity').index[-1]
min_ind = sim_frame.sort_values('similarity').index[0]

min_spikes = flat_spikes[min_ind][...,time_lims[0]:time_lims[1]]
max_spikes = flat_spikes[max_ind][...,time_lims[0]:time_lims[1]]

min_firing = gauss_filt(min_spikes, 50, axis=-1)
max_firing = gauss_filt(max_spikes, 50, axis=-1)

mean_min_firing = min_firing.mean(axis=0)
mean_max_firing = max_firing.mean(axis=0)

min_weights = all_scaled_weights[min_ind]
max_weights = all_scaled_weights[max_ind]
min_weights = min_weights / min_weights.sum(axis=-1)[:,np.newaxis]
max_weights = max_weights / max_weights.sum(axis=-1)[:,np.newaxis]

min_rank = np.tile(np.arange(states), (min_weights.shape[0],1))
max_rank = np.tile(np.arange(states), (max_weights.shape[0],1))

min_cog = (min_weights * min_rank).sum(axis=-1)
max_cog = (max_weights * max_rank).sum(axis=-1)

min_order = np.argsort(min_cog)
max_order = np.argsort(max_cog)

min_raw_sim = all_similarity_raw[min_ind]
max_raw_sim = all_similarity_raw[max_ind]

fig,ax = plt.subplots(3,2, sharey = 'row',
        figsize = (7,10))
im = ax[0,0].imshow(mean_min_firing[min_order], **img_kwargs)
fig.colorbar(im, ax=ax[0,0])
im = ax[0,1].imshow(mean_max_firing[max_order], **img_kwargs)
fig.colorbar(im, ax=ax[0,1])
im = ax[1,0].imshow(zscore(mean_min_firing[min_order],axis=-1), **img_kwargs)
fig.colorbar(im, ax=ax[1,0])
im = ax[1,1].imshow(zscore(mean_max_firing[max_order],axis=-1), **img_kwargs)
fig.colorbar(im, ax=ax[1,1])
ax[2,0].bar(np.arange(states), min_raw_sim)
ax[2,1].bar(np.arange(states), max_raw_sim)
ax[0,0].set_title('Min Dynamics')
ax[0,1].set_title('Max Dynamics')
ax[0,0].set_ylabel('Raw Values')
ax[1,0].set_ylabel('Zscore Values')
ax[2,0].set_ylabel('Similarity Per Epoch')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,'min_max_dynamics.png'))
plt.close(fig)
#plt.show()

#plt.scatter(nrn_count, all_similarity);plt.show()


