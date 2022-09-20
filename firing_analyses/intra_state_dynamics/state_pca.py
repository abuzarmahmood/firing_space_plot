"""
Mark state transitions on PCA plot 
This will allow us to see
    1) How similar are trials in their population trajectory
        1.1) Can do PCA on data processed for model fit
    2) Whether states of different length are truncated or stretched/squeezed
    ** Can also compare vanilla changepoint to Mixture Changepoint
"""

import sys
import os
import numpy as np
base_dir = '/media/bigdata/projects/pytau'
#sys.path.append(os.path.join(base_dir, 'utils'))
sys.path.append(base_dir)
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz
from pytau.changepoint_io import FitHandler
from pytau.changepoint_preprocess import preprocess_single_taste
from pytau.changepoint_model import (single_taste_poisson, 
                                    advi_fit,
                                    single_taste_poisson_trial_switch)
import pandas as pd
import numpy as np
from glob import glob
import pylab as plt
from sklearn.decomposition import PCA
from scipy.stats import mode,zscore
from scipy.ndimage import gaussian_filter1d as gauss_filt
from sklearn.preprocessing import StandardScaler
import pymc3 as pm


############################################################
# Load Params
############################################################
job_file_path = '/media/bigdata/firing_space_plot/firing_analyses/intra_state_dynamics/fit_params.json'
input_params = pd.read_json(f'{job_file_path}').T

############################################################
# Load Data 
############################################################
data_dir = '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511'
dat = ephys_data(data_dir)
dat.get_spikes()
dat.firing_rate_params = dat.default_firing_params
dat.get_firing_rates()
dat.get_info_dict()

spike_array = np.stack(dat.spikes)
taste_list = dat.info_dict['taste_params']['tastes']

vz.firing_overview(dat.all_normalized_firing)
plt.show()

taste_ind = 0
this_spikes = spike_array[taste_ind]
vz.firing_overview(dat.normalized_firing[taste_ind]);plt.show()

vz.firing_overview(dat.normalized_firing[taste_ind].swapaxes(0,1));plt.show()

############################################################
# Define Model Params 
############################################################
model_parameters_keys = ['states','fit','samples', 'model_kwargs']
preprocess_parameters_keys = ['time_lims','bin_width','data_transform']

model_parameters = dict(zip(model_parameters_keys, 
                        input_params[model_parameters_keys].iloc[0]))
preprocess_parameters = dict(zip(preprocess_parameters_keys, 
                        input_params[preprocess_parameters_keys].iloc[0]))

pre_dat = preprocess_single_taste(this_spikes, **preprocess_parameters)
vz.firing_overview(pre_dat);plt.show()

#model = single_taste_poisson(pre_dat, **model_parameters)
model = single_taste_poisson_trial_switch(
                            pre_dat, 
                            switch_components = 2,
                            states = model_parameters['states'])
with model:
    inference = pm.ADVI('full-rank')
    approx = pm.fit(n=model_parameters['fit'], method=inference)
    trace = approx.sample(draws=model_parameters['samples'])

tau_trial = np.squeeze(mode(np.vectorize(np.int)(trace['tau_trial']))[0])

#outs = advi_fit(model, model_parameters['fit'], model_parameters['samples']) 
#varnames = ['model', 'approx', 'lambda', 'tau', 'data']
#outs = dict(zip(varnames, outs))
tau_array = trace['tau']
int_tau = np.vectorize(np.int)(tau_array)
mode_tau = np.squeeze(mode(int_tau, axis = 0)[0])

lambda_array = trace['selected_trial_lambda'].mean(axis=0).swapaxes(0,1).swapaxes(1,2)
fig, ax = vz.firing_overview(lambda_array)
plt.show()

############################################################
# Plot state durations
#tau_frame = pd.DataFrame(mode_tau, columns = [str(x) for x in range(len(mode_tau.T))])
#tau_frame.sort_values(by = list(tau_frame.columns), inplace = True)
#sorted_tau = tau_frame.values
sorted_tau = mode_tau#[np.argsort(mode_tau[:,-1])[::-1]]
sorted_tau = np.concatenate(
        [
        np.array([0] * len(mode_tau))[:,np.newaxis],
        sorted_tau,
        np.array([pre_dat.shape[-1]] * len(mode_tau))[:,np.newaxis]
        ],
        axis=1
        )

state_array = np.zeros((sorted_tau.shape[0], sorted_tau.max()))
for i in range(sorted_tau.shape[-1]-1):
    for trial in range(state_array.shape[0]):
        state_lims = (sorted_tau[trial,i], sorted_tau[trial,i+1])
        state_array[trial,state_lims[0]:state_lims[1]] = i

vz.imshow(state_array, cmap = 'jet')
plt.axhline(tau_trial - 0.5, color = 'k', linestyle = '--', linewidth = 2)
plt.show()

############################################################
# Smooth pre_dat with gaussian kern
gauss_firing = gauss_filt(this_spikes, sigma = 100, axis=-1)
vz.firing_overview(gauss_firing);plt.show()

gauss_firing_long = gauss_firing.swapaxes(0,1)
gauss_firing_long = np.reshape(gauss_firing_long, (len(gauss_firing_long),-1))
vz.imshow(gauss_firing_long);plt.show()

zscore_obj = StandardScaler().fit(gauss_firing_long.T)
gauss_firing_z_long = zscore_obj.transform(gauss_firing_long.T).T
#vz.imshow(gauss_firing_z_long);plt.show()
zscore_gauss= np.stack([zscore_obj.transform(x.T).T for x in gauss_firing])

vz.firing_overview(gauss_firing);
vz.firing_overview(zscore_gauss);plt.show()

gauss_firing_pre = preprocess_single_taste(zscore_gauss, **preprocess_parameters)
gauss_firing_long_pre = gauss_firing_pre.swapaxes(0,1) 
gauss_firing_long_pre = np.reshape(gauss_firing_long_pre, 
        (len(gauss_firing_long_pre),-1))
vz.firing_overview(gauss_firing_pre);plt.show()
vz.imshow(gauss_firing_long_pre);plt.show()

############################################################
## PCA
pca_obj = PCA(n_components = 10).fit(gauss_firing_long_pre.T)
plt.plot(np.cumsum(pca_obj.explained_variance_ratio_), '-x');plt.show()
pca_obj = PCA(n_components = 3).fit(gauss_firing_long_pre.T)
long_pca_dat = pca_obj.transform(gauss_firing_long_pre.T).T
vz.imshow(long_pca_dat);plt.show()

pca_dat = np.stack([pca_obj.transform(x.T).T for x in gauss_firing_pre])

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0,1,len(mode_tau.T)))
for trans in range(len(mode_tau.T)):
    this_tau = mode_tau[:,trans]
    this_tau_inds = np.stack([trial[:,ind] for trial,ind in zip(pca_dat, this_tau)])
    ax.scatter(
            this_tau_inds[:,0], 
            this_tau_inds[:,1], 
            this_tau_inds[:,2], 
            c = np.stack([colors[trans]]*len(this_tau.T)))
for trial in pca_dat:
    ax.plot(*trial, color = 'grey', alpha = 0.5)
line_cmap = plt.get_cmap('viridis')
line_cols = line_cmap(np.linspace(0,1,pca_dat.shape[-1])) 
mean_line = pca_dat.mean(axis=0)
im = ax.scatter(
        mean_line[0],
        mean_line[1],
        mean_line[2], 
        c = line_cols,
        s = 200)
fig.colorbar(im,ax=ax)
plt.show()

############################################################
## State Specific Firing of Neurons
############################################################
time_lims = preprocess_parameters['time_lims']
bin_width = preprocess_parameters['bin_width']
scaled_tau = ((mode_tau / bin_width)*np.abs(np.diff(time_lims))) 
scaled_tau = np.vectorize(np.int)(scaled_tau)

spike_array = this_spikes[...,time_lims[0]:time_lims[1]]

def chop_by_state(spike_array, tau_array):
    """Calculate firing rates within states given changepoint positions on data

    Args:
        spike_array (3D Numpy array): trials x nrns x bins
        tau_array (2D Numpy array): trials x switchpoints

    Returns:
        Numpy array: Average firing given state bounds
    """

    states = tau_array.shape[-1] + 1
    # Get mean firing rate for each STATE using model
    state_inds = np.hstack([np.zeros((tau_array.shape[0], 1)),
                            tau_array,
                            np.ones((tau_array.shape[0], 1))*spike_array.shape[-1]])
    state_lims = np.array([state_inds[:, x:x+2] for x in range(states)])
    state_lims = np.vectorize(np.int)(state_lims)
    state_lims = np.swapaxes(state_lims, 0, 1)

    #state_firing = \
    #    [[trial_dat[:, start:end]
    #               for start, end in trial_lims]
    #              for trial_dat, trial_lims in zip(spike_array, state_lims)]

    ### Length of states
    #state_dur = np.abs(np.squeeze(np.diff(state_lims,axis=-1)))
    #max_state_dur = np.max(state_dur,axis=None)

    # Register by onset of state
    #reg_state_spikes = np.zeros((states, *spike_array.shape[:-1], max_state_dur)).swapaxes(0,1) 
    #reg_state_spikes_mask = np.ones((states, *spike_array.shape[:-1], max_state_dur)).swapaxes(0,1) 
    reg_state_spikes = np.zeros((states, *spike_array.shape)).swapaxes(0,1) 
    reg_state_spikes_mask = np.ones((states, *spike_array.shape)).swapaxes(0,1) 
    for trial in range(len(state_lims)):
        for state in range(len(state_lims[0])):
            lims = state_lims[trial,state]
            reg_state_spikes[trial,state,:,0:(lims[1]-lims[0])] = \
                    spike_array[trial,:,lims[0]:lims[1]]
            reg_state_spikes_mask[trial,state,:,0:(lims[1]-lims[0])] = 0
    reg_state_spikes = np.ma.array(reg_state_spikes, mask = reg_state_spikes_mask)    
    #reg_state_firing = gauss_filt(reg_state_spikes, sigma = 100, axis=-1) 
    #reg_state_firing = np.ma.array(reg_state_firing, mask = reg_state_spikes_mask)    
    #vz.firing_overview(reg_state_firing[:,:,0].swapaxes(0,1));plt.show()

    # Mask by actual state duration
    state_spikes = np.zeros((states, *spike_array.shape)).swapaxes(0,1) 
    state_spikes_mask = np.ones((states, *spike_array.shape)).swapaxes(0,1) 
    for trial in range(len(state_lims)):
        for state in range(len(state_lims[0])):
            lims = state_lims[trial,state]
            state_spikes[trial,state,:,lims[0]:lims[1]] = \
                    spike_array[trial,:,lims[0]:lims[1]]
            state_spikes_mask[trial,state,:,lims[0]:lims[1]] = 0

    state_spikes = np.ma.array(state_spikes, mask = state_spikes_mask)    

    return state_spikes, reg_state_spikes

    #state_firing = gauss_filt(state_spikes, sigma = 100, axis=-1) 
    #state_firing = np.ma.array(state_firing, mask = state_spikes_mask)    

    #vz.firing_overview(state_firing[:,:,0].swapaxes(0,1));plt.show()

    #fig,ax = plt.subplots(state_firing.shape[2], state_firing.shape[1],
    #        sharex=True, sharey=True)
    #inds = list(np.ndindex(ax.shape))
    #img_kwargs = dict(interpolation = 'nearest', aspect = 'auto')
    #for this_ind in inds:
    #    ax[this_ind].imshow(state_firing[:,this_ind[1], this_ind[0]], **img_kwargs)
    #plt.show()

    nrn = 0
    this_reg_firing = reg_state_firing[:,:,nrn].swapaxes(0,1)
    this_firing = state_firing[:,:,nrn].swapaxes(0,1)
    fig,ax = plt.subplots(2, this_reg_firing.shape[0], sharey = True, sharex=True)
    img_kwargs = dict(interpolation = 'nearest', aspect = 'auto')
    for num in range(states): 
        ax[0,num].imshow(this_firing[num], **img_kwargs)
        ax[1,num].imshow(this_reg_firing[num], **img_kwargs)
    plt.show()
