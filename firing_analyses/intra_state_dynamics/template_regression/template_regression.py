"""
Find loadings for pre-defined temporal functions which approximate epochal durations
If projection of data into template_space is similar to templates, then
population follows the same pattern
"""
import os

import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/projects/pytau')
from ephys_data import ephys_data
import visualize as vz
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#from joblib import Parallel, cpu_count, delayed
import seaborn as sns
from scipy.stats import zscore, mode, spearmanr
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
from glob import glob

import pymc3 as pm
from pytau.changepoint_preprocess import preprocess_single_taste
from pytau.changepoint_model import (single_taste_poisson, 
                                    advi_fit)

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'intra_state_dynamics/template_regression/plots'

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
NM_file_list_path = '/media/bigdata/firing_space_plot/NM_gape_analysis/fin_NM_emg_dat.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
NM_file_list = [x.strip() for x in open(NM_file_list_path,'r').readlines()]
file_list = file_list + NM_file_list
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
all_raw_weights = []
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
    all_raw_weights.append(estim_weights)


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


plt.imshow(this_basis, **img_kwargs)
plt.xlabel('Time post-stim (ms)')
plt.ylabel('Templates')
plt.colorbar()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'templates.png'))
plt.close(fig)

psths = np.concatenate([x.mean(axis=0) for x in flat_spikes], axis=0)
psths = psths[...,time_lims[0]:time_lims[1]]
firing = gauss_filt(psths, 50, axis =-1)
weight_array = np.concatenate(all_scaled_weights, axis=0)

## Select neurons by specific epoch projection
weight_ratio = weight_array / weight_array.sum(axis=-1)[:,np.newaxis]
weight_frame = pd.DataFrame(weight_ratio, columns = [str(x) for x in range(states)])

count = 5
fig,ax = plt.subplots(states,count,sharex=True)
for epoch, this_ax_row in enumerate(ax):
    #epoch = 1
    wanted_nrns = weight_frame.sort_values(str(epoch), 
            ascending=False).index[:count]
    for this_nrn, this_ax in zip(wanted_nrns, this_ax_row):
        this_ax.plot(firing[this_nrn])
        this_ax.axvspan(epoch_lims[epoch][0], epoch_lims[epoch][1], 
                color = 'y', alpha = 0.5)
        this_ax.axis('off')
#plt.show()
fig.savefig(os.path.join(plot_dir,'example_single_neurons'))
plt.close(fig)

raw_sim_array = np.stack(all_similarity_raw)

sim_frame = pd.DataFrame(
        dict(
            nrn_count = nrn_count,
            similarity = all_similarity_met,
            basenames = flat_basenames
            ))
sim_frame['experimenter'] = [x[:2] for x in sim_frame['basenames']]
mean_sim_exp = sim_frame.groupby('experimenter').mean()['similarity']

fig,ax = plt.subplots(figsize = (15,5))
im = ax.scatter(sim_frame.basenames, sim_frame.similarity, c = sim_frame.nrn_count)
NM_inds = np.where(sim_frame.basenames.str.contains('NM'))
NM_vals = sim_frame.basenames.loc[NM_inds]
#plt.axvspan(NM_vals.iloc[0], NM_vals.iloc[-1], color = 'y', alpha = 0.3)
ax.axvline(NM_vals.iloc[0], color = 'red', alpha = 0.3)
ax.plot([sim_frame.basenames.iloc[0], NM_vals.iloc[0]], 
        [mean_sim_exp[0], mean_sim_exp[0]], 
        color = 'k', alpha = 0.3) 
ax.plot([NM_vals.iloc[0], NM_vals.iloc[-1]],
        [mean_sim_exp[1], mean_sim_exp[1]],
        color = 'k', alpha = 0.3) 
plt.xticks(rotation = 90)
fig.tight_layout()
fig.colorbar(im, ax=ax, label = 'Neuron Count')
#plt.show()
fig.savefig(os.path.join(plot_dir,'across_session_comparison.png'))
plt.close(fig)

fig, ax = plt.subplots()
sns.regplot(data = sim_frame,
        x = 'nrn_count',
        y = 'similarity',
        ax=ax
        )
ax.scatter(*sim_frame.sort_values('similarity').iloc[-1][['nrn_count','similarity']], color = 'k', s = 100)
ax.scatter(*sim_frame.sort_values('similarity').iloc[0][['nrn_count','similarity']], color = 'k', s = 100)
fig.savefig(os.path.join(plot_dir,'nrns_vs_dynamics.png'))
plt.close(fig)
#plt.show()

max_ind = sim_frame.sort_values('similarity').index[-1]
min_ind = sim_frame.sort_values('similarity').index[0]

min_spikes_raw = flat_spikes[min_ind]
max_spikes_raw = flat_spikes[max_ind]

min_spikes = min_spikes_raw[...,time_lims[0]:time_lims[1]]
max_spikes = max_spikes_raw[...,time_lims[0]:time_lims[1]]

min_firing_raw = gauss_filt(min_spikes_raw, 50, axis=-1)
max_firing_raw = gauss_filt(max_spikes_raw, 50, axis=-1)

min_firing = min_firing_raw[...,time_lims[0]:time_lims[1]]
max_firing = max_firing_raw[...,time_lims[0]:time_lims[1]]

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

min_trial_proj = np.stack([np.linalg.pinv(all_raw_weights[min_ind]).dot(x) \
        for x in min_firing])
max_trial_proj = np.stack([np.linalg.pinv(all_raw_weights[max_ind]).dot(x) \
        for x in max_firing])

############################################################
min_mean_proj = min_trial_proj.mean(axis=0)
max_mean_proj = max_trial_proj.mean(axis=0)

fig,ax = plt.subplots(5,2, sharey = 'row',
        figsize = (7,10))
im = ax[0,0].imshow(mean_min_firing[min_order], **img_kwargs)
fig.colorbar(im, ax=ax[0,0])
im = ax[0,1].imshow(mean_max_firing[max_order], **img_kwargs)
fig.colorbar(im, ax=ax[0,1])
im = ax[1,0].imshow(zscore(mean_min_firing[min_order],axis=-1), **img_kwargs)
fig.colorbar(im, ax=ax[1,0])
im = ax[1,1].imshow(zscore(mean_max_firing[max_order],axis=-1), **img_kwargs)
fig.colorbar(im, ax=ax[1,1])
ax[2,0].imshow(zscore(min_mean_proj,axis=-1), **img_kwargs)
ax[2,1].imshow(zscore(max_mean_proj,axis=-1), **img_kwargs)
ax[3,0].plot(zscore(min_mean_proj,axis=-1).T, linewidth = 2)
ax[3,1].plot(zscore(max_mean_proj,axis=-1).T, linewidth = 2)
ax[4,0].bar(np.arange(states), min_raw_sim)
ax[4,1].bar(np.arange(states), max_raw_sim)
ax[0,0].set_title('Min Dynamics')
ax[0,1].set_title('Max Dynamics')
ax[0,0].set_ylabel('Raw Values')
ax[1,0].set_ylabel('Zscore Values')
ax[2,0].set_ylabel('Template Recounstruction')
ax[3,0].set_ylabel('Template Recounstruction')
ax[4,0].set_ylabel('Similarity Per Epoch')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,'min_max_dynamics.png'))
plt.close(fig)
#plt.show()

#plt.scatter(nrn_count, all_similarity);plt.show()
############################################################
## Fit Changepoints to all datasets and calculate correlation between 
## dynamicity and transition variance
############################################################
############################################################
# Load Params
############################################################
data_dir = '/media/bigdata/firing_space_plot/firing_analyses/intra_state_dynamics/template_regression/'
job_file_path = os.path.join(data_dir, 'fit_params.json')
input_params = pd.read_json(f'{job_file_path}').T
############################################################
# Define Model Params 
############################################################
model_outs_dir = os.path.join(data_dir, 'model_outputs')
if not os.path.exists(model_outs_dir):
    os.makedirs(model_outs_dir)

model_parameters_keys = ['states','fit','samples', 'model_kwargs']
preprocess_parameters_keys = ['time_lims','bin_width','data_transform']

model_parameters = dict(zip(model_parameters_keys, 
                        input_params[model_parameters_keys].iloc[0]))
preprocess_parameters = dict(zip(preprocess_parameters_keys, 
                        input_params[preprocess_parameters_keys].iloc[0]))

model_parameters['states'] = states
preprocess_parameters['time_lims'] = time_lims

#mode_tau_list = []
#raw_tau_list = []
#for num, this_spikes in enumerate(tqdm(flat_spikes)):
#    try:
#        pre_dat = preprocess_single_taste(this_spikes, **preprocess_parameters)
#        model = single_taste_poisson(pre_dat, **model_parameters)
#        with model:
#            inference = pm.ADVI('full-rank')
#            approx = pm.fit(n=model_parameters['fit'], method=inference)
#            trace = approx.sample(draws=model_parameters['samples'])
#        #tau_trial = np.squeeze(mode(np.vectorize(np.int)(trace['tau_trial']))[0])
#        raw_tau = trace['tau']
#        mode_tau = np.squeeze(mode(np.vectorize(np.int)(raw_tau))[0])
#        mode_tau_list.append(mode_tau)
#        raw_tau_list.append(raw_tau)
#        np.save(os.path.join(model_outs_dir, f'raw_tau_{num}.npy'), raw_tau, allow_pickle = True)
#        np.save(os.path.join(model_outs_dir, f'mode_tau_{num}.npy'), mode_tau, allow_pickle = True)
#    except:
#        print(f'Error with run {num}')

# Get inds
raw_file_list = glob(os.path.join(model_outs_dir, '*raw_tau*'))
mode_file_list = glob(os.path.join(model_outs_dir, '*mode_tau*'))
raw_tau_list = [np.load(x) for x in raw_file_list]
mode_tau_list = [np.load(x) for x in mode_file_list]
inds = [os.path.basename(x).split('_')[-1].split('.')[0] for x in raw_file_list]
inds = [int(x) for x in inds]
sort_inds = np.argsort(inds)
raw_tau_list = [raw_tau_list[x] for x in sort_inds]
mode_tau_list = [mode_tau_list[x] for x in sort_inds]
inds = [inds[x] for x in sort_inds]

#inds = sorted([int(x) for x in inds])

tau_var = [np.std(x,axis=0) for x in raw_tau_list]
mean_tau_var = [x.mean(axis=None) for x in tau_var]
std_tau_var = [x.std(axis=None) for x in tau_var]

sim_frame.loc[inds, 'mean_var'] = mean_tau_var
sim_frame.loc[inds, 'std_var'] = std_tau_var


AM_frame = sim_frame.loc[sim_frame['experimenter'] == 'AM']
sim_var_corr = spearmanr(AM_frame.similarity, AM_frame.mean_var)
sim_var_corr = [np.round(x,4) for x in sim_var_corr]

# Only use AM data because NM data also has laser
sns.regplot(
        x = 'similarity',
        y = 'mean_var',
        data = AM_frame 
        )
plt.errorbar(x = AM_frame.similarity,
                y = AM_frame.mean_var,
                yerr = AM_frame.std_var,
                fmt = 'o',
                zorder = -10)
fig = plt.gcf()
plt.xlabel('Dynamicity')
plt.ylabel('Changepoint uncertainty')
plt.title('SpearmanR' + '\n' + f'Corr : {sim_var_corr[0]}, p_val : {sim_var_corr[1]}')
#plt.show()
fig.savefig(os.path.join(plot_dir,'dynamicity_vs_changepoint_uncertainty.png'),
        dpi = 300)
plt.close(fig)

#inds = np.array(list(np.ndindex(tau_var.shape)))
#tau_var_frame = pd.DataFrame(
#        dict(
#            session = np.array(['min','max'])[inds[:,0]],
#            trial = inds[:,1],
#            transition = inds[:,2],
#            var = tau_var.flatten()
#            )
#        )

############################################################
## Fit Changepoints to min and max datasets and compare transitions
############################################################

mode_tau_list = []
raw_tau_list = []
for this_spikes in [min_spikes_raw, max_spikes_raw]:
    pre_dat = preprocess_single_taste(this_spikes, **preprocess_parameters)
    model = single_taste_poisson(pre_dat, **model_parameters)
    with model:
        inference = pm.ADVI('full-rank')
        approx = pm.fit(n=model_parameters['fit'], method=inference)
        trace = approx.sample(draws=model_parameters['samples'])
    #tau_trial = np.squeeze(mode(np.vectorize(np.int)(trace['tau_trial']))[0])
    raw_tau = trace['tau']
    mode_tau = np.squeeze(mode(np.vectorize(np.int)(raw_tau))[0])
    mode_tau_list.append(mode_tau)
    raw_tau_list.append(raw_tau)


tau_var = np.stack([np.std(x,axis=0) for x in raw_tau_list])
inds = np.array(list(np.ndindex(tau_var.shape)))
tau_var_frame = pd.DataFrame(
        dict(
            session = np.array(['min','max'])[inds[:,0]],
            trial = inds[:,1],
            transition = inds[:,2],
            var = tau_var.flatten()
            )
        )

sns.catplot(data = tau_var_frame,
        x = 'transition',
        y = 'var',
        col = 'session',
        kind = 'box'
        )
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'min_max_var_comparison.png'))
plt.close(fig)
#plt.show()

def return_state_array(tau, trial_len):
    state_array = np.zeros((tau.shape[0], trial_len))
    for i in range(tau.shape[-1]-1):
        for trial in range(state_array.shape[0]):
            state_lims = (tau[trial,i], tau[trial,i+1])
            state_array[trial,state_lims[0]:state_lims[1]] = i
    return state_array

mode_tau_list = np.stack(mode_tau_list)
cat_tau = np.concatenate(
        [
            np.tile(np.array([0]), (2,len(mode_tau),1)),
            mode_tau_list,
            np.tile(np.array(pre_dat.shape[-1]), (2, len(mode_tau),1))
        ],
        axis=2)
cat_tau_scaled = (cat_tau / np.max(cat_tau,axis=None)) * int(np.abs(np.diff(time_lims)))
mode_tau_scaled = (mode_tau_list / pre_dat.shape[-1]) * int(np.abs(np.diff(time_lims)))
state_array_list = [return_state_array(x, pre_dat.shape[-1]) for x in cat_tau]

fig,ax = plt.subplots(1,2)
ax[0].imshow(state_array_list[0], **img_kwargs)
ax[1].imshow(state_array_list[1], **img_kwargs)
fig.savefig(os.path.join(plot_dir,'min_max_transition_latency_comparison.png'))
plt.close(fig)
#plt.show()

cmap = plt.get_cmap('tab10')
spike_dat = [min_spikes, max_spikes]
fig, ax = plt.subplots(len(mode_tau), len(spike_dat), sharex=True)
for i in range(len(mode_tau)):
    for j in range(len(spike_dat)):
        ax[i,j] = vz.raster(ax[i,j], spike_dat[j][i], marker = '|', color = 'k')
        #ax[i,1] = vz.raster(ax[i,1], max_spikes[i], marker = '|')
        this_tau = cat_tau_scaled[j,i]
        for k in range(len(this_tau) - 1):
            ax[i,j].axvspan(this_tau[k], this_tau[k+1], color = cmap(k), alpha = 0.3, zorder = -1)
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
#plt.tight_layout()
plt.show()

############################################################
# Transition aligned changes in population activity
############################################################
def return_pca_trials(array, n_components = 4):
    long_array = np.reshape(array, (len(array),-1))
    pca_obj = PCA(n_components = n_components).fit(long_array.T)
    temp = np.stack([pca_obj.transform(x.T).T for x in array.swapaxes(0,1)])
    return temp.swapaxes(0,1) 

firing_list = [ min_firing_raw[...,time_lims[0]-200 : time_lims[1]+200], 
                max_firing_raw[...,time_lims[0]-200 : time_lims[1]+200]]
spike_list = [ min_spikes_raw[...,time_lims[0]-200 : time_lims[1]+200], 
                max_spikes_raw[...,time_lims[0]-200 : time_lims[1]+200]]
firing_list = [np.swapaxes(x,0,1) for x in firing_list]
spike_list = [np.swapaxes(x,0,1) for x in spike_list]
firing_pca = [return_pca_trials(x) for x in firing_list]

fig,ax = plt.subplots(2,1)
for this_dat, this_ax in zip(firing_pca, ax):
    this_ax.plot(this_dat.mean(axis=1).T, linewidth = 2)
ax[0].set_title('Min Dynamics')
ax[1].set_title('Max Dynamics')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,'min_max_raw_pca_comparison.png'))
plt.close(fig)
#plt.show()

window = 200 #ms
temp_mode_tau_scaled = mode_tau_scaled + 200
aligned_pc_list = []
#for this_tau, this_pca in zip(temp_mode_tau_scaled, firing_pca):
for this_tau, this_pca in zip(temp_mode_tau_scaled, spike_list):
    this_aligned_pca = np.zeros((*this_pca.shape[:2], states-1, window*2))
    for this_trial in range(len(mode_tau)):
        for this_transition in range(len(mode_tau.T)):
            lat = int(this_tau[this_trial, this_transition])
            min_lim = np.max([0, lat - window])
            max_lim = np.min([firing_pca[0].shape[-1], lat + window])
            snippet = this_pca[:, this_trial, min_lim : max_lim]
            this_aligned_pca[:,this_trial, this_transition] = snippet
    aligned_pc_list.append(this_aligned_pca)
#aligned_pc_list = np.stack(aligned_pc_list)

mean_aligned_pc = [x.mean(axis=1).swapaxes(0,1) for x in aligned_pc_list]
#mean_aligned_pc = aligned_pc_list.mean(axis=2).swapaxes(1,2)
#mean_aligned_firing = [gauss_filt(x,25,axis=-1) for x in mean_aligned_pc]
fig, ax = plt.subplots(len(firing_list), states-1)
inds = list(np.ndindex(ax.shape))
for this_ind in inds:
    #this_firing = mean_aligned_firing[this_ind[0]][this_ind[1]]
    #ax[this_ind].imshow(zscore(mean_aligned_pc[this_ind],axis=-1),
    #        **img_kwargs)
    ax[this_ind].plot(zscore(mean_aligned_pc[this_ind],axis=-1).T)
    #ax[this_ind].plot(this_firing.T)
ax[0,0].set_ylabel('Min Dynamics')
ax[1,0].set_ylabel('Max Dynamics')
fig.savefig(os.path.join(plot_dir,'aligned_pcs.png'))
plt.close(fig)
#plt.show()

fig, ax = plt.subplots(len(firing_list), states-1, 
        figsize = (7,7),sharex=True, sharey = 'row')
inds = list(np.ndindex(ax.shape))
for this_ind in inds:
    ax[this_ind].scatter(
            *np.where(mean_aligned_pc[this_ind[0]][this_ind[1]])[::-1],
            marker = '|',
            alpha = 0.5)
fig.savefig(os.path.join(plot_dir,'aligned_spikes.png'))
plt.close(fig)
#plt.show()

# For max data, sort neurons by magnitude of change
max_spikes = mean_aligned_pc[1]
time_split = np.stack(np.array_split(max_spikes, 2, axis=-1)).mean(axis=-1)
change = time_split[0] / time_split[1]
sorted_order = [np.argsort(x)[::-1] for x in change]

sorted_max_spikes = np.stack([x[y] for x,y in zip(max_spikes, sorted_order)])

#plt.plot(gauss_filt(sorted_max_spikes[:,0],3, axis=-1).T);plt.show()
fig, ax = plt.subplots(1, states-1, 
        figsize = (7,3),sharex=True, sharey = 'row')
for this_ax, this_dat in zip(ax, sorted_max_spikes):
    this_ax.scatter(
            *np.where(this_dat)[::-1],
            marker = '|',
            alpha = 0.5)
plt.show()

fig, ax = plt.subplots(1, states-1, 
        figsize = (7,3),sharex=True, sharey = 'row')
for this_ax, this_dat in zip(ax, sorted_max_spikes):
    this_ax.imshow(gauss_filt(this_dat, 5, axis=-1),
            interpolation = 'nearest', aspect = 'auto',
            cmap = 'jet')
plt.show()

#fig.savefig(os.path.join(plot_dir,'aligned_spikes.png'))
#plt.close(fig)
