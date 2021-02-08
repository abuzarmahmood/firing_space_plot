"""
PyMC3 Blackbox Variational Inference implementation
of Poisson Likelihood Changepoint for spike trains.
- Changepoint distributions are shared across all tastes
- Each taste has it's own emission matrix
"""

########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import os
import sys
import scipy
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt
import theano

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import glob
import json
from scipy.stats import spearmanr
from tqdm import tqdm
from scipy.stats import percentileofscore
import pickle
import tables
import argparse

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('dir_name',  help = 'Directory containing data files')
parser.add_argument('--states', '-s', help = 'Number of States to fit')
args = parser.parse_args()
data_dir = args.dir_name 
#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'

plot_super_dir = os.path.join(data_dir,'changepoint_plots')
if not os.path.exists(plot_super_dir):
        os.makedirs(plot_super_dir)

dat = \
    ephys_data(data_dir)

#dat.firing_rate_params = dat.default_firing_params

dat.get_unit_descriptors()
dat.get_spikes()
#dat.get_firing_rates()
dat.default_stft_params['max_freq'] = 50
fin_spikes = np.array(dat.spikes)
nrn = np.arange(np.array(fin_spikes).shape[2])
taste_dat = np.array(fin_spikes)[:,:,nrn]

##########
# PARAMS 
##########
time_lims = [1500,4000]
bin_width = 10
#states = 4
states = int(args.states)
fit = 40000
samples = 20000

model_save_dir = os.path.join(data_dir,'saved_models',f'vi_{states}_states')
if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
model_name = f'vi_{states}_states_{fit}fit_'\
        f'time{time_lims[0]}_{time_lims[1]}_bin{bin_width}'
plot_dir = os.path.join(plot_super_dir,model_name)
if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

##########
# Bin Data
##########
t_vec = np.arange(taste_dat.shape[-1])
binned_t_vec = np.mean(t_vec[time_lims[0]:time_lims[1]].\
                    reshape((-1,bin_width)),axis=-1)
whole_dat_binned = \
        np.sum(taste_dat.reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = \
        np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
        reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = np.vectorize(np.int)(this_dat_binned)

# Unroll arrays along taste axis
dat_binned_long = np.reshape(this_dat_binned,(-1,*this_dat_binned.shape[-2:]))
whole_dat_binned_long = \
        np.reshape(whole_dat_binned,(-1,*whole_dat_binned.shape[-2:]))

############################################################
#  ____                _         __  __           _      _ 
# / ___|_ __ ___  __ _| |_ ___  |  \/  | ___   __| | ___| |
#| |   | '__/ _ \/ _` | __/ _ \ | |\/| |/ _ \ / _` |/ _ \ |
#| |___| | |  __/ (_| | ||  __/ | |  | | (_) | (_| |  __/ |
# \____|_|  \___|\__,_|\__\___| |_|  |_|\___/ \__,_|\___|_|
############################################################
# Find mean firing for initial values
tastes = this_dat_binned.shape[0]
split_list = np.array_split(this_dat_binned,states,axis=-1)
# Cut all to the same size
min_val = min([x.shape[-1] for x in split_list])
split_array = np.array([x[...,:min_val] for x in split_list])
mean_vals = np.mean(split_array,axis=(2,-1)).swapaxes(0,1)
mean_vals += 0.01 # To avoid zero starting prob
mean_nrn_vals = np.mean(mean_vals,axis=(0,1))

# Find evenly spaces switchpoints for initial values
idx = np.arange(this_dat_binned.shape[-1]) # Index
array_idx = np.broadcast_to(idx, dat_binned_long.shape)
idx_range = idx.max() - idx.min()
even_switches = np.linspace(0,idx.max(),states+1)
even_switches_normal = even_switches/np.max(even_switches)

taste_label = np.repeat([0,1,2,3],30)
trial_num = array_idx.shape[0]
# Being constructing model
with pm.Model() as model:

    # Hierarchical firing rates
    # Refer to model diagram
    # Mean firing rate of neuron AT ALL TIMES
    lambda_nrn = pm.Exponential('lambda_nrn',
                                1/mean_nrn_vals, 
                                shape = (mean_vals.shape[-1]))
    # Priors for each state, derived from each neuron
    # Mean firing rate of neuron IN EACH STATE (averaged across tastes)
    lambda_state = pm.Exponential('lambda_state',
                                    lambda_nrn, 
                                    shape = (mean_vals.shape[1:]))
    # Mean firing rate of neuron PER STATE PER TASTE
    lambda_latent = pm.Exponential('lambda', 
                                    lambda_state[np.newaxis,:,:], 
                                    testval = mean_vals, 
                                    shape = (mean_vals.shape))

    # Changepoint time variable
    # INDEPENDENT TAU FOR EVERY TRIAL
    a = pm.HalfNormal('a_tau', 3., shape = states - 1)
    b = pm.HalfNormal('b_tau', 3., shape = states - 1)

    # Stack produces states x trials --> That gets transposed 
    # to trials x states and gets sorted along states (axis=-1)
    # Sort should work the same way as the Ordered transform --> 
    # see rv_sort_test.ipynb
    tau_latent = pm.Beta('tau_latent', a, b, 
                           shape = (trial_num, states-1),
                           testval = \
                                   tt.tile(even_switches_normal[1:(states)],
                                       (array_idx.shape[0],1))).sort(axis=-1)
           
    tau = pm.Deterministic('tau', idx.min() + (idx.max() - idx.min()) * tau_latent)

    # Sigmoing to create transitions based off tau
    # Hardcoded 3-5 states
    weight_1_stack = tt.nnet.sigmoid(\
            array_idx - tau[:,0][...,np.newaxis,np.newaxis])
    weight_2_stack = tt.nnet.sigmoid(\
            array_idx - tau[:,1][...,np.newaxis,np.newaxis])
    if states > 3:
        weight_3_stack = tt.nnet.sigmoid(\
                array_idx - tau[:,2][...,np.newaxis,np.newaxis])
    if states > 4:
        weight_4_stack = tt.nnet.sigmoid(\
                array_idx - tau[:,3][...,np.newaxis,np.newaxis])

    # Generate firing rates from lambda and sigmoid weights
    if states == 3:
        # 3 states
        lambda_ = np.multiply(1 - weight_1_stack, 
                            lambda_latent[taste_label,0][:,:,np.newaxis]) + \
                np.multiply(weight_1_stack * (1 - weight_2_stack), 
                            lambda_latent[taste_label][:,1][:,:,np.newaxis]) + \
                np.multiply(weight_2_stack, 
                            lambda_latent[taste_label,2][:,:,np.newaxis])

    elif states == 4:
        # 4 states
        lambda_ = np.multiply(1 - weight_1_stack, 
                            lambda_latent[taste_label,0][:,:,np.newaxis]) + \
                np.multiply(weight_1_stack * (1 - weight_2_stack), 
                            lambda_latent[taste_label][:,1][:,:,np.newaxis]) + \
                np.multiply(weight_2_stack * (1 - weight_3_stack), 
                            lambda_latent[taste_label][:,2][:,:,np.newaxis]) + \
                np.multiply(weight_3_stack, 
                            lambda_latent[taste_label,3][:,:,np.newaxis])

    elif states == 5:
        # 5 states
        lambda_ = np.multiply(1 - weight_1_stack, 
                            lambda_latent[taste_label,0][:,:,np.newaxis]) + \
                np.multiply(weight_1_stack * (1 - weight_2_stack), 
                            lambda_latent[taste_label][:,1][:,:,np.newaxis]) + \
                np.multiply(weight_2_stack * (1 - weight_3_stack), 
                            lambda_latent[taste_label][:,2][:,:,np.newaxis]) +\
                np.multiply(weight_3_stack * (1 - weight_4_stack), 
                            lambda_latent[taste_label][:,3][:,:,np.newaxis])+ \
                np.multiply(weight_4_stack, 
                            lambda_latent[taste_label,4][:,:,np.newaxis])
        
    # Add observations
    observation = pm.Poisson("obs", lambda_, observed=dat_binned_long)

########################################
# ___        __                              
#|_ _|_ __  / _| ___ _ __ ___ _ __   ___ ___ 
# | || '_ \| |_ / _ \ '__/ _ \ '_ \ / __/ _ \
# | || | | |  _|  __/ | |  __/ | | | (_|  __/
#|___|_| |_|_|  \___|_|  \___|_| |_|\___\___|
########################################
# If the unnecessarily detailed model name exists
# It will be loaded without running the inference
# Otherwise model will be fit and saved
model_dump_path = os.path.join(model_save_dir,f'dump_{model_name}.pkl')
if os.path.exists(model_dump_path):
    print('Trace loaded from cache')
    with open(model_dump_path, 'rb') as buff:
        data = pickle.load(buff)
    model = data['model']
    #inference = data['inference']
    approx = data['approx']
    # Recreate samples
    trace = approx.sample(draws=samples)
else:
    with model:
        #step= pm.Metropolis()
        #step= pm.NUTS()
        #trace = pm.sample(100, tune=10,
        #                  step = step,
        #                  chains = 30, cores = 30)
        inference = pm.ADVI('full-rank')
        approx = pm.fit(n=fit, method=inference)
        trace = approx.sample(draws=samples)

    print('Dumping trace to cache')
    with open(model_dump_path, 'wb') as buff:
        pickle.dump({'model' : model,
                    'approx' : approx}, buff)
                    #'trace': trace,
                    #'inference': inference,

##############################
# ____  _       _       
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
##############################

dat.get_stft(recalculate=False)

##################################################
# ELBO Plot
##################################################
fig,ax = plt.subplots()                       
ax.plot(-approx.hist, label='new ADVI', alpha=.3)
plt.legend()
plt.ylabel('ELBO')
plt.xlabel('iteration');
plt.savefig(os.path.join(plot_dir,'ELBO_plot'))
plt.close(fig)

##################################################
# Lambda Plot
##################################################
lambda_stack = trace['lambda'].swapaxes(0,1)
tau_samples = trace['tau']
mean_tau = np.mean(tau_samples, axis=0)
mean_lambda = np.mean(lambda_stack,axis=1).swapaxes(1,2)
sd_lambda = np.std(lambda_stack,axis=1).swapaxes(1,2)
zscore_mean_lambda = np.array([stats.zscore(nrn,axis=None) \
        for nrn in mean_lambda.swapaxes(0,1)]).swapaxes(0,1)

fig,ax = plt.subplots(2,mean_lambda.shape[0]);
for ax_num, (this_dat,this_zscore_dat) in \
                    enumerate(zip(mean_lambda,zscore_mean_lambda)):
    ax[0,ax_num].imshow(this_dat, interpolation = 'nearest', 
                aspect = 'auto', cmap = 'viridis',
                vmin = 0, vmax = np.max(mean_lambda,axis=None))
    ax[1,ax_num].imshow(this_zscore_dat, interpolation = 'nearest', 
                aspect = 'auto', cmap = 'viridis')

plt.savefig(os.path.join(plot_dir,'lambda_plot'))
plt.close(fig)

##################################################
# Changepoint Plot
##################################################
mean_mean_tau = np.mean(tau_samples,axis=(0,1))

plot_spikes = dat_binned_long>0

channel = 0
stft_cut = stats.zscore(dat.amplitude_array[:,:],axis=-1)
stft_cut = stft_cut[:,channel,...,time_lims[0]:time_lims[1]]
stft_cut = np.reshape(stft_cut,(-1,*stft_cut.shape[2:]))
stft_ticks = dat.time_vec[time_lims[0]:time_lims[1]]*1000
stft_tick_inds = np.arange(0,len(stft_ticks),250)

mean_tau_stft = (mean_tau/np.max(mean_tau,axis=None))*stft_cut.shape[-1]

# Overlay raster with CDF of switchpoints
tick_interval = 5
vline_kwargs = {'color': 'red', 'linewidth' :3, 'alpha' : 0.7}
imshow_kwargs = {'interpolation':'nearest','aspect':'auto','origin':'lower'}

for this_taste in np.sort(np.unique(taste_label)):
    trial_inds = np.where(taste_label==this_taste)[0]

    fig, ax = plt.subplots(len(trial_inds),3,sharex='col', figsize = (15,50))
    for num,trial in enumerate(trial_inds):
        ax[num,0].imshow(plot_spikes[trial], **imshow_kwargs)
        ax[num,0].set_ylabel(taste_label[trial])
        ax[num,1].imshow(stft_cut[trial],**imshow_kwargs)
        ax[num,0].vlines(mean_tau[trial],-0.5,
                            mean_lambda.shape[1]-0.5, **vline_kwargs)
        ax[num,1].vlines(mean_tau_stft[trial],
                            -0.5,stft_cut.shape[1]-0.5,**vline_kwargs)
        ax[num,1].set_xticks(stft_tick_inds)
        ax[num,1].set_xticklabels(stft_ticks[stft_tick_inds],rotation='vertical')

        for state in range(tau_samples.shape[-1]):
            ax[num,-1].hist(tau_samples[:,trial,state], bins = 100, density = True)
            
    plt.savefig(os.path.join(\
            plot_dir,'taste{}_changepoints'.format(this_taste)),dpi=300)
    plt.close(fig)


##################################################
# Good Trial Changepoint Plot
##################################################
# Find trials where the mean tau for one changepoint is outside the 95% interval for other taus 

percentile_array = np.zeros((*mean_tau.shape,mean_tau.shape[-1]))
for trial_num, (this_mean_tau, this_tau_dist) in \
            enumerate(zip(mean_tau, np.moveaxis(tau_samples,0,-1))):
    for tau1_val, this_tau in enumerate(this_mean_tau):
        for tau2_val, this_dist in enumerate(this_tau_dist):
            percentile_array[trial_num, tau1_val, tau2_val] = \
                    percentileofscore(this_dist, this_tau)

# Visually, threshold of <1 percentile seems compelling
# Find all trials where all the upper triangular elements are <1
# and lower triangular elements are >99
lower_thresh = 1
upper_thresh = 100 - lower_thresh
good_trial_list = np.where([all(x[np.triu_indices(states-1,1)] < lower_thresh) \
                  and all(x[np.tril_indices(states-1,-1)] > upper_thresh) \
                  for x in percentile_array])[0]

# Plot only good trials
# Overlay raster with CDF of switchpoints
tick_interval = 5
max_trials = 15
num_plots = int(np.ceil(len(good_trial_list)/max_trials))
trial_inds_list = [good_trial_list[x*max_trials:(x+1)*max_trials] \
                        for x in np.arange(num_plots)]

for fig_num in np.arange(num_plots):
    trial_inds = trial_inds_list[fig_num]
    trial_count = len(trial_inds)
    
    fig, ax = plt.subplots(trial_count,3,sharex='col', figsize = (20,trial_count*3))
    for num,trial in enumerate(trial_inds):
        ax[num,0].imshow(plot_spikes[trial], **imshow_kwargs)
        ax[num,0].set_ylabel(taste_label[trial])
        ax[num,1].imshow(stft_cut[trial], **imshow_kwargs)
        ax[num,0].vlines(mean_tau[trial],-0.5,mean_lambda.shape[1]-0.5,
                            **vline_kwargs)
        ax[num,1].vlines(mean_tau_stft[trial],-0.5,stft_cut.shape[1]-0.5,
                            **vline_kwargs)

        ax[num,1].set_xticks(stft_tick_inds)
        ax[num,1].set_xticklabels(stft_ticks[stft_tick_inds],rotation='vertical')

        for state in range(tau_samples.shape[-1]):
            ax[num,-1].hist(tau_samples[:,trial,state], bins = 100, density = True)

    plt.savefig(os.path.join(plot_dir,'good_changepoints{}'.format(fig_num)),dpi=300)
    plt.close(fig)
