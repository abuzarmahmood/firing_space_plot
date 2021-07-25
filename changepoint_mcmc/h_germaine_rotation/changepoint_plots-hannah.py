"""
Genearte plots from fit changepoint models
ISSUES:
    - Add control for cases where exp info file doesn't exist
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
import scipy.stats as stats
import pymc3 as pm
#import theano.tensor as tt
import json
import re
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from scipy.stats import percentileofscore
import pickle
import argparse

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc')
from ephys_data import ephys_data
import visualize
import poisson_all_tastes_changepoint_model as changepoint 

def create_changepoint_plots(spike_array, tau_samples, trial_inds_list, 
                    suptitle, taste_label, region_units_list, 
                    binned_tick_inds, binned_tick_vals,
                    plot_type = 'raster'):
                    #stft_ticks, stft_tick_inds, 

    mean_tau = np.mean(tau_samples,axis=0)
    #mean_tau_stft = (mean_tau/spike_array.shape[-1])*stft_cut.shape[-1]

    for fig_num in tqdm(np.arange(len(trial_inds_list))):
        trial_inds = trial_inds_list[fig_num]
        trial_count = len(trial_inds)
        
        fig, ax = plt.subplots(trial_count,2,#sharex='col', 
                            figsize = (20,trial_count*3))
        for num,trial in enumerate(trial_inds):
            if plot_type == 'heatmap':
                ax[num,0].imshow(spike_array[trial], **imshow_kwargs)
            elif plot_type == 'raster':
                ax[num,0].scatter(*np.where(spike_array[trial])[::-1], marker = "|")
            ax[num,0].set_ylabel(taste_label[trial])
            #ax[num,1].imshow(stft_array[trial], **imshow_kwargs)
            ax[num,0].hlines(len(region_units_list[0]) -0.5 ,**hline_kwargs)
            ax[num,0].vlines(mean_tau[trial],-0.5,spike_array.shape[1]-0.5,
                                **vline_kwargs)
            #ax[num,1].vlines(mean_tau_stft[trial],-0.5,stft_cut.shape[1]-0.5,
            #                    **vline_kwargs)

            #ax[num,1].set_xticks(stft_tick_inds)
            #ax[num,1].set_xticklabels(stft_ticks[stft_tick_inds],rotation='vertical')

            for state in range(tau_samples.shape[-1]):
                ax[num,-1].hist(tau_samples[:,trial,state], 
                    bins = 100, density = True)
        ax[-1,0].set_xticks(binned_tick_inds)
        ax[-1,0].set_xticklabels(binned_tick_vals, rotation = 'vertical')
        ax[-1,-1].set_xticks(binned_tick_inds)
        ax[-1,-1].set_xticklabels(binned_tick_vals, rotation = 'vertical')

        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('model_path',  help = 'Path to model pkl file')
parser.add_argument('use_saved_data_array',  
        help = 'Boolean indicating whether unit numbers should '\
                'be taken from origin data or from model', default = False)
args = parser.parse_args()
model_path = args.model_path 
saved_data_bool = bool(args.use_saved_data_array)

#model_path = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'\
#        'saved_models/vi_4_states/dump_vi_4states_40000fit_1500_4000time_50bin.pkl'
#saved_data_bool = True

##########
# PARAMS 
##########
# Extract model params from basename
model_name = os.path.basename(model_path).split('.')[0]
states = int(re.findall("\d+states",model_name)[0][:-6])
time_lims = [int(x) for x in \
        re.findall("\d+_\d+time",model_name)[0][:-4].split('_')]
bin_width = int(re.findall("\d+bin",model_name)[0][:-3])

# Exctract data_dir from model_path
data_dir = "/".join(model_path.split('/')[:-3])

dat = ephys_data(data_dir)
#dat.firing_rate_params = dat.default_firing_params
dat.get_unit_descriptors()
dat.get_spikes()
#dat.get_firing_rates()
dat.default_stft_params['max_freq'] = 50
#dat.get_stft(recalculate=False, dat_type = ['amplitude'])
dat.get_region_units()

########################################
# Create dirs and names
########################################
plot_super_dir = os.path.join(data_dir,'changepoint_plots',f'{states}_states')
plot_dir = os.path.join(plot_super_dir,model_name)

if not os.path.exists(plot_super_dir):
        os.makedirs(plot_super_dir)
if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

print(model_path)

if os.path.exists(model_path):
    print('Trace loaded from cache')
    with open(model_path, 'rb') as buff:
        data = pickle.load(buff)
    model = data['model']
    approx = data['approx']
    lambda_stack = data['lambda']
    tau_samples = data['tau']
    binned_dat = data['data']
    unbinned_dat = data['fulldata']
    # Remove pickled data to conserve memory
    del data
    # Recreate samples

binned_t_vec = np.arange(time_lims[0],time_lims[1])[::bin_width]
taste_label = np.sort(list(range(len(dat.spikes)))*dat.spikes[0].shape[0])

##############################
# ____  _       _       
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
##############################

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
# Average Tau Plot 
##################################################
long_tau_samples = tau_samples.reshape((-1, tau_samples.shape[-1]))
fig,ax = plt.subplots()
for switch in range(long_tau_samples.shape[-1]):
     plt.hist(long_tau_samples[...,switch],bins = 100, density = True,alpha = 0.8)
plt.savefig(os.path.join(plot_dir,'cumulated_changepoint_hist'))#,dpi=300)
plt.close(fig)

##################################################
# Lambda Plot
##################################################
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
#def create_changepoint_plots(spike_array, stft_array, tau_samples, trial_inds_list, 
# Sort units be region
# Mark change in region using hline
if saved_data_bool:
    unit_order = np.arange(binned_dat.shape[1])
else:
    unit_order = np.concatenate(dat.region_units)

plot_spikes = binned_dat>0
plot_spikes = plot_spikes[:,unit_order]

#channel = 0
#stft_cut = stats.zscore(dat.amplitude_array[:,:],axis=-1)
#stft_cut = stft_cut[:,channel,...,time_lims[0]:time_lims[1]]
#stft_cut = np.reshape(stft_cut,(-1,*stft_cut.shape[2:]))
#stft_ticks = dat.time_vec[time_lims[0]:time_lims[1]]*1000
#stft_ticks = np.vectorize(np.int)(stft_ticks)
#stft_tick_inds = np.arange(0,len(stft_ticks),250)
#
#mean_tau_stft = (mean_tau/binned_dat.shape[-1])*stft_cut.shape[-1]

# Overlay raster with CDF of switchpoints
this_plot_dir = os.path.join(plot_dir,'all_changepoints')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

tick_interval = 5
time_tick_count = 10
binned_tick_inds = np.arange(0,len(binned_t_vec),
                        len(binned_t_vec)//time_tick_count)
binned_tick_vals = np.arange(time_lims[0],time_lims[1],
                        np.abs(np.diff(time_lims))//time_tick_count)
vline_kwargs = {'color': 'red', 'linewidth' :3, 'alpha' : 0.7}
hline_kwargs = {'color': 'red', 'linewidth' :1, 'alpha' : 1,
                    'xmin': -0.5, 'xmax' : plot_spikes.shape[-1] -0.5}
imshow_kwargs = {'interpolation':'nearest','aspect':'auto','origin':'lower'}

region_unit_count = dict(zip(dat.region_names,[len(x) for x in dat.region_units]))

full_trial_inds_list = [np.where(taste_label==this_taste)[0] \
        for this_taste in np.sort(np.unique(taste_label))]

#create_changepoint_plots(plot_spikes, stft_cut, tau_samples, full_trial_inds_list,
create_changepoint_plots(plot_spikes, tau_samples, full_trial_inds_list,
            region_unit_count, taste_label, [unit_order],
            binned_tick_inds, binned_tick_vals)
            #stft_ticks, stft_tick_inds,

for fig_num in tqdm(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(os.path.join(\
            this_plot_dir,'taste{}_changepoints'.format(fig_num)))#,dpi=300)
    plt.close(fig_num)
                

##################################################
# Good Trial Changepoint Plot
##################################################
# Find trials where the mean tau for one changepoint is outside the 95% interval for other taus 
this_plot_dir = os.path.join(plot_dir, 'good_changepoints')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

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
good_trial_inds_list = [good_trial_list[x*max_trials:(x+1)*max_trials] \
                        for x in np.arange(num_plots)]

#create_changepoint_plots(plot_spikes, stft_cut, tau_samples, good_trial_inds_list,
create_changepoint_plots(plot_spikes, tau_samples, good_trial_inds_list,
            region_unit_count, taste_label, [unit_order],
            binned_tick_inds, binned_tick_vals)
            #stft_ticks, stft_tick_inds,

for fig_num in tqdm(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(os.path.join(\
            this_plot_dir,'good_changepoints{}'.format(fig_num)))#,dpi=300)
    plt.close(fig_num)
                
##################################################
# Good Trial Changepoint Plot - Color
##################################################
#imshow_kwargs = {'interpolation':'nearest','aspect':'auto','origin':'lower'}
#
#create_changepoint_plots(binned_dat, stft_cut, tau_samples, good_trial_inds_list,
#            region_unit_count, taste_label, dat.region_units,
#            stft_ticks, stft_tick_inds,
#            binned_tick_inds, binned_tick_vals, plot_type = 'heatmap')
#
#for fig_num in tqdm(plt.get_fignums()):
#    plt.figure(fig_num)
#    plt.savefig(os.path.join(\
#            this_plot_dir,'good_changepoints_color{}'.format(fig_num)),dpi=300)
#    plt.close(fig_num)

##################################################
# Changepoint plots on RAW spikes 
##################################################
# Create same plots as above on unbinned data
cut_spikes = np.array(unbinned_dat)[...,time_lims[0]:time_lims[1]]
spike_array_long = np.reshape(cut_spikes,(-1,*cut_spikes.shape[-2:]))

raw_tick_inds = np.arange(0,cut_spikes.shape[-1],
                        cut_spikes.shape[-1]//time_tick_count)
scaled_tau_samples = (tau_samples/binned_dat.shape[-1])*cut_spikes.shape[-1]

hline_kwargs = {'color': 'red', 'linewidth' :1, 'alpha' : 1,
                    'xmin': -0.5, 'xmax' : spike_array_long.shape[-1] -0.5}

this_plot_dir = os.path.join(plot_dir, 'full_spike_changepoints')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

#create_changepoint_plots(spike_array_long, stft_cut, scaled_tau_samples, 
create_changepoint_plots(spike_array_long, scaled_tau_samples, 
            full_trial_inds_list,
            region_unit_count, taste_label, [unit_order],
            raw_tick_inds, binned_tick_vals)
            #stft_ticks, stft_tick_inds,

for fig_num in tqdm(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(os.path.join(\
            this_plot_dir,'full_spike_changepoints{}'.format(fig_num)))#,dpi=300)
    plt.close(fig_num)

##################################################
# Good Changepoint plots on RAW spikes
##################################################

this_plot_dir = os.path.join(plot_dir, 'full_spike_good_changepoints')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

#create_changepoint_plots(spike_array_long, stft_cut, scaled_tau_samples, 
create_changepoint_plots(spike_array_long, scaled_tau_samples, 
            good_trial_inds_list,
            region_unit_count, taste_label, [unit_order],
            raw_tick_inds, binned_tick_vals)
            #stft_ticks, stft_tick_inds,

for fig_num in tqdm(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(os.path.join(\
            this_plot_dir,'full_spike_good_changepoints{}'.format(fig_num)))#,dpi=300)
    plt.close(fig_num)

## ** NO POINT PLOTTING FULL SPIKES AS HEATMAP
## ** SPIKES DON'T SHOW UP IN IMAGE
