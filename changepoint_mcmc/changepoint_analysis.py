"""
Input :: Only actual data
            Shuffled and simulated datasets are loaded automatically

1)
Fit ANOVA to "firing rates" of neurons
based off of changepoint predictions to test which and how many neurons
are strongly adhering to the changepoint predictions

2)
Compare magnitude of change in activity across transitions between
actual data and shuffle and simulated controls on a per-neuron
and population-vector basis
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
import shutil
#import pymc3 as pm
import re
from glob import glob
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import percentileofscore,mode,zscore

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def create_changepoint_plots(spike_array, tau_samples, trial_inds_list, 
                    suptitle, taste_label, region_units_list, 
                    binned_tick_inds, binned_tick_vals,
                    plot_type = 'raster'):
                    #stft_ticks, stft_tick_inds, 

    #mean_tau = np.mean(tau_samples,axis=0)
    int_tau = np.vectorize(np.int)(tau_samples)
    mode_tau = np.squeeze(mode(int_tau,axis=0)[0])
    state_inds = np.concatenate([np.zeros((mode_tau.shape[0],1)),
                    mode_tau, 
                    np.ones((mode_tau.shape[0],1))*spike_array.shape[-1]],
                    axis=-1)
    if len(region_units_list) > 1:
        hline_val = np.squeeze(np.cumsum([len(x) for x in region_units_list])[:-1])

    for fig_num in tqdm(np.arange(len(trial_inds_list))):
        trial_inds = trial_inds_list[fig_num]
        trial_count = len(trial_inds)
        
        fig, ax = plt.subplots(trial_count,2,sharex='col', 
                            figsize = (10,trial_count*2))
        if ax.ndim < 2:
            ax = ax[np.newaxis,:]
        for num,trial in enumerate(trial_inds):

            cmap = plt.get_cmap("tab10")
            for state in range(tau_samples.shape[-1]+1):
                ax[num,0].axvspan(state_inds[trial,state],
                        state_inds[trial,state+1],alpha=0.2,
                        color = cmap(state))
                ax[num,0].axvline(state_inds[trial,state],
                        -0.5,spike_array.shape[1]-0.5,
                        linewidth = 2, color = cmap(state), alpha = 0.6)
                                #**vline_kwargs)

            if plot_type == 'heatmap':
                ax[num,0].imshow(spike_array[trial], **imshow_kwargs)
            elif plot_type == 'raster':
                ax[num,0].scatter(*np.where(spike_array[trial])[::-1], 
                        color = 'k',marker = "|")
            ax[num,0].set_ylabel(taste_label[trial])
            if len(region_units_list) > 1:
                ax[num,0].hlines(hline_val-0.5, 
                        state_inds[0,0] , state_inds[0,-1], **hline_kwargs)

            for state in range(tau_samples.shape[-1]):
                ax[num,-1].hist(tau_samples[:,trial,state], 
                    bins = 100, density = True)

            #plt.show()

        ax[-1,0].set_xticks(binned_tick_inds)
        ax[-1,0].set_xticklabels(binned_tick_vals, rotation = 'vertical')
        ax[-1,-1].set_xticks(binned_tick_inds)
        ax[-1,-1].set_xticklabels(binned_tick_vals, rotation = 'vertical')

        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

def get_state_firing(spike_array,tau_array):
    """
    spike_array : trials x nrns x bins
    tau_array : trials x switchpoints
    """
    states = tau_array.shape[-1] + 1
    # Get mean firing rate for each STATE using model
    state_inds = np.hstack([np.zeros((tau_array.shape[0],1)),
                            tau_array,
                            np.ones((tau_array.shape[0],1))*spike_array.shape[-1]])
    state_lims = np.array([state_inds[:,x:x+2] for x in range(states)])
    state_lims = np.vectorize(np.int)(state_lims)
    state_lims = np.swapaxes(state_lims,0,1)

    state_firing = \
            np.array([[np.mean(trial_dat[:,start:end],axis=-1) \
            for start, end in trial_lims] \
            for trial_dat, trial_lims in zip(spike_array,state_lims)])

    state_firing = np.nan_to_num(state_firing)
    return state_firing

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

parser = argparse.ArgumentParser(description = 'Script to analyze fit models')
parser.add_argument('model_path',  help = 'Path to model pkl file')
args = parser.parse_args()
model_path = args.model_path 

#model_path = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'\
#        'saved_models/vi_4_states/vi_4states_40000fit_1500_4000time_50bin.pkl'
#model_path = '/media/bigdata/Abuzar_Data/AS18/AS18_4Tastes_200228_151511/'\
#        'saved_models/vi_4_states/vi_4states_40000fit_1500_4000time_50bin.pkl'
#model_path = '/media/bigdata/Abuzar_Data/AM28/AM28_4Tastes_201004_120804/'\
#        'saved_models/vi_4_states/vi_4states_40000fit_2000_4000time_50bin.pkl'
#model_path = '/media/bigdata/Abuzar_Data/AS18/AS18_4Tastes_200229_154608/'\
#        'saved_models/vi_4_states/'\
#        'actual_vi_4states_40000fit_2000_4000time_50bin_type_good.pkl'

if not os.path.exists(model_path):
    raise Exception('Model path does not exist')

##########
# PARAMS 
##########
# Extract model params from basename
model_name = os.path.basename(model_path).split('.')[0]
states = int(re.findall("\d+states",model_name)[0][:-6])
time_lims = [int(x) for x in \
        re.findall("\d+_\d+time",model_name)[0][:-4].split('_')]
bin_width = int(re.findall("\d+bin",model_name)[0][:-3])
fit_type = re.findall("type_.+",model_name)[0].split('_')[1]

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
plot_super_dir = os.path.join(data_dir,
        'changepoint_plots',f'{states}_states', f'type_{fit_type}')
plot_dir = os.path.join(plot_super_dir,model_name,'analysis_plots')

if not os.path.exists(plot_super_dir):
        os.makedirs(plot_super_dir)
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir)

if os.path.exists(model_path):
    print('Trace loaded from cache')
    with open(model_path, 'rb') as buff:
        data = pickle.load(buff)
    lambda_stack = data['lambda']
    tau_samples = data['tau']
    binned_dat = data['data']
    # Remove pickled data to conserve memory
    del data
    # Recreate samples

binned_t_vec = np.arange(time_lims[0],time_lims[1])[::bin_width]
taste_label = np.sort(list(range(len(dat.spikes)))*dat.spikes[0].shape[0])

##################################################
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           
##################################################
#lambda_stack = lambda_stack.swapaxes(0,1)
#mean_lambda = np.mean(lambda_stack,axis=1).swapaxes(1,2)

# To be able to move between mean and mode more easily
#stat_tau = np.mean(tau_samples, axis=0)
int_tau = np.vectorize(np.int)(tau_samples)
stat_tau = np.squeeze(mode(int_tau,axis=0)[0])

state_firing = get_state_firing(binned_dat, stat_tau)
## Zscore firing for later plotting
#state_firing = np.array([zscore(nrn) for nrn in state_firing.T]).T
#state_firing = np.nan_to_num(state_firing)

# Reshape state_firing to have separate axis for tastes
# Otherwise ANOVA will pull all tastes together
taste_state_firing = np.reshape(state_firing,
                        (len(dat.spikes),-1,*state_firing.shape[1:]))
frame_inds = np.array(list(np.ndindex(taste_state_firing.shape)))

mean_firing_frame = pd.DataFrame({\
                    'taste' : frame_inds[:,0],
                    'trial' : frame_inds[:,1],
                    'state' : frame_inds[:,2],
                    'neuron' : frame_inds[:,3],
                    'firing' : taste_state_firing.flatten()})

#import seaborn as sns
#
#g = sns.catplot(data=mean_firing_frame, x = 'state', y = 'firing',
#                    hue = 'state', kind = 'box', col='neuron',
#                    col_wrap = 8)
#plt.show()

########################################
## ANOVA
########################################
# See which neurons are significantly different between states
# Discounting first state (pre-stim firing) since that change is expected
if time_lims[0] < 2000:
    fin_firing_frame = mean_firing_frame.query('state > 0')
else:
    fin_firing_frame = mean_firing_frame

#_,this_frame = list(fin_firing_frame.groupby('neuron'))[0]
anova_list = [this_frame.rm_anova(\
        dv = 'firing', within = ['taste','state'], subject = 'trial') \
        for num,this_frame in fin_firing_frame.groupby('neuron')]
pval_array = np.array([x['p-unc'][1] for x in anova_list])

# Sort neurons by p-values
sort_order = np.argsort(pval_array)

sort_order_index = np.array([np.where(sort_order == x)[0][0] \
                        for x in range(len(sort_order))])
fin_firing_frame['sort_order'] = sort_order_index[fin_firing_frame['neuron']]

g = sns.catplot(data=fin_firing_frame, col = 'sort_order', y = 'firing',
                    x = 'taste', hue='state', kind = 'box', col_wrap=8, sharey=False)
for num, this_ax in enumerate(g.axes):
    this_ax.set_title(f"{pval_array[sort_order][num]:.2E}")
plt.tight_layout(rect=[0,0,0.95,0.95])
plt.savefig(os.path.join(plot_dir,'sorted_neuron_order'))
plt.close()
#plt.show()

########################################
## PLOTS 
########################################
plt.rcParams.update(plt.rcParamsDefault)

# Remove neurons which don't pass the threshold
# And order the remaining ones by p-values
taste_label = np.repeat(np.arange(len(dat.spikes)),30)
alpha = 0.001
pval_cutoff = np.min(np.where(pval_array[sort_order] > alpha))
max_nrn_num = 10
cutoff_post_sort = np.min([max_nrn_num,pval_cutoff]) 
#cut_sort_order = sort_order[:cutoff_post_sort]

# Take all neurons which pass p_value thresholf
# Resort them to have highest spiking neurons
mean_nrn_firing = np.mean(state_firing,axis=(0,1))
thresh_sort_order = sort_order[:pval_cutoff]
thresh_sorted_mean_firing = mean_nrn_firing[thresh_sort_order]
rate_sort_order = np.argsort(thresh_sorted_mean_firing)[::-1]
cut_sort_order = thresh_sort_order[rate_sort_order][:cutoff_post_sort]
# Return to neuron number order
cut_sort_order = np.sort(cut_sort_order)

# Get mean firing for units being plotted
cut_taste_firing = taste_state_firing[...,cut_sort_order]
## tastes x staets x nrns
mean_cut_taste_firing = np.mean(cut_taste_firing,axis=1)/(bin_width/1000)
max_firing = np.max(mean_cut_taste_firing,axis=None)

cut_firing_frame = fin_firing_frame.loc[fin_firing_frame.neuron.isin(cut_sort_order)]
cut_firing_frame.firing = cut_firing_frame['firing']/(bin_width/1000)

#sort_ind_frame = pd.DataFrame({
#                'neuron' : cut_sort_order,
#                'sort_order' : np.arange(len(cut_sort_order))})
#cut_firing_frame.drop('sort_order',axis=1,inplace=True)
#cut_firing_frame = cut_firing_frame.merge(sort_ind_frame,on='neuron')

cmap = plt.get_cmap("tab10")
if time_lims[0] < 2000:
    this_cmap = cmap(np.arange(states))[1:]
else:
    this_cmap = cmap(np.arange(states))
this_cmap[:,-1] = 0.4

g = sns.catplot(data=cut_firing_frame, y = 'neuron', x = 'firing',
                    row = 'taste', col = 'state', kind = 'bar', 
                    orient = 'h', ci=None, facecolor='white',
                    edgecolor=".2", sharey=False) 
for num,this_row in enumerate(g.axes.T):
    for this_ax in this_row:
        this_ax.set_facecolor(this_cmap[num])
        #max_x_lim = this_ax.get_xlim()[1]
        this_ax.set_xlim([0,max_firing*1.05])
        this_ax.invert_yaxis()
plt.tight_layout(rect=[0,0,0.95,0.95])
plt.subplots_adjust(hspace=0.2)
plt.savefig(os.path.join(plot_dir,'sorted_neuron_state_firing'))
plt.close()
#plt.show()


# Recluster sorted units by which region they belong to
sorted_region_units = [[] for region in dat.region_names]
for unit in cut_sort_order:
    for region_num, region_list in enumerate(dat.region_units):
        if unit in region_list:
            sorted_region_units[region_num].append(unit)

sorted_region_units_lens = [len(x) for x in sorted_region_units]
title_dict = dict(zip(dat.region_names, sorted_region_units_lens))

fin_sort_order = np.concatenate(sorted_region_units)

##################################################
# Good Trial Changepoint Plot
##################################################
# Find trials where the mean tau for one changepoint is outside the 95% interval for other taus 

this_plot_dir = os.path.join(plot_dir,'good_changepoints_sorted')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

plot_spikes = binned_dat>0
plot_spikes = plot_spikes[:,fin_sort_order]

#channel = 0
#stft_cut = stats.zscore(dat.amplitude_array[:,:],axis=-1)
#stft_cut = stft_cut[:,channel,...,time_lims[0]:time_lims[1]]
#stft_cut = np.reshape(stft_cut,(-1,*stft_cut.shape[2:]))
#stft_ticks = dat.time_vec[time_lims[0]:time_lims[1]]*1000
#stft_tick_inds = np.arange(0,len(stft_ticks),250)

percentile_array = np.zeros((*stat_tau.shape,stat_tau.shape[-1]))
for trial_num, (this_stat_tau, this_tau_dist) in \
            enumerate(zip(stat_tau, np.moveaxis(tau_samples,0,-1))):
    for tau1_val, this_tau in enumerate(this_stat_tau):
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
t_stim = 2000
vline_kwargs = {'color': 'red', 'linewidth' :3, 'alpha' : 0.7}
hline_kwargs = {'color': 'black', 'linewidth' :1, 'alpha' : 1}#,
                    #'xmin': -0.5, 'xmax' : plot_spikes.shape[-1] -0.5}
imshow_kwargs = {'interpolation':'none','aspect':'auto',
            'origin':'lower', 'cmap':'viridis'}

tick_interval = 5
time_tick_count = 10
binned_tick_inds = np.arange(0,len(binned_t_vec),
                        len(binned_t_vec)//time_tick_count)
binned_tick_vals = np.arange(time_lims[0],time_lims[1],
                        np.abs(np.diff(time_lims))//time_tick_count)
binned_tick_vals -= t_stim
max_trials = 15
num_plots = int(np.ceil(len(good_trial_list)/max_trials))
trial_inds_list = [good_trial_list[x*max_trials:(x+1)*max_trials] \
                        for x in np.arange(num_plots)]

create_changepoint_plots(plot_spikes, tau_samples , 
            trial_inds_list,
            title_dict, taste_label, sorted_region_units,
            binned_tick_inds, binned_tick_vals)

for fig_num in tqdm(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(os.path.join(\
            this_plot_dir,'good_trials_sorted_nrns{}'.format(fig_num)))#,dpi=300)
    plt.close(fig_num)

########################################
cut_spikes = np.array(dat.spikes)[...,time_lims[0]:time_lims[1]]
spike_array_long = np.reshape(cut_spikes,(-1,*cut_spikes.shape[-2:]))
spike_array_long = spike_array_long[:,fin_sort_order]

raw_tick_inds = np.arange(0,cut_spikes.shape[-1],
                        cut_spikes.shape[-1]//time_tick_count)
scaled_tau_samples = (tau_samples/binned_dat.shape[-1])*cut_spikes.shape[-1]

this_plot_dir = os.path.join(plot_dir, 'full_spike_good_changepoints_sorted_nrns')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

create_changepoint_plots(spike_array_long, scaled_tau_samples, 
        trial_inds_list,
            title_dict, taste_label, sorted_region_units,
            raw_tick_inds, binned_tick_vals)

for fig_num in tqdm(plt.get_fignums()):
    plt.figure(fig_num)
    plt.savefig(os.path.join(\
            this_plot_dir,
            'full_spike_good_changepoints_sorted_nrns{}'.format(fig_num)))
    plt.close(fig_num)

##################################################
## Comparison of actual data with shuffles
##################################################
############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

# Check that simulate and shuffle fits exist
# Checking done here so initial plots can be made despite 
# not have control fits
search_pattern = "_".join(os.path.basename(model_path).split('_')[1:])
files_of_interest = glob(os.path.dirname(model_path) + '/*' +\
                        search_pattern)

if not len(files_of_interest) > 1:
    quit()
control_patterns = ['shuffle','simulate']
control_check = \
        [[len(re.findall(pattern,x))>0 for x in files_of_interest] \
        for pattern in control_patterns]
control_paths = [files_of_interest[np.where(x)[0][0]] for x in control_check]
control_check_bool = [any(x) for x in control_check]
if not all(control_check_bool):
    raise Exception('Simulate/shuffle splits absent \n'\
            f'{dict(zip(control_patterns,control_check))}')

all_file_paths = [model_path,*control_paths]
all_names = ['actual',*control_patterns]

if os.path.exists(model_path):
    print('Trace loaded from cache')
    data = [pickle.load(open(this_path,'rb')) for this_path in all_file_paths]
    lambda_list = [x['lambda'] for x in data]
    tau_list = [x['tau'] for x in data]
    binned_dat_list = [x['data'] for x in data]
    lambda_stack = lambda_list[0]
    tau_samples = tau_list[0]
    binned_dat = binned_dat_list[0]
    # Remove pickled data to conserve memory
    del data

##################################################
## ANALYSIS
##################################################
# Difference in MAGNITUDE of firing changes at state boundaries
# Calculate mean state firing for all datasets
int_tau_list = [np.vectorize(np.int)(x) for x in tau_list]
stat_tau_list = [np.squeeze(mode(x,axis=0)[0]) for x in int_tau_list]
state_firing_array = np.array([get_state_firing(this_dat, this_tau) \
        for this_dat,this_tau in zip(binned_dat_list, stat_tau_list)])

# Zscore activity of each neuron across all present data
zscore_state_firing = np.array([zscore(nrn,axis=None)
        for nrn in np.moveaxis(state_firing_array,-1,0)])
zscore_state_firing = np.moveaxis(zscore_state_firing,0,-1)

# Calculate mean of absolute difference between states
# On single-neuron, and population basis

## Single Neuron
#delta_state_firing = [np.diff(x,axis=1) for x in state_firing_list]
delta_state_firing = np.diff(zscore_state_firing,axis=2) 
# Split by taste
#delta_state_firing = [np.array(np.split(x, len(dat.spikes),axis=0)) \
#        for x in delta_state_firing]
# taste x data_type x trial x changepoint x neuron
delta_state_firing = np.array(np.split(delta_state_firing, len(dat.spikes),axis=1))
delta_state_firing = np.moveaxis(delta_state_firing,0,1)
#abs_delta = [np.abs(x) for x in delta_state_firing]
abs_delta = np.abs(delta_state_firing)

#mean_single_difference = np.array([np.mean(x,axis=1) for x in abs_delta])
mean_single_difference = np.mean(abs_delta,axis=2)

min_val = np.min(mean_single_difference,axis=None)
max_val = np.max(mean_single_difference,axis=None)

fig,ax = plt.subplots(*mean_single_difference.shape[:2])
inds = np.ndindex(ax.shape)
for this_ind in inds:
    im = ax[this_ind].imshow(mean_single_difference[this_ind],
                            vmin = min_val, vmax = max_val, aspect='auto')
for num in range(ax.shape[0]):
    ax[num,0].set_ylabel(all_names[num])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig(os.path.join(plot_dir,'single_nrn_delta_magnitude'))
plt.close(fig)
#plt.show()

single_greater_list = [(mean_single_difference[0] > x)*1 \
        for x in mean_single_difference]
single_greater_list = np.array(single_greater_list)[1:]
mean_greater = np.round(np.mean(single_greater_list,axis=(1,2,3)),2)

fig,ax = plt.subplots(mean_single_difference.shape[0]-1,
                            mean_single_difference.shape[1])
inds = np.ndindex(ax.shape)
for this_ind in inds:
    ax[this_ind].imshow(single_greater_list[this_ind],
                            aspect='auto',vmin=0,vmax=1)
for num in range(ax.shape[0]):
    ax[num,0].set_ylabel(all_names[num])
fig.suptitle(f'Average greater : {dict(zip(control_patterns,mean_greater))}')
fig.savefig(os.path.join(plot_dir,'single_nrn_delta_magnitude_bool'))
plt.close(fig)
#plt.show()

## Population
## ** Need to account for different firing rates
population_delta_mag = [np.linalg.norm(x,axis=-1) for x in delta_state_firing]
mean_population_delta_mag = [np.mean(x,axis=1) for x in population_delta_mag] 

min_val = np.min(mean_population_delta_mag,axis=None)
max_val = np.max(mean_population_delta_mag,axis=None)

# Plot population difference magnitudes
fig,ax = plt.subplots(1,len(mean_population_delta_mag))
for this_ax, this_dat,this_name \
        in zip(ax.flatten(),mean_population_delta_mag, all_names):
    im = this_ax.imshow(this_dat,vmin = min_val, vmax = max_val)
    this_ax.set_title(this_name)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.suptitle('Mean population vector difference magnitude')
fig.savefig(os.path.join(plot_dir,'pop_vec_delta_magnitude'))
plt.close(fig)
#plt.show()

# Plot actual > control
population_greater_list = np.array([(mean_population_delta_mag[0] > x)*1 \
        for x in mean_population_delta_mag])[1:]
mean_population_greater = np.mean(population_greater_list,axis=(1,2))

fig,ax = plt.subplots(1,len(mean_population_delta_mag)-1)
for this_ax, this_dat,this_name \
        in zip(ax.flatten(),population_greater_list, all_names[1:]):
    this_ax.imshow(this_dat,vmin=0,vmax=1)
    this_ax.set_title(this_name)
fig.suptitle('Mean Data population vector difference magnitude > Control : '\
        f'\nAverage greater : {dict(zip(control_patterns,mean_population_greater))}')
fig.savefig(os.path.join(plot_dir,'pop_vec_delta_magnitude_bool'))
plt.close(fig)
#plt.show()

