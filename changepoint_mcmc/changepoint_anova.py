"""
Fit ANOVA to "firing rates" of neurons
based off of changepoint predictions to test which and how many neurons
are strongly adhering to the changepoint predictions
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
#import pymc3 as pm

import numpy as np
from matplotlib import pyplot as plt
import pickle
import argparse
import pandas as pd
import pingouin as pg
from scipy.stats import percentileofscore

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
parser.add_argument('states', type = int, help = 'Number of States to fit')
args = parser.parse_args()
data_dir = args.dir_name 
#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201230_115322/'
#states = 4

plot_super_dir = os.path.join(data_dir,'changepoint_plots')
if not os.path.exists(plot_super_dir):
        os.makedirs(plot_super_dir)

dat = ephys_data(data_dir)

dat.get_unit_descriptors()
dat.get_spikes()
dat.default_stft_params['max_freq'] = 50
taste_dat = np.array(dat.spikes)
#dat.get_stft(recalculate=False, dat_type = ['amplitude'])

##########
# PARAMS 
##########
time_lims = [1500,4000]
bin_width = 10
states = int(args.states)
fit = 40000
samples = 20000

# Create dirs and names
model_save_dir = os.path.join(data_dir,'saved_models',f'vi_{states}_states')
if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
model_name = f'vi_{states}_states_{fit}fit_'\
        f'time{time_lims[0]}_{time_lims[1]}_bin{bin_width}'
plot_dir = os.path.join(plot_super_dir,model_name)
if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

model_dump_path = os.path.join(model_save_dir,f'dump_{model_name}.pkl')

##########
# Bin Data
##########
t_vec = np.arange(taste_dat.shape[-1])
binned_t_vec = np.min(t_vec[time_lims[0]:time_lims[1]].\
                    reshape((-1,bin_width)),axis=-1)
whole_dat_binned = \
        np.sum(taste_dat.reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = \
        np.sum(taste_dat[...,time_lims[0]:time_lims[1]].\
        reshape(*taste_dat.shape[:-1],-1,bin_width),axis=-1)
this_dat_binned = np.vectorize(np.int)(this_dat_binned)

# Unroll arrays along taste axis
dat_binned_long = np.reshape(this_dat_binned,(-1,*this_dat_binned.shape[-2:]))

########################################
# Load Data 
########################################
trace_dump_path = os.path.join(model_save_dir,f'traces_{model_name}.pkl')

if os.path.exists(trace_dump_path):
    print('Data loaded from cache')
    with open(trace_dump_path, 'rb') as buff:
        data = pickle.load(buff)
    #lambda_stack = data['lambda']
    tau_samples = data['tau']
else:
    raise Exception('Saved data not found')

##################################################
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           
##################################################
#lambda_stack = lambda_stack.swapaxes(0,1)
#mean_lambda = np.mean(lambda_stack,axis=1).swapaxes(1,2)
mean_tau = np.mean(tau_samples, axis=0)

# Get mean firing rate for each STATE using model
state_inds = np.hstack([np.zeros((mean_tau.shape[0],1)),
                        mean_tau,
                        np.ones((mean_tau.shape[0],1))*dat_binned_long.shape[-1]])
state_lims = np.array([state_inds[:,x:x+2] for x in range(states)])
state_lims = np.vectorize(np.int)(state_lims)
state_lims = np.swapaxes(state_lims,0,1)

mean_state_firing = \
        np.array([[np.mean(trial_dat[:,start:end],axis=-1) \
        for start, end in trial_lims] \
        for trial_dat, trial_lims in zip(dat_binned_long,state_lims)])
frame_inds = np.array(list(np.ndindex(mean_state_firing.shape)))

mean_firing_frame = pd.DataFrame({\
                    'trial' : frame_inds[:,0],
                    'state' : frame_inds[:,1],
                    'neuron' : frame_inds[:,2],
                    'firing' : mean_state_firing.flatten()})

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
fin_firing_frame = mean_firing_frame.query('state > 0')

#_,this_frame = list(fin_firing_frame.groupby('neuron'))[0]
anova_list = [this_frame.rm_anova(\
        dv = 'firing', within = 'state', subject = 'trial') \
        for num,this_frame in fin_firing_frame.groupby('neuron')]
pval_array = np.array([x['p-unc'][0] for x in anova_list])

# Sort neurons by p-values
sort_order = np.argsort(pval_array)
sort_order_index = np.array([np.where(sort_order == x)[0][0] \
                        for x in range(len(sort_order))])

#fin_firing_frame['sort_order'] = sort_order_index[fin_firing_frame['neuron']]
#g = sns.catplot(data=fin_firing_frame, x = 'sort_order', y = 'firing',
#                    hue = 'state', kind = 'box')
#plt.show()

########################################
## PLOTS 
########################################
plt.rcParams.update(plt.rcParamsDefault)

this_plot_dir = os.path.join(plot_dir,'analysis_plots')
if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

# Remove neurons which don't pass the threshold
# And order the remaining ones by p-values
taste_label = np.repeat([0,1,2,3],30)
alpha = 0.001
pval_cutoff = np.min(np.where(pval_array[sort_order] > alpha))
cutoff_post_sort = np.min([15,pval_cutoff]) 

##################################################
# Good Trial Changepoint Plot
##################################################
# Find trials where the mean tau for one changepoint is outside the 95% interval for other taus 

plot_spikes = dat_binned_long>0
plot_spikes = plot_spikes[:,sort_order]
plot_spikes = plot_spikes[:,:cutoff_post_sort]

#channel = 0
#stft_cut = stats.zscore(dat.amplitude_array[:,:],axis=-1)
#stft_cut = stft_cut[:,channel,...,time_lims[0]:time_lims[1]]
#stft_cut = np.reshape(stft_cut,(-1,*stft_cut.shape[2:]))
#stft_ticks = dat.time_vec[time_lims[0]:time_lims[1]]*1000
#stft_tick_inds = np.arange(0,len(stft_ticks),250)

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
vline_kwargs = {'color': 'red', 'linewidth' :3, 'alpha' : 0.7}
hline_kwargs = {'color': 'red', 'linewidth' :1, 'alpha' : 1,
                    'xmin': -0.5, 'xmax' : plot_spikes.shape[-1] -0.5}
imshow_kwargs = {'interpolation':'none','aspect':'auto',
            'origin':'lower', 'cmap':'viridis'}

tick_interval = 5
time_tick_count = 10
binned_tick_inds = np.arange(0,len(binned_t_vec),
                        len(binned_t_vec)//time_tick_count)
binned_tick_vals = np.arange(time_lims[0],time_lims[1],
                        np.abs(np.diff(time_lims))//time_tick_count)
max_trials = 15
num_plots = int(np.ceil(len(good_trial_list)/max_trials))
trial_inds_list = [good_trial_list[x*max_trials:(x+1)*max_trials] \
                        for x in np.arange(num_plots)]

for fig_num in np.arange(num_plots):
    trial_inds = trial_inds_list[fig_num]
    trial_count = len(trial_inds)
    
    fig, ax = plt.subplots(trial_count,2,sharex='col', figsize = (20,trial_count*3))
    for num,trial in enumerate(trial_inds):
        #ax[num,0].imshow(plot_spikes[trial], **imshow_kwargs)
        ax[num,0].scatter(*np.where(plot_spikes[trial])[::-1], marker = "|")
        ax[num,0].set_ylabel(taste_label[trial])
        #ax[num,1].imshow(stft_cut[trial], **imshow_kwargs)
        #ax[num,0].hlines(len(dat.region_units[0]) -0.5 ,**hline_kwargs)
        ax[num,0].vlines(mean_tau[trial],-0.5,plot_spikes.shape[1]-0.5,
                            **vline_kwargs)
        #ax[num,1].vlines(mean_tau_stft[trial],-0.5,stft_cut.shape[1]-0.5,
        #                    **vline_kwargs)

        #ax[num,1].set_xticks(stft_tick_inds)
        #ax[num,1].set_xticklabels(stft_ticks[stft_tick_inds],rotation='vertical')

        for state in range(tau_samples.shape[-1]):
            ax[num,-1].hist(tau_samples[:,trial,state], bins = 100, density = True)

    ax[-1,0].set_xticks(binned_tick_inds)
    ax[-1,0].set_xticklabels(binned_tick_vals, rotation = 'vertical')
    ax[-1,-1].set_xticks(binned_tick_inds)
    ax[-1,-1].set_xticklabels(binned_tick_vals, rotation = 'vertical')

    #fig.suptitle(region_unit_count)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(this_plot_dir,
            'good_changepoints_sorted_units{}'.format(fig_num)),dpi=300)
    plt.close(fig)

