"""
Compile correlation percentiles across sessions 
(and subaggregate on a per-animal basis)

Calculate correlations for all-to-all transitions

Refine results using
1) Trials where the model fits are more confident
2) Recordings with more discriminative and responsive neurons
3) Recordings with "stable" neurons

** Note about dependencies
** Using theano and not theano-pymc
pip uninstall theano-pymc  # run a few times until it says not installed
pip install "pymc3<3.10" "theano==1.0.5"
"""

import numpy as np
import re
import json
from glob import glob
import os
import pandas as pd
import pickle 
import sys
from scipy import stats
from scipy.stats import percentileofscore as p_of_s
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange 
import tables
import pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler as ss

#import theano
#theano.config.compute_test_value = "ignore"

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs/all_tastes')
from check_data import check_data 
import itertools as it

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def corr_percentile_single(a,b, shuffles = 1000):
    #shuffles = 1000
    #this_comp = comparison_list[1]
    #a,b = tau_array[0, this_comp[0]], tau_array[1,this_comp[1]]
    corr_val = stats.spearmanr(a,b)[0]
    shuffle_vals = [stats.spearmanr(a, 
                    np.random.permutation(b))[0] \
            for i in range(shuffles)]
    percentile_val = p_of_s(shuffle_vals, corr_val)
    return percentile_val, corr_val, shuffle_vals

def return_corr_percentile(tau_array, shuffles = 5000):
    """
    tau_array : regions x transitions x trials
    """
    #tau_array = tau_list[0]
    trans_list = np.arange(tau_array.shape[1])
    # **Note: The transitions in BLA and GC are not the same,
    #           therefore we must look at all permutations, not simply
    #           all combinations.
    comparison_list = list(it.product(trans_list, trans_list))
    #comparison_list = list(it.combinations_with_replacement(trans_list, 2))
    percentile_array = np.zeros((tau_array.shape[1], tau_array.shape[1]))
    corr_array = np.zeros((tau_array.shape[1], tau_array.shape[1]))
    shuffle_array = np.zeros((tau_array.shape[1], tau_array.shape[1], shuffles))
    for this_comp in tqdm(comparison_list):
        percentile_val, corr_val, shuffle_vals = \
                corr_percentile_single(tau_array[0, this_comp[0]],
                                        tau_array[1, this_comp[1]],
                                        shuffles = shuffles)
        percentile_array[this_comp] = percentile_val
        corr_array[this_comp] = corr_val
        shuffle_array[this_comp] = shuffle_vals
    return percentile_array, corr_array, shuffle_array

class params_from_path:
    def __init__(self, path):
        # Extract model params from basename
        self.path = path
        self.model_name = os.path.basename(self.path).split('.')[0]
        self.states = int(re.findall("\d+states",self.model_name)[0][:-6])
        self.time_lims = [int(x) for x in \
                re.findall("\d+_\d+time",self.model_name)[0][:-4].split('_')]
        self.bin_width = int(re.findall("\d+bin",self.model_name)[0][:-3])
        #self.fit_type = re.findall("type_.+",self.model_name)[0].split('_')[1]
        # Exctract data_dir from model_path
        self.data_dir = "/".join(self.path.split('/')[:-3])
        self.session_name = self.data_dir.split('/')[-1]
        self.animal_name = self.session_name.split('_')[0]
    def to_dict(self):
        return dict(zip(['states','time_lims','bin_width','session_name'],
            [self.states,self.time_lims,self.bin_width,self.session_name]))

def load_mode_tau(model_path):
    if os.path.exists(model_path):
        print('Trace loaded from cache')
        with open(model_path, 'rb') as buff:
            data = pickle.load(buff)
        tau_samples = data['tau']
        # Convert to int first, then take mode
        int_tau = np.vectorize(int)(tau_samples)
        int_mode_tau = stats.mode(int_tau,axis=0)[0][0]
        # Remove pickled data to conserve memory
        del data
    #return tau_samples#, int_mode_tau
    return int_mode_tau

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################
plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/plots/multi_region'

########################################

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/inter_region'
wanted_names = ['rho_percentiles','mode_tau','rho_shuffles',
        'tau_corrs','tau_list'] 

# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/multi_region_frame.pkl'
inter_frame = pd.read_pickle(data_dir_pkl)
inter_frame['animal_name'] = [x.split('_')[0] for x in inter_frame['name']]

black_list = ['AM26_4Tastes_200828_112822', 'AM18','AM37', 'AM39']

fin_dir_list = []
for num, data_dir in tqdm(enumerate(inter_frame.path)):
    if data_dir[-1] == '/':
        temp_name = data_dir[:-1]
    else:
        temp_name = data_dir
    basename = os.path.basename(temp_name)
    if (basename in black_list) or any([x in basename for x in black_list]):
        continue
    fin_dir_list.append(data_dir)

# Pull out fit params for one file
gc_pkls = []
bla_pkls = []

for data_dir in fin_dir_list:
    #data_dir = fin_dir_list[0] 
    this_info = check_data(data_dir)
    this_info.run_all()
    inter_region_paths = [path for  num,path in enumerate(this_info.pkl_file_paths) \
                    if num in this_info.region_fit_inds]
    state4_models = [path for path in inter_region_paths if '4state' in path]

    # Check params for both fits add up
    check_params_bool = params_from_path(state4_models[0]).to_dict() ==\
                        params_from_path(state4_models[1]).to_dict()
    region_check = all([any([region_name in params_from_path(x).model_name \
                        for region_name in ['gc','bla']])\
                        for x in state4_models])
    if not (check_params_bool and region_check):
        raise Exception('Fit params dont match up')
    gc_pkls.append([x for x in state4_models if 'gc' in os.path.basename(x)][0])
    bla_pkls.append([x for x in state4_models if 'bla' in os.path.basename(x)][0])
    #params = params_from_path(inter_region_paths[0]) 

session_pkls_sorted = list(zip(gc_pkls, bla_pkls))
# Indexing = (gc,bla)

tau_list = [[load_mode_tau(x) for x in y] for y in tqdm(session_pkls_sorted)]
tau_list = np.stack(tau_list).swapaxes(-2,-1)

########################################
## All to all transition correlation
########################################

gc_list = np.stack([tau_list[:,0],tau_list[:,0]]).swapaxes(0,1)
bla_list = np.stack([tau_list[:,1],tau_list[:,1]]).swapaxes(0,1)

# Plot corrs
corr_dict = dict(gc = gc_list, bla = bla_list)
#corr_dat = gc_list
for name, corr_dat in corr_dict.items():
    iters = list(it.combinations(range(3),2))
    fig,ax = plt.subplots(len(iters), len(corr_dat),
            sharex=True, sharey=True,
            figsize = (15,5))
    for num, this_dat in enumerate(corr_dat):
        for iter_num, this_iter in enumerate(iters):
            ax[iter_num,num].scatter(this_dat[0][this_iter[0]],
                                    this_dat[1][this_iter[1]],
                                    alpha = 0.5, s = 5)
    for num, val in enumerate(iters):
        ax[num,0].set_ylabel(str(val))
    plt.suptitle(name)
    fig.savefig(os.path.join(plot_dir, name + '_transition_plots')) 
    #plt.show()

############################################################
# Compare actual data with simulation for purely sequential with no 
# influence

#region_percs = []
#for region_list in [gc_list, bla_list]:
#    session_percs = []
#    for this_dat in tqdm(region_list):
#        #gc_ind = 0
#        #this_dat = region_list[gc_ind]
#        iter_percs = []
#        iters = list(it.combinations(range(3),2))
#        for this_iter in iters:
#            #this_iter = iters[0] 
#            trans_comp = np.stack([this_dat[0][this_iter[0]], 
#                                this_dat[1][this_iter[1]]])
#            x = trans_comp[0]
#            y = trans_comp[1]
#            actual_corr = stats.spearmanr(x,y)[0]
#            max_y = int(y.max() + 1)
#            sim_num = 10000
#            sim_y = np.stack([np.random.randint(this_x, max_y, sim_num) \
#                        for this_x in x]).T
#            sim_corrs = [stats.spearmanr(x, this_sim_y)[0] for this_sim_y in sim_y] 
#            actual_perc = p_of_s(sim_corrs, actual_corr)
#            iter_percs.append(actual_perc)
#        session_percs.append(iter_percs)
#    region_percs.append(session_percs)

def return_region_percs(this_dat):
    iter_percs = []
    iters = list(it.combinations(range(3),2))
    for this_iter in iters:
        #this_iter = iters[0] 
        trans_comp = np.stack([this_dat[0][this_iter[0]], 
                            this_dat[1][this_iter[1]]])
        x = trans_comp[0]
        y = trans_comp[1]
        actual_corr = stats.spearmanr(x,y)[0]
        max_y = int(y.max() + 1)
        sim_num = 10000
        # Strictly, y >= x
        sim_y = np.stack([np.random.randint(this_x, max_y, sim_num) \
                    for this_x in x]).T
        ## For each x_i, pick a y_sim such that y_sim >= y_i
        #sim_y = np.stack([np.random.randint(this_y, max_y, sim_num) \
        #            for this_y in y]).T
        sim_corrs = [stats.spearmanr(x, this_sim_y)[0] for this_sim_y in sim_y] 
        actual_perc = p_of_s(sim_corrs, actual_corr)
        iter_percs.append(actual_perc)
    return iter_percs

region_percs = []
for region_list in [gc_list, bla_list]:
    outs = parallelize(return_region_percs, region_list)
    region_percs.append(outs)

## Example plot of shuffle
min_val = np.min(np.concatenate([x,y]), axis=None)
max_val = np.max(np.concatenate([x,y]), axis=None)
linx = liny = np.linspace(min_val, max_val) 
plt.scatter(x,y, zorder = 2, label = 'Actual Data')
plt.scatter(np.broadcast_to(x[np.newaxis,:], sim_y.shape), sim_y,
                alpha = 0.01,
                zorder = 1, s=5, label = 'Simulated Data')
plt.xlabel('First Transition')
plt.ylabel('Second Transition')
plt.plot(linx,liny, color = 'red')
plt.legend()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'sequential_corr_sim_example'), format='png')
plt.close(fig)
#plt.show()

#session_percs_array = np.stack(session_percs)
region_percs_array = np.stack(region_percs)

perc_thresh = 90
summed_sig_counts = np.sum(region_percs_array >= perc_thresh, axis = 1)

# Calculate binomial probability
total_count = region_percs_array.shape[1] 
x = np.arange(total_count)
rv = stats.binom(total_count, (100-perc_thresh)/100)
prob = rv.pmf(x) 
#plt.plot(x,prob)
#plt.show()
binom_p_vals = np.empty(summed_sig_counts.shape)
for this_iter in list(np.ndindex(summed_sig_counts.shape)):
    binom_p_vals[this_iter] = np.sum(prob[int(summed_sig_counts[this_iter]):])

alpha = 0.05 / binom_p_vals.shape[1]
binom_sig = binom_p_vals < alpha

#plt.hist(sim_corrs, bins = 20, alpha = 0.7)
#plt.axvline(actual_corr, color = 'red')
#plt.suptitle(f'Percentile : {actual_perc}') 
#plt.show()

############################################################

# Compare state onset with duration
gc_onset = tau_list[:,0]
gc_dur = np.diff(gc_onset, axis=1)

bla_onset = tau_list[:,0]
bla_dur = np.diff(bla_onset, axis=1)

# Session x Transition x (onset, dur) x Trial
#fin_dat = np.stack([gc_onset[:,:-1], gc_dur]).swapaxes(0,1).swapaxes(1,2)
fin_dat = np.stack([bla_onset[:,:-1], bla_dur]).swapaxes(0,1).swapaxes(1,2)

fig,ax = plt.subplots(fin_dat.shape[1], fin_dat.shape[0],
        sharex=True, sharey=True)
for num, this_dat in enumerate(fin_dat):
    for trans_num, this_trans in enumerate(this_dat):
        ax[trans_num,num].scatter(*this_trans,
                                alpha = 0.5, s = 5)
plt.show()

########################################
gc_outs = parallelize(return_corr_percentile, gc_list)
bla_outs = parallelize(return_corr_percentile, bla_list)

gc_percentile_array, _, _ = list(zip(*gc_outs))
bla_percentile_array, _, _ = list(zip(*bla_outs))
gc_percentile_array = np.stack(gc_percentile_array)
bla_percentile_array = np.stack(bla_percentile_array)

percentile_array = np.stack([gc_percentile_array, bla_percentile_array])
region_order = ['gc','bla']
#corr_array = np.stack(corr_array)
#shuffle_array = np.stack(shuffle_array)


# Find number of significant correlations
sig_perc = 90
sig_frac = np.mean(percentile_array >= sig_perc, axis=1)
sig_count = np.sum(percentile_array >= sig_perc, axis=1)

# Calculate binomial probability
total_count = percentile_array.shape[1]
x = np.arange(total_count)
rv = stats.binom(total_count, (100-sig_perc)/100)
prob = rv.pmf(x) 
#plt.plot(x,prob)
#plt.show()
binom_p_vals = np.empty(sig_count.shape)
for this_iter in list(np.ndindex(sig_count.shape)):
    binom_p_vals[this_iter] = np.sum(prob[int(sig_count[this_iter]):])

alpha = 0.05
binom_sig = binom_p_vals < alpha

########################################
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
########################################

plot_sig_frac = np.tril(sig_frac, -1)
plot_sig_frac = np.ma.masked_equal(plot_sig_frac, 0)

fig,ax = plt.subplots(2,1, figsize = (7,10))
#cbar_ax = fig.add_subplot(1,3,3)
#cmap = plt.cm.get_cmap('viridis') # jet doesn't have white color
#cmap.set_bad('w') # default value is 'k'
for num in range(len(plot_sig_frac)): 
    im = ax[num].matshow(plot_sig_frac[num], 
            interpolation="nearest", cmap='jet',
            vmin = 0, vmax = 1)
    this_binom_sig = np.tril(binom_sig, -1)[num]
    iters = list(np.ndindex(this_binom_sig.shape))
    for this_iter in iters:
        if this_binom_sig[this_iter]:
            text = ax[num].text(this_iter[1], this_iter[0], "*",
                    ha="center", va="center", color="black",)
            text.set_fontsize(20)
    tick_vals = np.arange(this_binom_sig.shape[0])
    ax[num].set_xticks(tick_vals)
    ax[num].set_xticklabels(tick_vals+1)
    ax[num].set_yticks(tick_vals)
    ax[num].set_yticklabels(tick_vals+1)
    ax[num].set_ylabel(f'Fraction of significant correlations : {region_order[num]}')
    #ax[num].set_xlabel("BLA Transition")
    for key,val in ax[num].spines.items():
        val.set_visible(False)
    ax[num].set_xticks(tick_vals - 0.5, minor = True)
    ax[num].set_yticks(tick_vals - 0.5, minor = True)
    ax[num].grid(which="minor", color="w", linestyle='-', linewidth=5)
    ax[num].tick_params(which="minor", bottom=False, left=False)
fig.subplots_adjust(top=0.8)
#plt.colorbar(im, ax = cbar_ax)
fig.colorbar(im, ax=ax, shrink=0.6, location = 'right')
plt.suptitle('All-to-all transition correlation \n' + \
        f"* = Upper tailed Binomial p-value (uncorrected) < alpha ({alpha})")
#plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'intra_region_all_corrs'), format='svg')
plt.close(fig)
#plt.show()

