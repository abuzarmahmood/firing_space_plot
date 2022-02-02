"""
Summarize transition order per transition per session
and aggregate across recordings

Aggreation groups : All sessions, "Good" sessions
"""

########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

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

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs/all_tastes')
from check_data import check_data 

def load_tau(model_path):
    if os.path.exists(model_path):
        print('Trace loaded from cache')
        with open(model_path, 'rb') as buff:
            data = pickle.load(buff)
        tau_samples = data['tau']
        # Remove pickled data to conserve memory
        del data
    return tau_samples

class params_from_path:
    def __init__(self, path):
        # Extract model params from basename
        self.path = path
        self.model_name = os.path.basename(self.path).split('.')[0]
        self.states = int(re.findall("\d+states",self.model_name)[0][:-6])
        self.time_lims = [int(x) for x in \
                re.findall("\d+_\d+time",self.model_name)[0][:-4].split('_')]
        self.bin_width = int(re.findall("\d+bin",self.model_name)[0][:-3])
        self.region_name = re.findall("region_\w+?\_",self.model_name)[0].split("_")[-2]
        # Exctract data_dir from model_path
        self.data_dir = "/".join(self.path.split('/')[:-3])
        self.session_name = self.data_dir.split('/')[-1]
        self.animal_name = self.session_name.split('_')[0]
    def to_dict(self):
        return dict(zip(['states','time_lims','bin_width','session_name'],
            [self.states,self.time_lims,self.bin_width,self.session_name]))

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/inter_region'
#wanted_names = ['rho_percentiles','mode_tau','rho_shuffles',
#        'tau_corrs','tau_list'] 

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/transition_order/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/multi_region_frame.pkl'
inter_frame = pd.read_pickle(data_dir_pkl)
inter_frame['animal_name'] = [x.split('_')[0] for x in inter_frame['name']]

black_list = ['AM26_4Tastes_200828_112822', 'AM18','AM37']

# Pull out fit params for one file
data_dir = inter_frame.path.iloc[0]
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
params = params_from_path(inter_region_paths[0]) 

##################################################
#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           
##################################################

session_num_list = []
tau_diff_list = []
region_order_list = []
percentile_list = []
for num, data_dir in tqdm(enumerate(inter_frame.path)):

    if data_dir[-1] == '/':
        temp_name = data_dir[:-1]
    else:
        temp_name = data_dir
    basename = os.path.basename(temp_name)
    if (basename in black_list) or any([x in basename for x in black_list]):
        continue

    dat = ephys_data(data_dir)

    with tables.open_file(dat.hdf5_path,'r') as hf5:
        if save_path in hf5:
            this_dat = hf5.get_node(save_path, 'rho_percentiles')[:] 
            percentile_list.append(this_dat)
        else:
            # If not present, ignore everything else
            continue

    this_info = check_data(data_dir)
    this_info.run_all()
    inter_region_paths = [path for  num,path in enumerate(this_info.pkl_file_paths) \
                    if num in this_info.region_fit_inds]
    state4_models = [path for path in inter_region_paths if '4state' in path]
    split_basenames = [os.path.basename(x) for x in state4_models]

    full_tau = np.array([load_tau(this_path).T for this_path in state4_models])
    mode_tau = np.squeeze(stats.mode(np.vectorize(np.int)(full_tau),axis=3)[0])
    region_names = [params_from_path(x).region_name for x in state4_models]

    sort_order = np.argsort(region_names)
    sorted_mode_tau = mode_tau[sort_order] 
    diff_mode_tau = np.diff(sorted_mode_tau, axis = 0)
    sorted_region_names = np.array(region_names)[sort_order]

    session_num_list.append(num)
    tau_diff_list.append(diff_mode_tau)
    region_order_list.append(sorted_region_names)


#dat_list_zip = list(zip(*dat_list))
#dat_list_zip = [np.stack(x) for x in dat_list_zip]
#for this_var, this_dat in zip(wanted_names, dat_list_zip):
#    globals()[this_var] = this_dat 

session_name_list = [inter_frame.path.iloc[x] for x in session_num_list]
session_basename_list = [os.path.basename(x) for x in session_name_list]

percentile_array = np.array(percentile_list)
perc_thresh = 90
perc_pass_inds = [np.where(x>=perc_thresh)[0] for x in percentile_array.T] 

tau_diff_array = np.squeeze(np.array(tau_diff_list))
select_tau_diff = [tau_diff_array[:,num][x] for num,x in enumerate(perc_pass_inds)] 
select_tau_names = [[session_basename_list[x] for x in y] \
                            for y in perc_pass_inds] 

########################################
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
########################################
stat_used = np.mean

## Plot individual sessions

for ses_num in trange(len(session_num_list)):

    #num = 0
    this_diff = np.squeeze(tau_diff_list[ses_num])

    # Bootstrap 95% CI for the median of each diff
    boot_samples = int(1e4)
    boot_inds = np.random.randint(0, this_diff.shape[1], 
            (this_diff.shape[1], boot_samples))
    boot_diff = np.array([x[boot_inds] for x in this_diff])
    boot_diff_med = stat_used(boot_diff, axis = 1)
    boot_med_percs = np.percentile(boot_diff_med, [2.5,97.5], axis = 1)

    fig,ax = plt.subplots(this_diff.shape[0],1, sharex=True, figsize = (5,10))
    for num, vals in enumerate(this_diff):

        effect_size = np.round(np.abs(stat_used(vals))/np.std(vals),3)

        ax[num].hist(vals, bins = 20, label = 'Tau difference')
        ax[num].axvspan(*boot_med_percs[:,num], alpha = 0.5, color = 'red')
        ax[num].axvline(stat_used(vals), color = 'red', 
                linestyle = 'dashed', label = 'Mean +/- 95 CI')
        ax[num].axvline(0, color = 'k', linewidth = 2, label = '0')
        ax[num].set_title(f'Trans {num} :: Effect size : {effect_size}')
    ax[-1].legend()
    plt.suptitle(session_basename_list[ses_num] + "\n" + str(region_order_list[ses_num]))
    fig.savefig(os.path.join(plot_dir, session_basename_list[ses_num] + "_tau_diff_hists"))
    plt.close(fig)
    #plt.show()

## Plot sessions aggregated
bins = np.linspace(np.min(tau_diff_array.flatten()),
                    np.max(tau_diff_array.flatten()), 30)

tau_diff_hists = np.zeros((*tau_diff_array.shape[:-1], len(bins)-1))
inds = list(np.ndindex(tau_diff_array.shape[:-1]))
for this_ind in inds:
    tau_diff_hists[this_ind] = np.histogram(tau_diff_array[this_ind], bins = bins)[0]

stat_tau_diff = stat_used(tau_diff_array,axis=-1) 
stat_tau_diff_stat = stat_used(stat_tau_diff, axis=0)
stat_tau_diff_effect = np.round(
        np.abs(stat_tau_diff_stat)/np.std(stat_tau_diff,axis=0),2)

boot_inds = np.random.randint(0, stat_tau_diff.shape[0], 
        (stat_tau_diff.shape[0], boot_samples))
boot_stat_tau_diff = np.array([x[boot_inds] for x in stat_tau_diff.T])
boot_stat_tau_diff_stat = stat_used(boot_stat_tau_diff, axis = 1)
boot_stat_diff_percs = np.percentile(boot_stat_tau_diff_stat, [2.5,97.5], axis = -1)

stat_bins = 20
#stat_tau_diff_hist = [np.histogram(x,bins=stat_bins) for x in stat_tau_diff.T]
#min_x = np.min([np.min(x[1]) for x in stat_tau_diff_hist])
#max_x = np.max([np.max(x[1]) for x in stat_tau_diff_hist])

fig, ax = plt.subplots(2,tau_diff_hists.shape[1], sharex = 'row', figsize = (10,10))
for num in range(tau_diff_hists.shape[1]):
    ax[0,num].pcolormesh(bins, np.arange(tau_diff_hists.shape[0]),tau_diff_hists[:,num])
    ax[0,num].axvline(0, color = 'red', linestyle = 'dashed')
    ax[0,num].set_title(f'Transition {num}')
    ax[1,num].hist(stat_tau_diff[:,num])#, bins = stat_bins)
    ax[1,num].axvline(0, color = 'k', linewidth = 2, label = '0')
    ax[1,num].axvline(stat_tau_diff_stat[num], 
            color = 'red', linewidth = 2, linestyle = 'dashed', 
            label = 'Mean +/- 95 CI')
    ax[1,num].axvspan(*boot_stat_diff_percs[:,num], 
            color = 'red', alpha = 0.5)
    ax[1,num].set_title(f'CI : {str(np.round(boot_stat_diff_percs[:,num],2))}, ' + \
            f'eta : {stat_tau_diff_effect[num]}')
    #ax[1,num].set_xlim(min_x,max_x)
ax[-1,-1].legend()
ax[0,0].set_ylabel('Tau Diff Hists')
ax[1,0].set_ylabel('Median Tau Diff')
plt.suptitle('Aggregate Tau difference')
fig.savefig(os.path.join(plot_dir, 'agg_tau_diff'))
plt.close(fig)
#plt.show()

## Plot "GOOD" sessions aggregated
stack_select_tau_diff = np.concatenate(select_tau_diff,axis=0)
bins = np.linspace(np.min(stack_select_tau_diff.flatten()),
                    np.max(stack_select_tau_diff.flatten()), 30)

select_tau_diff_hists = [np.stack([np.histogram(x, bins = bins)[0] for x in y]) \
                for y in select_tau_diff]

fig, ax = plt.subplots(2,len(select_tau_diff_hists), sharex = 'row', figsize = (10,10))
for num, (this_hist, this_tau) in enumerate(zip(select_tau_diff_hists, select_tau_diff)): 
    ax[0,num].pcolormesh(bins, np.arange(this_hist.shape[0]+1), this_hist)
    ax[0,num].axvline(0, color = 'red', linestyle = 'dashed')
    ax[0,num].set_title(f'Transition {num}')
    this_stat_tau = stat_used(this_tau, axis = 1)
    this_stat_stat = stat_used(this_stat_tau)
    this_stat_effect = np.round(np.abs(this_stat_stat)/np.std(this_stat_tau),2)
    boot_inds = np.random.randint(0, len(this_stat_tau),
            (len(this_stat_tau), boot_samples))
    boot_stat_tau_diff = this_stat_tau[boot_inds]
    boot_stat_tau_diff_stat = stat_used(boot_stat_tau_diff, axis = 0)
    boot_stat_diff_percs = np.percentile(boot_stat_tau_diff_stat, [2.5,97.5])
    ax[1,num].hist(this_stat_tau)#, bins = stat_bins)
    ax[1,num].axvline(0, color = 'k', linewidth = 2, label = '0')
    ax[1,num].axvline(this_stat_stat,
            color = 'red', linewidth = 2, linestyle = 'dashed', 
            label = 'Mean +/- 95 CI')
    ax[1,num].axvspan(*boot_stat_diff_percs,
            color = 'red', alpha = 0.5)
    ax[1,num].set_title(f'CI : {str(np.round(boot_stat_diff_percs,2))}, ' + \
            f'eta : {this_stat_effect}')
ax[-1,-1].legend()
ax[0,0].set_ylabel('Tau Diff Hists')
ax[1,0].set_ylabel('Median Tau Diff')
plt.suptitle('Aggregate Tau difference "GOOD"')
fig.savefig(os.path.join(plot_dir, 'agg_tau_diff_good'))
plt.close(fig)
#plt.show()
