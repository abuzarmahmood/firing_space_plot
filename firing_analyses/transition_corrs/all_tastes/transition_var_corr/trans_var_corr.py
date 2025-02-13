"""
Compare strength of correlation between BLA and GC to
uncertainty in estimation of changepoint position

Do this for both significant correlations and all data
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
from scipy.stats import linregress
from pprint import pprint

#import theano
#theano.config.compute_test_value = "ignore"

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs/all_tastes')
from check_data import check_data 
import itertools as it

def parallelize(func, iterator):
    """parallelize.

    Args:
        func:
        iterator:
    """
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def corr_percentile_single(a,b, shuffles = 1000):
    """corr_percentile_single.

    Args:
        a:
        b:
        shuffles:
    """
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
    """params_from_path.
    """

    def __init__(self, path):
        """__init__.

        Args:
            path:
        """
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
        """to_dict.
        """
        return dict(zip(['states','time_lims','bin_width','session_name'],
            [self.states,self.time_lims,self.bin_width,self.session_name]))

def load_mode_tau(model_path):
    """load_mode_tau.

    Args:
        model_path:
    """
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
    return int_mode_tau, int_tau

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/inter_region'
wanted_names = ['rho_percentiles','mode_tau','rho_shuffles',
        'tau_corrs','tau_list'] 

# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/inter_region/multi_region_frame.pkl'
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

# Get tau samples as well as mode_tau
tau_list = [[load_mode_tau(x) for x in y] for y in tqdm(session_pkls_sorted)]

mode_tau = np.stack([[x[0] for x in y] for y in tau_list])
raw_tau = np.stack([[x[1] for x in y] for y in tau_list])

mode_tau = mode_tau.swapaxes(-2,-1)
raw_tau = raw_tau.swapaxes(-2,-1)

raw_tau_var = np.var(raw_tau, axis = 2)
## Use summed variance as a measure
#sum_tau_var = np.sum(raw_tau_var,axis=1) 
## Average summed variance across trials
#mean_sum_tau_var = np.mean(sum_tau_var, axis=-1)

########################################
## All to all transition correlation
########################################
# Check whether the latency correlation was significant

outs = parallelize(return_corr_percentile, mode_tau)
percentile_array, corr_array, shuffle_array = list(zip(*outs))
percentile_array = np.stack(percentile_array)
corr_array = np.stack(corr_array)
shuffle_array = np.stack(shuffle_array)


# Find number of significant correlations
sig_perc = 90
sig_perc_array = percentile_array >= sig_perc
sig_frac = np.mean(sig_perc_array, axis=0)
sig_count = np.sum(sig_perc_array, axis=0)

########################################
# Diagonal comparisons only
########################################
diag_corrs = np.stack([x[np.diag_indices(len(x))] for x in corr_array])
diag_sig = np.stack([x[np.diag_indices(len(x))] for x in sig_perc_array])

# Correlate variances of all transitions 
temp_tau_var = np.swapaxes(raw_tau_var, 1,2)
inds = np.array(list(np.ndindex(temp_tau_var.shape[:2])))

var_corr_rho_array = np.zeros(temp_tau_var.shape[:2])
var_corr_p_array = np.zeros(temp_tau_var.shape[:2])

for this_ind in inds:
    this_dat = temp_tau_var[tuple(this_ind)]
    rho, p_val = stats.spearmanr(this_dat[0], this_dat[1])
    var_corr_rho_array[tuple(this_ind)] = rho
    var_corr_p_array[tuple(this_ind)] = p_val

inds = np.array(list(np.ndindex(diag_corrs.shape)))
corr_var_frame = pd.DataFrame(dict(
            session_num = inds[:,0],
            transition_num = inds[:,1],
            corr = diag_corrs.flatten(),
            var_corr_rho = var_corr_rho_array.flatten(),
            var_corr_p = var_corr_p_array.flatten(),
            sig_bool = diag_sig.flatten()))

corr_var_frame[corr_var_frame['var_corr_p'] < 0.05]
corr_var_frame[corr_var_frame['sig_bool'] == True]
