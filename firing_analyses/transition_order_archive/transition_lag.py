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

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/transition_order'

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs/all_tastes')
from check_data import check_data 
import itertools as it

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

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
region_order = ['gc','bla']
# Indexing = (gc,bla)

tau_list = [[load_mode_tau(x) for x in y] for y in tqdm(session_pkls_sorted)]
tau_list = np.stack(tau_list).swapaxes(-2,-1)
tau_list = tau_list.swapaxes(0,1)

#array = tau_list[0,0]
def all_to_all_diffs(array1, array2):
    """
    Input
    =====
    array1, array2  :: transitions x trials

    Output
    ======
    diff_array :: transitions x transitions x trials 
    """
    assert array1.shape == array2.shape

    trans_vec = np.arange(array1.shape[0])
    iters = list(it.product(trans_vec, trans_vec))
    diff_array = np.empty((len(trans_vec), len(trans_vec), array.shape[1]))
    for this_iter in iters:
        diff_array[this_iter] = array1[this_iter[0]] - array2[this_iter[1]]
    return diff_array

intra_diffs = np.stack(
        [[all_to_all_diffs(session,session) for session in tqdm(region)]\
        for region in tau_list])
inter_diffs = np.stack([all_to_all_diffs(x1,x2) \
        for x1,x2 in tau_list.swapaxes(0,1)])

intra_diff_inds = np.array(list(np.ndindex(intra_diffs.shape)))
intra_diff_frame = pd.DataFrame(
                    dict(
                        region_num = intra_diff_inds[:,0],
                        session_num = intra_diff_inds[:,1],
                        trans1 = intra_diff_inds[:,2],
                        trans2 = intra_diff_inds[:,3],
                        trials = intra_diff_inds[:,4],
                        vals = intra_diffs.flatten()))

inter_diff_inds = np.array(list(np.ndindex(inter_diffs.shape)))
inter_diff_frame = pd.DataFrame(
                    dict(
                        session_num = inter_diff_inds[:,0],
                        trans1 = inter_diff_inds[:,1],
                        trans2 = inter_diff_inds[:,2],
                        trials = inter_diff_inds[:,3],
                        vals = inter_diffs.flatten()))


gc_frame = intra_diff_frame.query('region_num == 0')
hists = [np.histogram(dat['vals'], 20) \
        for num,dat in gc_frame.groupby(['trans1','trans2'])]
modes = [val[np.argmax(count)] for count,val in hists]
fig,ax = plt.subplots(gc_frame.trans1.unique().size,
        gc_frame.trans2.unique().size,
        sharex = True, sharey=False,
        figsize = (10,7))
for (this_t1, this_t2), dat in gc_frame.groupby(['trans1','trans2']):
    ax[this_t1,this_t2].hist(dat['vals'], bins = 20)
    ax[this_t1,this_t2].set_title((this_t1,this_t2))
for val, this_ax in zip(modes, ax.flatten()):
    this_ax.axvline(val, color = 'red', linestyle = '--')
    this_ax.text(-35,150,f'mode : {int(val * 50)} ms') 
plt.suptitle('GC Intra Transition Lags')
plt.subplots_adjust(top=0.9, hspace = 0.2)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'gc_intra_region_lags'))
plt.close(fig)
#plt.show()

#ax = sns.displot(gc_frame,
#        x = 'vals',
#        row = 'trans1', col = 'trans2',
#        hue = 'region_num',
#        facet_kws=dict(sharey=False),
#        bins = 30, kde = True)
#for val, this_ax in zip(modes, ax.axes.flatten()):
#    this_ax.axvline(val, color = 'red', linestyle = '--')
#    this_ax.text(-35,150,f'mode : {int(val * 50)} ms') 
#plt.show()

bla_frame = intra_diff_frame.query('region_num == 1')
hists = [np.histogram(dat['vals'], 20) \
        for num,dat in bla_frame.groupby(['trans1','trans2'])]
modes = [val[np.argmax(count)] for count,val in hists]
fig,ax = plt.subplots(bla_frame.trans1.unique().size,
        bla_frame.trans2.unique().size,
        sharex = True, sharey=False,
        figsize = (10,7))
for (this_t1, this_t2), dat in bla_frame.groupby(['trans1','trans2']):
    ax[this_t1,this_t2].hist(dat['vals'], bins = 20)
    ax[this_t1,this_t2].set_title((this_t1,this_t2))
for val, this_ax in zip(modes, ax.flatten()):
    this_ax.axvline(val, color = 'red', linestyle = '--')
    this_ax.text(-35,150,f'mode : {int(val * 50)} ms') 
plt.suptitle('BLA Intra Transition Lags')
plt.subplots_adjust(top=0.9, hspace = 0.2)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'bla_intra_region_lags'))
plt.close(fig)
#plt.show()

#ax = sns.displot(bla_frame,
#        x = 'vals',
#        row = 'trans1', col = 'trans2',
#        hue = 'region_num',
#        facet_kws=dict(sharey=False),
#        bins = 30, kde = True)
#for val, this_ax in zip(modes, ax.axes.flatten()):
#    this_ax.axvline(val, color = 'red', linestyle = '--')
#    this_ax.text(-35,150,f'mode : {int(val * 50)} ms') 
#plt.show()


# Calculate mode of each histogram manually
hists = [np.histogram(dat['vals'], 30) \
        for num,dat in inter_diff_frame.groupby(['trans1','trans2'])]

modes = [val[np.argmax(count)] for count,val in hists]

ax = sns.displot(inter_diff_frame,
        x = 'vals',
        row = 'trans1', col = 'trans2',
        #bins = 30,
        facet_kws=dict(sharey=False),
        aspect = 1,
        kde = True)
for val, this_ax in zip(modes, ax.axes.flatten()):
    this_ax.axvline(val, color = 'red', linestyle = '--')
    this_ax.text(-35,150,f'Mode : {int(val * 50)} ms') 
plt.suptitle('Inter Region Transition Lags' 
        '\n' + 'Trans1 = GC, Trans2 = BLA')
plt.subplots_adjust(top=0.9, hspace = 0.2)
#plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'inter_region_lags'))
plt.close(fig)
#plt.show()

# Average by session
mean_inter_diff_frame = inter_diff_frame.groupby(
        ['session_num','trans1','trans2']).mean().reset_index()
mean_vals = [dat['vals'].mean() \
        for num,dat in mean_inter_diff_frame.groupby(['trans1','trans2'])]

ax = sns.displot(mean_inter_diff_frame,
        x = 'vals',
        row = 'trans1', col = 'trans2',
        bins = 30,
        aspect = 1,
        kde = True)
for val, this_ax in zip(mean_vals, ax.axes.flatten()):
    this_ax.axvline(val, color = 'red', linestyle = '--')
    this_ax.text(-25,5,f'Mode : {int(val * 50)} ms') 
plt.suptitle('Inter Region Transition Lags (Mean per recording)' 
        '\n' + 'Trans1 = GC, Trans2 = BLA')
plt.subplots_adjust(top=0.9, hspace = 0.2)
#plt.show()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'mean_inter_region_lags'))
plt.close(fig)

