"""
Conglomerate of analysis to explore variability in firing properties
and taste responses of GC neurons

Repeat analyses with non-parametric tests
"""

# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   

import numpy as np
import tables
import glob
######################### Import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt
import sys 
from tqdm import trange
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing as mp

sys.path.append('/media/bigdata/firing_space_plot/_old')
from ephys_data import ephys_data

from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

## Load data
dir_list = dir_list = ['/media/bigdata/jian_you_data/des_ic',
                        '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

all_off_data = []
for file in trange(len(file_list)):

    this_dir = file_list[file].split(sep='/')[-2]
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                                   [25,250,7000,'conv',269]))
    data.get_data()
    data.get_firing_rates()
    data.get_normalized_firing()
    
    all_off_data.append(np.asarray(data.normal_off_firing))

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

stim_delivery_ind = int(2000/25)
end_ind = int(4500/25)
# Get 2000 ms post-stim firing
all_poststim_data = [np.swapaxes(file[:,:,:,stim_delivery_ind:end_ind],0,1) \
        for file in all_off_data]

# Sort to have top list be neurons
neuron_list = [file[nrn,:,:,:] for file in all_poststim_data \
        for nrn in range(file.shape[0])]
neuron_array = np.asarray(neuron_list)

# Unroll array along taste for plotting
neuron_array_long = np.reshape(neuron_array,\
        (neuron_array.shape[0],neuron_array.shape[1]*neuron_array.shape[2],
            neuron_array.shape[3]))

# Convert array to pandas object for ease with 2 X Anova

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs

idx = make_array_identifiers(neuron_array)
neuron_frame = pd.DataFrame(\
        data = { 'neuron' : idx[0].flatten(),
                'taste' : idx[1].flatten(),
                'trial' : idx[2].flatten(),
                'time' : idx[3].flatten(),
                'firing_rate' : neuron_array.flatten() })

# Add infinitesimal noise to firing rate to avoid 0-related errors
neuron_frame['firing_rate'] += \
        np.random.rand(neuron_frame['firing_rate'].shape[0])* 1e-9

# Convert time into discrete 500ms bins
time_bin_frame = neuron_frame.copy()
time_bin_frame['time_bin'] = pd.cut(time_bin_frame.time,
        bins =4 ,include_lowest = True, labels = np.arange(4))

time_bin_frame.drop('time',inplace=True,axis=1)
time_bin_frame =\
time_bin_frame.groupby(['neuron','taste','trial','time_bin'])\
                .sum().reset_index()

# _____         _       
#|_   _|_ _ ___| |_ ___ 
#  | |/ _` / __| __/ _ \
#  | | (_| \__ \ ||  __/
#  |_|\__,_|___/\__\___|
#                       
# ____  _               _           _             _   _           
#|  _ \(_)___  ___ _ __(_)_ __ ___ (_)_ __   __ _| |_(_)_   _____ 
#| | | | / __|/ __| '__| | '_ ` _ \| | '_ \ / _` | __| \ \ / / _ \
#| |_| | \__ \ (__| |  | | | | | | | | | | | (_| | |_| |\ V /  __/
#|____/|_|___/\___|_|  |_|_| |_| |_|_|_| |_|\__,_|\__|_| \_/ \___|
                                                                 
# Mark which neurons are taste discriminative 

anova_list = [
    time_bin_frame.loc[time_bin_frame.neuron == nrn,:]\
            .rm_anova(dv = 'firing_rate', \
            within = ['time_bin','taste'], subject = 'trial') \
            for nrn in tqdm(time_bin_frame.neuron.unique())]

pairwise_ttest_list  = [
    time_bin_frame.loc[time_bin_frame.neuron == nrn,:]\
            .pairwise_ttests(dv = 'firing_rate', \
            within = 'taste', subject = 'trial', padjust = 'holm') \
            for nrn in tqdm(time_bin_frame.neuron.unique())]

#this_frame = time_bin_frame.query('neuron == 0').drop('neuron',axis=1)
#sns.pointplot(data = this_frame,
#   x='time_bin',y='firing_rate',hue='taste');plt.show()

#anova_results = pg.rm_anova( data = this_frame, dv = 'firing_rate', within =
#        ['time_bin','taste'], subject = 'trial')
#taste_p = anova_results['p-unc'][1]

# Extract number of taste discriminative units
taste_p_vec = np.asarray([anova_result['p-unc'][1] \
        for anova_result in anova_list])

# Mark taste responsive neurons
taste_responsive = taste_p_vec < 0.05

# _____ _      _                ____ _                            
#|  ___(_)_ __(_)_ __   __ _   / ___| |__   __ _ _ __   __ _  ___ 
#| |_  | | '__| | '_ \ / _` | | |   | '_ \ / _` | '_ \ / _` |/ _ \
#|  _| | | |  | | | | | (_| | | |___| | | | (_| | | | | (_| |  __/
#|_|   |_|_|  |_|_| |_|\__, |  \____|_| |_|\__,_|_| |_|\__, |\___|
#                      |___/                           |___/      

# Mark units which change firing rate in first vs last half of trials
trial_bin_frame = time_bin_frame.copy()
trial_bin_frame['trial_bin'] = pd.cut(trial_bin_frame.trial,
        bins =2 ,include_lowest = True, labels = range(2))


# Mark how many units have significant difference in first 
# and last half of trials

#def trial_bin_anova(dframe):
#    return pg.mixed_anova(data = dframe, dv = 'firing_rate', \
#            within = ['taste'], \
#            between = 'trial_bin', subject = 'trial') 
#
#trial_anova_list = Parallel(n_jobs = mp.cpu_count())\
#    (delayed(trial_bin_anova)\
#    (trial_bin_frame.loc[trial_bin_frame.neuron == nrn,:])\
#            for nrn in tqdm(trial_bin_frame.neuron.unique()))

# Serial version of loop
#trial_anova_list = [
#    trial_bin_anova(
#    trial_bin_frame.loc[trial_bin_frame.neuron == nrn,:])\
#            for nrn in tqdm(trial_bin_frame.neuron.unique())]

# 3 Way ANOVA for time, trial and taste

#def trial_bin_anova(dframe):
#    return pg.anova(data = dframe,
#            dv = 'firing_rate', \
#            between = ['time_bin','trial_bin', 'taste']) 
#
#trial_anova_list = Parallel(n_jobs = mp.cpu_count())\
#    (delayed(trial_bin_anova)\
#    (trial_bin_frame.loc[trial_bin_frame.neuron == nrn,:])\
#            for nrn in tqdm(trial_bin_frame.neuron.unique()))


# ANOVA for each taste for every neuron

def trial_bin_anova(dframe):
    return [pg.anova(data = dframe.loc[dframe.taste==taste,:], 
            dv = 'firing_rate', \
            between = 'trial_bin') \
            for taste in dframe.taste.unique()]

trial_anova_list = Parallel(n_jobs = mp.cpu_count())\
    (delayed(trial_bin_anova)\
    (trial_bin_frame.loc[trial_bin_frame.neuron == nrn,:])\
            for nrn in tqdm(trial_bin_frame.neuron.unique()))

# Extract p-value from anova list
trial_plist = [[taste['p-unc'][0] \
        for taste in neuron] \
        for neuron in trial_anova_list]


# Mark neurons with any change in responsesi
# For 4 tastes, correcting alpha to be < 0.01
changed_tastes = np.sum(np.asarray(trial_plist) < 0.01,axis=1)

#sns.distplot(changed_tastes[changed_tastes>0],\
#        bins = np.arange(1,6),kde=False);plt.show()

#nrn = [125,125] 
#this_frame = trial_bin_frame.query('neuron >= @nrn[0] and  neuron <= @nrn[1]')
this_frame = pd.concat(
        [trial_bin_frame.loc[trial_bin_frame.neuron == nrn,:] \
                for nrn in np.where(changed_tastes == 4)[0]])
g = sns.FacetGrid(this_frame, row = 'taste', \
        col = "neuron", hue = "trial_bin")
g = g.map(sns.pointplot, 'time_bin', 'firing_rate', ci = 'sd')\
        .add_legend()
plt.show()

# Mark how many taste discriminatory vs non-discriminatory units
# showed drift in responses
discrimatory_drift = sum((changed_tastes > 0) * (taste_responsive == 1))
nondiscrimintory_drift = sum((changed_tastes > 0) * (taste_responsive == 0))

# How many discriminatory units became non after splitting and
# vice versa
# Neurons becoming taste discriminative after splitting would mean
# that they are instantaneously taste responsive (and not canonically)

# How many significantly different time bins were increased
# in all by splitting
