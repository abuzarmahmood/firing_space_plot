"""
Testing single-trial specific firing changes in different neuronal populations
using linear regression
- If these populations share activity changes specific to single trials,
    trial-matched models will be better than trial-shuffled models
- Calculate goodness of fit for intra-region, inter-region,
    and trial-shuffled for both cases
    -- Perform intra-region by splitting single region recordings (as currently)
    -- Error bounds can be found by leaving neurons out
    -- Results for different inter-regional calculations can be scaled
        by performing the calculation with the same number of neurons 
        in an intra-region session
ISSUES:
    - Test only on taste discriminative neurons
        - Separate out by neurons responsive to a particular taste
        - If a neuron doesn't respond to a particular taste,
            no point in trying to compare responses if a neuron isn't
            responding to a taste
    - Should prediction always be on trial-matched data even if
        training is on trial-shuffled data
    - Should latent regression (e.g. PC Regresson or PLSR) be used
        to avoid overfitting
"""
#############################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
#############################

# Import modules

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore, percentileofscore
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
#from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from joblib import Parallel, delayed, cpu_count
import argparse
import sys
import pickle

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

vector_percentile = lambda dist,vals: np.array(\
        list(map(lambda vals: percentileofscore(dist, vals), vals)))

def trial_shuffle_gen(label_list):
    new_trial_order = np.random.permutation(np.unique(label_list))
    return np.concatenate([np.where(label_list == x)[0] for x in new_trial_order])

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

###################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
###################################################

parser = argparse.ArgumentParser(\
        description = 'Script to calculate linear regression fit'\
                            'between neural populations')
parser.add_argument('data_dir',
                help = 'Where to load spike_trains from')
args = parser.parse_args()
data_dir = args.data_dir
#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201228_124547'
#data_dir = '/media/bigdata/Abuzar_Data/AM28/AM28_2Tastes_201005_134840'


plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
                            'firing_regression/Plots'
dat_save_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
                            'firing_regression/saved_data'
name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]
fin_plot_dir = os.path.join(plot_dir, fin_name)
fin_save_path = os.path.join(dat_save_dir, fin_name + ".pkl")

if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)
if not os.path.exists(dat_save_dir):
    os.makedirs(dat_save_dir)

dat = ephys_data(data_dir)
dat.firing_rate_params = dict(zip(\
    ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
    ('conv',25,250,1,25e-3,1e-3)))
dat.extract_and_process()
dat.get_region_units()

# Use half of neurons to predict activity in other half
time_lims = [2000,4000]
#step_size = dat.firing_rate_params['baks_resolution']
#dt = dat.firing_rate_params['baks_dt']
step_size = dat.firing_rate_params['step_size']
dt = dat.firing_rate_params['dt']
time_inds = np.vectorize(np.int)(np.array(time_lims)/(step_size/dt))


#taste = 0
#firing_array = dat.firing_array[taste][...,time_inds[0]:time_inds[1]]
#firing_array_long = firing_array.reshape((firing_array.shape[0],-1))
#firing_long_scaled = StandardScaler().fit_transform(firing_array_long.T) 

firing_array = dat.firing_array[...,time_inds[0]:time_inds[1]]
firing_array_long = firing_array.reshape((*firing_array.shape[:2],-1))
firing_long_scaled = [StandardScaler().fit_transform(x.T) \
        for x in firing_array_long]

trial_labels = np.repeat(np.arange(firing_array.shape[2]), firing_array.shape[3])


###################################################
# ____                              _             
#|  _ \ ___  __ _ _ __ ___  ___ ___(_) ___  _ __  
#| |_) / _ \/ _` | '__/ _ \/ __/ __| |/ _ \| '_ \ 
#|  _ <  __/ (_| | | |  __/\__ \__ \ | (_) | | | |
#|_| \_\___|\__, |_|  \___||___/___/_|\___/|_| |_|
#           |___/                                 
###################################################

##################################################
## Perform regression for all trials collectively
##################################################
## No crossvalidation
def run_regression(firing0, firing1 = None, 
        trial_labels = None, shuffle_repeats = 100, 
        cv_splitter = None, scoring_metric = None, cv_bool = False):
        """
        firing0 : 2D : (trials x time_bins) x nrns
        firing1 : 2D : Firing for second region
                        If firing1 not present, firing0 gets chopped up
        trial_labels : REQUIRED, but set to None to make it a kwarg
        cv_bool : use cross validation?
        """
        if trial_labels is None:
            raise Exception('trial_labels are required to generate shuffles')
        if cv_bool:
            if cv_splitter is None or scoring_metric is None:
                raise Exception('cv_splitter or scoring_metric not provided')

        if firing1 is None:
            grp1,grp2 = np.array_split(\
                    np.random.permutation(np.arange(firing0.shape[1])),2)

            X,y = firing0[:,grp1], firing0[:,grp2]
        else:
            X,y = firing0,firing1

        if cv_bool:
            lm = LinearRegression()
            cv_iter = cv_splitter.split(X, y, trial_labels)
            actual_score = cross_val_score(lm, X, y, 
                    cv=cv_iter, scoring = scoring_metric)
        else:
            reg = LinearRegression().fit(X, y)
            actual_score = r2_score(y,reg.predict(X))

        # Shuffled trials
        shuffled_scores = []
        for repeat in range(shuffle_repeats):

            X_sh = X[trial_shuffle_gen(trial_labels)] 
            y_sh = y[trial_shuffle_gen(trial_labels)]

            if cv_bool:
                lm = LinearRegression()
                cv_iter = cv_splitter.split(X_sh, y_sh, trial_labels)
                shuffled_scores.append(cross_val_score(lm, X_sh, y_sh, 
                    cv=cv_iter, scoring = scoring_metric))
            else:
                reg = LinearRegression().fit(X_sh, y_sh)
                shuffled_scores.append(r2_score(y_sh,reg.predict(X_sh)))

        if not cv_bool:
            shuffled_scores = [shuffled_scores]
            actual_score = [actual_score]
        actual_data_percentile = \
                vector_percentile(np.concatenate(shuffled_scores), actual_score)

        return actual_data_percentile, actual_score, shuffled_scores

def run_regression2(firing0, firing1, 
        trial_labels = None, shuffle_repeats = 100, scoring_metric = None):
        """
        Train on full data, and predict on full data and shuffled data

        firing0 : 2D : (trials x time_bins) x nrns
        firing1 : 2D : Firing for second region
                        If firing1 not present, firing0 gets chopped up
        trial_labels : REQUIRED, but set to None to make it a kwarg
        """
        if trial_labels is None:
            raise Exception('trial_labels are required to generate shuffles')

        X,y = firing0,firing1

        reg = LinearRegression().fit(X, y)
        actual_score = r2_score(y,reg.predict(X))

        ## If we're using r2 on the whole dataset as a metric for
        ## goodness of fit, our shuffles have to fit and tested
        ## on shuffled data aswell
        # Shuffled trials
        shuffled_scores = []
        for repeat in range(shuffle_repeats):

            X_sh = X[trial_shuffle_gen(trial_labels)] 
            y_sh = y[trial_shuffle_gen(trial_labels)]

            reg = LinearRegression().fit(X_sh, y_sh)
            shuffled_scores.append(r2_score(y_sh,reg.predict(X_sh)))

        actual_data_percentile = percentileofscore(shuffled_scores, actual_score)

        return actual_data_percentile, actual_score, shuffled_scores

########################################
#split_names = np.array(['firing0','firing1'])
#pkl_list = []
#
#if len(dat.region_names) < 2:
#    dat_arrays = [np.array_split(x,2,axis=-1) for x in firing_long_scaled]
#    region_names = [dat.region_names[0] + str(x) for x in range(2)]
#    region_sizes = [x.shape[1] for x in dat_arrays[0]]  
#
#else:
#    dat_arrays = [\
#                [x[:,dat.region_units[0]] for x in firing_long_scaled],
#                [x[:,dat.region_units[1]] for x in firing_long_scaled]]
#    region_names = dat.region_names
#    region_sizes = [len(x) for x in dat.region_units]
#
#for split_order in [[0,1],[1,0]]:
#    name_order = split_names[np.array(split_order)]
#    for this_dat, this_name in zip(dat_arrays,name_order):
#        globals()[this_name] = this_dat
#    fin_region_names = np.array(region_names)[np.array(split_order)]
#    fin_region_sizes = np.array(region_sizes)[np.array(split_order)]
#    # For ease with naming
#    fin_names_list = [x for x in fin_region_names]
#    fin_plot_name = "_".join(fin_names_list)
#
#
#    ########################################
#    # WITHOUT cross-validation
#    ########################################
#    percentile_list = []
#    actual_score_list = []
#    shuffled_score_list = []
#
#    for this_firing0, this_firing1 in tqdm(zip(firing0,firing1)):
#        percentile_val, actual_score, shuffled_score  = \
#                        run_regression2(this_firing0, this_firing1,
#                        trial_labels = trial_labels)
#        percentile_list.append(percentile_val)
#        actual_score_list.append(actual_score)
#        shuffled_score_list.append(shuffled_score)
#
#    fig,ax = plt.subplots(len(percentile_list),1, sharex=True, figsize=(5,10))
#    for this_shuffle, this_actual, this_percentile, this_ax in \
#            zip(shuffled_score_list, actual_score_list, 
#                    percentile_list, ax.flatten()):
#        this_ax.hist(this_shuffle)
#        this_ax.axvline(this_actual, linewidth = 2, color='red')
#        percentile_val = np.round(this_percentile)
#        this_ax.set_title(f'Actual Data : {percentile_val} percentile')
#    plt.suptitle(f'{dat.hdf5_name} \n no_cv_percentiles \n'\
#            f'X : {fin_region_names[0]} ({fin_region_sizes[0]}), '\
#            f'y : {fin_region_names[1]} ({fin_region_sizes[1]})')
#    #plt.show()
#
#    fig.savefig(os.path.join(fin_plot_dir,fin_plot_name + '_no_cv_score')) 
#
#    pkl_list.append(\
#            {'type' :'no_cv',
#            'percentiles' :  percentile_list, 
#            'actual_score' :  actual_score_list, 
#            'shuffled_score' :  shuffled_score_list, 
#            'region_order' : fin_region_names, 
#            'region_sizes' : fin_region_sizes})
#
#
#with open(fin_save_path, 'wb') as buff:
#    pickle.dump(pkl_list, buff)
#
########################################
# Since the quality of prediction depends on the size of the training data
# For inter-region testing, want to train and test on both BLA and GC
split_names = np.array(['firing0','firing1'])
split_repeat_num = 100
cv_split_count = 10
gss = GroupShuffleSplit(n_splits=cv_split_count, train_size=.9)
cv_metric = 'r2'

pkl_list = []

for split_order in [[0,1],[1,0]]:
    if len(dat.region_names) < 2:
        dat_arrays = firing_long_scaled,[None]*len(firing_long_scaled)
        region_names = ['split0','split1']
        region_sizes = [firing_long_scaled[0].shape[1], 0]
        name_order = split_names

        #firing0 = firing_long_scaled
        #firing1 = [None]*len(firing_long_scaled)
        #region_names = ['split0','split1']
    else:
        dat_arrays = [\
                    [x[:,dat.region_units[0]] for x in firing_long_scaled],
                    [x[:,dat.region_units[1]] for x in firing_long_scaled]]
        region_names = dat.region_names
        region_sizes = [len(x) for x in dat.region_units]
        name_order = split_names[np.array(split_order)]

        #firing0 = [x[:,dat.region_units[0]] for x in firing_long_scaled]
        #firing1 = [x[:,dat.region_units[1]] for x in firing_long_scaled]

    for this_dat, this_name in zip(dat_arrays,name_order):
        globals()[this_name] = this_dat
    fin_region_names = np.array(region_names)[np.array(split_order)]
    fin_region_sizes = np.array(region_sizes)[np.array(split_order)]
    # For ease with naming
    fin_names_list = [x for x in fin_region_names]
    fin_plot_name = "_".join(fin_names_list)


    ########################################
    # WITHOUT cross-validation
    ########################################
    no_cv_all_percentiles = []

    for this_firing0, this_firing1 in tqdm(zip(firing0,firing1)):
        this_func = lambda x : run_regression(this_firing0, this_firing1,
                        trial_labels = trial_labels,
                        cv_splitter = gss, scoring_metric = cv_metric, 
                        cv_bool = False)

        no_cv_outs = parallelize(this_func, range(split_repeat_num))
        no_cv_actual_data_percentile, no_cv_actual_score, no_cv_shuffled_scores = \
                                    list(zip(*no_cv_outs))
        no_cv_actual_data_percentile = np.concatenate(no_cv_actual_data_percentile)
        no_cv_all_percentiles.append(no_cv_actual_data_percentile)

    fig,ax = plt.subplots(len(no_cv_all_percentiles),1, 
            sharex=True, figsize=(5,10))
    for this_dat, this_ax in zip(no_cv_all_percentiles, ax.flatten()):
        vals,bins,patches = \
                this_ax.hist(this_dat, bins = np.linspace(0,100,21))
        patches[-1].set_fc('red')
        post_thresh = np.round(np.mean(this_dat > 95)*100,1)
        this_ax.set_title(f'{post_thresh}% > 95th percentile')
    plt.suptitle(f'{dat.hdf5_name} \n no_cv_percentiles \n'\
            f'X : {fin_region_names[0]} ({fin_region_sizes[0]}), '\
            f'y : {fin_region_names[1]} ({fin_region_sizes[1]})')
    fig.savefig(os.path.join(fin_plot_dir,fin_plot_name + '_no_cv_score')) 
    #plt.show()

    pkl_list.append(\
            {'type' :'no_cv',
            'percentiles' : no_cv_all_percentiles, 
            'region_order' : fin_region_names, 
            'region_sizes' : fin_region_sizes})

#    ########################################
#    # WITH cross-validation
#    ########################################
#    cv_all_percentiles = []
#
#    for this_firing0, this_firing1 in tqdm(zip(firing0,firing1)):
#
#        this_func = lambda x : run_regression(this_firing0, this_firing1,
#                        trial_labels = trial_labels,
#                        cv_splitter = gss, scoring_metric = cv_metric, 
#                        cv_bool = True)
#
#        cv_outs = parallelize(this_func, range(split_repeat_num))
#        cv_actual_data_percentile, cv_actual_score, cv_shuffled_scores = \
#                                    list(zip(*cv_outs))
#        cv_actual_data_percentile = np.concatenate(cv_actual_data_percentile)
#        cv_all_percentiles.append(cv_actual_data_percentile)
#
#    fig,ax = plt.subplots(len(cv_all_percentiles),1, sharex=True, figsize=(5,10))
#    for this_dat, this_ax in zip(cv_all_percentiles, ax.flatten()):
#        vals,bins,patches = \
#                this_ax.hist(this_dat, bins = np.linspace(0,100,21))
#        patches[-1].set_fc('red')
#        post_thresh = np.round(np.mean(this_dat > 95)*100,1)
#        this_ax.set_title(f'{post_thresh}% > 95th percentile')
#    plt.suptitle(f'{dat.hdf5_name} \n cv_percentiles \n'\
#            f'X : {fin_region_names[0]} ({fin_region_sizes[0]}), '\
#            f'y : {fin_region_names[1]} ({fin_region_sizes[1]})')
#    fig.savefig(os.path.join(fin_plot_dir,fin_plot_name + '_cv_score')) 
#    #plt.show()
#
#    pkl_list.append(\
#            {'type' :'cv',
#            'percentiles' : cv_all_percentiles, 
#            'region_order' : fin_region_names, 
#            'region_sizes' : fin_region_sizes})
#
#with open(fin_save_path, 'wb') as buff:
#    pickle.dump(pkl_list, buff)
