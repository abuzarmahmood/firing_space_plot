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
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel,delayed
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore, percentileofscore
from sklearn.model_selection import cross_val_predict, cross_val_score
from statsmodels.tsa.stattools import acf
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

data_dir = '/media/bigdata/Abuzar_Data/AM34/AM34_4Tastes_201218_131632'
dat = ephys_data(data_dir)
dat.firing_rate_params = dict(zip(\
    ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
    ('conv',25,250,1,25e-3,1e-3)))
dat.extract_and_process()

# Use half of neurons to predict activity in other half
time_lims = [2000,4000]
step_size = dat.firing_rate_params['baks_resolution']
dt = dat.firing_rate_params['baks_dt']
time_inds = np.vectorize(np.int)(np.array(time_lims)/(step_size/dt))
taste = 0
firing_array = dat.firing_array[taste][...,time_inds[0]:time_inds[1]]

split_repeat_num = 5

## No crossvalidation

all_percentiles = []
all_pls_percentiles = []
for this_split in trange(split_repeat_num):
    pop1_neurons = np.random.choice(np.arange(firing_array.shape[0]),
                                firing_array.shape[0]*1//2, replace = False)
    pop2_neurons = np.array([x for x in np.arange(firing_array.shape[0]) \
                            if x not in pop1_neurons])
    pop1_firing = firing_array[pop1_neurons] 
    pop2_firing = firing_array[pop2_neurons] 


    ##################################################
    ## Perform regression for all trials collectively
    ##################################################
    # Actual data
    pop1_firing_long = pop1_firing.reshape((pop1_firing.shape[0],-1))
    pop2_firing_long = pop2_firing.reshape((pop2_firing.shape[0],-1))

    pop1_long_scaled = StandardScaler().fit_transform(pop1_firing_long.T)
    pop2_long_scaled = StandardScaler().fit_transform(pop2_firing_long.T)

    X = pop1_long_scaled 
    y = pop2_long_scaled 
    reg = LinearRegression().fit(X, y)
    actual_score = r2_score(y,reg.predict(X))

    ## Use PLSR to avoid overfitting
    pls2 = PLSRegression(n_components=2)
    pls2.fit(X, y)

    actual_pls_score = r2_score(y,pls2.predict(X))

    # Shuffled trials
    repeats = 100
    shuffled_scores = []
    shuffled_pls_scores = []
    for repeat in range(repeats):
        pop1_shuffled = \
                pop1_firing[:,np.random.permutation(np.arange(pop1_firing.shape[1]))] 
        pop2_shuffled = \
                pop2_firing[:,np.random.permutation(np.arange(pop2_firing.shape[1]))] 

        pop1_shuffled_long = pop1_shuffled.reshape((pop1_shuffled.shape[0],-1))
        pop2_shuffled_long = pop2_shuffled.reshape((pop2_shuffled.shape[0],-1))

        pop1_shuffled_long_scaled = \
                StandardScaler().fit_transform(pop1_shuffled_long.T)
        pop2_shuffled_long_scaled = \
                StandardScaler().fit_transform(pop2_shuffled_long.T)

        X = pop1_shuffled_long_scaled 
        y = pop2_shuffled_long_scaled 

        reg = LinearRegression().fit(X, y)
        shuffled_scores.append(r2_score(y,reg.predict(X)))

        pls2 = PLSRegression(n_components=2).fit(X, y)
        shuffled_pls_scores.append(r2_score(y,pls2.predict(X)))

    actual_data_percentile = percentileofscore(shuffled_scores, actual_score)
    actual_pls_data_percentile = \
            percentileofscore(shuffled_pls_scores, actual_pls_score)
    all_percentiles.append(actual_data_percentile)
    all_pls_percentiles.append(actual_pls_data_percentile)

plt.hist(shuffled_scores,bins=50)
plt.axvline(actual_score)
plt.title(f'Percentile : {actual_data_percentile}')
plt.show()

plt.hist(shuffled_pls_scores,bins=50)
plt.axvline(actual_pls_score)
plt.title(f'Percentile : {actual_pls_data_percentile}')
plt.show()

##################################################
## With cross-validation
##################################################
cv_split = 5
all_percentiles = []
all_pls_percentiles = []
for this_split in trange(split_repeat_num):
    ##################################################

    pop1_neurons = np.random.choice(np.arange(firing_array.shape[0]),
                                firing_array.shape[0]*1//2, replace = False)
    pop2_neurons = np.array([x for x in np.arange(firing_array.shape[0]) \
                            if x not in pop1_neurons])
    pop1_firing = firing_array[pop1_neurons] 
    pop2_firing = firing_array[pop2_neurons] 


    ##################################################
    ## Perform regression for all trials collectively
    ##################################################
    # Actual data
    pop1_firing_long = pop1_firing.reshape((pop1_firing.shape[0],-1))
    pop2_firing_long = pop2_firing.reshape((pop2_firing.shape[0],-1))

    pop1_long_scaled = StandardScaler().fit_transform(pop1_firing_long.T)
    pop2_long_scaled = StandardScaler().fit_transform(pop2_firing_long.T)

    permutation_inds = np.random.permutation(np.arange(pop1_long_scaled.shape[0]))
    # To avoid cross-validation being affected by position/trial
    X = pop1_long_scaled[permutation_inds] 
    y = pop2_long_scaled[permutation_inds] 
    lm = LinearRegression()
    actual_score = cross_val_score(lm, X, y, cv=cv_split, scoring = 'r2')


    ## Use PLSR to avoid overfitting
    pls2 = PLSRegression(n_components=2)
    actual_pls_score = cross_val_score(pls2, X, y, cv=cv_split, scoring = 'r2')

    # Shuffled trials
    repeats = 100
    shuffled_scores = []
    shuffled_pls_scores = []
    #full_shuffled_scores = []
    for repeat in range(repeats):
        pop1_shuffled = \
                pop1_firing[:,np.random.permutation(np.arange(pop1_firing.shape[1]))] 
        pop2_shuffled = \
                pop2_firing[:,np.random.permutation(np.arange(pop2_firing.shape[1]))] 

        pop1_shuffled_long = pop1_shuffled.reshape((pop1_shuffled.shape[0],-1))
        pop2_shuffled_long = pop2_shuffled.reshape((pop2_shuffled.shape[0],-1))

        pop1_shuffled_long_scaled = \
                StandardScaler().fit_transform(pop1_shuffled_long.T)
        pop2_shuffled_long_scaled = \
                StandardScaler().fit_transform(pop2_shuffled_long.T)

        # Can probably use same permutation inds as above
        X = pop1_shuffled_long_scaled[permutation_inds] 
        y = pop2_shuffled_long_scaled[permutation_inds] 

        lm = LinearRegression()
        shuffled_scores.append(cross_val_score(lm, X, y, cv=10, scoring = 'r2'))
        #full_shuffled_scores.append(\
        #        cross_val_score(lm, X, np.random.permutation(y), cv=10, scoring = 'r2'))

        pls2 = PLSRegression(n_components=2)
        shuffled_pls_scores.append(cross_val_score(pls2, X, y, cv=10, scoring = 'r2'))

    mean_actual_score = np.mean(actual_score)
    mean_shuffled_scores = np.mean(np.array(shuffled_scores),axis=1)
    #mean_full_shuffled_scores = np.mean(np.array(full_shuffled_scores),axis=1)
    actual_data_percentile = percentileofscore(mean_shuffled_scores, mean_actual_score)

    mean_pls_actual_score = np.mean(actual_pls_score)
    mean_pls_shuffled_scores = np.mean(np.array(shuffled_pls_scores),axis=1)
    actual_pls_data_percentile = \
            percentileofscore(mean_pls_shuffled_scores, mean_pls_actual_score)

    all_percentiles.append(actual_data_percentile)
    all_pls_percentiles.append(actual_pls_data_percentile)

plt.hist(mean_shuffled_scores,bins=50, label = 'Trial shuffled')
#plt.hist(mean_full_shuffled_scores,bins=50, label = 'Complete Random')
plt.axvline(mean_actual_score, label = 'Actual', linewidth = 2, color = 'red')
plt.legend()
plt.title(f'Percentile : {actual_data_percentile}')
plt.show()

plt.hist(mean_pls_shuffled_scores,bins=50)
plt.axvline(mean_pls_actual_score)
plt.title(f'Percentile : {actual_pls_data_percentile}')
plt.show()
