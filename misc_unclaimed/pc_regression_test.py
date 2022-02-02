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
from scipy.stats import zscore

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

########################################
# Import Data
########################################

data_dir = '/media/bigdata/Abuzar_Data/AS18/AS18_4Tastes_200229_154608'
dat = \
    ephys_data(data_dir)
dat.firing_rate_params = dict(zip(\
    ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
    ('baks',1,250,1,1e-3,1e-3)))
dat.extract_and_process()

# Use half of neurons to predict activity in other half
time_lims = [1500,4000]
taste = 0
firing_array = dat.normalized_firing[taste,...,time_lims[0]:time_lims[1]]

pop1_neurons = np.random.choice(np.arange(firing_array.shape[0]),
                            firing_array.shape[0]*4//5, replace = False)
pop2_neurons = np.array([x for x in np.arange(firing_array.shape[0]) \
                        if x not in pop1_neurons])
pop1_firing = firing_array[pop1_neurons] 
pop2_firing = firing_array[pop2_neurons] 

pop1_firing_long = pop1_firing.reshape((pop1_firing.shape[0],-1)).T
pop2_firing_long = pop2_firing.reshape((pop2_firing.shape[0],-1)).T

pop1_long_standard = StandardScaler().fit_transform(pop1_firing_long)
pop2_long_standard = StandardScaler().fit_transform(pop2_firing_long)

pop1_standard_reshape = pop1_long_standard.reshape((*pop1_firing.shape[1:],-1))
pop2_standard_reshape = pop2_long_standard.reshape((*pop2_firing.shape[1:],-1))

# Convert firing to principle componenets


score_cal_list = []
score_cv_list = []
shuffle_score_cv_list = []
mse_cal_list = []
mse_cv_list = []

for selected_components in trange(1,pop1_long_standard.shape[-1]):

    pop1_pca_object = pca(n_components = selected_components).fit(pop1_long_standard)
    #pop2_pca_object = pca(n_components = selected_components).fit(pop2_long_standard)
    pop1_pca = pop1_pca_object.transform(pop1_long_standard)
    #pop2_pca = pop2_pca_object.transform(pop2_long_standard)

    pop1_pca_reshape = pop1_pca.reshape((*pop1_firing.shape[1:],-1))
    #pop2_pca_reshape = pop2_pca.reshape((*pop2_firing.shape[1:],-1))

    #visualize.firing_overview(np.moveaxis(pop1_pca_reshape,-1,0))
    #visualize.firing_overview(np.moveaxis(pop2_pca_reshape,-1,0))
    #plt.show()

    # Divide trials in train and test sets
    regr = linear_model.LinearRegression()
    regr.fit(pop1_pca, pop2_long_standard)

    # visualize prediction
    #trial = 1
    #y = regr.predict(pop1_pca_reshape[trial])
    ##inverse_y = pop2_pca_object.inverse_transform(y) 
    #min_val, max_val = np.min(y,axis=None),np.max(y,axis=None)
    #fig,ax = plt.subplots(2,1)
    #ax[0].imshow(y.T, aspect = 'auto', vmin = min_val, vmax = max_val)
    #ax[1].imshow(pop2_standard_reshape[trial].T, 
    #        aspect = 'auto', vmin = min_val, vmax = max_val)
    #plt.show()
    #fig,ax = visualize.gen_square_subplots(inverse_y.shape[-1])
    #for nrn_num, this_ax in enumerate(ax.flatten()):
    #    this_ax.plot(y[:,nrn_num])
    #    this_ax.plot(pop2_standard_reshape[trial,:,nrn_num])
    #plt.show()

    shuffle_count = 20

    # Calibration
    pop2_cal = regr.predict(pop1_pca)

    # Cross-validation
    pop2_cross_val = cross_val_predict(regr, pop1_pca, pop2_long_standard, cv = 10)

    this_shuffle_cross_val_list = []
    for repeat in range(shuffle_count):
        pop2_shuffle_cross_val = \
                cross_val_predict(regr, pop1_pca, 
                        np.random.permutation(pop2_long_standard), 
                        cv = 10)
        this_shuffle_cross_val_list.append(\
                        r2_score(pop2_long_standard, pop2_shuffle_cross_val))
    shuffle_score_cv_list.append(this_shuffle_cross_val_list)

    # Calculate scores for calibration and cross-validation
    score_cal_list.append(r2_score(pop2_long_standard, pop2_cal))
    score_cv_list.append(r2_score(pop2_long_standard, pop2_cross_val))

    # Calculate MSE for calibration and cross-validation
    mse_cal_list.append(mean_squared_error(pop2_long_standard, pop2_cal))
    mse_cv_list.append(mean_squared_error(pop2_long_standard, pop2_cross_val))

fig, ax = plt.subplots(2,1)
ax[0].plot(score_cal_list,'o', label = 'Calibration')
ax[0].plot(score_cv_list,'o', label = 'Cross-validation')
ax[0].boxplot(np.array(shuffle_score_cv_list).T)
ax[0].set_ylabel('R^2')
ax[0].set_xlabel('# of components')
ax[0].legend()
ax[1].plot(mse_cal_list,'o', label = 'Calibration')
ax[1].plot(mse_cv_list,'o', label = 'Cross-validation')
ax[1].set_ylabel('Mean Squared Error')
ax[1].set_xlabel('# of components')
ax[1].legend()
plt.show()
