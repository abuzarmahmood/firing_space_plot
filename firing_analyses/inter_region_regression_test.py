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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import zscore, percentileofscore
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

vector_percentile = lambda dist,vals: np.array(\
        list(map(lambda vals: percentileofscore(dist, vals), vals)))

###################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
###################################################

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

data_dir = '/media/bigdata/Abuzar_Data/AM34/AM34_4Tastes_201218_131632'
dat = ephys_data(data_dir)
dat.firing_rate_params = dict(zip(\
    ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
    ('baks',25,250,1,25e-3,1e-3)))
dat.extract_and_process()

# Use half of neurons to predict activity in other half
time_lims = [2000,4000]
step_size = dat.firing_rate_params['baks_resolution']
dt = dat.firing_rate_params['baks_dt']
time_inds = np.vectorize(np.int)(np.array(time_lims)/(step_size/dt))
taste = 0

firing_array = dat.firing_array[taste][...,time_inds[0]:time_inds[1]]
firing_array_long = firing_array.reshape((firing_array.shape[0],-1))
firing_long_scaled = StandardScaler().fit_transform(firing_array_long.T) 

trial_labels = np.repeat(np.arange(firing_array.shape[1]), firing_array.shape[2])

def trial_shuffle_gen(label_list):
    new_trial_order = np.random.permutation(np.unique(label_list))
    return np.concatenate([np.where(label_list == x)[0] for x in new_trial_order])

split_repeat_num = 10

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

all_percentiles = []
for this_split in trange(split_repeat_num):
    grp1,grp2 = np.array_split(\
            np.random.permutation(np.arange(firing_long_scaled.shape[1])),2)

    X,y = firing_long_scaled[:,grp1], firing_long_scaled[:,grp2]
    reg = LinearRegression().fit(X, y)
    actual_score = r2_score(y,reg.predict(X))

    # Shuffled trials
    repeats = 100
    shuffled_scores = []
    for repeat in range(repeats):

        X_sh = X[trial_shuffle_gen(trial_labels)] 
        y_sh = y[trial_shuffle_gen(trial_labels)]

        reg = LinearRegression().fit(X_sh, y_sh)
        shuffled_scores.append(r2_score(y_sh,reg.predict(X_sh)))

    actual_data_percentile = percentileofscore(shuffled_scores, actual_score)
    all_percentiles.append(actual_data_percentile)

plt.hist(shuffled_scores,bins=50)
plt.axvline(actual_score)
plt.title(f'Percentile : {actual_data_percentile}')
plt.show()

##################################################
## With cross-validation
##################################################
## K-Fold cross validation will hold out chunks in CONTINUOUS TIME
## There is no guarantee that the activity of the populations will be stable
## across the session. This means that the relationship between the populations
## could be different on the trained data and the validation data
## To mitigate this issue:
## 1) We could train and test on smaller chunks of data (e.g. 10 trial blocks)
## 2) Use GroupKFold or GroupShuffleSplit (with groups as trials) to
##      sample adequately from each time period
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

cv_split = 10
shuffle_repeats = 100
split_repeat_num = 30
metric = 'r2'
all_percentiles = []

gss = GroupShuffleSplit(n_splits=cv_split, train_size=.9)

#all_inds = []
#for train_idx, test_idx in gss.split(X, y, trial_labels):
#    all_inds.append(train_idx)
#ind_mat = np.zeros((cv_split,len(trial_labels)))
#for num,this_inds in enumerate(all_inds):
#    ind_mat[num,this_inds] = 1
#visualize.imshow(ind_mat)
#plt.show()

for this_split in trange(split_repeat_num):

    # Actual data
    grp1,grp2 = np.array_split(\
            np.random.permutation(np.arange(firing_long_scaled.shape[1])),2)

    X,y = firing_long_scaled[:,grp1], firing_long_scaled[:,grp2]

    lm = LinearRegression()
    cv_iter = gss.split(X, y, trial_labels)
    actual_score = cross_val_score(lm, X, y, cv=cv_iter, scoring = metric)

    # Shuffled trials
    shuffled_scores = []
    for repeat in range(shuffle_repeats):

        X_sh = X[trial_shuffle_gen(trial_labels)] 
        y_sh = y[trial_shuffle_gen(trial_labels)]

        lm = LinearRegression()
        cv_iter = gss.split(X_sh, y_sh, trial_labels)
        shuffled_scores.append(cross_val_score(lm, X_sh, y_sh, cv=cv_iter, 
            scoring = metric))

    actual_data_percentile = vector_percentile(np.concatenate(shuffled_scores), 
                                            actual_score)

    all_percentiles.append(actual_data_percentile)

vals,bins,patches = \
        plt.hist(np.concatenate(all_percentiles), bins = np.linspace(0,100,21))
patches[-1].set_fc('red')
plt.show()

