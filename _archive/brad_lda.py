import os
import sys
import numpy as np
from tqdm import tqdm
import pylab as plt
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, LeavePOut

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

data_dir = '/media/fastdata/'

dat = ephys_data(data_dir)
dat.get_spikes()

dat.firing_rate_params = dat.default_firing_params
#dat.firing_rate_params['type'] = 'baks'
#dat.firing_rate_params['baks_resolution'] = 1e-2
#dat.firing_rate_params['baks_dt'] = 1e-3
dat.get_firing_rates()

#visualize.firing_overview(dat.all_normalized_firing)
#plt.show()

## Calculate LDA
# Disregard timing of trial, just work with order

time_lims = np.array([2500,5000]) 
# Start earlier to accomodate smoothing
time_inds = time_lims//dat.firing_rate_params['step_size']

########################################
# dat.firing_array : size (tastes x neurons x trials x time)
# time = 271 bins

firing_array = dat.firing_array.swapaxes(1,0)
firing_array = firing_array[...,time_inds[0]:time_inds[1]]

# Features : All neurons and all timepoints

# Scale firing of each neuron
scaled_firing_array = np.empty(firing_array.shape)
for num, nrn in enumerate(firing_array):
    scaled_firing_array[num] = (nrn - np.nanmean(nrn,axis=None))/np.std(nrn,axis=None)

scaled_firing_array = np.moveaxis(scaled_firing_array, 0,2)

long_firing_array = np.reshape(scaled_firing_array,
                    (*scaled_firing_array.shape[:2],-1))
long_firing_array = np.reshape(long_firing_array, (-1, long_firing_array.shape[-1]))

#n_components = 10
#pca_obj = pca(n_components=n_components).fit(long_firing_array)
#print(np.sum(pca_obj.explained_variance_ratio_))
#pca_firing_array = pca_obj.transform(long_firing_array)
pca_firing_array = long_firing_array.copy()

trial_inds = np.array(list(np.arange(firing_array.shape[2]))*firing_array.shape[1])
taste_labels = np.repeat(np.arange(firing_array.shape[1]),firing_array.shape[2])

trial_window_len = 5 # Rolling window of # trials
trial_step = 1
window_num = int((firing_array.shape[2]-trial_window_len)/trial_step) + 1
inds_gen = np.array([np.arange(firing_array.shape[2])[i:i+trial_window_len] \
        for i in np.arange(window_num)])
get_array_inds = lambda x: np.sort(np.concatenate(\
        [np.where(trial_inds == this_ind)[0] for this_ind in x]))

#this_window = 0
#this_trial_inds = list(inds_gen)[this_window]

score_list = []
for this_trial_inds in tqdm(inds_gen):
    this_array_inds = get_array_inds(this_trial_inds)
    this_labels = taste_labels[this_array_inds] 
    this_data = pca_firing_array[this_array_inds]

    clf = lda()
    loo = LeaveOneOut()
    scores = cross_val_score(clf, this_data, this_labels, cv = loo, n_jobs = -1)
    #lpo = LeavePOut(p=4)
    #scores = cross_val_score(clf, this_data, this_labels, cv = lpo, n_jobs = -1)

    score_list.append(scores)

score_array = np.array(score_list)
mean_score = np.mean(score_array,axis=-1)
hist_bins = np.linspace(0,1,len(np.unique(score_array[0]))+1)
score_hist_list = np.array([np.histogram(x,bins=hist_bins,density=True)[0] for x in score_array])

#plt.pcolormesh(np.arange(score_hist_list.shape[0]), 
#        np.linspace(0,1, score_hist_list.shape[1]+1), score_hist_list.T)
#plt.plot(np.mean(score_array,axis=-1), color='red', linewidth=2)
#plt.ylabel('Cross-validated accuracy')
#plt.xlabel('Window Number')
##plt.colorbar()
#plt.show()
