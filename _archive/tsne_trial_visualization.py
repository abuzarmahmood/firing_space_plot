#####################import dat ish #########################
import os
import numpy as np
import matplotlib.pyplot as plt

import tensortools as tt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat
from sklearn.cluster import KMeans as kmeans
from sklearn.manifold import TSNE as tsne

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure

import glob

import ruptures as rpt
from sklearn.decomposition import FastICA as ica

# =============================================================================# =============================================================================
#dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']#dir_list = ['/media/bigdata/Jenn_Data/']
file_list = []
dir_list = ['/media/bigdata/jian_you_data/des_ic']
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

plot_dir = '/media/bigdata/pomegranate_hmm/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    
file  = 0

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                               [25,250,7000]))
data.get_data()
data.get_firing_rates()

all_spikes_array = np.asarray(data.off_spikes)
all_firing_array = np.asarray(data.normal_off_firing)

X = data.all_normal_off_firing.swapaxes(1,2)[:,80:160,:]

#taste = 0
#X = data.normal_off_firing[taste][:,:,80:160].swapaxes(-1,-2)

# Reduce dimensions of every timepoint using TSNE
X_long = X[:,:,0]
for trial in range(1,X.shape[2]):
    X_long = np.concatenate((X_long,X[:,:,trial]),axis=-1)
    
perm  = np.random.permutation(np.arange(X_long.shape[1]))
X_long_perm = X_long[:,perm] 
    
X_embedded = tsne(n_components = 2,perplexity = 35).fit_transform(X_long_perm.T)

colors = range(np.int(X.shape[1]))/np.max(range(np.int(X.shape[1])))
for trial_num in range(60):
    fig = plt.figure()
    plt.scatter(X_embedded[(trial_num+1)*np.arange(np.int(X.shape[1])),0],X_embedded[(trial_num+1)*np.arange(np.int(X.shape[1])),1],c=colors)
    plt.colorbar()
    plt.plot(X_embedded[(trial_num+1)*np.arange(np.int(X.shape[1])),0],X_embedded[(trial_num+1)*np.arange(np.int(X.shape[1])),1])
    plt.savefig(plot_dir + '/' + 'tsne_trial_%i.png' % trial_num)
    plt.close(fig)
    
colors = np.matlib.repmat(range(np.int(X.shape[1])),1,X.shape[2])[0,perm]
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=colors.flatten())

# =============================================================================
# fig,ax = plt.subplots()
# trial_labels = np.matlib.repmat(range(np.int(X.shape[1])),1,X.shape[-1])[0]
# ax.scatter(X_embedded[:,0],X_embedded[:,1],c=trial_labels);
# 
# colors = range(np.int(X.shape[1]))/np.max(range(np.int(X.shape[1])))
# for time in (trial_num+1)*np.arange(np.int(X.shape[1])):
#     ax.scatter(X_embedded[time,0],X_embedded[time,1],c='red')
# =============================================================================



# Fit CP tensor decomposition (two times).
rank = 5
repeats = 30
all_models = []
all_obj = []
for repeat in tqdm(range(repeats)):
    U = tt.cp_als(X, rank=rank, verbose=False)
    all_models.append(U)
    all_obj.append(U.obj)

U = all_models[np.argmin(all_obj)]

tt.plot_factors(U.factors)

## We should be able to see differences in tastes by using distance matrices on trial factors
trial_factors = U.factors.factors[-1]
trial_distances = dist_mat(trial_factors,trial_factors)
plt.figure();plt.imshow(exposure.equalize_hist(trial_distances))

## Cluster trials using tsne
trial_labels = np.sort([0,1,2,3]*15)
X_embedded = tsne(n_components = 2,perplexity = 40).fit_transform(trial_factors)
plt.figure();plt.scatter(X_embedded[:,0],X_embedded[:,1],c=trial_labels)
