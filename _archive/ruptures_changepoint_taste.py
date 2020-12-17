######################### Import dat ish #########################
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

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure

import glob

import ruptures as rpt
from sklearn.decomposition import FastICA as ica

# =============================================================================
# =============================================================================
#dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
#dir_list = ['/media/bigdata/Jenn_Data/']
dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


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

# =============================================================================
# =============================================================================
# Changepoint detection on a per-trial basis

trial = 1

# Perform PCA to reduce dimensions
this_dat = X[:,:,trial]
pca_obj = pca(n_components = 6).fit(this_dat.T)
reduced_signal = pca_obj.transform(this_dat.T)

model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Window(width=10, model=model).fit(reduced_signal)

my_bkps = algo.predict(n_bkps=4)

data.imshow(this_dat)
for point in my_bkps:
    plt.vlines(point,-0.5,this_dat.shape[0]-0.5,'r')

# show results
rpt.show.display(reduced_signal, my_bkps, figsize=(5, 3))
plt.show()

# =============================================================================
# =============================================================================
#Changepoint detection on a per-neuron basis
nrn = 3
this_dat = X[nrn,:,:]

pca_obj = pca(n_components = 5).fit(this_dat)
reduced_signal = pca_obj.transform(this_dat)

model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
algo = rpt.Window(width=10, model=model).fit(reduced_signal)
my_bkps = algo.predict(n_bkps=3)

rpt.show.display(reduced_signal, my_bkps, figsize=(5, 3))
plt.show()

plt.figure()
data.imshow(this_dat.T)
for point in my_bkps:
    plt.vlines(point,-0.5,this_dat.T.shape[0]-0.5,'r')