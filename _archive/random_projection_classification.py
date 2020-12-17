
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   

import numpy as np
import tables
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import sys 
from tqdm import trange
import pandas as pd
import pingouin as pg
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing as mp
from itertools import groupby
import itertools

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.cluster import AgglomerativeClustering as hier_clust
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA as pca
from scipy.stats import zscore
#from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM12/AM12_extracted/AM12_4Tastes_191106_085215')
dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))
dat.extract_and_process()
dat.firing_overview(dat.all_normalized_firing);plt.show()

# ____                 _ ____            _ 
#|  _ \ __ _ _ __   __| |  _ \ _ __ ___ (_)
#| |_) / _` | '_ \ / _` | |_) | '__/ _ \| |
#|  _ < (_| | | | | (_| |  __/| | | (_) | |
#|_| \_\__,_|_| |_|\__,_|_|   |_|  \___// |
#                                     |__/ 

# Randomly project entire trial of single neurons to 2D space
time_range = range(80,160)
this_dat = dat.all_normalized_firing[:,:,time_range]
taste_labels = np.sort(list(range(dat.normalized_firing.shape[0]))*dat.normalized_firing.shape[2])

this_nrn = this_dat[18]
dat.imshow(this_nrn);plt.show()
n_proj_dims = 30
random_mat = np.random.normal(size=(this_nrn.shape[1],n_proj_dims))
proj_nrn = np.matmul(this_nrn,random_mat)

# Plot output with colors as labels
dat.imshow(proj_nrn.T);plt.show()

# Perform PCA and then plot scatter
pca_dims = 3
all_dim_combs = list(itertools.combinations(range(pca_dims),2))
this_pca = pca(n_components = pca_dims).fit_transform(proj_nrn)
fig,ax = plt.subplots(1,pca_dims)
for ax_num,dim_comb in enumerate(all_dim_combs):
    plt.sca(ax[ax_num])
    plt.scatter(this_pca[:,dim_comb[0]],this_pca[:,dim_comb[1]],
            c=taste_labels,cmap='jet')
    plt.title(dim_comb)
plt.show()

# Randomly project single trial of entire population 
this_dat_re = this_dat.swapaxes(0,1)
n_proj_dims = 10
random_mat = np.random.normal(size=tuple((n_proj_dims,*this_dat_re.shape[-2:])))
proj_nrn = np.array([np.matmul(this_trial,random_mat) for trial in this_dat_re])

# Plot output with colors as labels
dat.imshow(proj_nrn.T);plt.show()

# Perform PCA and then plot scatter
pca_dims = 3
all_dim_combs = list(itertools.combinations(range(pca_dims),2))
this_pca = pca(n_components = pca_dims).fit_transform(proj_nrn)
fig,ax = plt.subplots(1,pca_dims)
for ax_num,dim_comb in enumerate(all_dim_combs):
    plt.sca(ax[ax_num])
    plt.scatter(this_pca[:,dim_comb[0]],this_pca[:,dim_comb[1]],
            c=taste_labels,cmap='jet')
    plt.title(dim_comb)
plt.show()

