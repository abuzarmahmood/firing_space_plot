
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
import pingouin as pg
import seaborn as sns
from joblib import Parallel, delayed
import multiprocessing as mp
from itertools import groupby

sys.path.append('/media/bigdata/firing_space_plot/_old')
from ephys_data import ephys_data

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.cluster import AgglomerativeClustering as hier_clust
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA as pca
from scipy.stats import zscore
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# _   _      _                    _              ____ _           _   
#| | | | ___(_)_ __ __ _ _ __ ___| |__  _   _   / ___| |_   _ ___| |_ 
#| |_| |/ _ \ | '__/ _` | '__/ __| '_ \| | | | | |   | | | | / __| __|
#|  _  |  __/ | | | (_| | | | (__| | | | |_| | | |___| | |_| \__ \ |_ 
#|_| |_|\___|_|_|  \__,_|_|  \___|_| |_|\__, |  \____|_|\__,_|___/\__|
#                                       |___/                         


## Load data
dir_list = ['/media/bigdata/jian_you_data/des_ic',
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
    
    all_off_data.append(np.asarray(data.off_firing))

stim_delivery_ind = int(2000/25)
end_ind = int(4500/25)
# Get 2000 ms post-stim firing
all_poststim_data = [np.swapaxes(file,0,1) \
        for file in all_off_data]

# Sort to have top list be neurons
neuron_list = [file[nrn,:,:,:] for file in all_poststim_data \
        for nrn in range(file.shape[0])]
neuron_array = np.asarray(neuron_list)

neuron_array = neuron_array[:,:,:,stim_delivery_ind:end_ind]
neuron_array += np.random.random(neuron_array.shape)* 1e-9

#zscore_array = np.asarray([zscore(x,axis=None) for x in neuron_array])
#zscore_array_long= np.reshape(zscore_array,\
#        (np.prod(zscore_array.shape[:2]),zscore_array.shape[2],\
#        zscore_array.shape[3]))

mean_taste_array = np.mean(neuron_array, axis=2)
mean_long_array = np.reshape(mean_taste_array,\
        (np.prod(mean_taste_array.shape[:2]),mean_taste_array.shape[2]))

mean_zscore_array = zscore(mean_long_array,axis=-1)
data.imshow(mean_zscore_array);plt.show()

#input_dat = pca(n_components = 10).fit_transform(mean_taste_array[:,taste,:])
input_dat = np.split(mean_zscore_array, indices_or_sections = 4, axis=-1)
mean_bin_array = np.array([np.mean(x,axis=-1) for x in\
    input_dat]).swapaxes(0,1)

#dist_mat = pdist(input_dat)
linkage_mat = linkage(mean_bin_array, method = 'centroid')
#linkage_mat = linkage(mean_zscore_array, method = 'centroid')

model = dendrogram(linkage_mat)
plt.show()

sorted_order = np.asarray([np.int(x) for x in model['ivl']])

data.imshow(mean_zscore_array[sorted_order,:]);plt.show()

# =============================================
# Control analysis to see how easy it is to pull out clusters from random data
# ============================================
# Noise is correlated normal random (therefore brownian noise)
rand_long_array = np.cumsum(np.random.normal(size = mean_long_array.shape),axis=-1)
mean_zscore_array = zscore(rand_long_array,axis=-1)
data.imshow(mean_zscore_array);plt.show()

#input_dat = pca(n_components = 10).fit_transform(mean_taste_array[:,taste,:])
input_dat = np.split(mean_zscore_array, indices_or_sections = 4, axis=-1)
mean_bin_array = np.array([np.mean(x,axis=-1) for x in\
    input_dat]).swapaxes(0,1)

#dist_mat = pdist(input_dat)
linkage_mat = linkage(mean_bin_array, method = 'centroid')
#linkage_mat = linkage(mean_zscore_array, method = 'centroid')

model = dendrogram(linkage_mat)
plt.show()

sorted_order = np.asarray([np.int(x) for x in model['ivl']])
data.imshow(mean_zscore_array[sorted_order,:]);plt.show()

# _____         _             _ _     _   
#|_   _|_ _ ___| |_ ___    __| (_)___| |_ 
#  | |/ _` / __| __/ _ \  / _` | / __| __|
#  | | (_| \__ \ ||  __/ | (_| | \__ \ |_ 
#  |_|\__,_|___/\__\___|  \__,_|_|___/\__|

# Cluster neurons by time-periods of taste discriminability
labels = \
np.sort(np.array(list(range(neuron_array.shape[1]))*neuron_array.shape[2]))
accuracy_array = np.empty((neuron_array.shape[0],neuron_array.shape[-1]))
for nrn in tqdm(range(neuron_array.shape[0])):
    for time_bin in range(neuron_array.shape[-1]):
        accuracy_array[nrn,time_bin] = \
                LDA().fit(neuron_array[nrn,:,:,time_bin].flatten().reshape(-1,1),\
                labels.reshape(-1,1)).\
                score(neuron_array[nrn,:,:,time_bin].flatten().reshape(-1,1),labels)
                                         
zscore_accuracy_array = zscore(accuracy_array,axis=-1)
linkage_mat = linkage(zscore_accuracy_array, method = 'average')
model = dendrogram(linkage_mat)
plt.show()
sorted_order = np.asarray([np.int(x) for x in model['ivl']])
data.imshow(zscore_accuracy_array[sorted_order,:]);plt.colorbar();plt.show()

#__     __         _                      
#\ \   / /_ _ _ __(_) __ _ _ __   ___ ___ 
# \ \ / / _` | '__| |/ _` | '_ \ / __/ _ \
#  \ V / (_| | |  | | (_| | | | | (_|  __/
#   \_/ \__,_|_|  |_|\__,_|_| |_|\___\___|
#                                         

neuron_array_long = np.reshape(neuron_array,
        (neuron_array.shape[0],np.prod(neuron_array.shape[1:3]),
            neuron_array.shape[3]))

var_array = np.var(neuron_array_long,axis=1)
zscore_var_array = zscore(var_array,axis=-1)

linkage_mat = linkage(zscore_var_array, method = 'average')
model = dendrogram(linkage_mat)
plt.show()
sorted_order = np.asarray([np.int(x) for x in model['ivl']])
data.imshow(zscore_var_array[sorted_order,:]);plt.colorbar();plt.show()

data.imshow(zscore_var_array[np.argsort(np.argmax(zscore_var_array,axis=-1)),:])
plt.show()



# ____  ____    _   _ _     _   
#|___ \|  _ \  | | | (_)___| |_ 
#  __) | | | | | |_| | / __| __|
# / __/| |_| | |  _  | \__ \ |_ 
#|_____|____/  |_| |_|_|___/\__|
#                               

bin_edges = np.quantile(mean_taste_array.flatten(),np.linspace(0,1)) 
hist_array = np.asarray([np.histogram(np.abs(mean_taste_array[:,time_bin]), \
            bins = bin_edges)[0] for time_bin in range(hist_array.shape[1])])
data.imshow(mean_taste_array[np.argsort(np.argmax(mean_taste_array,axis=-1)),:])
plt.show()
data.imshow(hist_array);plt.show()
