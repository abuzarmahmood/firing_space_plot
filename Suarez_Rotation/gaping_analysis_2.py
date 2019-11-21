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

sys.path.append('/media/bigdata/firing_space_plot/_old')
from ephys_data import ephys_data

import glob
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.decomposition import PCA as pca
from mpl_toolkits.mplot3d.axes3d import Axes3D


# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

def dat_imshow(x):
    plt.imshow(x,interpolation='nearest',aspect='auto')

## Load data
#dir_list = ['/media/bigdata/Abuzar_Data/run_this_file']
dir_list = ['/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


file_num = 0
hf5 = tables.open_file(file_list[file_num])

gape_array = hf5.root.ancillary_analysis.gapes[:]
ltp_array = hf5.root.ancillary_analysis.ltps[:]

gape_array_long = gape_array.reshape(\
        (gape_array.shape[0],np.prod(gape_array.shape[1:3]),gape_array.shape[-1]))
#gape_array_long = gape_array_long[:,:,2000:4500]

gape_array_really_long = gape_array.reshape(\
        (np.prod(gape_array.shape[:3]),gape_array.shape[3]))

ltp_array_long = ltp_array.reshape(\
        (ltp_array.shape[0],np.prod(ltp_array.shape[1:3]),ltp_array.shape[-1]))
ltp_array_long = ltp_array_long[:,:,2000:4500]

fig, ax = plt.subplots(2,2)
plt.sca(ax[0,0]);dat_imshow(gape_array_long[0,:,:])
plt.sca(ax[0,1]);dat_imshow(gape_array_long[1,:,:])
plt.sca(ax[1,0]);dat_imshow(ltp_array_long[0,:,:])
plt.sca(ax[1,1]);dat_imshow(ltp_array_long[1,:,:])
plt.show()

# Find bout length
this_array = gape_array_long[0,:,:]

def calc_bout_duration(gape_array):
    gape_array[gape_array < 0.5] = 0
    gape_array[gape_array>0] = 1
    # Setting first and last index to 0 to ensure even numbers of markers
    gape_array[:,0] = 0
    gape_array[:,-1] = 0
    marker_array = np.where(np.abs(np.diff(gape_array,axis=-1)))
    marker_list = [marker_array[1][marker_array[0]==trial] \
            for trial in range(gape_array.shape[0])]
    bout_duration = [np.diff(np.split(trial,len(trial)//2)).flatten() \
            for trial in marker_list]
    return bout_duration

taste_bout_durations = \
        [np.concatenate(calc_bout_duration(taste)) for taste in gape_array[0]]

gape_array_pca = pca(n_components = 10).fit_transform(gape_array_long[0])

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(gape_array_pca[:,0],gape_array_pca[:,1],gape_array_pca[:,2],c=labels)
plt.show()

import umap

gape_array_umap = umap.UMAP(n_components = 2).fit_transform(gape_array_pca)
plt.scatter(gape_array_umap[:,0], gape_array_umap[:,1],c=labels);plt.show()


emg_bsa_results = np.asarray([x[:] for x in hf5.root.emg_BSA_results \
        if 'taste' in str(x)])
