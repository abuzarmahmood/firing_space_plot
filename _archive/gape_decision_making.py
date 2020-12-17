# =============================================================================
# #### Import Modules #####
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tables

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data

from sklearn.decomposition import PCA as pca
from scipy.spatial import distance_matrix as dist_mat

import glob
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans as kmeans
from sklearn.decomposition import PCA

# HMM imports

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

from skimage import exposure

# =============================================================================
# #### Load Data ####
# =============================================================================

dir_list = ['/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)


file  = 3

this_dir = file_list[file].split(sep='/')[-2]
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'conv',700]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()
data.firing_overview('off')
data.firing_overview('on')

# =============================================================================
#  Load Gaping Data
# =============================================================================
file_list = os.listdir(data_dir)
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files
        
hf5 = tables.open_file(data_dir + '/' + hdf5_name, 'r+')

bsa_dat = hf5.root.ancillary_analysis.emg_BSA_results
bsa_cumsum = np.cumsum(bsa_dat,axis=3)

li_gape_trials = hf5.root.ancillary_analysis.gape_trials_Li[:]

# =============================================================================
# Sort trials for every taste by gaping
# =============================================================================
off_data = np.asarray(data.normal_off_firing)

# Remove neurons with missing trials
bad_nrns = []
good_nrns = np.asarray([x for x in range(off_data.shape[1]) if x not in bad_nrns])

off_data = off_data[:,good_nrns,:,:]

taste = 1
gape_firing = np.asarray([off_data[taste,:,trial,:] for trial in range(li_gape_trials.shape[-1]) if 
               li_gape_trials[0,taste,trial] == 1])
nongape_firing = np.asarray([off_data[taste,:,trial,:] for trial in range(li_gape_trials.shape[-1]) if 
               li_gape_trials[0,taste,trial] == 0])

# =============================================================================
# plt.figure()
# for trial in range(gape_firing.shape[0]):
#     plt.subplot(gape_firing.shape[0],1,trial+1)
#     data.imshow(gape_firing[trial,:,:])
#     
# plt.figure()
# for trial in range(nongape_firing.shape[0]):
#     plt.subplot(nongape_firing.shape[0],1,trial+1)
#     data.imshow(nongape_firing[trial,:,:])
# =============================================================================

plot_range = range(1000//25,4500//25)
plt.subplot(121)
data.imshow(np.mean(gape_firing,axis=0)[:,plot_range])
plt.title('Gape, n = %i' % gape_firing.shape[0])
plt.subplot(122)
data.imshow(np.mean(nongape_firing,axis=0)[:,plot_range])
plt.title('Non-Gape, n = %i' % nongape_firing.shape[0])