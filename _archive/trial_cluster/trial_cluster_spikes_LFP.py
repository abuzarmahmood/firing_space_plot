######################### Import dat ish #########################
import tables
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp
import tensortools as tt
from sklearn.cluster import KMeans as kmeans


from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca

from scipy.stats import mannwhitneyu as mnu

from skimage import exposure
from scipy import signal
from scipy.signal import butter
from scipy.signal import filtfilt
import glob

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y

#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#

dir_list = ['/media/bigdata/brads_data/BS28_4Tastes_180801_112138']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    
file = 0

# Get firing rate data

data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'baks',269]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()

firing_array = np.asarray(data.normal_off_firing)
spikes_array = np.asarray(data.off_spikes)

# Get LFP data
hf5 = tables.open_file(file_list[file])
lfp_array = np.asarray([x[:] for x in hf5.root.Parsed_LFP._f_list_nodes()])
hf5.close()

# Mean lfp across all channels (no justification :p)
lfp_array = np.mean(lfp_array,axis=1)


#   _____ _           _              _______   _       _     
#  / ____| |         | |            |__   __| (_)     | |    
# | |    | |_   _ ___| |_ ___ _ __     | |_ __ _  __ _| |___ 
# | |    | | | | / __| __/ _ \ '__|    | | '__| |/ _` | / __|
# | |____| | |_| \__ \ ||  __/ |       | | |  | | (_| | \__ \
#  \_____|_|\__,_|___/\__\___|_|       |_|_|  |_|\__,_|_|___/
#

# Clustering using tensor rank reduction
firing_inds = range(80,160)
lfp_inds = range(2000,4000)

taste = 0
this_firing_data = np.rollaxis(firing_array[taste,:,:,firing_inds],0,3) # nrn x trial x time
this_lfp_data = np.rollaxis(lfp_array[taste,:,lfp_inds],0,2)
this_spike_data = np.rollaxis(spikes_array[taste,:,:,lfp_inds],0,3)

rank = 7

# Fit CP tensor decomposition (two times).
U = tt.cp_als(this_firing_data.swapaxes(1,2), rank=rank, verbose=False)

# Compare the low-dimensional factors from the two fits.
fig, _, _ = tt.plot_factors(U.factors)

## We should be able to see differences in tastes by using distance matrices on trial factors
trial_factors = U.factors.factors[-1]
trial_distances = exposure.equalize_hist(dist_mat(trial_factors,trial_factors))
plt.figure();plt.imshow(trial_distances)

# Kmean clustering on trial factors
n_components = 4
clf = kmeans(n_clusters = n_components, n_init = 500)
this_groups = clf.fit_predict(trial_distances)


trial_order = np.argsort(this_groups)

# Pull out and cluster distance matrices
clust_post_dist = trial_distances[trial_order,:]
clust_post_dist = clust_post_dist[:,trial_order]


#  __  __       _          _____  _       _       
# |  \/  |     | |        |  __ \| |     | |      
# | \  / | __ _| | _____  | |__) | | ___ | |_ ___ 
# | |\/| |/ _` | |/ / _ \ |  ___/| |/ _ \| __/ __|
# | |  | | (_| |   <  __/ | |    | | (_) | |_\__ \
# |_|  |_|\__,_|_|\_\___| |_|    |_|\___/ \__|___/
#                                                

## Distance matrix cluster plots
plt.figure()
plt.subplot(121);plt.imshow(exposure.equalize_hist(trial_distances));plt.title('Un Stim')
plt.subplot(122);plt.imshow(exposure.equalize_hist(clust_post_dist));plt.title('Clust Stim')
line_num = np.where(np.diff(np.sort(this_groups)))[0]
for point in line_num:
    plt.axhline(point+0.5,color = 'red')
    plt.axvline(point+0.5,color = 'red')

## Firing rate plots (image)
clust_list = []
for cluster in range(n_components):
    this_cluster = this_firing_data[:,this_groups == cluster,:]
    clust_list.append(this_cluster)

max_vals = []
clust_means = []   
for cluster in range(len(clust_list)):
    dat = np.mean(clust_list[cluster],axis=1)
    clust_means.append(dat)
    max_vals.append(np.max(dat))

## Firing rate Plots (lines)
plt.figure()
for cluster in range(len(clust_list)):
    plt.subplot(n_components,1,cluster+1)
    dat = np.mean(clust_list[cluster],axis=1)
    plt.imshow(dat,
               interpolation='nearest',aspect='auto',vmin=0,vmax=max(max_vals))
    plt.title('n = %i' % clust_list[cluster].shape[1])
    plt.colorbar()

plt.figure()
for nrn in range(this_firing_data.shape[0]):
    plt.subplot(this_firing_data.shape[0],1,nrn+1)
    for cluster in range(len(clust_list)):
        dat = clust_list[cluster][nrn,:,:]
        y = np.mean(dat,axis=0)
        x = range(dat.shape[1])
        yerr = np.std(dat,axis=0)/np.sqrt(dat.shape[0])
        plt.errorbar(x = x,y = y,yerr = yerr)
        
# Mean LFP plots
# Cluster LFP by using same trials as firing rate
lfp_clust_list = []
for cluster in range(n_components):
    this_cluster = this_lfp_data[this_groups == cluster,:]
    lfp_clust_list.append(this_cluster)
    
fs = 1000
# Average spectrogram across trials
max_freq = 30
lfp_spect_list = []
for cluster in lfp_clust_list:
    this_clust_spec = []
    for trial in cluster:
        f, t, Sxx = signal.spectrogram(trial, fs,nperseg = 500, noverlap = 450,scaling='spectrum')
        f,Sxx = f[f <= max_freq],Sxx[f <= max_freq,:]
        this_clust_spec.append(Sxx)
    lfp_spect_list.append(np.asarray(this_clust_spec))
    

plt.figure()
for cluster in range(len(lfp_spect_list)):
    plt.subplot(n_components,1,cluster+1)
    dat = np.mean(lfp_spect_list[cluster],axis=0)
    log_dat = np.log10(10+dat)
    #plt.pcolormesh(t, f, np.log10(dat),cmap='jet')
    plt.contourf(t, f, log_dat,cmap='jet',levels=np.linspace(np.min(log_dat),np.max(log_dat),30))
    #plt.ylim([0,100])
    #plt.plot(dat)
    plt.title('n = %i' % clust_list[cluster].shape[1])
    
# =============================================================================
# =============================================================================
# Make sure single neuron firing is still appropriately clustered
## Clustered raster plots
nrn = 9
plt.figure();dot_raster(this_spike_data[nrn,trial_order,:]);plt.title('taste %i' % taste)
line_num = np.append(-0.5,np.where(np.diff(np.sort(this_groups)))[0])
for point in range(len(line_num)):
    plt.axhline(line_num[point]+0.5,color = 'red')
    plt.text(0,line_num[point]+0.5,point,fontsize = 20,color = 'r')