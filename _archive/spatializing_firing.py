"""
Parametrize trajectory taken by population in 
response to a taste
1) See how strongly mean spatial trajectory reflects
    all trials
    (since according to HMM, the content of the states is 
    similar, the temporal features are different)
2) Can number of states be identified using a space-time plot
    (path along the spatial trajectory vs time -- there should
    be overlap at the mean time value for different states)
3) Can state transitions be identified using changes in 
    trajectory velocity
4) Can key points in firing-rate space be identified using
    a running velocity orthogonality measure
"""

## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import zscore
from scipy.stats import median_absolute_deviation as MAD
from sklearn.preprocessing import StandardScaler as scaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize
from BAKS import BAKS

def _calc_baks_rate(resolution, dt, spike_array):
    t = np.linspace(0,spike_array.shape[-1]*dt, 
            spike_array.shape[-1]*dt/resolution)
    array_inds = list(np.ndindex((spike_array.shape[:-1])))
    spike_times = [np.where(spike_array[this_inds])[0]*dt \
            for this_inds in array_inds]
    # Calculate firing rates in parallel
    firing_rates = [BAKS(this_spike_times,t) \
            for this_spike_times in tqdm(spike_times)]
    #firing_rates = Parallel(n_jobs = cpu_count()-2)\
    #        (delayed(BAKS)(this_spike_times,t) \
    #        for this_spike_times in spike_times)
    # Put back into array
    firing_rate_array = np.zeros((*spike_array.shape[:-1],len(t)))
    for this_inds, this_firing in zip(array_inds, firing_rates):
        firing_rate_array[this_inds] = this_firing
    return firing_rate_array

dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AS18/AS18_4Tastes_200229_154608')
dat.firing_rate_params = dat.default_firing_params
dat.firing_rate_params['type'] = 'baks'
dat.firing_rate_params['baks_resolution'] = 1e-2
dat.firing_rate_params['baks_dt'] = 1e-3
#dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
#                                    (1,250,1)))

dat.extract_and_process()
dat.separate_laser_data()

#baks_firing = np.array([_calc_baks_rate(1e-2, 1e-3, spike_array)\
#        for spike_array in dat.spikes]).swapaxes(1,2)
#all_baks_firing = baks_firing.swapaxes(0,1).reshape(
#        (baks_firing.shape[1],-1,baks_firing.shape[-1]))

baks_firing = dat.firing_array
all_baks_firing = dat.all_normalized_firing

#visualize.firing_overview(dat.all_normalized_firing)
visualize.firing_overview(all_baks_firing)
plt.show()

# Extract data for single taste and perform PCA
time_range = (0,700)
#this_taste = dat.normalized_firing[0][...,time_range[0]:time_range[1]]
this_taste = baks_firing[0][...,time_range[0]:time_range[1]]
this_taste_long = np.reshape(this_taste, (this_taste.shape[0],-1))
scaler_object = scaler().fit(this_taste_long.T)
scaled_long_data = scaler_object.transform(this_taste_long.T).T
visualize.imshow(scaled_long_data);plt.show()
n_components = 10
pca_object = pca(n_components = n_components)\
        .fit(scaled_long_data.T)
plt.plot(np.cumsum(pca_object.explained_variance_ratio_),'x');plt.show()
pca_long_data = pca_object.transform(scaled_long_data.T).T
visualize.imshow(pca_long_data);plt.show()

# Use pca to convert data trial by trial
this_taste_swap = this_taste.swapaxes(0,1)
scaled_taste = np.array([scaler_object.transform(trial.T).T \
        for trial in this_taste_swap])
pca_taste = np.array([pca_object.transform(trial.T).T \
        for trial in scaled_taste])

pca_median_taste = np.median(pca_taste,axis=0)
pca_mean_taste = np.mean(pca_taste,axis=0)

# This plot well demonstrates that the mean population
# firing across trials is not very representative of all trials

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(pca_long_data[0],pca_long_data[1],pca_long_data[2], alpha = 0.5)
ax.scatter(pca_median_taste[0],pca_median_taste[1],pca_median_taste[2])
ax.plot(pca_mean_taste[0],pca_mean_taste[1],pca_mean_taste[2])
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pca_median_taste[0],pca_median_taste[1],pca_median_taste[2],
        c = 'red', alpha = 0.5)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(pca_long_data[0],pca_long_data[1], alpha = 0.5)
ax.scatter(pca_median_taste[0],pca_median_taste[1])
ax2 = fig.add_subplot(122)
ax2.scatter(pca_median_taste[0],pca_median_taste[1],
        c = 'red', alpha = 0.5)
plt.show()

##################################################
## Generate time and velocity colored plots of mean trajectory 
##################################################
taste_3d = pca_median_taste[:3]

pca_velocity = np.diff(pca_taste, axis=-1)
pca_speed = np.linalg.norm(pca_velocity,axis=1)
#median_speed = np.median(pca_speed,axis=0)
mean_speed = np.mean(pca_speed,axis=0)
smooth_speed = savgol_filter(mean_speed, 5, 1)

# The appearance of only 2 sharp peaks can be rationalized
# by the idea that only the stimulus aligned transitions are 
# visible in the mean speed
plt.plot(mean_speed);plt.plot(smooth_speed);plt.show()

time_color = np.arange(0,pca_taste.shape[-1])

#fig = plt.figure()
#ax = fig.add_subplot(121, projection='3d')
#ax.plot(pca_long_data[0],pca_long_data[1],pca_long_data[2], alpha = 0.5)
#ax.scatter(pca_median_taste[0],pca_median_taste[1],pca_median_taste[2])
#ax.plot(pca_mean_taste[0],pca_mean_taste[1],pca_mean_taste[2])
#ax2 = fig.add_subplot(122, projection='3d')

# Mark every second
fig = plt.figure()
ax2 = fig.add_subplot(111, projection = '3d')
im2 = ax2.scatter(pca_median_taste[0],pca_median_taste[1],pca_median_taste[2],
        c = time_color, alpha = 0.5, cmap = 'jet')
ax2.scatter(pca_median_taste[0],pca_median_taste[1],pca_median_taste[2],
        s = 0.5, c='k')
fig.colorbar(im2,ax = ax2)
plt.show()


##################################################
## Plotting 3D errorbars
##################################################
fig = plt.figure(dpi=100)
ax = fig.add_subplot(111, projection='3d')

#data
fx = [0.673574075,0.727952994,0.6746285]
fy = [0.331657721,0.447817839,0.37733386]
fz = [18.13629648,8.620699842,9.807536512]

#error data
xerror = [0.041504064,0.02402152,0.059383144]
yerror = [0.015649804,0.12643117,0.068676131]
zerror = [3.677693713,1.345712547,0.724095592]

#plot points
ax.plot(fx, fy, fz, linestyle="None", marker="o")

#plot errorbars
for i in np.arange(0, len(fx)):
    ax.plot([fx[i]+xerror[i], fx[i]-xerror[i]], [fy[i], fy[i]], [fz[i], fz[i]], marker="_")
    ax.plot([fx[i], fx[i]], [fy[i]+yerror[i], fy[i]-yerror[i]], [fz[i], fz[i]], marker="_")
    ax.plot([fx[i], fx[i]], [fy[i], fy[i]], [fz[i]+zerror[i], fz[i]-zerror[i]], marker="_")

#configure axes
ax.set_xlim3d(0.55, 0.8)
ax.set_ylim3d(0.2, 0.5)
ax.set_zlim3d(8, 19)

plt.show()
