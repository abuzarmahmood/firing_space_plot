## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Qt5Agg')
import tables
#import h5py
#import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from itertools import product
from scipy.stats import zscore
import glob
from collections import namedtuple
from scipy.signal import convolve
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.decomposition import PCA as pca
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans as kmeans

def img_plot(array):
    plt.imshow(array,interpolation='nearest',
            aspect='auto',cmap='viridis')

h5_path = '/media/bigdata/Abuzar_Data/AM23/AM23_4Tastes_200316_134649/AM23_4Tastes_200316_134649_repacked.bk'

h5_file = tables.open_file(h5_path,'r')

unit_descriptors = h5_file.root.unit_descriptor[:]
sorted_units_path = '/sorted_units'
unit_num = 3

this_unit_waves = h5_file.get_node(os.path.join(sorted_units_path,
    'unit{0:03d}'.format(unit_num),'waveforms'))[:]

this_unit_pca = pca(n_components = 3).fit_transform(this_unit_waves)

ac_cluster = AC().fit(this_unit_pca)
kmeans_cluster = kmeans(n_clusters = 3).fit(this_unit_pca)

clust_method = ac_cluster
mean_wavs = [(np.mean(this_unit_waves[clust_method.labels_ == clust],
    axis=0),
    np.std(this_unit_waves[clust_method.labels_ == clust],axis=0)) \
            for clust in np.sort(np.unique(clust_method.labels_))]

img_plot(this_unit_waves[np.argsort(kmeans_cluster.labels_)]);plt.show()

for wav in mean_wavs:
    plt.fill_between(range(len(wav[0])), 
            wav[0]+2*wav[1], wav[0]-2*wav[1], alpha = 0.5)
    plt.plot(wav[0])
plt.show()


fig,ax = plt.subplots(2)
plt.sca(ax[0])
img_plot(this_unit_pca[np.argsort(kmeans.labels_)])
plt.sca(ax[1])
img_plot(this_unit_pca[np.argsort(ac_cluster.labels_)]);plt.show()

mean_wave = np.mean(this_unit_waves,axis=0)
std_wave = np.std(this_unit_waves,axis=0)

fig,ax = plt.subplots(1,3)
plt.sca(ax[0])
img_plot(this_unit_waves)
plt.sca(ax[1])
img_plot(this_unit_pca)
#ax[2].scatter(this_unit_pca[:,0],this_unit_pca[:,1],alpha = 0.5,s=2)
ax[2].fill_between(np.arange(len(mean_wave)), mean_wave - 2*std_wave,
        mean_wave + 2*std_wave,alpha = 0.5)
ax[2].plot(mean_wave,'k')
plt.show()

## Final code
unit_num = 2

this_unit_waves = h5_file.get_node(os.path.join(sorted_units_path,
    'unit{0:03d}'.format(unit_num),'waveforms'))[:]
len(this_unit_waves)
this_unit_pca = pca(n_components = 3).fit_transform(this_unit_waves)

#ac_cluster = AC().fit(this_unit_pca)
kmeans_cluster = kmeans(n_clusters = 3).fit(this_unit_pca)
clust_method = kmeans_cluster
mean_wavs = [(np.mean(this_unit_waves[clust_method.labels_ == clust],
    axis=0),
    np.std(this_unit_waves[clust_method.labels_ == clust],axis=0)) \
            for clust in np.sort(np.unique(clust_method.labels_))]

fig = plt.figure(figsize = (10,6))
ax0 = fig.add_subplot(161)
ax1 = fig.add_subplot(162)
ax2 = fig.add_subplot(163)
ax3 = fig.add_subplot(164)
ax4 = fig.add_subplot(133)
plt.sca(ax0)
img_plot(this_unit_waves)
plt.sca(ax1)
img_plot(this_unit_waves[np.argsort(clust_method.labels_)])
plt.sca(ax2)
img_plot(this_unit_pca)
plt.sca(ax3)
img_plot(this_unit_pca[np.argsort(clust_method.labels_)])
plt.sca(ax4)
for wav in mean_wavs:
    plt.fill_between(range(len(wav[0])), 
            wav[0]+2*wav[1], wav[0]-2*wav[1], alpha = 0.2)
    plt.plot(wav[0])
for this_ax in ax:
    this_ax.set_axis_off()
plt.show()


fig,ax = plt.subplots(1,5,figsize = (10,10))
plt.sca(ax[0])
img_plot(this_unit_waves)
plt.sca(ax[1])
img_plot(this_unit_waves[np.argsort(clust_method.labels_)])
plt.sca(ax[2])
img_plot(this_unit_pca)
plt.sca(ax[3])
img_plot(this_unit_pca[np.argsort(clust_method.labels_)])
plt.sca(ax[4])
for wav in mean_wavs:
    plt.fill_between(range(len(wav[0])), 
            wav[0]+2*wav[1], wav[0]-2*wav[1], alpha = 0.2)
    plt.plot(wav[0])
for this_ax in ax:
    this_ax.set_axis_off()
plt.show()
