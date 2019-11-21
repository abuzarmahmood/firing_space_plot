print('Importing modules')

import os

start_dir = os.getcwd()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import umap
import dill
#from sklearn.decomposition import PCA as pca
os.chdir('/media/bigdata/firing_space_plot/NeuronCluster')
from polygon_selector_demo import SelectFromCollection, NeuronCluster
from scipy.signal import decimate

#data_directory = \
#'/media/bigdata/brads_data/BS26_4Tastes_1080202/'
#
#waveforms = np.load(data_directory + \
#        'spike_waveforms/electrode17/spike_waveforms.npy')
#predictions = np.load(data_directory + \
#        'clustering_results/electrode17/clusters3/predictions.npy')
#
#pca_waveforms = pca(n_components = 10).fit_transform(waveforms)
#umap_waveforms = umap.UMAP(n_components = 2).fit_transform(pca_waveforms)

#pca_waveforms = pca_waveforms[:5000,:] 
#plotting_waveforms = decimate(waveforms,10)

print('Loading Data')

dill_file = 'global.pkl'
#dill.dump_session(dill_file)
dill.load_session(dill_file)


#cluster = NeuronCluster(waveforms, umap_waveforms)
cluster = NeuronCluster(plotting_waveforms, pca_waveforms[:,:2])
cluster.initiate_plots()
exit()

#def onclick(event):
#    print((event.xdata,event.ydata))
#
#fig, ax = plt.subplots()
##ax.hexbin(umap_waveforms[:,0],umap_waveforms[:,1])
#ax.scatter(umap_waveforms[:,0],umap_waveforms[:,1])
#cid = fig.canvas.mpl_connect('button_press_event', onclick)
#plt.show()
#
#exit()

# Scatter plot
plt.scatter(umap_waveforms[:,0],umap_waveforms[:,1],c=predictions)
plt.show()

# Desnity plot
nbins = 100
fig, ax = plt.subplots()
ax.hexbin(umap_waveforms[:,0],umap_waveforms[:,1], gridsize = nbins)
plt.show()
