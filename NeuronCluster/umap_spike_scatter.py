import numpy as np
import os
import umap
import pylab as plt
import glob
from tqdm import trange
from joblib import Parallel, delayed
import multiprocessing as mp

# Read blech.dir, and cd to that directory
with open('blech.dir','r') as blech_dir:
    data_dir = blech_dir.readline()[:-1]

# Read the clustering params for the file
with open(glob.glob(data_dir + '/*params*')[0],'r') as param_file:
    params = [float(line) for line in param_file.readlines()[:-1]]
cluster_num = int(params[0])

# Get PCA waveforms from spike_waveforms
# Get cluster predictions from clustering_results
# Plot output in Plots


def umap_plots(data_dir, electrode_num):
    # If processing has happened, the file will exist
    pca_file = data_dir + \
                '/spike_waveforms/electrode{}/pca_waveforms.npy'.format(electrode_num)

    if os.path.isfile(pca_file):

        pca_waveforms = np.load(data_dir + \
                '/spike_waveforms/electrode{}/pca_waveforms.npy'.format(electrode_num))

        umap_waveforms = umap.UMAP(n_components = 2).\
                fit_transform(pca_waveforms[:,:20])
        
        clustering_results = [np.load(data_dir + \
                '/clustering_results/electrode{0}/clusters{1}/predictions.npy'.\
                format(electrode_num, cluster)) for cluster in \
                range(2,cluster_num+1)] 
        
        print('Processing for Electrode {} complete'.format(electrode_num))

        for cluster in range(2,cluster_num+1):
            fig1, ax1 = plt.subplots()
            scatter = ax1.scatter(umap_waveforms[:,0],umap_waveforms[:,1],\
                    c = clustering_results[cluster-2], s = 2, cmap = 'jet')
            legend = ax1.legend(*scatter.legend_elements())
            ax1.add_artist(legend)
            fig1.savefig(data_dir + \
                '/Plots/{0}/Plots/{1}_clusters_waveforms_ISIs/cluster{1}_umap.png'.\
                format(electrode_num, cluster), 
                dpi = 300)
            plt.close(fig1)

            nbins = np.min([100,int(umap_waveforms.shape[0]/100)])
            fig2, ax2 = plt.subplots()
            ax2.hexbin(umap_waveforms[:,0],umap_waveforms[:,1], gridsize = nbins)
            fig2.savefig(data_dir + \
                '/Plots/{0}/Plots/{1}_clusters_waveforms_ISIs/cluster{1}_umap_hist.png'.\
                format(electrode_num, cluster),
                dpi = 300)
            plt.close(fig2)

for electrode_num in trange(len(os.listdir(data_dir + '/clustering_results'))):
    umap_plots(data_dir, electrode_num)

#Parallel(n_jobs = mp.cpu_count())\
#        (delayed(umap_plots)(data_dir, electrode_num, pca_waveforms) \
#        for electrode_num in \
#        trange(len(os.listdir(data_dir + '/clustering_results'))))
