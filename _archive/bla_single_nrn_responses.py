"""
Confirm that single unit responses from recorded BLA neurons
agree with past literature. Checks done in particular:

1) BLA palatability response is confined to about GC Identity 
    epoch (i.e. 250-750 ms post-stimulus delivery)
2) Descriminability in BLA single-neuron responses is confined
    to being palatability related i.e. BLA does not show 
    discriminability other than palatability

***
While we're at it, might aswell do that same for GC responses
***

README and file_list in /media/fastdata/bla_firing_corroborate
"""


## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt4Agg')
import tables
import easygui
import scipy
import json
import glob
import numpy as np
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import pingouin
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as pca
from sklearn.cluster import KMeans as kmeans
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import ttest_rel 

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Create file to store data + analyses in
file_list_path = '/media/fastdata/bla_firing_corroborate/file_list.txt' 
plot_dir = os.path.join(os.path.dirname(file_list_path),'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

file_list = open(file_list_path,'r').readlines()
file_list = [x.rstrip() for x in file_list]
basename_list = [os.path.basename(x) for x in file_list]
name_date_str_list = ['_'.join([x.split('_')[-1],x.split('_')[2]]) \
        for x in basename_list]
dirname_list = [os.path.dirname(x) for x in file_list]
json_path_list = [glob.glob(dirname+'/*json')[-1] for dirname in dirname_list]

# Find all electrodes corresponding to BLA and GC using the jsons
json_list = [json.load(open(file,'r')) for file in json_path_list]
bla_electrodes = [x['regions']['bla'] for x in json_list]
gc_electrodes = [x['regions']['gc'] for x in json_list]

all_spike_array = []
all_firing = []
all_unit_descriptors = []
waveform_thresh = 1500
for file_num, file_name in tqdm(enumerate(file_list)):
    data = ephys_data(data_dir = dirname_list[file_num]) 
    data.firing_rate_params = dict(zip(\
        ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
        ('conv',25,250,1,25e-3,1e-3)))
    data.get_unit_descriptors()
    data.get_spikes()
    data.get_firing_rates()
    all_firing.append(data.normalized_firing)
    all_spike_array.append(np.array(data.spikes).swapaxes(1,2))
    all_unit_descriptors.append(data.unit_descriptors)

# Remove cases where size of firing array doesn't match with
# unit descriptors
# WTF WOULD THAT HAPPEN THOUGH???

all_unit_electrodes = [x["electrode_number"] for x in all_unit_descriptors]
matchings_inds = [num for num, (firing, electrodes) in \
        enumerate(zip(all_firing, all_unit_electrodes)) \
        if firing.shape[1] == len(electrodes)]

fin_filenames = [name_date_str_list[x] for x in matchings_inds]
fin_firing = [all_firing[x] for x in matchings_inds]
fin_spikes = [all_spike_array[x] for x in matchings_inds]
fin_unit_electrodes = [all_unit_electrodes[x] for x in matchings_inds]
fin_gc_electrodes = [gc_electrodes[x] for x in matchings_inds]
fin_bla_electrodes = [bla_electrodes[x] for x in matchings_inds]

gc_unit_inds = [np.where(np.isin(unit_electrodes, this_gc_electrodes))[-1]\
        for unit_electrodes, this_gc_electrodes \
        in zip(fin_unit_electrodes, fin_gc_electrodes)]
bla_unit_inds = [np.where(np.isin(unit_electrodes, this_bla_electrodes))[-1]\
        for unit_electrodes, this_bla_electrodes \
        in zip(fin_unit_electrodes, fin_bla_electrodes)]
gc_nrn_count = [len(x) for x in gc_unit_inds]
bla_nrn_count = [len(x) for x in bla_unit_inds]

gc_nrn_list = [firing[:,inds] for firing,inds in zip(fin_firing, gc_unit_inds)]
bla_nrn_list = [firing[:,inds] for firing,inds in zip(fin_firing, bla_unit_inds)]
gc_spike_list = [spikes[:,inds] for spikes,inds in zip(fin_spikes, gc_unit_inds)]
bla_spike_list = [spikes[:,inds] for spikes,inds in zip(fin_spikes, bla_unit_inds)]

gc_nrn_array = np.concatenate(gc_nrn_list, axis=1).swapaxes(0,1) 
bla_nrn_array = np.concatenate(bla_nrn_list, axis=1).swapaxes(0,1)
gc_spike_array = np.concatenate(gc_spike_list, axis=1).swapaxes(0,1) 
bla_spike_array = np.concatenate(bla_spike_list, axis=1).swapaxes(0,1)

##################################################
# Fraction of taste DISCRIMINATIVE neurons in BLA and GC
##################################################
stim_inds = np.arange(2000,4500) # 2000 - 4500 ms
x_vec = np.linspace(0,7000,gc_nrn_array.shape[-1])
x_vec_binned = np.round(np.mean(x_vec[stim_inds].reshape((bin_count,-1)),axis=-1))
bin_count = 5
gc_spikes_binned = np.mean(
        gc_spike_array[...,stim_inds].reshape(\
                        (*gc_spike_array.shape[:-1],bin_count, -1)),
                    axis = -1)
bla_spikes_binned = np.mean(
        bla_spike_array[...,stim_inds].reshape(\
                        (*bla_spike_array.shape[:-1],bin_count, -1)),
                    axis = -1)

#gc_firing_binned = np.mean(
#        gc_nrn_array[...,stim_inds].reshape(\
#                        (*gc_nrn_array.shape[:-1],bin_count, -1)),
#                    axis = -1)
#bla_firing_binned = np.mean(
#        bla_nrn_array[...,stim_inds].reshape(\
#                        (*bla_nrn_array.shape[:-1],bin_count, -1)),
#                    axis = -1)

gc_dim_inds = np.stack(list(np.ndindex(gc_spikes_binned.shape))).T
bla_dim_inds = np.stack(list(np.ndindex(bla_spikes_binned.shape))).T

gc_firing_frame = pd.DataFrame(\
        data = {'neuron' : gc_dim_inds[0],
                'taste' : gc_dim_inds[1],
                'trial' : gc_dim_inds[2],
                'bin' : gc_dim_inds[3],
                'firing' : gc_spikes_binned.flatten()}) \

bla_firing_frame = pd.DataFrame(\
        data = {'neuron' : bla_dim_inds[0],
                'taste' : bla_dim_inds[1],
                'trial' : bla_dim_inds[2],
                'bin' : bla_dim_inds[3],
                'firing' : bla_spikes_binned.flatten()}) \

# Perform ANOVA on each neuron individually
# 3 Way ANOVA : Taste, laser, time
gc_anova_list = [
    gc_firing_frame.loc[gc_firing_frame.neuron == nrn,:]\
            .anova(dv = 'firing', \
             between = ['taste'])\
            for nrn in tqdm(gc_firing_frame.neuron.unique())]
bla_anova_list = [
    bla_firing_frame.loc[bla_firing_frame.neuron == nrn,:]\
            .anova(dv = 'firing', \
             between = ['taste'])\
            for nrn in tqdm(bla_firing_frame.neuron.unique())]

gc_anova_p_list = [x[['Source','p-unc']][:3] for x in gc_anova_list]
bla_anova_p_list = [x[['Source','p-unc']][:3] for x in bla_anova_list]

# Get main taste and interaction
gc_anova_parray = np.array([
                x['p-unc'][[0]] for x in gc_anova_p_list])
bla_anova_parray = np.array([
                x['p-unc'][[0]] for x in bla_anova_p_list])

# Find neurons with significant responses
alpha = 1e-3
gc_sig_array = np.sum(gc_anova_parray < alpha,axis=-1)
bla_sig_array = np.sum(bla_anova_parray < alpha,axis=-1)

gc_fraction = np.mean(gc_sig_array > 0)
bla_fraction = np.mean(bla_sig_array > 0)

# Output plots of firing for each significant and non-significant
# neuron to confirm ANOVA results
for nrn in range(gc_sig_array.shape[0]):
    fig = plt.figure()
    mean_firing = np.mean(gc_nrn_array[nrn],axis=1)
    std_firing = np.std(gc_nrn_array[nrn],axis=1)
    for mean_vec, std_vec in zip(mean_firing, std_firing):
        plt.fill_between(x = x_vec,
                y1 = mean_vec + std_vec,
                y2 = mean_vec - std_vec,
                alpha = 0.5)
        plt.plot(x_vec,mean_vec)
    if gc_sig_array[nrn] > 0:
        this_plot_dir = os.path.join(plot_dir,'gc','taste_discrim')
    else:
        this_plot_dir = os.path.join(plot_dir,'gc','non_taste_discrim')
    if not os.path.exists(this_plot_dir):
        os.makedirs(this_plot_dir)
    fig.savefig(os.path.join(this_plot_dir,str(nrn)))
    plt.close(fig)

for nrn in range(bla_sig_array.shape[0]):
    mean_firing = np.mean(bla_nrn_array[nrn],axis=1)
    std_firing = np.std(bla_nrn_array[nrn],axis=1)
    fig = plt.figure()
    for mean_vec, std_vec in zip(mean_firing, std_firing):
        plt.fill_between(x = x_vec,
                y1 = mean_vec + std_vec,
                y2 = mean_vec - std_vec,
                alpha = 0.5)
        plt.plot(x_vec,mean_vec)
    if bla_sig_array[nrn] > 0:
        this_plot_dir = os.path.join(plot_dir,'bla','taste_discrim')
    else:
        this_plot_dir = os.path.join(plot_dir,'bla','non_taste_discrim')
    if not os.path.exists(this_plot_dir):
        os.makedirs(this_plot_dir)
    fig.savefig(os.path.join(this_plot_dir,str(nrn)))
    plt.close(fig)


########################################
## Find palatability correlation for both BLA and GC
########################################
# spearmanr only supports upto 2D arrays
# reshape nrn_arrays to be long and iterate over neurons
gc_nrn_long = gc_nrn_array.reshape(
            (gc_nrn_array.shape[0],-1,gc_nrn_array.shape[-1])).swapaxes(1,2)
bla_nrn_long = bla_nrn_array.reshape(
            (bla_nrn_array.shape[0],-1,bla_nrn_array.shape[-1])).swapaxes(1,2)

#palatability_ranks = [3,4,2,1]
palatability_ranks = [0,0,1,1]
rank_vec = np.broadcast_to(np.reshape(palatability_ranks,(-1,1)),
        (gc_nrn_array.shape[1:3])).flatten()

gc_pal_corrs = np.empty((*gc_nrn_long.shape[:2],2))
bla_pal_corrs = np.empty((*bla_nrn_long.shape[:2],2))

gc_inds = list(np.ndindex(gc_pal_corrs.shape[:2]))
bla_inds = list(np.ndindex(bla_pal_corrs.shape[:2]))

for this_ind in tqdm(gc_inds):
    gc_pal_corrs[this_ind] = spearmanr(rank_vec, gc_nrn_long[this_ind])
for this_ind in tqdm(bla_inds):
    bla_pal_corrs[this_ind] = spearmanr(rank_vec, bla_nrn_long[this_ind])


# Plot all correlations
fix, ax  = plt.subplots(1,2)
plt.sca(ax[0])
imshow(np.abs(gc_pal_corrs[...,0]))
plt.sca(ax[1])
imshow(np.abs(bla_pal_corrs[...,0]))
plt.show()

# Plot mean correlations
fig,ax = plt.subplots(2,1)
ax[0].plot(np.abs(np.mean(gc_pal_corrs[...,0],axis=0)))
ax[1].plot(np.abs(np.mean(bla_pal_corrs[...,0],axis=0)))
plt.show()

# Sort neurons into groups with hierarchical clustering
alpha = 0.05
gc_pal_sig = gc_pal_corrs[...,1] < alpha
bla_pal_sig = bla_pal_corrs[...,1] < alpha

# Break down significant array by session to see differences
gc_nrns_markers = np.append(0,np.cumsum(gc_nrn_count))
bla_nrns_markers = np.append(0,np.cumsum(bla_nrn_count))
gc_sig_list = [gc_pal_sig[gc_nrns_markers[i]:gc_nrns_markers[i+1]] \
        for i in range(len(gc_nrns_markers)-1)]
bla_sig_list = [bla_pal_sig[bla_nrns_markers[i]:bla_nrns_markers[i+1]] \
        for i in range(len(bla_nrns_markers)-1)]
gc_corrs_list = [gc_pal_corrs[gc_nrns_markers[i]:gc_nrns_markers[i+1]][...,0] \
        for i in range(len(gc_nrns_markers)-1)]
bla_corrs_list = [bla_pal_corrs[bla_nrns_markers[i]:bla_nrns_markers[i+1]][...,0] \
        for i in range(len(bla_nrns_markers)-1)]

x = np.arange(gc_pal_sig.shape[-1],step=40) 
fig, ax = gen_square_subplots(len(gc_sig_list))
for num, (this_firing,this_ax) in enumerate(zip(gc_sig_list,ax.flatten())):
    plt.sca(this_ax)
    imshow(this_firing)
    this_ax.set_title(fin_filenames[num])
    plt.xticks(ticks = x, labels = x)
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.set_size_inches(15,10)
#fig.savefig(os.path.join(plot_dir, 'gc_palcorr_sig'),dpi = 300)
plt.show()

fig, ax = gen_square_subplots(len(bla_sig_list))
for num, (this_firing,this_ax) in enumerate(zip(bla_sig_list,ax.flatten())):
    plt.sca(this_ax)
    imshow(this_firing)
    this_ax.set_title(fin_filenames[num])
    plt.xticks(ticks = x, labels = x)
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
fig.set_size_inches(15,10)
#fig.savefig(os.path.join(plot_dir, 'bla_palcorr_sig'),dpi = 300)
plt.show()



# Plot all significant bins
fix, ax  = plt.subplots(1,2)
plt.sca(ax[0])
imshow(gc_pal_sig)
plt.sca(ax[1])
imshow(bla_pal_sig)
plt.show()

n_components = 5
gc_pal_sig_pca = pca(n_components = n_components).fit_transform(gc_pal_sig)
bla_pal_sig_pca = pca(n_components = n_components).fit_transform(bla_pal_sig)

clusters = 5
gc_pal_kmeans = kmeans(n_clusters = clusters).fit(gc_pal_sig_pca)
bla_pal_kmeans = kmeans(n_clusters = clusters).fit(bla_pal_sig_pca)
# Plot sorted units
fig, ax = plt.subplots(2)
plt.sca(ax[0])
imshow(gc_pal_sig[np.argsort(gc_pal_kmeans.labels_)])
plt.sca(ax[1])
imshow(bla_pal_sig[np.argsort(bla_pal_kmeans.labels_)])
plt.show()


fraction_sig_gc = np.mean(gc_pal_corrs[...,1] < alpha,axis=0)
fraction_sig_bla = np.mean(bla_pal_corrs[...,1] < alpha,axis=0)

fig, ax = plt.subplots(2)
ax[0].plot(fraction_sig_gc)
ax[1].plot(fraction_sig_bla)
plt.show()

#gc_palatability_rank_array = np.broadcast_to(
#                    np.reshape(np.array(palatability_ranks),(1,-1,1,1)),
#                                gc_nrn_array.shape)
#bla_palatability_rank_array = np.broadcast_to(
#                    np.reshape(np.array(palatability_ranks),(1,-1,1,1)),
#                                bla_nrn_array.shape)
#gc_ranks_long = gc_palatability_rank_array.reshape(
#            (gc_nrn_array.shape[0],-1,gc_nrn_array.shape[-1]))
#bla_ranks_long = bla_palatability_rank_array.reshape(
#            (bla_nrn_array.shape[0],-1,bla_nrn_array.shape[-1]))
#gc_palatability_correlation = np.array([
#                                spearmanr(this_nrn, this_ranks,axis=0)
#            for this_nrn, this_ranks in tqdm(zip(gc_nrn_long, gc_ranks_long))])
#bla_palatability_correlation = np.array([
#                                spearmanr(this_nrn, this_ranks,axis=0)
#            for this_nrn, this_ranks in tqdm(zip(bla_nrn_long, bla_ranks_long))])
