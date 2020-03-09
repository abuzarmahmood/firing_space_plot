## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel,delayed
import pingouin
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import zscore

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extract data
dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM21/AM21_4Tastes_200305_085032')
    #ephys_data('/media/bigdata/Abuzar_Data/AM21/AM21_4Tastes_200306_085755/')
dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()
dat.separate_laser_data()

#dat.firing_overview(dat.all_on_lfp.swapaxes(0,1),zscore_bool=True)
#dat.firing_overview(dat.all_off_lfp.swapaxes(0,1),zscore_bool=True)
#plt.show()

on_hilbert = hilbert(dat.all_on_lfp)
off_hilbert = hilbert(dat.all_off_lfp)

on_spikes = dat.on_spikes
off_spikes = dat.off_spikes

# Bin firing into larger bins for anova
time_lims = (2000,4500)
bin_width = 250
fs = 1000

binned_on_firing = np.sum(np.reshape(on_spikes[...,time_lims[0]:time_lims[1]],
                    (*on_spikes.shape[:3],bin_width,-1)),axis=-2)/ (bin_width/fs)
binned_off_firing = np.sum(np.reshape(off_spikes[...,time_lims[0]:time_lims[1]],
                    (*off_spikes.shape[:3],bin_width,-1)),axis=-2)/ (bin_width/fs)

dim_inds = np.stack(list(np.ndindex(binned_on_firing.shape))).T
all_electrode_numbers = [x['electrode_number'] for x in dat.unit_descriptors]

firing_frame_list = [pd.DataFrame(\
        data = {'taste' : dim_inds[0],
                'trial' : dim_inds[1],
                'neuron' : dim_inds[2],
                'bin' : dim_inds[3],
                'laser' : num,
                'firing' : dat.flatten()}) \
                        for num,dat in enumerate(list((binned_off_firing,binned_on_firing)))]
firing_frame = pd.concat(firing_frame_list)

# Plot firing to make sure everything looks good
# Plot all discriminative neurons
#g = sns.FacetGrid(data = firing_frame.loc[firing_frame.neuron < 10],
#            col = 'neuron', row = 'taste', hue = 'laser', sharey = False)
#g.map(sns.pointplot, 'bin', 'firing')
#plt.show()

# Perform ANOVA on each neuron individually
# 3 Way ANOVA : Taste, laser, time
anova_list = [
    firing_frame.loc[firing_frame.neuron == nrn,:]\
            .anova(dv = 'firing', \
             between = ['taste','laser','bin'])\
            for nrn in tqdm(firing_frame.neuron.unique())]

anova_p_list = [x[['Source','p-unc']][:3] for x in anova_list]
p_val_tresh = 0.05 #/ len(firing_frame.bin.unique()) 
sig_taste = [True if x['p-unc'][0]<p_val_tresh else False for x in anova_p_list]
sig_laser = [True if x['p-unc'][1]<p_val_tresh else False for x in anova_p_list]

# Perform pairwise ttests
#pairwise_pvals = firing_frame.pairwise_ttests(dv = 'firing',
#        between = 'laser', within=['neuron','taste','bin'])

# Plot neurons with significant laser values
# Only neurons with significant laser effects and >1500 waveforms
waveform_thresh = 1500
sig_waveform = np.array([x['waveform_count']>waveform_thresh for x in dat.unit_descriptors]) 
relevant_laser = sig_laser * sig_waveform 

g = sns.FacetGrid(data = firing_frame.loc[firing_frame.neuron.isin(np.where(relevant_laser)[0])],
            col = 'neuron', row = 'taste', hue = 'laser', sharey = False)
g.map(sns.pointplot, 'bin', 'firing')
plt.show()

# Generate venn diagrams to visualize taste discriminatoriness
# and laser affectedness

v3 = venn3([set(np.where(sig_waveform)[0]),
            set(np.where(relevant_laser)[0]),
            set(np.where(sig_waveform*sig_taste)[0])],
            set_labels = ('All neurons','Laser','Taste'))
v3_c = venn3_circles([set(np.where(sig_waveform)[0]),
            set(np.where(relevant_laser)[0]),
            set(np.where(sig_waveform*sig_taste)[0])],
            linestyle = 'dashed')
plt.show()

# Venn Diagram for which regions laser affected units belong to
electrode_neuron = [(all_electrode_numbers[num],num) for num in range(len(sig_waveform))]
#all_unit_channels = np.array([x['electrode_number'] for x in dat.unit_descriptors[sig_waveform]])
#relevant_laser_channels = np.array([x['electrode_number'] \
#        for x in dat.unit_descriptors[relevant_laser]])
port_a = np.arange(32)
port_b = np.arange(32,64)
port_a_present = [x for x,wav in zip(electrode_neuron,sig_waveform) if (x[0] in port_a) and wav]
port_b_present = [x for x,wav in zip(electrode_neuron,sig_waveform) if (x[0] in port_b) and wav]
laser_present = [x for x,laser in zip(electrode_neuron,relevant_laser) if laser]
v3 = venn3([set(port_a_present),
            set(port_b_present),
            set(laser_present)],
            set_labels = ('Port A','Port B','Laser'))
v3_c = venn3_circles([set(port_a_present),
            set(port_b_present),
            set(laser_present)],
            linestyle = 'dashed')
plt.show()

##################################################
# ____                   _       _   _             
#|  _ \ ___  _ __  _   _| | __ _| |_(_) ___  _ __  
#| |_) / _ \| '_ \| | | | |/ _` | __| |/ _ \| '_ \ 
#|  __/ (_) | |_) | |_| | | (_| | |_| | (_) | | | |
#|_|   \___/| .__/ \__,_|_|\__,_|\__|_|\___/|_| |_|
#           |_|                                    
##################################################

# Changes in population firing
# Laser differences in principal components
# Elongate along all time dimensions
concat_firing = np.concatenate((binned_on_firing,binned_off_firing),axis=-1)
# Remove neurons below waveform threshold
concat_firing = concat_firing[:,:,sig_waveform]
concat_firing = np.rollaxis(concat_firing,2,0)
concat_firing_long = np.reshape(concat_firing,(concat_firing.shape[0],-1))
# Add noise to remove spurious effects
concat_firing_long += np.random.random(concat_firing_long.shape)*1e-6
zscore_concat_firing_long = zscore(concat_firing_long,axis=-1) 
pca_object = pca(n_components = 20).fit(zscore_concat_firing_long.T)
variance_thresh = 0.8
needed_components = np.sum(
                        np.cumsum(
                            pca_object.explained_variance_ratio_) < variance_thresh)+1
pca_object = pca(n_components = needed_components).fit(zscore_concat_firing_long.T)

red_zscore_firing = pca_object.transform(zscore_concat_firing_long.T)
labels = 1+(np.linspace(0,1,red_zscore_firing.shape[0])<0.5)*1
plt.scatter(red_zscore_firing[:,0],red_zscore_firing[:,1], c=labels, alpha = 0.5,cmap='jet');plt.show()
plt.plot(red_zscore_firing[:,1]);plt.show()


# Transform all data into principal components and plot average values of PC's for every taste
array = binned_on_firing[:,:,sig_waveform]
array = np.moveaxis(array,2,0)

def reduce_array_intact(array, dim_red_object):
    """
    array : Array to be reduced along the 0-th dimension
    """
    array = np.moveaxis(array,0,-1)
    reduced_array = np.zeros((*array.shape[:-1],dim_red_object.n_components)) 
    inds = list(np.ndindex(array.shape[:-1]))
    for index in inds:
        reduced_array[index] = dim_red_object.transform(array[index].reshape(1,-1))
    return reduced_array

red_on_firing = reduce_array_intact(np.moveaxis(binned_on_firing,2,0)[sig_waveform], pca_object)
red_off_firing = reduce_array_intact(np.moveaxis(binned_off_firing,2,0)[sig_waveform], pca_object)

red_firing_concat = np.concatenate((red_on_firing,red_off_firing),axis=-1) 
red_firing_concat = np.moveaxis(red_firing_concat,2,0)
red_firing_long = np.reshape(red_firing_concat,(red_firing_concat.shape[0],-1))

dim_inds = np.stack(list(np.ndindex(red_on_firing.shape))).T
red_firing_frame_list = [pd.DataFrame(\
        data = {'taste' : dim_inds[0],
                'trial' : dim_inds[1],
                'pc' : dim_inds[2],
                'bin' : dim_inds[3],
                'laser' : num,
                'firing' : dat.flatten()}) \
                        for num,dat in enumerate(list((red_off_firing,red_on_firing)))]
red_firing_frame = pd.concat(red_firing_frame_list)

g = sns.FacetGrid(data = red_firing_frame,
            col = 'pc', row = 'taste', hue = 'laser', sharey = False)
g.map(sns.pointplot, 'bin', 'firing')
plt.show()



mean_red_on_firing = np.mean(red_on_firing,axis=1)
mean_red_off_firing = np.mean(red_off_firing,axis=1)
std_red_on_firing = np.std(red_on_firing,axis=1)
std_red_off_firing = np.std(red_off_firing,axis=1)
