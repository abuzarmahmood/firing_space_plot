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

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extract data
dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM21/AM21_4Tastes_200306_085755/')
dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()
dat.separate_laser_data()

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
g = sns.FacetGrid(data = firing_frame.loc[firing_frame.neuron < 10],
            col = 'neuron', row = 'taste', hue = 'laser', sharey = False)
g.map(sns.pointplot, 'bin', 'firing')
plt.show()

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
