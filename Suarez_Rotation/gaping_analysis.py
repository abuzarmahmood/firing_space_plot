# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   

import numpy as np
import tables
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import sys 
from tqdm import trange
import pingouin as pg
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp
import seaborn as sns
from scipy.stats import zscore

# Class to simplify loading of ephys data from Katz H5 files
sys.path.append('/media/bigdata/firing_space_plot/_old')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/PyHMM/PyHMM')
from fake_firing import *

import glob
from tqdm import tqdm

# ____        __   _____                     
#|  _ \  ___ / _| |  ___|   _ _ __   ___ ___ 
#| | | |/ _ \ |_  | |_ | | | | '_ \ / __/ __|
#| |_| |  __/  _| |  _|| |_| | | | | (__\__ \
#|____/ \___|_|   |_|   \__,_|_| |_|\___|___/
#                                            

def firing_overview(data, time_step = 25, cmap = 'jet'):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron
    """
    num_nrns = data.shape[0]
    t_vec = np.arange(data.shape[-1])*time_step 

    # Plot firing rates
    square_len = np.int(np.ceil(np.sqrt(num_nrns)))
    fig, ax = plt.subplots(square_len,square_len)
    
    nd_idx_objs = []
    for dim in range(ax.ndim):
        this_shape = np.ones(len(ax.shape))
        this_shape[dim] = ax.shape[dim]
        nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(ax.shape[dim]),this_shape.astype('int')), ax.shape).flatten())
    
    for nrn in range(num_nrns):
        plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
        plt.gca().set_title(nrn)
        plt.gca().pcolormesh(t_vec, np.arange(data.shape[1]),
                data[nrn,:,:],cmap=cmap)
        #self.imshow(data[nrn,:,:])
        #plt.show()
    return ax

# Create array index identifiers
# Used to convert array to pandas dataframe
def make_array_identifiers(array):
    nd_idx_objs = []
    for dim in range(array.ndim):
        this_shape = np.ones(len(array.shape))
        this_shape[dim] = array.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to(
                    np.reshape(
                        np.arange(array.shape[dim]),
                                this_shape.astype('int')), 
                    array.shape).flatten())
    return nd_idx_objs

def dat_imshow(x):
    plt.imshow(x,interpolation='nearest',aspect='auto')

# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               

## Find relevant HF5 files
dir_list = ['/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

all_sorted_off_firing = []
for file_num in trange(len(file_list)):

    
    # Load data
    # Load firing rate data

    # Use ephys_data class to load firing rate data for all neurons
    this_dir = file_list[file_num].split(sep='/')[-2]
    data_dir = os.path.dirname(file_list[file_num])
    data = ephys_data(data_dir = data_dir ,file_id = file_num, use_chosen_units = True)
    data.firing_rate_params = \
            dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                                   [25,250,7000,'conv',269]))
    data.get_data()
    data.get_firing_rates()
    data.get_normalized_firing()

    off_firing = np.asarray(data.normal_off_firing)
    on_firing = np.asarray(data.normal_on_firing)

    # Load gaping data
    hf5 = tables.open_file(file_list[file_num])
    gapes_array = hf5.root.ancillary_analysis.gapes[:]
    gapes_Li_array = hf5.root.ancillary_analysis.gapes_Li[:]
    # Load desciptor for laser condition
    laser_condition_array = hf5.root.ancillary_analysis.laser_combination_d_l[:]  
    # Since single condition, axis with all zeros will be off-laser
    laser_on_condition = np.sum(laser_condition_array,axis=-1)
    hf5.close()

    #    _                _           _     
    #   / \   _ __   __ _| |_   _ ___(_)___ 
    #  / _ \ | '_ \ / _` | | | | / __| / __|
    # / ___ \| | | | (_| | | |_| \__ \ \__ \
    #/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
    #                       |___/           

    # Reshape gape arrays to make plots
    gapes_array_long = gapes_array.reshape(\
            (gapes_array.shape[0],np.prod(gapes_array.shape[1:3]),gapes_array.shape[-1]))

    gapes_Li_long = gapes_Li_array.reshape(\
                (gapes_Li_array.shape[0], np.prod(gapes_Li_array.shape[1:3]),\
                gapes_Li_array.shape[-1]))


    ### Comparison of gapes_Li and gapes
    fig, ax = plt.subplots(1,3)
    plt.sca(ax[0])
    dot_raster(gapes_Li_long[0,:,:])
    plt.pcolormesh(gapes_array_long[0,:,:],cmap='viridis')
    plt.show()

    plt.sca(ax[1])
    #dot_raster(gapes_Li_long[1,:,:])
    plt.pcolormesh(gapes_array_long[1,:,:])
    plt.sca(ax[2])
    plt.plot(np.sum(gapes_array,axis=(0,2)).T)
    plt.legend(labels = ['Dil S', 'Conc S', 'Dil Q', 'Conc Q'], loc = 'upper right')
    plt.show()


    # Calculate total gaping and sort trials
    # Window to index for gaping probabilites
    gaping_window = (3000,5000)
    # Total gaping for a single trial from the window
    total_gaping = np.sum(gapes_array[:,:,:,gaping_window[0]:gaping_window[1]],axis=-1) 
    # Sort EVERYTHING along the trial axis
    sorted_gaping_trials = np.argsort(total_gaping, axis = -1)

    # Plot sorted gaping trials to make sure sorting worked
    sorted_off_gapes_array = np.asarray(\
            [x[sorted_gaping_trials[0,taste,:]] for taste,x in \
                enumerate(gapes_array[0])])

    sorted_on_gapes_array = np.asarray(\
            [x[sorted_gaping_trials[1,taste,:]] for taste,x in \
                enumerate(gapes_array[1])])
    firing_overview(sorted_off_gapes_array[:,:,gaping_window[0]:gaping_window[1]]);plt.show()
    firing_overview(sorted_on_gapes_array[:,:,gaping_window[0]:gaping_window[1]]);plt.show()

    ## Sort neural firing by "total gaping"
    sorted_off_firing = np.asarray(\
            [x[:,sorted_gaping_trials[0,taste,:]] for taste,x in \
                enumerate(off_firing)])

    all_sorted_off_firing.append(sorted_off_firing)

    sorted_on_firing = np.asarray(\
            [x[:,sorted_gaping_trials[1,taste,:]] for taste,x in \
                enumerate(on_firing)])

    # Visualize sorted firing
    #firing_overview(sorted_off_firing[2]);plt.title('Off, T2');plt.show()     
    #firing_overview(sorted_off_firing[3]);plt.title('Off, T3');plt.show()     

    #firing_overview(sorted_on_firing[2]);plt.title('On, T2');plt.show()     
    #firing_overview(sorted_on_firing[3]);plt.title('On, T3');plt.show()     

# Concatenate firing from ALL files along neuron axis
all_sorted_off_firing = np.concatenate(all_sorted_off_firing,axis=1)

# Visually remove bad neurons
firing_overview(all_sorted_off_firing[2]);plt.show()
firing_overview(all_sorted_off_firing[3]);plt.show()

bad_neurons = [17,30,32,38]
good_neurons = [x for x in range(all_sorted_off_firing.shape[1]) if x not in
        bad_neurons]

all_sorted_good_off_firing = all_sorted_off_firing[:,good_neurons]

# Make array identifiers to convert np array to dataframe
sorted_idx = make_array_identifiers(all_sorted_good_off_firing)
sorted_off_frame = pd.DataFrame({
    'taste': sorted_idx[0],
    'neuron' : sorted_idx[1],
    'trial' : sorted_idx[2],
    'time': sorted_idx[3],
    'firing_rate' : all_sorted_good_off_firing.flatten()})

# Mark units which change firing rate in first vs last half of trials
firing_window = [2000,5000]
step_size = 25
test_frame = sorted_off_frame.copy()
# Only look at post-stimulus firing
test_frame = test_frame.loc[(test_frame.time >= firing_window[0]/step_size) & \
        (test_frame.time < firing_window[1]/step_size),:]
# Chop time into 500ms bins 
bin_num = (firing_window[1]-firing_window[0])//500
test_frame['time_bin'] = pd.cut(test_frame.time,
        bins = bin_num,
        include_lowest = True, 
        labels = np.arange(bin_num))
        #labels = ['a','b','c','d','e','f'])

# Average firing by groups
mean_test_frame = test_frame.groupby(['neuron','taste','time_bin','trial'])\
                       .mean().reset_index()
# Chop trials into first and second half (they're sorted by gaping already)
mean_test_frame['trial_bin'] = pd.cut(mean_test_frame.trial,
        bins =2 ,include_lowest = True, 
        labels = np.arange(2))
        #labels = ['Low','High'])
# Drop time for clarity
mean_test_frame.drop('time',axis=1,inplace=True)

# Convert time_bin and trial_bin to integers
# Because pingouin has issues with categorical variables
mean_test_frame[['time_bin','trial_bin']] = \
        mean_test_frame[['time_bin','trial_bin']].astype('int')

# Plot firing rates for some neurons
g = sns.FacetGrid(test_frame.loc[test_frame.neuron < 10],\
        col = 'neuron', row = 'taste', \
        hue = 'trial_bin', sharey = False)
g = g.map(sns.pointplot, 'time_bin', 'firing_rate', ci = 'sd')\
        .add_legend()
plt.show()

# Perform ANOVA on trial_bins using pingouin
def trial_bin_anova(dframe):
    return [pg.anova(data = dframe.loc[dframe.taste==taste,:], 
            dv = 'firing_rate', \
            between = 'trial_bin') \
            for taste in dframe.taste.unique()]

trial_anova_list = Parallel(n_jobs = mp.cpu_count())\
    (delayed(trial_bin_anova)\
    (mean_test_frame.loc[mean_test_frame.neuron == nrn,:])\
            for nrn in tqdm(mean_test_frame.neuron.unique()))

# Extract p-value from anova list
trial_parray= np.asarray(\
        [[taste['p-unc'][0] if taste['p-unc'][0] > 0 else 1 \
        for taste in neuron] \
        for neuron in trial_anova_list])

alpha = 0.05
# JY said no need for correction
#corrected_alpha = alpha/len(test_frame.time_bin.unique())

# Dil Q only
dil_q_only = np.sum((trial_parray[:,2] < alpha) * (trial_parray[:,3] > alpha))
# Conc Q only
conc_q_only = np.sum((trial_parray[:,3] < alpha) * (trial_parray[:,2] > alpha))
# Both Q
both_q = np.sum(np.sum(trial_parray[:,2:] < alpha ,axis=-1) > 0)

# Mixed ANOVA on time-bins and trial-bins
def trial_time_anova(dframe):
    return [pg.mixed_anova(data = dframe.loc[dframe.taste==taste,:], 
                    dv = 'firing_rate', \
                    within = 'time_bin',
                    subject = 'trial',
                    between = 'trial_bin') \
                    for taste in dframe.taste.unique()]

trial_time_anova_list = [trial_time_anova(\
        mean_test_frame.loc[mean_test_frame.neuron == nrn,:]) \
        for nrn in tqdm(mean_test_frame.neuron.unique())]

# Extract p-value from anova list
trial_time_parray= np.asarray(\
        [[taste['p-unc'][0] if taste['p-unc'][0] > 0 else 1 \
        for taste in neuron] \
        for neuron in trial_time_anova_list])

# Dil Q only
dil_q_only = np.sum((trial_time_parray[:,2] < alpha) * (trial_time_parray[:,3] > alpha))
# Conc Q only
conc_q_only = np.sum((trial_time_parray[:,3] < alpha) * (trial_time_parray[:,2] > alpha))
# Both Q
both_q = np.sum(np.sum(trial_time_parray[:,2:] < alpha ,axis=-1) > 0)

# RM ANOVA
def trial_time_anova(dframe):
    return [pg.rm_anova(data = dframe.loc[dframe.taste==taste,:], 
                    dv = 'firing_rate', \
                    within = ['time_bin', 'trial_bin'],
                    subject = 'trial')
                    for taste in dframe.taste.unique()]

trial_time_anova_list = [trial_time_anova(\
        mean_test_frame.loc[mean_test_frame.neuron == nrn,:]) \
        for nrn in tqdm(mean_test_frame.neuron.unique())]

# Extract p-value from anova list
trial_time_parray= np.asarray(\
        [[taste['p-unc'][1] \
        for taste in neuron] \
        for neuron in trial_time_anova_list])

# Dil Q only
dil_q_only = np.sum((trial_time_parray[:,2] < alpha) * (trial_time_parray[:,3] > alpha))
# Conc Q only
conc_q_only = np.sum((trial_time_parray[:,3] < alpha) * (trial_time_parray[:,2] > alpha))
# Both Q
both_q = np.sum(np.sum(trial_time_parray[:,2:] < alpha ,axis=-1) > 0)

#this_frame = mean_test_frame.query('neuron == 0 and taste == 0')
#this_frame.drop(['neuron','taste'],axis=1,inplace=True)

#this_frame.rename(columns = {'trial':'Subject',
#                            'firing_rate':'Scores',
#                            'trial_bin':'Group',
#                            'time_bin':'Time'
#                           }, inplace=True)

#aov = mixed_anova(dv='Scores', between='Group',
#                   within='Time', subject='Subject', data=this_frame)

#aov = mixed_anova(dv = 'firing_rate', between = 'trial_bin',
#        within = 'time_bin', subject = 'trial', data = this_frame)

# ____  _       _       
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
                       

# (Seaborn) Plot firing rates for signifcantly different neurons 
this_frame = test_frame.loc[test_frame.neuron.isin(\
        np.where(np.sum(trial_parray[:,2:]<alpha,axis=-1))[0]),:]
this_frame = this_frame.loc[this_frame.taste >=2,:]
g = sns.FacetGrid(this_frame,\
        col = 'neuron', row = 'taste', \
        hue = 'trial_bin', sharey = False)
g = g.map(sns.pointplot, 'time_bin', 'firing_rate', ci = 'sd')\
        .add_legend()
plt.show()

# (Heatmap) Plot firing rates for signifcantly different neurons 
firing_overview(all_sorted_off_firing[2,np.where(trial_parray[:,2]<alpha)[0],:,\
        :(firing_window[1]//step_size)], cmap = 'viridis');plt.show()
firing_overview(all_sorted_off_firing[3,np.where(trial_parray[:,3]<alpha)[0],:,\
        :(firing_window[1]//step_size)], cmap = 'viridis');plt.show()

# Zscore plots
firing_overview(zscore(all_sorted_off_firing[2,np.where(trial_parray[:,2]<alpha)[0],:,\
        :(gaping_window[1]//step_size)]), cmap = 'viridis');plt.show()
firing_overview(zscore(all_sorted_off_firing[3,np.where(trial_parray[:,3]<alpha)[0],:,\
        :(gaping_window[1]//step_size)]), cmap = 'viridis');plt.show()

# (Seaborn) Plot firing rates for signifcantly different neurons 
# Tastes individually
this_frame = test_frame.loc[test_frame.neuron.isin(\
        np.where(trial_parray[:,2]<alpha)[0]),:]
this_frame = this_frame.loc[this_frame.taste == 2,:]
g = sns.FacetGrid(this_frame,\
        col = 'neuron', row = 'taste', \
        hue = 'trial_bin', sharey = False)
g = g.map(sns.pointplot, 'time_bin', 'firing_rate', ci = 'sd')\
        .add_legend()
plt.show()

this_frame = test_frame.loc[test_frame.neuron.isin(\
        np.where(trial_parray[:,3]<alpha)[0]),:]
this_frame = this_frame.loc[this_frame.taste == 3,:]
g = sns.FacetGrid(this_frame,\
        col = 'neuron', row = 'taste', \
        hue = 'trial_bin', sharey = False)
g = g.map(sns.pointplot, 'time_bin', 'firing_rate', ci = 'sd')\
        .add_legend()
plt.show()

 

# _____      _             
#| ____|_  _| |_ _ __ __ _ 
#|  _| \ \/ / __| '__/ _` |
#| |___ >  <| |_| | | (_| |
#|_____/_/\_\\__|_|  \__,_|
#                          

#ltp_array = hf5.root.ancillary_analysis.ltps[:]
#
#gapes_array_really_long = gapes_array.reshape(\
#        (np.prod(gapes_array.shape[:3]),gapes_array.shape[3]))
#
#ltp_array_long = ltp_array.reshape(\
#        (ltp_array.shape[0],np.prod(ltp_array.shape[1:3]),ltp_array.shape[-1]))
#ltp_array_long = ltp_array_long[:,:,2000:4500]
#
#fig, ax = plt.subplots(2,2)
#plt.sca(ax[0,0]);dat_imshow(gapes_array_long[0,:,:])
#plt.sca(ax[0,1]);dat_imshow(gapes_array_long[1,:,:])
#plt.sca(ax[1,0]);dat_imshow(ltp_array_long[0,:,:])
#plt.sca(ax[1,1]);dat_imshow(ltp_array_long[1,:,:])
#plt.show()
#
## Find bout length
#this_array = gapes_array_long[0,:,:]
#
#def calc_bout_duration(gapes_array):
#    gapes_array[gapes_array < 0.5] = 0
#    gapes_array[gapes_array>0] = 1
#    # Setting first and last index to 0 to ensure even numbers of markers
#    gapes_array[:,0] = 0
#    gapes_array[:,-1] = 0
#    marker_array = np.where(np.abs(np.diff(gapes_array,axis=-1)))
#    marker_list = [marker_array[1][marker_array[0]==trial] \
#            for trial in range(gapes_array.shape[0])]
#    bout_duration = [np.diff(np.split(trial,len(trial)//2)).flatten() \
#            for trial in marker_list]
#    return bout_duration
#
#taste_bout_durations = \
#        [np.concatenate(calc_bout_duration(taste)) for taste in gapes_array[0]]
#
#gapes_array_pca = pca(n_components = 10).fit_transform(gapes_array_long[0])
#
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(gapes_array_pca[:,0],gapes_array_pca[:,1],gapes_array_pca[:,2],c=labels)
#plt.show()
#
#import umap
#
#gapes_array_umap = umap.UMAP(n_components = 2).fit_transform(gapes_array_pca)
#plt.scatter(gapes_array_umap[:,0], gapes_array_umap[:,1],c=labels);plt.show()
#
#
#emg_bsa_results = np.asarray([x[:] for x in hf5.root.emg_BSA_results \
#        if 'taste' in str(x)])
