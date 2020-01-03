"""
Script to load all files from Brad's paradigm for a single 
session and perform the following tests/visualisations:
    1) Are there differences in taste ITI LFPs between tastes
        for different bands
    2) Do changes in sickness persists into the taste delivery period
        and are they detectable in the LFP during the ITI period
    3) What are the dynamics of the LFP during the ITI periods
        and if sickness is noticeable in the LFP, does it rebound
        towards health

Precise analyses and outputs:
    1) 
        - 2 Way ANOVA on ITI LFP Power by taste and time for every band
        - FacetGrid heatmap output (taste x band) for each ITI LFP power
    2)
        - Bandwise timeseries plot of power in affective and whole taste periods
        - Heatmap for power for Saline and LiCl sessions
        - Difference in ITI LFP power for taste after Saline or LiCl 
    3) 
    *)  Items saved to HDF5 file for animal
        - Firing rate output for all 5 recordings
        - Bandwise LFP power output for all 5 recordings
        -
"""

# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   

## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import h5py
#import easygui
import scipy
from scipy.signal import spectrogram
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from itertools import product
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import zscore
import glob
from collections import namedtuple
from scipy.signal import convolve


##############
# Define functions to extract data
#############

def get_spikes(hdf5_name):
    """
    Extract spike arrays from specified HD5 files
    """
    with tables.open_file(hdf5_name, 'r+') as hf5: 
    
        # Iterate through tastes and extract spikes from laser on and off conditions
        # If no array for laser durations, put everything in laser off
        
        dig_in_list = \
            [x for x in hf5.list_nodes('/spike_trains') if 'dig_in' in x.__str__()]
        
        spikes = np.asarray([dig_in.spike_array[:] for dig_in in dig_in_list])
        return spikes

def calc_normalized_firing(spike_array,kern_length):
    """
    spike_array :: N-D array with time as last dim
    """
    kern_shape = (*np.ones(len(spike_array.shape)-1),kern_length)
    kern_shape = [int(x) for x in kern_shape]
    firing_rates = convolve(spike_array, np.ones(kern_shape))
    return firing_rates

def get_parsed_lfp(hdf5_name):
    """
    Extract parsed lfp arrays from specified HD5 files
    """
    with tables.open_file(hdf5_name, 'r+') as hf5: 
        if 'Parsed_LFP' in hf5.list_nodes('/').__str__():
            lfp_nodes = [node for node in hf5.list_nodes('/Parsed_LFP')\
                    if 'dig_in' in node.__str__()]
            lfp_array = np.asarray([node[:] for node in lfp_nodes])
            all_lfp_array = \
                    lfp_array.\
                        swapaxes(1,2).\
                        reshape(-1, lfp_array.shape[1],\
                                lfp_array.shape[-1]).\
                        swapaxes(0,1)
        else:
            raise Exception('Parsed_LFP node absent in HDF5')
    return all_lfp_array


#def get_parsed_lfp_h5py(hdf5_name):
#    """
#    Extract parsed lfp arrays from specified HD5 files
#    """
#    with h5py.File(hdf5_name, 'r') as hf5: 
#        if 'Parsed_LFP' in hf5.keys():
#            lfp_nodes = [node for node in hf5['Parsed_LFP']\
#                    if 'dig_in' in node]
#            lfp_array = np.asarray([hf5['Parsed_LFP'][node] for node in lfp_nodes])
#            all_lfp_array = \
#                    lfp_array.\
#                        swapaxes(1,2).\
#                        reshape(-1, lfp_array.shape[1],\
#                                lfp_array.shape[-1]).\
#                        swapaxes(0,1)
#        else:
#            raise Exception('Parsed_LFP node absent in HDF5')
#    return all_lfp_array

def get_whole_session_lfp(hdf5_name):
    with tables.open_file(hdf5_name, 'r+') as hf5: 
        whole_session_lfp_node = hf5.list_nodes('/Whole_session_raw_LFP') 
        whole_lfp = whole_session_lfp_node[0][:]
    return whole_lfp

#def get_whole_session_lfp_h5py(hdf5_name):
#    with h5py.File(hdf5_name, 'r') as hf5: 
#        whole_session_lfp_node = list(hf5['Whole_session_raw_LFP'].keys())
#        whole_lfp = hf5['Whole_session_raw_LFP'][whole_session_lfp_node[0]].value
#    return whole_lfp

def get_delivery_times(hdf5_name):
    delivery_times = \
            pd.read_hdf(hdf5_name,'/Whole_session_spikes/delivery_times')
    delivery_times['taste'] = delivery_times.index
    delivery_times = \
            pd.melt(delivery_times,
                    id_vars = 'taste',
                    var_name ='trial',
                    value_name='delivery_time')
    delivery_times.sort_values(by='delivery_time',inplace=True)
    # Delivery times are in 30kHz samples, convert to ms
    delivery_times['delivery_time'] = delivery_times['delivery_time'] // 30
    delivery_times['chronological'] = np.argsort(delivery_times.delivery_time)
    return delivery_times


# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#

# All HDF5 files need to be in the same folder
# Load files and make sure the order is right
dir_name = '/media/bigdata/brads_data/Brad_LFP_ITI_analyses/BS23'

final_confirmation = 'n'
while 'y' not in final_confirmation:
    hdf5_name = glob.glob(
            os.path.join(dir_name, '**.h5'))
    selection_list = ['{}) {} \n'.format(num,os.path.basename(file)) \
            for num,file in enumerate(hdf5_name)]
    selection_string = 'Please enter the number of the HDF5 files in the following'\
                ' order \n (as a comma separated string e.g. 2,1,3,4,0):'\
                '\n Day1 Saline \n Day1 LiCl \n Day1 Taste \n' \
                ' Day3 Saline \n Day3 Taste:\n{}'.\
                    format("".join(selection_list))
    file_selection = input(selection_string)
    file_order = [int(x) for x in file_selection.split(',')]
    # Check with user that the order is right
    final_list = [hdf5_name[x] for x in file_order]
    exp_list = ['Day1 Saline','Day1 LiCl','Day1 Taste',
                'Day3 Saline', 'Day3 Taste']
    final_selection_list = ['{})\t{}\n'.format(exp,os.path.basename(file)) \
            for exp,file in zip(exp_list,final_list)]
    final_selection_string = 'Is this order correct (y/n): \n{}'.\
            format("".join(final_selection_list))
    final_confirmation = input(final_selection_string)

affective_recordings = [0,1,3]
taste_files = [final_list[2], final_list[-1]]

# Pull in spike trains from all sessions to calculate firing rates
# Replace list with

#spike_tuple = namedtuple('SpikeTrains',['filename','spiketrains'])
#
#all_spike_trains = [spike_tuple(file_name, get_spikes(file_name)) \
#        for file_name in tqdm(final_list)]
#
## Elements 0,1,3 will be from the affective recording 
## So they're shaped a little weirdly
#fin_spike_trains = [spike_train.spiketrains[0,0] \
#        if recording_num in affective_recordings \
#        else spike_train.spiketrains \
#        for recording_num, spike_train in enumerate(all_spike_trains)]
#
## Calculate firing rates for all spike_trains
#fin_firing_rates = [calc_normalized_firing(x,250) for x in tqdm(fin_spike_trains)]
#
## Downsample spiketrains and delete originals
#down_ratio = 25
#down_firing_rate = [x[..., np.arange(0,x.shape[-1],down_ratio)] \
#        for x in fin_firing_rates]
#
#del fin_firing_rates

# Extract LFP from all sessions
whole_lfp = [np.squeeze(get_parsed_lfp(file_name))
        if recording_num in affective_recordings \
        else get_whole_session_lfp(file_name) \
        for recording_num, file_name in tqdm(enumerate(final_list))]

taste_whole_lfp = [whole_lfp[2],whole_lfp[-1]]
affective_whole_lfp = [whole_lfp[x] for x in [0,1,3]]

# Extract ITI's from taste sessions
trial_time_data = [get_delivery_times(file_name) \
        for file_name in taste_files]
delivery_time_list = [x.delivery_time for x in trial_time_data]

# Define parameters to extract ITI data
time_before_delivery = 10 #seconds
padding = 1 #second before taste delivery won't be extracted
Fs = 1000 # Sampling frequency

# (trials x channels x time)
iti_array_list = np.asarray(\
        [[lfp_array[:,(x-(time_before_delivery*Fs)):(x-(padding*Fs))]\
        for x in delivery_times] \
        for delivery_times,lfp_array in \
        zip(delivery_time_list,taste_whole_lfp)])

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

# Bandpass filter lfp into relevant bands

#define bandpass filter parameters to parse out frequencies
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

band_freqs = [(1,4),
                (4,7),
                (7,12),
                (12,25),
                (25,50)]


affective_whole_bandpassed = \
            [np.asarray([
                    butter_bandpass_filter(
                        data = data, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])\
                for data in affective_whole_lfp]

taste_whole_bandpassed  = \
            [np.asarray([
                    butter_bandpass_filter(
                        data = data, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])\
                for data in taste_whole_lfp]

iti_lfp_bandpassed  = \
            [np.asarray([
                    butter_bandpass_filter(
                        data = data, 
                        lowcut = band[0],
                        highcut = band[1],
                        fs = Fs) \
                                for band in tqdm(band_freqs)])\
                for data in iti_array_list]
    
# Remove to preserve memory
del affective_whole_lfp, taste_whole_lfp, iti_array_list, whole_lfp

# Calculate Hilbert and amplitude
affective_whole_hilbert = [hilbert(data) for data in tqdm(affective_whole_bandpassed)]
#taste_whole_hilbert = [hilbert(data) for data in tqdm(taste_whole_bandpassed)]
iti_lfp_hilbert = [hilbert(data) for data in tqdm(iti_lfp_bandpassed)]

affective_lfp_amplitude = [np.abs(data) for data in tqdm(affective_whole_hilbert)]
#taste_lfp_amplitude = [np.abs(data) for data in taste_whole_hilbert]
iti_lfp_amplitude = [np.abs(data) for data in tqdm(iti_lfp_hilbert)]


#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           

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

# Take average of power across channels
mean_iti_lfp_amplitude = [np.mean(data,axis=2) for data in iti_lfp_amplitude]

# Check for outliers
zscore_trials_power = np.asarray([\
        [zscore(band,axis = None) for band in data] for data in mean_iti_lfp_amplitude])
mean_trials_power = [np.mean(data,axis=-1) for data in zscore_trials_power]

# Plot zscore trial averaged power
fig, ax = plt.subplots(len(mean_trials_power),1,sharex=True)
for num, this_ax in enumerate(ax):
   this_ax.plot(mean_trials_power[num].T,'x')
   this_ax.title.set_text(os.path.basename(taste_files[num]))
   this_ax.set_xlabel('Trial num');this_ax.set_ylabel('Zscore mean power')
plt.tight_layout()
plt.show()

# Ask user to input threshold zscore to remove bad trials from ANOVA analysis
file_str = ("".join(['{}) {} \n'.format(file_num,os.path.basename(file_name)) \
        for file_num,file_name in enumerate(taste_files)]))

thresh_string = 'Please enter threshold zscore for the following files'\
        '\n(separated by commas e.g. 0.6,0.3):\n{}'.format(file_str) 
user_check = 'n'
while 'y' not in user_check:
    try:
        bad_trial_threshes = [float(x) for x in input(thresh_string).split(',')]
        user_check = input('Please confirm (y/n): {}\n'.format(bad_trial_threshes))
    except:
        raise Exception('Please check the formatting of your string')

bad_trials = [np.unique(np.where(mean_trials_power[num] > bad_trial_threshes[num])[1]) \
            for num in range(len(bad_trial_threshes))]

# Replot trials with threshold and marked trials
# Plot zscore trial averaged power
fig, ax = plt.subplots(len(mean_trials_power),1,sharex=True)
for num, this_ax in enumerate(ax):
   this_ax.plot(mean_trials_power[num].T,'x')
   this_ax.plot(np.arange(mean_trials_power[num].shape[-1])[bad_trials[num]],
                mean_trials_power[num][:,bad_trials[num]].T,'o',c='r')
   this_ax.hlines(bad_trial_threshes[num],0,mean_trials_power[num].shape[-1],color='r')
   this_ax.title.set_text('Thresh {} , {}'.\
           format(bad_trial_threshes[num],os.path.basename(taste_files[num])))
   this_ax.set_xlabel('Trial num');this_ax.set_ylabel('Zscore mean power')
plt.tight_layout()
plt.show()

# Plot Average power for (taste x band)
taste_band_power = [np.asarray([\
        data[:,trial_time_data[num].taste == taste,:] \
        for taste in np.sort(trial_time_data[num].taste.unique())])\
        for num,data in enumerate(mean_iti_lfp_amplitude)]

# Set points with values > 3*std as masked
masked_taste_band_power = [np.asarray(\
        [np.ma.masked_greater(data[:,band],
            np.mean(data[:,band],axis=None)+3*\
                    np.std(data[:,band],axis=None))\
        for band in range(data.shape[1])]).swapaxes(0,1) \
        for data in taste_band_power]

zscore_taste_band_power = [np.asarray(\
        [zscore(data[:,band],axis=None) \
        for band in range(data.shape[1])]).swapaxes(0,1)\
        for data in masked_taste_band_power]

for num,data in enumerate(zscore_taste_band_power):
    fig,ax = plt.subplots(data.shape[0],
                            data.shape[1])
    for taste in range(data.shape[0]):
        for band in range(data.shape[1]):
            plt.sca(ax[taste,band])
            im = plt.imshow(data[taste,band],
                    interpolation='nearest',aspect='auto',
                    cmap = 'viridis',vmin=-1,vmax=3)
            im.cmap.set_over('k')
    plt.suptitle(os.path.basename(taste_files[num]))
    fig.subplots_adjust(bottom = 0.2)
    cbar_ax = fig.add_axes([0.15,0.1,0.7,0.02])
    plt.colorbar(im, cax = cbar_ax,orientation = 'horizontal', pad = 0.2, extend='max')
plt.show()

#    _    _   _  _____     ___    
#   / \  | \ | |/ _ \ \   / / \   
#  / _ \ |  \| | | | \ \ / / _ \  
# / ___ \| |\  | |_| |\ V / ___ \ 
#/_/   \_\_| \_|\___/  \_/_/   \_\
#                                 

# Convert power from spectrogram to DataFrame
# Average across entire trial
mean_zscore_taste_band_power = [np.mean(data,axis=-1) for axis in zscore_taste_band_power]
nd_idx = [make_array_identifiers(data) for data in mean_zscore_taste_band_power]
mean_band_df = [pd.DataFrame({\
        'taste' : nd_idx[num][0],
        'band' : nd_idx[num][1],
        'trial' : nd_idx[num][2],
        'power' : data.flatten()}) \
                for num,data in enumerate(mean_zscore_taste_band_power)]

mean_band_df = [mean_band_df[num].merge(trial_time_data[num],'inner') \
                    for num in range(len(mean_band_df))]

# Remove "bad trials"
mean_band_df = [mean_band_df[num].loc[~mean_band_df[num].chronological.isin(bad_trials[num])] \
        for num in range(len(mean_band_df))]

mean_band_df[1].to_pickle('test_anova_frame.pkl')

# Find average power across ITI interval
#taste_band_avg_power = np.mean(zscore_taste_band_spectrograms,axis=-1)
#nd_idx_objs = make_array_identifiers(taste_band_avg_power)
#mean_band_df = pd.DataFrame({\
#                        'taste' : nd_idx_objs[0],
#                        'band' : nd_idx_objs[1],
#                        'trial' : nd_idx_objs[2],
#                        'power' : taste_band_avg_power.flatten()})


#mean_trial_df = [mean_band_df[num].\
#                        groupby(['chronological','band']).\
#                            aggregate('mean').\
#                                reset_index() \
#                                for num in range(len(mean_band_df))]
#
#mean_trial_df = mean_trial_df.merge(delivery_times[['chronological','trial']],'inner')

########################################
# Something wrong with trial removal
########################################

# Cluster trials into bins for anova
trial_bin_num = 5
for dat in mean_band_df:
    dat['trial_bin'] = pd.cut(dat.trial,
            bins = trial_bin_num, include_lowest = True, labels = range(trial_bin_num))

#mean_trial_df['trial_bin'] = mean_trial_df['trial_bin'].astype(np.dtype('int16')) 
#mean_trial_df = mean_trial_df.astype('float')

# Plot dataframe to visualize
for dat in mean_band_df:
    g = sns.FacetGrid(data = \
                dat, col = 'band', hue = 'taste', sharey=False)
    g.map(sns.pointplot, 'trial_bin','power',ci='sd').add_legend()
plt.show()

# Perform 2-way ANOVA to look at differences in taste and trial_bin
taste_trial_anova = [\
    [dat.loc[dat.band == band_num].anova(dv = 'power', \
        between= ['trial_bin','taste'])[['Source','p-unc','np2']][:3] \
            for band_num in np.sort(dat.band.unique())] \
        for dat in mean_band_df]

taste_trial_anova_df = [ [\
        pd.DataFrame({  'File' : os.path.basename(taste_files[file_num]),
                        'Band' : band_num,
                        'Source' : band['Source'],
                        'p-unc' : band['p-unc'],
                        'np2' : band['np2']})
            for band_num,band in enumerate(file)]\
            for file_num, file in enumerate(taste_trial_anova)]

from itertools import chain
taste_trial_anova_df = pd.concat(list(chain(*taste_trial_anova_df)))
