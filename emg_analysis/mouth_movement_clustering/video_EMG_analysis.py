#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 14:57:39 2023

@author: natasha
"""


import numpy as np
import tables
import glob
import os
import scipy.stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
#import pingouin as pg
from tqdm import tqdm, trange
import math
from scipy import signal
import scipy.stats as stats
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from tempfile import TemporaryFile
from matplotlib.colors import LinearSegmentedColormap


"""
######################################
IMPORTING DATA AND GETTING SETUP
######################################
"""
#functions to search through base folder
def find_test_day_folder(test_subject, test_day):
    base_folder = "/media/natasha/drive2/Natasha_Data"  # Path to drive where sorted data is stored

    test_subject_folder = os.path.join(base_folder, test_subject)
    test_day_folder = os.path.join(test_subject_folder, f"Test{test_day}")
    
    if os.path.exists(test_day_folder):
        return test_day_folder
    else:
        return None


def search_for_file(root_folder, file_extension):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                return file_path  # Return the file path if found

    return None  # Return None if file is not found

"""
INPUT RAT NAME, TEST DAY, AND DIG_INs HERE
"""
rat_name = "NB27"
test_day = 2
dig_in_numbers = [12, 13, 14, 15] #for taste only
taste_names = ['sucrose', 'nacl', 'citric acid', 'qhcl'] #IN SEQUENCE with DIG-INs

#Finding path to test day folder inside of extradrive (i.e. drive2)
dirname = find_test_day_folder(rat_name, test_day)
if dirname:
    print(f"dirname is: {dirname}")
else:
    print("Data folder not found.")


# === Setting up csv data from scored video into tables ===
#Searching and reading video scoring results in csv format
file_extension = '.csv' 
csv_path = search_for_file(dirname, file_extension)
scoring_data = pd.read_csv(csv_path)

#deleting superflous columns from csv files
columns_to_delete = ['Observation id', 'Observation date', 'Description', 'Observation duration', 'Observation type', 
                     'Source', 'Media duration (s)', 'FPS', 'date', 'test num', 'Subject', 
                     'Behavioral category', 'Media file name', 'Image index', 'Image file path', 'Comment']
for column in columns_to_delete:
    del scoring_data[column]
 
#seperating scoring table into two tables.
#trial_table is only all trial starts (120 rows)
#behavior_table is all scored behavior of rat
mask = scoring_data.iloc[:, 0] == 'trial start'
vid_trial_table = scoring_data[mask]
behavior_table = scoring_data[~mask]

## Saving scoring table as an .npy file to dirname
#os.chdir(dirname)
#temp_filename = f'{rat_name}_test{test_day}_scoring_table'
#np.save(temp_filename, scoring_data)

# === Importing EMG blech_clust results ===
#finding path to EMG data
file_extension = '.h5' 
h5_path = search_for_file(dirname, file_extension)
if not h5_path:
    print("Path to H5 file not found!")
h5 = tables.open_file(h5_path, 'r')    


'''
# === Creating table_with_trial_info ===
# Each row is trial # out of total (usually 120) ===
# and it tells you (taste #, presentation # of that taste) ===
#  Needed to match with video trials ===
'''

dig_in_index = [i for i in range(len(dig_in_numbers))]

trial_time_index = []

###this part takes a while to run
for number in tqdm(dig_in_numbers):
    dig_in = h5.get_node("/digital_in", f"dig_in_{number}")[:]
    indices = [i for i in range(1, len(dig_in)) if dig_in[i] == 1 and dig_in[i-1] == 0]
    trial_time_index.append(indices)

all_trial_indices = sorted(sum(trial_time_index, []))

table_with_trial_info = []
counters = [0] * len(dig_in_index)

for i in all_trial_indices:
    for j, valve_index in enumerate(trial_time_index):
        if i in valve_index:
            table_with_trial_info.append([dig_in_index[j], counters[j]])
            counters[j] += 1
            break
    else:
        print("Match not found between trial_time_index and all_trial_index")
   
    
#importing EMG data
emg_path = os.path.join(os.path.dirname(h5_path), "emg_output")

### DOUBLE CHECK THIS IS RIGHT FOR YOUR RAT
filt_ad = np.load(emg_path + '/emgAD/emg_filt.npy')
filt_sty = np.load(emg_path + '/emgSTYone/emg_filt.npy')
env_ad = np.load(emg_path + '/emgAD/emg_env.npy')
env_sty = np.load(emg_path + '/emgSTYone/emg_env.npy')


"""
######################################
Creating figure with 3 suplots
top subplot is scored behavior
middle is AD EMG, bottom is STY EMG
Across a few seconds of a given trial
######################################
NECESSARY PLOTTING INPUT HERE:
Input the single trial (first trial = 1)
and time pre-stimulus and post-stimulus delivery
"""

trial = 106 #trial num out of 120 with first trial =1
trial_pre_stim = 500 #original: 500
trial_post_stim = 5000 # original: 5000

trial_len = trial_pre_stim + trial_post_stim

# === Preparing behavior data for plotting ===
# determine start and end time of the trial based on video time
trial_start_time = vid_trial_table.iloc[trial-1, 4]
trial_end_time = trial_start_time + (trial_post_stim/1000) #determine end of trial in video time 
trial_start_time -= (trial_pre_stim/1000) #original: 0.5

#table of all behavior data within trial start and stop time
trial_behaviors = behavior_table[(behavior_table['Time'] >= trial_start_time) 
                                 & (behavior_table['Time'] <= trial_end_time)]

trial_behaviors = trial_behaviors[trial_behaviors['Behavior'] != 'out of view']


#Creating dictionary where key is all unique behaviors
unique_behaviors = list(set(behavior_table['Behavior']))
behaviors_dict = {i: None for i in unique_behaviors}

# loop through all behaviors in dict
# append time any time the behavior starts or stops
for index,row in trial_behaviors.iterrows(): 
    #converting video time to ephys time
    temp_time = [((row[4]-trial_start_time)/(trial_end_time - trial_start_time))*trial_len]
    if behaviors_dict[row[0]] == None:
        behaviors_dict[row[0]] = temp_time
    else:
        behaviors_dict[row[0]].extend(temp_time)

#converts the list within each index
# into a list of tuples containing pairs of consecutive values
# [(first_start_time, first_stop_time), (etc.)}]
for key, value in behaviors_dict.items():
    if isinstance(value, list):
        behaviors_dict[key] = [(value[i], value[i + 1]) 
                               for i in range(0, len(value) - 1, 2)]
#removing behavior "out of view"
#behaviors_dict.pop('out of view')

# Making list of behavior names and time intervals
# Needed for ease of plotting
behavior_names = list(behaviors_dict.keys())
time_intervals = list(behaviors_dict.values())



'''
# === Actually creating figure ===
'''
# Create a figure and axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, 
                                    figsize=(8,8), 
                                    sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1, 1]})

### Subplot 1 stuff (behavior)
# Set the y-axis ticks and labels
ax1.set_yticks(range(len(behavior_names)))
ax1.set_yticklabels(behavior_names)

# Define the linewidth of the bars
bar_linewidth = 25  # Adjust the value as needed

# Create a colormap with the number of behaviors
colors = cm.get_cmap('Accent', len(behavior_names))

# Iterate over the behavior names and corresponding time intervals
for i, intervals in enumerate(time_intervals):
    if intervals is not None:
        # Iterate over the time intervals and plot horizontal bars
        for interval in intervals:
            start_time, end_time = interval
            ax1.hlines(i, start_time, end_time, linewidth=bar_linewidth, color=colors(i))

### Subplot 2 + 3 stuff (EMG)
emg_trial = table_with_trial_info[trial-1][1]
emg_taste = table_with_trial_info[trial-1][0] #suc=0 ; nacl=1 ; ca=2 ; qhcl=3

# making xlims for EMG data based on input in section above
# EMG trial is 0>7000ms, 2000 pre-stim + 5000 post-stim
my_xlims = np.arange((-1*trial_pre_stim+2000), (trial_post_stim+2000))

#Putting color on subplot 2+3 that match with behavior intervals
for index, i in enumerate(time_intervals):
    if i is not None:
        for j in i:
            ax2.axvspan(j[0], j[1], color=colors(index), alpha=0.5)
            ax3.axvspan(j[0], j[1], color=colors(index), alpha=0.5)

#plotting AD and STY EMG
# #4285F4 = nice blue/ #DB4437 = nice red

emg_data_plot = [filt_ad, filt_sty]
ax2.plot(env_ad[emg_taste, emg_trial, my_xlims], '0.4', color='#DB4437')
ax2.set_ylabel('Envoleopped\nAnterior Digastric')

ax3.plot(env_sty[emg_taste, emg_trial, my_xlims], '0.4', color='#4285F4')
ax3.set_ylabel('Envelopped\nStyloglossus')


### Putting dashed line at trial delviery,
#as set in section above.
ax1.axvline(trial_pre_stim, linestyle='--', color='gray') #original = 500
ax2.axvline(trial_pre_stim, linestyle='--', color='gray')
ax3.axvline(trial_pre_stim, linestyle='--', color='gray')

# Set title
fig.suptitle(
    f'{rat_name}, Test Day {test_day}\nTrial {trial}: delivery#{emg_trial} of {taste_names[emg_taste]}',
    y = 0.95)

# Set x-axis label, and limits 
ax3.set_xlabel('Time (ms)')
ax1.set_ylim(-0.5, len(behavior_names) - 0.1)  # Adjust the limits as needed
#ax3.get_shared_y_axes().join(ax2, ax3) #AD and STY share same y-axis range
# Show the plot
plt.show()




'''
NEW PLOT IN PROGRESS
'''

'''
#figuring out like-trials 
to create heat map of intensity for pal/non-pal
'''

#table_with_trial_info
in_view_mask = vid_trial_table.iloc[:,2] == 'in view'
good_trial_table = vid_trial_table[in_view_mask]

suc_trials = [] 
nacl_trials = []
ca_trials = []
qhcl_trials = []
for index, row in good_trial_table.iterrows():
    modifier_value = row['Modifier #1']
    temp_tastant = table_with_trial_info[int(modifier_value)][0]
    if temp_tastant == 0:
        suc_trials.append(int(modifier_value))
    elif temp_tastant == 1:
        nacl_trials.append(int(modifier_value))
    elif temp_tastant == 2:
        ca_trials.append(int(modifier_value))
    elif temp_tastant == 3:
        qhcl_trials.append(int(modifier_value))

pal_trials = suc_trials + nacl_trials
unpal_trials = ca_trials + qhcl_trials

'''Input condition you want to plot here'''
plotting_condition = 'pal'


# Define the time bin parameters (adjust these as needed)
time_bin_width_ms = 100  # Width of each time bin in milliseconds
num_time_bins = int((trial_pre_stim + trial_post_stim) / time_bin_width_ms)

# Create a matrix to store behavior occurrences (initialize to zeros)
behavior_matrix = np.zeros((len(behavior_names), num_time_bins))

# Iterate through either unpalatable trials or palatable
for trial in tqdm(unpal_trials if plotting_condition == 'unpal' else suc_trials):
    trial_start_time = vid_trial_table.iloc[trial - 1, 4]
    trial_start_time -= (trial_pre_stim / 1000)

    trial_behaviors = behavior_table[(behavior_table['Time'] >= trial_start_time) & (behavior_table['Time'] <= trial_start_time + (trial_post_stim / 1000))]

    for _, row in trial_behaviors.iterrows():
        behavior_index = behavior_names.index(row['Behavior'])
        time_bin = int((row['Time'] - trial_start_time) / (time_bin_width_ms / 1000))
        behavior_matrix[behavior_index, time_bin] += 1

# Create a heatmap of behavior occurrences
plt.figure(figsize=(12, 8))
cax = plt.imshow(behavior_matrix, cmap='gray', aspect='auto', interpolation='none')
plt.colorbar(cax, label='Behavior Occurrences')
plt.xlabel('Time (ms)')
plt.ylabel('Behaviors')
#plt.title('Behavior Occurrences Across Trials')
plt.yticks(np.arange(len(behavior_names)), behavior_names)
plt.xticks(np.arange(0, num_time_bins, num_time_bins // 10), np.arange(-trial_pre_stim, trial_post_stim + 1, (trial_pre_stim + trial_post_stim) // 10))

if plotting_condition == 'unpal':
    title_suffix = f'Unpalatable tastants (total trials = {len(unpal_trials)})'
else:
    title_suffix = f'Palatable tastants (total trials = {len(pal_trials)})'

plt.title(
    f'{rat_name}, Test Day {test_day}\n{title_suffix}')



plt.show()

