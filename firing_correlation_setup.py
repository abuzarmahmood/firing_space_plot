
## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
from scipy.signal import spectrogram
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

########################################################
# Define functions and relevant variables to extract data
########################################################

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),os.path.basename(path_to_node))



# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)
log_file_name = os.path.join(data_folder, 'file_list.txt')

with tables.open_file(data_hdf5_path,'r+') as hf5:
    if '/firing' not in hf5:
        hf5.create_group('/','firing')
        hf5.flush()

# Ask user for all relevant files
# If file_list already exists then ask the user if they want to use it
if os.path.exists(log_file_name):
    old_list = open(log_file_name,'r').readlines()
    old_list = [x.rstrip() for x in old_list]
    old_basename_list = [os.path.basename(x) for x in old_list]
    old_bool = easygui.ynbox(msg = "Should this file list be used: \n\n Old list: \n\n{}"\
            .format("\n".join(old_basename_list), 
            title = "Save these files?"))

if old_bool:
    file_list = old_list
else:
    print('No other options at this point')

# Break down filename into parts so they can be used for naming nodes
file_name_parts = [os.path.basename(x).split('_') for x in file_list]
animal_name_list = [x[0] for x in file_name_parts]
date_str_list = [x[2] for x in file_name_parts]

# Check output is correct, if not, ask user to define names
for file_num in range(len(file_list)):
    node_check = 'a'
    while node_check not in ['y','n']:
        question_str = \
                'Please confirm (y/n): \nNode:\t{}\nDate:\t{}\n:::'\
                .format(animal_name_list[file_num], date_str_list[file_num])
        node_check = input(question_str)
    if node_check == 'n':
        animal_name_list[file_num], date_str_list[file_num] = \
                easygui.multenterbox('Please enter names for nodes in HF5'\
                        '\n{}'.format(os.path.basename(file_list[file_num])),
                            'Enter node names',['Animal Name','Date'])

for file_num in range(len(file_list)):
    animal_name = animal_name_list[file_num] 
    date_str = date_str_list[file_num]

    with tables.open_file(data_hdf5_path,'r+') as hf5:
        if '/firing/{}'.format(animal_name) not in hf5:
            hf5.create_group('/firing',animal_name)
        if '/firing/{}/{}'.format(animal_name,date_str) not in hf5:
            hf5.create_group('/firing/{}'.format(animal_name),date_str)
        hf5.flush()

    dat = ephys_data(os.path.dirname(file_list[file_num]))
    dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                        (25,250,1)))
    dat.get_unit_descriptors()
    dat.get_spikes()
    dat.get_firing_rates()

    # Write arrays to data HF5
    with tables.open_file(data_hdf5_path,'r+') as hf5:
        # Write STFT axis values (frequencies and time) under STFT node
        # This is to signify that all STFT arrays have the same size

        # If arrays already present then remove them and rewrite
        remove_node('/firing/{}/{}/{}'.format(animal_name, date_str, 'spike_array'),hf5) 
        remove_node('/firing/{}/{}/{}'.format(animal_name, date_str, 'firing_array'),hf5) 
        remove_node('/firing/{}/{}/{}'.format(animal_name, date_str, 'normalized_firing_array'),hf5) 
        remove_node('/firing/{}/{}/{}'.format(animal_name, date_str, 'unit_electrode'),hf5) 

        hf5.create_array('/firing/{}/{}'.format(animal_name,date_str),
                'spike_array',np.array(dat.spikes))
        hf5.create_array('/firing/{}/{}'.format(animal_name,date_str),
                'firing_array', dat.firing_array)
        hf5.create_array('/firing/{}/{}'.format(animal_name,date_str),
                'normalized_firing_array', dat.normalized_firing)
        hf5.create_array('/firing/{}/{}'.format(animal_name,date_str),
                'unit_electrode', np.array([x[0] for x in dat.unit_descriptors]))
        hf5.flush()

