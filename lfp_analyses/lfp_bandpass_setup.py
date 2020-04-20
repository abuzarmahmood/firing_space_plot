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

# Define functions to streamline sequence of processing
def img_plot(array):
    plt.imshow(array, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

def get_lfp_channels(hdf5_name):
    with tables.open_file(hdf5_name,'r') as hf5:
        parsed_lfp_channels = hf5.root.Parsed_LFP_channels[:]
    return parsed_lfp_channels

# Calculate absolute and phase
def parallelize(func, iterator):
    return Parallel(n_jobs = mp.cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),os.path.basename(path_to_node))

# Define filtering functions
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


# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/fastdata/lfp_analyses'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)
log_file_name = os.path.join(data_folder, 'file_list.txt')

# Create Phase node in data hdf5
#if not os.path.exists(data_hdf5_path):
#    hf5 = tables.open_file(data_hdf5_path,'w')
#    hf5.flush()
#    hf5.close()

with tables.open_file(data_hdf5_path,'r+') as hf5:
    if '/bandpass_lfp' not in hf5:
        hf5.create_group('/','bandpass_lfp')
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
    old_list = ''
    file_list = []
    last_dir = None
    while True:
        if last_dir is not None:
            file_name = easygui.fileopenbox(msg = 'Please select files to extract'\
                    ' data from, CANCEL to stop', default = last_dir)
        else:
            file_name = easygui.fileopenbox(msg = 'Please select files to extract'\
                    ' data from, CANCEL to stop')
        if file_name is not None:
            file_list.append(file_name)
            last_dir = os.path.dirname(file_name)
        else:
            break

    file_basename_list = [os.path.basename(x) for x in file_list]
    save_bool = easygui.ynbox(msg = 'Should this list of files be saved: '\
            '\n\nOld list:\n\n{}\n\nNew list:\n\n{}\n\n'\
            'Files will be saved to : {}'\
            .format("\n".join(old_basename_list),"\n".join(file_basename_list),log_file_name), 
            title = "Save these files?")

    if save_bool:
        open(log_file_name,'w').writelines("\n".join(file_list))

# Break down filename into parts so they can be used for naming nodes
file_name_parts = [os.path.basename(x).split('_') for x in file_list]
animal_name_list = [x[0] for x in file_name_parts]
date_str_list = [x[2] for x in file_name_parts]

# Check output is correct, if not, ask user to define names
parts_check_str = 'Node\tDate\n\n' + \
        "\n".join(['{}\t{}'.format(animal,date) \
        for animal, date in zip(animal_name_list,date_str_list)])
choice_list = ['{} | {}'.format(animal,date) \
        for animal, date in zip(animal_name_list,date_str_list)]
node_check_choices = easygui.multchoicebox(msg = 'Please select all correct answers',
        choices = choice_list) 
unselected_choices = \
        [num  for num,x in enumerate(choice_list) if x not in node_check_choices]

if len(unselected_choices) > 0:
    for file_num in unselected_choices:
        animal_name_list[file_num], date_str_list[file_num] = \
                easygui.multenterbox('Please enter names for nodes in HF5'\
                        '\n{}'.format(os.path.basename(file_list[file_num])),
                            'Enter node names',['Animal Name','Date'])

band_freqs = [(1,4),
                (4,7),
                (7,12),
                (12,25)]

with tables.open_file(data_hdf5_path,'r+') as hf5:
    if '/bandpass_lfp/frequency_bands' in hf5:
        remove_node('/bandpass_lfp/frequency_bands',hf5) 
    hf5.create_array('/bandpass_lfp', 'frequency_bands', np.array(band_freqs))

# Extract data
for file_num in tqdm(range(len(file_list))):
    animal_name = animal_name_list[file_num] 
    date_str = date_str_list[file_num]

    with tables.open_file(data_hdf5_path,'r+') as hf5:
        if '/bandpass_lfp/{}'.format(animal_name) not in hf5:
            hf5.create_group('/bandpass_lfp',animal_name)
        if '/bandpass_lfp/{}/{}'.format(animal_name,date_str) not in hf5:
            hf5.create_group('/bandpass_lfp/{}'.format(animal_name),date_str)
        hf5.flush()

    dat = ephys_data(os.path.dirname(file_list[file_num]))
    dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                        (25,250,1)))
    dat.extract_and_process()

    Fs = 1000 

    # Bandpass lfp
    lfp_bandpassed = np.asarray([
                        butter_bandpass_filter(
                            data = dat.lfp_array, 
                            lowcut = band[0],
                            highcut = band[1],
                            fs = Fs) \
                                    for band in tqdm(band_freqs)])

    hilbert_bandpassed_lfp = hilbert(lfp_bandpassed)
    phase_array = np.angle(hilbert_bandpassed_lfp)
    amplitude_array = np.abs(hilbert_bandpassed_lfp)

    # Write arrays to data HF5
    with tables.open_file(data_hdf5_path,'r+') as hf5:
        # Write STFT axis values (frequencies and time) under STFT node
        # This is to signify that all STFT arrays have the same size

        # If arrays already present then remove them and rewrite
        remove_node('/bandpass_lfp/{}/{}/{}'.\
                format(animal_name, date_str, 'bandpassed_lfp_array'),hf5) 
        remove_node('/bandpass_lfp/{}/{}/{}'.\
                format(animal_name, date_str, 'hilbert_array'),hf5) 
        remove_node('/bandpass_lfp/{}/{}/{}'.\
                format(animal_name, date_str, 'phase_array'),hf5) 
        remove_node('/bandpass_lfp/{}/{}/{}'.\
                format(animal_name, date_str, 'amplitude_array'),hf5) 

        hf5.create_array('/bandpass_lfp/{}/{}'.format(animal_name,date_str),
                'bandpassed_lfp_array', lfp_bandpassed)
        hf5.create_array('/bandpass_lfp/{}/{}'.format(animal_name,date_str),
                'hilbert_array', hilbert_bandpassed_lfp)
        hf5.create_array('/bandpass_lfp/{}/{}'.format(animal_name,date_str),
                'phase_array',phase_array)
        hf5.create_array('/bandpass_lfp/{}/{}'.format(animal_name,date_str),
                'amplitude_array', amplitude_array)
        hf5.flush()

