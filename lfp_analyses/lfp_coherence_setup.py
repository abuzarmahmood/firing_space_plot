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
def get_lfp_channels(hdf5_name):
    with tables.open_file(hdf5_name,'r') as hf5:
        parsed_lfp_channels = hf5.root.Parsed_LFP_channels[:]
    return parsed_lfp_channels

# Define function to parse out only wanted frequencies in STFT
def calc_stft(trial, max_freq,time_range_tuple,
        Fs,signal_window,window_overlap):
    """
    trial : 1D array
    max_freq : where to lob off the transform
    time_range_tuple : (start,end) in seconds
    """
    f,t,this_stft = scipy.signal.stft(
                scipy.signal.detrend(trial), 
                fs=Fs, 
                window='hanning', 
                nperseg=signal_window, 
                noverlap=signal_window-(signal_window-window_overlap)) 
    this_stft =  this_stft[np.where(f<max_freq)[0]]
    this_stft = this_stft[:,np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))[0]]
    return f[f<max_freq],t[np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))],this_stft
                        
# Calculate absolute and phase
def parallelize(func, iterator):
    return Parallel(n_jobs = mp.cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

# Convert list to array
def convert_to_array(iterator, iter_inds):
    temp_array  =\
            np.empty(tuple((*(np.max(np.array(stft_iters),axis=0) + 1),*iterator[0].shape)),
                    dtype=np.dtype(iterator[0].flatten()[0]))
    for iter_num, this_iter in tqdm(enumerate(iter_inds)):
        temp_array[this_iter] = iterator[iter_num]
    return temp_array

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),os.path.basename(path_to_node))

def normalize_timeseries(array, time_vec, stim_time):
    mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    array = array/mean_baseline
    # Recalculate baseline
    #mean_baseline = np.mean(array[:,time_vec<stim_time],axis=1)[:,np.newaxis]
    #array -= mean_baseline
    return array

# Define middle channels in board
middle_channels = np.arange(8,24)

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

# Create Phase node in data hdf5
if not os.path.exists(data_hdf5_path):
    hf5 = tables.open_file(data_hdf5_path,'w')
    hf5.flush()
    hf5.close()

with tables.open_file(data_hdf5_path,'r+') as hf5:
    if '/stft' not in hf5:
        hf5.create_group('/','stft')
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

# Ask user to select files to perform anaysis on
selected_files = easygui.multchoicebox(msg = 'Please select files to run analysis on',
    choices = ['{}) '.format(num)+os.path.basename(x) for num,x in enumerate(file_list)])
file_list = [file_list[int(x.split(')')[0])] for x in selected_files]

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

# Extract data
for file_num in range(len(file_list)):
    animal_name = animal_name_list[file_num] 
    date_str = date_str_list[file_num]

    with tables.open_file(data_hdf5_path,'r+') as hf5:
        if '/stft/{}'.format(animal_name) not in hf5:
            hf5.create_group('/stft',animal_name)
        if '/stft/{}/{}'.format(animal_name,date_str) not in hf5:
            hf5.create_group('/stft/{}'.format(animal_name),date_str)
        hf5.flush()

    dat = ephys_data(os.path.dirname(file_list[file_num]))
    dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                        (25,250,1)))
    dat.extract_and_process()

    # Extract channel numbers for lfp
    with tables.open_file(dat.hdf5_name,'r') as hf5:
        parsed_lfp_channels = hf5.root.Parsed_LFP_channels[:]

    middle_channels_bool = np.array([True if channel in middle_channels else False \
            for channel in parsed_lfp_channels ])

    # ____ _____ _____ _____ 
    #/ ___|_   _|  ___|_   _|
    #\___ \ | | | |_    | |  
    # ___) || | |  _|   | |  
    #|____/ |_| |_|     |_|  

    # Resolution has to be increased for phase of higher frequencies
    Fs = 1000 
    signal_window = 500 
    window_overlap = 499
    max_freq = 20 
    time_range_tuple = (0,5)

    # Generate list of individual trials to be fed into STFT function
    stft_iters = list(product(*list(map(np.arange,dat.lfp_array.shape[:3]))))

    # Calculate STFT over lfp array
    stft_list = Parallel(n_jobs = mp.cpu_count()-2)\
            (delayed(calc_stft)(dat.lfp_array[this_iter],max_freq,time_range_tuple,
                                Fs,signal_window,window_overlap)\
            for this_iter in tqdm(stft_iters))

    freq_vec = stft_list[0][0]
    time_vec = stft_list[0][1]
    fin_stft_list = [x[-1] for x in stft_list]
    del stft_list
    amplitude_list = parallelize(np.abs, fin_stft_list)
    phase_list = parallelize(np.angle, fin_stft_list)

    # (taste, channel, trial, frequencies, time)
    stft_array = convert_to_array(fin_stft_list, stft_iters)
    del fin_stft_list
    amplitude_array = convert_to_array(amplitude_list, stft_iters)**2
    del amplitude_list
    phase_array = convert_to_array(phase_list, stft_iters)
    del phase_list

    # Write arrays to data HF5
    with tables.open_file(data_hdf5_path,'r+') as hf5:
        # Write STFT axis values (frequencies and time) under STFT node
        # This is to signify that all STFT arrays have the same size

        # If arrays already present then remove them and rewrite
        remove_node('/stft/freq_vec',hf5) 
        remove_node('/stft/time_vec',hf5) 
        remove_node('/stft/{}/{}/{}'.format(animal_name, date_str, 'stft_array'),hf5) 
        remove_node('/stft/{}/{}/{}'.format(animal_name, date_str, 'amplitude_array'),hf5) 
        remove_node('/stft/{}/{}/{}'.format(animal_name, date_str, 'phase_array'),hf5) 
        remove_node('/stft/{}/{}/{}'.format(animal_name, date_str, 'parsed_lfp_channels'),hf5) 

        hf5.create_array('/stft','freq_vec',freq_vec)
        hf5.create_array('/stft','time_vec',time_vec)
        hf5.create_array('/stft/{}/{}'.format(animal_name,date_str),'stft_array',stft_array)
        hf5.create_array('/stft/{}/{}'.format(animal_name,date_str),
                'amplitude_array',amplitude_array)
        hf5.create_array('/stft/{}/{}'.format(animal_name,date_str),'phase_array',phase_array)
        hf5.create_array('/stft/{}/{}'.format(animal_name,date_str),
                'parsed_lfp_channels',parsed_lfp_channels)
        hf5.flush()

    del stft_array
        
    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    #                       

    plot_dir = os.path.join(data_folder,animal_name,date_str)
    # If directory exists, delete and remake. Otherwise just remake
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

    # Plot firing rates for all neurons
    region_label = [1 if any(x[0] == middle_channels) else 0 for x in dat.unit_descriptors]
    dat.firing_overview(dat.all_normalized_firing)#,subplot_labels = region_label);
    fig = plt.gcf()
    #for ax in fig.get_axes():
    #    ax.axis('off')
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'firing_rate_overview.png'))

    # Plot raw LFP for all channels
    # Calculate clims
    #mean_val = np.mean(dat.all_lfp_array, axis = None)
    #sd_val = np.std(dat.all_lfp_array, axis = None)
    dat.firing_overview(dat.all_lfp_array, 
                        cmap = 'viridis',
                        subplot_labels = middle_channels_bool, zscore_bool = True)
    fig = plt.gcf()
    #for ax in fig.get_axes():
    #    ax.axis('off')
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'raw_lfp_overview.png'))

    # Plot mean spectrogram for all channels to make sure nothing is off
    dat.firing_overview(
            data = np.array([normalize_timeseries(x,time_vec,2) \
                    for x in np.mean(amplitude_array,axis=(0,2))]),
            t_vec = time_vec,
            y_values_vec = freq_vec,
            subplot_labels = middle_channels_bool)
    fig = plt.gcf()
    #for ax in fig.get_axes():
    #    ax.axis('off')
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'spectrogram_overview.png'))

    del amplitude_array

    # Plot mean phase for all channels to make sure nothing is off
    dat.firing_overview(np.angle(np.mean(np.exp(phase_array*-1.j),axis=(0,2))), 
            subplot_labels = middle_channels_bool)#, min_val = -np.pi, max_val = np.pi)
    fig = plt.gcf()
    #for ax in fig.get_axes():
    #    ax.axis('off')
    fig.set_size_inches(10,8)
    fig.savefig(os.path.join(plot_dir,'phase_overview.png'))

    del phase_array
    plt.close('all')

