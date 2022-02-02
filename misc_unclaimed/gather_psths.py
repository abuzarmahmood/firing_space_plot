"""
Go through all specified files and generate PSTHs
for GC and BLA neurons to save in a consolidated location
"""
########################################
# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   
########################################
import os
import sys

import pylab as plt
import numpy as np
import argparse
from glob import glob
import json

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

file_list_path = '/media/bigdata/Abuzar_Data/hdf5_file_list.txt'
plot_save_dir = '/media/bigdata/Abuzar_Data/all_overlay_psths'
if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)

region_name_list = ['gc','bla']
region_plot_dirs = [os.path.join(plot_save_dir,this_name) \
        for this_name in region_name_list]
for this_plot_dir in region_plot_dirs:
    if not os.path.exists(this_plot_dir):
        os.makedirs(this_plot_dir)

def get_plot_dir(region_name):
   return [plot_dir for name,plot_dir \
           in zip(region_name_list,region_plot_dirs)\
           if region_name == name ][0] 

counter_list = [0,0]

def add_to_counter(region_name):
    ind = [num for num,this_name \
           in enumerate(region_name_list)\
           if region_name == this_name][0]
    current_count = counter_list[ind]
    counter_list[ind] +=1
    return current_count

#parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
#parser.add_argument('dir_name',  help = 'Directory containing data files')
#parser.add_argument('states', type = int, help = 'Number of States to fit')
#args = parser.parse_args()
#data_dir = args.dir_name 

with open(file_list_path,'r') as this_file:
    file_list = this_file.read().splitlines()
dir_list = [os.path.dirname(x) for x in file_list]

#For each file, calculate baks firing, split by region
# and save PSTH in a folder with file name and 
# unit details
for data_dir in dir_list:
    #data_dir = os.path.dirname(file_list[0])
    #data_dir = '/media/bigdata/Abuzar_Data/AM28/AM28_2Tastes_201005_134840'

    # Look for info file
    # If absent, skip this file because we won't know tastant names
    info_file_path = glob(os.path.join(data_dir,"*.info"))
    if len(info_file_path) == 0:
        continue

    with open(info_file_path[0], 'r') as params_file:
        info_dict = json.load(params_file)
    taste_names = info_dict['taste_params']['tastes']

    dat = ephys_data(data_dir)
    # Try to get spikes, if can't, skip file
    try:
        dat.get_spikes()
    except:
        continue

    if not dat.spikes[0].shape[-1]==7000:
        continue

    dat.firing_rate_params = dat.default_firing_params
    dat.firing_rate_params['type'] = 'conv'

    dat.get_unit_descriptors()
    dat.get_region_units()
    dat.get_firing_rates()

    stim_t = 2000
    time_lims = [1000,5000]
    time_vec = np.arange(dat.spikes[0].shape[-1])-stim_t
    time_vec = time_vec[time_lims[0]:time_lims[1]]
    if dat.firing_rate_params['type'] == 'baks':
        bin_width = int(dat.firing_rate_params['baks_resolution']/\
                        dat.firing_rate_params['baks_dt'] )
    else:
        bin_width = int(dat.firing_rate_params['step_size'])
    baks_time_vec = time_vec[::bin_width] 
    mean_firing = np.mean(dat.firing_array,axis=2)
    mean_firing = mean_firing[...,time_lims[0]//bin_width:time_lims[1]//bin_width]

    for this_region_name, this_unit_list in zip(dat.region_names,dat.region_units):
        for unit_num in this_unit_list:
            fig,ax = plt.subplots()
            for taste_num,this_taste in enumerate(mean_firing[:,unit_num]):
                ax.plot(baks_time_vec, this_taste, label = taste_names[taste_num])
            ax.legend()
            fig.suptitle(os.path.basename(dat.data_dir) + \
                    f'\nUnit {unit_num}, '\
                    f'Electrode {dat.unit_descriptors[unit_num][0]}, '\
                    f'Region : {this_region_name}')
            ax.set_xlabel('Time post-stimulus delivery (ms)')
            ax.set_ylabel('Firing Rate (Hz)')
            fig.savefig(os.path.join(get_plot_dir(this_region_name),
                f'unit{add_to_counter(this_region_name)}'))
            plt.close(fig)
