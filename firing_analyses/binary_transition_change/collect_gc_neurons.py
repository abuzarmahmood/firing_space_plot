"""
Iterate over all data
Find recordings with:
    1) GC recording
    2) No laser
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import itertools as it

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc/v2')
from changepoint_io import fit_handler

file_list_path = '/media/bigdata/firing_space_plot/firing_analyses/'\
                    'binary_transition_change/all_h5_files.txt'

file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
dir_list = [os.path.dirname(x) for x in file_list]

num_list = []
count_list = []
unit_descriptor_list = []
for dir_num, this_dir in tqdm(enumerate(dir_list)):
    #this_dir = dir_list[0]
    dat = ephys_data(this_dir)
    dat.get_region_units()
    dat.check_laser()
    if not dat.laser_exists:
        if 'gc' in dat.region_names:
            num_list.append(dir_num)
            gc_units = dict(zip(dat.region_names, dat.region_units))['gc']
            unit_descriptor_list.append(dat.unit_descriptors[gc_units])
            count_list.append(len(gc_units))

########################################
## Fit changepoints
########################################
gc_dir_list = [dir_list[i] for i in num_list]
taste_num_list = range(3)
arg_list = list(it.product(gc_dir_list,taste_num_list))

for data_dir, taste_num in tqdm(arg_list):
    #data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191029_171714'
    #taste_num = 0
    region_name = 'gc'
    experiment_name = 'natasha_gc_binary'

    model_parameters = dict(zip(['states','fit','samples'],[4,40000,20000]))
    preprocess_parameters = dict(zip(['time_lims','bin_width','data_transform'],
                                    [[2000,4000],50, None]))

    fit_handler_kwargs = {'data_dir' : data_dir,
                        'taste_num' : taste_num,
                        'region_name' : region_name,
                        'experiment_name' : experiment_name}
    handler = fit_handler(**fit_handler_kwargs)
    handler.set_model_params(**model_parameters)
    handler.set_preprocess_params(**preprocess_parameters)

    error_file_path = os.path.join(
            handler.database_handler.model_save_dir,
            'error_log_file.txt')

    try:
        handler.run_inference()
        handler.save_fit_output()
    except:
        with open(error_file_path,'a') as error_file:
            error_file.write(\
                str((data_dir,taste_num,region_name, preprocess_transform)) + "\n")
