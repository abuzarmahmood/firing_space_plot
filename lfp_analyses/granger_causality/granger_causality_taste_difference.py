"""
Look for taste differences in spectral granger causality
"""
import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

############################################################
# Load Data
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

basename_list = [os.path.basename(this_dir) for this_dir in dir_list]
animal_name = [this_name.split('_')[0] for this_name in basename_list]
animal_count = np.unique(animal_name, return_counts=True)
session_count = len(basename_list)

n_string = f'N = {session_count} sessions, {len(animal_count[0])} animals'

save_path = '/ancillary_analysis/granger_causality/'
# Get node paths for individual tastes
with tables.open_file(os.path.join(this_dir,'*.h5'), 'r') as h5:
    node_list = h5.list_nodes(save_path)
    node_list = [this_node._v_pathname for this_node in node_list \
            if 'all' not in os.path.basename(this_node._v_pathname)]

names = ['granger_actual',
         'wanted_window',
         'time_vec',
         'freq_vec']

loaded_dat_list = []
for this_dir in dir_list:
    h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
    with tables.open_file(h5_path, 'r') as h5:
        loaded_dat = [h5.get_node(save_path, this_name)[:]
                      for this_name in names]
        loaded_dat_list.append(loaded_dat)

zipped_dat = zip(*loaded_dat_list)
zipped_dat = [np.stack(this_dat) for this_dat in zipped_dat]

(
    granger_actual,
    wanted_window,
    time_vec,
    freq_vec) = zipped_dat

wanted_window = np.array(wanted_window[0])/1000
stim_t = 2
corrected_window = wanted_window-stim_t
freq_vec = freq_vec[0]
time_vec = time_vec[0]
time_vec += corrected_window[0]

wanted_freq_range = [1, 100]
wanted_freq_inds = np.where(np.logical_and(freq_vec >= wanted_freq_range[0],
                                           freq_vec <= wanted_freq_range[1]))[0]
freq_vec = freq_vec[wanted_freq_inds]
# granger_actual shape = (session, repeats, time, freq, d1, d2)
granger_actual = granger_actual[:, :, :, wanted_freq_inds]
granger_actual = np.stack(
        [granger_actual[:, :, :, :, 0, 1],
         granger_actual[:, :, :, :, 1, 0]])
# granger_actual shape = (session, repeats, time, freq, direction)
granger_actual = np.moveaxis(granger_actual, 0, -1)

# Convert to pandas dataframe for easier handling with pingouin
inds = list(np.ndindex(granger_actual.shape))
dim_names=  ['session', 'repeats', 'time', 'freq', 'direction']
df = pd.DataFrame(
        dict(
            zip(dim_names, inds)
            )
        )
df['granger'] = granger_actual.flatten()

############################################################
# Peform ANOVA for each time point, frequency, direction
############################################################
grouped_dat = df.groupby(['time', 'freq', 'direction'])
                  
