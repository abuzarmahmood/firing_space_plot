"""
Plot granger causality per taste and differences between tastes
"""
import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
from tqdm import tqdm
from joblib import Parallel, delayed

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

names = [
    'taste_anova_pval_array',
    'taste_anova_pval_dim_order',
    'taste_anova_pval_dim_time',
    'taste_anova_pval_dim_freq',
    'taste_anova_pval_dim_direction',
]


loaded_dat_list = []
for this_dir in dir_list:
    save_path = '/ancillary_analysis/granger_causality/taste_anova'
    h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
    with tables.open_file(h5_path, 'r') as h5:
        loaded_dat = [h5.get_node(save_path, this_name)[:]
                      for this_name in names]
        loaded_dat_list.append(loaded_dat)

# loaded_dat_list: [session][data_type]

# zipped_dat : [data_type][session]
zipped_dat = list(zip(*loaded_dat_list))
zipped_dat = [np.stack(this_dat) for this_dat in zipped_dat]

# taste_pval_array shape = (session, time, freq, direction)
(
    taste_pval_array,
    dim_order,
    time_vec,
    freq_vec,
    direction_vec) = zipped_dat

wanted_window = np.array([1500, 4000])/1000
stim_t = 2
corrected_window = wanted_window-stim_t
freq_vec = freq_vec[0]
time_vec = time_vec[0]
time_vec += corrected_window[0]

# Also load raw granger causality per taste
names = ['granger_actual',
         'wanted_window',
         'time_vec',
         'freq_vec']


loaded_dat_list = []
node_names = []
for this_dir in dir_list:
    #this_dir = dir_list[0]
    save_path = '/ancillary_analysis/granger_causality/'
    # Get node paths for individual tastes
    h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
    with tables.open_file(h5_path) as h5:
        node_list = [this_node._v_pathname for this_node
                     in h5.list_nodes(save_path)]
        node_list = [x for x in node_list if
                     os.path.basename(x) not in ['all', 'taste_anova']]
    node_names.append([os.path.basename(this_node) for this_node in node_list])
    with tables.open_file(h5_path, 'r') as h5:
        # Outer list is for each taste
        # Inner list is for each data type
        loaded_dat = [[h5.get_node(this_save_path, this_name)[:]
                      for this_name in names]
                      for this_save_path in tqdm(node_list)]
        loaded_dat_list.append(loaded_dat)

node_names = np.stack(node_names)
if not np.mean(node_names == node_names[0][None, :]) == 1:
    raise ValueError('Node names do not match across sessions')

# loaded_dat_list: [session][taste][data_type]

#zipped_dat = [list(zip(*x)) for x in loaded_dat_list]
# zipped_dat : [taste][session][data_type]
zipped_dat = list(zip(*loaded_dat_list))
# zipped_dat : [taste][data_type][session]
zipped_dat = [list(zip(*x)) for x in zipped_dat]
# zipped_dat : [taste][data_type][session]
zipped_dat = list(zip(*zipped_dat))
zipped_dat = [np.stack(this_dat) for this_dat in zipped_dat]

# granger_actual shape = (taste, session repeats, time, freq, d1, d2)
(
    granger_actual,
    wanted_window,
    time_vec,
    freq_vec) = zipped_dat

wanted_window = np.array(wanted_window[0][0])/1000
stim_t = 2
corrected_window = wanted_window-stim_t
freq_vec = freq_vec[0][0]
time_vec = time_vec[0][0]
time_vec += corrected_window[0]

wanted_freq_range = [1, 100]
wanted_freq_inds = np.where(np.logical_and(freq_vec >= wanted_freq_range[0],
                                           freq_vec <= wanted_freq_range[1]))[0]
freq_vec = freq_vec[wanted_freq_inds]
# granger_actual shape = (session, repeats, time, freq, d1, d2)
granger_actual = granger_actual[:, :, :, :, wanted_freq_inds]
granger_actual = np.stack(
    [granger_actual[..., 0, 1],
     granger_actual[..., 1, 0]])
# granger_actual shape = (tastes, session, repeats, time, freq, direction)
granger_actual = np.moveaxis(granger_actual, 0, -1)

# Analyze single sessions at a time
# Otherwize memory blows up
# granger_actual shape = (session, tastes, repeats, time, freq, direction)
granger_actual = np.moveaxis(granger_actual, 0, 1)

# mean_taste_granger shape = (sessions, tastes, time, freq, direction)
mean_taste_granger = np.nanmean(granger_actual, axis=2)

############################################################
# Plot differences
############################################################

plot_dir_base = '/media/bigdata/firing_space_plot/lfp_analyses/' +\
    'granger_causality/plots/taste_difference_plots'

if not os.path.exists(plot_dir_base):
    os.makedirs(plot_dir_base)

# Plot all tastes AND plot mask
#session_num = 0
for session_num in range(mean_taste_granger.shape[0]): 
    pval_dat = np.moveaxis(taste_pval_array[session_num], -1, 0)
    log10_pval_dat = np.log10(pval_dat)
    granger_dat = np.moveaxis(mean_taste_granger[session_num], -1, 0)
    vmin, vmax = np.min(granger_dat), np.max(granger_dat)
    fig, ax = plt.subplots(2, 5, figsize=(15, 5),
                           sharex=True, sharey=True)
    inds = list(np.ndindex(granger_dat.shape[:2]))
    for this_ind in inds:
        ax[this_ind].pcolormesh(
                time_vec, freq_vec,
                granger_dat[this_ind].T, 
                cmap='jet', vmin=vmin, vmax=vmax)
        ax[this_ind].axvline(0, color='red', linestyle = '--', linewidth = 2)
        ax[0, this_ind[1]].set_title(node_names[session_num][this_ind[1]])
    #cbar_ax_list = [
    #    fig.add_axes([0.92, 0.1, 0.02, 0.8]),
    #    fig.add_axes([0.92, 0.1, 0.02, 0.8])]
    #    ]
    for num, this_dat in enumerate(log10_pval_dat):
        im = ax[num, -1].pcolormesh(
                time_vec, freq_vec,
                this_dat.T, 
                cmap='jet')
        plt.colorbar(im, ax=ax[num, -1])
    ax[0,-1].set_title('log10 pval')
    session_name = basename_list[session_num]
    fig.suptitle('Session {}'.format(session_name))
    fig.savefig(os.path.join(plot_dir_base, session_name + '.png'))
    plt.close(fig)

