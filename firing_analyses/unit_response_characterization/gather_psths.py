"""
Go through all specified files and generate PSTHs
for GC and BLA neurons to save in a consolidated location

For each neuron, also calculate discriminability and palatability correlation
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
import pandas as pd
import pingouin as pg
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm,trange
from scipy.stats import spearmanr, pearsonr

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

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
taste_names = ['nacl', 'suc', 'ca', 'qhcl']
pal_map = dict(zip(taste_names, [3,4,2,1]))

with open(file_list_path,'r') as this_file:
    file_list = this_file.read().splitlines()
dir_list = [os.path.dirname(x) for x in file_list]
#dir_list = [x for x in dir_list if 'bla_gc'in x]

wanted_sessions = ['AM34_4Tastes_201215', 'AM37_4Tastes_210112']
dir_list = [[x for x in dir_list if y in x] for y in wanted_sessions]
dir_list = [x for y in dir_list for x in y]

#For each file, calculate baks firing, split by region
# and save PSTH in a folder with file name and 
# unit details

alpha = 0.05

#black_list = [
#        '/media/storage/gc_only/AS18/AS18_4Tastes_200228_151511_copy/AS18_4Tastes_200228_151511'
#        ]

#dir_list = [x for x in dir_list if x not in black_list]
#dir_list = [x for x in dir_list if 'AM34' in x]

#for data_dir in dir_list:
for ind in trange(len(dir_list)):
    #for ind in trange(53, len(dir_list)):
    data_dir = dir_list[ind]
    #data_dir = os.path.dirname(file_list[0])
    #data_dir = '/media/bigdata/Abuzar_Data/AM28/AM28_2Tastes_201005_134840'

    data_basename = os.path.basename(data_dir)
    # Look for info file
    # If absent, skip this file because we won't know tastant names
    info_file_path = glob(os.path.join(data_dir,"*.info"))
    if len(info_file_path) == 0:
        continue

    with open(info_file_path[0], 'r') as params_file:
        info_dict = json.load(params_file)
    taste_names = info_dict['taste_params']['tastes']
    taste_pals = np.array([pal_map[x] for x in taste_names])

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

    unit_region_map = [{x:region_name for x in this_region} \
                        for this_region,region_name \
                        in zip(dat.region_units, dat.region_names)]
    fin_unit_map = {}
    for x in unit_region_map:
        fin_unit_map.update(x)

    # For each neuron, calculate disciriminability per bin
    inds = np.array(list(np.ndindex(dat.firing_array.shape)))
    firing_frame = pd.DataFrame(
            dict(
                taste = inds[:,0],
                neurons = inds[:,1],
                trials = inds[:,2],
                bins = inds[:,3],
                firing = dat.firing_array.flatten()
                    )
                )

    group_keys = ['neurons','bins']
    grouped_frame = list(firing_frame.groupby(group_keys))
    group_tuples = [x[0] for x in grouped_frame]
    group_tuple_dicts = [dict(zip(group_keys, x)) for x in group_tuples]
    group_dat = [x[1] for x in grouped_frame]
    anova_lambda = lambda x : \
            pg.anova(data=x, dv = 'firing', between = 'taste')['p-unc'].values[0]
    p_vals = parallelize(anova_lambda, group_dat)
    # It seems like sometimes the anova conks out
    # Replace any strings with int(1)
    p_vals = [x if isinstance(x, np.float) else 1 for x in p_vals]
    discrim_frame = pd.DataFrame(group_tuple_dicts)
    discrim_frame['discrim_p_vals'] = p_vals
    discrim_frame['discrim_bool'] = (discrim_frame['discrim_p_vals'] < alpha )*1

    # Conservative criterion, significance has to persist for 75ms otherwise toss
    # This is from 3 consecutive windows of firing rate with 25ms steps
    kern_len = 4
    box_kern = np.ones(kern_len)/kern_len
    discrim_frame['discrim_bool_cons'] = \
            np.convolve(discrim_frame['discrim_bool'], box_kern, mode = 'same') == 1
    discrim_frame['discrim_bool_cons'] *= 1 
    discrim_frame['p_vals_conv'] = \
            np.convolve(discrim_frame['discrim_p_vals'], box_kern, mode = 'same') 

    # Also calculate palatability correlation for sinle neurons
    taste_pal_broad = np.expand_dims(taste_pals, (1,2,3))
    taste_pal_broad = np.broadcast_to(taste_pal_broad, 
                            dat.firing_array.shape)

    firing_array = dat.firing_array.copy()
    #firing_array = np.moveaxis(firing_array, 1,2)
    #firing_array = np.reshape(firing_array, (-1, *firing_array.shape[2:]))

    #taste_pal_broad = np.moveaxis(taste_pal_broad, 1,2)
    #taste_pal_broad = np.reshape(taste_pal_broad, (-1, *taste_pal_broad.shape[2:]))

    #firing_array = firing_array.T
    #taste_pal_broad = taste_pal_broad.T

    iter_inds = list(np.ndindex((
                        firing_array.shape[1],
                        firing_array.shape[-1])))
    corr_lambda = lambda inds: \
            pearsonr(  firing_array[:,inds[0],:,inds[1]].flatten(), 
                        taste_pal_broad[:,inds[0],:,inds[1]].flatten()
                        )
    corr_outs = parallelize(corr_lambda, iter_inds)
    corr_pvals = [x[1] for x in corr_outs]
    corr_rhos = [np.abs(x[0]) for x in corr_outs]

    iter_array = np.array(iter_inds)
    corr_frame = pd.DataFrame(
            dict(
                neurons = iter_array[:,0],
                bins = iter_array[:,1],
                corr_pvals = corr_pvals,
                corr_rhos = corr_rhos
                )
            )

    corr_frame['pvals_cons'] = \
            np.convolve(corr_frame['corr_pvals'], box_kern, mode = 'same') 
    corr_frame['sig_bool'] = corr_frame['pvals_cons'] <= alpha

    #corr_array = corr_frame.pivot(
    #                    index = 'neurons', 
    #                    columns = 'bins', 
    #                    values = 'corr_pvals').to_numpy()

    #fig,ax = plt.subplots()
    #ax.imshow(corr_array < alpha, 
    #        interpolation = 'nearest', aspect='auto')
    #fig.savefig(
    #        os.path.join(plot_save_dir, f'{data_basename}_corr.png'),
    #        dpi = 300)
    #plt.close(fig)

    ############################################################
    #fin_pval_frame = discrim_frame.join(corr_frame, 
    #        lsuffix = 'x', rsuffix = 'y')
    #fin_pval_frame.drop(columns = ['binsy','neuronsy'], inplace=True)
    #fin_pval_frame.rename(columns = dict(neuronsx = 'neurons',
    #                                binsx = 'bins'), inplace=True)

    #fin_pval_frame['region'] = [fin_unit_map[x] for x in \
    #        fin_pval_frame.neurons.values]

    #fin_pval_frame.to_json(
    #        os.path.join(plot_save_dir, f'{data_basename}_unit_pvals.json')
    #        )

    #fin_pval_frame['time'] = (fin_pval_frame.bins * bin_width)-stim_t
    ############################################################ 
    ############################################################ 

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

    #fin_pval_frame = fin_pval_frame[fin_pval_frame.time.isin(baks_time_vec)]
    corr_frame['time'] = (corr_frame.bins * bin_width)-stim_t
    discrim_frame['time'] = (discrim_frame.bins * bin_width)-stim_t
    discrim_frame = discrim_frame[discrim_frame.time.isin(baks_time_vec)]
    corr_frame = corr_frame[corr_frame.time.isin(baks_time_vec)]

    # Add region name
    corr_frame['region'] = [fin_unit_map[x] for x in \
            corr_frame.neurons.values]
    discrim_frame['region'] = [fin_unit_map[x] for x in \
            discrim_frame.neurons.values]

    discrim_frame.to_json(
            os.path.join(plot_save_dir, f'{data_basename}_discrim_frame.json')
           )
    corr_frame.to_json(
            os.path.join(plot_save_dir, f'{data_basename}_corr_frame.json')
           )

    mean_firing = np.mean(dat.firing_array,axis=2)
    mean_firing = mean_firing[...,time_lims[0]//bin_width:time_lims[1]//bin_width]

    for this_region_name, this_unit_list in zip(dat.region_names,dat.region_units):
        for unit_num in this_unit_list:

            #unit_frame = fin_pval_frame[fin_pval_frame.neurons.isin([unit_num])]
            unit_corr_frame = \
                    corr_frame[corr_frame.neurons.isin([unit_num])]
            unit_sig_corr = unit_corr_frame[unit_corr_frame.corr_pvals < alpha]
            unit_discrim_frame = \
                    discrim_frame[discrim_frame.neurons.isin([unit_num])]

            unit_discrim_frame['bool'] = \
                    1*(unit_discrim_frame.discrim_p_vals < alpha)

            #fig,ax = plt.subplots(3,1, sharex=True)

            fig = plt.figure()
            ax = []
            ax.append(fig.add_subplot(2,1,1))
            ax.append(fig.add_subplot(4,1,3))#, sharex = ax[0]))
            ax.append(fig.add_subplot(4,1,4))#, sharex = ax[0]))

            xlims = [-500, 1500]
            xinds = np.logical_and(baks_time_vec >= xlims[0],
                                    baks_time_vec <= xlims[1])
            fin_time_vec = baks_time_vec[xinds]
            unit_discrim_frame = unit_discrim_frame[unit_discrim_frame.time.isin(fin_time_vec)]
            unit_corr_frame = unit_corr_frame[unit_corr_frame.time.isin(fin_time_vec)]
            unit_sig_corr = unit_sig_corr[unit_sig_corr.time.isin(fin_time_vec)]


            for taste_num,this_taste in enumerate(mean_firing[:,unit_num]):
                ax[0].plot(fin_time_vec, 
                        this_taste[xinds], label = taste_names[taste_num],
                        linewidth = 2)
            #ax[0].legend()
            fig.suptitle(os.path.basename(dat.data_dir) + \
                    f'\nUnit {unit_num}, '\
                    f'Electrode {dat.unit_descriptors[unit_num][0]}')#, '\
                    #f'Region : {this_region_name}')
            ax[-1].set_xlabel('Time post-stimulus delivery (ms)')
            ax[0].set_ylabel('Firing Rate (Hz)')
            #ax[0].set_xlim([-500, 1500])
            #ax[1].set_xlim([-500, 1500])
            #ax[2].set_xlim([-500, 1500])

            cmap = plt.get_cmap('binary')

            #ax[1].plot(unit_discrim_frame.time, unit_discrim_frame['bool'])
            ax[1].plot(unit_discrim_frame.time, 
                    unit_discrim_frame['discrim_bool_cons'],
                    color = cmap(0.5))
            ax[1].fill_between(
                    x = unit_discrim_frame.time, 
                    y1 = unit_discrim_frame['discrim_bool_cons'],
                    y2 = 0,
                    alpha = 0.7,
                    color = cmap(0.5))
            #ax[1].plot(unit_discrim_frame.time, 
            #        np.log10(unit_discrim_frame['p_vals_conv']))
            #ax[1].axhline(np.log10(0.05))
            ax[1].set_ylabel('Discrim sig')
            ax[2].plot(unit_corr_frame.time, unit_corr_frame.corr_rhos,
                    color = cmap(0.7))
            ax[2].fill_between(
                    x = unit_corr_frame.time, 
                    y1 = unit_corr_frame['corr_rhos'],
                    #where = unit_corr_frame['corr_pvals'] <= 0.05,
                    where = unit_corr_frame['sig_bool'], 
                    y2 = 0,
                    alpha = 0.7,
                    color = cmap(0.7))
            #ax[2].plot(unit_sig_corr.time, unit_sig_corr.corr_rhos, 'x')
            ax[2].set_ylabel('Pal Corr sig')

            #ax[0].tick_params(axis='x', which = 'both', bottom = False)
            ax[0].set_xticklabels([])
            ax[1].set_xticklabels([])
            #plt.show()

            fig.savefig(os.path.join(get_plot_dir(this_region_name),
                f'{data_basename}_unit{add_to_counter(this_region_name)}' + '.svg'))
            plt.close(fig)
