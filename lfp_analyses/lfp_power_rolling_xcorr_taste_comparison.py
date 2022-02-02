"""
Compare XCorr between tastes by palatability
Take Fold change of palatable vs unpalatable tastants
"""
########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

########################################
# Import modules
########################################

import os
import sys
import numpy as np
from scipy.stats import zscore
from tqdm import tqdm,trange
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
import pylab as plt
from glob import glob
import xarray as xr
import json

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import firing_overview, gen_square_subplots

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

save_dir = '/media/bigdata/firing_space_plot/lfp_analyses/lfp_amp_xcorr/data'
plot_dir = '/media/bigdata/firing_space_plot/'\
                    'lfp_analyses/lfp_amp_xcorr/Plots/rolling'
dir_list_path = '/media/bigdata/firing_space_plot/lfp_analyses'\
        '/lfp_amp_xcorr/dir_list.txt'
dir_list = open(dir_list_path,'r').readlines()
dir_list = [x.strip() for x in dir_list]

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM12/AM12_4Tastes_191105_083246'
#data_dir = sys.argv[1]
dir_list = [this_dir+'/' for this_dir in dir_list if this_dir[-1] != '/']

all_tastant_names = []
for data_dir in dir_list:
    json_path = glob(os.path.join(data_dir,"*.info"))[0]
    with open(json_path, 'r') as params_file:
        info_dict = json.load(params_file)
    tastant_names = info_dict['taste_params']['tastes']
    all_tastant_names.append(tastant_names)

len_check = [len(x)==4 for x in all_tastant_names]
data_list = [glob(os.path.join(
                save_dir, os.path.basename(data_dir[:-1])[:-1]
             + "*inter*")) for data_dir in dir_list]
data_check = [len(x)==1 for x in data_list]
fin_dir_list = [x for x,this_bool,that_bool in zip(dir_list, len_check, data_check) \
                    if this_bool and that_bool]
fin_tastant = [x for x,this_bool,that_bool in \
                    zip(all_tastant_names, len_check, data_check) \
                    if this_bool and that_bool]
fin_data_list = [x[0] for x,this_bool,that_bool in \
                    zip(data_list, len_check, data_check) \
                    if this_bool and that_bool]


def pal_func(a_str):
    if a_str in ['nacl','suc']:
        return 1
    else:
        return 2

########################################
## Load XCorr frames
########################################
fold_change_plot_dir = os.path.join(plot_dir, 'fold_change')
if not os.path.exists(fold_change_plot_dir):
    os.makedirs(fold_change_plot_dir)

array_list = []

#ind = 0
for ind in trange(len(fin_data_list)):

    inter_data = xr.load_dataarray(fin_data_list[ind]) 
    inter_data = inter_data.assign_coords(tastes = tastant_names) 
    pals = [pal_func(x) for x in inter_data.tastes.values]
    inter_data = inter_data.assign_coords(pals = ('tastes',pals)) 
    inter_data = inter_data[:,inter_data.order_type == 'actual']
    inter_data = inter_data.mean(dim = ['pairs','trials']).squeeze()
    inter_data = inter_data.groupby('pals').mean()

    fold_change = inter_data.sel(pals = 1) - inter_data.sel(pals=2)
    fold_change.values = zscore(fold_change, axis=-1)

    name_splits = os.path.basename(fin_dir_list[ind][:-1]).split('_')
    animal_name = name_splits[0]
    #fin_name = name_splits[0]+'_'+name_splits[2]
    ## Somehow messed up naming
    basename = os.path.basename(fin_dir_list[ind][:-1])[:-1]
    fin_plot_dir = os.path.join(plot_dir, basename)
    #fold_change = fold_change.assign_attrs(
    #        animal_name = animal_name, basename = basename)
    fold_change = fold_change.expand_dims('name')
    fold_change = fold_change.assign_coords(name = np.atleast_1d(basename))
    fold_change = fold_change.expand_dims('animal_name')
    fold_change = fold_change.assign_coords(animal_name = np.atleast_1d(animal_name))

    xr.plot.pcolormesh(np.squeeze(fold_change),
            x = 'bins', y = 'freqs', cmap = 'viridis')
    fig = plt.gcf()
    fig.suptitle(basename + ": Pal/Unpal fold_change")
    fig.savefig(os.path.join(fin_plot_dir,basename+'_pal_unpal_fold_change'))
    plt.close(fig)

    array_list.append(fold_change)

fin_array = xr.concat(array_list, dim = 'name')

mean_fin_array = fin_array.groupby('animal_name').mean(dim = 'name', skipna=True)

xr.plot.line(mean_fin_array, x = 'bins', col = 'freqs', 
        color = 'blue', alpha = 0.5)
fig = plt.gcf()
fig.suptitle('All animal' + ": Line Mean Pal/Unpal fold_change")
fig.savefig(os.path.join(fold_change_plot_dir, \
        'all_animal' + '_line_pal_unpal_fold_change'))
plt.close(fig)

for animal_array in mean_fin_array:
    xr.plot.pcolormesh(np.squeeze(animal_array),
            x = 'bins', y = 'freqs', cmap = 'viridis')#, robust = True)
    fig = plt.gcf()
    animal_name = str(animal_array.animal_name.values)
    fig.suptitle(animal_name + ": Mean Pal/Unpal fold_change")
    fig.savefig(os.path.join(fold_change_plot_dir, \
            animal_name + 'pal_unpal_fold_change'))
    plt.close(fig)

mean_mean_array = fin_array.mean(dim = ['animal_name','name'], skipna=True) 
xr.plot.pcolormesh(np.squeeze(mean_mean_array),
        x = 'bins', y = 'freqs', cmap = 'viridis')#, robust = True)
fig = plt.gcf()
fig.suptitle('All Animals' + ": Mean Pal/Unpal fold_change")
fig.savefig(os.path.join(fold_change_plot_dir, \
        'All Animals' + '_pal_unpal_fold_change'))
plt.close(fig)
