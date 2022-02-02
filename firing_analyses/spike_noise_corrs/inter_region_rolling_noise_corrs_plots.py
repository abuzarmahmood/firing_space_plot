"""
1) Plot average significant comparisons
    a) In all
    b) Per taste
    for all groups (intra/inter and actual/shuffle)
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
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
from scipy.cluster import hierarchy as hc
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

save_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'spike_noise_corrs/data/rolling'
plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'spike_noise_corrs/Plots/rolling'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM12/AM12_4Tastes_191106_085215'
data_dir = sys.argv[1]
if data_dir[-1] != '/':
    data_dir += '/'
dat = ephys_data(data_dir)

json_path = glob(os.path.join(dat.data_dir,"*.info"))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)
tastant_names = info_dict['taste_params']['tastes']

name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]
# Somehow messed up naming
basename = os.path.basename(data_dir[:-1])[:-1]
fin_plot_dir = os.path.join(plot_dir, basename)

if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

########################################
## Load Corr frames
########################################
file_list = glob(os.path.join(save_dir, basename+"*"))

alpha = 0.05
xr_datasets = [xr.load_dataset(path) for path in file_list]
p_sets = [x.p for x in xr_datasets]

# Remove pairs with ANY NANs ANYWHERE
nan_inds = [np.unique(np.where(np.isnan(x))[0]) for x in p_sets]
wanted_pairs = [[x for x in np.arange(dat_set.shape[0]) if x not in y]\
                    for y,dat_set in zip(nan_inds, p_sets)]
p_sets = [x[inds] for x,inds in zip(p_sets, wanted_pairs)]

sig_datasets = [x <= alpha for x in p_sets]
#sig_datasets = [x['rho'] for x in xr_datasets]
for x,y in zip(sig_datasets,xr_datasets):
    x.attrs = y.attrs

comp_list = [x.attrs['comparison_type'] for x in sig_datasets]
region_list = [x.attrs['region_type'] for x in sig_datasets]
attrs_frame = pd.DataFrame(dict(
    comp = comp_list, region = region_list, ind = np.arange(len(comp_list))))
attrs_frame.sort_values(by = ['comp','region'], inplace=True)
grouped_inds = [x.ind.values for num,x in list(attrs_frame.groupby('comp'))]

## Removing any NANs clears out all shuffles, so just take them out for now
sig_datasets = [sig_datasets[x] for x in grouped_inds[0]]

##################################################
# ____  _       _   _   _             
#|  _ \| | ___ | |_| |_(_)_ __   __ _ 
#| |_) | |/ _ \| __| __| | '_ \ / _` |
#|  __/| | (_) | |_| |_| | | | | (_| |
#|_|   |_|\___/ \__|\__|_|_| |_|\__, |
#                               |___/ 
##################################################
wanted_vars = ['time_lims','window_size','step_size']
for this_name in wanted_vars:
    globals()[this_name] = sig_datasets[0].attrs[this_name]
x = np.arange(sig_datasets[0].shape[-1])
stim_t = 2000 - window_size
x_fin = x*step_size + time_lims[0] - stim_t

def get_mean_dims(x, leave_dims = ['taste','time']):
    return [this_dim for this_dim in x.dims if this_dim not in leave_dims] 

##################################################
## Line plots of average significance (across pairs) per taste
##################################################
mean_datasets = [x.mean(dim = get_mean_dims(x), keep_attrs = True) \
                                for x in sig_datasets]
#var_percs = [0.025, 0.975]
#std_datasets = [x.quantile(var_percs, dim = get_mean_dims(x)) for x in xr_datasets]
#std_datasets = [x.std(dim = get_mean_dims(x), keep_attrs = True) \
#        for x in sig_datasets]

fig,ax = plt.subplots(mean_datasets[0].shape[0] + 1, 1, 
        sharex = True, sharey = True, figsize = (5,10))
wanted_attrs = ['region_type','comparison_type']
for this_mean in mean_datasets:
    this_wanted_attrs = [this_mean.attrs[x] for x in wanted_attrs]
    for num, this_ax in enumerate(ax.flatten()[:-1]):
        this_ax.plot(x_fin, this_mean[num], 
                label = "_".join(this_wanted_attrs))
        this_ax.axvline(0, alpha = 0.5, color = 'red', linestyle = 'dashed')
        this_ax.axhline(0.05, alpha = 0.5, color = 'red', linestyle = 'dashed')
    ax[-1].plot(x_fin, this_mean.mean(dim = 'taste'),
            label = "_".join(this_wanted_attrs))
    ax[-1].axvline(0, alpha = 0.5, color = 'red', linestyle = 'dashed')
    ax[-1].axhline(0.05, alpha = 0.5, color = 'red', linestyle = 'dashed')
ax[-1].legend(bbox_to_anchor=(1.1, 1.05))
plt.suptitle(basename + "_mean_significance")
#fig.savefig(os.path.join(fin_plot_dir, basename + "_mean_significance"),
fig.savefig(os.path.join(fin_plot_dir, basename + "_mean_significance_clean"),
        bbox_inches = 'tight')
plt.close(fig)

##################################################
## Heatmap of average significance (across tastes) for each pair
##################################################
mean_datasets = [x.mean(dim = get_mean_dims(x, ['pair','time']), 
                keep_attrs = True) for x in sig_datasets]
this_wanted_attrs = [[this_mean.attrs[x] for x in wanted_attrs] \
                for this_mean in mean_datasets]
attr_strings = ["_".join(x) for x in this_wanted_attrs]

max_val = np.max([x.max() for x in mean_datasets])

def sorted_array(xarr):
    Z = hc.linkage(xarr, optimal_ordering = True)
    dn = hc.dendrogram(Z, no_plot = True)
    sort_inds = [int(x) for x in dn['ivl']]
    return xarr[sort_inds]

fig,ax = plt.subplots(len(grouped_inds[0]), len(grouped_inds) + 2,
        figsize = (20,7), sharex=True)#, sharey='row')
cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
imshow_kwargs = dict(#aspect = 'auto', 
        vmin = 0, vmax = max_val, cmap = 'viridis')
#for num, (act_num, shuff_num) in enumerate(zip(*grouped_inds)):
for num, act_num in enumerate(range(len(mean_datasets))):
    y = np.arange(mean_datasets[act_num].shape[0])
    try:
        this_map = ax[num,0].pcolormesh(x_fin, y, 
                sorted_array(mean_datasets[act_num]), **imshow_kwargs) 
    except:
        this_map = ax[num,0].pcolormesh(x_fin, y, 
                mean_datasets[act_num], **imshow_kwargs) 
    try:
        ax[num,1].pcolormesh(x_fin, y, 
            sorted_array(zscore(mean_datasets[act_num],axis=-1)), **imshow_kwargs) 
    except:
        ax[num,1].pcolormesh(x_fin, y, 
            zscore(mean_datasets[act_num],axis=-1), **imshow_kwargs) 

    ax[num,0].axvline(0, alpha = 0.5, color = 'yellow', linestyle = 'dashed')
    #map = ax[num,-1].pcolormesh(x_fin, y, 
    #        mean_datasets[shuff_num].values, **imshow_kwargs) 
    #ax[num,-1].axvline(0, alpha = 0.5, color = 'yellow', linestyle = 'dashed')
    ax[num,0].set_ylabel(attr_strings[act_num])
    #ax[num,-1].set_ylabel(attr_strings[shuff_num])
    ax[num,1].axvline(0, alpha = 0.5, color = 'yellow', linestyle = 'dashed')
    #ax[num,2].plot(x_fin, 
    #        zscore(np.mean(zscore(mean_datasets[act_num],axis=-1),axis=0)), 
    #        label = 'Zscore', alpha = 0.7)
    ax[num,2].plot(x_fin, 
            np.mean(mean_datasets[act_num],axis=0), 
            alpha = 0.7)
    ax[num,2].axvline(0, alpha = 0.5, color = 'red', linestyle = 'dashed')
#ax[-1,2].legend()
plt.colorbar(this_map, cax = cax)
plt.subplots_adjust(right = 0.8)
plt.suptitle(basename + "_mean_pair_significance")
fig.savefig(os.path.join(fin_plot_dir, basename + "_mean_pair_significance_clean"))
plt.close(fig)

##################################################
## Heatmap of average significance (across tastes) for each pair
##################################################
mean_datasets = [x.mean(dim = get_mean_dims(x, ['taste','time']), 
                keep_attrs = True) for x in sig_datasets]

fig,ax = plt.subplots(len(grouped_inds[0]), 1, figsize = (10,7))
#for num, act_num in enumerate(grouped_inds[0]):
#for num, act_num in enumerate(grouped_inds[0]):
for num, act_num in enumerate(range(len(mean_datasets))):
    for taste_name, this_taste in zip(tastant_names, mean_datasets[act_num]):
        ax[num].plot(x_fin, this_taste, label = taste_name)
    ax[num].set_ylabel(attr_strings[act_num])
    ax[num].axvline(0, alpha = 0.5, color = 'red', linestyle = 'dashed')
ax[-1].legend()
plt.suptitle(basename + "_mean_region_significance")
fig.savefig(os.path.join(fin_plot_dir, basename + "_mean_region_significance_clean"))
plt.close(fig)

##################################################
## Heatmap of average significance pairs and tastes 
##################################################
mean_datasets = [x.mean(dim = get_mean_dims(x, ['pair','taste','time']), 
                keep_attrs = True) for x in sig_datasets]
max_val = np.max([x.max() for x in mean_datasets])

fig,ax = plt.subplots(len(grouped_inds[0]), mean_datasets[0].shape[1], 
        figsize = (20,7), sharex=True, sharey='row')
cax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
imshow_kwargs = dict(#aspect = 'auto', 
        vmin = 0, vmax = max_val, cmap = 'viridis')
#for num, act_num in enumerate(grouped_inds[0]):
for num, act_num in enumerate(range(len(mean_datasets))):
    for taste_ind in range(mean_datasets[0].shape[1]):
        y = np.arange(mean_datasets[act_num].shape[0])
        m = ax[num,taste_ind].pcolormesh(x_fin, y, 
            mean_datasets[act_num][:,taste_ind], **imshow_kwargs) 
        ax[num,taste_ind].axvline(0, alpha = 0.5, 
                            color = 'yellow', linestyle = 'dashed')
        ax[0,taste_ind].set_title(f'Taste {taste_ind}')
    ax[num,0].set_ylabel(attr_strings[act_num])
plt.colorbar(m, cax = cax)
plt.subplots_adjust(right = 0.8)
plt.suptitle(basename + "_taste_region_significance")
fig.savefig(os.path.join(fin_plot_dir, basename + "_taste_region_significance_clean"))
plt.close(fig)
#plt.show()
