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
from scipy.ndimage import gaussian_filter1d as gauss_filt
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
        'spike_noise_corrs/Plots/rolling/aggregate'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

dir_list = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'spike_noise_corrs/good_inter_region_paths.txt'
dir_paths = [x.strip() for x in open(dir_list,'r').readlines()]
dir_basenames = [os.path.basename(x) for x in dir_paths]
dir_animal_names = [x.split("_")[0] for x in dir_basenames]
dir_names_dicts = [dict(zip(['animal_name','basename'],[x,y]))\
                for x,y in zip(dir_animal_names, dir_basenames)]
dat_paths = [glob(os.path.join(save_dir, x + "*")) \
        for x in dir_basenames]

alpha = 0.05

def get_mean_dims(x, leave_dims = ['taste','time']):
    return [this_dim for this_dim in x.dims if this_dim not in leave_dims] 

wanted_attrs = ['region_type','comparison_type']
concat_data_list = []

for this_session in tqdm(dat_paths):
    #this_session = dat_paths[0]
    xr_datasets = [xr.load_dataset(x) for x in this_session]

    p_sets = [x.p for x in xr_datasets]

    # Remove pairs with ANY NANs ANYWHERE
    nan_inds = [np.unique(np.where(np.isnan(x))[0]) for x in p_sets]
    wanted_pairs = [[x for x in np.arange(dat_set.shape[0]) if x not in y]\
                        for y,dat_set in zip(nan_inds, p_sets)]
    p_sets = [x[inds] for x,inds in zip(p_sets, wanted_pairs)]

    p_dat = [x <= alpha for x in p_sets]
    #p_dat = [x.p <= alpha for x in xr_datasets]
    for x,y in zip(p_dat,xr_datasets):
        x.attrs = y.attrs
    mean_xr_data = [x.mean(dim=get_mean_dims(x), keep_attrs = True) \
            for x in p_dat]

    ## Merge into single array along region and comparison dimensions
    comp_list = [x.attrs['comparison_type'] for x in mean_xr_data]
    region_list = [x.attrs['region_type'] for x in mean_xr_data]
    attrs_frame = pd.DataFrame(dict(
        comp = comp_list, region = region_list, ind = np.arange(len(comp_list))))
    attrs_frame.sort_values(by = ['comp','region'], inplace=True)
    grouped_inds = [x.ind.values for num,x in list(attrs_frame.groupby('comp'))]

    attr_list = [dict(zip(wanted_attrs,[x.attrs[key] for key in wanted_attrs]))\
            for x in mean_xr_data]

    mean_xr_data = [x.assign_coords(**this_attrs) \
            for this_attrs,x in zip(attr_list,mean_xr_data)] 
    mean_xr_data = [x.expand_dims(wanted_attrs) for x in mean_xr_data] 
    nested_data = [[mean_xr_data[i] for i in this_group] \
            for this_group in grouped_inds]

    concat_ds = xr.concat([xr.concat(this_list, dim = 'region_type') 
                        for this_list in nested_data],
                        dim = 'comparison_type')
    concat_data_list.append(concat_ds)

# Merge into one giant frame
concat_data_list = [x.assign_coords(basename = y) \
        for y,x in zip(dir_basenames,concat_data_list)] 
concat_data_list = [x.expand_dims('basename') \
        for x in concat_data_list] 
fin_concat_data = xr.concat(concat_data_list, dim = 'basename')

fin_concat_data.values = gauss_filt(fin_concat_data.values, 2)

mean_fin_dat = fin_concat_data.mean(dim = ['basename','taste'])
std_fin_dat = fin_concat_data.std(dim = ['basename','taste'])
sem_scaling = np.sqrt(len(fin_concat_data.basename))# * len(fin_concat_data.taste))
std_fin_dat = std_fin_dat / sem_scaling
mean_fin_dat.name = 'mean'
std_fin_dat.name = 'std'
stat_dataset = xr.merge([mean_fin_dat, std_fin_dat])

#stat_dataset = stat_dataset.sel(region_type = 'inter')
#stat_dataset = stat_dataset.expand_dims('region_type')
stat_dataset = stat_dataset.stack(region_comp = ['region_type','comparison_type'])

enc_vals, enc_keys = pd.factorize(stat_dataset.region_comp.values)
this_cmap = plt.get_cmap('tab10')
colors = this_cmap(enc_vals)
lut = dict(zip(enc_keys, colors))

wanted_vars = ['time_lims','window_size','step_size']
for this_name in wanted_vars:
    globals()[this_name] = xr_datasets[0].attrs[this_name]
x = np.arange(xr_datasets[0].p.shape[-1])
stim_t = 2000 - window_size
x_fin = x*step_size + time_lims[0] - stim_t

## Plot mean for different conditions
fig, ax = plt.subplots(1,2, figsize = (10,5))
for this_mean, this_std in zip(stat_dataset['mean'].T, stat_dataset['std'].T):
    this_coords = this_mean.coords['region_comp'].values.flatten()[0]
    if this_coords == ('inter','actual'):
        this_col = lut[this_coords]
        if 'shuffle' not in this_coords:
            label = "_".join(this_coords)
        else:
            label = None
        ax[0].plot(x_fin, this_mean, label = label, linewidth = 2, color = this_col)
        ax[0].fill_between(x = x_fin, 
                y1 = this_mean + this_std,
                y2 = this_mean - this_std, alpha = 0.3, color = this_col)
ax[0].axvline(0, linewidth = 2, linestyle = 'dashed', alpha = 0.6, color = 'red')
#ax[0].legend(loc = 'upper right')
ax[0].legend(bbox_to_anchor = (0, 1.3))
ax[1].text(0.1,0.5,"\n".join(dir_basenames))
ax[0].set_ylim([0.00, 0.16])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Mean Significant Fraction')
fig.suptitle('Aggregate mean Spike Count Corr significance')
#fig.savefig(os.path.join(plot_dir, 'inter_mean_sig'), dpi = 300,
fig.savefig(os.path.join(plot_dir, 'agg_inter_mean_sig'), dpi = 300,
        bbox_inches = 'tight')
