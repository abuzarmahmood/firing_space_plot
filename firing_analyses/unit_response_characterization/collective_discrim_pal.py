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
from pathlib import Path
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

plot_save_dir = Path('/media/bigdata/Abuzar_Data/all_overlay_psths')
discrim_plot_dir = Path('/media/bigdata/firing_space_plot/firing_analyses/'\
        'unit_response_characterization/plots/discrim_plots')
discrim_file_list = list(plot_save_dir.glob('*discrim*.json'))
pal_corr_file_list = list(plot_save_dir.glob('*corr*.json'))

############################################################
## Discrim collective
############################################################
discrim_list = [pd.read_json(x) for x in discrim_file_list]
region_names =['gc','bla'] 

region_discrim_array_list = []
for this_region in region_names:
    this_discrim_list = [x[x.region == this_region] for x in discrim_list]
    # Convert to arrays
    time_vec = np.sort(this_discrim_list[0].time.unique())
    discrim_array_list = [] 
    for discrim_frame in this_discrim_list:
        discrim_array = discrim_frame.pivot(
                            index = 'neurons', 
                            columns = 'bins', 
                            values = 'discrim_p_vals').to_numpy()
        discrim_array_list.append(discrim_array)
    region_discrim_array_list.append(discrim_array_list)


alpha = 0.05
for dat_num, this_path in enumerate(discrim_file_list):
    this_region_discrim = [region_discrim_array_list[0][dat_num],
                            region_discrim_array_list[1][dat_num]]
    basename = this_path.stem
    fig,ax = plt.subplots(2,1)
    #ax.imshow(this_array < alpha, aspect= 'auto', interpolation = 'nearest')
    for num, (this_array, this_region_name) \
            in enumerate(zip(this_region_discrim, region_names)):
        ax[num].pcolormesh(time_vec, np.arange(this_array.shape[0]),
                this_array < alpha, shading = 'nearest') 
        ax[num].set_title(this_region_name)
    #x_ticks = ax.get_xticks()
    #x_ticks = [int(x) for x in x_ticks if x>= 0]
    #ax.set_xticks(x_ticks, labels = [time_vec[x] for x in x_ticks])
    plt.tight_layout()
    fig.savefig(str(discrim_plot_dir / basename) + '_gc.png')
    plt.close(fig)

#fin_discrim_array = np.concatenate(discrim_array_list, axis = 0) < alpha
fin_discrim_array = [np.concatenate(x,axis=0)<alpha \
        for x in region_discrim_array_list]

def resample_mean_array(array, resample_frac, resample_num):
    inds_list = [np.random.choice(np.arange(array.shape[0]),
                            size = int(array.shape[0] * resample_frac)) \
                            for i in range(resample_num)]
    mean_array = [np.mean(array[this_inds],axis=0) \
                        for this_inds in inds_list]
    return mean_array

# Bootstrap mean value
resampled_fracs = np.stack([resample_mean_array(x, 0.5, 1000) \
        for x in fin_discrim_array])
mean_discrim_frac = np.mean(resampled_fracs,axis=1)
std_discrim_frac = np.std(resampled_fracs,axis=1)

fig,ax = plt.subplots(2,1, sharex=True)
for num, this_region in enumerate(region_names):
    ax[num].plot(time_vec, mean_discrim_frac[num])
    ax[num].fill_between(
            x = time_vec,
            y1 = mean_discrim_frac[num] - 3*std_discrim_frac[num],
            y2 = mean_discrim_frac[num] + 3*std_discrim_frac[num],
            alpha = 0.5
            )
    ax[num].set_title(this_region)
    plt.tight_layout()
    fig.savefig(discrim_plot_dir / 'discrim_frac.png')
    plt.close(fig)

############################################################
## Palatability collective 
############################################################
pal_list = [pd.read_json(x) for x in pal_corr_file_list]
