"""
Per animal, aggregate unit response characteristics by region
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
import scipy.stats as stats
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare, ttest_rel
from scipy.stats import kruskal
import pylab as plt
from glob import glob
import json
from scipy.spatial import distance_matrix as distmat
from sklearn.preprocessing import LabelEncoder as LE
import pingouin as pg
import matplotlib as mpl
import shutil

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

dir_list_path = "/media/bigdata/Abuzar_Data/dir_list.txt"
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
#data_dir = dir_list[39]

# First get data from all files
# Then chop up by animal
kw_p_array_path = '/ancillary_analysis/taste_discriminability/taste_discrim_kw'
time_vec_path = '/ancillary_analysis/taste_discriminability/post_stim_time'
kw_frame_list = []

for data_dir in tqdm(dir_list):

    dat = ephys_data(data_dir)
    dat.get_region_units()

    with tables.open_file(dat.hdf5_path,'r+') as h5:
        if kw_p_array_path in h5:
            proceed_bool = True
            kw_array = h5.get_node(kw_p_array_path)[:]
            time_vec = h5.get_node(time_vec_path)[:]

    if proceed_bool:
        region_name_list = [[name]*len(units) \
                for name,units in zip(dat.car_names, dat.car_units)]

        nrn_region_frame = pd.DataFrame({
            'neuron' : np.concatenate(dat.car_units),
            'region' : np.concatenate(region_name_list)})

        inds = np.array(list(np.ndindex(kw_array.shape)))
        kw_frame = pd.DataFrame({
            'neuron' : inds[:,0],
            'time_bin' : time_vec[inds[:,1]],
            'p_val' : kw_array.flatten()})

        fin_frame = kw_frame.merge(nrn_region_frame, how = 'inner', on = 'neuron')
        animal_name, exp_type, exp_date = dat.hdf5_name.split('_')[:3]
        fin_frame['animal_name'] = animal_name
        fin_frame['exp_date'] = exp_date

        kw_frame_list.append(fin_frame)

fin_kw_frame = pd.concat(kw_frame_list)
fin_kw_frame.time_bin = fin_kw_frame.time_bin.round()
alpha = 0.001
fin_kw_frame['sig'] = fin_kw_frame['p_val'] < alpha
time_lims = [-500, 2000]
fin_kw_frame = fin_kw_frame.loc[fin_kw_frame['time_bin'] <= time_lims[1]]
fin_kw_frame = fin_kw_frame.loc[fin_kw_frame['time_bin'] >= time_lims[0]]

# Pull out values for each animal
multi_ind_fin_kw_frame = fin_kw_frame.set_index(
        ['animal_name','region']).sort_index()
multi_ind_fin_kw_frame.sort_values(['exp_date','neuron'], inplace=True)
multi_ind_fin_kw_frame['ind'] = \
        multi_ind_fin_kw_frame['exp_date'].astype('str') + "_" +\
        multi_ind_fin_kw_frame['neuron'].astype('str')

plot_super_dir = '/media/bigdata/firing_space_plot/firing_analyses'\
        '/unit_response_characterization/unit_response_plots'

if not os.path.exists(plot_super_dir):
    os.makedirs(plot_super_dir)
else:
    shutil.rmtree(plot_super_dir)
    os.makedirs(plot_super_dir)

for animal, region in tqdm(multi_ind_fin_kw_frame.index.unique()):

    print(animal,region)
    x = multi_ind_fin_kw_frame.loc[animal, region]

    #x = list(multi_ind_fin_kw_frame.groupby(level = [0,1]))[0][1]
    inds,vals = zip(*list(x.groupby(['exp_date','neuron'])))
    dates = [x[0] for x in inds]
    x_pivot = x.pivot('ind','time_bin','sig')


    cmap = sns.color_palette('tab10', len(np.unique(dates))) 
    lut = dict(zip(np.unique(dates),cmap))
    row_colors = [lut[x] for x in dates]


    g = sns.clustermap(x_pivot, row_cluster=  False, col_cluster= False, 
            row_colors = row_colors) 
    g.gs.update(bottom=0.55, top=0.95)
    g.cax.set_position([1, 1, 1, 1])
    for label in np.unique(dates):
            g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)
            g.ax_col_dendrogram.legend(loc="center", ncol=5)
    gs2 = mpl.gridspec.GridSpec(1,1, left = 0.2, top=0.45)
    # create axes within this new gridspec
    ax2 = g.fig.add_subplot(gs2[0])
    ax2.plot(np.mean(x_pivot,axis=0), marker = 'x')
    plt.suptitle(animal + '_' + region)
    fig = plt.gcf()
    fig.savefig(os.path.join(plot_super_dir, animal + '_' + region))
    plt.close(fig)
    #plt.show()
