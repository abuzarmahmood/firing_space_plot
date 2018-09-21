

#  ____                 _ _              _____  _       
# |  _ \               | (_)            |  __ \(_)      
# | |_) | __ _ ___  ___| |_ _ __   ___  | |  | |___   __
# |  _ < / _` / __|/ _ \ | | '_ \ / _ \ | |  | | \ \ / /
# | |_) | (_| \__ \  __/ | | | | |  __/ | |__| | |\ V / 
# |____/ \__,_|___/\___|_|_|_| |_|\___| |_____/|_| \_/  ergence
#

# 
######################### Import dat ish #########################
import tables
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from scipy.spatial import distance_matrix as dist_mat
from scipy.stats.mstats import zscore
from scipy.stats import pearsonr
from scipy import signal

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal
import copy

from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp


#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#
corr_dat = pd.DataFrame()

for file in range(1,7):
    data_dir = '/media/bigdata/jian_you_data/des_ic/file_%i/' % file
    data = ephys_data(data_dir = data_dir ,file_id = file)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    data.correlation_params = dict(zip(['stimulus_start_time', 'stimulus_end_time', 
            'baseline_start_time', 'baseline_end_time',
            'baseline_window_sizes', 'shuffle_repeats', 'accumulated'], 
            [2000, 4000, 100, 2000, np.arange(0 ,2000, 100), 100, False]))
    data.all_baseline_windows = (0,2000)
    data.get_correlations()
    
    # Run correlation analysis
    # Off_firing
    pool = mp.Pool(processes = mp.cpu_count())
    results = [pool.apply_async(baseline_stimulus_correlation_mean, args = (off_firing, all_baseline_windows[i], stimulus_time,
                                      stimulus_window_size, step_size, shuffle_repeats, file, False)) for i in range(len(all_baseline_windows))]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    for i in range(len(output)):
        corr_dat = pd.concat([corr_dat,output[i][0]])
        
    print('file %i off' % file)
    
    # On_firing
    pool = mp.Pool(processes = mp.cpu_count())
    results = [pool.apply_async(baseline_stimulus_correlation_mean, args = (on_firing, all_baseline_windows[i], stimulus_time,
                                      stimulus_window_size, step_size, shuffle_repeats, file, True)) for i in range(len(all_baseline_windows))]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    for i in range(len(output)):
        corr_dat = pd.concat([corr_dat,output[i][0]])
        
    print('file %i on' % file)

#############################
mean_squared = lambda x: np.mean(x**2)
for file in [6]:#range(1,7):
    g = sns.FacetGrid(corr_dat.query('file == %i and shuffle == False' % file),col = 'taste', 
                      row = 'pre_stim_window_size', hue = 'laser', 
                      sharey = 'row')
    #g.set(ylim=(0,None)
    g.map(sns.regplot,'baseline_end','rho', 
          x_estimator = np.mean, x_ci = 'sd').add_legend()
    g.fig.suptitle('FILE %i' % file)
    #g.fig.suptitle('All files')
    g.savefig('file%i_JY.png' % file)
    plt.close('all')
