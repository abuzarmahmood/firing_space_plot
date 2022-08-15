"""
Check the distribution of phase difference between BLA and GC
Non-zero diference would argue against volume conduction

Local or Not Local: Investigating the Nature of Striatal 
Theta Oscillations in Behaving Rats
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5616191/

Emergent modular neural control drives coordinated motor actions
Supplementary Fig. 4
https://www.nature.com/articles/s41593-019-0407-2
"""

import numpy as np
import re
import json
from glob import glob
import os
import pandas as pd
import pickle 
import sys
from scipy import stats
from scipy.stats import percentileofscore as p_of_s
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm, trange 
import tables
import pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler as ss

#import theano
#theano.config.compute_test_value = "ignore"

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs/all_tastes')
from check_data import check_data 
import itertools as it

dir_list_path = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/fin_inter_list_3_14_22.txt'

dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]

this_dir = dir_list[0]
dat = ephys_data(this_dir)
dat.get_stft(dat_type = 'phase')
dat.get_lfp_electrodes()
dat.lfp_region_electrodes
dat.region_names

## Convert phase to polar
polar_phase_array = np.exp(-1j * dat.phase_array)

# Get single electrodes from each region most reflective of mean phase
region_phase_list = [polar_phase_array[:,inds] for inds in dat.lfp_region_electrodes]
mean_region_phase = [np.mean(x,axis=1) for x in region_phase_list]
phase_diff_list = [np.abs(x-y[:,np.newaxis]) \
        for x,y in zip(region_phase_list, mean_region_phase)]
mean_phase_diff = [np.mean(x, axis=(0,2,3,4)) for x in phase_diff_list]
rep_inds = [np.argmin(x) for x in mean_phase_diff]

select_region_electrodes = np.stack([x[:,i] \
        for x,i in zip(region_phase_list, rep_inds)])
select_phase_diff = np.squeeze(np.diff(select_region_electrodes,axis=0))
select_phase_diff = np.real(np.log(select_phase_diff)/-1j)

### Test
#x = np.array([0.1,0.2,0.3,0.35,0.37,0.39,0.36,0.7])
#polar_x = np.exp(-1j*x)
#mean_polar_x = np.mean(polar_x)
#diff_polar = polar_x - mean_polar_x
#x[np.argmin(np.abs(diff_polar))]
