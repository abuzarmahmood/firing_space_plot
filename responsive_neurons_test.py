"""
Check fraction of taste responsive neurons in BLA vs GC
"""


## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import json
import glob
import numpy as np
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import ttest_rel 

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

# Create file to store data + analyses in
#data_hdf5_name = 'AM_LFP_analyses.h5'
#data_ffileer = '/media/bigdata/Abuzar_Data/lfp_analysis'
#data_hdf5_path = os.path.join(data_ffileer, data_hdf5_name)
file_list_path = '/media/fastdata/taste_response_analysis/file_list.txt' 

file_list = open(file_list_path,'r').readlines()
file_list = [x.rstrip() for x in file_list]
basename_list = [os.path.basename(x) for x in file_list]
dirname_list = [os.path.dirname(x) for x in file_list]
json_path_list = [glob.glob(dirname+'/*json')[0] for dirname in dirname_list]

# Find all electrodes corresponding to BLA and GC using the jsons
json_list = [json.load(open(file,'r')) for file in json_path_list]
bla_electrodes = [x['regions']['bla'] for x in json_list]
gc_electrodes = [x['regions']['gc'] for x in json_list]

all_firing = []
all_unit_descriptors = []
waveform_thresh = 1500

for file_num, file_name in tqdm(enumerate(file_list)):
    data = ephys_data(data_dir = dirname_list[file_num]) 
    data.firing_rate_params = dict(zip(['step_size','window_size','dt'],
                                   [25,250,1]))
    data.extract_and_process()
    all_firing.append(data.normalized_firing)
    all_unit_descriptors.append(data.unit_descriptors)

# Remove cases where size of firing array doesn't match with
# unit descriptors
# WTF WOULD THAT HAPPEN THOUGH???

all_unit_electrodes = [x["electrode_number"] for x in all_unit_descriptors]
matchings_inds = [num for num, (firing, electrodes) in \
        enumerate(zip(all_firing, all_unit_electrodes)) \
        if firing.shape[1] == len(electrodes)]

fin_firing = [all_firing[x] for x in matchings_inds]
fin_unit_electrodes = [all_unit_electrodes[x] for x in matchings_inds]

gc_unit_electrodes = [np.where(np.isin(unit_electrodes, this_gc_electrodes))[0]\
        for unit_electrodes, this_gc_electrodes \
        in zip(fin_unit_electrodes, gc_electrodes)]
bla_unit_electrodes = [np.where(np.isin(unit_electrodes, this_bla_electrodes))[0]\
        for unit_electrodes, this_bla_electrodes \
        in zip(fin_unit_electrodes, bla_electrodes)]

gc_nrn_list = [x[0][:,x[1]] for x in zip(fin_firing, gc_unit_electrodes)]
bla_nrn_list = [x[0][:,x[1]] for x in zip(fin_firing, bla_unit_electrodes)]

gc_nrn_array = np.concatenate(gc_nrn_list, axis=1).swapaxes(0,1) 
gc_array_long = gc_nrn_array.reshape((gc_nrn_array.shape[0],-1,
                                        gc_nrn_array.shape[-1]))
bla_nrn_array = np.concatenate(bla_nrn_list, axis=1).swapaxes(0,1)
bla_array_long = bla_nrn_array.reshape((bla_nrn_array.shape[0],-1,
                                        bla_nrn_array.shape[-1]))

# Use paired t-tests to find all neurons whose firing rates
# significantly chaned in 0-250ms post-stimulus delivery compared to baseline
baseline_inds = np.arange(60)
stim_inds = np.arange(80,90)
gc_baseline = np.mean(gc_array_long[...,baseline_inds],axis=-1)
gc_stim = np.mean(gc_array_long[...,stim_inds],axis=-1)
bla_baseline = np.mean(bla_array_long[...,baseline_inds],axis=-1)
bla_stim = np.mean(bla_array_long[...,stim_inds],axis=-1)

gc_ttest = [ttest_rel(base,stim) for base,stim in zip(gc_baseline, gc_stim)]
gc_tstat , gc_pvals = np.array([x[0] for x in gc_ttest]), \
                        np.array([x[1] for x in gc_ttest])
bla_ttest = [ttest_rel(base,stim) for base,stim in zip(bla_baseline, bla_stim)]
bla_tstat , bla_pvals = np.array([x[0] for x in bla_ttest]), \
                        np.array([x[1] for x in bla_ttest])

alpha = 0.05
# Plot cumulative distribution of p-values for both regions
percentile_num = 100
gc_pval_hist = np.histogram(np.log10(gc_pvals), percentile_num)
bla_pval_hist = np.histogram(np.log10(bla_pvals), percentile_num)
gc_ecdf = np.cumsum(gc_pval_hist[0])/np.sum(gc_pval_hist[0])
bla_ecdf = np.cumsum(bla_pval_hist[0])/np.sum(bla_pval_hist[0])

plt.plot(gc_pval_hist[1][1:], gc_ecdf)
plt.plot(bla_pval_hist[1][1:], bla_ecdf)
plt.show()

# Plot distribution of t-statistics for significant neurons
hist_bins = 20
plt.hist(gc_tstat[gc_pvals<alpha], hist_bins, density = True, alpha = 0.5)
plt.hist(bla_tstat[bla_pvals<alpha],hist_bins, density = True, alpha = 0.5)
plt.show()

# Fraction of neurons with significant response
# And average absolute t-statistic
fig, ax = plt.subplots(2,1)
ax[0].boxplot([np.abs(gc_tstat[gc_pvals < alpha]),
                np.abs(bla_tstat[bla_pvals < alpha])])
ax[0].set_xticklabels(['GC','BLA'])
plt.suptitle('Ratio of signficant responses (alpha = {}) :\n'\
        'GC : {}/{} ({:.3}) ,  BLA : {}/{} ({:.3})'.format(\
                alpha,
                np.sum(gc_pvals<alpha),
                len(gc_pvals),
                np.mean(gc_pvals<alpha), 
                np.sum(bla_pvals<alpha),
                len(bla_pvals),
                np.mean(bla_pvals<alpha)))
ax[0].set_title('Abs t-statistic for significant units')
#plt.sca(ax[1])
ax[1].hist(np.abs(gc_tstat[gc_pvals < alpha]), 
        hist_bins, density = True, alpha = 0.5)
ax[1].hist(np.abs(bla_tstat[bla_pvals < alpha]), 
        hist_bins, density = True, alpha = 0.5)
plt.legend(['GC','BLA'])
plt.show()
