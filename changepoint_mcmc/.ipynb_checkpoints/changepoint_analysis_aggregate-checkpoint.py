"""
Aggregate comparison of magnitude of transitions in actual data vs.
shuffled and simulated data
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
import pymc3 as pm
import re
from glob import glob
from tqdm import tqdm,trange
import tables
from shutil import copyfile

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
import pickle
import argparse
import pandas as pd
import seaborn as sns

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

############################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
############################################################

parser = argparse.ArgumentParser(description = 'Script to compare models '\
                            'with different numbers of states')
parser.add_argument('file_list',  help = 'models to perform analysis on')
parser.add_argument('title',  help = 'name to give to this comparison')
args = parser.parse_args()
file_list_path = args.file_list 
analysis_title = args.title 

save_dir = os.path.join('/media/bigdata/Abuzar_Data','aggregate_analysis',
                                analysis_title)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#file_list_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/file_lists/bla_only_files_for_analysis.txt'
#analysis_title = 'bla_only_comparison'
firing_anova_list = []
single_unit_diff_list = []
pop_diff_list = []
session_name_list = []

with open(file_list_path,'r') as this_file:
    file_list = this_file.read().splitlines()
copyfile(file_list_path, os.path.join(save_dir,'file_list.txt'))

# Get per-transition firing rate change arrays from each file
for file_num in range(len(file_list)):
    #file_num = 0
    this_file = file_list[file_num]
    model_name = os.path.basename(this_file).split('.')[0]
    data_dir = "/".join(this_file.split('/')[:-3])
    session_name = os.path.basename(data_dir)
    session_name_list.append(session_name)
    hdf5_path = glob(os.path.join(data_dir,"*h5"))[0]

    analysis_save_path = os.path.join('/changepoint_analysis/',model_name)

    with tables.open_file(hdf5_path,'r') as hf5:
        single_unit_diff_list.append(\
                hf5.get_node(\
                    os.path.join(\
                        analysis_save_path,'single_nrn_abs_difference'))[:])
        pop_diff_list.append(\
                hf5.get_node(\
                    os.path.join(\
                        analysis_save_path,'population_abs_difference'))[:])
        this_frame = \
            pd.read_hdf(hdf5_path,
                    os.path.join(analysis_save_path,'firing_rate_comparison'))
        this_frame['session_name'] = session_name
        firing_anova_list.append(this_frame)

fin_firing_frame = pd.concat(firing_anova_list)

# Plot distribution of p-values for firing rate comparisons
thresh = 0.05
plt.hist(np.log10(fin_firing_frame.pval))
plt.xlabel('Log10 transformed p-values')
plt.ylabel('Frequency')
plt.gcf().suptitle('Actual vs Shuffle firing rate comparison'\
    f'\n{sum(fin_firing_frame.pval < thresh)}/{len(fin_firing_frame)} < {thresh}')
plt.axvline(np.log10(thresh), color = 'red', alpha = 0.5, linewidth = 5)
plt.savefig(os.path.join(save_dir,'actual_vs_simulated_firing_comparison'))
plt.close()
#plt.show()

# Calculate average number of times actual data transitions were
# larger than controls
# POPULATION
# Don't perform as array to allow cases with unequal numbers of states
pop_greater_list = [(x[0]>x)[1:] for x in pop_diff_list]
mean_pop_greater_array = np.array([[np.mean(x,axis=None) for x in this_list] \
                            for this_list in pop_greater_list])

inds = np.array(list(np.ndindex(mean_pop_greater_array.shape)))
pop_greater_frame = pd.DataFrame({
                        'session' : np.array(session_name_list)[inds[:,0]],
                        'type' : np.array(['Shuffled','Simulated'])[inds[:,1]],
                        'val' : mean_pop_greater_array.flatten()})

sns.boxplot(data=pop_greater_frame, x = 'type', y = 'val',
        boxprops=dict(alpha=.5))
sns.swarmplot(data=pop_greater_frame, x = 'type', y = 'val', 
        s = 10, linewidth = 1, edgecolor = 'gray')
plt.xlabel('Control Type')
plt.ylabel('Fraction of changes smaller than ACTUAL DATA')
plt.gcf().suptitle('Comparison of magnitude of peri-transition \n'\
        'population vector difference \n between actual data and controls')
plt.savefig(os.path.join(save_dir,'population_transition_comparison'))
plt.close()
#plt.show()


# SINGLE NEURON
single_unit_greater_array = np.concatenate(\
                [(x[0]>x)[1:] for x in single_unit_diff_list],axis=-1)
mean_single_unit_greater = np.mean(single_unit_greater_array,axis=(1,2))
inds = np.array(list(np.ndindex(mean_single_unit_greater.shape)))
single_unit_greater_frame = pd.DataFrame({
                        'type' : np.array(['Shuffled','Simulated'])[inds[:,0]],
                        'neuron': inds[:,1],
                        'val' : mean_single_unit_greater.flatten()})

#n,bins = np.hist(mean_single_unit_greater[0])
bins = np.linspace(0,1,12)
cmap = plt.get_cmap("tab10")
fig,ax = plt.subplots(2,1, sharex=True, figsize=(5,5))
for num, (this_dat,this_ax) in enumerate(zip(mean_single_unit_greater,ax)):
    n,bins,patches = this_ax.hist(
                            this_dat,bins, 
                            color = cmap(num), histtype='step')
    n,bins,patches = this_ax.hist(
                            this_dat,bins, 
                            color = cmap(num))
    mid_ind = np.where(bins>0.5)[0][0]-1
    this_fc = patches[mid_ind].get_fc()
    pre_fc = np.array(this_fc) 
    pre_fc[-1] = 0.2
    mid_fc = np.array(this_fc)
    mid_fc[-1] = 0.5
    post_fc = np.array(this_fc)
    post_fc[-1] = 0.8
    patches[mid_ind].set_fc(mid_fc)
    for this_patch in patches[:mid_ind]:
        this_patch.set_fc(pre_fc)
    for this_patch in patches[(mid_ind+1):]:
        this_patch.set_fc(post_fc)
    y_val = np.max(n)/2
    text_kwargs = {'fontweight' : 'bold', 
            'horizontalalignment' : 'center', 'size' : 'x-large'}
    t1 = this_ax.text(0.25,y_val, f'{int(np.mean(this_dat < 0.5)*100)}%',
            color = pre_fc, **text_kwargs) 
    t2 = this_ax.text(0.5,y_val, f'{int(np.mean(this_dat == 0.5)*100)}%',
            color = mid_fc, **text_kwargs)
    t3 = this_ax.text(0.75,y_val, f'{int(np.mean(this_dat > 0.5)*100)}%',
            color = post_fc, **text_kwargs) 
    for this_t in [t1,t2,t3]:
        this_t.set_path_effects([path_effects.Stroke(linewidth=1, 
                                    foreground='black'),
                                   path_effects.Normal()])
    #this_ax.axvline(0.5, alpha = 0.7, color = 'red', linewidth = 2)
    this_ax.set_title(np.array(['Shuffled','Simulated'])[num])
    this_ax.set_ylabel('Frequency')
plt.xlim([0,1])
ax[1].set_xlabel('Fraction of sharper changepoints relative to control')
plt.savefig(os.path.join(save_dir,'single_unit_transition_comparison'))
plt.close()
#plt.show()


