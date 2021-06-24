"""
Use ELBO to compare changepoint fits with
different numbers of states
1) On a per-recording basis
2) By aggregating across recordings for a particular
    class of recordings (e.g. BLA neurons)
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

import numpy as np
from matplotlib import pyplot as plt
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

#file_list_path = '/media/bigdata/firing_space_plot/changepoint_mcmc/'\
#                        'file_lists/bla_model_comparison.txt'
#analysis_title = 'bla_state_comparison'                    

with open(file_list_path,'r') as this_file:
    file_list = this_file.read().splitlines()

# For every file in file list, search for models with the 
# same parameters but different numbers of states
search_pattern = [os.path.basename(x).split('states')[1] \
                        for x in file_list]
search_dirs = ["/".join(x.split('/')[:-2]) for x in file_list]
session_names = [x.split('/')[-4] for x in file_list]

all_model_list = [glob(os.path.join(this_dir,"*","*"+this_pattern)) \
        for this_dir,this_pattern in zip(search_dirs,search_pattern)]
# Remove shuffle and simulate sets
def keep_file_bool(x):
    if ('shuffle' in x) or ('simulate' in x):
        keep_bool = False
    else:
        keep_bool = True
    return keep_bool

all_model_list = [[this_file for this_file in x if keep_file_bool(this_file)]\
                        for x in all_model_list]
all_model_list = [sorted(x) for x in all_model_list]

states = [[int(re.findall("\d+states",x)[0][:-6]) for x in this_model_set]\
        for this_model_set in all_model_list]
all_elbo_list = []

# Hardcode output folder for aggregated analysis
aggregrate_out_dir = '/media/bigdata/Abuzar_Data/elbo_model_comparison'
if not os.path.exists(aggregrate_out_dir):
        os.makedirs(aggregrate_out_dir)

# Get elbo from each model
#set_num = 0
for set_num in trange(len(all_model_list)):
    this_model_set = all_model_list[set_num]
    this_states = states[set_num]

    data_list = [pickle.load(open(this_path,'rb')) for this_path in this_model_set]
    fit_list = [this_data['approx'] for this_data in data_list]
    elbo_list = [-this_fit.hist for this_fit in fit_list]
    # Remove pickled data to conserve memory
    del data_list

    # Create plots
    this_plot_dir = os.path.join(os.path.dirname(search_dirs[set_num]),
                    'changepoint_plots')
    if not os.path.exists(this_plot_dir):
        os.makedirs(this_plot_dir)

    wanted_fraction = 0.05
    wanted_ind = int(len(elbo_list[0])*(1-wanted_fraction))
    end_elbo_list = [this_elbo[wanted_ind:] for this_elbo in elbo_list] 
    median_end_elbo = [np.median(x) for x in end_elbo_list]
    rank_order = np.array(this_states)[np.argsort(median_end_elbo)]

    all_elbo_list.append(median_end_elbo)

    fig,ax = plt.subplots(1,3,figsize=(15,5))
    for num,this_elbo in enumerate(elbo_list):
        ax[0].plot(this_elbo, label=this_states[num], alpha = 0.5)
        ax[1].plot(end_elbo_list[num], label = this_states[num], alpha = 0.5)
        ax[2].plot(this_states,median_end_elbo,'-x',c='blue')
        ax[2].scatter(this_states[np.argmax(median_end_elbo)], np.max(median_end_elbo),
                            facecolors='none',edgecolors='r',s=100)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Elbo trace')
    ax[1].set_title(f'Final {100*wanted_fraction}% of ELBO trace')
    ax[2].set_title('Final ELBO per state'+f'\nRank Order : {rank_order}')
    for num,val in enumerate(median_end_elbo):
        ax[2].text(this_states[num],val, int(val),fontweight='bold')
    set_title = search_pattern[set_num].split('.')[0][1:]
    plt.suptitle(set_title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(this_plot_dir, set_title + "_state_comparison"))
    plt.close(fig)
    #plt.show()

all_elbo_ranks = np.array([np.array(this_states)[np.argsort(this_elbo)] \
                    for this_states,this_elbo in zip(states,all_elbo_list)])

inds = np.array(list(np.ndindex(all_elbo_ranks.shape)))
rank_frame = pd.DataFrame({
                'rank' : inds[:,1],
                'session' : inds[:,0],
                'state' : all_elbo_ranks.flatten()})

fig,ax = plt.subplots(2,1)
#g = sns.violinplot(x='state',y='rank',data=rank_frame,ax=ax[1],inner=None)
g = sns.swarmplot(x='state',y='rank',data=rank_frame,ax=ax[1])#,
        #color="white", edgecolor="gray")
plt.suptitle(f'{analysis_title}\n{set_title}')
ax[0].text(0.01,0.1,"\n".join(session_names),wrap=True)
ax[1].set_xlabel('Number of states')
ax[1].set_ylabel('Model Rank')
fig.savefig(os.path.join(aggregrate_out_dir, analysis_title))
plt.close(fig)
#plt.show()
