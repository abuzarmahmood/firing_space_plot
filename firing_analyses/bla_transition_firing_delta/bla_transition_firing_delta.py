"""
Magnitude of change of BLA single neurons and population vectors across transitions
For:
    1) Actual Data
    2) Trial shuffled data
    3) Spike shuffled data

Plot outputs show:
    Number of transitions for which delta_firing of actual data > shuffle.
    Plots show, fraction of unit of comparison (single nrns/populations),
    for which AVG NUMBER OF TRANSITIONS is lower, equal, or higher
"""

import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/projects/pytau')
from ephys_data import ephys_data
import visualize as vz
from pytau.changepoint_io import DatabaseHandler
from pytau.changepoint_analysis import PklHandler, get_state_firing
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import os
#from joblib import Parallel, cpu_count, delayed
import seaborn as sns
from scipy.stats import zscore
import numpy as np
import pingouin as pg
import matplotlib as mpl
import matplotlib.patches as mpatches
from  matplotlib import colors
import itertools as it
from scipy.stats import mannwhitneyu as mwu
import pymc3 as pm
import xarray as xr


plot_dir = '/media/bigdata/firing_space_plot/'\
        'firing_analyses/bla_transition_firing_delta' 

file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in file_list]

data_dir = '/media/bigdata/projects/pytau/pytau/analyses'

fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()
#
dframe = fit_database.fit_database

wanted_exp_name = 'bla_population_elbo_repeat'
exp_name_bool = dframe['exp.exp_name'] == wanted_exp_name
state_num_bool = dframe['model.states'] == 4
pm_bool = dframe['module.pymc3_version'] == pm.__version__
bool_stack = np.array((exp_name_bool, state_num_bool, pm_bool))
fin_bool = np.logical_and.reduce(bool_stack,axis=0)

wanted_frame = dframe.loc[fin_bool] 

# Make sure every session has all 3 conditions
wanted_frame = pd.concat(\
    [x for num,x in wanted_frame.groupby('data.basename') if x.shape[0]==3])
wanted_grouped_frame = [x[1] for x in list(wanted_frame.groupby('data.basename'))]

############################################################
## Get state-aligned firing
############################################################
wanted_cols = ['preprocess.data_transform', 'data.basename','exp.save_path']
window_radius = 500 # ms, before and after

all_diffs = []

for this_group_temp in tqdm(wanted_grouped_frame):
    #this_group = wanted_grouped_frame[0][wanted_cols]
    this_group = this_group_temp[wanted_cols]

    session_diffs = {}
    for version_ind in range(len(this_group)):
        #version_ind = 0
        pkl_path = this_group['exp.save_path'].iloc[version_ind]
        this_transform = this_group['preprocess.data_transform'].iloc[version_ind]
        this_handler = PklHandler(pkl_path) 
        this_dat = ephys_data(this_handler.metadata['data']['data_dir'])
        this_dat.get_info_dict()
        taste_num = len(this_dat.info_dict['taste_params']['concs'])
        #bin_width = this_handler.metadata['preprocess']['bin_width']
        #window_radius_inds = window_radius//bin_width
        scaled_tau = this_handler.tau.scaled_mode_tau
        spike_array_long = np.concatenate(this_handler.firing.raw_spikes, axis=0) 
        snippets_lims = np.moveaxis(
                np.stack([
                    scaled_tau - window_radius, scaled_tau + window_radius]),
                0,-1)

        # No need to worry about boundary problems 
        # because t_lims for spikes is [0,7000]
        snippet_array = np.empty((
                            *spike_array_long.shape[:2], 
                            snippets_lims.shape[1],
                            window_radius*2))
        snippet_array = snippet_array.swapaxes(1,2)

        inds = list(np.ndindex(snippet_array.shape[:-2]))
        # Ind : trial x changepoint 
        for this_ind in tqdm(inds):
            #this_ind = inds[0]
            this_tlims = snippets_lims[this_ind]
            this_snippet = \
                    spike_array_long[this_ind[0],:,this_tlims[0]:this_tlims[1]]
            snippet_array[this_ind] = this_snippet

        split_snippets = np.stack(np.split(snippet_array,2, axis=-1))
        sum_snippets = split_snippets.sum(axis=-1)
        diff_snippets = np.squeeze(np.abs(np.diff(sum_snippets,axis=0)))
        taste_diff_snippets = np.stack(np.split(diff_snippets, taste_num, axis=0))
        # Taste x Changepoints x Neurons
        mean_diff_snippets = np.mean(taste_diff_snippets,axis=1)
        session_diffs[this_transform] = mean_diff_snippets
    all_diffs.append(session_diffs)

############################################################
## Compare None condition to transformed datasets 
############################################################
inds_list = [['None','trial_shuffled'],['None','spike_shuffled']]

mpl.rcParams.update({'font.size': 16})
mpl.rc('axes', titlesize=13, labelsize = 13)

########################################
## Single neurons analysis
########################################

## Compare magnitude of transition strength across conditions
# Convert all_diffs to dataframe for convenience
############################################################

frame_list = []
for session_num, session in tqdm(enumerate(all_diffs)):
    for shuffle_type, shuffle_values in session.items():
        this_inds = np.array(list(np.ndindex(shuffle_values.shape)))
        this_frame = pd.DataFrame(dict(
            session_num = session_num,
            shuffle_type = shuffle_type,
            taste_num = this_inds[:,0],
            transition_num = this_inds[:,1],
            nrn_num = this_inds[:,2],
            diff_mag = shuffle_values.flatten()
            ))
        frame_list.append(this_frame)

diff_frame = pd.concat(frame_list)
diff_frame = diff_frame.groupby(['session_num','shuffle_type','nrn_num'])\
        .mean().reset_index()

# Compare zscored values for session x neuron
#grouped_frame = list(diff_frame.groupby(['session_num','taste_num',
#                                    'transition_num','nrn_num']))
grouped_frame = list(diff_frame.groupby(['session_num','nrn_num']))
for keys, frame in grouped_frame:
    wanted_val = frame[frame['shuffle_type'] == 'None']['diff_mag'].values
    frame['norm_mag'] = frame['diff_mag']/wanted_val
    #frame['zscore_mag'] = zscore(frame['diff_mag']) 
diff_frame = pd.concat([x[1] for x in grouped_frame])
diff_frame['fold_change'] = diff_frame['norm_mag'] - 1

hue_order = ['None','trial_shuffled','spike_shuffled']
wanted_xlabels = ['None','Trial Shuffled','Spike Shuffled']
sns.stripplot(data = diff_frame,
        #x = 'shuffle_type', y = 'fold_change',
        x = 'shuffle_type', y = 'norm_mag',
        edgecolor = 'black', linewidth = 1, alpha = 0.5,
        order = hue_order
        )
sns.boxplot(data = diff_frame,
        #x = 'shuffle_type', y = 'fold_change', showfliers = False,
        x = 'shuffle_type', y = 'norm_mag', showfliers = False,
        order = hue_order)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison_box'))
plt.close(fig)
#plt.show()

out = sns.displot(data = diff_frame.reset_index(drop=True), x = 'norm_mag',
        row = 'shuffle_type', facet_kws=dict(sharey=False),
        hue = 'shuffle_type', kde = True)
for this_ax in out.axes.flatten()[:2]:
    x,y = this_ax.lines[0].get_data()
    mode_ind = np.argmax(y)
    this_ax.set_title(f'Mode : {np.round(x[mode_ind],3)}')
    this_ax.set_ylim([0.1, 80])
    this_ax.set_xlim([0, 2])
plt.suptitle('Single Neuron comparison')
plt.subplots_adjust(top = 0.9)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison_hist'))
plt.close(fig)
#plt.show()

out = sns.displot(data = diff_frame.reset_index(drop=True), x = 'norm_mag',
        row = 'shuffle_type', facet_kws=dict(sharey=False),
        hue = 'shuffle_type', kde = True)
for this_ax in out.axes.flatten()[:2]:
    x,y = this_ax.lines[0].get_data()
    mode_ind = np.argmax(y)
    #this_ax.axvline(mode_ind, color = 'red', linestyle = '--', linewidth = 2, zorder = 2)
    this_ax.set_title(f'Mode : {np.round(x[mode_ind],3)}')
    this_ax.set_yscale('log')
    this_ax.set_xlim([0,2])
    this_ax.set_ylim([0.1, 80])
plt.suptitle('Single Neuron comparison Log')
plt.subplots_adjust(top = 0.9)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison_hist_log'))
plt.close(fig)
#plt.show()

## Compare number of transitions which were sharper
############################################################
session_list = []
for session in tqdm(all_diffs):
    #session = all_diffs[0]
    comp_list = []
    for this_comp in inds_list:
        #this_comp = inds_list[0]
        # Condition x Taste x Changepoints x Neurons
        wanted_sets = np.stack([session[x] for x in this_comp]) 
        # For each neuron, whether ON AVERAGE it had sharper transition
        # across (Tastes x Changepoints)
        mean_nrn_frac = (wanted_sets[0] > wanted_sets[1]).mean(axis=(0,1))
        comp_list.append(mean_nrn_frac)
    session_list.append(comp_list)

flat_session_list = list(zip(*session_list))
flat_session_list = [[x for y in z for x in y] for z in flat_session_list]

labels = [[str(x)]*len(y) for x,y in zip(inds_list, flat_session_list)]
single_nrn_frame = pd.DataFrame(dict(
    label = [x for y in labels for x in y],
    values = [x for y in flat_session_list for x in y]))

# Cut into bins of lower than 0.5, 0.5, and higher than 0.5
bin_labels = ['Lower','Equal','Higher']
single_nrn_frame['binned_values'] = \
        pd.cut(single_nrn_frame['values'], 
                bins = [0,0.495,0.505,1], labels = bin_labels) 

bin_fracs = [x.groupby('binned_values')['values'].count() / len(x) \
        for num,x in list(single_nrn_frame.groupby('label'))]

fig,ax = plt.subplots(2,1, figsize = (7,15))
for num in range(len(ax)):
    ax[num].pie(bin_fracs[num], labels = bin_labels, autopct='%.1f%%')
    ax[num].set_title(inds_list[num][1])
plt.suptitle('Single Neuron Comparisons' + '\n' + wanted_exp_name + '\n' +\
                f'Total : {len(flat_session_list[0])} Neurons')
plt.subplots_adjust(top = 0.8)
#plt.show()
fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison'))
plt.close(fig)

#fig,ax = plt.subplots()
#sns.stripplot(data = single_nrn_frame,
#        x = 'label', y = 'values', ax=ax,
#        edgecolor = 'black', linewidth = 1,
#        jitter=  True)
#sns.violinplot(data = single_nrn_frame,
#        x = 'label', y = 'values', ax=ax)
#ax.set_title('Single neuron distribution')
#ax.axhline(0.5, color = 'red', linestyle = '--', linewidth = 2)
#plt.show()

#fig,ax = plt.subplots(2,1, sharex=True, sharey=True)
#for num in range(len(ax)):
#    this_ax = ax[num]
#    this_dat = flat_session_list[num]
#    this_title = inds_list[num]
#    this_ax.hist(this_dat, bins = 20)
#    this_ax.set_title(this_title)
#plt.suptitle('Single Neuron analysis')
#plt.show()

########################################
## Population analysis
########################################

## Compare magnitude of transition strength across conditions
# Convert all_diffs to dataframe for convenience
############################################################

# Normalize single neurons across conditions 
pop_grouped_frame = list(diff_frame.groupby(['session_num','nrn_num']))

for keys, frame in pop_grouped_frame:
    frame['zscore_mag'] = zscore(frame['diff_mag']) 

pop_diff_frame = pd.concat([x[1] for x in pop_grouped_frame])

# Calculate population vector magnitude
pop_mag_frame = pop_diff_frame.groupby(
        ['session_num','shuffle_type','taste_num','transition_num'])\
                .agg(np.linalg.norm).reset_index()
pop_mag_frame.drop(columns = 'norm_mag', inplace=True)

# Average for populations
pop_mag_frame = pop_mag_frame.\
        groupby(['session_num','shuffle_type']).\
        mean().reset_index()

# Normalize population to None condition
#grouped_frame = list(pop_mag_frame.groupby(['session_num','taste_num',
#                                    'transition_num']))
grouped_frame = list(pop_mag_frame.groupby(['session_num']))

for keys, frame in grouped_frame:
    wanted_val = frame[frame['shuffle_type'] == 'None']['zscore_mag'].values
    frame['pop_zscore_mag'] = frame['zscore_mag']/wanted_val

fin_pop_frame = pd.concat([x[1] for x in grouped_frame])

out = sns.displot(data = fin_pop_frame, x = 'pop_zscore_mag',
        row = 'shuffle_type', facet_kws=dict(sharey=False),
        hue = 'shuffle_type', kde = True, bins = 20)
for this_ax in out.axes.flatten()[1:]:
    x,y = this_ax.lines[0].get_data()
    mode_ind = np.argmax(y)
    this_ax.set_title(f'Mode : {np.round(x[mode_ind],3)}')
    #this_ax.set_ylim([0.1, 15])
    #this_ax.set_xlim([0, 2])
plt.suptitle('Population comparison')
plt.subplots_adjust(top = 0.9)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'population_comparison_hists'))
plt.close(fig)
#plt.show()

sns.swarmplot(data = fin_pop_frame, x = 'shuffle_type', y = 'pop_zscore_mag',
        edgecolor = 'black', linewidth = 1, order = hue_order)
sns.boxplot(data = fin_pop_frame, x = 'shuffle_type', y = 'pop_zscore_mag', order = hue_order)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'population_comparison_swarm.png'))
plt.close(fig)
#plt.show()


## Compare number of transitions which were sharper
############################################################
pop_session_list = []
for session in tqdm(all_diffs):
    #session = all_diffs[0]
    comp_list = []
    for this_comp in inds_list:
        #this_comp = inds_list[0]
        # Condition x Taste x Changepoints x Neurons
        wanted_sets = np.stack([session[x] for x in this_comp]) 
        zscored_sets = np.stack(
                [zscore(x,axis=None) for x in wanted_sets.T]).T
        # Calculate population vector magnitude for each transition
        zscored_mag = np.linalg.norm(zscored_sets, axis=-1)
        # For each neuron, whether ON AVERAGE it had sharper transition
        # across (Tastes x Changepoints)
        mean_pop_frac = (zscored_mag[0] > zscored_mag[1]).mean(axis=(0,1))
        comp_list.append(mean_pop_frac)
    pop_session_list.append(comp_list)

pop_flat_session_list = list(zip(*pop_session_list))

labels = [[str(x)]*len(y) for x,y in zip(inds_list, pop_flat_session_list)]
population_frame = pd.DataFrame(dict(
    label = [x for y in labels for x in y],
    values = [x for y in pop_flat_session_list for x in y]))

# Cut into bins of lower than 0.5, 0.5, and higher than 0.5
bin_labels = ['Lower','Equal','Higher']
population_frame['binned_values'] = \
        pd.cut(population_frame['values'], 
                bins = [0,0.495,0.505,1], labels = bin_labels) 

bin_fracs = [x.groupby('binned_values')['values'].count() / len(x) \
        for num,x in list(population_frame.groupby('label'))]

fig,ax = plt.subplots(2,1, figsize = (7,15))
for num in range(len(ax)):
    ax[num].pie(bin_fracs[num], labels = bin_labels, autopct='%.1f%%')
    ax[num].set_title(inds_list[num][1])
plt.suptitle('Population Comparisons' + '\n' + wanted_exp_name + '\n' +\
                f'Total : {len(pop_flat_session_list[0])} Populations')
plt.subplots_adjust(top = 0.8)
#plt.show()
fig.savefig(os.path.join(plot_dir, 'population_comparison'))
plt.close(fig)
