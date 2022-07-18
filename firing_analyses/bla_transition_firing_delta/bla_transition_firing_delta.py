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
from scipy import stats
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

# Thin out recordings to avoid over-representation bias
wanted_counts = [4,5]
wanted_basenames = []
for num, val in enumerate(wanted_counts):
    this_basenames = wanted_frame.groupby('data.animal_name')\
            ['data.basename'].unique()[num][:val]
    wanted_basenames.append(this_basenames)
wanted_basenames = [x for y in wanted_basenames for x in y]
wanted_frame = wanted_frame[wanted_frame['data.basename'].isin(wanted_basenames)]

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

#############################################################
#############################################################
#hue_order = ['None','trial_shuffled','spike_shuffled']
hue_order = ['trial_shuffled','spike_shuffled']
wanted_types = ['trial_shuffled','spike_shuffled']
#wanted_xlabels = ['None','Trial Shuffled','Spike Shuffled']
wanted_xlabels = ['Trial Shuffled','Spike Shuffled']
wanted_data = diff_frame[diff_frame.shuffle_type.isin(wanted_types)]
fig,ax = plt.subplots(figsize = (5,5))
#sns.stripplot(data = wanted_data,
#        #x = 'shuffle_type', y = 'fold_change',
#        x = 'shuffle_type', y = 'norm_mag',
#        edgecolor = 'black', linewidth = 1, alpha = 0.5,
#        order = hue_order, ax=ax
#        )
sns.boxplot(data = wanted_data,
        #x = 'shuffle_type', y = 'fold_change', showfliers = False,
        x = 'shuffle_type', y = 'norm_mag', showfliers = False,
        order = hue_order, ax=ax)
ax.set_xlabel('Shuffle Type')
ax.set_ylabel('Normalized Change')
ax.axhline(1, color = 'red', linewidth = 4, linestyle = '--')
#ax.set_xticks(ticks = [0,1], 
ax.set_xticklabels(labels = wanted_xlabels)
#fig = plt.gcf()
ax.set_ylim([0.5,1.2])
plt.tight_layout()
#fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison_box'))
#plt.close(fig)
plt.show()

###############################################################
##############################################################
#out = sns.displot(data = diff_frame.reset_index(drop=True), x = 'norm_mag',
#        row = 'shuffle_type', facet_kws=dict(sharey=False),
#        hue = 'shuffle_type', kde = True)
#for this_ax in out.axes.flatten()[:2]:
#    x,y = this_ax.lines[0].get_data()
#    mode_ind = np.argmax(y)
#    this_ax.set_title(f'Mode : {np.round(x[mode_ind],3)}')
#    this_ax.set_ylim([0.1, 80])
#    this_ax.set_xlim([0, 2])
#plt.suptitle('Single Neuron comparison')
#plt.subplots_adjust(top = 0.9)
#fig = plt.gcf()
#fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison_hist'))
#plt.close(fig)
##plt.show()
#
##############################################################
##############################################################
#out = sns.displot(data = diff_frame.reset_index(drop=True), x = 'norm_mag',
#        row = 'shuffle_type', facet_kws=dict(sharey=False),
#        hue = 'shuffle_type', kde = True)
#for this_ax in out.axes.flatten()[:2]:
#    x,y = this_ax.lines[0].get_data()
#    mode_ind = np.argmax(y)
#    #this_ax.axvline(mode_ind, color = 'red', linestyle = '--', linewidth = 2, zorder = 2)
#    this_ax.set_title(f'Mode : {np.round(x[mode_ind],3)}')
#    this_ax.set_yscale('log')
#    this_ax.set_xlim([0,2])
#    this_ax.set_ylim([0.1, 80])
#plt.suptitle('Single Neuron comparison Log')
#plt.subplots_adjust(top = 0.9)
#fig = plt.gcf()
#fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison_hist_log'))
#plt.close(fig)
##plt.show()

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
    values = [x for y in flat_session_list for x in y],
    nrn_ind = list(range(len(flat_session_list[0])))*len(flat_session_list)
    ))
    

# Cut into bins of lower than 0.5, 0.5, and higher than 0.5
bin_labels = ['Lower','Equal','Higher']
single_nrn_frame['binned_values'] = \
        pd.cut(single_nrn_frame['values'], 
                bins = [0,0.495,0.505,1], labels = bin_labels) 

############################################################
# Plot fraction of transitions faster for actual data vs shuffle
cond_order = ['None', 'trial_shuffled','spike_shuffled']
# Stack diffs and calc arg_max for each neuron
# diff_stack : condition x taste x transition x neuron
single_diff_stack = [np.stack(
            [
                this_diff[this_cond] for this_cond in cond_order
                ]
            )
            for this_diff in all_diffs
            ]

single_diff_frame_list = []
for num, this_diff in enumerate(single_diff_stack):
    inds = np.array(list(np.ndindex(this_diff.shape)))
    this_frame = pd.DataFrame(
            dict(
                session = num,
                cond = inds[:,0],
                tastes = inds[:,1],
                transition = inds[:,2],
                neuron = inds[:,3],
                diff_val = this_diff.flatten()
                )
            )
    single_diff_frame_list.append(this_frame)
single_fin_diff_frame = pd.concat(single_diff_frame_list).reset_index(drop=True)

#g = sns.boxplot(
#        data = single_fin_diff_frame,
#        x = 'cond',
#        y = 'diff_val'
#        )
#plt.show()


group_list = list(single_fin_diff_frame.groupby(
    ['session','tastes','transition','neuron']))
single_max_cond_list = []
for num, this_group in group_list: 
    max_ind = np.argmax(this_group.diff_val)
    max_cond = this_group.iloc[max_ind:max_ind+1]
    single_max_cond_list.append(max_cond)
single_max_cond_frame = pd.concat(single_max_cond_list)\
        .reset_index(drop=True).drop_duplicates()
single_max_cond_count = single_max_cond_frame.groupby(
        ['session','neuron','cond'])['tastes'].count().\
                reset_index()
single_max_cond_count.rename(columns = {'tastes':'count'}, inplace=True)

grouped_count_list = list(single_max_cond_count.groupby(
        ['session','neuron']
        ))
frac_frame_list = []
for num, this_frame in grouped_count_list:
    this_frame['frac'] = this_frame['count'] / this_frame['count'].sum()
    frac_frame_list.append(this_frame)
fin_frac_frame = pd.concat(frac_frame_list)
fin_frac_frame['id'] = fin_frac_frame.session.astype('str') \
                        + '_' + fin_frac_frame.neuron.astype('str')
single_fin_frac_frame = fin_frac_frame.copy()

single_fin_frac_frame.rm_anova(
        dv = 'frac',
        within = 'cond',
        subject = 'id'
        ).to_csv(os.path.join(plot_dir,'single_nrn_anova.csv'))

single_fin_frac_frame.pairwise_tukey(
        dv = 'frac',
        between = 'cond'
        ).to_csv(os.path.join(plot_dir,'single_nrn_tukey.csv'))


sns.barplot(
        data = single_fin_frac_frame,
        x = 'cond',
        hue = 'cond',
        y = 'frac',
        )
fig = plt.gcf()
plt.title('Single neuron fastest transition fraction')
fig.savefig(os.path.join(plot_dir,'single_neuron_best_frac.png'))
plt.close(fig)
#plt.show()

############################################################

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
[np.argmax(x.values) for a,x in fin_pop_frame.groupby('session_num')['pop_zscore_mag']]

#out = sns.displot(data = fin_pop_frame, x = 'pop_zscore_mag',
#        row = 'shuffle_type', facet_kws=dict(sharey=False),
#        hue = 'shuffle_type', kde = True, bins = 20)
#for this_ax in out.axes.flatten()[1:]:
#    x,y = this_ax.lines[0].get_data()
#    mode_ind = np.argmax(y)
#    this_ax.set_title(f'Mode : {np.round(x[mode_ind],3)}')
#    #this_ax.set_ylim([0.1, 15])
#    #this_ax.set_xlim([0, 2])
#plt.suptitle('Population comparison')
#plt.subplots_adjust(top = 0.9)
#fig = plt.gcf()
#fig.savefig(os.path.join(plot_dir, 'population_comparison_hists'))
#plt.close(fig)
##plt.show()

wanted_data = fin_pop_frame[fin_pop_frame.shuffle_type.isin(wanted_types)]
#sns.swarmplot(data = fin_pop_frame, x = 'shuffle_type', y = 'pop_zscore_mag',
#        edgecolor = 'black', linewidth = 1, order = hue_order)

fig,ax = plt.subplots(figsize = (5,5))
sns.boxplot(data = wanted_data, x = 'shuffle_type', y = 'pop_zscore_mag', 
        order = hue_order)
ax.set_xlabel('Shuffle Type')
ax.set_ylabel('Normalized Change')
ax.axhline(1, color = 'red', linewidth = 4, linestyle = '--')
#ax.set_xticks(ticks = [0,1], 
ax.set_xticklabels(labels = wanted_xlabels)
ax.set_ylim([0.5,1.2])
#fig = plt.gcf()
plt.tight_layout()
#fig.savefig(os.path.join(plot_dir, 'population_comparison_box.png'))
#plt.close(fig)
plt.show()

############################################################
## Put both boxplots on same figure
############################################################
from matplotlib.patches import PathPatch
def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    #for ax in g.axes:

    # iterating through axes artists:
    for c in ax.get_children():

        # searching for PathPatches
        if isinstance(c, PathPatch):
            # getting current width of box:
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5*(xmin+xmax)
            xhalf = 0.5*(xmax - xmin)

            # setting new width of box
            xmin_new = xmid-fac*xhalf
            xmax_new = xmid+fac*xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            # setting new width of median line
            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])


hue_order = ['trial_shuffled', 'spike_shuffled']
wanted_types = ['trial_shuffled','spike_shuffled']
wanted_xlabels = ['Trial Shuffled','Spike Shuffled']
single_wanted_data = diff_frame[diff_frame.shuffle_type.isin(wanted_types)]
pop_wanted_data = fin_pop_frame[fin_pop_frame.shuffle_type.isin(wanted_types)]

fin_single_data = single_wanted_data[['shuffle_type','norm_mag']]
fin_single_data['data_type'] = 'single_nrn'
fin_single_data.rename(columns = dict(norm_mag = 'val'), inplace=True)
fin_pop_data = pop_wanted_data[['shuffle_type','pop_zscore_mag']]
fin_pop_data['data_type'] = 'population'
fin_pop_data.rename(columns = dict(pop_zscore_mag = 'val'), inplace=True)

############################################################
## Significance tests 
single_dat = [1-x[1].val for x in list(fin_single_data.groupby('shuffle_type'))]
single_tests = [stats.wilcoxon(x) for x in single_dat]
with open(os.path.join(plot_dir, 'single_nrn_delta.txt'), 'w') as this_file:
    this_file.writelines([str(x)+'\n' for x in single_tests])
pop_dat = [1-x[1].val for x in list(fin_pop_data.groupby('shuffle_type'))]
pop_tests = [stats.wilcoxon(x) for x in pop_dat]
with open(os.path.join(plot_dir, 'pop_delta.txt'), 'w') as this_file:
    this_file.writelines([str(x)+'\n' for x in pop_tests])
############################################################


fin_plot_dat = pd.concat([fin_single_data, fin_pop_data], axis = 0)

fig,ax = plt.subplots(figsize = (5,3), linewidth = 10)
g = sns.boxplot(data = fin_plot_dat,
        #x = 'shuffle_type', y = 'fold_change', showfliers = False,
        x = 'data_type', y = 'val', showfliers = True,
        hue = 'shuffle_type', ax=ax,
        dodge=  True, width = 0.5, zorder = 2,
        #palette = 'Set2', linewidth = 1)
        palette =sns.color_palette()[1:3][::-1], linewidth = 2)
adjust_box_widths(g, 0.8)
ax.set_xlabel('Shuffle Type')
ax.set_ylabel('Normalized Change')
ax.axhline(1, color = 'red', linewidth = 4, 
        linestyle = '--', alpha = 0.7, zorder = 0)
ax.set_xticklabels(labels = ['Single Neuron', 'Population'])
ax.set_ylim([0.5,1.2])
plt.tight_layout()
plt.legend([],[], frameon=False)
#ax.legend(title = 'Shuffle Type', labels = ['Spike Shuffle','Trial Shuffle'])
#plt.show()
fig.savefig(os.path.join(plot_dir, 'joint_box_comparison.png'), dpi = 300)
plt.close(fig)

#fig,ax = plt.subplots(figsize = (5,5))
#sns.boxplot(data = wanted_data,
#        #x = 'shuffle_type', y = 'fold_change', showfliers = False,
#        x = 'shuffle_type', y = 'norm_mag', showfliers = False,
#        order = hue_order, ax=ax)
#ax.set_xlabel('Shuffle Type')
#ax.set_ylabel('Normalized Change')
#ax.axhline(1, color = 'red', linewidth = 4, linestyle = '--')
#ax.set_xticklabels(labels = wanted_xlabels)
#ax.set_ylim([0.5,1.2])
#plt.tight_layout()
##fig.savefig(os.path.join(plot_dir, 'single_nrn_comparison_box'))
##plt.close(fig)
#plt.show()
#
##sns.swarmplot(data = fin_pop_frame, x = 'shuffle_type', y = 'pop_zscore_mag',
##        edgecolor = 'black', linewidth = 1, order = hue_order)
#
#fig,ax = plt.subplots(figsize = (5,5))
#sns.boxplot(data = wanted_data, x = 'shuffle_type', y = 'pop_zscore_mag', 
#        order = hue_order)
#ax.set_xlabel('Shuffle Type')
#ax.set_ylabel('Normalized Change')
#ax.axhline(1, color = 'red', linewidth = 4, linestyle = '--')
##ax.set_xticks(ticks = [0,1], 
#ax.set_xticklabels(labels = wanted_xlabels)
#ax.set_ylim([0.5,1.2])
##fig = plt.gcf()
#plt.tight_layout()
##fig.savefig(os.path.join(plot_dir, 'population_comparison_box.png'))
##plt.close(fig)
#plt.show()


############################################################
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

############################################################
# Plot fraction of transitions faster for actual data vs shuffle
cond_order = ['None', 'trial_shuffled','spike_shuffled']
# Stack diffs and calc arg_max for each neuron
# all_diffs : dict : {cond} x Taste x Changepoints x Neurons
# diff_stack : neuron x transition x taste x condition 
pop_diff_stack = [np.stack(
            [
                this_diff[this_cond] for this_cond in cond_order
                ]
            ).T
            for this_diff in all_diffs
            ]

# Zscore for each neuron, taste, transition
pop_diff_stack = [
        [
            zscore(nrn,axis=None) for nrn in session
            ]
        for session in pop_diff_stack
        ]

# transition x taste x condition
pop_diff_mag_stack = [
        np.linalg.norm(x, axis=0) for x in pop_diff_stack
        ]

pop_diff_frame_list = []
for num, this_diff in enumerate(pop_diff_mag_stack):
    inds = np.array(list(np.ndindex(this_diff.shape)))
    this_frame = pd.DataFrame(
            dict(
                session = num,
                cond = inds[:,2],
                tastes = inds[:,1],
                transition = inds[:,0],
                diff_val = this_diff.flatten()
                )
            )
    pop_diff_frame_list.append(this_frame)
pop_diff_frame = pd.concat(pop_diff_frame_list).reset_index(drop=True)

#g = sns.boxplot(
#        data = pop_diff_frame,
#        x = 'cond',
#        y = 'diff_val'
#        )
#plt.show()

group_list = list(pop_diff_frame.groupby(
    ['session','tastes','transition']))
pop_max_cond_list = []
for num, this_group in group_list: 
    max_ind = np.argmax(this_group.diff_val)
    max_cond = this_group.iloc[max_ind:max_ind+1]
    pop_max_cond_list.append(max_cond)
pop_max_cond_frame = pd.concat(pop_max_cond_list)\
        .reset_index(drop=True).drop_duplicates()
pop_max_cond_count = pop_max_cond_frame.groupby(
        ['session','cond'])['tastes'].count().\
                reset_index()
pop_max_cond_count.rename(columns = {'tastes':'count'}, inplace=True)

grouped_count_list = list(pop_max_cond_count.groupby(
        ['session']
        ))
frac_frame_list = []
for num, this_frame in grouped_count_list:
    this_frame['frac'] = this_frame['count'] / this_frame['count'].sum()
    frac_frame_list.append(this_frame)
fin_frac_frame = pd.concat(frac_frame_list)
fin_frac_frame['id'] = fin_frac_frame.session.astype('str')
pop_fin_frac_frame = fin_frac_frame.copy()

pop_fin_frac_frame.rm_anova(
        dv = 'frac',
        within = 'cond',
        subject = 'id'
        ).to_csv(os.path.join(plot_dir,'pop_nrn_anova.csv'))

pop_fin_frac_frame.pairwise_tukey(
        dv = 'frac',
        between = 'cond'
        ).to_csv(os.path.join(plot_dir,'pop_nrn_tukey.csv'))


sns.barplot(
        data = pop_fin_frac_frame,
        x = 'cond',
        hue = 'cond',
        y = 'frac',
        )
fig = plt.gcf()
plt.title('Population neuron fastest transition fraction')
fig.savefig(os.path.join(plot_dir,'pop_neuron_best_frac.png'))
plt.close(fig)
#plt.show()

############################################################
# Aggreagate results in single plot
temp_pop_frame = pop_fin_frac_frame.copy()
temp_pop_frame['type'] = 'population'
temp_single_frame = single_fin_frac_frame.copy()
temp_single_frame['type'] = 'single'
agg_frame = pd.concat([temp_single_frame, temp_pop_frame])
pretty_cond_order = ['None','Trial Shuffled','Spike Shuffled']
agg_frame['cond'] = [pretty_cond_order[x] for x in agg_frame.cond] 
agg_frame.rename(
        columns = dict(
            cond = 'Shuffle Type',
            frac = 'Fraction',
            type = 'Comparison Level'
            ),
        inplace = True
        )
comparison_map = dict(
        single = 'Single Neuron',
        population = 'Population'
        )
agg_frame['Comparison Level'] = \
        [comparison_map[x] for x in agg_frame['Comparison Level']]

wanted_colors = [sns.color_palette()[0]]
wanted_colors.append(sns.color_palette()[2])
wanted_colors.append(sns.color_palette()[1])
fig,ax = plt.subplots(figsize = (5,3))
sns.barplot(
        data = agg_frame,
        x = 'Comparison Level',
        hue = 'Shuffle Type',
        palette = wanted_colors,
        y = 'Fraction',
        alpha = 0.7,
        capsize = 0.1,
        linewidth = 2,
        edgecolor = '.2',
        ax=ax
        )
fig = plt.gcf()
plt.title('Fraction of Dataset with fastest transition')
plt.ylim([0,0.7])
plt.legend('')
fig.savefig(os.path.join(plot_dir,'agg_best_frac.svg'))
plt.close(fig)
#plt.show()

############################################################
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
