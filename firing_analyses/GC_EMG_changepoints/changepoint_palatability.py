"""
Check palatability on a per-state basis
"""

## Import modules
base_dir = '/media/bigdata/projects/pytau'
import sys
sys.path.append(base_dir)
from pytau.changepoint_io import FitHandler
import pylab as plt
from pytau.utils import plotting
# from pytau.utils import ephys_data
from tqdm import tqdm
from pytau.changepoint_io import DatabaseHandler
from pytau.changepoint_analysis import PklHandler, get_transition_snips
import os
import pandas as pd
import numpy as np
from ast import literal_eval
# sys.path.append(os.path.expanduser('~/Desktop/blech_clust'))
# from blech_clust.utils import ephys_data
import json
from glob import glob
from scipy.stats import spearmanr

fit_database = DatabaseHandler()
fit_database.drop_duplicates()
fit_database.clear_mismatched_paths()

# Get fits for a particular experiment
dframe = fit_database.fit_database.copy()
wanted_exp_name = 'GC_EMG_changepoints_single_taste'
wanted_frame = dframe.loc[dframe['exp.exp_name'] == wanted_exp_name].copy() 
# Save as artifact
wanted_frame.to_pickle(os.path.join(artifact_dir, 'wanted_changepoint_frame.pkl'))

basename_list = wanted_frame['data.basename'].unique()
blacklist_basenames = [
       'KM50_5tastes_EMG_210911_104510_copy',
       'KM50_5tastes_EMG_210913_100710_copy',
       'KM29_dual_4tastes_emg_200620_165523_copy',
       ]
# make sure all blacklisted basenames are in the frame
assert all([x in basename_list for x in blacklist_basenames])

wanted_frame = wanted_frame.loc[~wanted_frame['data.basename'].isin(blacklist_basenames)]

# Pull out a single data_directory

base_plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/plots'
change_plot_dir = os.path.join(base_plot_dir, 'changepoint_plots')
artifact_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/artifacts'

# transition_snips_list = []
basename_list = []
taste_num_list = []
state_firing_list = []
pal_ranking_dict = {}
tau_list = []
for i, this_row in tqdm(wanted_frame.iterrows()):

    # i = 0
    # this_row = wanted_frame.iloc[i]
    taste_num = this_row['data.taste_num']
    basename = this_row['data.basename']
    pkl_path = this_row['exp.save_path']
    basename_list.append(basename)
    taste_num_list.append(taste_num)

    data_dir = this_row['data.data_dir']
    info_file_path = glob(os.path.join(data_dir, '*.info'))[0]
    with open(info_file_path, 'r') as info_file:
        info_dict = json.load(info_file)
    taste_names = info_dict['taste_params']['tastes']
    pal_rankings = info_dict['taste_params']['pal_rankings']
    pal_ranking_dict[basename] = pal_rankings

    time_lims = literal_eval(this_row['preprocess.time_lims'])
    # From saved pkl file
    this_handler = PklHandler(pkl_path)
    # shape: n_trials x n_states x n_neurons
    state_firing = this_handler.firing.state_firing
    state_firing_list.append(state_firing)

    scaled_mode_tau = this_handler.tau.scaled_mode_tau
    tau_list.append(scaled_mode_tau)

##############################
# Plot tau's across all sessions

# List of arrays with shape: n_states x n_neurons
# mean_state_firing = [x.mean(axis=0) for x in state_firing_list]

firing_frame = pd.DataFrame(
        dict(
            basename = basename_list,
            taste_num = taste_num_list,
            # mean_state_firing = mean_state_firing
            state_firing = state_firing_list,
            tau = tau_list
            )
        )

# Save as artifact
firing_frame.to_pickle(os.path.join(artifact_dir, 'firing_frame.pkl'))

cat_session = np.concatenate([np.repeat(x, y.shape[0]) for x, y in zip(basename_list, state_firing_list)])
cat_tau = np.concatenate(tau_list)
# mean_cat_tau = cat_tau.median(axis=0)
mean_cat_tau = np.median(cat_tau, axis=0)
cat_codes = pd.Categorical(cat_session)
cat_code_names = cat_codes.categories
cat_code_tuples = list(zip(np.unique(cat_codes.codes), cat_code_names))
tuple_str = "\n".join([str(x) for x in cat_code_tuples])


cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in np.arange(cat_tau.shape[1])]
fig, ax = plt.subplots(2,2, sharex='col', figsize = (10,10), sharey='row')
for i, this_tau in enumerate(cat_tau.T):
    ax[0,0].hist(this_tau, bins=20, alpha = 0.5, color = colors[i])
    ax[0,0].hist(this_tau, bins=20, histtype='step', color = colors[i]) 
    ax[0,0].axvline(mean_cat_tau[i], color = colors[i], linestyle='--')
    ax[1,0].scatter(this_tau, np.arange(this_tau.shape[0]), s=10, alpha = 0.7)
ax00_ylim = ax[0,0].get_ylim()
mid_y_val = np.mean(ax00_ylim)
for i, val in enumerate(mean_cat_tau):
    ax[0,0].text(val*1.05, mid_y_val, f'Median: {val:.0f}', rotation=90, color=colors[i])
ax[1,1].imshow(cat_codes.codes[:,None], aspect='auto', interpolation='none')
fig.suptitle('Tau values across all sessions')
ax[0,0].set_title('Tau values')
ax[1,0].set_title('Tau values per session')
ax[1,1].set_title('Session code\n' + tuple_str)
plt.tight_layout()
fig.savefig(os.path.join(base_plot_dir, 'tau_values.png'))
plt.close(fig)

##############################

pal_array_list = []
basename_array_list = []
for i, this_basename in enumerate(tqdm(np.unique(basename_list))):
    wanted_frame = firing_frame.loc[firing_frame['basename'] == this_basename]
    wanted_frame = wanted_frame.sort_values(by='taste_num')
    wanted_pal_rankings = pal_ranking_dict[this_basename]
    wanted_pal_rankings = [wanted_pal_rankings[int(x)] for x in wanted_frame.taste_num]
    assert len(wanted_pal_rankings) == wanted_frame.shape[0]
    # shape: n_tastes x n_trials x n_states x n_neurons
    min_trials = min([x.shape[0] for x in wanted_frame['state_firing'].values])
    firing_stack = np.stack([x[:min_trials] for x in wanted_frame['state_firing'].values])
    
    # np.array_equal(firing_stack[0], firing_stack[1])

    # shape: n_neurons x n_states x (n_tastes*n_trials)
    firing_stack_long = firing_stack.reshape(-1, *firing_stack.shape[2:]).T
    
    # nrn_ind = 16
    # plt.imshow(firing_stack_long[nrn_ind].T, 
    #            aspect='auto', interpolation='none')
    # plt.show()

    pal_vec_long = np.repeat(wanted_pal_rankings, firing_stack.shape[1], axis=0)
    pal_array = np.zeros(firing_stack_long.shape[:2])
    inds = list(np.ndindex(firing_stack_long.shape[:2]))
    for this_ind in inds:
        this_state_firing = firing_stack_long[this_ind]
        this_pal_vec = pal_vec_long
        r_val, p_val = spearmanr(this_state_firing, this_pal_vec)
        pal_array[this_ind] = r_val
    pal_array_list.append(np.abs(pal_array))
    basename_vec = np.repeat(this_basename, pal_array.shape[0])
    basename_array_list.append(basename_vec)

pal_array_cat = np.concatenate(pal_array_list)
basename_cat = np.concatenate(basename_array_list)
basename_categorical = pd.Categorical(basename_cat)
basename_code = np.unique(basename_categorical.codes)
basename_categories = basename_categorical.categories
basename_cat_code_name = list(zip(basename_code, basename_categories))
basename_cat_code_name_str = str('\n'.join([str(x) for x in basename_cat_code_name]))
max_pal_state = pal_array_cat.argmax(axis=1)

fig, ax = plt.subplots(2,2, sharey=False, sharex=False,
                       figsize=(10,10))
ax[0,0].hist(max_pal_state, bins = np.arange(pal_array_cat.shape[1])-0.5)
ax[0,1].bar(np.arange(pal_array_cat.shape[1]), pal_array_cat.mean(axis=0))
# ax[0,1].errorbar(
#         np.arange(pal_array_cat.shape[1]), 
#         y = pal_array_cat.mean(axis=0),
#         yerr = pal_array_cat.std(axis=0),
#         )
ax[0,1].set_title('Mean palatability per state')
ax[1,0].imshow(pal_array_cat, aspect='auto', interpolation='none')
ax[1,0].scatter(max_pal_state, np.arange(pal_array_cat.shape[0]), color='r', s=5)
im = ax[1,1].imshow(basename_code[:,None], aspect='auto', interpolation='none')
plt.colorbar(im, ax=ax[1,1])
fig.suptitle('Palatability correlation with state firing')
ax[0,0].set_title('Max palatability state')
ax[1,0].set_title('Palatability correlation')
ax[1,1].set_title('Basename code' + '\n' + basename_cat_code_name_str)
plt.tight_layout()
fig.savefig(os.path.join(base_plot_dir, 'palatability_correlation_state_aligned.png'))
plt.close(fig)
# plt.show()
