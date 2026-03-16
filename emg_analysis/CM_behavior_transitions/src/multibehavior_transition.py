import pytau.changepoint_model as models
from pickle import load
import os
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import zscore
import numpy as np
from tqdm import tqdm

plot_base_dir = '/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions/plots'
plot_dir = os.path.join(plot_base_dir, 'two_test_changepoint_plots')
os.makedirs(plot_dir, exist_ok = True)

data_dir = '/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions/data'
# data_file = 'CM74_CTATest2_h2o_nacl_lowqhcl_highqhcl_250614_120546_behavior_score_dict.pkl'

data_file = 'behavior_dict_df_all_two_test_animals.pkl'
with open(os.path.join(data_dir, data_file), 'rb') as f:
    # behavior_score_dict = load(f)
    behavior_score_df = load(f)

out_dict_list = []
for ind, row in tqdm(behavior_score_df.iterrows()): 
    basename = row['session_name']
    behavior_score_dict = row['behavior_dict']

    # (trials x taste) x behavior
    behavior_array = behavior_score_dict['behavior_array']

    # Chop by taste
    trial_map = behavior_score_dict['global_trial_map']
    tastes = [key[0] for key in trial_map.keys()]
    rel_trials = [key[1] for key in trial_map.keys()]
    abs_trials = list(trial_map.values()) 
    trial_map_df = pd.DataFrame(
            dict(
                taste = tastes,
                rel_trial = rel_trials,
                abs_trial = abs_trials,
                )
            )

    # get single taste trials
    taste_data_list = []
    zscored_taste_data_list = []
    taste_cp_list = []
    unique_tastes = trial_map_df['taste'].unique()

    for taste in unique_tastes:
        taste_trials = trial_map_df.query(f'taste == "{taste}"')['abs_trial'].values
        taste_behavior_array = behavior_array[taste_trials, :]
        # NEED TO ZSCORE -- OTHERWISE GAUSSIAN ASSUMPTION IS VIOLATED
        # this DOES affect the perfomance of the model
        zscored_taste_behavior_array = zscore(taste_behavior_array, axis = 0)
        # Add slight noise to avoid zero variance issues
        zscored_taste_behavior_array += 0.1 * np.random.randn(*zscored_taste_behavior_array.shape)
        taste_data_list.append(taste_behavior_array)
        zscored_taste_data_list.append(zscored_taste_behavior_array)

        # Use GaussianChangepointMean2D

        model = models.GaussianChangepointMeanVar2D(zscored_taste_behavior_array.T, n_states = 2).generate_model()
        fit = models.advi_fit(
                model,
                fit = 50_000,
                samples = 1000,
                )

        # The API for the output is a little wonky
        tau_samples = fit[-2]
        taste_cp_list.append(tau_samples)

        out_dict = dict(
                basename = basename,
                taste = taste,
                taste_trials = taste_trials,
                taste_behavior_array = taste_behavior_array,
                zscored_taste_behavior_array = zscored_taste_behavior_array,
                tau_samples = tau_samples,
                )
        out_dict_list.append(out_dict)


    fig, ax = plt.subplots(3,len(unique_tastes), sharex = 'col', sharey= 'row',
                           figsize = (len(unique_tastes)*3, 5))
    for taste_idx, taste in enumerate(unique_tastes):
        ax[0, taste_idx].plot(taste_data_list[taste_idx])
        ax[1, taste_idx].plot(zscored_taste_data_list[taste_idx])
        bins = np.arange(0, taste_data_list[taste_idx].shape[0], 1)
        bin_counts, _ = np.histogram(taste_cp_list[taste_idx], bins = bins)
        ax[2, taste_idx].bar(bins[:-1], bin_counts, width = 0.8, alpha=0.7)
        ax[2, taste_idx].set_xlabel('Trial Number')
        # Plot mode
        mode_bin = bins[np.argmax(bin_counts)]
        ax[2, taste_idx].axvline(mode_bin, color = 'red', linestyle = '--', zorder=-1, label='Mode')
    ax[0, 0].set_ylabel('Behavior Score')
    ax[1, 0].set_ylabel('Z-scored Behavior Score')
    ax[2, 0].set_ylabel('Changepoint Samples')
    plt.legend(loc='upper right')
    fig.suptitle(f'{basename}\nBehavior Score and Changepoint Distributions by Taste')
    fig.savefig(os.path.join(plot_dir, f'{basename}_behavior_changepoint_plots.png'))
    plt.close(fig)
    # plt.show()

    # Also make a plot using absolute trial number instead of relative trial number
    fig, ax = plt.subplots(3,len(unique_tastes), sharex = True, sharey= 'row',
                           figsize = (len(unique_tastes)*3, 5))
    for taste_idx, taste in enumerate(unique_tastes):
        taste_trials = trial_map_df.query(f'taste == "{taste}"')['abs_trial'].values
        ax[0, taste_idx].plot(taste_trials, taste_data_list[taste_idx], 'x')
        ax[1, taste_idx].plot(taste_trials, zscored_taste_data_list[taste_idx], 'x')
        bins = np.arange(0, len(taste_trials)+1, 1)
        bin_counts, _ = np.histogram(taste_cp_list[taste_idx], bins = bins)
        # Remap bin edges to absolute trial numbers
        taste_trial_bins = np.arange(taste_trials.min(), taste_trials.max() + 1, 1)
        ax[2, taste_idx].bar(taste_trial_bins, bin_counts, width = 0.8, alpha=0.7)
        ax[2, taste_idx].set_xlabel('Absolute Trial Number')
        # Plot mode
        mode_bin = taste_trial_bins[np.argmax(bin_counts)]
        ax[2, taste_idx].axvline(mode_bin, color = 'red', linestyle = '--', zorder=-1, label='Mode')
    ax[0, 0].set_ylabel('Behavior Score')
    ax[1, 0].set_ylabel('Z-scored Behavior Score')
    ax[2, 0].set_ylabel('Changepoint Samples')
    plt.legend(loc='upper right')
    fig.suptitle(f'{basename}\Behavior Score and Changepoint Distributions by Taste (Absolute Trial Number)')
    fig.savefig(os.path.join(plot_dir, f'{basename}_behavior_changepoint_plots_absolute_trial.png'))
    plt.close(fig)
    # plt.show()

##############################
# Compile out_dict_list into a dataframe and save
out_df = pd.DataFrame(out_dict_list)

# Add a column for mode changepoint
change_hists = [
    np.histogram(out_dict['tau_samples'], bins = np.arange(0, out_dict['taste_behavior_array'].shape[0]+1, 1))[0]
    for out_dict in out_dict_list
    ]
mode_changepoints = [
    np.arange(0, out_dict['taste_behavior_array'].shape[0], 1)[np.argmax(change_hist)]
    for out_dict, change_hist in zip(out_dict_list, change_hists)
    ]
out_df['mode_changepoint'] = mode_changepoints

out_df_file = 'behavior_changepoint_out_df.pkl'
with open(os.path.join(data_dir, out_df_file), 'wb') as f:
    pd.to_pickle(out_df, f)
