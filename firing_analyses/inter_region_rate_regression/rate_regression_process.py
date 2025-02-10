import sys
import os
from tqdm import tqdm
import pandas as pd
blech_clust_dir = os.path.expanduser('~/Desktop/blech_clust')
sys.path.append(blech_clust_dir)
from utils.ephys_data.ephys_data import ephys_data
from utils.ephys_data import visualize as vz
from pprint import pprint as pp
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import spearmanr, pearsonr, wilcoxon, zscore
from sklearn.metrics import make_scorer, mean_squared_error
import seaborn as sns
import base64
import hashlib
import json


def shuffle_groups(groups, time_vals, X): 
    # Shuffle groups
    group_shuffle_map = dict(zip(groups.unique(), np.random.permutation(groups.unique())))
    new_groups = groups.map(group_shuffle_map)
    X_frame = pd.DataFrame(X)
    X_frame['groups'] = new_groups
    X_frame['time'] = time_vals
    X_frame.sort_values(['groups','time'], inplace=True)
    shuffled_X = X_frame.drop(columns = ['groups','time']).values
    return shuffled_X, new_groups

############################################################
############################################################

# data_dir_file_path = '/media/fastdata/Thomas_Data/all_data_dirs.txt'
data_dir_file_path = '/media/fastdata/Thomas_Data/data/sorted_new/data_dir_list.txt'
data_dir_list = [x.strip() for x in open(data_dir_file_path, 'r').readlines()]

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/inter_region_rate_regression'
artifact_dir =  os.path.join(base_dir,'artifacts')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)
plot_dir = os.path.join(base_dir,'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

recollect_data = True

if recollect_data:
    for this_dir in tqdm(data_dir_list):
        try:
            this_ephys_data = ephys_data(this_dir)

            print(' ===================================== ')
            print(this_dir)
            print(' ===================================== ')

            this_ephys_data.default_firing_params['type'] = 'basis'

            this_ephys_data.get_region_units()
            region_dict = dict(
                    zip(
                        this_ephys_data.region_names,
                        this_ephys_data.region_units,
                        )
                    )
            inv_region_map = {}
            for k,v in region_dict.items():
                for unit in v:
                    inv_region_map[unit] = k
            this_ephys_data.get_spikes()
            n_tastes = len(this_ephys_data.spikes)

            # Get trial-changepoints
            qa_output_dir = os.path.join(this_dir, 'QA_output')
            best_change_path = os.path.join(qa_output_dir, 'best_change.txt')
            best_change = int(open(best_change_path, 'r').readlines()[1])

            session_artifact_dir = os.path.join(this_dir, 'QA_output', 'artifacts')
            # Get all files
            all_files = sorted(os.listdir(session_artifact_dir))
            # Keep only csv
            all_files = [x for x in all_files if 'csv' in x]
            # Look for pattern taste_* in each name
            # taste_ind = [int(x.split('taste_')[1][0]) for x in all_files if 'taste' in x]

            # Look for 'taste_trial'
            all_files = [x for x in all_files if 'taste_trial' in x][0]

            # Load pkl files
            # trial_change_dat = [pd.read_csv(os.path.join(artifact_dir, x), index_col = 0) \
            #         for x in all_files if 'taste' in x]
            trial_change_dat = pd.read_csv(os.path.join(session_artifact_dir, all_files), index_col = 0)

            # # Get median lowest elbo
            # # best_change = []
            # # for this_df in trial_change_dat:
            # med_elbo = trial_change_dat.groupby('changes').median()
            # # best_change.append(med_elbo['elbo'].idxmin())
            # best_change = med_elbo['elbo'].idxmin()

            best_change_df = trial_change_dat[trial_change_dat['changes'] == best_change]
            mode_list = np.stack([literal_eval(x) for x in best_change_df['mode']])
            median_mode_list = np.median(mode_list, axis=0)
            time_bins = trial_change_dat['time_bins'].iloc[0]
            time_bins = time_bins.replace(' ',',').replace('\n','').replace('[','').replace(']','').split(',')
            # Drop any empty strings
            time_bins = [x for x in time_bins if x]
            # Convert to float
            time_bins = [float(x) for x in time_bins]

            # Convert median_mode_list to timepoints using interpolated time_bins 
            interpolated_time = np.interp(median_mode_list, np.arange(len(time_bins)), time_bins)

            # Make "sections"
            time_sections = np.concatenate(([0], interpolated_time, [np.max(time_bins)]))

            # Get trial_info_frame and get sections for all trials
            trial_info_path = os.path.join(this_dir, 'trial_info_frame.csv') 
            trial_info_frame = pd.read_csv(trial_info_path)
            trial_info_frame['section'] = pd.cut(trial_info_frame['start_taste'], time_sections, labels=False)
            
            wanted_columns = ['dig_in_num_taste', 'start_taste', 'section', 'taste_rel_trial_num']
            trial_info_frame = trial_info_frame[wanted_columns]

            dig_in_num_taste_map = dict(zip(
                np.sort(trial_info_frame['dig_in_num_taste'].unique()), 
                range(n_tastes)))
            trial_info_frame['taste_ind'] = trial_info_frame['dig_in_num_taste'].map(dig_in_num_taste_map) 

            vz.firing_overview(this_ephys_data.all_normalized_firing)
            plt.show()

            this_ephys_data.get_sequestered_firing()
            seq_firing_frame = this_ephys_data.sequestered_firing_frame
            seq_firing_frame.reset_index(inplace=True)

            merge_frame = pd.merge(
                    seq_firing_frame, 
                    trial_info_frame, 
                    left_on = ['taste_num','trial_num'],
                    right_on = ['taste_ind','taste_rel_trial_num'],
                    how = 'inner',
                    )
            merge_frame.drop(columns = ['index','taste_rel_trial_num','taste_ind'], inplace=True)
            # Add time values
            t_vec = np.vectorize(int)(np.linspace(-2000, 5000, merge_frame.time_num.max()+1))
            merge_frame['time'] = t_vec[merge_frame['time_num']]

            # Add region values
            merge_frame['region'] = merge_frame['neuron_num'].map(inv_region_map)

            basename = os.path.basename(this_dir)
            animal = os.path.basename(os.path.dirname(this_dir))
            merge_frame['basename'] = basename
            merge_frame['animal'] = animal

            # Save to pkl
            save_path = os.path.join(artifact_dir, f'{animal}_{basename}_section_rate.pkl')
            merge_frame.to_pickle(save_path)
        
        except Exception as e:
            print(f'Error with {this_dir}')
            print(e)

df_list_paths = os.listdir(artifact_dir)
df_list_paths = [x for x in df_list_paths if 'section_rate.pkl' in x]
df_list = [pd.read_pickle(os.path.join(artifact_dir, x)) for x in df_list_paths]

############################################################
# Perform regression
############################################################

# Generate list of pivotted data
time_lims = [2000, 4000]
basename_list = []
animal_list = []
taste_list = []
section_list = []
pivots_list = []
region_name_list = []
for this_df in tqdm(df_list):
    section_group_list = [x[1] for x in list(this_df.groupby(['taste_num','section']))]
    # Only keep sections with at least 3 trials
    min_trials = 3
    section_group_list = [x for x in section_group_list if x.trial_num.nunique() >= min_trials]
    for this_section in section_group_list:
        region_names, region_data = zip(*list(this_section.groupby('region'))) 
        # Pivot to make neuron_num the columns
        pivot_frames = []
        for i, this_region in enumerate(region_data):
            this_region.drop(columns = 'start_taste', inplace=True)
            this_pivot = this_region.pivot(
                    index = ['trial_num','time_num'],
                    columns = 'neuron_num', 
                    values = 'firing')
            # Drop according to time_lims
            this_pivot = this_pivot.loc[(slice(None), slice(*time_lims)), :]
            pivot_frames.append(this_pivot)

        basename_list.append(this_df['basename'].iloc[0])
        animal_list.append(this_df['animal'].iloc[0])
        taste_list.append(this_section['taste_num'].iloc[0])
        section_list.append(this_section['section'].iloc[0])
        pivots_list.append(pivot_frames)
        region_name_list.append(region_names)

# plt.plot(pivot_frames[0].iloc[:,0].values)
# plt.show()

# Perform regression
all_pivot_frame = pd.DataFrame(
        dict(
            animal = animal_list,
            basename = basename_list,
            taste = taste_list,
            section = section_list,
            pivots = pivots_list,
            region_names = region_name_list,
            )
        )

spearman_score = lambda x,y: spearmanr(x.flatten(),y.flatten())[0]
pearson_score = lambda x,y: pearsonr(x.flatten(),y.flatten())[0]
spearman_score = make_scorer(spearman_score)
pearson_score = make_scorer(pearson_score)
# Make mse scorer
mse_score = make_scorer(mean_squared_error)

cv_out_dir = os.path.join(artifact_dir, 'cv_results')
if not os.path.exists(cv_out_dir):
    os.makedirs(cv_out_dir)

cv_dict_list = []
mean_cv_dict_list = []
region_names_list = []
iden_list = []
for i, this_row in tqdm(all_pivot_frame.iterrows()):
    iden_cols = this_row[['animal','basename','taste','section']]
    orig_pivot_frames = this_row['pivots']
    orig_region_names = this_row['region_names']
    # Rename 'oc' to 'pc'
    orig_region_names = ['pc' if x == 'oc' else x for x in region_names]

    for this_order in [[0,1], [1,0]]:
        pivot_frames = [orig_pivot_frames[x] for x in this_order]
        region_names = [orig_region_names[x] for x in this_order]
        region_names_list.append(region_names)
        iden_list.append(iden_cols)

        unique_iden = iden_cols.to_list() + region_names
        unique_iden_str = '_'.join([str(x) for x in unique_iden])
        hasher = hashlib.sha1(unique_iden_str.encode())
        unique_iden_hash = base64.urlsafe_b64encode(hasher.digest()).decode('utf-8')[:10]

        out_path = os.path.join(cv_out_dir, f'{unique_iden_hash}.pkl')
        if os.path.exists(out_path):
            print(f'Skipping {iden_cols.to_list()}')
            continue

        groups = pivot_frames[0].index.get_level_values('trial_num')
        # time_vals = pivot_frames[0].index.get_level_values('time')
        time_vals = pivot_frames[0].index.get_level_values('time_num')
        n_components = np.min([5, min([x.shape[1] for x in pivot_frames])])
        pivot_pca = [PCA(n_components = n_components, whiten=True).fit_transform(x) for x in pivot_frames] 
        # Shuffle groups
        trial_shuffled_pivot_pca = [shuffle_groups(groups, time_vals, pivot_pca[0])[0], pivot_pca[1]]
        scrambled_pivot_pca = [np.random.permutation(pivot_pca[0]), pivot_pca[1]]

        # img_kwargs = dict(interpolation='none', aspect='auto')
        # fig, ax = plt.subplots(3,1,sharex=True)
        # ax[0].imshow(pivot_pca[0].T, **img_kwargs)
        # ax[1].imshow(trial_shuffled_pivot_pca[0].T, **img_kwargs)
        # ax[2].imshow(scrambled_pivot_pca[0].T, **img_kwargs)
        # plt.show()

        gkf = GroupKFold(n_splits=np.min([groups.nunique(), 5]))
        # mlp = MLPRegressor(hidden_layer_sizes=(50,50,50), max_iter=1000)
        lr = LinearRegression()
        estimator = lr

        # spearman_score = lambda x,y: np.abs(spearmanr(x.flatten(),y.flatten())[0])

        cross_val_kwargs = dict(
                estimator = estimator, 
                groups = groups,
                cv = gkf,
                scoring = mse_score,
                # scoring = 'neg_mean_squared_error',
                # scoring = spearman_score,
                # scoring = pearson_score,
                # return_estimator = True,
                )
                
        cv_results = cross_validate(
                X = pivot_pca[0],
                y = pivot_pca[1],
                **cross_val_kwargs
                )

        # # Get best model
        # best_model = cv_results['estimator'][np.argmax(cv_results['test_score'])]
        # pred_y = best_model.predict(pivot_pca[0])
        #
        # cmin = np.min([pivot_pca[1].min(), pred_y.min()])
        # cmax = np.max([pivot_pca[1].max(), pred_y.max()])
        # img_kwargs = dict(interpolation='none', aspect='auto', vmin=cmin, vmax=cmax)
        # fig, ax = plt.subplots(2,1,sharex=True)
        # ax[0].imshow(pivot_pca[1].T, **img_kwargs)
        # ax[1].imshow(pred_y.T, **img_kwargs)
        # plt.show()

        group_shuffle_cv_results = cross_validate(
                X = trial_shuffled_pivot_pca[0], 
                y = trial_shuffled_pivot_pca[1],
                **cross_val_kwargs
                )

        # scrambled_cv_results = cross_validate(
        #         X = scrambled_pivot_pca[0],
        #         y = scrambled_pivot_pca[1],
        #         **cross_val_kwargs
        #         )

        cv_test_score = cv_results['test_score']
        group_shuffle_cv_test_score = group_shuffle_cv_results['test_score']
        # scrambled_cv_test_score = scrambled_cv_results['test_score']
        
        x_names = ['cv_test_score', 'group_shuffle_cv_test_score']#, 'scrambled_cv_test_score']
        y_vals_list = [cv_test_score, group_shuffle_cv_test_score]#, scrambled_cv_test_score]
        y_vals_list = [y.tolist() for y in y_vals_list]
        # y_vals = np.concatenate(y_vals_list)
        # x_vals = np.concatenate([[x]*len(y) for x,y in zip(x_names, y_vals_list)])
        cv_dict = dict(zip(x_names, y_vals_list))
        iden_cols['cv_results'] = cv_dict

        iden_cols.to_pickle(out_path)

        # cv_dict_list.append(cv_dict)
        # with open(out_path, 'w') as f:
        #     json.dump(cv_dict, f)
        # mean_cv_dict = {x:np.mean(y) for x,y in cv_dict.items()}
        # mean_cv_dict_list.append(mean_cv_dict)


    # plt.scatter(x_vals, y_vals, alpha=0.5, edgecolor='k', linewidth=1)
    # plt.scatter(x_names, [np.mean(x) for x in y_vals_list], c='r', s=100, alpha=0.5)
    # plt.show()

# Get all pickle files
cv_files = os.listdir(cv_out_dir)
cv_files = [x for x in cv_files if '.pkl' in x]
cv_dict_list = [pd.read_pickle(os.path.join(cv_out_dir, x)) for x in cv_files]

cv_frame = pd.DataFrame(cv_dict_list)
cv_frame.reset_index(drop=True, inplace=True)
mean_cv_test = cv_frame.cv_results.apply(lambda x: np.mean(x['cv_test_score']))
mean_group_shuffle_test = cv_frame.cv_results.apply(lambda x: np.mean(x['group_shuffle_cv_test_score']))

cv_frame['mean_cv_test'] = mean_cv_test
cv_frame['mean_group_shuffle_test'] = mean_group_shuffle_test
cv_frame['cv_diff'] = cv_frame['mean_cv_test'] - cv_frame['mean_group_shuffle_test']

mean_cols = ['mean_cv_test', 'mean_group_shuffle_test', 'cv_diff']
mean_cv_frame = cv_frame.groupby(['animal','basename'])[mean_cols].mean()
mean_cv_frame.reset_index(inplace=True)

# Throw out anything with abs > 30
mean_cv_frame = mean_cv_frame[mean_cv_frame['cv_diff'].abs() < 30]
# mean_mean_cv_diff = np.mean(mean_cv_frame['cv_diff'])

# Test for significance
# wanted_cv_diff = cv_frame['cv_diff']
wilcoxon_out = wilcoxon(mean_cv_frame['cv_diff'])

fig, ax = plt.subplots(1,2, sharey=True)
sns.scatterplot(data = mean_cv_frame, x = 'animal', y = 'cv_diff', hue = 'basename',
                legend = False, s = 100, ax = ax[0], alpha = 0.5, edgecolor = 'k', linewidth = 1)
ax[0].axhline(0, c='r')
ax[0].set_title('CV Diff')
ax[0].set_ylabel('Mean CV Diff\n<-- Actual better | Shuffled better -->')
ax[1].hist(mean_cv_frame['cv_diff'], orientation='horizontal', bins=20)
# Annotate histogram with mean
ax[1].annotate(f'Mean: {mean_mean_cv_diff:.2f}', xy=(0, mean_mean_cv_diff), xytext=(0, mean_mean_cv_diff),
             arrowprops=dict(facecolor='black', shrink=0.05))
ax[1].set_title(f'CV Diff Histogram\nWilcoxon p: {wilcoxon_out.pvalue:.2e}')
ax[1].axhline(0, c='r')
fig.suptitle('Mean CV Diff by Animal and Session')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'mean_cv_diff.svg'))
plt.close(fig)
# plt.show()

############################################################
############################################################
# Make Average and single trial plots

# Get highest cv_test_score (lowest mse score)
cv_frame['min_cv_test'] = cv_frame.cv_results.apply(lambda x: np.min(x['cv_test_score']))

min_cv_frame = cv_frame[cv_frame['min_cv_test'] == cv_frame['min_cv_test'].min()]
# Get details from all_pivot_frame
row_bool = (all_pivot_frame['animal'] == min_cv_frame['animal'].iloc[0]) & \
        (all_pivot_frame['basename'] == min_cv_frame['basename'].iloc[0]) & \
        (all_pivot_frame['taste'] == min_cv_frame['taste'].iloc[0]) & \
        (all_pivot_frame['section'] == min_cv_frame['section'].iloc[0])
row_ind = np.where(row_bool)[0][0]
wanted_pivots = pivots_list[row_ind]
wanted_region_names = region_name_list[row_ind]
# all_pivot_frame.loc[np.where(row_bool)[0][0]]
wanted_index = wanted_pivots[0].index
n_trials = wanted_index.get_level_values('trial_num').nunique()
pivot_pca = [PCA(n_components = 3, whiten=True).fit_transform(x) for x in wanted_pivots] 
# Split into trials and average
pivot_pca_trials = [np.stack(np.split(x, n_trials, axis=0)) for x in pivot_pca]
mean_pivot_pca = [np.mean(x, axis=0) for x in pivot_pca_trials]

# Align both using regression
X = mean_pivot_pca[0]
y = mean_pivot_pca[1]
proj_y = np.linalg.lstsq(X, y, rcond=None)[0] @ X.T

# Plot in 3D
fig = plt.figure(figsize=(12, 6))
ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection='3d')
ax0.scatter(*X.T, alpha=0.5)
ax1.scatter(*proj_y, alpha=0.5)
plt.show()

# Average pivots by trial and plot
mean_pivots = []
for this_pivot in wanted_pivots:
    this_pivot = this_pivot.reset_index()
    mean_pivots.append(this_pivot.groupby('time_num').mean())

# fig, ax = plt.subplots(2,1, sharex=True, figsize=(12,8))
fig = plt.figure(figsize=(12, 6))
ax0 = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122, projection='3d')
ax = [ax0, ax1]
for i, this_mean_pivot in enumerate(mean_pivots):
    zscore_mean_pivot = zscore(this_mean_pivot.values, axis=0)
    # Drop nans
    zscore_mean_pivot = zscore_mean_pivot[:,~np.isnan(zscore_mean_pivot).any(axis=0)]
    # ax[i].imshow(zscore_mean_pivot.T, aspect='auto', interpolation='none')
    ax[i].scatter(*zscore_mean_pivot[:,:3].T, alpha=0.5)
plt.show()

############################################################
############################################################

mean_firing_plot_dir = os.path.join(plot_dir, 'mean_firing')
if not os.path.exists(mean_firing_plot_dir):
    os.makedirs(mean_firing_plot_dir)

mean_cos_sim_list = []
basename_list = []
taste_list = []
for this_dir in tqdm(data_dir_list):
    this_ephys_data = ephys_data(this_dir)

    print(' ===================================== ')
    print(this_dir)
    print(' ===================================== ')

    firing_type = 'baks'

    this_ephys_data.firing_rate_params = this_ephys_data.default_firing_params.copy()
    this_ephys_data.firing_rate_params['type'] = firing_type
    this_ephys_data.firing_rate_params['step_size'] = 10
    step_size = this_ephys_data.firing_rate_params['step_size']

    this_ephys_data.get_region_units()
    region_dict = dict(
            zip(
                this_ephys_data.region_names,
                this_ephys_data.region_units,
                )
            )
    inv_region_map = {}
    for k,v in region_dict.items():
        for unit in v:
            inv_region_map[unit] = k
    this_ephys_data.get_spikes()
    this_ephys_data.get_firing_rates()

    cat_spikes = np.concatenate(this_ephys_data.spikes, axis=0)

    norm_firing = this_ephys_data.normalized_firing_list.copy()
    # List (tastes) -> list (regions) - > array (trials, neurons, time)
    region_firing = [[taste[:,x] for k, x in region_dict.items()] for taste in norm_firing] 
    region_spikes = [cat_spikes[:,x] for k, x in region_dict.items()]
    cat_region_spikes = np.concatenate(region_spikes, axis=1)

    # region_firing = [this_ephys_data.get_region_firing(x) for x in this_ephys_data.region_names]
    # # Chop by time_lims
    time_lims_raw = np.array([1000, 5000])
    if firing_type == 'conv':
        time_lims = time_lims_raw // step_size
    elif firing_type == 'basis':
        time_lims = time_lims_raw.copy()
    elif firing_type == 'baks':
        step_size = this_ephys_data.firing_rate_params['baks_resolution'] / \
                this_ephys_data.firing_rate_params['baks_dt']
        step_size = int(step_size)
        time_lims = time_lims_raw // step_size
    region_firing = [[x[...,time_lims[0]:time_lims[1]] for x in taste] for taste in region_firing]

    # Zip to have regions as outer list
    region_firing = list(zip(*region_firing))

    # # Normalize for each neuron
    # norm_region_firing = region_firing.copy()
    # for i, this_region in enumerate(norm_region_firing):
    #     for nrn_ind in range(this_region.shape[1]):
    #         norm_region_firing[i][:,nrn_ind] = zscore(this_region[:,nrn_ind])

    # cat_norm_firing = np.concatenate(norm_region_firing, axis=1)

    # vz.firing_overview(np.concatenate(cat_norm_firing.swapaxes(1,2),0).swapaxes(0,1))
    # plt.show()

    # mean_region_firing = [np.mean(x, axis=2).swapaxes(0,1) for x in region_firing]
    mean_region_firing = [np.stack([np.mean(x, axis=0) for x in taste]).swapaxes(0,1) for taste in region_firing]
    cat_mean_region_firing = np.concatenate(mean_region_firing, axis=0)
    cat_region_labels = np.concatenate([[k]*x.shape[0] for k,x in region_dict.items()])

    if len(cat_mean_region_firing) > 1:
    
        fig, ax = vz.gen_square_subplots(len(cat_mean_region_firing), 
                                         sharex=True, sharey=True,
                                         figsize=(12,12))
        font_color_dict = dict(
                zip(
                    this_ephys_data.region_names, 
                    sns.color_palette('tab10', len(this_ephys_data.region_names))
                    )
                )
        for i, this_firing in enumerate(cat_mean_region_firing):
            this_ax = ax.flatten()[i]
            this_ax.plot(this_firing.T)
            this_ax.set_title(cat_region_labels[i], 
                              color=font_color_dict[cat_region_labels[i]],
                              fontweight='bold')
        fig.suptitle(os.path.basename(this_dir))
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(mean_firing_plot_dir, f'{os.path.basename(this_dir)}_mean_{firing_type}_firing.svg'))
        plt.close(fig)
    #
    #     ############################## 
    #     # Plot spikes
    #     cat_region_spikes = cat_region_spikes[...,time_lims_raw[0]:time_lims_raw[1]]
    #
    #     fig, ax = vz.gen_square_subplots(cat_region_spikes.shape[1],
    #                                      sharex=True, sharey=True,
    #                                      figsize=(12,12))
    #     for i in range(cat_region_spikes.shape[1]):
    #         this_ax = ax.flatten()[i]
    #         this_ax = vz.raster(this_ax, cat_region_spikes[:, i], marker = '|', color = 'k') 
    #         this_ax.set_title(cat_region_labels[i], 
    #                           color=font_color_dict[cat_region_labels[i]],
    #                           fontweight='bold')
    #     fig.suptitle(os.path.basename(this_dir))
    #     plt.tight_layout()
    #     # plt.show()
    #     fig.savefig(os.path.join(mean_firing_plot_dir, f'{os.path.basename(this_dir)}_spikes.png'))
    #     plt.close(fig)
    
    if len(cat_mean_region_firing) > 1:
        region_firing_long = region_firing.copy()
        region_firing_long = [np.concatenate(x, axis=0) for x in region_firing_long]
        region_firing_long = [x.swapaxes(0,1) for x in region_firing_long]
        region_firing_long = [np.reshape(x, (x.shape[0], -1)) for x in region_firing_long]

        # Perform pca
        pca_list = [PCA(n_components=3).fit(x.T) for x in region_firing_long]

        # Also get a projection between PC components of regions
        pca_region_long = [x.transform(y.T) for x,y in zip(pca_list, region_firing_long)]
        X = pca_region_long[0]
        y = pca_region_long[1]
        y_projection = np.linalg.lstsq(X, y, rcond=None)[0]

        # Peform pca on single_trials
        region_pca_arrays = []
        for this_pca_obj, this_region in zip(pca_list, region_firing):
            this_region_tastes = []
            for this_taste in this_region:
                taste_inds = list(np.ndindex(this_taste.shape[:1]))
                pca_array = np.empty((this_taste.shape[0], 3, this_taste.shape[-1]))
                for i, this_taste_ind in enumerate(taste_inds):
                    pca_array[this_taste_ind] = this_pca_obj.transform(this_taste[this_taste_ind].T).T
                this_region_tastes.append(pca_array)
            region_pca_arrays.append(this_region_tastes)
            # this_region = this_region.swapaxes(1,2)
            # region_inds = list(np.ndindex(this_region.shape[:2]))
            # pca_array = np.empty((this_region.shape[0], this_region.shape[1], 3, this_region.shape[3]))
            # for i, this_region_ind in enumerate(region_inds):
            #     pca_array[this_region_ind] = this_pca_obj.transform(this_region[this_region_ind].T).T
            # region_pca_arrays.append(pca_array)

        # Shape : region, taste, trial, pca, time
        # all_region_pca_array = np.stack(region_pca_arrays)
        # region_pca_arrays: region (list) --> taste (list) --> trials, pca, time

        # mean_region_pca_array = np.mean(all_region_pca_array, axis=2)
        # shape: region, taste, pca, time
        mean_region_pca_array = np.stack(
                [[x.mean(axis=0) for x in region] \
                        for region in region_pca_arrays]
                )

        # vz.firing_overview(mean_region_pca_array[0])
        # plt.show()

        mean_region_pca_proj = np.tensordot(
                mean_region_pca_array[0],
                y_projection,
                [1, 0],
                ).swapaxes(1,2)

        fig = plt.figure(figsize=(12, 6))
        ax0 = fig.add_subplot(131, projection='3d')
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(133, projection='3d')
        ax = [ax0, ax1, ax2]
        for i, this_region in enumerate(mean_region_pca_array):
            for taste in this_region:
                ax[i].plot(*taste, alpha=0.5)
        for taste in mean_region_pca_proj:
            ax[2].plot(*taste, alpha=0.5)
        ax[0].set_title('Region 1 PCA')
        ax[1].set_title('Region 2 PCA')
        ax[2].set_title('Region 1 PCA projected onto Region 2 PCA')
        basename = os.path.basename(this_dir)
        fig.suptitle(f'{basename}')
        fig.savefig(os.path.join(plot_dir, f'{basename}_pca.svg'))
        plt.close(fig)
        # plt.show()

        ############################## 
        # Perform and align pca on a single-taste basis
        # region_firing_long = [x[0] for x in region_firing_long]
        n_tastes = len(region_firing[0])
        region_pca_list = []
        region_pca_proj_list = []
        for taste_ind in range(n_tastes):
            # taste_ind = 0
            region_firing_long = region_firing.copy()
            taste_firing = [x[taste_ind] for x in region_firing_long]
            region_firing_long = taste_firing.copy()
            # region_firing_long = [np.concatenate(x, axis=0) for x in region_firing_long]
            region_firing_long = [x.swapaxes(0,1) for x in region_firing_long]
            region_firing_long = [np.reshape(x, (x.shape[0], -1)) for x in region_firing_long]

            # Perform pca
            pca_list = [PCA(n_components=3).fit(x.T) for x in region_firing_long]

            # Also get a projection between PC components of regions
            pca_region_long = [x.transform(y.T) for x,y in zip(pca_list, region_firing_long)]
            X = pca_region_long[0]
            y = pca_region_long[1]
            y_projection = np.linalg.lstsq(X, y, rcond=None)[0]

            # Peform pca on single_trials
            region_pca_arrays = []
            # for this_pca_obj, this_region in zip(pca_list, region_firing):
            for this_pca_obj, this_region in zip(pca_list, taste_firing):
                taste_inds = list(np.ndindex(this_region.shape[:1]))
                pca_array = np.empty((this_region.shape[0], 3, this_region.shape[-1]))
                for i, this_taste_ind in enumerate(taste_inds):
                    pca_array[this_taste_ind] = this_pca_obj.transform(this_region[this_taste_ind].T).T
                region_pca_arrays.append(pca_array)

            # Shape : region, taste, trial, pca, time
            # all_region_pca_array = np.stack(region_pca_arrays)
            # region_pca_arrays: region (list) --> taste (list) --> trials, pca, time

            # shape: region, taste, pca, time
            mean_region_pca_array = np.stack(
                    [x.mean(axis=0) for x in region_pca_arrays]
                    )

            lr = LinearRegression()
            X = mean_region_pca_array[0]
            y = mean_region_pca_array[1]
            lr.fit(X.T, y.T)
            y_proj  = lr.predict(X.T).T
            mean_region_pca_proj = y_proj.copy()

            # mean_region_pca_proj = np.tensordot(
            #         mean_region_pca_array[0],
            #         y_projection,
            #         [0, 0],
            #         ).T

            region_pca_list.append(mean_region_pca_array)
            region_pca_proj_list.append(mean_region_pca_proj)

            # Calculate mean cosine similarity
            def cos_sim(x,y):
                return np.dot(x,y) / (np.linalg.norm(x) * np.linalg.norm(y))

            cos_sim_list = [cos_sim(a,b) for a,b in zip(X.T, y_proj.T)]
            mean_cos_sim = np.mean(cos_sim_list)

            # plt.imshow(np.dot(y.T, y_proj), interpolation='nearest')
            # plt.show()

            fig = plt.figure(figsize=(8, 6))
            ax0 = fig.add_subplot(121, projection='3d')
            ax1 = fig.add_subplot(122, projection='3d')
            ax = [ax0, ax1]
            for i, this_region in enumerate(mean_region_pca_array):
                    ax[i].plot(*this_region, alpha=0.5)
            ax[1].plot(*mean_region_pca_proj, alpha=0.5, c='r', label = 'Projected')
            ax[0].set_title('Region 1 PCA')
            ax[1].set_title(f'Region 2 PCA, Mean Cosine Sim: {mean_cos_sim:.2f}')
            ax[1].legend()
            basename = os.path.basename(this_dir)
            fig.suptitle(f'{basename}')
            fig.savefig(os.path.join(plot_dir, f'{basename}_pca_taste_{taste_ind}.svg'))
            plt.close(fig)
            # plt.show()
            
            mean_cos_sim_list.append(mean_cos_sim)
            basename_list.append(basename)
            taste_list.append(taste_ind)

        ############################## 
        # # Also perform partial least squares regression 
        # for taste_ind in range(n_tastes):
        #     # taste_ind = 0
        #     region_firing_long = region_firing.copy()
        #     taste_firing = [x[taste_ind] for x in region_firing_long]
        #     region_firing_long = taste_firing.copy()
        #     # region_firing_long = [np.concatenate(x, axis=0) for x in region_firing_long]
        #     region_firing_long = [x.swapaxes(0,1) for x in region_firing_long]
        #     region_firing_long = [np.reshape(x, (x.shape[0], -1)) for x in region_firing_long]
        #
        #     pls = PLSRegression(n_components=3)
        #     pls.fit(region_firing_long[0].T, region_firing_long[1].T)
        #     # y_proj = pls.transform(region_firing_long[0].T).T
        #     # x_proj, y_proj = pls.transform(region_firing_long[0].T, region_firing_long[1].T)
        #     proj_out = [pls.transform(a.T,b.T) for a,b in zip(taste_firing[0], taste_firing[1])]
        #     x_proj, y_proj = zip(*proj_out)
        #     x_proj = np.stack(x_proj)
        #     y_proj = np.stack(y_proj)
        #
        #     mean_x_proj = x_proj.mean(axis=0)
        #     mean_y_proj = y_proj.mean(axis=0)
        #
        #     fig = plt.figure(figsize=(6, 6))
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot(*mean_x_proj.T, alpha=0.5, label='Region 1')
        #     ax.plot(*mean_y_proj.T, alpha=0.5, label='Region 2')
        #     ax.set_title(f'Taste {taste_ind} PLS')
        #     ax.legend()
        #     basename = os.path.basename(this_dir)
        #     fig.suptitle(f'{basename}')
        #     fig.savefig(os.path.join(plot_dir, f'{basename}_pls_taste_{taste_ind}.svg'))
        #     plt.close(fig)
        #

cos_sim_frame = pd.DataFrame(
        dict(
            basename = basename_list,
            taste = taste_list,
            cos_sim = mean_cos_sim_list,
            )
        )
cos_sim_frame.to_csv(os.path.join(artifact_dir, 'cos_sim_frame.csv'))
