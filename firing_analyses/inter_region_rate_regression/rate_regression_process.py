import sys
import os
from tqdm import tqdm
import pandas as pd
blech_clust_dir = os.path.expanduser('~/Desktop/blech_clust')
sys.path.append(blech_clust_dir)
from utils.ephys_data.ephys_data import ephys_data
from pprint import pprint as pp
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
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

recollect_data = False

if recollect_data:
    for this_dir in tqdm(data_dir_list):
        try:
            this_ephys_data = ephys_data(this_dir)

            print(' ===================================== ')
            print(this_dir)
            print(' ===================================== ')

            this_ephys_data.get_region_units()
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
else:
    df_list_paths = os.listdir(artifact_dir)
    df_list_paths = [x for x in df_list_paths if 'section_rate.pkl' in x]
    df_list = [pd.read_pickle(os.path.join(artifact_dir, x)) for x in df_list_paths]

############################################################
# Perform regression
############################################################

# Generate list of pivotted data
time_lims = [0, 2000]
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
            this_pivot = this_region.pivot(
                    index = ['trial_num','time'],
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

# cv_dict_list = []
# mean_cv_dict_list = []
# region_names_list = []
# iden_list = []
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
        time_vals = pivot_frames[0].index.get_level_values('time')
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

        gkf = GroupKFold(n_splits=groups.nunique())
        mlp = MLPRegressor(hidden_layer_sizes=(50,50,50), max_iter=1000)
        # lr = LinearRegression()

        # spearman_score = lambda x,y: np.abs(spearmanr(x.flatten(),y.flatten())[0])

        cross_val_kwargs = dict(
                estimator = mlp,
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

all_pivot_frame['cv_results'] = cv_dict_list
all_pivot_frame['mean_cv_results'] = mean_cv_dict_list
all_pivot_frame.to_pickle(os.path.join(artifact_dir, 'all_pivot_frame_cv.pkl'))

mean_cv_dict_list = [
        {this_key:np.mean(this_dict[this_key]) for this_key in this_dict.keys()} \
        for this_dict in cv_dict_list
        ]
mean_cv_frame = pd.DataFrame(mean_cv_dict_list)
# mean_cv_frame['region_names'] = region_names_list
# iden_frame = pd.DataFrame(iden_list)
# iden_frame.reset_index(drop=True, inplace=True)
# mean_cv_frame = pd.concat([iden_frame, mean_cv_frame], axis=1)

# melted_mean_cv_frame = mean_cv_frame.melt(var_name='cv_type', value_name='score')

# Drop scrambled
# plot_mean_cv_frame = mean_cv_frame.drop(columns = 'scrambled_cv_test_score')
delta_cv = mean_cv_frame['cv_test_score'] - mean_cv_frame['group_shuffle_cv_test_score']
mean_delta_cv = np.mean(delta_cv)
median_delta_cv = np.median(delta_cv)

bins = np.linspace(-1,1,50)
plt.hist(delta_cv)#, bins=bins)
plt.axvline(0, c='r')
# Annotate mean and median with arrows
plt.annotate(f'Mean: {mean_delta_cv:.2f}', xy=(mean_delta_cv, 0), xytext=(mean_delta_cv, 10),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate(f'Median: {median_delta_cv:.2f}', xy=(median_delta_cv, 0), xytext=(median_delta_cv, 10),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.ylabel('Count')
plt.xlabel('Delta CV\n<-- Actual better | Shuffled better -->')
# plt.tight_layout()
plt.show()
