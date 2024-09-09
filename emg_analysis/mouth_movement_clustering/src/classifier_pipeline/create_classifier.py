"""
Generate training dataset and train a classifier.
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import xgboost as xgb
import sys
from ast import literal_eval
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
code_dir = os.path.join(base_dir, 'src')
sys.path.append(code_dir)
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            find_segment,
                                            calc_peak_interval,
                                            gen_gape_frame,
                                            threshold_movement_lengths,
                                            )

artifact_dir = os.path.join(base_dir, 'artifacts')

############################################################
# Load data
############################################################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

##############################
# Specificitiy of annotations 
##############################
merge_gape_pal_path = os.path.join(artifact_dir, 'merge_gape_pal.pkl')
merge_gape_pal = pd.read_pickle(merge_gape_pal_path)

feature_names_path = os.path.join(artifact_dir, 'merge_gape_pal_feature_names.npy')
feature_names = np.load(feature_names_path)

# Load scored_df anew to have the updated codes and event_types
# scored_df = merge_gape_pal[merge_gape_pal.scored == True]
scored_df_path = os.path.join(artifact_dir, 'scored_df.pkl')
scored_df = pd.read_pickle(scored_df_path)

# Over-write with updated event coes and types
scored_df['event_type'] = scored_df['updated_event_type']
scored_df['event_codes'] = scored_df['updated_codes']

# Correct event_types
types_to_drop = ['to discuss', 'other', 'unknown mouth movement','out of view']
scored_df = scored_df[~scored_df.event_type.isin(types_to_drop)]

# Remap event_types
event_type_map = {
        'mouth movements' : 'mouth or tongue movement',
        'tongue protrusion' : 'mouth or tongue movement',
        'mouth or tongue movement' : 'mouth or tongue movement',
        'lateral tongue movement' : 'lateral tongue protrusion',
        'lateral tongue protrusion' : 'lateral tongue protrusion',
        'gape' : 'gape',
        'no movement' : 'no movement',
        'nothing' : 'no movement',
        }

scored_df['event_type'] = scored_df['event_type'].map(event_type_map)
scored_df['event_codes'] = scored_df['event_type'].astype('category').cat.codes
scored_df['is_gape'] = (scored_df['event_type'] == 'gape')*1

scored_df.dropna(subset=['event_type'], inplace=True)

scored_df = scored_df[scored_df.event_type != 'lateral tongue protrusion']

scored_df['event_type'] = scored_df['event_type'].replace(
        ['mouth or tongue movement'],
        'MTMs'
        )

event_code_dict = {
        'gape' : 1,
        'MTMs' : 2,
        'no movement' : 0,
        }

scored_df['event_codes'] = scored_df['event_type'].map(event_code_dict)

############################################################
# Train model
############################################################
# Load hyperparameters
hyperparam_path = os.path.join(artifact_dir, 'xgb_optim_artifacts', 'best_xgb_hyperparams.csv')
best_hyperparams = pd.read_csv(hyperparam_path)
hparam_names = []
hparam_vals = []
for i, row in best_hyperparams.iterrows():
    hparam_names.append(row['name'])
    raw_value = row['value']
    dtype = row['dtype']
    if dtype == 'int':
        hparam_vals.append(int(raw_value))
    elif dtype == 'float':
        hparam_vals.append(float(raw_value))
hparam_dict = dict(zip(hparam_names, hparam_vals))

# Train on all data and save model
X_train = np.stack(scored_df.features.values)
y_train = scored_df.event_codes.values

# Calculate sample weights and normalize weight for each class
class_weights = scored_df.event_codes.value_counts(normalize=True)
inv_class_weights = 1 / class_weights
sample_weights = inv_class_weights.loc[y_train].values

# Train model
clf = xgb.XGBClassifier(**hparam_dict)
clf.fit(X_train, y_train, sample_weight=sample_weights)

# Save model
save_dir = os.path.join(artifact_dir, 'xgb_model')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
clf.save_model(os.path.join(save_dir, 'xgb_model.json'))

############################################################
# Data testing
# Making sure the training data is loaded correctly
############################################################
# # Leave one-animal-out cross validation
# unique_animals = scored_df.animal_num.unique()
# 
# for i, this_animal in enumerate(tqdm(unique_animals)):
#     train_df = scored_df[scored_df.animal_num != this_animal]
#     test_df = merge_gape_pal[merge_gape_pal.animal_num == this_animal]
# 
#     # Train model
#     X_train = np.stack(train_df.features.values)
#     y_train = train_df.event_codes.values
# 
#     # Calculate sample weights and normalize weight for each class
#     class_weights = train_df.event_codes.value_counts(normalize=True)
#     inv_class_weights = 1 / class_weights
#     sample_weights = inv_class_weights.loc[y_train].values 
# 
#     X_pred = np.stack(test_df.features.values)
# 
#     clf = xgb.XGBClassifier()
#     clf.fit(X_train, y_train, sample_weight=sample_weights)
#     # clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_pred)
# 
#     merge_gape_pal.loc[merge_gape_pal.animal_num == this_animal, 'xgb_pred'] = y_pred
# 
#     # For sanity checking, check accuracy on y_train
#     y_train_pred = clf.predict(X_train)
#     train_acc = accuracy_score(y_train, y_train_pred) 
#     print(f'Train Accuracy: {train_acc:.2f}')
# 
# # Convert xgb_pred to int
# merge_gape_pal['xgb_pred'] = merge_gape_pal['xgb_pred'].astype('int')
# 
# # Add event name to xgb_pred
# bsa_event_map = {
#         0 : 'nothing',
#         1 : 'gapes',
#         2 : 'MTMs',
#         }
# 
# merge_gape_pal['xgb_pred_event'] = merge_gape_pal['xgb_pred'].map(bsa_event_map)
# 
# ###############
# # Convert to array so downstream processing can be same as BSA
# JL_gape_shape = (4,30,7000)
# xgb_pred_array_list = []
# for session_num in tqdm(merge_gape_pal.session_ind.unique()):
#     this_peak_frame = merge_gape_pal.loc[merge_gape_pal.session_ind == session_num] 
#     event_array = np.zeros(JL_gape_shape)
#     inds = this_peak_frame[['taste','trial','segment_bounds']]
#     pred_vals = this_peak_frame['xgb_pred'].values
#     for ind, pred_val in enumerate(pred_vals):
#         taste, trial, segment_bounds = inds.iloc[ind]
#         updated_bounds = segment_bounds.copy()
#         updated_bounds += 2000
#         this_pred = pred_vals[ind]
#         event_array[taste,trial,updated_bounds[0]:updated_bounds[1]] = this_pred
# 
#     xgb_pred_array_list.append(event_array)
# 
# xgb_pred_array_list = np.array(xgb_pred_array_list)
# # Convert to int
# xgb_pred_array_list = xgb_pred_array_list.astype('int')
# 
# # Plot
# xgb_pred_plot_dir = os.path.join(plot_dir, 'pipeline_test_plots', 'xgb')
# if not os.path.exists(xgb_pred_plot_dir):
#     os.makedirs(xgb_pred_plot_dir)
# 
# event_color_map = {
#         0 : '#D1D1D1',
#         1 : '#EF8636',
#         2 : '#3B75AF',
#         }
# 
# taste_map_list = all_data_frame.taste_map.tolist()
# taste_list = [list(x.keys()) for x in taste_map_list]
# basenames = all_data_frame.basename.tolist()
# x_vec = np.arange(-2000, 5000)
# 
# # Create segmented colormap
# from matplotlib.colors import ListedColormap
# cmap = ListedColormap(list(event_color_map.values()), name = 'NBT_cmap')
# 
# for i, this_xgb_pred  in enumerate(xgb_pred_array_list):
#     fig, ax = plt.subplots(4,1,sharex=True,sharey=True,
#                            figsize=(5,10))
#     this_tastes = taste_list[i]
#     for taste in range(4):
#         im = ax[taste].pcolormesh(
#                 x_vec, np.arange(30), 
#                 this_xgb_pred[taste],
#                   cmap=cmap,vmin=0,vmax=2,)
#         ax[taste].set_ylabel(f'{this_tastes[taste]}' + '\nTrial #')
#         ax[0].set_title('XGB')
#         ax[0].set_title('BSA')
#     ax[-1].set_xlabel('Time (ms)')
#     cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
#     cbar = fig.colorbar(im, cax=cbar_ax)
#     cbar.set_ticks([0.5,1,1.5])
#     cbar.set_ticklabels(['nothing','gape','MTMs'])
#     basename = basenames[i]
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     fig.suptitle(basename)
#     fig.savefig(os.path.join(xgb_pred_plot_dir, basename + '_xgb_bsa_pred.png'),
#                 bbox_inches='tight', dpi = 300)
#     plt.close(fig)
