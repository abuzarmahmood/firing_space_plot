"""
Calculate cross-validated accuracy of all algorithms on updated label
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
code_dir = os.path.join(base_dir, 'src')
sys.path.append(code_dir)
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            find_segment,
                                            calc_peak_interval,
                                            JL_process,
                                            gen_gape_frame,
                                            threshold_movement_lengths,
                                            )

sys.path.append(os.path.expanduser('~/Desktop/blech_clust/emg/gape_QDA_classifier'))
from QDA_classifier import QDA

artifact_dir = os.path.join(base_dir, 'artifacts')

############################################################
# Load data
############################################################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

updated_plot_dir = os.path.join(plot_dir, 'updated_annotation_accuracy')
if not os.path.exists(updated_plot_dir):
    os.makedirs(updated_plot_dir)

##############################
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

# scored_df['event_type'] = scored_df['event_type'].replace(
#         ['lateral tongue protrusion','no movement'],
#         'other'
#         )

scored_df['event_type'] = scored_df['event_type'].replace(
        ['mouth or tongue movement'],
        'MTMs'
        )

event_code_dict = {
        'gape' : 1,
        'MTMs' : 2,
        # 'other' : 0,
        'no movement' : 0,
        }

scored_df['event_codes'] = scored_df['event_type'].map(event_code_dict)

############################################################


session_group_list = [x[1] for x in scored_df.groupby('session')]

##############################
# JL Accuracy - Updated Annotations
##############################
scored_df['updated_gapes'] = scored_df['updated_event_type'] == 'gape'
scored_df['updated_gapes'] = scored_df['updated_gapes'].map({True:'gape', False:'not gape'})

event_counts = scored_df.groupby('session').updated_gapes.value_counts()
event_counts = pd.DataFrame(event_counts)
event_counts.reset_index(inplace=True)
event_counts = event_counts.pivot(
        index = 'session', columns = 'updated_gapes', values = 'count')
event_counts.reset_index(inplace=True)

event_fractions = event_counts.copy()
event_fractions[['gape','not gape']] = event_fractions[['gape','not gape']].div(
        event_fractions[['gape','not gape']].sum(axis=1), axis=0)

event_counts.plot(x = 'session', kind='bar', stacked=True)
plt.tight_layout()
plt.legend(title = 'Is Gape?')
plt.xlabel('Session')
plt.ylabel('Count')
plt.title('Updated Gape Counts Across Sessions')
plt.savefig(os.path.join(updated_plot_dir, 'updated_gape_counts.png'),
            bbox_inches='tight')
plt.close()

event_fractions.plot(x = 'session', kind='bar', stacked=True)
plt.tight_layout()
plt.legend(title = 'Is Gape?')
plt.xlabel('Session')
plt.ylabel('Fraction')
plt.title('Updated Gape Fractions Across Sessions')
plt.savefig(os.path.join(updated_plot_dir, 'updated_gape_fractions.png'),
            bbox_inches='tight')
plt.close()

jl_session_acc = []
jl_session_f1 = []
for this_group in session_group_list:
    jl_pred = this_group.classifier.values
    updated_annot = 1*((this_group.updated_event_type == 'gape').values)
    this_acc = accuracy_score(jl_pred, updated_annot)
    this_f1 = f1_score(jl_pred, updated_annot)
    jl_session_acc.append(this_acc)
    jl_session_f1.append(this_f1)

acc_frac_corr = pearsonr(jl_session_acc, event_fractions['gape'])
rho, pval = acc_frac_corr
plt.scatter(jl_session_acc, event_fractions['gape'])
plt.xlabel('Accuracy')
plt.ylabel('Fraction of Gapes')
plt.title('JL Accuracy vs Gape Fraction\n' + f'rho = {rho:.2f}, p = {pval:.2f}')
plt.savefig(os.path.join(updated_plot_dir, 'jl_acc_vs_gape_fraction.png'),
            bbox_inches='tight')
plt.close()

############################################################
# BSA Accuracy - Updated Annotations 
############################################################
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

# Get BSA for given events
# gape = 6:11
# LTP = 11:
wanted_bsa_p_list = []
for i, this_event_row in scored_df.iterrows():
    taste = this_event_row['taste']
    trial = this_event_row['trial']
    time_lims = np.array(this_event_row['segment_bounds'])+2000
    all_data_ind = all_data_frame.loc[all_data_frame.basename == this_event_row['basename']].index[0]
    bsa_p_array = all_data_frame.loc[all_data_ind,'bsa_p']
    bsa_dat = bsa_p_array[taste, trial, time_lims[0]:time_lims[1]]
    bsa_mode = stats.mode(bsa_dat, axis = 0).mode
    wanted_bsa_p_list.append(bsa_mode)
wanted_bsa_p_list = np.array(wanted_bsa_p_list).astype('int')

# Convert bsa_p to predictions
def bsa_to_pred(x):
    if np.logical_and(x>=6, x<11):
        return 0
    elif x>=11:
        return 1
    else:
        return 2

bsa_aligned_event_map = {
        'gape' : 0,
        'mouth or tongue movement' : 1,
        'lateral tongue protrusion' : 2,
        'no movement' : 2,
        }

bsa_labels = ['gape', 'mouth or tongue movement', 'LTP/Nothing']

scored_df['bsa_aligned_event_codes'] = scored_df['updated_event_type'].map(bsa_aligned_event_map)
scored_df['bsa_aligned_event_names'] = scored_df['bsa_aligned_event_codes'].map({0:'gape', 1:'MTMs', 2:'LTP/Nothing'})

# Get metrics for BSA
wanted_bsa_pred_list = np.array([bsa_to_pred(x) for x in wanted_bsa_p_list]).astype('int')
scored_df['bsa_pred'] = wanted_bsa_pred_list

##############################
# Plots of counts and fractions for bsa_aligned_event_codes

event_counts = scored_df.groupby('session').bsa_aligned_event_names.value_counts()
event_counts = pd.DataFrame(event_counts)
event_counts.reset_index(inplace=True)
event_counts = event_counts.pivot(
        index = 'session', columns = 'bsa_aligned_event_names', values = 'count')
# event_counts.reset_index(inplace=True)

event_fractions = event_counts.copy()
event_fractions = event_fractions.div(event_fractions.sum(axis=1), axis=0)

event_counts.plot(kind='bar', stacked=True)
plt.tight_layout()
plt.legend(title = 'Event Type')
plt.xlabel('Session')
plt.ylabel('Count')
plt.title('BSA Aligned Event Counts Across Sessions')
plt.savefig(os.path.join(updated_plot_dir, 'bsa_aligned_event_counts.png'),
            bbox_inches='tight')
plt.close()

event_fractions.plot(kind='bar', stacked=True)
plt.tight_layout()
plt.legend(title = 'Event Type')
plt.xlabel('Session')
plt.ylabel('Fraction')
plt.title('BSA Aligned Event Fractions Across Sessions')
plt.savefig(os.path.join(updated_plot_dir, 'bsa_aligned_event_fractions.png'),
            bbox_inches='tight')
plt.close()

##############################

session_group_list = [x[1] for x in scored_df.groupby('session')]

bsa_session_acc = []
bsa_session_f1 = []
bsa_conf_mat_list = []
for this_group in session_group_list:
    bsa_pred = this_group.bsa_pred.values
    updated_annot = this_group.bsa_aligned_event_codes.values
    this_acc = accuracy_score(bsa_pred, updated_annot)
    this_f1 = f1_score(bsa_pred, updated_annot, average = 'weighted')
    this_conf_mat = confusion_matrix(bsa_pred, updated_annot, normalize = 'true')
    bsa_session_acc.append(this_acc)
    bsa_session_f1.append(this_f1)
    bsa_conf_mat_list.append(this_conf_mat)

############################################################
# XGB Accuracy - Updated Annotations 
############################################################
X_mat = np.stack(scored_df.features.values)
X_mat_scaled = StandardScaler().fit_transform(X_mat)
scored_df['scaled_features'] = [x for x in X_mat_scaled]

test_y_list = []
pred_y_list = []
for i, this_session in enumerate(tqdm(scored_df.session_ind.unique())):

    this_session_data = scored_df[scored_df.session_ind == this_session]
    this_animal_num = this_session_data.animal_num.values[0]

    # Leave out this session
    train_df = scored_df[scored_df.animal_num != this_animal_num]
    # test_df = scored_df[scored_df.animal_num == this_animal_num]
    test_df = this_session_data.copy() 

    # Train classifier
    X_train = np.stack(train_df.scaled_features.values)
    y_train = train_df.bsa_aligned_event_codes.values
    X_test = np.stack(test_df.scaled_features.values)
    y_test = test_df.bsa_aligned_event_codes.values

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_y_list.append(y_test)
    pred_y_list.append(y_pred)

xgb_accu_list = [accuracy_score(x,y) for x,y in zip(test_y_list, pred_y_list)]
xgb_f1_list = [f1_score(x,y, average = 'weighted') for x,y in zip(test_y_list, pred_y_list)]
xgb_conf_mat_list = [confusion_matrix(x,y, normalize = 'true') for x,y in zip(test_y_list, pred_y_list)]

############################################################
# Make plots
############################################################

# XGB vs BSA accuracy and f1
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].scatter(xgb_accu_list, bsa_session_acc)
min_val = min(min(xgb_accu_list), min(bsa_session_acc))
max_val = max(max(xgb_accu_list), max(bsa_session_acc))
ax[0].plot([min_val, max_val], [min_val, max_val], 'k--')
ax[0].set_xlabel('XGB Accuracy')
ax[0].set_ylabel('BSA Accuracy')
ax[0].set_title('XGB vs BSA Accuracy')
ax[1].scatter(xgb_f1_list, bsa_session_f1)
min_val = min(min(xgb_f1_list), min(bsa_session_f1))
max_val = max(max(xgb_f1_list), max(bsa_session_f1))
ax[1].plot([min_val, max_val], [min_val, max_val], 'k--')
ax[1].set_xlabel('XGB F1')
ax[1].set_ylabel('BSA F1')
ax[1].set_title('XGB vs BSA F1')
fig.suptitle('XGB vs BSA Accuracy and F1\nLeave one animal out cross-validation')
plt.tight_layout()
plt.savefig(os.path.join(updated_plot_dir, 'xgb_bsa_acc_f1.png'),
            bbox_inches='tight')
plt.close()

# Comparison of confusion matrices
mean_bsa_confusion = np.mean(bsa_conf_mat_list, axis = 0)
std_bsa_confusion = np.std(bsa_conf_mat_list, axis = 0)
mean_xgb_confusion = np.mean(xgb_conf_mat_list, axis = 0)
std_xgb_confusion = np.std(xgb_conf_mat_list, axis = 0)

fig, ax = plt.subplots(1,2, figsize=(10,5))
img = ax[0].matshow(mean_bsa_confusion, cmap='gray', vmin=0, vmax=1)
# plt.colorbar(img, ax=ax[0])
for i in range(mean_bsa_confusion.shape[0]):
    for j in range(mean_bsa_confusion.shape[1]):
        mean_val = mean_bsa_confusion[i,j]
        if mean_val < 0.5:
            c = 'white'
        else:
            c = 'black'
        ax[0].text(j, i, f'{mean_val:.2f}\n±{std_bsa_confusion[i,j]:.2f}',
                 ha='center', va='center', color=c, fontweight='bold')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')
ax[0].set_xticks(np.arange(3))
ax[0].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[0].set_yticks(np.arange(3))
ax[0].set_yticklabels(bsa_labels)
ax[0].set_title('BSA: Mean+Std Confusion Matrix')
img = ax[1].matshow(mean_xgb_confusion, cmap='gray', vmin=0, vmax=1)
# plt.colorbar(img, ax=ax[1])
for i in range(mean_xgb_confusion.shape[0]):
    for j in range(mean_xgb_confusion.shape[1]):
        mean_val = mean_xgb_confusion[i,j]
        if mean_val < 0.5:
            c = 'white'
        else:
            c = 'black'
        ax[1].text(j, i, f'{mean_val:.2f}\n±{std_xgb_confusion[i,j]:.2f}',
                 ha='center', va='center', color=c, fontweight='bold')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')
ax[1].set_xticks(np.arange(3))
ax[1].set_xticklabels(bsa_labels, rotation=45, ha='left')
ax[1].set_yticks(np.arange(3))
ax[1].set_yticklabels(bsa_labels)
ax[1].set_title('XGB: Mean+Std Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(updated_plot_dir, 'mean_std_confusion_diff_session.png'),
            bbox_inches='tight', dpi = 300)
plt.close()


