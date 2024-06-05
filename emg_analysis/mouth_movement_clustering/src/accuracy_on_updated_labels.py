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
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from scipy import stats

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

X_mat = np.stack(scored_df.features.values)
# X_mat_raw = np.stack(scored_df.raw_features.values)
X_mat_scaled = StandardScaler().fit_transform(X_mat)
# X_mat_raw_scaled = StandardScaler().fit_transform(X_mat_raw)

# fig, ax = plt.subplots(2,2)
# ax[0,0].imshow(X_mat_raw, aspect='auto', interpolation='nearest')
# ax[0,1].imshow(X_mat, aspect='auto', interpolation='nearest')
# ax[1,0].imshow(X_mat_raw_scaled, aspect='auto', interpolation='nearest')
# ax[1,1].imshow(X_mat_scaled, aspect='auto', interpolation='nearest')
# plt.show()

session_group_list = [x[1] for x in scored_df.groupby('session')]

##############################
# JL Accuracy - Updated Annotations
##############################
jl_session_acc = []
jl_session_f1 = []
for this_group in session_group_list:
    jl_pred = this_group.classifier.values
    updated_annot = 1*((this_group.updated_event_type == 'gape').values)
    this_acc = accuracy_score(jl_pred, updated_annot)
    this_f1 = f1_score(jl_pred, updated_annot)
    jl_session_acc.append(this_acc)
    jl_session_f1.append(this_f1)

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

scored_df['bsa_aligned_event_codes'] = scored_df['event_type'].map(bsa_aligned_event_map)

# Get metrics for BSA
wanted_bsa_pred_list = np.array([bsa_to_pred(x) for x in wanted_bsa_p_list]).astype('int')
scored_df['bsa_pred'] = wanted_bsa_pred_list

#######################################
# plot frequency ranges for all movements
max_freq_ind = np.where(feature_names == 'max_freq')[0][0]
freq_vec = scored_df.raw_features.apply(lambda x: x[max_freq_ind])
event_vec = scored_df.event_type.values

bsa_omega = all_data_frame.bsa_omega.values[0]
omega_cutoff_inds = [6,11]
omega_cutoff_freqs = bsa_omega[omega_cutoff_inds]

bsa_inferred_freq = bsa_omega[wanted_bsa_p_list]

freq_df = pd.DataFrame(
        dict(
            max_freq = freq_vec,
            bsa_freq = bsa_inferred_freq,
            event = event_vec
            )
        )

fig, ax = plt.subplots(freq_df.event.nunique(), 2, sharex=True,
                       figsize = (10, 3*freq_df.event.nunique()))
for i, this_event in enumerate(freq_df.event.unique()):
    this_freqs = freq_df[freq_df.event == this_event].max_freq
    ax[i,0].hist(this_freqs, bins = 25, alpha=0.7)
    ax[i,0].set_ylabel(this_event)
    for this_cutoff in omega_cutoff_freqs:
        ax[i,0].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
    this_bsa_freqs = freq_df[freq_df.event == this_event].bsa_freq
    ax[i,1].hist(this_bsa_freqs, bins = 25, alpha=0.7)
    ax[i,1].set_ylabel(this_event)
    for this_cutoff in omega_cutoff_freqs:
        ax[i,1].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
ax[0,0].set_title('Max Freq')
ax[0,1].set_title('BSA Freq')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count')
plt.suptitle('Frequency Distribution of Mouth Movements')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'event_frequency_ranges.png'),
            bbox_inches='tight')
plt.close()

# Direct overlays of gape and 'mouth or tongue movement'
freq_regions = [[0, 4.15], [4.15, 6.4], [6.4, 20]]
freq_region_labels = ['Null', 'Gape', 'LTP']
fig,ax = plt.subplots(2,1, sharex=True, sharey=True)
for event_type in ['gape', 'mouth or tongue movement']:
    dat_inds = np.where(scored_df.event_type == event_type)
    max_freq = freq_vec.values[dat_inds]
    bsa_freq = bsa_inferred_freq[dat_inds]
    bsa_freq += np.random.normal(0, 0.1, len(bsa_freq))
    ax[0].hist(max_freq, bins = 25, alpha=0.7, label = event_type,
               density=True)
    ax[1].hist(bsa_freq, bins = 25, alpha=0.7, label = event_type,
               density=True)
ax[0].legend()
for this_cutoff in omega_cutoff_freqs:
    ax[0].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
    ax[1].axvline(this_cutoff, color='r', linestyle='--', alpha=0.7)
ax1_y_lims = ax[1].get_ylim()
for label, region in zip(freq_region_labels, freq_regions):
    ax[1].text(np.mean(region), ax1_y_lims[1]*0.75, label, fontweight='bold',
               ha='center')
ax[0].set_title('Max Freq')
ax[1].set_title('BSA Freq')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count')
plt.suptitle('Frequency Distribution of Gape and Mouth Movements')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'gape_mouth_freq_comparison.png'),
            bbox_inches='tight')
plt.close()


# Correlate max freq with bsa freq
rho, p = stats.spearmanr(freq_df.max_freq, freq_df.bsa_freq)
rho, p = np.round(rho, 3), np.round(p, 3)

fig, ax = plt.subplots()
ax.scatter(freq_df.max_freq, freq_df.bsa_freq,
            alpha=0.1)
ax.plot([0,15],[0,15],'k--')
ax.set_xlabel('Max Freq')
ax.set_ylabel('BSA Freq')
ax.set_title('Max Freq vs BSA Freq\n' + \
        f'Spearman Rho: {rho}, p: {p}')
ax.set_aspect('equal')
plt.savefig(os.path.join(plot_dir, 'max_freq_vs_bsa_freq.png'),)
plt.close()

##############################
# Leave one session out
##############################
test_y_list = []
pred_y_list = []
bsa_pred_y_list = []
mean_abs_shap_list = []
for i, this_session in enumerate(tqdm(unique_sessions)):
    # Leave out this session
    train_df = scored_df[scored_df.session_ind != this_session]
    test_df = scored_df[scored_df.session_ind == this_session]

    # Train classifier
    X_train = np.stack(train_df.features.values)
    y_train = train_df.bsa_aligned_event_codes.values
    X_test = np.stack(test_df.features.values)
    y_test = test_df.bsa_aligned_event_codes.values

    bsa_pred_y = test_df.bsa_pred.values
    bsa_pred_y_list.append(bsa_pred_y)

    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_y_list.append(y_test)
    pred_y_list.append(y_pred)

    # Get shap values
    # explainer = shap.Explainer(clf)
    # shap_values = explainer(X_train)
    # mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # mean_abs_shap_list.append(mean_abs_shap)
    mean_abs_shap_list.append(clf.feature_importances_)


# Get accuracy and f1
bsa_accuracy = [accuracy_score(y_test, bsa_pred_y) \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
accuracy = [accuracy_score(y_test, pred_y) for y_test, pred_y in zip(test_y_list, pred_y_list)]
bsa_f1 = [f1_score(y_test, bsa_pred_y, average='macro') \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
f1 = [f1_score(y_test, pred_y, average='macro') \
        for y_test, pred_y in zip(test_y_list, pred_y_list)]
bsa_confusion = [confusion_matrix(y_test, bsa_pred_y, normalize='true') \
        for y_test, bsa_pred_y in zip(test_y_list, bsa_pred_y_list)]
confusion = [confusion_matrix(y_test, pred_y, normalize='true') \
        for y_test, pred_y in zip(test_y_list, pred_y_list)]

