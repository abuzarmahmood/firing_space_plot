"""
Timing comparison of JL-QDA, BSA, and XGB-classifier on datasets

Starting point will be extracted EMG envelopes

Make a copy of current "*BSA_results" folder in case something goes wrong
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import xgboost as xgb

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')

############################################################
# Load data
############################################################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
session_specific_plot_dir = os.path.join(plot_dir, 'session_specific_plots')
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

############################################################
# Get BSA Inference Times 
############################################################
# Load BSA inference results
all_emg_env_array_path = os.path.join(artifact_dir, 'all_emg_env_array.npy')
# all_emg_env_array = np.stack(all_data_frame.env.tolist())
# np.save(all_emg_env_array_path, all_emg_env_array)
all_emg_env_array = np.load(all_emg_env_array_path)

# Import R related stuff - use rpy2 for Python->R and pandas for R->Python
# Needed for the next line to work on Anaconda. 
# Also needed to do conda install -c r rpy2 at the command line
import readline 
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
from rpy2.robjects import r

# Fire up BaSAR on R
basar = importr('BaSAR')

all_inds = list(np.ndindex(all_emg_env_array.shape[:-1]))
run_n = 100

bsa_preprocess_time_list_path = os.path.join(artifact_dir, 'bsa_preprocess_time_list.npy')
if os.path.exists(bsa_preprocess_time_list_path):
    bsa_preprocess_time_list = np.load(bsa_preprocess_time_list_path)
else:
    bsa_preprocess_time_list = []

left_runs = run_n - len(bsa_preprocess_time_list)
run_inds = np.random.choice(
        np.arange(len(all_inds)), left_runs, replace=False)
inds = [all_inds[ind] for ind in run_inds]

for i, ind in enumerate(tqdm(inds)):

    start_time = time()

    input_data = all_emg_env_array[ind]

    # Make the time array and assign it to t on R
    T = (np.arange(7000) + 1)/1000.0
    t_r = ro.r.matrix(T, nrow = 1, ncol = 7000)
    ro.r.assign('t_r', t_r)
    ro.r('t = c(t_r)')

    # Run BSA on trial 'trial' of taste 'taste' and assign the results to p and omega.
    # input_data = emg_env[taste, trial, :]
    # input_data = emg_env[task]
    # Check that trial is non-zero, if it isn't, don't try to run BSA
    if not any(np.isnan(input_data)):

        Br = ro.r.matrix(input_data, nrow = 1, ncol = 7000)
        ro.r.assign('B', Br)
        ro.r('x = c(B[1,])')

        # x is the data, 
        # we scan periods from 0.1s (10 Hz) to 1s (1 Hz) in 20 steps. 
        # Window size is 300ms. 
        # There are no background functions (=0)
        ro.r('r_local = BaSAR.local(x, 0.1, 1, 20, t, 0, 300)') 
        p_r = r['r_local']
        # r_local is returned as a length 2 object, 
        # with the first element being omega and the second being the 
        # posterior probabilities. These need to be recast as floats
        p = np.array(p_r[1]).astype('float')
        omega = np.array(p_r[0]).astype('float')/(2.0*np.pi) 
        # print(f'Trial {task:03} succesfully processed')
    else:
        print(f'NANs in trial {task:03}, BSA will also output NANs')
        p = np.zeros((7000,20))
        omega = np.zeros(20)
        p[:] = np.nan
        omega = np.nan

    print(f'{i} of {run_n} done')

    end_time = time()
    preprocess_time = end_time - start_time
    bsa_preprocess_time_list.append(preprocess_time)

    np.save(bsa_preprocess_time_list_path, bsa_preprocess_time_list)


############################################################
# Get BSA Prediction times
############################################################
bsa_p_array = np.stack(all_data_frame.bsa_p.tolist())

def bsa_to_pred_trial(x):
    """
    Convert BSA inference to prediction

    Input:
        x: np.array of shape (num_time_bins)

    Output:
        y: np.array of shape (num_time_bins)
    """
    pred_array = np.zeros_like(x)
    gape_inds = np.logical_and(x>=6, x<11)
    ltp_inds = x>=11
    other_inds = np.logical_not(np.logical_or(gape_inds, ltp_inds))

    pred_array[gape_inds] = 0
    pred_array[ltp_inds] = 1
    pred_array[other_inds] = 2

    return pred_array

inds = list(np.ndindex(bsa_p_array.shape[:-1]))
bsa_pred_time_list = []
for ind in tqdm(inds):
    start_time = time()
    _ = bsa_to_pred_trial(bsa_p_array[ind])
    end_time = time()
    classification_time = end_time - start_time
    bsa_pred_time_list.append(classification_time)

bsa_timing_frame = pd.DataFrame(bsa_pred_time_list, columns=['classification'])
bsa_timing_frame.to_csv(os.path.join(artifact_dir, 'bsa_timing_frame.csv'))


############################################################
# Get XGBoost Prediction times 
############################################################
merge_gape_pal_path = os.path.join(artifact_dir, 'merge_gape_pal.pkl')
merge_gape_pal = pd.read_pickle(merge_gape_pal_path)

scored_df = merge_gape_pal[merge_gape_pal.scored == True]

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
        }

scored_df['event_type'] = scored_df['event_type'].map(event_type_map)
scored_df['event_codes'] = scored_df['event_type'].astype('category').cat.codes
scored_df['is_gape'] = (scored_df['event_type'] == 'gape')*1

scored_df.dropna(subset=['event_type'], inplace=True)

bsa_aligned_event_map = {
        'gape' : 0,
        'mouth or tongue movement' : 1,
        'lateral tongue protrusion' : 2,
        'no movement' : 2,
        }

bsa_labels = ['gape', 'mouth or tongue movement', 'LTP/Nothing']

scored_df['bsa_aligned_event_codes'] = scored_df['event_type'].map(bsa_aligned_event_map)

##############################
# Train model

X_train = np.stack(scored_df.features.values)
y_train = scored_df.bsa_aligned_event_codes.values
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

##############################
# Predict : all events in one trial at a time 
events_trial_group = list(merge_gape_pal.groupby(['session', 'taste', 'trial']))
events_trial_group_inds = [x[0] for x in events_trial_group]
events_trial_group_data = [x[1] for x in events_trial_group]

xgb_pred_time_list = []
for i, trial_group in tqdm(enumerate(events_trial_group_data)):
    start_time = time()
    X_test = np.stack(trial_group.features.values)
    y_pred = clf.predict(X_test)
    end_time = time()
    classification_time = end_time - start_time
    xgb_pred_time_list.append(classification_time)

############################################################
# Compile 
############################################################
# Get timing frames from artifacts
li_timing_frame_path = os.path.join(artifact_dir, 'Li_time_frame.csv')
am_timing_frame_path = os.path.join(artifact_dir, 'AM_time_frame.csv')
bsa_timing_frame_path = os.path.join(artifact_dir, 'bsa_timing_frame.csv')

li_timing_frame = pd.read_csv(li_timing_frame_path, index_col=0)
am_timing_frame = pd.read_csv(am_timing_frame_path, index_col=0)
bsa_timing_frame = pd.read_csv(bsa_timing_frame_path, index_col=0)

am_timing_frame['classification'] = xgb_pred_time_list

li_melted = li_timing_frame.melt(var_name='step', value_name='time')
li_melted['classifier'] = 'Li'

am_melted = am_timing_frame.melt(var_name='step', value_name='time')
am_melted['classifier'] = 'AM'

bsa_melted = bsa_timing_frame.melt(var_name='step', value_name='time')
bsa_melted['classifier'] = 'BSA'

# Since BSA preprocessing times are not all full trial-set,
# will need to add them to the melted frame
bsa_preprocess_time_list = np.load(bsa_preprocess_time_list_path)
bsa_preprocess_time_frame = pd.DataFrame(bsa_preprocess_time_list, columns=['time'])
bsa_preprocess_time_frame['step'] = 'preprocessing'
bsa_preprocess_time_frame['classifier'] = 'BSA'
bsa_melted = pd.concat([bsa_melted, bsa_preprocess_time_frame])

timing_frame = pd.concat([li_melted, am_melted, bsa_melted])
timing_frame.dropna(inplace=True)
timing_frame.reset_index(drop=True, inplace=True)

# Plot time for each trial per classifier
g = sns.catplot(
        data=timing_frame, 
        x='classifier', y='time', 
        hue='step', 
        kind='boxen'
        )
g.axes[0,0].set_ylabel('Time (s)')
g.set(yscale='log')
g.axes[0,0].set_xticklabels(['JL-QDA', 'XGB', 'BSA'])
g.axes[0,0].set_ylim(1e-5)
# plt.show()
plt.title('Timing Comparison of JL-QDA, XGB, and BSA\n' +\
        'Run-time for single trial (7000ms)')
plt.savefig(os.path.join(plot_dir, 'timing_comparison.png'),
            bbox_inches='tight')
plt.close()

# Plot total time taken per classifier
am_total_time = am_timing_frame.sum(axis=1)
li_total_time = li_timing_frame.sum(axis=1)
am_total_time = pd.DataFrame(am_total_time, columns=['time'])
li_total_time = pd.DataFrame(li_total_time, columns=['time'])
am_total_time['classifier'] = 'AM'
li_total_time['classifier'] = 'Li'

bsa_total_time = bsa_timing_frame.mean().values[0] + bsa_preprocess_time_frame.time.mean()
bsa_total_time = pd.DataFrame(
        [bsa_total_time, 'BSA'],
        index=['time', 'classifier']).T

total_time_frame = pd.concat([li_total_time, am_total_time, bsa_total_time])

g = sns.barplot(data=total_time_frame, x='classifier', y='time',
                errorbar='sd')
g.set(yscale='log')
g.set_ylabel('Total Time (s)')
g.set_xticklabels(['JL-QDA', 'XGB', 'BSA'])
plt.title('Total Time (Preprocessing + Classification)\n' +\
        'Comparison of JL-QDA, XGB, and BSA\n' +\
        'Run-time for single trial (7000ms)')
plt.savefig(os.path.join(plot_dir, 'total_timing_comparison.png'),
            bbox_inches='tight')
plt.close()

# Get total time normalized to xgb
mean_am_time = total_time_frame.loc[total_time_frame.classifier == 'AM', 'time'].mean()
ratio_time_frame = total_time_frame.copy()
ratio_time_frame['time'] = ratio_time_frame['time']/mean_am_time
ratio_time_frame['orders'] = np.vectorize(np.log10)(ratio_time_frame['time'].values)
ratio_time_frame.dropna(inplace=True)
ratio_time_frame.reset_index(drop=True, inplace=True) 

g = sns.barplot(data=ratio_time_frame, x='classifier', y='orders',
                errorbar='sd')
g.set_ylabel('Orders of Magnitude Relative to XGB')
g.set_xticklabels(['JL-QDA', 'XGB', 'BSA'])
plt.title('Total Time (Preprocessing + Classification)\n' +\
        'Normalized to XGB\n' +\
        'Comparison of JL-QDA, XGB, and BSA\n' +\
        'Run-time for single trial (7000ms)')
plt.savefig(os.path.join(plot_dir, 'total_timing_comparison_ratio.png'),
            bbox_inches='tight')
plt.close()

