"""
Given classifier predictions, create plots of

1. Features UMAP colored by predicted class
2. Line overlap plot for each predicted class
3. Heatmap of features per class and session
4. Raster plot of predicted classes
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from sklearn.decomposition import PCA
from tqdm import tqdm

def return_pred_array(taste_frame):
    """
    Given a taste_frame, return a 2D array of predictions
    
    Inputs:
        taste_frame : pd.DataFrame

    Outputs:
        pred_array : np.array
            2D array with shape (n_trials, max_time)
    """

    assert len(taste_frame.taste.unique()) == 1
    assert len(taste_frame.basename.unique()) == 1
    n_trials = taste_frame.trial.max() + 1
    max_time = np.max([x for y in taste_frame.segment_bounds for x in y] )
    # Round up to nearest 1000
    max_time = int(np.ceil(max_time/100) * 100)
    pred_array = np.zeros((n_trials, max_time))
    pred_array[:] = np.nan 
    for _, this_row in taste_frame.iterrows():
        this_trial = this_row.trial
        this_bounds = this_row.segment_bounds
        this_pred = this_row.xgb_pred_code
        pred_array[this_trial, this_bounds[0]:this_bounds[1]] = this_pred
    return pred_array

############################################################
artifact_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering/src/classifier_pipeline/artifacts' 

xgb_pred_plot_dir = os.path.join(artifact_dir, 'pipeline_test_plots')
if not os.path.exists(xgb_pred_plot_dir):
    os.makedirs(xgb_pred_plot_dir)

bsa_event_map = {
        0 : 'no movement',
        1 : 'gape',
        2 : 'MTMs',
        }
event_color_map = {
        0 : '#D1D1D1',
        1 : '#EF8636',
        2 : '#3B75AF',
        }
inv_bsa_event_map = {v: k for k, v in bsa_event_map.items()}

predicted_df = pd.read_pickle(os.path.join(artifact_dir, 'all_datasets_emg_pred.pkl'))
predicted_df.reset_index(inplace=True)
predicted_df['xgb_pred_code'] = predicted_df['pred_event_type'].map(inv_bsa_event_map)
feature_names = open(os.path.join(artifact_dir, 'all_datasets_feature_names.txt')).read().split('\n') 
predicted_df['animal_num'] = predicted_df.basename.str.split('_').str[0]
predicted_df['animal_code'] = predicted_df.animal_num.astype('category').cat.codes
predicted_df['session_code'] = predicted_df.basename.astype('category').cat.codes

cmap = ListedColormap(list(event_color_map.values()), name = 'NBT_cmap')

############################################################
# 1- Features UMAP colored by predicted class
# Create UMAP
predicted_df = predicted_df.sort_values(['xgb_pred_code', 'animal_code', 'session_code'])
feature_array = np.stack(predicted_df.features.values)
# Clip at +/- 3
feature_array = np.clip(feature_array, -3, 3)

umap = UMAP(n_components=2)
# umap = PCA(n_components=2) 
umap.fit(feature_array)
X_umap = umap.transform(feature_array) 

# plt.imshow(umap.components_, interpolation='none', cmap='viridis')
# plt.show()

# Plot
fig, ax = plt.subplots()
plt.scatter(X_umap[:,0], X_umap[:,1], c=predicted_df.xgb_pred_code, cmap=cmap,
            s=1, alpha=0.5)
ax.legend(title = 'Predicted Class')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.title('UMAP of Features Colored by Predicted Class')
fig.savefig(os.path.join(xgb_pred_plot_dir, 'xgb_pred_umap.png'),
            bbox_inches='tight', dpi = 300)
plt.close(fig)


############################################################
# 3- Heatmap of features per class and session
fig, ax = plt.subplots(1,5, sharey=True, sharex='col',
                       figsize=(20,5))
# Sort by prediction
pred_array = np.stack(predicted_df.xgb_pred_code.values)
animal_codes = np.stack(predicted_df.animal_code.values)
session_codes = np.stack(predicted_df.session_code.values)
ax[0].imshow(X_umap, aspect='auto', cmap='viridis', 
             interpolation='none')
ax[1].imshow(feature_array, aspect='auto', cmap='viridis', vmin = -3, vmax = 3,
           interpolation='none')
ax[1].set_xlabel('Feature #')
ax[1].set_xticks(np.arange(len(feature_names)))
ax[1].set_xticklabels(feature_names, rotation=90)
ax[2].imshow(pred_array[:,None], aspect='auto', cmap=cmap)
ax[3].imshow(animal_codes[:,None], aspect='auto', cmap='tab20')
ax[4].imshow(session_codes[:,None], aspect='auto', cmap='tab20')
ax[0].set_title('PCA Features')
ax[1].set_title('Feature Heatmap')
ax[2].set_title('Predicted Class')
ax[3].set_title('Animal Code')
ax[4].set_title('Session Code')
fig.savefig(os.path.join(xgb_pred_plot_dir, 'xgb_pred_heatmap_ind.png'),
            bbox_inches='tight', dpi = 300)
plt.close(fig)

###############

# g = sns.clustermap(feature_array, row_colors=row_colors, cmap='viridis',
#                    row_cluster=False, col_cluster=False,
#                    vmin = -2, vmax = 2)
# # Set feature_names as x-axis labels
# g.ax_heatmap.set_xticks(np.arange(len(feature_names))+0.5)
# g.ax_heatmap.set_xticklabels(feature_names, rotation=90)
# legend_elements = [mpl.lines.Line2D([0], [0], color=cmap(i), label=event_type,
#                                     linewidth = 5) \
#         for i, event_type in bsa_event_map.items()] 
# g.ax_heatmap.legend(handles=legend_elements, title='Event Type',
#                     bbox_to_anchor=(1.04,1), loc='upper left')
# g.ax_heatmap.set_xlabel('Feature #')
# plt.suptitle('XGB Predicted Class Heatmap')
# plt.savefig(os.path.join(xgb_pred_plot_dir, 'xgb_pred_heatmap.png'),
#             bbox_inches='tight', dpi = 300)
# plt.close()



############################################################

# Add event name to xgb_pred



###############

pred_dict = {}
for session_name, session_frame in tqdm(list(predicted_df.groupby('basename'))):
    for taste_name, taste_frame in list(session_frame.groupby('taste_name')):
        pred_array = return_pred_array(taste_frame)
        this_pred_dict = dict(
                pred_array = pred_array,
                taste_name = taste_name,
                session_name = session_name,
                )
        pred_dict[f'{session_name}_{taste_name}'] = this_pred_dict

pred_frame = pd.DataFrame(pred_dict).T

# Plot
# xgb_pred_plot_dir = os.path.join(plot_dir, 'pipeline_test_plots', 'xgb')

# Create segmented colormap

# Plot one session at a time
# for i, this_xgb_pred  in enumerate(xgb_pred_array_list):
for session_name, session_frame in list(pred_frame.groupby('session_name')):
    this_xgb_pred = session_frame.pred_array.to_numpy()
    this_tastes = session_frame.taste_name.tolist()

    fig, ax = plt.subplots(len(this_tastes),1,sharex=True,sharey=True,
                           figsize=(5,10))
    for taste in range(4):
        try:
            this_array = this_xgb_pred[taste]
            max_trials = this_array.shape[0]
            x_vec = np.arange(this_array.shape[1])
            im = ax[taste].pcolormesh(
                    x_vec, np.arange(max_trials), 
                    this_xgb_pred[taste],
                      cmap=cmap,vmin=0,vmax=2,)
            ax[taste].set_ylabel(f'{this_tastes[taste]}' + '\nTrial #')
        except:
            print(f'{session_name} {taste} ind not found')
    ax[0].set_title('XGB')
    ax[-1].set_xlabel('Time (ms)')
    cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0.5,1,1.5])
    cbar.set_ticklabels(['nothing','gape','MTMs'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle(session_name)
    fig.savefig(os.path.join(xgb_pred_plot_dir, session_name + '_xgb_bsa_pred_test.png'),
                bbox_inches='tight', dpi = 300)
    plt.close(fig)
