"""
XGB classifer has strong confusion between gape and MTM
Compare with JL-QDA classifier performance
"""
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from scipy.spatial.distance import mahalanobis
from pickle import dump, load
from matplotlib import pyplot as plt
from umap import UMAP
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import pingouin as pg

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
code_dir = os.path.join(base_dir, 'src')
sys.path.append(code_dir)  # noqa
from utils.extract_scored_data import return_taste_orders, process_scored_data  # noqa
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            JL_process,
                                            # gen_segment_frame,
                                            parse_segment_dat_list,
                                            parse_gapes_Li,
                                            threshold_movement_lengths,
                                            )  # noqa

# artifact_dir = os.path.join(base_dir, 'artifacts')
artifact_dir = os.path.join(code_dir, 'classifier_pipeline/artifacts')
plot_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering/plots/jl_comparison'

scored_df = pd.read_pickle(os.path.join(artifact_dir, 'fin_training_dataset.pkl'))
all_pred_frame = pd.read_pickle(os.path.join(artifact_dir, 'all_datasets_emg_pred.pkl'))
##############################
# Createa NCA plot
##############################
feature_array = np.stack(scored_df['features'].values)
X = feature_array.copy()
y = scored_df['updated_codes'].values

# Do not rescale as classifier was trained on unscaled data
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

n_dims = 2
nca = NeighborhoodComponentsAnalysis(n_components=n_dims)
X_transform = nca.fit_transform(X, y)
X_transform_plot = X_transform.copy()

# umap_obj = UMAP(n_components=n_dims)
# X_transform = umap_obj.fit_transform(X_transform)
# # Give everything a normal jitter to generate clouds
# sd = 1
# X_transform_plot = X_transform + np.random.normal(0, sd, X_transform.shape)

# Count each 'updated_event_type'
event_counts = scored_df['updated_event_type'].value_counts()

############################################################
# Perform predictions 
############################################################
event_code_dict_path = os.path.join(artifact_dir, 'event_code_dict.json')
with open(event_code_dict_path, 'r') as f:
    event_code_dict = json.load(f)

inv_event_code_dict = {v: k for k, v in event_code_dict.items()}

model_save_dir = os.path.join(artifact_dir, 'xgb_model')
clf = xgb.XGBClassifier() 
clf.load_model(os.path.join(model_save_dir, 'xgb_model.json'))

xgb_pred = clf.predict(X)
xgb_pred_names = [inv_event_code_dict[x] for x in xgb_pred]

##############################
fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(15,5))
for i, this_y in enumerate(np.unique(y)):
    y_name = scored_df.loc[scored_df['updated_codes'] == this_y, 'updated_event_type'].values[0]
    y_count = event_counts[y_name]
    if y_name == 'gape':
        zorder = 10
        edgecolor = 'k'
    else:
        zorder = 1
        edgecolor = 'none'
    ax[0].scatter(*list(X_transform_plot[y == this_y].T),
                alpha=0.3,
                label=f'{y_name} ({y_count})',
                zorder=zorder,
                edgecolor=edgecolor)
    ax[1].scatter(*list(X_transform_plot[y == this_y].T),
                alpha=0.3,
                label=f'{y_name} ({y_count})',
                zorder=zorder,
                edgecolor=edgecolor)
    xgb_pred_name = inv_event_code_dict[i]
    ax[2].scatter(*list(X_transform_plot[xgb_pred == i].T),
                alpha=0.3,
                label=f'{xgb_pred_name}',
                  )
# Mark JL gapes as black dots
jl_gapes = (scored_df['classifier'] == 1).values
xgb_gape_code = event_code_dict['gape']
xgb_gapes = (xgb_pred == xgb_gape_code)
ax[0].scatter(*list(X_transform_plot[jl_gapes].T), c='k', s=2, label='JL-QDA Gapes')
ax[0].set_title('True Labels (with JL Gapes)')
# Put legend below plot
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
ax[1].scatter(*list(X_transform_plot[xgb_gapes].T), c='k', s=2, label='XGB Gapes')
ax[1].set_title('True Labels (with XGB Gapes)')
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
ax[2].set_title('XGB Predictions')
ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
plt.suptitle('Neighborhood Components Analysis')
plt.tight_layout()
# plt.show()
fig.savefig(os.path.join(plot_dir, 'jl_comparison_nca.svg'),
            bbox_inches='tight') 
plt.close(fig)

##############################
# Calculate precision, recall, and f1 for
# both classifiers for gape vs everything-else
labelled_gapes = (scored_df['updated_event_type'] == 'gape').values * 1
jl_gapes = (scored_df['classifier'] == 1).values * 1
xgb_gapes = (xgb_pred == xgb_gape_code) * 1

# Bootstrap to estimate error
n_repeats = 100

metric_df_list = []
for i in trange(n_repeats):
    # Sample with replacement
    sample_inds = np.random.choice(len(labelled_gapes), len(labelled_gapes), replace=True)
    this_labelled_gapes = labelled_gapes[sample_inds]
    this_jl_gapes = jl_gapes[sample_inds]
    this_xgb_gapes = xgb_gapes[sample_inds]

    jl_precision = precision_score(this_labelled_gapes, this_jl_gapes)
    jl_recall = recall_score(this_labelled_gapes, this_jl_gapes)
    jl_f1 = f1_score(this_labelled_gapes, this_jl_gapes)

    xgb_precision = precision_score(this_labelled_gapes, this_xgb_gapes)
    xgb_recall = recall_score(this_labelled_gapes, this_xgb_gapes)
    xgb_f1 = f1_score(this_labelled_gapes, this_xgb_gapes)

    metric_df = pd.DataFrame({
        'classifier': ['JL-QDA', 'XGB'],
        'precision': [jl_precision, xgb_precision],
        'recall': [jl_recall, xgb_recall],
        'f1': [jl_f1, xgb_f1]
    })

    # Melt
    metric_df = metric_df.melt(id_vars='classifier', var_name='metric', value_name='score')
    metric_df['repeat'] = i

    metric_df_list.append(metric_df)

final_metric_df = pd.concat(metric_df_list)

# Perform nonparametric paired t-test for f1 score
jl_f1 = final_metric_df.loc[final_metric_df['classifier'] == 'JL-QDA', 'score'].values
xgb_f1 = final_metric_df.loc[final_metric_df['classifier'] == 'XGB', 'score'].values
wilcoxon_res = pg.wilcoxon(jl_f1, xgb_f1)
wilcoxon_p = wilcoxon_res['p-val'].values[0]

# Plot using seaborn
g = sns.catplot(
    data=final_metric_df,
    x='metric',
    y='score',
    hue='classifier',
    kind='bar',
    errorbar='sd',
)
g.set_axis_labels('Classifier', 'Score')
fig = plt.gcf()
fig.suptitle('Gape Classification Metrics\n'+\
        f'N={n_repeats} bootstraps' +\
        '\n' + f'Wilcoxon p for f1 ={wilcoxon_p:.3f}')
g.savefig(os.path.join(plot_dir, 'gape_classification_metrics.svg'),
          bbox_inches='tight')
plt.close()

# plt.imshow(feature_array, aspect='auto', interpolation='nearest')
# plt.show()
