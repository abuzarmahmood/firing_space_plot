import os
import sys
from glob import glob
from pickle import dump, load

import numpy as np
import tables
import pylab as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Patch
import seaborn as sns
import pingouin as pg

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import shap
from umap import UMAP
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.metrics import classification_report

############################################################
############################################################

##############################
# Plotting 
##############################
def plot_event_examples(scored_gape_frame, plot_dir, basename):
    wanted_event_types = scored_gape_frame.event_type.unique()

    # Plot examples of all wanted events
    plot_gape_frame = scored_gape_frame.loc[scored_gape_frame.event_type.isin(wanted_event_types)]

    fig,ax = plt.subplots(len(wanted_event_types), 1, 
                          sharex=True, sharey=True,
                          figsize = (5,15))
    for this_event, this_ax in zip(wanted_event_types, ax):
        this_dat = plot_gape_frame.loc[plot_gape_frame.event_type == this_event]
        this_plot_dat = this_dat.segment_raw.values.T
        for this_seg in this_plot_dat:
            this_ax.plot(this_seg, color='k', alpha=0.1)
        this_ax.set_title(this_event)
    fig.savefig(os.path.join(plot_dir, f'{basename}_wanted_event_examples'), dpi = 150,
                bbox_inches='tight')
    plt.close(fig)

def plot_UMAP(scored_gape_frame, plot_dir, basename):
    raw_X = np.stack(scored_gape_frame['features'].values)
    X = StandardScaler().fit_transform(raw_X)

    # UMAP
    umap_model = UMAP(n_components=2)
    umap_out = umap_model.fit_transform(X)

    wanted_event_types = scored_gape_frame.event_type.unique()
    cmap = plt.get_cmap('tab10')
    plot_c_list = [cmap(i) for i in range(len(wanted_event_types))]
    cmap_dict = {x:cmap(i) for i,x in enumerate(wanted_event_types)}

    fig,ax = plt.subplots()
    for i, this_event_type in enumerate(wanted_event_types):
        this_umap_out = umap_out[scored_gape_frame.event_type == this_event_type]
        ax.scatter(this_umap_out[:,0], this_umap_out[:,1], 
                   label = this_event_type.title(), alpha = 0.7,
                   c = plot_c_list[i])
    ax.legend(loc = 'upper left', bbox_to_anchor=(1.3, 1))
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_aspect('equal')
    fig.savefig(os.path.join(plot_dir, f'{basename}_umap_features.png'),
                bbox_inches='tight')
    plt.close(fig)

def plot_NCA(scored_gape_frame, plot_dir, basename):
    raw_X = np.stack(scored_gape_frame['features'].values)
    X = StandardScaler().fit_transform(raw_X)
    wanted_event_types = scored_gape_frame.event_type.unique()
    cmap = plt.get_cmap('tab10')
    plot_c_list = [cmap(i) for i in range(len(wanted_event_types))]
    cmap_dict = {x:cmap(i) for i,x in enumerate(wanted_event_types)}

    nca_model = NCA(n_components=2)
    nca_out = nca_model.fit_transform(X, scored_gape_frame['event_type'].values)

    fig,ax = plt.subplots()
    for i, this_event_type in enumerate(wanted_event_types):
        this_nca_out = nca_out[scored_gape_frame.event_type == this_event_type]
        ax.scatter(this_nca_out[:,0], this_nca_out[:,1], 
                   label = this_event_type.title(), alpha = 0.7,
                   c = plot_c_list[i])
    ax.legend(loc = 'upper left', bbox_to_anchor=(1.3, 1))
    ax.set_xlabel('NCA 1')
    ax.set_ylabel('NCA 2')
    ax.set_aspect('equal')
    fig.savefig(os.path.join(plot_dir, f'{basename}_nca_features.png'),
                bbox_inches='tight')
    plt.close(fig)


def plot_clustermap(scored_gape_frame, plot_dir, basename):
    raw_X = np.stack(scored_gape_frame['features'].values)
    X = StandardScaler().fit_transform(raw_X)
    # Plot heatmap of scaled features using seaborn
    wanted_event_types = scored_gape_frame.event_type.unique()
    cmap = plt.get_cmap('tab10')
    plot_c_list = [cmap(i) for i in range(len(wanted_event_types))]
    cmap_dict = {x:cmap(i) for i,x in enumerate(wanted_event_types)}
    feature_df = pd.DataFrame(data = X, columns = np.arange(X.shape[1])) 
    row_colors = scored_gape_frame['event_type'].map(cmap_dict)
    row_colors.reset_index(drop = True, inplace = True)

    sns.clustermap(feature_df, cmap = 'viridis', 
                   row_colors = row_colors,
                   figsize = (10,10))
    plt.savefig(os.path.join(plot_dir, f'{basename}_feature_clustermap.png'),
                bbox_inches='tight')
    plt.close()
    # plt.show()

def plot_event_counts(event_counts, plot_dir, basename):
    fig, ax = plt.subplots()
    ax.bar(event_counts.index, event_counts.values)
    ax.set_xticklabels(event_counts.index, rotation = 45,
                       horizontalalignment = 'right')
    ax.set_ylabel('Count')
    ax.set_title('Event Counts')
    fig.savefig(os.path.join(plot_dir, f'{basename}_event_counts.svg'),
                bbox_inches='tight')
    plt.close(fig)


def plot_accuracy_histograms_one_vs_all(
        xgb_accuracy_list,
        wanted_event_types,
        JL_accuracy,
        basename,
        plot_dir,
        event_counts=None,
        ):
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(len(wanted_event_types),1, 
                           sharex=True, sharey=True,
                           figsize = (5,len(wanted_event_types)*3))
    for i, this_event_type in enumerate(wanted_event_types):
        this_accuracy = xgb_accuracy_list[i]
        if event_counts is not None:
            this_count = event_counts[this_event_type]
            label = f'{this_event_type.title()} (n={this_count})'
        ax[i].hist(this_accuracy, label=label,
                alpha=0.5, bins = np.linspace(0,1), color = cmap(i))
        ax[i].hist(this_accuracy, 
                bins = np.linspace(0,1), histtype = 'step',
                color = cmap(i))
        if this_event_type == 'gape':
            ax[i].axvline(JL_accuracy, color = 'k', linewidth = 2, alpha = 0.7,
               linestyle = '--', label = 'JL Classifier Gapes')
        ax[i].legend(loc = 'upper left')
        ax[i].set_ylabel('Count')
    ax[-1].set_xlabel('Cross-validated Accuracy')
    fig.suptitle('Classification of Mouth Movements (One vs All)')
    fig.savefig(os.path.join(plot_dir, f'{basename}_classification_accuracy.svg'), 
                             bbox_inches='tight')
    plt.close(fig)

def plot_average_confusion_multiclass(
        wanted_event_types,
        xgb_confusion,
        plot_dir,
        basename,
        ):
    label_len = len(wanted_event_types)
    wanted_xgb_confusion = [x for x in xgb_confusion if x.shape == (label_len,label_len)]
    avg_confusion = np.mean(wanted_xgb_confusion, axis = 0)
    std_confusion = np.std(wanted_xgb_confusion, axis = 0)

    # Normalize over predicted
    norm_avg_confusion = avg_confusion / avg_confusion.sum(axis=-1)[:,None]
    norm_std_confusion = std_confusion / avg_confusion.sum(axis=-1)[:,None]

    plt.matshow(norm_avg_confusion, vmin = 0, vmax = 1)
    plt.xticks(range(label_len), y_labels, rotation = 45,
               horizontalalignment = 'left')
    plt.yticks(range(label_len), y_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Confusion Matrix')
    plt.colorbar(label = 'Fraction of Predictions')
    # Also plot text in each square
    for i in range(label_len):
        for j in range(label_len):
            plt.text(j, i, '{:.2f}'.format(norm_avg_confusion[i,j]) + '\n' + '± {:.2f}'.format(norm_std_confusion[i,j]), 
                     horizontalalignment="center", 
                     verticalalignment="center",
                     color="white" if norm_avg_confusion[i,j] < 0.5 else "black")
    plt.savefig(os.path.join(plot_dir, f'{basename}_average_confusion_matrix.svg'),
                bbox_inches='tight')
    plt.close()

def plot_accuracy_histogram_multiclass(
        xgb_accuracy,
        wanted_event_types,
        plot_dir,
        basename,
        ):

    label_len = len(wanted_event_types)
    fig, ax = plt.subplots(1,1)
    ax.hist(xgb_accuracy, bins = np.linspace(0,1),
            label='All', alpha=0.5)
    ax.set_xlabel('Cross-validated Accuracy')
    ax.set_ylabel('Count')
    ax.set_title('Classification of Mouth Movements (Multiclass)')
    mean_acc = np.mean(xgb_accuracy)
    ax.axvline(mean_acc, color = 'k', linewidth = 2, alpha = 0.7,
               label = 'Mean Accuracy')
    ax.text(mean_acc*1.05, 0.5, f'{mean_acc:.2f}',
            horizontalalignment = 'center',
            verticalalignment = 'center',
            rotation = 90,
            transform = ax.transAxes)
    ax.axvline(1/label_len, color = 'r', linewidth = 2, alpha = 0.7,
               linestyle = '--', label = 'Chance')
    ax.legend(loc = 'upper left')
    fig.savefig(os.path.join(plot_dir, f'{basename}_classification_accuracy_multiclass.svg'),
                bbox_inches='tight')
    plt.close(fig)

##############################
# Classification
##############################

def xgb_one_vs_all_accuracy(
        scored_gape_frame,
        wanted_event_types,
        n_cv,
        ):
    xgb_accuracy_list = []
    xgb_confusion_list = []
    JL_accuracy = None
    JL_confusion = None

    for this_event_type in wanted_event_types:
        #this_event_type = wanted_event_types[0]

        # Train new classifier on data
        # And calculate cross-validation accuracy score
        X = np.stack(scored_gape_frame['features'].values)
        y = np.array(scored_gape_frame['event_type'].values == this_event_type) 
        # Pull out classes for stratified sampling

        if this_event_type == 'gape':
            JL_accuracy = accuracy_score(y, scored_gape_frame.classifier)
            JL_confusion = confusion_matrix(y, scored_gape_frame.classifier,
                                            normalize = 'all')

        xgb_accuracy = []
        xgb_confusion = []
        for i in tqdm(range(n_cv)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=0.3,
                                                                #stratify=classes
                                                                )
            clf = XGBClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cv_accuracy = accuracy_score(y_test, y_pred)
            xgb_accuracy.append(cv_accuracy)

            # cv_confusion = confusion_matrix(y_test, y_pred, normalize = 'all')
            # xgb_confusion.append(cv_confusion)

        xgb_accuracy = np.array(xgb_accuracy)
        # xgb_confusion = np.array(xgb_confusion)

        xgb_accuracy_list.append(xgb_accuracy)
        # xgb_confusion_list.append(xgb_confusion)

    xgb_accuracy_list = np.stack(xgb_accuracy_list)
    # xgb_confusion_list = np.stack(xgb_confusion_list)

    return (
            xgb_accuracy_list, 
            # xgb_confusion_list, 
            JL_accuracy, 
            JL_confusion
            )

def xgb_multiclass_accuracy(
        scored_gape_frame,
        wanted_event_types,
        n_cv,
        ):
    X = np.stack(scored_gape_frame['features'].values)
    y = scored_gape_frame['event_type']
    y_bool = [x in wanted_event_types for x in y] 
    X = X[y_bool]
    y = y[y_bool]
    y_labels = y.astype('category').cat.categories.values
    y = y.astype('category').cat.codes.values

    xgb_accuracy = []
    xgb_confusion = []
    y_test_list = []
    y_pred_list = []
    for i in tqdm(range(n_cv)):

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=0.3,
                                                                )
            clf = XGBClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            cv_accuracy = accuracy_score(y_test, y_pred)
            cv_confusion = confusion_matrix(y_test, y_pred, normalize = 'all')
            xgb_accuracy.append(cv_accuracy)
            xgb_confusion.append(cv_confusion)
            y_test_list.append(y_test)
            y_pred_list.append(y_pred)
        except:
            print('Error in CV')
            continue

    # Drop indices where all classes aren't represented
    class_len = len(y_labels)
    wanted_inds = [x.shape == (class_len,class_len) for x in xgb_confusion]
    wanted_inds = np.where(wanted_inds)[0]

    xgb_accuracy = np.array([xgb_accuracy[i] for i in wanted_inds])
    xgb_confusion = np.array([xgb_confusion[i] for i in wanted_inds])
    y_test_list = np.array([y_test_list[i] for i in wanted_inds])
    y_pred_list = np.array([y_pred_list[i] for i in wanted_inds])

    xgb_accuracy = np.array(xgb_accuracy)
    xgb_confusion = np.array(xgb_confusion)
    y_test_list = np.array(y_test_list)
    y_pred_list = np.array(y_pred_list)

    return (
            xgb_accuracy, 
            xgb_confusion, 
            y_test_list, 
            y_pred_list,
            y_labels,
            )

############################################################
# Load data
############################################################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
all_data_pkl_path = os.path.join(artifact_dir, 'all_data_frame.pkl')
all_data_frame = pd.read_pickle(all_data_pkl_path)

############################################################
# Intra-session analysis 
############################################################
for ind in range(len(all_data_frame)):
    # ind = 0
    basename = all_data_frame.loc[ind,'basename']
    scored_gape_frame = all_data_frame.loc[ind,'gape_frame']
    scored_gape_frame.dropna(inplace=True)
    wanted_event_types = scored_gape_frame.event_type.unique()

    ############################################################
    # Classifier comparison on gapes 
    ############################################################

    #wanted_event_types = ['gape','tongue protrusion','lateral tongue protrusion',]
    # wanted_event_types = ['gape','tongue protrusion',]

    plot_event_examples(
            scored_gape_frame,
            plot_dir,
            basename
            )

    # Also plot dimensionally reduced data
    plot_UMAP(scored_gape_frame, plot_dir, basename)
    plot_NCA(scored_gape_frame, plot_dir, basename)
    plot_clustermap(scored_gape_frame, plot_dir, basename)

    # Get count of each type of event
    event_counts = scored_gape_frame.event_type.value_counts()
    event_counts = event_counts.loc[wanted_event_types]

    plot_event_counts(event_counts, plot_dir, basename)

    ############################################################
    ############################################################
    # One vs All
    n_cv = 500

    (
        one_vs_all_xgb_accuracy_list, 
        # xgb_confusion_list, 
        JL_accuracy, 
        JL_confusion
        )= xgb_one_vs_all_accuracy(
                    scored_gape_frame,
                    wanted_event_types,
                    n_cv,
                    )

    # Histograms of accuracy per event type
    plot_accuracy_histograms_one_vs_all(
            one_vs_all_xgb_accuracy_list,
            wanted_event_types,
            JL_accuracy,
            basename,
            plot_dir,
            event_counts,
            )

    ############################################################
    # Multiclass 

    # Train new classifier on data
    # And calculate cross-validation accuracy score

    (
        multiclass_xgb_accuracy, 
        multiclass_xgb_confusion, 
        y_test_list, 
        y_pred_list,
        y_labels,
        ) = xgb_multiclass_accuracy(
                    scored_gape_frame,
                    wanted_event_types,
                    n_cv,
                    )

    # Average confusion matrix
    # Only take cases with all 3 labels

    plot_average_confusion_multiclass(
            wanted_event_types,
            multiclass_xgb_confusion,
            plot_dir,
            basename,
            )

    # Plot average accuracy
    plot_accuracy_histogram_multiclass(
            multiclass_xgb_accuracy,
            wanted_event_types,
            plot_dir,
            basename,
            )

    # Pool all artifacts and save
    wanted_artifacts = dict(
            basename = basename,
            event_counts = event_counts,
            one_vs_all_accuracy = one_vs_all_xgb_accuracy_list,
            JL_accuracy = JL_accuracy,
            multiclass_accuracy = multiclass_xgb_accuracy,
            multiclass_confusion = multiclass_xgb_confusion,
            )

    dump(
            wanted_artifacts, 
            open(os.path.join(artifact_dir, f'{basename}_wanted_artifacts.pkl'), 'wb')
            )

############################################################
# Inter-session analysis 
############################################################
# Description of variation in data
# 1) Variation in baseline mean+/-std
# 2) Variation in features of labelled events
# 3) Ability to normalize new data (if large variation exists)

##############################
# Get baseline data and statistics
##############################
all_envs = np.stack(all_data_frame.env)
all_basenames = all_data_frame.basename.values
animal_nums = [x.split('_')[0] for x in all_basenames]
session_nums = [x.split('_')[1] for x in all_basenames]

baseline_lims = [0, 2000]
baseline_envs = all_envs[:,baseline_lims[0]:baseline_lims[1]]

mean_baseline = np.mean(baseline_envs, axis = -1)
std_baseline = np.std(baseline_envs, axis = -1)
inds = np.array(list(np.ndindex(mean_baseline.shape)))

baseline_frame = pd.DataFrame(
        data = np.concatenate(
            [
                inds, 
                mean_baseline.flatten()[:,None],
                std_baseline.flatten()[:,None]
                ], 
            axis = -1
            ),
        columns = ['session','taste','trial', 'mean','std']
        )
# Convert ['session','taste','trial'] to int
baseline_frame = baseline_frame.astype({'session':'int','taste':'int','trial':'int'})
baseline_frame['animal'] = [animal_nums[i] for i in baseline_frame['session']]
baseline_frame['session_day'] = [session_nums[i] for i in baseline_frame['session']]
baseline_frame['session_name'] = [animal + '\n' + day for \
        animal,day in zip(baseline_frame['animal'],baseline_frame['session_day'])]

sns.boxplot(
        data = baseline_frame, 
        x = 'session_name', 
        y = 'mean',
        hue = 'taste'
        )
plt.savefig(os.path.join(plot_dir, 'baseline_mean_boxplot.png'),
            bbox_inches='tight')
plt.close()

sns.boxplot(
        data = baseline_frame, 
        x = 'session_name', 
        y = 'std',
        hue = 'taste'
        )
plt.savefig(os.path.join(plot_dir, 'baseline_std_boxplot.png'),
            bbox_inches='tight')
plt.close()

# Correlation between mean and std
sns.scatterplot(
        data = baseline_frame,
        x = 'mean',
        y = 'std',
        hue = 'taste'
        )
plt.savefig(os.path.join(plot_dir, 'baseline_mean_std_scatter.png'),
            bbox_inches='tight')
plt.close()

##############################
# Similar analysis for labelled events 
##############################
all_gape_frames = all_data_frame.gape_frame

# Update gape frames to include animal_num and session_nums
for ind in range(len(all_gape_frames)):
    all_gape_frames[ind]['animal_num'] = animal_nums[ind]
    all_gape_frames[ind]['session_num'] = session_nums[ind]

fin_gape_frame = pd.concat(all_gape_frames.values, axis = 0)
fin_gape_frame['session_name'] = [animal + '\n' + day for \
        animal,day in zip(fin_gape_frame['animal_num'],fin_gape_frame['session_num'])]

wanted_cols = ['taste','trial','features','event_type','session_name']
fin_gape_frame = fin_gape_frame[wanted_cols]
fin_gape_frame['feature_ind'] = [np.arange(len(fin_gape_frame.features.iloc[0])) \
        for i in range(len(fin_gape_frame))]

# Explode features
fin_gape_frame = fin_gape_frame.explode(['features','feature_ind'])
fin_gape_frame.reset_index(drop = True, inplace = True)

feature_names = [
    'duration',
    'amplitude_rel',
    'amplitude_abs',
    'left_interval',
    'right_interval',
    'pca_1',
    'pca_2',
    'pca_3',
    'max_freq',
]

fin_gape_frame['feature_name'] = [feature_names[i] for i in fin_gape_frame['feature_ind']]

sns.catplot(
        data = fin_gape_frame, 
        x = 'session_name',
        y = 'features',
        hue = 'taste',
        row = 'feature_name',
        col = 'event_type',
        kind = 'box',
        sharey = False,
        )
plt.savefig(os.path.join(plot_dir, 'feature_boxplot.png'),
            bbox_inches='tight')
plt.close()

# Run ANOVA across features for every mouth movement type
# to see where there are significant differences between sessions
feature_event_groups = list(fin_gape_frame.groupby(['feature_name','event_type']))
feature_event_inds = [x[0] for x in feature_event_groups]
feature_event_data = [x[1] for x in feature_event_groups]

anova_results = []
for this_feature_event in tqdm(feature_event_data):
    this_feature_event = this_feature_event.astype({'features':'float'})
    this_anova = pg.anova(data = this_feature_event, dv = 'features', between = 'session_name')
    anova_results.append(this_anova)

feature_event_diff_frame = pd.DataFrame(
        data = feature_event_inds,
        columns = ['feature_name','event_type']
        )
punc_list = []
np2_list = []
for this_anova in anova_results:
    try:
        punc_list.append(this_anova['p-unc'].values[0])
        np2_list.append(this_anova['np2'].values[0])
    except:
        punc_list.append(np.nan)
        np2_list.append(np.nan)
feature_event_diff_frame['punc'] = punc_list
feature_event_diff_frame['np2'] = np2_list

# Plot both punc and np2 as heatmaps
punc_pivot = feature_event_diff_frame.pivot(
            columns = 'feature_name',
            index = 'event_type',
            values = 'punc')
punc_pivot = np.round(punc_pivot, 3)
fig, ax = plt.subplots(1,1)
sns.heatmap(
        data = punc_pivot,
        ax = ax,
        annot = True,
        cmap = 'viridis',
        )
plt.savefig(os.path.join(plot_dir, 'feature_event_punc_heatmap.png'),
            bbox_inches='tight')
plt.close()

np2_pivot = feature_event_diff_frame.pivot(
        columns = 'feature_name',
        index = 'event_type',
        values = 'np2')
np2_pivot = np.round(np2_pivot, 3)
fig, ax = plt.subplots(1,1)
sns.heatmap(
        data = np2_pivot,
        ax = ax,
        annot = True,
        cmap = 'viridis',
        )
plt.savefig(os.path.join(plot_dir, 'feature_event_np2_heatmap.png'),
            bbox_inches='tight')
plt.close()



# Comparison of inter-session (but intra-animal) prediction
# and inter-animal comparisons (using leave-one-animal-out)