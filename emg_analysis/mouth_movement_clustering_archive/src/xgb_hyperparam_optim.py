"""
Optimize hyperparameters for XGBoost model using skopt.
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from time import time
import xgboost as xgb
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
import json

from skopt import Optimizer, dump, load
from skopt.plots import plot_convergence, plot_objective
from skopt.space import Real, Integer
from joblib import Parallel, delayed

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
code_dir = os.path.join(base_dir, 'src')
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')

xgb_optim_artifacts_dir = os.path.join(artifact_dir, 'xgb_optim_artifacts')
if not os.path.exists(xgb_optim_artifacts_dir):
    os.makedirs(xgb_optim_artifacts_dir)

##############################
# XGB Predictions 
##############################

# merge_gap_pal can only be loaded in base env

# merge_gape_pal_path = os.path.join(artifact_dir, 'merge_gape_pal.pkl')
# merge_gape_pal = pd.read_pickle(merge_gape_pal_path)
# 
# feature_names_path = os.path.join(artifact_dir, 'merge_gape_pal_feature_names.npy')
# feature_names = np.load(feature_names_path)
# 
# scored_df = merge_gape_pal[merge_gape_pal.scored == True]
# 
# # Correct event_types
# types_to_drop = ['to discuss', 'other', 'unknown mouth movement','out of view']
# scored_df = scored_df[~scored_df.event_type.isin(types_to_drop)]
# 
# # Remap event_types
# event_type_map = {
#         'mouth movements' : 'mouth or tongue movement',
#         'tongue protrusion' : 'mouth or tongue movement',
#         'mouth or tongue movement' : 'mouth or tongue movement',
#         'lateral tongue movement' : 'lateral tongue protrusion',
#         'lateral tongue protrusion' : 'lateral tongue protrusion',
#         'gape' : 'gape',
#         'no movement' : 'no movement',
#         }
# 
# scored_df['event_type'] = scored_df['event_type'].map(event_type_map)
# scored_df['event_codes'] = scored_df['event_type'].astype('category').cat.codes
# scored_df['is_gape'] = (scored_df['event_type'] == 'gape')*1
# 
# scored_df.dropna(subset=['event_type'], inplace=True)
# 
# # plt.imshow(np.stack(scored_df.features.values), aspect='auto', interpolation='none')
# # plt.xticks(np.arange(len(feature_names)), feature_names, rotation=90)
# # plt.tight_layout()
# # plt.show()
# 
# # Generate leave-one-animal-out predictions
# # Convert 'lateral tongue protrustion' and 'no movement' to 'other'
# scored_df['event_type'] = scored_df['event_type'].replace(
#         ['lateral tongue protrusion','no movement'],
#         'other'
#         )
# 
# scored_df['event_type'] = scored_df['event_type'].replace(
#         ['mouth or tongue movement'],
#         'MTMs'
#         )
# 
# event_code_dict = {
#         'gape' : 1,
#         'MTMs' : 2,
#         'other' : 0,
#         }
# 
# scored_df['event_codes'] = scored_df['event_type'].map(event_code_dict)
# 
# X_array = np.stack(scored_df.features.values)
# y_array = scored_df.event_codes.values
# animals_array = scored_df.animal_num.values
# 
# # Output to artifact_dir
# 
# np.save(os.path.join(xgb_optim_artifacts_dir, 'X_array.npy'), X_array)
# np.save(os.path.join(xgb_optim_artifacts_dir, 'y_array.npy'), y_array)
# np.save(os.path.join(xgb_optim_artifacts_dir, 'animals_array.npy'), animals_array)
# 
# # Train on scored data but predict on all data
# unique_animals = scored_df.animal_num.unique()

X_array = np.load(os.path.join(xgb_optim_artifacts_dir, 'X_array.npy'))
y_array = np.load(os.path.join(xgb_optim_artifacts_dir, 'y_array.npy'))
animals_array = np.load(os.path.join(xgb_optim_artifacts_dir, 'animals_array.npy'), allow_pickle=True)
unique_animals = np.unique(animals_array)

############################################################
search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    }

dimensions = [
        Real(0.001, 1.0, name='learning_rate', prior='log-uniform'),
        Integer(1, 10, name='min_child_weight'),
        Integer(1, 100, name='max_depth'),
        Integer(0, 20, name='max_delta_step'), 
        Real(0.01, 1.0, name='subsample', prior='uniform'),
        Real(0.01, 1.0, name='colsample_bytree', prior='uniform'),
        Real(0.01, 1.0, name='colsample_bylevel', prior='uniform'),
        Real(1e-9, 1000, name='reg_lambda', prior='log-uniform'),
        Real(1e-9, 1.0, name='reg_alpha', prior='log-uniform'),
        Real(1e-9, 0.5, name='gamma', prior='log-uniform'),
        Integer(50, 100, name='n_estimators'),
        # Real(1e-6, 500, name='scale_pos_weight', prior='log-uniform')
        ]

# Convert search dimensions to dataframe
dim_names = [dim.name for dim in dimensions]
dim_lows = [dim.low for dim in dimensions]
dim_ups = [dim.high for dim in dimensions]
dim_priors = [dim.prior for dim in dimensions]

dim_df = pd.DataFrame(
        columns=['name', 'low', 'high', 'prior'],
        data = list(zip(dim_names, dim_lows, dim_ups, dim_priors))
        )
# make low and high floats
dim_df['low'] = dim_df['low'].astype(float)
dim_df['high'] = dim_df['high'].astype(float)

# Print to artifact_dir
dim_df.to_csv(os.path.join(xgb_optim_artifacts_dir, 'xgb_optim_search_dimensions.csv'))


def run_xgb_cv(params_kwargs, X_array, y_array, animals_array):
    cv_accs = []
    for i, this_animal in enumerate(unique_animals):

        test_idx = animals_array == this_animal
        train_idx = ~test_idx

        X_train = X_array[train_idx]
        y_train = y_array[train_idx]
        X_test = X_array[test_idx]
        y_test = y_array[test_idx]

        clf = xgb.XGBClassifier(
                **params_kwargs, 
                use_label_encoder=False,
                n_jobs=1,
                eval_metric='mlogloss'
                )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_accs.append(acc)

    mean_acc = np.mean(cv_accs)
    return mean_acc

def run_xgb_cv_wrapper(params):
    return run_xgb_cv(params, X_array, y_array, animals_array)


###############
# Run optimizer
n_optim_repeats = 20

results_list = []
optimizer_list = []

for repeat_ind in range(len(results_list), n_optim_repeats):

    optimizer = Optimizer(
            dimensions=dimensions,
            base_estimator='GBRT',
            n_initial_points=10,
            acq_func='EI',
            acq_optimizer='auto',
            )

    n_iter = 100
    n_parallel = 4

    for i in trange(n_iter):
        params = optimizer.ask(n_points=n_parallel)
        params_kwargs = [{key: val for key, val in zip([dim.name for dim in dimensions], this_params)} \
                for this_params in params]

        # mean_acc = run_xgb_cv_wrapper(params_kwargs)
        mean_acc = Parallel(n_jobs=n_parallel)\
                (delayed(run_xgb_cv_wrapper)(this_params) for this_params in params_kwargs)
        mean_acc = [-this_acc for this_acc in mean_acc]

        result = optimizer.tell(params, mean_acc)

    results_list.append(result)
    optimizer_list.append(optimizer)

    # Save optimizer results
    dump(
            optimizer, 
            os.path.join(xgb_optim_artifacts_dir, f'xgb_optimization_results_{repeat_ind}.pkl')
            )

# Save best hyperparameters
best_result_ind = np.argmax([np.max(-result.func_vals) for result in results_list])
best_result = results_list[best_result_ind]
best_params = best_result.x
param_names = [dim.name for dim in dimensions]
best_params_kwargs = {key: val for key, val in zip(param_names, best_params)}
params_df = pd.DataFrame(
        columns=['name', 'value'],
        data=list(best_params_kwargs.items())
        )

def return_type(val):
    if type(val) == float:
        return 'float'
    elif type(val) in [int, np.int64]:
        return 'int'
    else:
        return 'other'

params_df['dtype'] = [return_type(val) for val in best_params]
params_df.to_csv(os.path.join(xgb_optim_artifacts_dir, 'best_xgb_hyperparams.csv'),
                 index=False)

# Plot convergence
def plot_convergence_custom(result, ax = None, alpha = 0.3, linewidth=5):
    if ax is None:
        fig, ax = plt.subplots()
    y_vals = -np.array(result.func_vals)
    y_vals_max = np.maximum.accumulate(y_vals)
    # ax.plot(y_vals_max, '-o', alpha=alpha)
    ax.plot(y_vals_max, alpha=alpha, linewidth=linewidth)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('XGB Optimization Convergence')
    return ax

# plot_convergence_custom(result)
# plt.show()
fig, ax = plt.subplots(1,2, sharey=True, figsize=(10,5))
for result in results_list:
    plot_convergence_custom(result, ax=ax[0], alpha=1, linewidth=2)
ax[0].set_title('XGB Optimization Convergence')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Accuracy')
max_y = [np.max(-result.func_vals) for result in results_list]
ax[1].hist(max_y, bins=10, orientation='horizontal')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'xgb_optimization_convergence.png'),
            bbox_inches='tight')
plt.close(fig)

##############################

importance_df_list = []

for ind, result in enumerate(results_list):
    results_df = pd.DataFrame(result.x_iters, columns=[dim.name for dim in dimensions])

    # Estimate hyperparameter importance
    clf = xgb.XGBRegressor()
    clf.fit(results_df, -result.func_vals)
    importances = clf.feature_importances_

    importance_df = pd.DataFrame(
            columns=[dim.name for dim in dimensions],
            data=[importances]
            )

    importance_df = pd.melt(importance_df)
    importance_df['run'] = ind
    importance_df_list.append(importance_df)

fin_importance_df = pd.concat(importance_df_list)
fin_importance_df.reset_index(inplace=True)

order = fin_importance_df.groupby('variable').mean().sort_values('value', ascending=False).index
g = sns.boxplot(data=fin_importance_df, x='variable', y='value',
                order=order)
sns.stripplot(data=fin_importance_df, x='variable', y='value', color='black', alpha=0.5,
              ax=g, order=order)
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_xlabel('Hyperparameter')
g.set_ylabel('Importance')
g.set_title('XGB Hyperparameter Importance')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'xgb_optimization_importance.png'),
            bbox_inches='tight')
plt.close(fig)


# Plot objective
best_result_ind = np.argmax(max_y)
best_result = results_list[best_result_ind]
plot_objective(best_result, sample_source='result', n_points=10)
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'xgb_optimization_objective.png'),
            bbox_inches='tight')
plt.close(fig)

##############################
# Plot final distributions of hyperparameters

final_params_list = [result.x for result in results_list]
final_params_df = pd.DataFrame(final_params_list, columns=[dim.name for dim in dimensions])
final_params_df_melt = pd.melt(final_params_df)

sns.pairplot(final_params_df)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'xgb_optimization_final_params.png'),
            bbox_inches='tight')
plt.close()


