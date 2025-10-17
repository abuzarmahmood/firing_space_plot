"""
Preliminary analysis to compare units across different experiments

1) Counts of units per region
2) Similarity of GC vs LH units within a single dataset
"""
import sys
ephys_data_path = '/media/bigdata/firing_space_plot/ephys_data'
sys.path.append(ephys_data_path)
from ephys_data import ephys_data
from matplotlib import pyplot as plt
from pprint import pprint as pp
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/plots'

##############################
# Data Dirs
data_dir_file = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/data_dir_list.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

region_unit_list = []
dir_list = []
for this_dir in tqdm(data_dir_list):
    try:
        # this_dir = data_dir_list[0]
        this_data = ephys_data(this_dir)
        this_data.get_region_units()
        region_unit_counts = [len(x) for x in this_data.region_units]
        region_names = [x for x in this_data.region_names if 'emg' not in x]
        this_region_count_dict = dict(zip(region_names,region_unit_counts))
        region_unit_list.append(this_region_count_dict)
        dir_list.append(this_dir)
    except:
        print(f"Error in {this_dir}")
        continue

# Convert to dataframe
region_unit_df = pd.DataFrame(region_unit_list,index=dir_list)
region_unit_df = region_unit_df.fillna(0)
region_unit_df.reset_index(inplace=True)
region_unit_df.rename(columns={'index':'data_dir'},inplace=True)
region_unit_df['basename'] = region_unit_df['data_dir'].apply(lambda x: x.split('/')[-1])
region_unit_df.drop(columns=['data_dir'],inplace=True)
region_unit_df.set_index('basename',inplace=True)
region_unit_df.sort_values(by=['basename'],inplace=True)
# Convert gc and lh to int
region_unit_df['gc'] = region_unit_df['gc'].astype(int)
region_unit_df['lh'] = region_unit_df['lh'].astype(int)

# Plot stacked barplot of unit counts
plt.figure()
# sns.barplot(data=region_unit_df,x='basename',y='unit_count',hue='region', stacked=True)
region_unit_df.plot.bar(stacked=True)
plt.xticks(rotation=45, ha='right')
plt.title('Unit Counts per Region')
plt.tight_layout()
plt.savefig(f'{plot_dir}/unit_counts_per_region.png')
plt.close()

############################################################

# Perform PLS regression on unit activity between GC and LH
# Compare single-trial firing rates between GC and LH to
# 1) trial-shuffled firing rates between GC and LH
# 2) trial-matched and trial-shuffled firing rates between intra-GC populations

# Get firing rates for all units

region_rates = {}
error_list = []
for this_dir in tqdm(data_dir_list):
    try:
        this_data = ephys_data(this_dir)
        this_data.firing_rate_params = this_data.default_firing_params
        this_data.get_spikes()
        this_data.get_region_units()
        if len(this_data.region_units) < 2:
            print(f"Skipping {this_dir}")
            continue
        region_rates[this_dir] = {}
        for region in this_data.region_names:
            if 'emg' in region:
                continue
            this_firing = this_data.get_region_firing(region)
            region_rates[this_dir][region] = this_firing
    except Exception as e:
        print(e)
        error_list.append(this_dir)
        print(f"Error in {this_dir}")
        continue

# If dict is empty, remove
region_rates = {k:v for k,v in region_rates.items() if v}

data_dirs = list(region_rates.keys())
basenames = [x.split('/')[-1] for x in data_dirs]
region_rates = list(region_rates.values())

time_vec = np.linspace(0, 7000, region_rates[0]['gc'].shape[-1]) 
wanted_inds = np.where((time_vec > 2000) & (time_vec < 4000))[0]
wanted_region_rates = []
for this_dict in region_rates:
    new_dict = {}
    for key, val in this_dict.items():
        new_dict[key] = val[...,wanted_inds]
    wanted_region_rates.append(new_dict)

# Perform PLS
r2_list = []
region_keys = ['gc','lh']
for i, this_session in enumerate(tqdm(wanted_region_rates)):
    gc_rates = this_session['gc']
    lh_rates = this_session['lh']
    gc_rates = gc_rates.swapaxes(0,1)
    lh_rates = lh_rates.swapaxes(0,1)
    gc_rates_long = gc_rates.reshape(gc_rates.shape[0],-1)
    lh_rates_long = lh_rates.reshape(lh_rates.shape[0],-1)
    # Scale each neuron
    gc_rates_long_scaled = StandardScaler().fit_transform(gc_rates_long.T)
    lh_rates_long_scaled = StandardScaler().fit_transform(lh_rates_long.T)

    X = gc_rates_long_scaled
    Y = lh_rates_long_scaled

    # Cut X and Y to same limits
    max_lim = 3
    X_cut = X.copy()
    Y_cut = Y.copy()
    X_cut[X_cut > max_lim] = max_lim
    X_cut[X_cut < -max_lim] = -max_lim
    Y_cut[Y_cut > max_lim] = max_lim
    Y_cut[Y_cut < -max_lim] = -max_lim

    ##############################
    # Perform MLP regression

    n_repeat = 10

    this_r2_list = []
    pred_list = []

    test_size = 0.3
    for _ in range(n_repeat):

        train_X, test_X, train_Y, test_Y = train_test_split(X_cut,Y_cut, test_size=test_size)

        mlp = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000,
                           verbose=False, tol=1e-4, n_iter_no_change=10)
        mlp.fit(train_X,train_Y)
        y_pred_mlp = mlp.predict(X_cut)
        r2_mlp = mlp.score(test_X,test_Y)
        this_r2_list.append(r2_mlp)
        pred_list.append(y_pred_mlp)

    r2_list.append(this_r2_list)

    mean_r2 = np.mean(this_r2_list)
    sd_r2 = np.std(this_r2_list)
    best_pred = pred_list[np.argmax(this_r2_list)]

    fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
    ax[0].imshow(best_pred,aspect='auto', interpolation='nearest',
                 vmin=-max_lim, vmax=max_lim)
    ax[0].set_title('Predicted LH')
    ax[1].imshow(Y_cut,aspect='auto', interpolation='nearest',
                 vmin=-max_lim, vmax=max_lim)
    ax[1].set_title('LH')
    ax[2].imshow(X_cut,aspect='auto', interpolation='nearest',
                 vmin=-max_lim, vmax=max_lim)
    ax[2].set_title('GC')
    r2_string = f'R^2: {mean_r2:.2f} +/- {sd_r2:.2f}'
    fig.suptitle(basenames[i] + '\n' + r2_string + \
            '\n' + f'R2 on held-out data, {test_size} test size')
    # plt.show()
    plt.tight_layout()
    fig.savefig(f'{plot_dir}/mlp_regression_{basenames[i]}.png')
    plt.close()

# Max boxplot of R^2 values
# Convert to dataframe
r2_df = pd.DataFrame(r2_list, index=basenames)
r2_df.reset_index(inplace=True)
r2_df.rename(columns={'index':'basename'},inplace=True)
r2_df.set_index('basename',inplace=True)
r2_df = r2_df.stack().reset_index()
r2_df.rename(columns={'level_1':'repeat',0:'r2'},inplace=True)
r2_df['r2'] = r2_df['r2'].astype(float)

plt.figure(figsize=(5,10))
g = sns.boxplot(data=r2_df,x='basename',y='r2')
plt.xticks(rotation=45, ha='right')
f_string = f'{n_repeat} repeats, {test_size} test size'
plt.title('MLP Regression Held Out R^2 \n GC vs LH Firing Rates \n' + f_string)
plt.tight_layout()
plt.savefig(f'{plot_dir}/mlp_regression_r2.png')
plt.close()

