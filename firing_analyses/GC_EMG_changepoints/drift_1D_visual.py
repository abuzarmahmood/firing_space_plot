"""
Visualize neural drift in 1D

Reduce single trial to 1D using PCA
"""

import sys
# ephys_data_path = '/media/bigdata/firing_space_plot/ephys_data'
blech_clust_path = '/home/abuzarmahmood/Desktop/blech_clust'
sys.path.append(blech_clust_path)
# sys.path.append(ephys_data_path)
# from ephys_data import ephys_data
from utils.ephys_data.ephys_data import ephys_data
from matplotlib import pyplot as plt
from pprint import pprint as pp
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import zscore, anderson, shapiro

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/plots'

# Data Dirs
data_dir_file = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/data_dir_list.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

rates_list = []
fin_dir_list = []
for this_dir in tqdm(data_dir_list):
    try:
        this_data = ephys_data(this_dir)
        this_data.firing_rate_params = this_data.default_firing_params
        this_data.get_spikes()
        this_data.get_firing_rates()
        # rates_list.append(this_data.firing_array)
        rates_list.append(this_data.firing_list)
        fin_dir_list.append(this_dir)
    except:
        print(f"Error in {this_dir}")
        continue

fin_basename_list = [x.split('/')[-1] for x in fin_dir_list]

# If rates are uneven, chop to be even
min_len = [np.min([len(x) for x in thi_session]) for thi_session in rates_list]
rates_list = [[x[:min_len[ind]] for x in thi_session] for ind, thi_session in enumerate(rates_list)]
rates_list = [np.stack(x) for x in rates_list]

# Treat each neuron and taste separately
pca_data_list = []
for session_ind, this_session in enumerate(rates_list):
    # this_session = this_session.swapaxes(1,2)
    this_session_long = this_session.reshape(*this_session.shape[:2],-1)
    session_pca_list = []
    for taste_ind, this_taste in enumerate(this_session_long):
        pca = PCA(n_components=1)
        pca_data = pca.fit_transform(this_taste)
        session_pca_list.append(pca_data)
    pca_data_list.append(session_pca_list)

# Plot PCA data
zscore_pca_data_list = [
        [zscore(x) for x in session] for session in pca_data_list
        ]

fig, ax = plt.subplots(len(pca_data_list),2,sharex=True,
                       figsize=(10,10))
for session_ind, this_session in enumerate(zscore_pca_data_list):
    for taste_ind, this_taste in enumerate(this_session):
        ax[session_ind,0].plot(this_taste,label=f'Taste {taste_ind}')
    session_array = np.squeeze(np.stack(this_session))
    mean_array = np.mean(session_array,axis=0)
    std_array = np.std(session_array,axis=0)
    # Test mean for normality
    shapiro_out = shapiro(mean_array)
    shapiro_p = shapiro_out[1]
    ax[session_ind,1].plot(mean_array,label='Mean')
    ax[session_ind,1].fill_between(np.arange(mean_array.shape[0]),
            mean_array-std_array,mean_array+std_array,alpha=0.3,
            label='Std')
    ax[session_ind, 0].set_title(f'{fin_basename_list[session_ind]}')
    p_str = f'Shapiro p for mean: {shapiro_p:.3f}'
    ax[session_ind, 1].set_title(p_str + '\n' + 'p < 0.05 is non-normal')
    ax[session_ind, 0].legend()
    ax[session_ind, 1].legend()
# plt.show()
plt.tight_layout()
plt.savefig(f'{plot_dir}/pca_drift.png')
plt.close()
