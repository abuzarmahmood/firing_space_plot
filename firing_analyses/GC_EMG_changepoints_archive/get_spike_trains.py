"""
Get spike trains from all relevant datasets and save as artifact
"""
import sys
import os
sys.path.append(os.path.expanduser('~/Desktop/blech_clust'))
from utils.ephys_data import ephys_data
import pandas as pd
from tqdm import tqdm
from pprint import pprint as pp
import matplotlib.pyplot as plt
from scipy.stats import zscore


base_plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/plots'
change_plot_dir = os.path.join(base_plot_dir, 'changepoint_plots')
artifact_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/artifacts'

wanted_frame = pd.read_pickle(os.path.join(artifact_dir, 'wanted_changepoint_frame.pkl'))
# firing_frame = pd.read_pickle(os.path.join(artifact_dir, 'firing_frame.pkl'))
blacklist_basenames = [
       'KM50_5tastes_EMG_210911_104510_copy',
       'KM50_5tastes_EMG_210913_100710_copy',
       'KM29_dual_4tastes_emg_200620_165523_copy',
       ]
# make sure all blacklisted basenames are in the frame
assert all([x in basename_list for x in blacklist_basenames])

wanted_frame = wanted_frame.loc[~wanted_frame['data.basename'].isin(blacklist_basenames)]
basename_list = wanted_frame['data.basename'].unique()

# Get data directories
data_directories = wanted_frame['data.data_dir'].unique()

# from importlib import reload
# reload(ephys_data)

pal_array_list = []
pal_df_list = []
for data_ind, data_dir in tqdm(enumerate(data_directories)):
    this_dat = ephys_data.ephys_data(data_dir)
    # this_dat.get_firing_rates()
    this_dat.calc_palatability()
    pal_df_list.append(this_dat.pal_df)
    pal_array_list.append(this_dat.pal_array)

for this_df, this_dir in zip(pal_df_list, data_directories):
    basename = os.path.basename(this_dir)
    print(basename)
    print(this_df)
    print()

time_lims = [2000, 4000]
time_vec = np.linspace(0, 7000, cat_pal_array.shape[-1])
lim_inds = np.logical_and(
        time_vec >= time_lims[0],
        time_vec < time_lims[1]
        )
cut_time_vec = time_vec[lim_inds]
cat_pal_array = np.concatenate(pal_array_list)
zscore_cat_pal_array = zscore(cat_pal_array, axis=-1)
max_pal_ind = np.argmax(cat_pal_array[:, lim_inds], axis=-1)
max_pal_time = [cut_time_vec[i] for i in max_pal_ind] 

fig, ax = plt.subplots(3,2, sharex=True)
ax[0,0].hist(max_pal_time, bins = 20)
ax[1,0].plot(time_vec, np.mean(cat_pal_array,axis=0))
ax[1,1].plot(time_vec, np.mean(zscore_cat_pal_array,axis=0))
ax[-1,0].pcolormesh(
        time_vec,
        np.arange(len(cat_pal_array)),
        cat_pal_array,
        )
ax[-1,1].pcolormesh(
        time_vec,
        np.arange(len(cat_pal_array)),
        zscore_cat_pal_array,
        )
for this_ax in ax.flatten():
    this_ax.set_xlim(time_lims)
ax[0,0].set_title('Max palatability time')
ax[1,0].set_title('Mean raw palatability')
ax[1,1].set_title('Mean zscored palatability')
ax[-1,0].set_title('Palatability')
ax[-1,1].set_title('Zscored Palatability')
plt.tight_layout()
fig.savefig(os.path.join(base_plot_dir, 'palatability_correlation_firing_rates.png'))
plt.close(fig)
# plt.show()

