"""
UMAP cannot be used for python < 3.8
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from tqdm import tqdm

import sys
import os

base_plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/plots'
change_plot_dir = os.path.join(base_plot_dir, 'changepoint_plots')

artifact_dir = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/artifacts'
snips_frame_path = os.path.join(artifact_dir, 'transition_snips.pkl')
snips_frame = pd.read_pickle(snips_frame_path)

for i, this_row in tqdm(snips_frame.iterrows()):
    
    # i = 0
    # this_row = snips_frame.iloc[i]
    basename = this_row['basename']
    taste_num = this_row['taste_num']
    snips = this_row['snips']

    bin_width = 25
    n_trials, n_nrns, n_time, n_trans = snips.shape 
    binned_snips = np.reshape(
            snips, 
            (n_trials, n_nrns, -1, bin_width, n_trans)).sum(axis=-2) 

    smooth_binned_snips = savgol_filter(binned_snips, 5, 3, axis=2)
    mean_smooth_binned_snips = np.mean(smooth_binned_snips, axis=0)
    scaled_snips = [StandardScaler().fit_transform(snip) for snip in mean_smooth_binned_snips.T] 

    time_vec = np.arange(-n_time/2, n_time/2, bin_width)
    
    # UMAP
    umap_snips = []
    for snip in tqdm(scaled_snips):
        this_umap = UMAP(n_components=1)
        this_snip_umap = this_umap.fit_transform(snip)
        umap_snips.append(this_snip_umap)

    midline = binned_snips.shape[2]/2
    fig, ax = plt.subplots(1 ,3, figsize = (7,2))
    for i, snip in enumerate(umap_snips):
        ax[i].plot(time_vec, snip, label=f'Trans {i}')
        ax[i].axvline(0, color='r', linestyle='--')
        ax[i].set_title(f'Trans {i}')
        ax[i].set_xlabel('Time (ms)')
        ax[i].set_ylabel('UMAP')
    fig.suptitle(f'{basename} Taste {taste_num}')
    plt.tight_layout()
    fig.savefig(os.path.join(change_plot_dir, f'{basename}_taste_{taste_num}_umap.png'))
    # plt.show()
