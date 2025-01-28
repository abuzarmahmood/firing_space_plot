"""
Elbo pickles are giving the following error:
    ValueError: unsupported pickle protocol: 5
This is likely because pickles were created in blech_clust environment (python 3.8.13)
and are being read in pytau_env (python 3.6.9).
To avoid reading pickles, convert them to csv files.
"""
## Import modules
import sys
import pylab as plt
from tqdm import tqdm
import os
from pprint import pprint as pp
import pandas as pd

##############################
# Data Dirs
data_dir_file = '/media/bigdata/firing_space_plot/firing_analyses/GC_EMG_changepoints/data_dir_list.txt'
with open(data_dir_file, 'r') as f:
    data_dir_list = f.read().splitlines()

pp(data_dir_list)

error_list = []
for this_dir in tqdm(data_dir_list):
    # Get trial-changepoints
    artifact_dir = os.path.join(this_dir, 'QA_output', 'artifacts')
    # Get all files
    all_files = sorted(os.listdir(artifact_dir))
    # Look for pattern taste_* in each name
    taste_ind = [int(x.split('taste_')[1][0]) for x in all_files if 'taste' in x]
    # Load pkl files
    trial_change_dat = [pd.read_pickle(os.path.join(artifact_dir, x)) for x in all_files if 'taste' in x]
    drop_cols = ['ppc']
    trial_change_dat = [x.drop(columns=drop_cols) for x in trial_change_dat]
    # Save to csv
    for i, this_df in enumerate(trial_change_dat):
        this_df.to_csv(os.path.join(artifact_dir, all_files[i].replace('.pkl','.csv')))
    print(f"Saved {len(trial_change_dat)} csv files in {artifact_dir}")

