"""
Match neurons from old and new sorted files

Use unit descriptor table,
If an electrode has a single neuron in both sets, print out:
    - session
    - electrode number
    - old waveform count
    - new waveform count
"""

import os
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import pandas as pd
import numpy as np
from pprint import pprint as pp

############################################################
# Old unit descriptors
############################################################
file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]

# Get unit descriptor table for each session
table_cols = ['unit_number','electrode_number','fast_spiking','regular_spiking','single_unit','waveform_count']
old_unit_list = []
for this_dir in file_list:
    try:
        print(f'Working on {this_dir}')
        this_data = ephys_data(this_dir)
        this_data.get_unit_descriptors()
        unit_table = this_data.unit_descriptors
        unit_dict_list = [dict(zip(table_cols, x)) for x in unit_table]
        unit_df = pd.DataFrame(unit_table)
        if 'unit_number' not in unit_df.columns:
            unit_df['unit_number'] = np.arange(unit_df.shape[0])
        if 'waveform_count' not in unit_df.columns:
            unit_df['waveform_count'] = 0
        basename = os.path.basename(this_dir)
        unit_df['session'] = basename
        old_unit_list.append(unit_df)
    except:
        print(f'Failed on {this_dir}')
        continue

old_unit_frame = pd.concat(old_unit_list)

# Convert "table_cols" to int
old_unit_frame[table_cols] = old_unit_frame[table_cols].astype(int)

############################################################
# New unit descriptors 
############################################################

new_dir_file_path = '/media/storage/for_transfer/bla_gc/data_dir_list.txt'
new_dir_list = [x.strip() for x in open(new_dir_file_path,'r').readlines()]

# Get unit descriptor table for each session
table_cols = ['unit_number','electrode_number','fast_spiking','regular_spiking','single_unit','waveform_count']
new_unit_list = []
for this_dir in new_dir_list:
    try:
        print(f'Working on {this_dir}')
        this_data = ephys_data(this_dir)
        this_data.get_unit_descriptors()
        unit_table = this_data.unit_descriptors
        unit_dict_list = [dict(zip(table_cols, x)) for x in unit_table]
        unit_df = pd.DataFrame(unit_table)
        basename = os.path.basename(this_dir)
        unit_df['session'] = basename
        new_unit_list.append(unit_df)
    except:
        print(f'Failed on {this_dir}')
        continue

new_unit_frame = pd.concat(new_unit_list)

# Convert "table_cols" to int
new_unit_frame[table_cols] = new_unit_frame[table_cols].astype(int)

############################################################
# Match neurons
############################################################

# For both dfs if any electrode has more than one neuron, drop the electrode
old_units_single = \
        old_unit_frame.groupby(['session','electrode_number']).filter(lambda x: x.shape[0] == 1)

new_units_single = \
        new_unit_frame.groupby(['session','electrode_number']).filter(lambda x: x.shape[0] == 1)

wanted_cols = ['session','unit_number','electrode_number','waveform_count']
old_units_single = old_units_single[wanted_cols]
new_units_single = new_units_single[wanted_cols]

# Merge on session and electrode number
merged_units = pd.merge(
        old_units_single, 
        new_units_single, 
        on = ['session','electrode_number'], 
        suffixes = ('_old','_new')
        )

artifacts_dir = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'
merged_units.to_csv(os.path.join(artifacts_dir,'single_neuron_match.csv'))

# Plot scatter plot of old and new waveform counts
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(2,1,figsize = (7,7))
ax[0].set(xscale = 'log', yscale = 'log', aspect = 'equal')
sns.scatterplot(
        data = merged_units.loc[merged_units['waveform_count_old'] > 0], 
        x = 'waveform_count_old', 
        y = 'waveform_count_new',
        hue = 'session',
        s = 50,
        ax = ax[0])
# plot x=y
min_count = min(merged_units['waveform_count_old'].min(), merged_units['waveform_count_new'].min())
max_count = max(merged_units['waveform_count_old'].max(), merged_units['waveform_count_new'].max())
ax[0].plot([min_count,max_count],[min_count,max_count],'k--')
ax[0].set_xlim([min_count,max_count])
ax[0].set_ylim([min_count,max_count])
ax[0].set_xlabel('Old waveform count')
ax[0].set_ylabel('New waveform count')
ax[0].set_title('Old vs New Waveform Counts\n' + f'N = {merged_units.shape[0]}')
# Turn off legend
ax[0].legend().remove()
waveform_diff = merged_units['waveform_count_new'] - merged_units['waveform_count_old']
ax[1].hist(waveform_diff, bins = 50, log = True)
ax[1].set_xlabel('New - Old Waveform Count\n(Positive = More New)')
plt.savefig(os.path.join(artifacts_dir,'old_vs_new_waveform_counts.png'))
plt.close()
