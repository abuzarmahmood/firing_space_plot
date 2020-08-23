"""
Code to analyze licks per taste for BAT rig data
"""


import numpy as np
import pandas as pd
import easygui as eg
import seaborn as sns
import matplotlib.pylab as plt
import glob
import os


# Import pandas dataframe containing extracted BAT data
data_dir = '/media/bigdata/Abuzar_Data/bat_data/data_files/'
file_path = np.sort(glob.glob(os.path.join(data_dir,'**.df')))[-1]
#file_path = eg.fileopenbox('Select file' , 'Select pickle file with BAT data')
relevant_columns = ['Animal','Notes','PRESENTATION','TUBE',
        'Trial_num','SOLUTION','LICKS']
bat_frame = pd.read_pickle(file_path)[relevant_columns]

# Remove zero lick trials
bat_frame = bat_frame[bat_frame.LICKS != 0]
group_names = ['Animal','Notes','SOLUTION']
bat_frame.sort_values(group_names, inplace=True)

# Group by and take MEDIAN
result_columns = group_names.copy()
result_columns.append('LICKS')
summary_dat = bat_frame[result_columns].groupby(group_names).describe()
summary_dat[[('LICKS','count'), ('LICKS','mean')]].sort_values(['Animal','SOLUTION','Notes'])

# Plot
wanted_animals = ['AM25','AM26']
labels = np.sort(bat_frame.Notes.unique())
g = sns.catplot(x = 'Notes', y = 'LICKS', col = 'SOLUTION', row = 'Animal', 
        data = bat_frame.loc[bat_frame['Animal'].isin(wanted_animals)], order = labels)
for ax in g.axes.flat:
    ax.set_xticklabels(labels,rotation = 45, horizontalalignment='right')
plt.tight_layout()
plt.show()
