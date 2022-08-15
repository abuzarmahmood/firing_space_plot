"""
PER RECORING SESSION:
1) Significance of difference in inter-region xcorr and corresponding shuffle
    - Per frequency
    - Over time
2) Significance of difference in intra-region xcorr
    - Per frequency
    - Over time
    ** Don't care about difference from shuffle

AGGREGATE OVER ALL SESSIONS
"""
########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

########################################
# Import modules
########################################

import os
import sys
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
from numba import jit
import ast
import pylab as plt
import pingouin as pg
from pathlib import Path

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
#import visualize

plot_dir = '/media/bigdata/firing_space_plot/lfp_analyses/lfp_amp_xcorr/Plots'
#file_list_path = '/media/bigdata/firing_space_plot/lfp_analyses/'\
#        'lfp_amp_xcorr/file_list.txt'

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]

#file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 
save_path = '/stft/analyses/amplitude_xcorr'
wanted_frames = ['inter_region_frame','intra_region_frame']

wanted_path = os.path.join(save_path, wanted_frames[0])
h5_files = [tables.open_file(x,'r') for x in file_list]

file_list = [x for x,this_file in zip(file_list,h5_files)\
        if wanted_path in this_file]
# Close h5_files
for this_file in h5_files:
    this_file.close()

basenames = [os.path.basename(x).split('.')[0] for x in file_list]

frame_list = [[pd.read_hdf(h5_path,
                os.path.join(save_path,this_frame_name)) \
                        for this_frame_name in wanted_frames] \
                        for h5_path in tqdm(file_list)]
grouped_frame_list = [[x.groupby(['label','freq']).median() \
        for x in this_session] for this_session in frame_list]
# Add session name before aggreageting
grouped_frame_list = [[x.assign(name=name) for x in this_session] \
        for this_session,name in tqdm(zip(grouped_frame_list, basenames))]
flat_grouped_frame = pd.concat([x for y in grouped_frame_list for x in y])
flat_grouped_frame.reset_index(inplace=True)

def group_mapper(x):
    if 'intra' in x and not 'shuffle' in x:
        return 'intra'
    if 'inter' in x and not 'shuffle' in x:
        return 'inter'
    if 'shuffle' in x:
        return 'shuffle'

def freq_mapper(x):
    if x >=4 and x<7:
        return 'theta'
    if x >=7 and x<=12:
        return 'alpha'
    if x >12 and x<30:
        return 'beta'

flat_grouped_frame['group'] = flat_grouped_frame['label'].apply(group_mapper)
flat_grouped_frame['band'] = flat_grouped_frame['freq'].apply(freq_mapper)
from pandas.api.types import CategoricalDtype
x_order = ['theta','alpha','beta']
cat_type = CategoricalDtype(x_order, ordered=True) 
flat_grouped_frame['band'] = flat_grouped_frame['band'].astype(cat_type)
flat_grouped_frame = flat_grouped_frame.groupby(['group','band', 'name']).mean()
flat_grouped_frame.reset_index(inplace=True)

g1 = sns.stripplot(data = flat_grouped_frame, x = 'band', y = 'xcorr',
                        hue = 'group', size = 7, order = x_order, alpha = 0.8)
handles, labels = g1.get_legend_handles_labels()
handles = handles[:3]
labels = [x.title() for x in labels[:3]]
g1.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
        title = 'Comparison Type')
plt.tight_layout()
plt.xlabel('Frequency Band')
plt.ylabel('Scaled XCorr Value')
fin_xticklabels = [x.get_text().title() for x in g1.get_xticklabels()]
g1.set_xticklabels(fin_xticklabels)
fig = plt.gcf()
plt.show()
#fig.savefig(os.path.join(plot_dir,'aggregate_plot_no_zscore.svg'),format='svg')
#plt.close(fig)

g1 = sns.stripplot(data = flat_grouped_frame, x = 'band', y = 'xcorr',
                        hue = 'group', size = 7, alpha = 0.5,
                        dodge = True, color = 'grey')
sns.boxplot(data = flat_grouped_frame, x = 'band', y = 'xcorr',
                        hue = 'group', ax = g1, showfliers=False)
handles, labels = g1.get_legend_handles_labels()
handles = handles[:3]
labels = [x.title() for x in labels[:3]]
g1.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
        title = 'Comparison Type')
plt.tight_layout()
plt.xlabel('Frequency Band')
plt.ylabel('Scaled XCorr Value')
fin_xticklabels = [x.get_text().title() for x in g1.get_xticklabels()]
g1.set_xticklabels(fin_xticklabels)
fig = plt.gcf()
plt.show()
#fig.savefig(os.path.join(plot_dir,'aggregate_plot2_no_zscore.svg'),format='svg')
#plt.close(fig)

# RM-ANOVA
pg.rm_anova(data = flat_grouped_frame.dropna(),
        dv = 'xcorr', within = ['group','band'], subject = 'name')

# VanillANOVA
pg.anova(data = flat_grouped_frame.dropna(),
        dv = 'xcorr', between = ['group','band'])

# Kruskal
pg.kruskal(data = flat_grouped_frame.dropna(),
        dv = 'xcorr', between = 'group')

# Post-hoc testing
group_combinations = list(it.combinations(flat_grouped_frame['group'].unique(),2))
grouped_frames = list(flat_grouped_frame.groupby('band'))
p_val_list = []
for name, this_frame in grouped_frames:
    for this_comb in group_combinations:
        dat1 = this_frame[this_frame['group'] == this_comb[0]]['xcorr']
        dat2 = this_frame[this_frame['group'] == this_comb[1]]['xcorr']
        p_val_dict = dict(
            band = name, 
            group1 = this_comb[0], 
            group2 = this_comb[1],
            p_val = stats.mannwhitneyu(dat1,dat2)[1])
        p_val_list.append(p_val_dict)
p_val_frame = pd.DataFrame(p_val_list)

#grouped_frames = list(flat_grouped_frame.groupby(['group','band']))
#group_names = ["_".join(x[0]) for x in grouped_frames]
#inds = np.arange(len(group_names))
#iter_inds = list(it.product(inds,inds))
#
#p_val_mat = np.empty((len(group_names),len(group_names)))
#for this_iter in tqdm(iter_inds):
#    frame1 = grouped_frames[this_iter[0]]
#    frame2 = grouped_frames[this_iter[1]]
#    dat1 = frame1[1]['xcorr']
#    dat2 = frame2[1]['xcorr']
#    p_val_mat[this_iter] = stats.mannwhitneyu(dat1,dat2)[1]
