"""
Look for taste differences in spectral granger causality
"""
import tables
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
from tqdm import tqdm
from joblib import Parallel, delayed

def parallelize(func, arg_list, n_jobs=10):
    return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in tqdm(arg_list))

def frame_to_array(df, value_col = 'p-unc'):
    """
    Convert dataframe to n-dimensional array
    """
    ind_cols = [x for x in df.columns if x != value_col]
    inds = df[ind_cols].values.T
    array_shape = [len(np.unique(x)) for x in inds]
    array = np.zeros(array_shape)
    array[tuple(inds)] = df[value_col].values
    return array

############################################################
# Load Data
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

basename_list = [os.path.basename(this_dir) for this_dir in dir_list]
animal_name = [this_name.split('_')[0] for this_name in basename_list]
animal_count = np.unique(animal_name, return_counts=True)
session_count = len(basename_list)

n_string = f'N = {session_count} sessions, {len(animal_count[0])} animals'

names = ['granger_actual',
         'wanted_window',
         'time_vec',
         'freq_vec']


loaded_dat_list = []
node_names = []
for this_dir in dir_list:
    #this_dir = dir_list[0]
    save_path = '/ancillary_analysis/granger_causality/'
    # Get node paths for individual tastes
    h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
    with tables.open_file(h5_path) as h5:
        node_list = h5.list_nodes(save_path)
        node_list = [this_node._v_pathname for this_node in node_list \
                if 'all' not in os.path.basename(this_node._v_pathname)]
    node_names.append([os.path.basename(this_node) for this_node in node_list])
    with tables.open_file(h5_path, 'r') as h5:
        # Outer list is for each taste
        # Inner list is for each data type
        loaded_dat = [[h5.get_node(this_save_path, this_name)[:]
                      for this_name in names] \
                              for this_save_path in tqdm(node_list)]
        loaded_dat_list.append(loaded_dat)

node_names = np.stack(node_names)
if not np.mean(node_names == node_names[0][None,:]) == 1:
    raise ValueError('Node names do not match across sessions')

# loaded_dat_list: [session][taste][data_type]

#zipped_dat = [list(zip(*x)) for x in loaded_dat_list]
# zipped_dat : [taste][session][data_type]
zipped_dat = list(zip(*loaded_dat_list))
# zipped_dat : [taste][data_type][session]
zipped_dat = [list(zip(*x)) for x in zipped_dat]
# zipped_dat : [taste][data_type][session]
zipped_dat = list(zip(*zipped_dat))
zipped_dat = [np.stack(this_dat) for this_dat in zipped_dat]

# granger_actual shape = (taste, session repeats, time, freq, d1, d2)
(
    granger_actual,
    wanted_window,
    time_vec,
    freq_vec) = zipped_dat

wanted_window = np.array(wanted_window[0][0])/1000
stim_t = 2
corrected_window = wanted_window-stim_t
freq_vec = freq_vec[0][0]
time_vec = time_vec[0][0]
time_vec += corrected_window[0]

wanted_freq_range = [1, 100]
wanted_freq_inds = np.where(np.logical_and(freq_vec >= wanted_freq_range[0],
                                           freq_vec <= wanted_freq_range[1]))[0]
freq_vec = freq_vec[wanted_freq_inds]
# granger_actual shape = (session, repeats, time, freq, d1, d2)
granger_actual = granger_actual[:, :, :, :, wanted_freq_inds]
granger_actual = np.stack(
        [granger_actual[..., 0, 1],
         granger_actual[..., 1, 0]])
# granger_actual shape = (tastes, session, repeats, time, freq, direction)
granger_actual = np.moveaxis(granger_actual, 0, -1)

# Analyze single sessions at a time
# Otherwize memory blows up
# granger_actual shape = (session, tastes, repeats, time, freq, direction)
granger_actual = np.moveaxis(granger_actual, 0, 1)

############################################################
# Peform ANOVA for each time point, frequency, direction
############################################################
# Convert to pandas dataframe for easier handling with pingouin

# Pingouin keeps warning about being outdated
os.environ['OUTDATED_IGNORE'] = '1'

n_comparisons = np.product(granger_actual.shape[-3:])
anova_frame_list = []
#this_granger = granger_actual[0]
for this_granger in tqdm(granger_actual):
    inds = np.array(list(np.ndindex(this_granger.shape)))
    dim_names=  ['taste', 'repeats', 'time', 'freq', 'direction']
    df = pd.DataFrame(
            dict(
                zip(dim_names, inds.T)
                )
            )
    df['granger'] = this_granger.flatten()

    grouped_dat = list(df.groupby(['time', 'freq', 'direction']))

    temp_anova = lambda x: pg.anova(data=x[1], 
                                    dv='granger', 
                                    between='taste')['p-unc'].values[0]
    #pg.anova(data=this_group[1], dv='granger', between='taste')

    anova_results = parallelize(temp_anova, grouped_dat, n_jobs=30)

    anova_frame = pd.DataFrame(
                    data = [x[0] for x in grouped_dat],
                    columns = ['time', 'freq', 'direction']
                    )
    anova_frame['p-unc'] = anova_results
    anova_frame_list.append(anova_frame)

anova_array_list = [frame_to_array(x) for x in anova_frame_list]

# Write out to files
anova_save_path = '/ancillary_analysis/granger_causality/taste_anova'

for num, this_dir in enumerate(dir_list):
    # Get node paths for individual tastes
    h5_path = glob(os.path.join(this_dir, '*.h5'))[0]
    with tables.open_file(h5_path,'r+') as h5:
        if anova_save_path in h5:
            h5.remove_node(anova_save_path, recursive=True)
        h5.create_group(os.path.dirname(anova_save_path),
                        os.path.basename(anova_save_path))
        h5.create_array(
                anova_save_path,
                'taste_anova_pval_array',
                anova_array_list[num])
        # Also write out dimension order
        h5.create_array(
                anova_save_path,
                'taste_anova_pval_dim_order',
                np.array(['time', 'freq', 'direction']))
        for name, val in zip(['time', 'freq', 'direction'],
                             [time_vec, freq_vec, ['0->1', '1->0']]):
            h5.create_array(
                    anova_save_path,
                    'taste_anova_pval_dim_%s' % name,
                    val)
        h5.create_array(
                anova_save_path,
                'taste_anova_pval_dim_units',
                np.array(['s', 'Hz', '']))

