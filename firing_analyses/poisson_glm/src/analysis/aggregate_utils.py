import numpy as np
import pylab as plt
import pandas as pd
import sys
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/src')
import utils.makeRaisedCosBasis as cb
from analysis import aggregate_utils
from pandas import DataFrame as df
from pandas import concat
import os
from tqdm import tqdm, trange
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from itertools import product
from glob import glob

############################################################
# Get Spikes

file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]


############################################################
# Load Data
############################################################

def return_data(save_path, run_str = 'run_1'):

    unit_region_frame_path = os.path.join(save_path,'unit_region_frame.csv')
    if not os.path.exists(unit_region_frame_path):
        unit_region_list = get_unit_region_list(file_list)
        unit_region_frame = pd.concat(unit_region_list)
    else:
        unit_region_frame = pd.read_csv(
                os.path.join(save_path,'unit_region_frame.csv'), index_col = 0)

    unit_region_frame.rename(columns = {'unit' : 'neuron'}, inplace=True)
    ind_frame = pd.read_csv(
            os.path.join(save_path,'ind_frame.csv'), index_col = 0)

    p_val_frame_paths = sorted(
            glob(os.path.join(save_path, run_str, '*p_val_frame.csv')))
    ll_frame_paths = sorted(
            glob(os.path.join(save_path, run_str, '*ll_frame.csv')))
    pred_spikes_paths = sorted(
            glob(os.path.join(save_path, run_str, '*pred_spikes.npy')))
    design_spikes_paths = sorted(
            glob(os.path.join(save_path, run_str, '*design_spikes.npy')))

    p_val_basenames = [os.path.basename(x) for x in p_val_frame_paths]
    ll_basenames = [os.path.basename(x) for x in ll_frame_paths]
    pred_spike_basenames = [os.path.basename(x) for x in pred_spikes_paths]
    design_spike_basenames = [os.path.basename(x) for x in design_spikes_paths]

    p_val_inds_str = [x.split('_p_val')[0] for x in p_val_basenames]
    ll_inds_str = [x.split('_ll')[0] for x in ll_basenames]
    pred_spike_inds_str = [x.split('_pred')[0] for x in pred_spike_basenames]
    design_spike_inds_str = [x.split('_design')[0] for x in design_spike_basenames]

    p_val_inds = np.array([list(map(int,x.split('_'))) for x in p_val_inds_str])
    ll_inds = np.array([list(map(int,x.split('_'))) for x in ll_inds_str])
    pred_spike_inds = np.array(
            [list(map(int,x.split('_'))) for x in pred_spike_inds_str])
    design_spike_inds = np.array(
            [list(map(int,x.split('_'))) for x in design_spike_inds_str])

    ind_names = ['session','taste', 'neuron']
    spike_dat_inds_frame = pd.DataFrame(columns = ind_names,
                                        data = pred_spike_inds)

    # Data indices for fits belonging to the same neuron
    session_taste_group_list = list(
            spike_dat_inds_frame.groupby(['session','neuron']))
    session_taste_inds = [x[0] for x in session_taste_group_list]
    all_taste_inds = [x[1].index.values for x in session_taste_group_list]

    # Make sure inds match up
    assert np.all(p_val_inds == ll_inds), "Mismatched inds"
    assert np.all(p_val_inds == pred_spike_inds), "Mismatched inds"
    assert np.all(p_val_inds == design_spike_inds), "Mismatched inds"

    p_val_frame_list = [pd.read_csv(x, index_col=0) for x in p_val_frame_paths]
    ll_frame_list = [pd.read_csv(x, index_col = 0) for x in ll_frame_paths]
    # Reset index and mark as fit_num
    for i in range(len(p_val_frame_list)):
        ll_frame_list[i].reset_index(inplace=True)
        ll_frame_list[i].rename(columns = {'index' : 'fit_num'}, inplace=True)
    pred_spikes_list = [np.load(x) for x in pred_spikes_paths]
    design_spikes_list = [np.load(x) for x in design_spikes_paths]

    # Add inds to frames
    # Order : [sessio, taste, neurons]
    p_val_frame_list_fin = []
    ll_frame_list_fin = []
    for i in trange(len(p_val_frame_list)):
        this_ind = p_val_inds[i]
        this_pval_frame = p_val_frame_list[i]
        this_ll_frame = ll_frame_list[i]
        this_pval_frame['session'] = this_ind[0]
        this_pval_frame['taste'] = this_ind[1]
        this_pval_frame['neuron'] = this_ind[2]
        this_ll_frame['session'] = this_ind[0]
        this_ll_frame['taste'] = this_ind[1]
        this_ll_frame['neuron'] = this_ind[2]
        p_val_frame_list_fin.append(this_pval_frame)
        ll_frame_list_fin.append(this_ll_frame)

    fin_pval_frame = pd.concat(p_val_frame_list_fin)
    fin_pval_frame = fin_pval_frame.sort_values(by=['session','taste','neuron'])
    # Reset index
    fin_pval_frame = fin_pval_frame.reset_index(drop=True)

    fin_ll_frame = pd.concat(ll_frame_list_fin)
    # Sort by inds
    fin_ll_frame = fin_ll_frame.sort_values(by=['session','taste','neuron'])
    # Merge fin_ll_frame and unit_region_frame
    fin_ll_frame = pd.merge(fin_ll_frame, unit_region_frame, on = ['session','neuron'])

    return (unit_region_frame,
            fin_pval_frame, 
            fin_ll_frame, 
            pred_spikes_list, 
            design_spikes_list, 
            ind_frame,
            )
