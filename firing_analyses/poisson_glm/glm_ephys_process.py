"""
First pass at applying glm framework to ephys data
"""

import numpy as np
import pandas as pd
import sys
#sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm'
#base_path = '/home/exouser/Desktop/ABU/firing_space_plot/firing_analyses/poisson_glm'
sys.path.append(base_path)
import glm_tools as gt
from pandas import DataFrame as df
from pandas import concat
import os
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed, cpu_count
from glob import glob
import json

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6

#base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm'
save_path = os.path.join(base_path,'artifacts')

def parallelize(func, iterator, n_jobs = 16):
    #return Parallel(n_jobs = cpu_count()-2)\
    return Parallel(n_jobs = n_jobs)\
            (delayed(func)(this_iter) for this_iter in tqdm(enumerate(iterator)))

def gen_spike_train(spike_inds):
    spike_train = np.zeros(spike_inds.max(axis=1)+1)
    spike_train[tuple(spike_inds)] = 1
    return spike_train

# Check if previous runs present
run_list = sorted(glob(os.path.join(save_path, 'run*')))
run_basenames = [os.path.basename(x) for x in run_list]
print(f'Present runs : {run_basenames}')
input_run_ind = int(input('Please specify current run (integer) :'))
fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03}')

if not os.path.exists(fin_save_path):
    os.makedirs(fin_save_path)

############################################################
# Parameters
hist_filter_len = 75
stim_filter_len = 300
coupling_filter_len = 75

bin_width = 2
# Reprocess filter lens
hist_filter_len_bin = hist_filter_len // bin_width
stim_filter_len_bin = stim_filter_len // bin_width
coupling_filter_len_bin = coupling_filter_len // bin_width

trial_start_offset = -2000
trial_lims = np.array([1500,4000])
stim_t = 2000

# Define basis kwargs
basis_kwargs = dict(
    n_basis = 15,
    basis = 'cos',
    basis_spread = 'log',
    )

# Number of fits on actual data (expensive)
n_fits = 1
n_max_tries = 20
# Number of shuffles tested against each fit
n_shuffles_per_fit = 5

# Save run parameters
params_dict = dict(
        hist_filter_len = hist_filter_len,
        stim_filter_len = stim_filter_len,
        coupling_filter_len = coupling_filter_len,
        bin_width = bin_width,
        hist_filter_len_bin = hist_filter_len_bin,
        stim_filter_len_bin = stim_filter_len_bin,
        coupling_filter_len_bin = coupling_filter_len_bin,
        trial_start_offset = trial_start_offset,
        trial_lims = list(trial_lims),
        stim_t = stim_t,
        basis_kwargs = basis_kwargs,
        n_fits = n_fits,
        n_max_tries = n_max_tries,
        n_shuffles_per_fit = n_shuffles_per_fit,
        )

params_save_path = os.path.join(fin_save_path, 'fit_params.json')
with open(params_save_path, 'w') as outf:
    json.dump(params_dict, outf, indent = 4, default = int)

############################################################

reprocess_data = False 
spike_list_path = os.path.join(save_path,'spike_save')

if reprocess_data:
    sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
    from ephys_data import ephys_data

    file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
    file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
    basenames = [os.path.basename(x) for x in file_list]

    spike_list = []
    unit_region_list = []
    for ind in trange(len(file_list)):
        dat = ephys_data(file_list[ind])
        dat.get_spikes()
        spike_list.append(np.array(dat.spikes))
        # Calc mean firing rate
        mean_fr = np.array(dat.spikes).mean(axis=(0,1,3))
        dat.get_region_units()
        region_units = dat.region_units
        region_names = dat.region_names
        region_names = [[x]*len(y) for x,y in zip(region_names,region_units)]
        region_units = np.concatenate(region_units)
        region_names = np.concatenate(region_names)
        unit_region_frame = df(
                {'region':region_names,
                 'unit':region_units,
                 }
                )
        unit_region_frame['basename'] = basenames[ind]
        unit_region_frame['session'] = ind
        unit_region_frame = unit_region_frame.sort_values(by=['unit'])
        unit_region_frame['mean_rate'] = mean_fr
        unit_region_list.append(unit_region_frame)

    # Save spike_list as numpy arrays

    spike_inds_list = [np.stack(np.where(x)) for x in spike_list]

    if not os.path.exists(spike_list_path):
        os.makedirs(spike_list_path)
    for ind,spikes in enumerate(spike_list):
        #np.save(os.path.join(spike_list_path,f'{ind}_spikes.npy'),spikes)
        np.save(os.path.join(spike_list_path,f'{ind:03}_spike_inds.npy'),spike_inds_list[ind])

    unit_region_frame = concat(unit_region_list)
    unit_region_frame = unit_region_frame.reset_index(drop=True)
    unit_region_frame.to_csv(os.path.join(save_path,'unit_region_frame.csv'))

    # Make sure all sessions have 4 tastes
    assert all([x.shape[0] == 4 for x in spike_list])

    # Find neurons per session
    nrn_counts = [x.shape[2] for x in spike_list]

    # Process each taste separately
    inds = np.array(list(product(range(len(spike_list)),range(spike_list[0].shape[0]))))
    fin_inds = []
    for ind in tqdm(inds):
        this_count = nrn_counts[ind[0]]
        for nrn in range(this_count):
            fin_inds.append(np.hstack((ind,nrn)))
    fin_inds = np.array(fin_inds)

    ind_frame = df(fin_inds,columns=['session','taste','neuron'])
    ind_frame.to_csv(os.path.join(save_path,'ind_frame.csv'))

else:
    ############################################################
    # Reconstitute data
    spike_inds_paths = sorted(glob(os.path.join(spike_list_path,'*_spike_inds.npy')))
    spike_inds_list = [np.load(x) for x in spike_inds_paths]
    spike_list = [gen_spike_train(x) for x in spike_inds_list]

    # Load unit_region_frame
    unit_region_frame = pd.read_csv(os.path.join(save_path,'unit_region_frame.csv'),index_col=0)

    # Load ind_frame
    ind_frame = pd.read_csv(os.path.join(save_path,'ind_frame.csv'),index_col=0)
    fin_inds = ind_frame.values

############################################################

# While iterating, will have to keep track of
# 1. Region each neuron belongs to
# 2. log-likehood for each data-type
# 3. p-values
stim_vec = np.zeros(spike_list[0].shape[-1])
stim_vec[stim_t] = 1


def process_ind(ind_num, this_ind):
    #this_ind = fin_inds[0]
    #this_ind_str = '_'.join([str(x) for x in this_ind])
    this_ind_str = '_'.join([f'{x:03}' for x in this_ind])
    pval_save_name = f'{this_ind_str}_p_val_frame.csv'
    pval_save_path = os.path.join(fin_save_path,pval_save_name)
    ll_save_name = f'{this_ind_str}_ll_frame.csv'
    ll_save_path = os.path.join(fin_save_path,ll_save_name)

    if os.path.exists(pval_save_path) and os.path.exists(ll_save_path):
        print(f'Already processed {this_ind_str}')
        return

    this_session_dat = spike_list[this_ind[0]]
    this_taste_dat = this_session_dat[this_ind[1]]
    this_nrn_ind = this_ind[2]
    other_nrn_inds = np.delete(np.arange(this_session_dat.shape[2]),
            this_nrn_ind)
    n_coupled_neurons = len(other_nrn_inds)

    this_nrn_dat = this_taste_dat[:,this_nrn_ind]
    other_nrn_dat = this_taste_dat[:,other_nrn_inds]
    stim_dat = np.tile(stim_vec,(this_taste_dat.shape[0],1))

    this_nrn_flat = np.concatenate(this_nrn_dat)
    other_nrn_flat = np.concatenate(np.moveaxis(other_nrn_dat, 1, -1)).T
    stim_flat = np.concatenate(stim_dat)

    #import importlib
    #importlib.reload(gt)

    # To convert to dataframe, make sure trials are not directly
    # concatenated as that would imply temporal continuity
    data_frame = gt.gen_data_frame(
            this_nrn_flat,
            other_nrn_flat,
            stim_flat,
            stim_filter_len = stim_filter_len,
            trial_start_offset = trial_start_offset,
            )

    # Replace coupling data names with actual indices
    coup_names = ['coup_{}'.format(x) for x in range(n_coupled_neurons)] 
    replace_names = ['coup_{}'.format(x) for x in other_nrn_inds]
    replace_dict = dict(zip(coup_names, replace_names))
    data_frame = data_frame.rename(columns=replace_dict)

    # Bin data
    data_frame['time_bins'] = data_frame.trial_time // bin_width
    data_frame = data_frame.groupby(['trial_labels','time_bins']).sum()
    data_frame['trial_time'] = data_frame['trial_time'] // bin_width
    data_frame = data_frame.reset_index()

    # Create design mat
    actual_design_mat = gt.dataframe_to_design_mat(
            data_frame,
            hist_filter_len = hist_filter_len_bin,
            stim_filter_len = stim_filter_len_bin,
            coupling_filter_len = coupling_filter_len_bin,
            basis_kwargs = basis_kwargs,
            )

    # Cut to trial_lims
    # Note, this needs to be done after design matrix is created
    # so that overlap of history between trials is avoided
    trial_lims_vec = np.arange(*trial_lims)
    actual_design_mat = actual_design_mat.loc[actual_design_mat.trial_time.isin(trial_lims_vec)]

    fit_list = []
    for i in trange(n_max_tries):
        if len(fit_list) < n_fits:
            try:
                res, temp_design_mat, = gt.gen_actual_fit(
                        data_frame, # Not used if design_mat is provided
                        hist_filter_len = hist_filter_len_bin,
                        stim_filter_len = stim_filter_len_bin,
                        coupling_filter_len = coupling_filter_len_bin,
                        basis_kwargs = basis_kwargs,
                        actual_design_mat = actual_design_mat,
                        )
                # Saving these is VERY expensive (~20MB per model)
                #res.save(os.path.join(save_path,f'{this_ind_str}_fit_{i}.pkl'))
                fit_list.append(res)
            except:
                print('Failed fit')
        else:
            print('Finished fitting')
            break

    p_val_list = [res.pvalues for res in fit_list]
    p_val_fin = []
    for i, p_vals in enumerate(p_val_list):
        p_vals = pd.DataFrame(p_vals)
        p_vals['fit_num'] = i
        p_vals['values'] = fit_list[i].params.values 
        p_val_fin.append(p_vals)

    p_val_frame = pd.concat(p_val_fin)
    p_val_frame.reset_index(inplace=True)
    p_val_frame.rename(columns={'index':'param', 0 : 'p_val'},inplace=True)
    p_val_frame.to_csv(pval_save_path)

    ll_names = ['actual','trial_sh','circ_sh','rand_sh']
    ll_outs = [[gt.calc_loglikelihood(actual_design_mat, res)\
            for i in range(n_shuffles_per_fit)]\
            for res in tqdm(fit_list)]
    ll_outs = np.array(ll_outs).reshape(-1,4)
    ll_frame = pd.DataFrame(ll_outs, columns=ll_names)
    ll_frame['fit_num'] = np.repeat(np.arange(len(fit_list)), n_shuffles_per_fit)
    ll_frame.to_csv(ll_save_path)

    del data_frame, actual_design_mat, fit_list, p_val_list, p_val_fin, ll_outs, ll_frame, temp_design_mat
    del this_session_dat, this_taste_dat, this_nrn_dat, other_nrn_dat, stim_dat
    del this_nrn_flat, other_nrn_flat, stim_flat
    print(f'Finished: {ind_num}, {this_ind}')

#for this_ind in tqdm(fin_inds):
def try_process(this_ind):
    try:
        process_ind(*this_ind)
    except:
        print('Failed')
        pass
parallelize(try_process,fin_inds, n_jobs = 8)
