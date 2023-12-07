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
from pprint import pprint
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

############################################################
## Helper Functions
############################################################

def parallelize(func, iterator, n_jobs = 16):
    #return Parallel(n_jobs = cpu_count()-2)\
    return Parallel(n_jobs = n_jobs, max_nbytes = 1e6)\
            (delayed(func)(this_iter) for this_iter in tqdm(enumerate(iterator)))

def gen_spike_train(spike_inds):
    """
    Generate spike train from spike indices
    """
    spike_train = np.zeros(spike_inds.max(axis=1)+1)
    spike_train[tuple(spike_inds)] = 1
    return spike_train

def process_ind(ind_num, this_ind):
    #this_ind = fin_inds[0]
    #this_ind_str = '_'.join([str(x) for x in this_ind])

    ############################## 
    # Generate save paths
    ############################## 
    this_ind_str = '_'.join([f'{x:03}' for x in this_ind])
    pval_save_name = f'{this_ind_str}_p_val_frame.csv'
    pval_save_path = os.path.join(fin_save_path,pval_save_name)
    ll_save_name = f'{this_ind_str}_ll_frame.csv'
    ll_save_path = os.path.join(fin_save_path,ll_save_name)

    if os.path.exists(pval_save_path) and os.path.exists(ll_save_path):
        print(f'Already processed {this_ind_str}')
        return

    ############################## 
    # Load data 
    ############################## 
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

    ############################## 
    # Process data 
    ############################## 
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
    # **Note**: this needs to be done after design matrix is created
    # so that overlap of history between trials is avoided
    trial_lims_vec = np.arange(*trial_lims)
    actual_design_mat = actual_design_mat.loc[actual_design_mat.trial_time.isin(trial_lims_vec)]

    ############################## 
    # Fit model 
    ############################## 

    # Try to get n_fits in max_tries 
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

    ############################## 
    # Process results 
    ############################## 
    if len(fit_list) > 0:

        ##############################
        # Extract info from model-fit

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

        # Also save predicted firing rates 
        if len(fit_list) > 1:
            best_fit_ind = int(ll_frame.loc[ll_frame.actual.idxmax()].fit_num)
            best_fit = fit_list[best_fit_ind]
        else:
            best_fit = fit_list[0]

        ##############################
        # Save original and predicted PSTHs

        # PSTH
        time_bins = actual_design_mat.trial_time.unique()
        design_trials = list(actual_design_mat.groupby('trial_labels'))
        design_spikes = [x.sort_values('trial_time').spikes.values for _,x in design_trials]
        design_spikes_array = np.stack(design_spikes)
        #design_spikes_tuple = np.array(np.where(design_spikes))
        #np.save(os.path.join(fin_save_path, f'{this_ind_str}_design_spikes.npy'), design_spikes_tuple)
        np.save(os.path.join(fin_save_path, f'{this_ind_str}_design_spikes.npy'), design_spikes_array)

        # Predicted PSTH
        pred_spikes = pd.DataFrame(best_fit.predict(actual_design_mat[best_fit.params.index]), 
                                   columns = ['spikes'])
        pred_spikes.loc[pred_spikes.spikes > bin_width, 'spikes'] = bin_width
        pred_spikes['trial_labels'] = actual_design_mat.trial_labels
        pred_spikes['trial_time'] = actual_design_mat.trial_time
        pred_trials = list(pred_spikes.groupby('trial_labels'))
        pred_spikes = [x.sort_values('trial_time').spikes.values for _,x in pred_trials]
        pred_spikes_array = np.stack(pred_spikes).astype(np.float16)
        np.save(os.path.join(fin_save_path, f'{this_ind_str}_pred_spikes.npy'), pred_spikes_array)
        del p_val_list, p_val_fin, ll_outs, ll_frame, temp_design_mat
    else:
        print('Could not fit a model')

    ##############################
    # Delete variables to save memory
    ##############################
    del data_frame, actual_design_mat, fit_list 
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
#parallelize(try_process,fin_inds, n_jobs = 8)
