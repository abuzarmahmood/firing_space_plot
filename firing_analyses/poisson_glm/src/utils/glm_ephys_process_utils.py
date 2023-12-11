import numpy as np
import pandas as pd
import sys
#sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm'
#base_path = '/home/exouser/Desktop/ABU/firing_space_plot/firing_analyses/poisson_glm'
sys.path.append(base_path)
##############################
# import glm_tools as gt
# from utils.utils import (
#         gen_data_frame,
#         dataframe_to_design_mat,
#         calc_loglikelihood,
#         )
from utils import utils
from utils import glm_fitting 
##############################
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
def gen_stim_vec(spike_list, params_dict):
    stim_t = params_dict['stim_t']
    stim_vec = np.zeros(spike_list[0].shape[-1])
    stim_vec[stim_t] = 1
    return stim_vec


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



def process_ind(
        ind_num, 
        this_ind,
        spike_list,
        stim_vec,
        params_dict,
        fin_save_path,
        ):
    """
    Process a single index

    Inputs:
        ind_num : index number for progress tracking
        this_ind : index tuple for extracting data
        spike_list : list of spike data
        stim_vec : stimulus vector
        params_dict : dictionary of parameters

    Outputs:
        p_val : p value for this index
        ll : log likelihood for this index
    """

    ############################## 
    # Extract values from params_dict
    ############################## 
    hist_filter_len = params_dict['hist_filter_len']
    stim_filter_len = params_dict['stim_filter_len']
    coupling_filter_len = params_dict['coupling_filter_len']

    trial_start_offset = params_dict['trial_start_offset']
    trial_lims = np.array(params_dict['trial_lims'])
    stim_t = params_dict['stim_t']

    bin_width = params_dict['bin_width']

    # Reprocess filter lens
    hist_filter_len_bin = params_dict['hist_filter_len_bin'] 
    stim_filter_len_bin = params_dict['stim_filter_len_bin']
    coupling_filter_len_bin = params_dict['coupling_filter_len_bin']

    # Define basis kwargs
    basis_kwargs = params_dict['basis_kwargs'] 

    # Number of fits on actual data (expensive)
    n_fits = params_dict['n_fits']
    n_max_tries = params_dict['n_max_tries']
    n_shuffles_per_fit = params_dict['n_shuffles_per_fit']

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

    data_frame = utils.gen_data_frame(
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
    actual_design_mat = utils.dataframe_to_design_mat(
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
    actual_design_mat = \
            actual_design_mat.loc[
                    actual_design_mat.trial_time.isin(trial_lims_vec)
                    ]

    ############################## 
    # Fit model 
    ############################## 

    # We want to make sure that we are capturing trial-specific
    # information. To do this, we will fit the model to the actual
    # data, and then fit the model to a trial-shuffled version of the
    # data. We will then compare the log-likelihoods of the two fits.
    # If the trial-shuffled model performs better, then we know that
    # the trial-specific information is not being captured by the
    # model.

    fit_types = [
             'actual',
             'trial_shuffled',
             # 'circle_shuffled',
             # 'random_shuffled',
             ]

    # # Plot a couple trials of the actual data and trial-shuffled data
    # actual_frame = data_frame.copy()
    # trial_shuffled_frame = utils.gen_trial_shuffle(data_frame)

    # single_fit_kwargs = dict(
    #         data_frame = data_frame, # Not used if design_mat is provided
    #         hist_filter_len = hist_filter_len_bin,
    #         stim_filter_len = stim_filter_len_bin,
    #         coupling_filter_len = coupling_filter_len_bin,
    #         basis_kwargs = basis_kwargs,
    #         design_mat = actual_design_mat,
    #         )

    # all_fit_lists = [] # List of lists of fit outs
    # for this_fit_type in fit_types:
    #     # single_fit_kwargs = {
    #     #         **single_fit_kwargs,
    #     #         'fit_type' : this_fit_type,
    #     #         }
    #     # Get at least n_fits in n_max_tries
    #     this_fit_list = gen_enough_fits(
    #             actual_design_mat,
    #             # single_fit_kwargs,
    #             n_fits = n_fits,
    #             n_max_tries = n_max_tries,
    #             )
    #     all_fit_lists.append(this_fit_list)

    # outs = list of lists
    # outer_list = # of fits
    # inner_list = res_list, train_dat_list, test_ll_list 
    # each element contains data for actual and trial-shuffled fits
    fit_outs = gen_enough_fits(
                actual_design_mat,
                # single_fit_kwargs,
                n_fits = n_fits,
                n_max_tries = n_max_tries,
                )

    # Separate out information for each type of fit
    actual_outs = [[x[0] for x in this_fit] for this_fit in fit_outs]
    trial_shuffled_outs = [[x[1] for x in this_fit] for this_fit in fit_outs]
    all_fit_lists = [actual_outs, trial_shuffled_outs]

    fin_pval_frame = pd.concat([
        return_pval_frame(this_fit_list, this_fit_type) \
                for this_fit_list, this_fit_type in \
                zip(all_fit_lists, fit_types)
                ])
    fin_pval_frame.to_csv(pval_save_path)

    fin_ll_frame = pd.concat([
       return_ll_frame(this_fit_list, this_fit_type) \
               for this_fit_list, this_fit_type in \
               zip(all_fit_lists, fit_types)
               ])

    # Pivot to make fit_type columns
    fin_ll_frame = pd.pivot(
            fin_ll_frame,
            index = 'fit_num',
            columns = 'fit_type',
            values = 'll',
            )
    fin_ll_frame = fin_ll_frame.reset_index(drop=True)
    fin_ll_frame.to_csv(ll_save_path)

    # # Fit matched differences in ll
    # fin_ll_frame.groupby('fit_num')['ll'].diff()

    ############################## 
    # Process results 
    ############################## 
    if len(fit_outs) > 0:

        ##############################
        # Extract info from model-fit

        # Also save predicted firing rates 
        if len(fit_outs) > 1:
            best_fit_ind = \
                    int(fin_ll_frame.loc[fin_ll_frame.actual.idxmax()].name)
            #best_fit = fit_list[best_fit_ind]
            best_fit = actual_outs[best_fit_ind][0] 
        else:
            best_fit = actual_outs[0][0]

        ##############################
        # Save original and predicted PSTHs

        # PSTH
        # We have to recreate the PSTHs from the design matrix
        # because the design matrix is cut to trial_lims whereas the
        # original data is not

        time_bins = actual_design_mat.trial_time.unique()
        design_trials = list(actual_design_mat.groupby('trial_labels'))
        design_spikes = [x.sort_values('trial_time').spikes.values \
                for _,x in design_trials]
        design_spikes_array = np.stack(design_spikes)
        #design_spikes_tuple = np.array(np.where(design_spikes))
        #np.save(os.path.join(fin_save_path, f'{this_ind_str}_design_spikes.npy'), design_spikes_tuple)
        np.save(
                os.path.join(
                    fin_save_path, 
                    f'{this_ind_str}_design_spikes.npy'
                    ), 
                design_spikes_array)

        # Predicted PSTH
        pred_spikes = pd.DataFrame(
                best_fit.predict(actual_design_mat[best_fit.params.index]), 
                                   columns = ['spikes'])
        pred_spikes.loc[pred_spikes.spikes > bin_width, 'spikes'] = bin_width
        pred_spikes['trial_labels'] = actual_design_mat.trial_labels
        pred_spikes['trial_time'] = actual_design_mat.trial_time
        pred_trials = list(pred_spikes.groupby('trial_labels'))
        pred_spikes = [x.sort_values('trial_time').spikes.values for _,x in pred_trials]
        pred_spikes_array = np.stack(pred_spikes).astype(np.float16)
        np.save(os.path.join(fin_save_path, f'{this_ind_str}_pred_spikes.npy'), pred_spikes_array)
        #del p_val_list, p_val_fin, ll_outs, ll_frame, temp_design_mat
        del fit_outs, fin_ll_frame, fin_pval_frame
    else:
        print('Could not fit a model')

    ##############################
    # Delete variables to save memory
    ##############################
    del data_frame, actual_design_mat
    del this_session_dat, this_taste_dat, this_nrn_dat, other_nrn_dat, stim_dat
    del this_nrn_flat, other_nrn_flat, stim_flat
    print(f'Finished: {ind_num}, {this_ind}')


def gen_enough_fits(
        design_mat,
        # single_fit_kwargs,
        n_fits = 10,
        n_max_tries = 100,
        ):
    """
    Generate enough fits to get n_fits in n_max_tries

    Inputs:
        single_fit_kwargs: dict
            kwargs for glm_fitting.gen_actual_fit
        n_fits: int
        n_max_tries: int

    Returns:
        fit_list: list of GLMResults
    """

    # Try to get n_fits in max_tries 
    fit_list = []
    for i in trange(n_max_tries):
        if len(fit_list) < n_fits:
            try:
                # res, temp_design_mat, test_ll = glm_fitting.perform_fit(
                #         **single_fit_kwargs,
                #         )
                res_list, train_dat_list, test_ll_list = \
                        glm_fitting.perform_fit_actual_and_trial_shuffled_fit(
                                design_mat,
                                )
                #res, temp_design_mat, test_ll = glm_fitting.gen_actual_fit(
                        #        data_frame, # Not used if design_mat is provided
                        #        hist_filter_len = hist_filter_len_bin,
                        #        stim_filter_len = stim_filter_len_bin,
                        #        coupling_filter_len = coupling_filter_len_bin,
                        #        basis_kwargs = basis_kwargs,
                        #        actual_design_mat = actual_design_mat,
                        #        fit_type = wanted_fit_type,
                        #        )
                # Saving these is VERY expensive (~20MB per model)
                #res.save(os.path.join(save_path,f'{this_ind_str}_fit_{i}.pkl'))
                # fit_list.append([res, train_dat_list, test_ll])
                fit_list.append([res_list, train_dat_list, test_ll_list])
            except:
                print('Failed fit')
        else:
            print('Finished fitting')
            break
    return fit_list


#for this_ind in tqdm(fin_inds):
def try_process(this_ind):
    try:
        process_ind(*this_ind)
    except:
        print('Failed')
        pass
#parallelize(try_process,fin_inds, n_jobs = 8)

def return_pval_frame(fit_list, this_fit_type):
    """
    Note: Outs per fit = (res, design_mat, test_ll)

    Process output of gen_enough_fits

    Input:
    - fit_list: list of fit outs
    - this_fit_type: str, type of fit
    """
    assert len(fit_list) > 0, 'No fits to process'

    p_val_list = [this_fit[0].pvalues for this_fit in fit_list]
    p_val_fin = []
    for i, p_vals in enumerate(p_val_list):
        p_vals = pd.DataFrame(p_vals)
        p_vals['fit_num'] = i
        p_vals['values'] = fit_list[i][0].params.values 
        p_vals['fit_type'] = this_fit_type
        p_val_fin.append(p_vals)

    p_val_frame = pd.concat(p_val_fin)
    p_val_frame.reset_index(inplace=True)
    p_val_frame.rename(columns={'index':'param', 0 : 'p_val'},inplace=True)
    # # Reformulate pval_save_path using this_fit_type
    # pval_save_pieces = pval_save_path.split('.')
    # pval_save_path = "_".join([
    #         pval_save_pieces[0], 
    #         this_fit_type, 
    #         ])
    # pval_save_path = pval_save_path + '.' + pval_save_pieces[1]
    # p_val_frame.to_csv(pval_save_path)
    return p_val_frame

def return_ll_frame(fit_list, this_fit_type):
    """
    Note: Outs per fit = (res, design_mat, test_ll)
    """
    ll_list = [x[2] for x in fit_list]
    ll_frame = pd.DataFrame(
            dict(
                ll = ll_list,
                fit_num = np.arange(len(ll_list)),
                fit_type = this_fit_type,
                ),
            )

    # # Reformulate ll_save_path using this_fit_type
    # ll_save_pieces = ll_save_path.split('.')
    # ll_save_path = "_".join([
    #         ll_save_pieces[0], 
    #         this_fit_type, 
    #         ])
    # ll_save_path = ll_save_path + '.' + ll_save_pieces[1]
    # ll_frame.to_csv(ll_save_path)
    return ll_frame
