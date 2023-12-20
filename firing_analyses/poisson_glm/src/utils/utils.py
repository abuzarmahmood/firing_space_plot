import os
import json
from pprint import pprint
import numpy as np
from scipy.stats import zscore
import pandas as pd
import sys
from glob import glob
sys.path.append('/media/bigdata/firing_space_plot/firing_analyses/poisson_glm')
##############################
import utils.makeRaisedCosBasis as cb
from utils.generate_tools import gen_stim_history_coupled_design
import utils.glm_ephys_process_utils as process_utils
##############################
from pandas import DataFrame as df
from pandas import concat
from sklearn.model_selection import train_test_split
from scipy.special import gammaln

##############################
## Functions
##############################

def get_unit_spikes_and_regions(file_list):
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
    return spike_list, unit_region_list

def load_dat_from_path_list(file_path_list, save_path):
    file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
    basenames = [os.path.basename(x) for x in file_list]

    spike_list, unit_region_list =\
            get_unit_spikes_and_regions(file_list)

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
    inds = np.array(
            list(product(range(len(spike_list)),range(spike_list[0].shape[0]))))
    fin_inds = []
    for ind in tqdm(inds):
        this_count = nrn_counts[ind[0]]
        for nrn in range(this_count):
            fin_inds.append(np.hstack((ind,nrn)))
    fin_inds = np.array(fin_inds)

    ind_frame = df(fin_inds,columns=['session','taste','neuron'])
    ind_frame.to_csv(os.path.join(save_path,'ind_frame.csv'))

    return spike_list, unit_region_frame, ind_frame, fin_inds

def load_dat_from_save(spike_list_path, save_path):
    spike_inds_paths = sorted(
            glob(os.path.join(spike_list_path,'*_spike_inds.npy')))
    spike_inds_list = [np.load(x) for x in spike_inds_paths]
    spike_list = [process_utils.gen_spike_train(x) for x in spike_inds_list]

    # Load unit_region_frame
    unit_region_frame = pd.read_csv(
            os.path.join(save_path,'unit_region_frame.csv'),index_col=0)

    # Load ind_frame
    ind_frame = pd.read_csv(os.path.join(save_path,'ind_frame.csv'),index_col=0)
    fin_inds = ind_frame.values

    return spike_list, unit_region_frame, ind_frame, fin_inds

def generate_params_dict(fin_save_path):
    # Save run parameters
    bin_width = 1 # ms
    params_dict = dict(
            hist_filter_len = 100, #ms
            stim_filter_len = 300, #ms
            coupling_filter_len = 100, #ms
            bin_width = bin_width, #ms
            hist_filter_len_bin = hist_filter_len // bin_width,
            stim_filter_len_bin = stim_filter_len // bin_width,
            coupling_filter_len_bin = coupling_filter_len // bin_width,
            trial_start_offset = -2000, # ms # No clue what this does
            trial_lims = [1000,4500],
            stim_t = 2000,
            basis_kwargs = dict(
                            n_basis = 10,
                            basis = 'cos',
                            basis_spread = 'log',
                            ),
            n_fits = 5,
            n_max_tries = 20,
            n_shuffles_per_fit = 10,
            )

    params_save_path = os.path.join(fin_save_path, 'fit_params.json')
    with open(params_save_path, 'w') as outf:
        json.dump(params_dict, outf, indent = 4, default = int)
    print('Creating run with following parameters :')
    pprint(params_dict)
    return params_dict

def calc_sta(data, filter_len):
    """
    Calculate the spike triggered average

    args:
        data: n x 1 vector of data
        filter_len: length of filter

    returns:
        sta: filter_len x 1 vector
    """
    inds = np.where(data)[0]
    cut_inds = inds[inds > filter_len]
    sta = np.stack([data[i-filter_len:i] for i in cut_inds]) 
    return np.mean(sta, axis = 0)

def dataframe_to_design_mat(
        data_frame,
        hist_filter_len,
        coupling_filter_len,
        stim_filter_len,
        basis_kwargs,
        ):
    """
    Split data into training and testing sets
    This NEEDS to be done at the design matrix level because
    temporal structure no longer matters then
    """
    coup_cols = [x for x in data_frame.columns if 'coup' in x]
    coup_inds = [int(x.split('_')[-1]) for x in coup_cols]
    glmdata = gen_stim_history_coupled_design(
                    spike_data = data_frame['spikes'].values, 
                    coupled_spikes = data_frame[coup_cols].values.T,
                    coupled_spike_inds = coup_inds,
                    stim_data = data_frame['stim'].values,
                    hist_filter_len = hist_filter_len,
                    coupling_filter_len = coupling_filter_len,
                    stim_filter_len = stim_filter_len,
                    n_basis = basis_kwargs['n_basis'],
                    basis = basis_kwargs['basis'],
                    basis_spread = basis_kwargs['basis_spread'],
                    )
    # Re-add trial_labels and trial_time
    trial_cols = ['trial_labels','trial_time']
    glmdata = pd.concat([glmdata, data_frame[trial_cols]], axis=1)
    glmdata = glmdata.dropna()
    glmdata.reset_index(inplace=True, drop=True)

    # Drop trials which are short
    trial_list = [x[1] for x in list(glmdata.groupby('trial_labels'))]
    trial_lens = [len(x) for x in trial_list]
    med_len = np.median(trial_lens)
    unwanted_trials = [i for i, this_len in enumerate(trial_lens) \
            if this_len != med_len]
    remaining_lens = [x for i,x in enumerate(trial_lens) \
            if i not in unwanted_trials]
    assert all([[x==y for x in remaining_lens] for y in remaining_lens]), \
            'Trial lengths are not equal'
    glmdata = glmdata.loc[~glmdata.trial_labels.isin(unwanted_trials)]
    glmdata.reset_index(inplace=True, drop=True)
    glmdata['intercept'] = 1
    return glmdata

def return_train_test_split(data_frame, test_size = 0.2, random_state = None):
    if random_state is None:
        random_state = np.random.randint(0,100)
    train_dat, test_dat = train_test_split(
            data_frame, test_size=test_size, random_state=random_state)
    return train_dat.sort_index(), test_dat.sort_index() 

class fit_handler():
    def __init__(self, param_names, param_values):
        self.params = pd.DataFrame(dict(params = param_names, values = param_values))

def calc_loglikelihood(actual_design_mat, res):
    """
    Generate shuffles and repeat testing
    Note: No need to refit as we're simply showing that destroying different
    parts of the predictors destroys the model's ability to predict actual data
    i.e. model has learned TRIAL-SPECIFIC features
    """
    trial_sh_design_mat = gen_trial_shuffle(actual_design_mat)
    circ_sh_design_mat = gen_circular_shuffle(actual_design_mat)
    rand_sh_design_mat = gen_random_shuffle(actual_design_mat)

    # Get train-test splits
    actual_train_dat, actual_test_dat = return_train_test_split(actual_design_mat)
    trial_sh_train_dat, trial_sh_test_dat = return_train_test_split(trial_sh_design_mat)
    circ_sh_train_dat, circ_sh_test_dat = return_train_test_split(circ_sh_design_mat)
    rand_sh_train_dat, rand_sh_test_dat = return_train_test_split(rand_sh_design_mat)

    # Calculate log-likelihoods
    actual_test_pred = res.predict(actual_test_dat[res.params.index])
    actual_test_ll = poisson_ll(actual_test_pred, actual_test_dat['spikes'].values)
    actual_test_ll = np.round(actual_test_ll, 2)

    trial_sh_test_pred = res.predict(trial_sh_test_dat[res.params.index])
    trial_sh_test_ll = poisson_ll(trial_sh_test_pred, actual_test_dat['spikes'].values)
    trial_sh_test_ll = np.round(trial_sh_test_ll, 2)

    circ_sh_test_pred = res.predict(circ_sh_test_dat[res.params.index])
    circ_sh_test_ll = poisson_ll(circ_sh_test_pred, actual_test_dat['spikes'].values)
    circ_sh_test_ll = np.round(circ_sh_test_ll, 2)

    rand_sh_test_pred = res.predict(rand_sh_test_dat[res.params.index])
    rand_sh_test_ll = poisson_ll(rand_sh_test_pred, actual_test_dat['spikes'].values)
    rand_sh_test_ll = np.round(rand_sh_test_ll, 2)

    return actual_test_ll, trial_sh_test_ll, circ_sh_test_ll, rand_sh_test_ll

def process_glm_res(
        res = None, 
        filter_values = None,
        filter_len = 200, 
        n_basis = 10,
        basis = 'cos',
        basis_spread = 'log',
        param_key = 'hist',):
    if res is not None:
        lag_params =  res.params
        lag_params = lag_params[[x for x in lag_params.index if param_key in x]]
        if basis == 'linear':
            lag_params.index = [int(x.split(param_key)[1]) for x in lag_params.index]
        elif basis == 'cos':
            cos_basis = cb.gen_raised_cosine_basis(
                    filter_len,
                    n_basis = n_basis,
                    spread = basis_spread,)
            lag_params = lag_params[None,:].dot(cos_basis).flatten()
    else:
        lag_params = filter_values
        if basis == 'linear':
            pass
        elif basis == 'cos':
            cos_basis = cb.gen_raised_cosine_basis(
                    filter_len,
                    n_basis = n_basis,
                    spread = basis_spread,)
            lag_params = lag_params[None,:].dot(cos_basis).flatten()
    return lag_params

def gen_data_frame(
        spike_data, 
        coupled_spikes,
        stim_vec,
        stim_filter_len,
        trial_start_offset = 0,
        ):
    """
    Convert array data to a pandas data frame

    Inputs:

        Outputs:
            data_frame: pandas data frame with columns:
                spikes: spike data
            coup_{i}: coupled spike data for neuron i
            stim: stimulus data
    """

    stacked_data = np.concatenate([
        spike_data[None,:], coupled_spikes, stim_vec[None,:]], 
                                  axis=0)
    labels = ['spikes',*[f'coup_{i}' for i in range(len(coupled_spikes))], 'stim']
    data_frame = pd.DataFrame(
            data = stacked_data.T,
            columns = labels)
    trial_starts = np.where(stim_vec[:-stim_filter_len])[0]
    trial_starts = trial_starts + trial_start_offset
    dat_len = len(spike_data)
    trial_labels = np.zeros(dat_len)
    trial_time = np.zeros(dat_len)
    counter = 0
    for i in range(len(trial_starts)):
        if i != len(trial_starts)-1:
            trial_labels[trial_starts[i]:trial_starts[i+1]] = counter
            counter +=1
            trial_time[trial_starts[i]:trial_starts[i+1]] = \
                    np.arange(0 , trial_starts[i+1] - trial_starts[i])
        else:
            trial_labels[trial_starts[i]:dat_len] = counter
            trial_time[trial_starts[i]:dat_len] = \
                    np.arange(0, dat_len - trial_starts[i])

    data_frame['trial_labels'] = trial_labels
    data_frame['trial_time'] = trial_time
    data_frame = data_frame.astype('int')
    return data_frame

############################################################
# Shuffling 
############################################################

def gen_trial_shuffle(data_frame, dv = 'spikes'):
    """
    Mismatch trials between dv and iv
    """
    spike_dat = data_frame[dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x != dv]]
    unique_trials = iv_dat['trial_labels'].unique()
    trial_map = dict(zip(unique_trials, np.random.permutation(unique_trials)))
    iv_dat.loc[:,'trial_labels'] = [trial_map[x] for x in iv_dat['trial_labels']]
    iv_dat = iv_dat.sort_values(by = ['trial_labels', 'trial_time'])
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat], axis=1)
    return out_frame

def gen_circular_shuffle(data_frame, dv = 'spikes'):
    """
    Shuffle timebins across trials (i.e. maintain the position of time bins but
                                    change trial indices)
    """
    spike_dat = data_frame[dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x != dv]]
    time_grouped_dat = [x[1] for x in list(iv_dat.groupby('trial_time'))]
    for this_dat in time_grouped_dat:
        this_dat['trial_labels'] = np.random.permutation(this_dat['trial_labels'])
    iv_dat = pd.concat(time_grouped_dat)
    iv_dat = iv_dat.sort_values(by = ['trial_labels', 'trial_time'])
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat([spike_dat.reset_index(drop=True), iv_dat], axis=1)
    return out_frame

def gen_random_shuffle(data_frame, dv = 'spikes'):
    """
    Randomly shuffled IV and DV separately
    """
    trial_cols = ['trial_labels','trial_time']
    spike_dat = data_frame[dv]
    trial_dat = data_frame[trial_cols]
    rm_cols = trial_cols + [dv]
    iv_dat = data_frame[[x for x in data_frame.columns if x not in rm_cols]]
    iv_dat = iv_dat.sample(frac = 1, replace=False)
    iv_dat.reset_index(inplace=True, drop=True)
    out_frame = pd.concat(
            [
                spike_dat.reset_index(drop=True), 
                iv_dat, 
                trial_dat.reset_index(drop=True)
                ], 
            axis=1)
    return out_frame

def poisson_ll(lam, k):
    """
    Poisson log likelihood
    # Note: This has been tested against scipy.stats.poisson.logpmf

    Inputs:
        lam: lambda parameter of poisson distribution
        k: observed counts

    Outputs:
        ll: log likelihood
    """
    lam += 1e-10 # To ensure there is no log(0)
    assert len(lam) == len(k), 'lam and k must be same length'
    assert all(lam > 0), 'lam must be non-negative'
    assert all(k >= 0), 'k must be non-negative'
    return np.sum(k*np.log(lam) - lam - gammaln(k+1))
