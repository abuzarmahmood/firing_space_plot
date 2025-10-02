"""
Run loop over all data files in a directory and fit models with multiple states to each one
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import argparse
from pathlib import Path
import pandas as pd
from pprint import pprint as pp
import pymc as pm
from tqdm import tqdm
import cloudpickle

from pytau.changepoint_model import (
    SingleTastePoisson,
    advi_fit,
    dpp_fit
)
from pytau.changepoint_io import DatabaseHandler, FitHandler

def load_data(data_path):
    """
    Load multidimensional timeseries data from file.
    
    Args:
        data_path (str): Path to data file (supports .npy, .pkl, .npz)
        
    Returns:
        data (ndarray): Shape (n_dims, n_timepoints) - the loaded data
        metadata (dict): Any additional metadata from the file
    """
    data_path = Path(data_path)
    metadata = {}
    
    if data_path.suffix == '.npy':
        data = np.load(data_path)
    elif data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, dict):
                data = loaded.get('data', loaded.get('timeseries', None))
                metadata = {k: v for k, v in loaded.items() if k not in ['data', 'timeseries']}
            else:
                data = loaded
    elif data_path.suffix == '.npz':
        loaded = np.load(data_path)
        data = loaded.get('data', loaded.get('timeseries', loaded[loaded.files[0]]))
        metadata = {k: loaded[k] for k in loaded.files if k not in ['data', 'timeseries']}
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    if data is None:
        raise ValueError("Could not find data in the loaded file")
    
    # Ensure data is in the correct shape (n_dims, n_timepoints)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2:
        # If n_timepoints > n_dims, assume data needs to be transposed
        if data.shape[0] > data.shape[1]:
            print(f"Transposing data from shape {data.shape} to {data.shape[::-1]}")
            data = data.T
    else:
        raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")
    
    n_dims, n_timepoints = data.shape
    print(f"Loaded data with {n_dims} dimensions and {n_timepoints} timepoints")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    return data, metadata

def advi_fit(model, fit, samples, convergence_tol=None):
    """Convenience function to perform ADVI fit on model

    Args:
        model (pymc model): model object to run inference on
        fit (int): Number of iterationst to fit the model for
        samples (int): Number of samples to draw from fitted model

    Returns:
        model: original model on which inference was run,
        approx: fitted model,
        lambda_stack: array containing lambda (emission) values,
        tau_samples,: array containing samples from changepoint distribution
        model.obs.observations: processed array on which fit was run
    """

    if convergence_tol is not None:
        callbacks = [pm.callbacks.CheckParametersConvergence(
            tolerance=convergence_tol)]
        print("Using convergence callback with tolerance:", convergence_tol)
    else:
        callbacks = None
    with model:
        inference = pm.ADVI("full-rank")
        approx = pm.fit(n=fit, method=inference, callbacks=callbacks)
        idata = approx.sample(draws=samples)

    return model, approx, idata


##############################

base_dir = '/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions'
artifacts_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')
data_dir = os.path.join(base_dir, 'data', 'Changepoint_data')

plot_dir = Path(plot_dir)
artifacts_dir = Path(artifacts_dir)

# 1. Load data
data_list = os.listdir(data_dir)
# Only keep files with "boolean" in the name
data_list = [f for f in data_list if 'boolean' in f]

for this_file in tqdm(data_list):
    # break    
    basename = this_file.split('.')[0]

    raw_data, raw_metadata = load_data(
            os.path.join(data_dir,
            [x for x in data_list if 'gape' in x][0]
                         )
            )

    bin_size = 250
    # raw_data shape: trials x time
    binned_data = np.reshape(raw_data, (raw_data.shape[0], raw_data.shape[1] // bin_size, bin_size)).sum(axis=2)


    fit_data = np.transpose(binned_data, (1, 0))[None,:,:]  # Now shape is time x neurons

    n_state_vec = np.arange(2, 8)

    save_dict_list = []
    for this_states in n_state_vec:
        # break
        model = SingleTastePoisson(fit_data, n_states=int(this_states)).generate_model()
        with model:
            inference = pm.ADVI("full-rank")
            approx = pm.fit(n=50_000, method=inference)
            idata = approx.sample(draws=2000)
        tau_samples = idata.posterior['tau'].values[0]
        save_dict = {
            'model': model,
            'approx': approx,
            'tau_samples': tau_samples,
            'fit_data': fit_data,
            'n_states': this_states,
            'bin_size': bin_size,
            'metadata': raw_metadata
        }
        save_dict_list.append(save_dict)
    fin_save_dict = dict(zip(n_state_vec, save_dict_list))
    # Save to artifacts directory
    with open(artifacts_dir / f'{basename}_fit_dict.pkl', 'wb') as f:
        pickle.dump(fin_save_dict, f)

    all_elbo_values = [val['approx'].hist[-1] for val in save_dict_list]
    all_tau_samples = [val['tau_samples'] for val in save_dict_list]

    fig, ax = plt.subplots(2+len(n_state_vec), 1, figsize=(8, 4+2*len(n_state_vec)), sharex=False)
    ax[0].plot(n_state_vec, all_elbo_values, '-o')
    ax[0].set_xlabel('Number of States')
    im = ax[1].imshow(binned_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[1].set_xlabel('Trials')
    ax[1].set_ylabel('Time Bins')
    ax[1].set_xlim(0, binned_data.shape[0])
    for i, this_tau in enumerate(all_tau_samples):
        for this_change in this_tau.T:
            ax[2+i].hist(this_change.flatten(), bins=30, density=True) 
            ax[2+i].set_ylabel(f'States={n_state_vec[i]}')
            ax[2+i].set_xlim(0, binned_data.shape[0])
    fig.suptitle(f'Fit results for {basename}')
    # plt.colorbar(im, ax=ax, label='Binned Counts')
    plt.tight_layout()
    fig.savefig(plot_dir / f'{basename}_binned_data.png')
    plt.close(fig)

##############################
# Also fit dpp model
for this_file in tqdm(data_list):
    # break    
    basename = this_file.split('.')[0]

    raw_data, raw_metadata = load_data(
            os.path.join(data_dir,
            [x for x in data_list if 'gape' in x][0]
                         )
            )

    bin_size = 250
    # raw_data shape: trials x time
    binned_data = np.reshape(raw_data, (raw_data.shape[0], raw_data.shape[1] // bin_size, bin_size)).sum(axis=2)


    fit_data = np.transpose(binned_data, (1, 0))[None,:,:]  # Now shape is time x neurons
    model = SingleTastePoisson(fit_data, n_states=int(this_states)).generate_model()
    dpp_trace = dpp_fit(model, n_chains = 24, n_cores = 24, use_numpyro=False) 

    dpp_tau_samples = dpp_trace['tau']

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    im = ax[0].imshow(binned_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[1].hist(dpp_tau_samples.flatten(), bins=50, density=True)
    ax[1].set_ylabel(f'DPP Tau Samples')
    ax[1].set_xlabel('Trials')
    fig.suptitle(f'DPP Fit results for {basename}')
    plt.tight_layout()
    fig.savefig(plot_dir / f'{basename}_dpp_binned_data.png')
    plt.close(fig)

    # Dump to artifacts directory
    save_dict = {
        'model': model,
        'dpp_trace': dpp_trace,
        'fit_data': fit_data,
        'bin_size': bin_size,
        'metadata': raw_metadata
    }
    with open(artifacts_dir / f'{basename}_dpp_fit_dict.pkl', 'wb') as f:
        cloudpickle.dump(save_dict, f)
