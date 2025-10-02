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

from pytau.changepoint_model import (
    SingleTastePoisson,
    advi_fit,
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

for this_file in data_list:
    
    basename = this_file.split('.')[0]

    raw_data, raw_metadata = load_data(
            os.path.join(data_dir,
            [x for x in data_list if 'gape' in x][0]
                         )
            )

    bin_size = 250
    # raw_data shape: trials x time
    binned_data = np.reshape(raw_data, (raw_data.shape[0], raw_data.shape[1] // bin_size, bin_size)).sum(axis=2)


    fig, ax = plt.subplots()
    im = ax.imshow(binned_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_title(f'Binned Data: {basename}')
    ax.set_xlabel('Trials')
    ax.set_ylabel('Time Bins')
    plt.colorbar(im, ax=ax, label='Binned Counts')
    fig.savefig(plot_dir / f'{basename}_binned_data.png')
    plt.close(fig)

    fit_data = np.transpose(binned_data, (1, 0))[None,:,:]  # Now shape is time x neurons

    n_state_vec = np.arange(1, 7)

    save_dict_list = []
    for this_states in n_state_vec:
        model = SingleTastePoisson(fit_data, n_states=int(this_states)).generate_model()
        with model:
            inference = pm.ADVI("full-rank")
            approx = pm.fit(n=50_000, method=inference)
            idata = approx.sample(draws=2000)
        # model, approx, idata = advi_fit(model, fit=50_000, samples=2000)
        # model, approx, lambda_samples , tau_samples, _ = outs
        trace = approx.sample(draws=2000)
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
    fin_save_dict = dict(zip(n_state_vec, save_dict_list))
