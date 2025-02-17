"""
Script for optimizing GLM hyperparameters.
"""

import os
import sys
import json
from glob import glob
import numpy as np
from skopt import gp_minimize
from skopt import callbacks, load
from skopt.callbacks import CheckpointSaver
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from datetime import datetime


base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/src'
save_path = os.path.join(
        os.path.dirname(base_path),
        'artifacts'
        )
sys.path.append(base_path)

############################################################
def parallelize(func, iterator, n_jobs = 8):
    return Parallel(n_jobs = n_jobs)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def obj_func(
        args,
        #hist_filter_len,
        #stim_filter_len,
        #coupling_filter_len,
        #n_basis_funcs,
        ):

    # Make sure input params are ints
    hist_filter_len = int(args[0])
    stim_filter_len = int(args[1])
    coupling_filter_len = int(args[2])
    n_basis_funcs = int(args[3])
    # hist_filter_len = int(hist_filter_len)
    # stim_filter_len = int(stim_filter_len)
    # coupling_filter_len = int(coupling_filter_len)
    # n_basis_funcs = int(n_basis_funcs)

    # Get fit params
    json_template_path = os.path.join(base_path, 'params', 'template_fit_params.json')
    params_dict = json.load(open(json_template_path))

    # hist_filter_len = 50
    # stim_filter_len = 100
    # coupling_filter_len = 75
    # n_basis_funcs = 5

    bin_width = params_dict['bin_width']

    hist_filter_len_bin = int(hist_filter_len/bin_width)
    stim_filter_len_bin = int(stim_filter_len/bin_width)
    coupling_filter_len_bin = int(coupling_filter_len/bin_width)

    # Write to params_dict
    params_dict['hist_filter_len'] = hist_filter_len
    params_dict['stim_filter_len'] = stim_filter_len
    params_dict['coupling_filter_len'] = coupling_filter_len
    params_dict['hist_filter_len_bin'] = hist_filter_len_bin
    params_dict['stim_filter_len_bin'] = stim_filter_len_bin
    params_dict['coupling_filter_len_bin'] = coupling_filter_len_bin
    params_dict['basis_kwargs']['n_basis'] = n_basis_funcs

    # Write to json
    with open(json_template_path, 'w') as json_file:
        json.dump(params_dict, json_template_path, indent=4)

    # Get run params
    run_params_path = os.path.join(base_path, 'params', 'run_params.json')
    run_params = json.load(open(run_params_path))
    input_run_ind = run_params['run_ind']
    fraction_used = run_params['fraction_used']
    fin_save_path = os.path.join(save_path, f'run_{input_run_ind:03}')

    # If save path exists, remove it and remake it
    if os.path.exists(fin_save_path):
        os.system(f'rm -r {fin_save_path}')
    os.mkdir(fin_save_path)

    # Run script
    script_path = os.path.join(base_path, 'glm_ephys_process.py')

    with open(script_path, 'r') as f:
        exec(f.read())

    ############################################################
    # Get aic results
    aic_file_list = glob(os.path.join(fin_save_path,'*crit.json'))

    # Load aic results
    aic_results = [json.load(open(f)) for f in aic_file_list]
    aic_results = [x['aic'] for x in aic_results]
    mean_aic = np.mean(aic_results)

    return mean_aic

# https://github.com/scikit-optimize/scikit-optimize/issues/388#issuecomment-307483219
# Have to keep n_calls > n_random_starts for partial dependence plots

if __name__ == '__main__':
    # keyword arguments will be passed to `skopt.dump`
    optim_out_dir = os.path.join(save_path, 'optimization_out')
    if not os.path.exists(optim_out_dir):
        os.mkdir(optim_out_dir)

    checkpoint_path = os.path.join(optim_out_dir, 'checkpoint.pkl')
    checkpoint_saver = CheckpointSaver(checkpoint_path, compress=9) 
    log_path = os.path.join(optim_out_dir, 'log.txt')

    dim_ranges = [
            (50, 400), # hist_filter_len
            (50, 400), # stim_filter_len
            (50, 400), # coupling_filter_len
            (5, 20), # n_basis_funcs
            ]

    total_calls = 10
    for call in range(total_calls):
        if os.path.exists(checkpoint_path):
            res = load(checkpoint_path)
            x0 = res.x_iters
            y0 = res.func_vals
            
            # Write out updates to log file
            iters = len(y0)
            with open(log_path, 'w') as f:
                f.write(f'Last save time: {datetime.now()}\n')
                f.write(f'Iteration {iters} complete\n')
                f.write(f'x: {x0}\n')
                f.write(f'y: {y0}\n')

            gp_minimize(
                        obj_func,            # the function to minimize
                        dim_ranges,    # the bounds on each dimension of x
                        x0= x0,          # the starting point
                        y0= y0,          # the starting point
                        acq_func="LCB",     # the acquisition function (optional)
                        n_calls=2,         # number of evaluations of f including at x0
                        n_initial_points=1,  # the number of random initial points
                        callback=[checkpoint_saver],
                        # a list of callbacks including the checkpoint saver
                        random_state=777,
                        n_jobs = 1)
        else:
            x0 = [200, 200, 200, 15]
            gp_minimize(
                        obj_func,            # the function to minimize
                        dim_ranges,    # the bounds on each dimension of x
                        x0= x0,          # the starting point
                        acq_func="LCB",     # the acquisition function (optional)
                        n_calls=2,         # number of evaluations of f including at x0
                        n_initial_points=1,  # the number of random initial points
                        callback=[checkpoint_saver],
                        # a list of callbacks including the checkpoint saver
                        random_state=777,
                        n_jobs = 1)

        # res = load('./checkpoint.pkl')
        # res.fun
