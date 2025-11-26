from skopt import plots
from skopt import load
import os
import pylab as plt

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

base_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/'
src_path = os.path.join(base_path,'src')
sys.path.append(src_path)

from glm_optimize import parallelize, obj_func

save_path = os.path.join(base_path, 'artifacts')
plot_dir = os.path.join(save_path,'plots')
fin_plot_dir = os.path.join(save_path,'optimization_plots')
optim_path = os.path.join(save_path,'optimization_out')
checkpoint_path = os.path.join(optim_path,'checkpoint.pkl')

if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

# Load optimization results
res = load(checkpoint_path)

# Plot convergence
plots.plot_convergence(res,)
plt.savefig(os.path.join(fin_plot_dir,'convergence.png'))
plt.close()

# Plot partial dependence
plots.plot_objective(res, 
                      dimensions = ['hist_filter_len',
                                     'stim_filter_len',
                                     'coupling_filter_len',
                                     #'n_basis_funcs'
                                    ]
                       )
plt.tight_layout()
plt.savefig(os.path.join(fin_plot_dir,'objective.png'))
plt.close()

# Plot evaluations
plots.plot_evaluations(res,
                      dimensions = ['hist_filter_len',
                                     'stim_filter_len',
                                     'coupling_filter_len',
                                     #'n_basis_funcs'
                                    ]
                       )
plt.tight_layout()
plt.savefig(os.path.join(fin_plot_dir,'evaluations.png'))
plt.close()

# Ignoring other variables (rather than accounting for them),
# how does the objective change with each variable?

fig, ax = plt.subplots(1,3,figsize = (20,5))
x_dat = np.stack(res.x_iters)
y_dat = res.func_vals
for i in range(3):
    ax[i].scatter(x_dat[:,i],y_dat)
    ax[i].set_xlabel(res.space.dimensions[i].name)
    ax[i].set_ylabel('Objective')
    ax[i].set_ylim([1400,2000])
plt.savefig(os.path.join(fin_plot_dir,'objective_vs_params.png'))
plt.close()
