"""
Perform granger causality anaylsis on all data and save results to file
:
# 1) Load data
# 2) Preprocess data:
#     a) Remove trials with artifacts
#     b) Detrend single-trial data
#     c) Remove temporal mean from single-trial data
#     d) Dvidide by temporal standard deviation
#     e) Subtract mean across trials from each trial (for each timepoint)
#     f) Divide by standard deviation across trials (for each timepoint)
# 3) Perform Augmented Dickey-Fuller test on each channel to check for stationarity
# 4) Perform Granger Causality test on each channel pair
# 5) Test for good fitting by checking that residuals are white noise
# 6) Calculate significance of granger causality by shuffling trials
# 7) Plot results
#
# References:
# Ding, Mingzhou, et al. “Short-Window Spectral Analysis of Cortical Event-Related Potentials by Adaptive Multivariate Autoregressive Modeling: Data Preprocessing, Model Validation, and Variability Assessment.” Biological Cybernetics, vol. 83, no. 1, June 2000, pp. 35–45, https://doi.org/10.1007/s004229900137.

"""

############################################################
## Imports
############################################################

import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
sys.path.append(ephys_data_dir)
from ephys_data import ephys_data
import numpy as np
import pylab as plt
from scipy.signal import detrend
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm, trange
from joblib import Parallel, delayed
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity

sys.path.append('/media/bigdata/firing_space_plot/lfp_analyses/granger_causality')
from granger_data_test import parallelize, lfp_preprocessing
