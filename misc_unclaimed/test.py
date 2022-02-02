import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from joblib import Parallel,delayed
import pingouin
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import zscore
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM21/AM21_4Tastes_200303_102607')
#dat.firing_rate_params = dict(zip(\
#        ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
#        ('baks',25,250,1,1e-3,1e-3)))
dat.firing_rate_params = dict(zip(\
        ('type', 'step_size','window_size','dt', 'baks_dt'),
        ('baks',25,250,1,1e-3)))
dat.get_unit_descriptors()
dat.get_spikes()
dat.get_firing_rates()
dat.separate_laser_spikes()

dat.firing_rate_method_selector()(dat.spikes[0])
