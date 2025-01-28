## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt4Agg')
import tables
import easygui
import scipy
import json
import glob
import numpy as np
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import pingouin
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA as pca
from sklearn.cluster import KMeans as kmeans
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import ttest_rel 

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

data_dir = '/media/bigdata/Abuzar_Data/AM17/AM17_4Tastes_191126_084934'
data = ephys_data(data_dir = data_dir) 
data.firing_rate_params = dict(zip(\
    ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
    ('conv',25,250,1,25e-3,1e-3)))
data.get_unit_descriptors()
data.get_spikes()
data.get_firing_rates()
time_vec = np.vectorize(np.int)(np.linspace(0,7000, data.normalized_firing.shape[-1]))

#json_file = glob.glob(data_dir+'/*json')[-1]
#file_params = json.load(open(json_file,'r'))
mean_firing = np.mean(data.normalized_firing,axis=2)

step = 20
plt.plot(mean_firing[:,4].T)
plt.xticks(np.arange(len(time_vec))[::step], time_vec[::step], rotation = 'vertical')
plt.show()
