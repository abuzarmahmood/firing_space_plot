
## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
#import h5py
#import easygui
import scipy
from scipy.signal import spectrogram
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from itertools import product
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import zscore
import glob
from collections import namedtuple
from scipy.signal import convolve

os.chdir('/media/bigdata/firing_space_plot')

test_dat = pd.read_pickle('test_anova_frame.pkl')


# Perform 2-way ANOVA to look at differences in taste and trial_bin
taste_trial_anova = test_dat.anova(dv='power',
                                    between = ['trial_bin','taste'])

taste_trial_anova = [\
    [dat.loc[dat.band == band_num].anova(dv = 'power', \
            between= ['trial_bin','taste'])[['Source','p-unc','np2']] \
            for band_num in np.sort(dat.band.unique())] \
        for dat in mean_band_df]

