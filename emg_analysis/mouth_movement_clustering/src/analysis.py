"""
Analysis of mouth movement classification

1) Comparison with previous methods
    a) Jenn Li (gapes)
    b) Narendra (gapes + LTPs)

"""

import os
import sys
from glob import glob
from pickle import dump, load

import numpy as np
import tables
import pylab as plt
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Patch
import seaborn as sns

############################################################
############################################################
base_dir = '/media/bigdata/firing_space_plot/emg_analysis/mouth_movement_clustering'
artifact_dir = os.path.join(base_dir, 'artifacts')
plot_dir = os.path.join(base_dir, 'plots')

wanted_artifacts_paths = glob(os.path.join(artifact_dir, '*wanted_artifacts.pkl'))
wanted_artifacts_paths.sort()
wanted_artifacts = [load(open(path, 'rb')) for path in wanted_artifacts_paths] 
basenames = [x['basename'] for x in wanted_artifacts]

############################################################
# Comparison of JL accuracy vs ours
############################################################
JL_accuracy_list = [x['JL_accuracy'] for x in wanted_artifacts]
our_accuracy_list = [x['one_vs_all_accuracy'] for x in wanted_artifacts]
our_accuracy_mean = np.array([np.mean(x) for x in our_accuracy_list])
our_accuracy_std = np.array([np.std(x) for x in our_accuracy_list])

min_vals = np.min([JL_accuracy_list, our_accuracy_mean - our_accuracy_std])
max_vals = np.max([JL_accuracy_list, our_accuracy_mean + our_accuracy_std])

fig, ax = plt.subplots(1,1)
ax.errorbar(JL_accuracy_list, our_accuracy_mean, yerr=our_accuracy_std, fmt='o')
ax.plot([min_vals, max_vals], [min_vals, max_vals], 'k--')
ax.set_xlabel('JL Accuracy')
ax.set_ylabel('Our Accuracy')
ax.set_title('Comparison of Jenn Li vs Our Accuracy\n(One vs All Classifier)')
ax.set_aspect('equal')
plt.savefig(os.path.join(plot_dir, 'JL_vs_our_accuracy.png'))
plt.close()

############################################################
# Comparison of JL accuracy vs ours
############################################################
