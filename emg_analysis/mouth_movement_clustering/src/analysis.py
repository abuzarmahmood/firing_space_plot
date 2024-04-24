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
# Comparison of NM-BSA accuracy vs ours
############################################################

bsa_accuracy_list = [x['bsa_accuracy'] for x in wanted_artifacts]
bsa_f1_list = [x['bsa_f1'] for x in wanted_artifacts]
our_multiclass_accuracy_list = [x['gape_ltp_accuracy'] for x in wanted_artifacts]
our_multiclass_f1_list = [x['gape_ltp_f1'] for x in wanted_artifacts]

our_multiclass_accuracy_mean = np.array([np.mean(x) for x in our_multiclass_accuracy_list])
our_multiclass_accuracy_std = np.array([np.std(x) for x in our_multiclass_accuracy_list])
our_multiclass_f1_mean = np.array([np.mean(x) for x in our_multiclass_f1_list])
our_multiclass_f1_std = np.array([np.std(x) for x in our_multiclass_f1_list])

fig, ax = plt.subplots(1,2)
ax[0].errorbar(
        bsa_accuracy_list, 
        our_multiclass_accuracy_mean, 
        yerr=our_multiclass_accuracy_std, 
        fmt='o')
min_vals = np.min([bsa_accuracy_list, our_multiclass_accuracy_mean - our_multiclass_accuracy_std])
max_vals = np.max([bsa_accuracy_list, our_multiclass_accuracy_mean + our_multiclass_accuracy_std])
ax[0].plot([min_vals, max_vals], [min_vals, max_vals], 'k--')
ax[0].set_xlabel('NM-BSA Accuracy')
ax[0].set_ylabel('Our Accuracy')
ax[0].set_title('Accuracy Comparison\n(Gape + LTP Classifier)')
ax[0].set_aspect('equal')
ax[1].errorbar(
        bsa_f1_list, 
        our_multiclass_f1_mean, 
        yerr=our_multiclass_f1_std, 
        fmt='o')
min_vals = np.min([bsa_f1_list, our_multiclass_f1_mean - our_multiclass_f1_std])
max_vals = np.max([bsa_f1_list, our_multiclass_f1_mean + our_multiclass_f1_std])
ax[1].plot([min_vals, max_vals], [min_vals, max_vals], 'k--')
ax[1].set_xlabel('NM-BSA F1')
ax[1].set_ylabel('Our F1')
ax[1].set_title('F1 Comparison\n(Gape + LTP Classifier)')
ax[1].set_aspect('equal')
fig.suptitle('Single Session\nComparison of NM-BSA vs Our Accuracy\n(Gape + LTP Classifier)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'NM_BSA_vs_our_accuracy_single_session.png'),
            bbox_inches='tight')
plt.close()
