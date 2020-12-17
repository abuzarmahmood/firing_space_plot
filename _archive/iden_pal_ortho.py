"""
Decompose population firing across tastes in the 
identity and palatability epochs to determine
overlap between the two subspaces
"""

## Import required modules
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
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from sklearn.decomposition import PCA as pca
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import zscore

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *

dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM11/AM11_4Tastes_191030_114043_copy/')
dat.firing_rate_params = dat.default_firing_params 

dat.extract_and_process()
dat.separate_laser_data()

# Extract firing according to the following windows
dt = 25
stimulus = 2000
identity_center = stimulus + 500
palatability_center = stimulus + 1000
window_radius = 125

iden_firing = dat.all_normalized_firing[...,(identity_center - window_radius)//dt:(identity_center + window_radius)//dt]
pal_firing = dat.all_normalized_firing[...,(palatability_center - window_radius)//dt:(palatability_center + window_radius)//dt]

def imshow(array):
    plt.imshow(array,interpolation='nearest',aspect='auto')

iden_firing_long = np.reshape(iden_firing,(iden_firing.shape[0],-1))
pal_firing_long = np.reshape(pal_firing,(pal_firing.shape[0],-1))

n_components = 3
red_iden_obj = pca(n_components = n_components).fit(iden_firing_long.T)
red_pal_obj = pca(n_components = n_components).fit(pal_firing_long.T)

# Plot eigenvectors for both states
fig,ax = plt.subplots(2)
ax[0].imshow(red_iden_obj.components_)
ax[1].imshow(red_pal_obj.components_)
plt.show()

# Matrix multiplication of both martices is dot product
# of all pairs of eigenvectors
orthogonal_distance = np.matmul(red_iden_obj.components_,
                        red_pal_obj.components_.T)

#fig,ax=plt.subplots(3)
#plt.sca(ax[0])
#imshow(red_iden_obj.components_)
#plt.sca(ax[1])
#imshow(np.flip(red_iden_obj.components_.T,axis=-1))
#plt.sca(ax[2])
imshow(orthogonal_distance)
plt.colorbar()
plt.show()

# Project identity and palatability into eachothers subspaces
# and plot results
iden_obj_iden = red_iden_obj.transform(iden_firing_long.T).T
pal_obj_iden = red_pal_obj.transform(iden_firing_long.T).T
iden_obj_pal = red_iden_obj.transform(pal_firing_long.T).T
pal_obj_pal = red_pal_obj.transform(pal_firing_long.T).T

fig,ax = plt.subplots(4)
plt.sca(ax[0])
imshow(iden_obj_iden)
plt.sca(ax[1])
imshow(pal_obj_iden)
plt.sca(ax[2])
imshow(iden_obj_pal)
plt.sca(ax[3])
imshow(pal_obj_pal)
plt.show()

variance_thresh = 0.8
needed_components = np.sum(
                        np.cumsum(
                            pca_object.explained_variance_ratio_) < variance_thresh)+1
pca_object = pca(n_components = needed_components).fit(zscore_concat_firing_long.T)
