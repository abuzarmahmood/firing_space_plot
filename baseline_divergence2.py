

#  ____                 _ _              _____  _       
# |  _ \               | (_)            |  __ \(_)      
# | |_) | __ _ ___  ___| |_ _ __   ___  | |  | |___   __
# |  _ < / _` / __|/ _ \ | | '_ \ / _ \ | |  | | \ \ / /
# | |_) | (_| \__ \  __/ | | | | |  __/ | |__| | |\ V / 
# |____/ \__,_|___/\___|_|_|_| |_|\___| |_____/|_| \_/  ergence
#

# 
######################### Import dat ish #########################
import tables
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

from scipy.spatial import distance_matrix as dist_mat
from scipy.stats.mstats import zscore
from scipy.stats import pearsonr
from scipy import signal

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import seaborn as sns
import glob

from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *
import multiprocessing as mp

from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu
import scipy

import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels



#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#
dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    
corr_dat = pd.DataFrame()

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    #for start, end in zip(range(2000,5500,500), range(2500,5500,500)):
    data.correlation_params = dict(zip(['stimulus_start_time', 'stimulus_end_time',
                                        'baseline_start_time', 'baseline_end_time',
                                        'shuffle_repeats', 'accumulated'],
                                       [2000, 4000, 0, 2000, 100, True]))
    data.get_correlations()
    data.get_dataframe()
        
    corr_dat = pd.concat([corr_dat, data.data_frame])
    print('file %i' % file)

#############################
sns.boxplot(x='taste',y='rho',hue='shuffle',data= corr_dat.query('laser == False'))
cw_lm=ols('rho ~ taste + shuffle', data=corr_dat.query('laser == False')).fit() #Specify C for Categorical
print(sm.stats.anova_lm(cw_lm, typ=2))
 
for file in [6]:#range(1,7):
    g = sns.FacetGrid(corr_dat.query('laser == False') ,
                      col = 'taste', 
                      hue = 'shuffle', 
                      sharey = 'row')
    #g.set(ylim=(0,None)
    g.map(sns.regplot,'stimulus_end','rho', 
          x_estimator = np.mean, x_ci = 'sd').add_legend()
    #g.fig.suptitle('FILE %i' % file)
    #g.fig.suptitle('All files')
    g.savefig('acc_dist_stim_window_move_2000ms_all_JY+NM.png')
    plt.close('all')

##############################
# Run stats on that ish
    
# NM laser on vs off
anova_dat = corr_dat.query('file < 6').filter(items = ['rho','taste','shuffle','laser'])    
anova_dat = anova_dat.assign(rho_2 = lambda x: (x.rho)**2)
#sns.swarmplot(x='taste',y='rho_2',hue='shuffle',data=anova_dat, dodge=True)
#sns.swarmplot(x='laser',y='rho_2', hue = 'shuffle',data=anova_dat, dodge=True)
sns.swarmplot(x='laser',y='rho_2', data=anova_dat.query('shuffle == False'), dodge=True)
plt.suptitle('Accumulated Distance Correlation, NM')
plt.title('n = %i, n = %i' % (anova_dat.query('shuffle==False and laser == True').shape[0], anova_dat.query('shuffle==True and laser == True').shape[0]))

mannwhitneyu(anova_dat.query('shuffle == False and laser == True').rho_2,anova_dat.query('shuffle == False and laser == False').rho_2)
f_oneway(anova_dat.query('shuffle == False and laser == True').rho_2,anova_dat.query('shuffle == False and laser == False').rho_2)
scipy.stats.ks_2samp(anova_dat.query('shuffle == False and laser == True').rho_2,anova_dat.query('shuffle == False and laser == False').rho_2)
scipy.stats.ttest_ind(anova_dat.query('shuffle == False and laser == True').rho_2,anova_dat.query('shuffle == False and laser == False').rho_2,
                      equal_var = False)

## Shuffle comparison
scipy.stats.ks_2samp(corr_dat.query('shuffle == True').rho,corr_dat.query('shuffle == False').rho)
mannwhitneyu(corr_dat.query('shuffle == True').rho,corr_dat.query('shuffle == False').rho)

### Difference across tastes
taste_comp = corr_dat.query('laser == False and shuffle == False').filter(items = ['rho','taste','shuffle','laser','file'])
taste_comp = taste_comp.assign(rho_2 = lambda x: (x.rho)**2)
formula = 'rho_2 ~ taste'
model = ols(formula, taste_comp).fit()
aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
print(aov_table)

sns.swarmplot(x='taste',y='rho', data=taste_comp, dodge=True)

### Differences in baseline across tastes
# Show that there are significant differences in distances of post-stimulus firing 
# but not baseline firing
dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

baseline_inds = range(80)
stimulus_inds = range(80,160)

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    base_stim = np.concatenate((data.all_normal_off_firing[:,:,baseline_inds],
                               data.all_normal_off_firing[:,:,stimulus_inds]),
                               axis = 1)
    base_stim_mean = np.mean(base_stim,axis=2)
    base_stim_dists = dist_mat(base_stim_mean.T,base_stim_mean.T)
    base_dists = base_stim_dists[range(60),0]
    stim_dists = base_stim_dists[range(60,120),0]

###
r_sq = []
r_sq_sh = []
for taste in range(4):
    r_sq.append(corr_dat.query('taste == %i and shuffle == False and laser == False' % taste).rho**2)
    r_sq_sh.append(corr_dat.query('taste == %i and shuffle == True and laser == False' % taste).rho**2)
    
f_oneway(r_sq[0],r_sq[1],r_sq[2],r_sq[3])#,r_sq_sh[0],r_sq_sh[1],r_sq_sh[2],r_sq_sh[3])

##
formula = 'rho_2 ~ shuffle + laser + shuffle:laser'
model = ols(formula, anova_dat).fit()
aov_table = statsmodels.stats.anova.anova_lm(model, typ=2)
print(aov_table)

## MDS for clustering trajectories from distances
from sklearn.manifold import MDS as mds
pre_dists = data.off_corr['pre_dists']
stim_dists = data.off_corr['stim_dists']

taste = 0
pre_clust = mds(n_components=2,dissimilarity='precomputed').fit_transform(pre_dists[taste])
stim_clust = mds(n_components=2,dissimilarity='precomputed').fit_transform(stim_dists[taste])

fig, ax = plt.subplots(ncols=1,nrows=2,constrained_layout=True)
labels = range(pre_clust.shape[0])
ax[0].scatter(pre_clust[:,0],pre_clust[:,1])
for i, txt in enumerate(labels):
    ax[0].annotate(txt, (pre_clust[i,0], pre_clust[i,1]))
ax[0].set_title('Pre-Stim')

labels = range(stim_clust.shape[0])    
ax[1].scatter(stim_clust[:,0],stim_clust[:,1])
for i, txt in enumerate(labels):
    ax[1].annotate(txt, (stim_clust[i,0], stim_clust[i,1]))
ax[1].set_title('Post-Stim')

