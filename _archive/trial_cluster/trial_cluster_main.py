"""
Atempting to cluster trials from basline and post-stimulus firing
on the basis of temporally-integrated euclidean distance of normalized
firing vectors
"""

######################### Import dat ish #########################
import tables
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import *

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

import multiprocessing as mp

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as pca

from scipy.stats import mannwhitneyu as mnu
from sklearn.cluster import KMeans as kmeans
from scipy.spatial import distance_matrix as dist_mat

from skimage import exposure

import glob


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
#corr_dat = pd.DataFrame()
all_diff = []

for file in range(len(dir_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    data.correlation_params = dict(zip(['stimulus_start_time', 'stimulus_end_time',
                                        'baseline_start_time', 'baseline_end_time',
                                        'shuffle_repeats', 'accumulated'],
                                       [2000, 4000, 0, 2000, 100, True]))
    data.get_correlations()
    
    off_pre_dists = data.off_corr['pre_dists']
    off_stim_dists = data.off_corr['stim_dists']
    on_pre_dists = data.on_corr['pre_dists']
    on_stim_dists = data.on_corr['stim_dists']
    
    #data.get_dataframe()    
    #corr_dat = pd.concat([corr_dat, data.data_frame])
    print('file %i' % file)

    n_components = 2
    
    for taste in range(4):
        this_off = data.normal_off_firing[taste]
        this_off = this_off[:,:,80:160]
        total_off = this_off[0,:,:]
        for nrn in range(1,this_off.shape[0]):
            total_off = np.concatenate((total_off,this_off[int(nrn),:,:]),axis=1)
        
        reduced_off_pca = pca(n_components = 15).fit(total_off)
        print(sum(reduced_off_pca.explained_variance_ratio_))
        reduced_off = reduced_off_pca.transform(total_off)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              n_init = 200).fit(reduced_off)
        print(gmm.predict(reduced_off))
        
        this_groups = gmm.predict(reduced_off)
        trial_order = np.argsort(this_groups)
        
        # Pull out and cluster distance matrices
        this_dist = off_stim_dists[taste]   
        clust_dist = this_dist[trial_order,:]
        clust_dist = clust_dist[:,trial_order]
        
        this_pre_dist = off_pre_dists[taste]
        clust_pre_dist = this_pre_dist[trial_order,:]
        clust_pre_dist = clust_pre_dist[:,trial_order]
        
# =============================================================================
#         ## Look at if trials clustered together at during post-stimulus firing
#         ## have baseline firing which is closer together
#         
#         # 1) Go through every group in 'this_groups'
#         # 2) Subset appropriate parts of the distance matrix and calculate in-group distances
#         # 3) For comparison, randomly pick equally sized subsets from unclustered distance matrx
#         # and calculate in-group distances
#         
#         pre_group_dists = []
#         pre_group_dists_sh = []
#         stim_group_dists = []
#         stim_group_dists_sh = []
#         
#         sh_repeats = 1000
#         
#         for group in np.unique(this_groups):
#             inds = this_groups == group
#             
#             this_stim_group_dist = this_dist[inds,:]
#             this_stim_group_dist = this_stim_group_dist[:,inds]
#             this_stim_group_dist = this_stim_group_dist[np.tril_indices(this_stim_group_dist.shape[0])]
#             this_stim_group_dist = this_stim_group_dist[np.nonzero(this_stim_group_dist)]
#             stim_group_dists.append(np.mean(this_stim_group_dist))
#             
#             this_stim_sh = []
#             for n in range(sh_repeats):
#                 sh_inds = np.random.choice(np.arange(15),sum(inds),replace=False)
#                 sh_stim = this_dist[sh_inds,:]
#                 sh_stim = sh_stim[:,sh_inds]
#                 sh_stim = sh_stim[np.tril_indices(sh_stim.shape[0])]
#                 sh_stim = sh_stim[np.nonzero(sh_stim)]
#                 this_stim_sh.append(np.mean(sh_stim))
#             stim_group_dists_sh.append(this_stim_sh)
#                 
#             #plt.hist(this_stim_sh,50,cumulative=True,histtype='step',density=True)
#             #plt.vlines(stim_group_dists[0],ymin=0,ymax=1)
#             #plt.vlines(stim_group_dists[1],ymin=0,ymax=1)
#             
#             this_pre_group_dist = this_pre_dist[inds,:]
#             this_pre_group_dist = this_pre_group_dist[:,inds]
#             this_pre_group_dist = this_pre_group_dist[np.tril_indices(this_pre_group_dist.shape[0])]
#             this_pre_group_dist = this_pre_group_dist[np.nonzero(this_pre_group_dist)]
#             pre_group_dists.append(np.mean(this_pre_group_dist))
#             
#             this_pre_sh = []
#             for n in range(sh_repeats):
#                 sh_inds = np.random.choice(np.arange(15),sum(inds),replace=False)
#                 sh_pre = this_pre_dist[sh_inds,:]
#                 sh_pre = sh_pre[:,sh_inds]
#                 sh_pre = sh_pre[np.tril_indices(sh_pre.shape[0])]
#                 sh_pre = sh_pre[np.nonzero(sh_stim)]
#                 this_pre_sh.append(np.mean(sh_pre))
#             pre_group_dists_sh.append(this_pre_sh)
#             
#             #plt.hist(this_pre_sh,50,cumulative=True,histtype='step',density=True)
#             #plt.vlines(pre_group_dists[0],ymin=0,ymax=1)
#             #plt.vlines(pre_group_dists[1],ymin=0,ymax=1)
# =============================================================================
            
        ## Distance matrix cluster plots
        plt.figure()
        plt.subplot(221);plt.imshow(exposure.equalize_hist(this_dist));plt.title('Un Stim')
        plt.subplot(222);plt.imshow(exposure.equalize_hist(clust_dist));plt.title('Clust Stim')
        line_num = np.where(np.diff(np.sort(this_groups)))[0]
        for point in line_num:
            plt.axhline(point+0.5,color = 'red')
            plt.axvline(point+0.5,color = 'red')
            
        plt.subplot(223);plt.imshow(exposure.equalize_hist(this_pre_dist));plt.title('Un Pre')
        plt.subplot(224);plt.imshow(exposure.equalize_hist(clust_pre_dist));plt.title('Clust Pre')
        line_num = np.where(np.diff(np.sort(this_groups)))[0]
        for point in line_num:
            plt.axhline(point+0.5,color = 'red')
            plt.axvline(point+0.5,color = 'red')
        
        clust_list = []
        for cluster in range(n_components):
            this_cluster = this_off[:,this_groups == cluster,:]
            clust_list.append(this_cluster)
        
        max_vals = []
        clust_means = []   
        for cluster in range(len(clust_list)):
            dat = np.mean(clust_list[cluster],axis=1)
            clust_means.append(dat)
            max_vals.append(np.max(dat))
        
        ## Firing rate Plots
        plt.figure()
        for cluster in range(len(clust_list)):
            plt.subplot(n_components,1,cluster+1)
            dat = np.mean(clust_list[cluster],axis=1)
            plt.imshow(dat,
                       interpolation='nearest',aspect='auto',vmin=0,vmax=max(max_vals))
            plt.title('n = %i' % clust_list[cluster].shape[1])
            plt.colorbar()
        
        plt.figure()
        for nrn in range(this_off.shape[0]):
            plt.subplot(this_off.shape[0],1,nrn+1)
            for cluster in range(len(clust_list)):
                dat = clust_list[cluster][nrn,:,:]
                y = np.mean(dat,axis=0)
                x = range(dat.shape[1])
                yerr = np.std(dat,axis=0)/np.sqrt(dat.shape[0])
                plt.errorbar(x = x,y = y,yerr = yerr)
                
# =============================================================================
#         mean_dat = np.asarray(clust_means)
#         mean_diff = np.diff(mean_dat,axis=0)[0,:,:]
# =============================================================================
# =============================================================================
#         mean_max = np.max(np.max(mean_dat,axis=2),axis=0)
#         norm_diff = np.zeros(mean_diff.shape)
#         for i in range(len(mean_max)):
#             norm_diff = np.divide(mean_diff,mean_max[i])
# =============================================================================
            
        all_diff.append(mean_diff)

plt.close('all')

all_diff_array = np.asarray(all_diff[0])
for i in all_diff[1:]:
    all_diff_array = np.concatenate((all_diff_array,i),axis=0)
all_diff_array = np.abs(all_diff_array)
plt.errorbar(x = range(80), y = np.mean(all_diff_array,axis=0),yerr = np.std(all_diff_array,axis=0))

plt.imshow(all_diff_array[np.argsort(np.argmax(all_diff_array,axis=1)),:],
                          interpolation='nearest',aspect='auto')

# =============================================================================
# =============================================================================
"""
See if trials from baseline and post-stimulus firing are enriched in the same clusters
"""

#dir_list = ['/media/bigdata/jian_you_data/des_ic']
dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)

baseline_inds = range(80)
stimulus_inds = range(80,160)

same_clust = []
same_clust_sh = []

base_group_nums = []
stim_group_nums = []

n_components = 2
shuffle_num = 100

clusts = list(range(n_components))

# Make permutations of clusters so items from all clusters can be compared
all_perms = []
for x in range(len(clusts)*100):
    this_perm = list(np.random.permutation(clusts))
    if this_perm not in all_perms:
        all_perms.append(this_perm)

for file in range(len(file_list)):
    data_dir = os.path.dirname(file_list[file])
    data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = False)
    data.firing_rate_params = dict(zip(['step_size','window_size','total_time'],
                                   [25,250,7000]))
    data.get_data()
    data.get_firing_rates()
    
    
    for taste in range(4):
        
        base_dat = data.normal_off_firing[taste][:,:,baseline_inds]
        stim_dat = data.normal_off_firing[taste][:,:,stimulus_inds]
        
        # Reduce data to PCAs
        
        base_long = base_dat[0,:,:]
        for nrn in range(1,base_dat.shape[0]):
            base_long = np.concatenate((base_long,base_dat[int(nrn),:,:]),axis=1)
            
        stim_long = stim_dat[0,:,:]
        for nrn in range(1,stim_dat.shape[0]):
            stim_long = np.concatenate((stim_long,stim_dat[int(nrn),:,:]),axis=1)
            
        base_pca = pca(n_components = 15).fit(base_long)
        stim_pca = pca(n_components = 15).fit(stim_long)
        
        reduced_base = base_pca.transform(base_long)
        reduced_stim = stim_pca.transform(stim_long)
        
        count = 0
        #this_base_group_nums, this_stim_group_nums = [0,0,0], [0,0,0]
        #while not (this_base_group_nums == [5,5,5]) & (this_stim_group_nums == [5,5,5]):
        base_gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              n_init = 500).fit(reduced_base)
        stim_gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              n_init = 500).fit(reduced_stim)
        
        base_groups = base_gmm.predict(reduced_base)
        stim_groups = base_gmm.predict(reduced_stim)
        
        this_base_group_nums = [sum(base_groups == x) for x in np.unique(base_groups)]
        this_stim_group_nums = [sum(stim_groups == x) for x in np.unique(stim_groups)]
        #count += 1
        #if count%100 ==0 :print(count)
            
        base_group_nums += this_base_group_nums
        stim_group_nums += this_stim_group_nums
        
        temp_same_clust = []
        for comparison in range(len(all_perms)):
            
            this_order = all_perms[comparison]
            temp_base_groups = np.zeros(base_groups.shape).astype(int)
            
            for group in range(len(this_order)):
                temp_base_groups[base_groups == clusts[group]] = this_order[group]
            
            temp_same_clust.append(sum(temp_base_groups == stim_groups))
        
        same_clust.append(np.max(temp_same_clust))
        
        
        for i in range(shuffle_num):
            
            base_groups_sh = np.random.permutation(base_groups)
            
            for comparison in range(len(all_perms)):
                
                this_order = all_perms[comparison]
                temp_base_groups_sh = np.zeros(base_groups.shape).astype(int)
                
                for group in range(len(this_order)):
                    temp_base_groups_sh[base_groups_sh == clusts[group]] = this_order[group]
            
                same_clust_sh.append(sum(temp_base_groups_sh == stim_groups))
        
    print(file)

fig, ax = plt.subplots(1,1)
ax.set_xticks([1,2])
ax.set_xticklabels(['Data','Shuffle'])
ax.violinplot([same_clust,same_clust_sh])
ax.set_ylabel('Number of trials in same group')
ax.set_xlabel('Group')

plt.figure()
plt.hist(stim_group_nums,20)
plt.hist(base_group_nums,20)

# =============================================================================
# Mean basis clustering
# =============================================================================
# Use PCA on mean firing rate for all trials --> Reduce to 1D
# Calculate distances between all trials then and cluster

dir_list = ['/media/bigdata/jian_you_data/des_ic', '/media/bigdata/NM_2500']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    
file = 0
taste = 0

data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'baks',100]))
data.get_data()
data.get_firing_rates()
data.get_normalized_firing()

spikes_array = np.asarray(data.spikes)
firing_array = np.asarray(data.normal_off_firing)
this_firing = firing_array[taste,:,:,20:50]
this_spikes = spikes_array[taste,:,:,2000:5000]

mean_firing = np.mean(this_firing,axis=1)

# Reduce to 1D
mean_pca = pca(n_components = 1).fit(mean_firing.T)
reduced_mean = mean_pca.transform(mean_firing.T)

# Reduce all other trials using same transformation
reduced_trials = np.zeros(this_firing.shape[1:])
for trial in range(this_firing.shape[1]):
    reduced_trials[trial,:] = mean_pca.transform(this_firing[:,trial,:].T).flatten()
    
# Calculate distances and Perform k-means on trials
nrn_dist = exposure.equalize_hist(dist_mat(reduced_trials,reduced_trials))
    
n_components = 3
clf = kmeans(n_clusters = n_components, n_init = 500)
this_groups = clf.fit_predict(nrn_dist)


trial_order = np.argsort(this_groups)

# Pull out and cluster distance matrices
clust_post_dist = nrn_dist[trial_order,:]
clust_post_dist = clust_post_dist[:,trial_order]

## Distance matrix cluster plots
plt.figure()
plt.subplot(221);plt.imshow(exposure.equalize_hist(nrn_dist));plt.title('Un Stim')
plt.subplot(222);plt.imshow(exposure.equalize_hist(clust_post_dist));plt.title('Clust Stim')
line_num = np.where(np.diff(np.sort(this_groups)))[0]
for point in line_num:
    plt.axhline(point+0.5,color = 'red')
    plt.axvline(point+0.5,color = 'red')
    
## Clustered raster plots
nrn = 9
plt.figure();dot_raster(this_spikes[nrn,trial_order,:])
line_num = np.append(-1,np.where(np.diff(np.sort(this_groups)))[0])
for point in range(len(line_num)):
    plt.axhline(line_num[point]+0.5,color = 'red')
    plt.text(0,line_num[point]+0.5,point,fontsize = 20,color = 'r')
    
    
clust_list = []
for cluster in range(n_components):
    this_cluster = this_firing[:,this_groups == cluster,:]
    clust_list.append(this_cluster)

max_vals = []
clust_means = []   
for cluster in range(len(clust_list)):
    dat = np.mean(clust_list[cluster],axis=1)
    clust_means.append(dat)
    max_vals.append(np.max(dat))

## Firing rate Plots
plt.figure()
for cluster in range(len(clust_list)):
    plt.subplot(n_components,1,cluster+1)
    dat = np.mean(clust_list[cluster],axis=1)
    plt.imshow(dat,
               interpolation='nearest',aspect='auto',vmin=0,vmax=max(max_vals))
    plt.title('n = %i' % clust_list[cluster].shape[1])
    plt.colorbar()

plt.figure()
for nrn in range(this_firing.shape[0]):
    plt.subplot(this_firing.shape[0],1,nrn+1)
    for cluster in range(len(clust_list)):
        dat = clust_list[cluster][nrn,:,:]
        y = np.mean(dat,axis=0)
        x = range(dat.shape[1])
        yerr = np.std(dat,axis=0)/np.sqrt(dat.shape[0])
        plt.errorbar(x = x,y = y,yerr = yerr)