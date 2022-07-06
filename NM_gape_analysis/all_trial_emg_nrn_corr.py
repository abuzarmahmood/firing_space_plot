"""
If we group all trials across session (with appropriate normalizations),
can we say something about the neural activity in relation to the EMG signals
"""
########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

########################################
# Import modules
########################################

import os
import sys
import numpy as np
from tqdm import tqdm
import tables
import itertools as it
import pylab as plt
from glob import glob
import pandas as pd
import seaborn as sns
import re
import pingouin as pg
from scipy.stats import zscore
import tensortools as tt
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import xarray as xr
from numpy import dot
from numpy.linalg import norm


sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

plot_dir = '/media/bigdata/firing_space_plot/NM_gape_analysis/plots'

file_list_path = '/media/fastdata/NM_sorted_data/h5_file_list.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
dir_names = [os.path.dirname(x) for x in file_list]

def time_box_conv(array, kern_width):
    """
    Convolution with 1D box kern along last dim
    """
    box_kern = np.ones((kern_width))/kern_width
    inds = list(np.ndindex(array.shape[:-1]))
    conv_array = np.empty(array.shape)
    for this_ind in tqdm(inds):
        conv_array[this_ind] = np.convolve(array[this_ind], box_kern, mode='same') 
    return conv_array

cos_sim = lambda a,b : dot(a, b)/(norm(a)*norm(b))
############################################################
# / ___| __ _ _ __   ___  |  _ \  __ _| |_ __ _ 
#| |  _ / _` | '_ \ / _ \ | | | |/ _` | __/ _` |
#| |_| | (_| | |_) |  __/ | |_| | (_| | || (_| |
# \____|\__,_| .__/ \___| |____/ \__,_|\__\__,_|
#            |_|                                
############################################################

gape_path = '/ancillary_analysis/gapes'
# laser_conds x taste x trials x time

emg_bsa_path = '/ancillary_analysis/emg_BSA_results'
# laser_conds x taste x trials x time x freq 

laser_dl_path = '/ancillary_analysis/laser_combination_d_l'
# condition_num x (duration + onset) 

# /emg_BSA_results
# taste_0_p : something? x time x something?

#ind = 1
#this_file_path = file_list[ind]
#this_dir = dir_names[ind]
#

condition_list = []
gapes_list = []
for this_path in file_list:
    # Gape related info
    #with tables.open_file(dat.hdf5_path,'r') as h5:
    with tables.open_file(this_path,'r') as h5:
        gape_laser_conditions = h5.get_node(laser_dl_path)[:] 
        #emg_bsa_results = h5.get_node(emg_bsa_path)[:] 
        gapes_array = h5.get_node(gape_path)[:] 
    condition_list.append(gape_laser_conditions)
    gapes_list.append(gapes_array)

flat_gapes = [x.flatten() for x in gapes_list]
flat_gapes = [x for y in flat_gapes for x in y]
#plt.hist(flat_gapes[::10]);plt.show()

# Some values are really low (e.g. e-88), clean those out
gapes_list = [(x>0.5)*1 for x in gapes_list]

wanted_condition = [np.where(x.sum(axis=-1)==0)[0][0] for x in condition_list]
wanted_gape_array = [x[i] for x,i in zip(gapes_list, wanted_condition)]

########################################
## Quin vs Suc Gape reponse
########################################

time_lims = [1000,5000]
real_time = np.arange(-2000, 5000)
cut_real_time = real_time[time_lims[0]:time_lims[1]]
stim_t = 2000 - time_lims[0]
taste_inds = np.array([0,3]) # 0:Sucrose, 3:quinine
taste_names = ['suc','quin']
quin_ind = 3
suc_ind = 0
wanted_gape_array = [x[taste_inds] for x in wanted_gape_array]
wanted_gape_array = [x[...,time_lims[0]:time_lims[1]] for x in wanted_gape_array]

# Check for which recordings distances are sufficiently separate
inds = [np.array(list(np.ndindex(x.shape))) for x in wanted_gape_array]
gape_frames = [pd.DataFrame(dict(
                session = num,
                taste = this_inds[:,0],
                trials = this_inds[:,1],
                time = this_inds[:,2],
                vals = this_dat.flatten())) \
        for num, (this_inds,this_dat) in enumerate(zip(inds, wanted_gape_array))]
fin_gape_frame = pd.concat(gape_frames)
fin_gape_frame['real_time'] = cut_real_time[fin_gape_frame['time']]

# Downsample for ANOVA
binsize = 500
bincount = int(np.diff(time_lims)[0]/binsize)
fin_gape_frame['time_bins'] = pd.cut(fin_gape_frame['time'], bincount, 
       labels = np.arange(bincount))
fin_gape_frame['vals'] += np.random.random(fin_gape_frame['vals'].shape)*0.01

group_cols = ['session','taste','trials','time_bins']
bin_gape_frame = fin_gape_frame.groupby(group_cols).mean().reset_index()
bin_gape_frame.dropna(inplace=True)

# Perform ANOVA
# Perform separately for each session
group_bin_gape = [x[1] for x in list(bin_gape_frame.groupby('session'))]
anova_list = [pg.anova(data = this_dat,
            dv = 'vals', between = ['taste','time_bins']) \
                    for this_dat in group_bin_gape]
pval_list = [x['p-unc'] for x in anova_list]
taste_bool = np.stack(pval_list)[:,0]<0.05

# Also check that quinine is HIGHER than sucrose
quin_bool = [x.groupby('taste')['vals'].mean().diff()[1]>0 for x in group_bin_gape]

#g = sns.relplot(data = bin_gape_frame,
#        x = 'real_time', y = 'vals',
#        hue = 'taste', col = 'session', col_wrap = 5,
#        kind = 'line')
#for this_ax, this_taste_bool, this_quin_bool  in zip(g.axes, taste_bool, quin_bool):
#    this_ax.set_title((this_taste_bool, this_quin_bool))
#plt.show()

# Only take session where both are true
fin_bool = np.logical_and(taste_bool, quin_bool)
fin_bool_inds = np.where(fin_bool)[0]
fin_bin_gape = bin_gape_frame[bin_gape_frame['session'].isin(fin_bool_inds)] 

#g = sns.relplot(data = fin_bin_gape,
#        x = 'real_time', y = 'vals',
#        hue = 'taste', col = 'session', col_wrap = 5,
#        kind = 'line')
#plt.show()

########################################
## Subtract AVERAGE sucrose response as non-specific EMG response 
########################################
taste_frame = fin_gape_frame.copy()

corr_frame = taste_frame[taste_frame.real_time > 0].groupby(['session','taste']).mean()
x = np.linspace(corr_frame.vals.min(), corr_frame.vals.max())
plt.scatter(*[x[1].vals for x in list(corr_frame.groupby('taste'))])
plt.xlabel('Average sucrose response')
plt.ylabel('Average quinine response')
plt.plot(x,x, color = 'red', alpha = 0.3, linestyle = '--')
fig = plt.gcf()
fig.suptitle('Correlated Taste Responses')
fig.savefig(os.path.join(plot_dir, 'correlated_taste_responses.png'))
plt.show()

mean_taste_frame = taste_frame.groupby(['session','taste','time']).mean()
mean_taste_frame = mean_taste_frame.drop(columns = 'trials')
mean_taste_array = mean_taste_frame.to_xarray()['vals']
mean_taste_array = mean_taste_array[fin_bool].reset_index('session')

taste_diff_array = mean_taste_array.diff(dim = 'taste').squeeze()

g = mean_taste_array.plot(
        x = 'time',
        y = 'session',
        col = 'taste',
        aspect = 2,
        size = 3
        );
for num, ax in enumerate(g.axes[0]):
    ax.axvline(stim_t, linestyle = '--', color = 'red', linewidth = 2,
            label = 'Stim Delivery')
    ax.set_title(taste_names[num])
    #ax.legend()
fig = plt.gcf()
fig.suptitle('Average EMG Resopnses')
#plt.subplots_adjust(top = 0.8)
fig.savefig(os.path.join(plot_dir, 'average_emg_responses.png'))
#plt.show()

taste_diff_array.plot(cmap = 'viridis', aspect = 2, size = 3);
ax = plt.gca()
ax.axvline(stim_t, linestyle = '--', color = 'k')
fig = plt.gcf()
fig.suptitle('Quin - Suc Responses')
fig.savefig(os.path.join(plot_dir, 'average_subtracted_emg_responses.png'))
#plt.show()

########################################
## Clustering in gape responses to quinine 
########################################
quin_gape_array = [x[1] for x in wanted_gape_array]
quin_gape_array = [quin_gape_array[i] for i in fin_bool_inds]

# Subtract mean sucrose response from respective quinine responses
suc_gape_array = [x[0] for x in wanted_gape_array]
suc_gape_array = [suc_gape_array[i] for i in fin_bool_inds]
mean_suc_gape = np.stack([np.mean(x,axis=0) for x in suc_gape_array])
quin_gape_array = [x-y for x,y in zip(quin_gape_array, mean_suc_gape)]

#vz.imshow(mean_suc_gape);plt.colorbar();plt.show()

gape_t_lims = [750,2500]
gape_t_lims = [x+time_lims[0] for x in gape_t_lims]

process_gape_array = [x[...,gape_t_lims[0]:gape_t_lims[1]] for x in quin_gape_array]
mean_gape_val = [x.mean(axis=-1) for x in process_gape_array]

mean_gape_frame = pd.concat(
        [
            pd.DataFrame(
                dict(
                    session_num = num,
                    trial_num = range(len(x)),
                    vals = x
                    )
            )
            for num, x in enumerate(mean_gape_val)
            ]
        ).sort_values('vals')

inds = list(zip(mean_gape_frame.session_num, mean_gape_frame.trial_num))
stacked_gapes = np.stack([process_gape_array[x][y] for x,y in inds])
vz.imshow(stacked_gapes);plt.show()

############################################################
# Given the above clusters, extract neural data and cluster similarly
fin_file_list = [file_list[i] for i in fin_bool_inds]
fin_basenames = [os.path.basename(x) for x in fin_file_list]
fin_dirnames = [os.path.dirname(x) for x in fin_file_list]

epoch_lims = dict(
        iden = [250,750],
        pal = [750,1250]
        )

diff_sim_list = []

#this_session_ind = 0
kern_width = 250
for this_session_ind in range(len(fin_file_list)):
    this_dir = fin_dirnames[this_session_ind]
    this_basename = fin_basenames[this_session_ind]
    dat = ephys_data(this_dir)
    dat.get_spikes()
    dat.check_laser()
    #dat.laser_durations

    quin_laser = dat.laser_durations[quin_ind]
    quin_spikes = dat.spikes[quin_ind] 
    quin_off_spikes = quin_spikes[quin_laser == 0]

    suc_laser = dat.laser_durations[suc_ind]
    suc_spikes = dat.spikes[suc_ind] 
    suc_off_spikes = suc_spikes[suc_laser == 0]

    # Trials x nrns x time
    quin_firing = time_box_conv(quin_off_spikes, kern_width)
    suc_firing = time_box_conv(suc_off_spikes, kern_width)
    mean_quin = quin_firing.mean(axis = (0,2))
    mean_suc = suc_firing.mean(axis = (0,2))

    norm_quin_firing = quin_firing / mean_quin[np.newaxis,:,np.newaxis]
    norm_suc_firing = suc_firing / mean_suc[np.newaxis,:,np.newaxis]

    norm_quin_epochs = [norm_quin_firing[...,x[0]:x[1]] for x in epoch_lims.values()]
    norm_suc_epochs = [norm_suc_firing[...,x[0]:x[1]] for x in epoch_lims.values()]

    mean_norm_quin = np.stack([x.mean(axis=(2)) for x in norm_quin_epochs])
    mean_norm_suc = np.stack([x.mean(axis=(2)) for x in norm_suc_epochs])

    norm_quin_template = [x.mean(axis=(0,2)) for x in norm_quin_epochs]
    norm_suc_template = [x.mean(axis=(0,2)) for x in norm_suc_epochs]

    #For each quinine trial and epoch, calculate difference in cosine similarity
    # for suc and quin i.e. f = cos_sim, metric = f(quin) - f(suc)
    # if metric is more negative, output is more similar to sucrose than quinine



    # Template epoch x test epochs x trials
    quin_sim = np.zeros((2,*mean_norm_quin.shape[:-1]))
    suc_sim = np.zeros((2,*mean_norm_suc.shape[:-1]))

    quin_iters = list(np.ndindex(quin_sim.shape))
    suc_iters = list(np.ndindex(suc_sim.shape))

    for this_iter in quin_iters:
        quin_sim[this_iter] = cos_sim(
                                norm_quin_template[this_iter[0]],
                                mean_norm_quin[this_iter[1:]])

    for this_iter in suc_iters:
        suc_sim[this_iter] = cos_sim(
                                norm_suc_template[this_iter[0]],
                                mean_norm_suc[this_iter[1:]])

    quin_sim_flat = np.reshape(quin_sim, (-1, quin_sim.shape[-1]))
    suc_sim_flat = np.reshape(suc_sim, (-1, suc_sim.shape[-1]))

    diff_sim_flat = quin_sim_flat - suc_sim_flat
    diff_sim_list.append(diff_sim_flat)

# Stack similarities similar to emg
stacked_sim = np.stack([diff_sim_list[x][:,y] for x,y in inds])

fig,ax = plt.subplots(1,2)
ax[0].imshow(stacked_gapes, 
        interpolation = 'nearest', aspect = 'auto', cmap = 'viridis');
ax[1].imshow(stacked_sim, 
        interpolation = 'nearest', aspect = 'auto', cmap = 'viridis');
plt.show()

plt.plot(stacked_sim);plt.show()
