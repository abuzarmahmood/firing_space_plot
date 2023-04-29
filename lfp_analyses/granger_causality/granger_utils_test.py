############################################################
## Imports
############################################################

import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
sys.path.append(ephys_data_dir)
from ephys_data import ephys_data
import numpy as np
import pylab as plt
sys.path.append('/media/bigdata/firing_space_plot/lfp_analyses/granger_causality')
import granger_utils as gu

############################################################
## Load Data 
############################################################

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path,'r') as f:
    dir_list = f.read().splitlines()
dir_name = dir_list[0]

dat = ephys_data(dir_name)
# Region lfps shape : (n_tastes, n_channels, n_trials, n_timepoints)
#region_lfps, region_names = dat.return_region_lfps()
lfp_channel_inds, region_lfps, region_names = \
        dat.return_representative_lfp_channels()

flat_region_lfps = np.reshape(region_lfps,(region_lfps.shape[0], -1,region_lfps.shape[-1]))

############################################################
## Preprocessing 
############################################################

# 1) Remove trials with artifacts
#good_lfp_trials_bool = dat.lfp_processing.return_good_lfp_trial_inds(dat.all_lfp_array)
good_lfp_data = dat.lfp_processing.return_good_lfp_trials(flat_region_lfps)

############################################################
## Compute Granger Causality 
############################################################
this_granger = gu.granger_handler(good_lfp_data)
this_granger.get_granger_sig_mask()

############################################################
## Generate plots 
############################################################
granger = this_granger.granger_actual
c = this_granger.c_actual
masked_granger = this_granger.masked_granger
mask_array = this_granger.mask_array
wanted_window = this_granger.wanted_window
alpha_mask = np.empty(mask_array.shape)
alpha_mask[~mask_array] = 1
alpha_mask[mask_array] = 0.3

# Plot values of granger
# Show values less than shuffle in red
t_vec = np.linspace(wanted_window[0], wanted_window[1], masked_granger.shape[0])
stim_t = 2000
t_vec = t_vec - stim_t

wanted_freqs = [0,60]
freq_inds = np.where((c.frequencies > wanted_freqs[0]) & \
        (c.frequencies < wanted_freqs[1]))[0]

cmap = plt.cm.viridis
cmap.set_bad(color='red')

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
im = ax[0].contourf(
        t_vec, c.frequencies[freq_inds], granger[:, freq_inds, 0, 1].T,
    cmap=cmap,
    levels = 15,
)
ax[0].set_title("x1 -> x2")
ax[0].set_ylabel("Frequency")
ax[0].axvline(0, color='red', linestyle='--')
plt.colorbar(im, ax = ax[0])
im = ax[1].contourf(
        t_vec, c.frequencies[freq_inds], granger[:, freq_inds, 1, 0].T,
    cmap=cmap, 
    levels = 15,
)
ax[1].set_title("x2 -> x1")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Frequency")
plt.colorbar(im, ax = ax[1])
ax[1].axvline(0, color='red', linestyle='--')
plt.show()
