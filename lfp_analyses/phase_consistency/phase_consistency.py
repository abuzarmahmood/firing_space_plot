"""
Show phase resetting and phase consistency in GC and BLA LFP

Figures:
    1. Phase resetting across trials
    2. Polar histogram of phase at t = 50ms post-stimulus
    3. Phase consistency across trials
"""
############################################################
## Import required modules
############################################################
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
from scipy.signal import spectrogram
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
from sklearn.utils import resample
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import zscore

############################################################
## Setup 
############################################################
base_dir = '/media/bigdata/firing_space_plot/lfp_analyses/phase_consistency'
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
plot_dir = os.path.join(base_dir,'plots')
if not os.path.isdir(plot_dir):
    os.mkdir(plot_dir)

file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in file_list]

phase_consistency_list = []
# ind = 0
for ind in trange(len(file_list)):
    this_dat = ephys_data(file_list[ind])
    basename = os.path.basename(file_list[ind])
    this_dat.get_stft(dat_type = ['phase']) 
    # phase_array = this_dat.phase_array.copy()
    freq_vec = this_dat.freq_vec.copy()
    time_vec = this_dat.time_vec.copy()
    wanted_channel_inds, wanted_lfp_electrodes, region_names = \
            this_dat.return_representative_lfp_channels()
    wanted_phases = phase_array[:, wanted_channel_inds]
    # Flatten out across tastes
    wanted_phases = np.swapaxes(wanted_phases,1,2)
    wanted_phases = np.concatenate(wanted_phases,axis=0)
    wanted_phases = wanted_phases.swapaxes(0,1)

    phase_vectors = np.exp(1j*wanted_phases)
    phase_consistency = np.abs(np.mean(phase_vectors,axis=1))
    phase_consistency_list.append(phase_consistency)

    ind_consistency_plot_dir = os.path.join(plot_dir,'ind_consistency_plots')
    if not os.path.isdir(ind_consistency_plot_dir):
        os.mkdir(ind_consistency_plot_dir)

    # Plot phase consistency
    vmin = np.min(phase_consistency)
    vmax = np.max(phase_consistency)
    fig, ax = plt.subplots(1, len(phase_consistency))
    for ind, (this_phase, this_ax) in enumerate(zip(phase_consistency, ax)):
        im = this_ax.pcolormesh(time_vec, freq_vec, 
                                phase_consistency[ind], 
                                cmap = 'viridis',
                                vmin = vmin, vmax = vmax)
        this_ax.set_xlim([1.5,3.5])
        this_ax.set_ylim([0,20])
        this_ax.set_xlabel('Time (s)')
        this_ax.set_ylabel('Frequency (Hz)')
        this_ax.set_title(region_names[ind])
    # Put color at bottom
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal',
                 label = basename + '\nPhase Consistency')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    # plt.show()
    plt.savefig(
            os.path.join(
                ind_consistency_plot_dir,basename+'_phase_consistency.png'))
    plt.close(fig)

phase_consistency_array = np.stack(phase_consistency_list,axis=0)
mean_phase_consistency = np.mean(phase_consistency_array,axis=0)

# Plot mean phase consistency
vmin = np.min(mean_phase_consistency)
vmax = np.max(mean_phase_consistency)
fig, ax = plt.subplots(2, len(mean_phase_consistency))
for ind, (this_phase, this_ax) in enumerate(zip(mean_phase_consistency, ax)):
    im = this_ax[0].pcolormesh(time_vec, freq_vec, 
                       mean_phase_consistency[ind], 
                       cmap = 'viridis',
                       vmin = vmin, vmax = vmax)
    this_ax[0].set_title(region_names[ind])
    this_ax[0].set_xlim([1.5,3.5])
    this_ax[0].set_ylim([0,20])
    this_ax[1].plot(time_vec, mean_phase_consistency[ind].T,
                    alpha = 0.5, color = 'k')
    this_ax[1].set_xlim([1.5,3.5])
    this_ax[1].set_ylim([0,1])
    this_ax[1].set_xlabel('Time (s)')
    this_ax[1].set_ylabel('Mean Phase Consistency')
# Put color at bottom
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation = 'horizontal',
             label = 'Mean Phase Consistency')
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig(os.path.join(plot_dir,'mean_phase_consistency.png'))
plt.close(fig)
# plt.show()

##############################
# Find data with max phase consistency 
max_phase_consistency = np.max(phase_consistency_array,axis=(3))
max_ind = np.where(max_phase_consistency == np.max(max_phase_consistency)) 
max_ind = np.squeeze(max_ind)

this_dat = ephys_data(file_list[max_ind[0]])
basename = os.path.basename(file_list[max_ind[0]])
this_dat.get_stft(dat_type = ['phase']) 

wanted_channel_inds, wanted_lfp_electrodes, region_names = \
        this_dat.return_representative_lfp_channels()
wanted_phases = phase_array[:, wanted_channel_inds]
# Flatten out across tastes
wanted_phases = np.swapaxes(wanted_phases,1,2)
wanted_phases = np.concatenate(wanted_phases,axis=0)
wanted_phases = wanted_phases.swapaxes(0,1)
wanted_phases = wanted_phases[max_ind[1]]

# Plot data for 7 Hz
freq_ind = np.argmin(np.abs(freq_vec - 7))

fig, ax = plt.subplots(2,1, figsize = (10,6), sharex=True)
im = ax[0].pcolormesh(time_vec*1000 -2000, np.arange(len(wanted_phases)), 
                   wanted_phases[:,freq_ind], 
                   cmap = 'twilight')
ax[0].set_ylabel('Trial #')
ax[0].set_title(basename + '\n7 Hz Phase')
cax = fig.add_axes([1, 0.2, 0.02, 0.6])
plt.colorbar(im, ax = ax[0], cax = cax, label = 'Phase (rad)')
ax[1].plot(time_vec*1000 -2000, np.sin(wanted_phases[:,freq_ind]).T,
           alpha = 0.03, color = 'k')
# Also plot mean
ax[1].plot(time_vec*1000 -2000, 
           np.sin(wanted_phases[:,freq_ind].mean(axis=0)),
           color = 'k', linewidth = 2, linestyle = '--', 
           label = 'Mean Phase')
ax[1].set_xlabel('Time post-stim (ms)')
ax[1].legend(loc = 'upper right')
for this_ax in ax:
    this_ax.axvline(0, color = 'k', linestyle = '--')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir,'max_phase_consistency.png'),
            bbox_inches = 'tight')
plt.close(fig)
# plt.show()

############################################################
# Simulate phase-coherence for random data
############################################################
