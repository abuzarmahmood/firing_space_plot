## Import required modules
import os
import matplotlib.pyplot as plt
import tables
import numpy as np
from tqdm import tqdm, trange
import shutil
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
from scipy.stats import zscore
from pathlib import Path
from joblib import Parallel, delayed, cpu_count
from scipy.signal import istft
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

##################################################
## Define functions
##################################################

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='center', width=widths,
           edgecolor='C0', 
           fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        #label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
        #          r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$-3\pi/4$', r'$-2\pi/2$', r'$-\pi/4$']
        ax.set_xticklabels(label)

plot_dir='/media/bigdata/firing_space_plot/lfp_analyses/lfp_phase_coherence/plots'
##################################################
## Read in data 
##################################################
dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in dir_list]
file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 


# Load data from parsed_lfp, bandpass filter, calculate instantaneous phase
# and calculate phase difference
with tables.open_file(file_list[0],'r') as hf5:
    time_vec = hf5.get_node('/stft/time_vec')[:]
    freq_vec = hf5.get_node('/stft/freq_vec')[:]

# Convert to bands
freq_bands_lims = [[3,8] , [8,13], [13, 30],[30,70],[70,100]] 
freq_map = np.array(['Theta', 'Alpha', 'Beta','low_Gamme','high_Gamma'])
freq_bands = [np.arange(*x) for x in freq_bands_lims]
freq_inds = [np.array([i for i,val in enumerate(freq_vec) if int(val) in band]) \
                    for band in freq_bands]

#for this_dir in tqdm(dir_list):
this_dir = dir_list[0]
dat = ephys_data(this_dir)
dat.get_stft(recalculate = False, dat_type = ['raw'])
#dat.get_stft(recalculate = False, dat_type = ['phase'])
dat.get_lfp_electrodes()
stft_array = dat.stft_array
#phase_array = dat.phase_array

region_order = np.argsort(dat.region_names)
sorted_region_names = np.array(dat.region_names)[region_order]
wanted_elecs = np.array([x[0] for x in dat.lfp_region_electrodes])[region_order]
stft_array = stft_array[:,wanted_elecs]
#phase_array = phase_array[:,wanted_elecs]

#for this_band in freq_bands_lims:
this_band = freq_bands_lims[0]
band_nums = np.arange(this_band[0],this_band[1])
#freq_inds = np.array([x in band_nums for x in freq_vec])
freq_inds = np.array([x in [2] for x in freq_vec])
# taste x region x trials x freq x time
this_stft = stft_array[:,:,:,freq_inds]
#this_phase = phase_array[:,:,:,freq_inds]

# Regions x freqs x time
#wanted_stft_trials = this_stft[0,:,0]
#wanted_phase_trials = this_phase[0,:,0]
this_freqs = freq_vec[freq_inds].reshape((-1,1))
carrier = np.exp(2*np.pi*1j*this_freqs*time_vec)#[np.newaxis,:,:]
carrier = np.expand_dims(carrier, axis = (0,1,2))

#orig_sig = np.sum(carrier * wanted_stft_trials,axis=1)
orig_sig = np.sum(carrier * this_stft,axis=3).swapaxes(1,2)
long_sig = np.concatenate(orig_sig)
phases = np.angle(orig_sig) 
long_phases = np.concatenate(phases)


#img_kwargs = dict(interpolation = 'nearest', aspect = 'auto')
#fig,ax = plt.subplots(2,1)
#ax[0].imshow(zscore(np.real(long_sig[:,0]),axis=-1), **img_kwargs)
#ax[1].imshow(zscore(np.real(long_sig[:,1]),axis=-1), **img_kwargs)
#plt.show()
#
#img_kwargs = dict(interpolation = 'nearest', aspect = 'auto')
#fig,ax = plt.subplots(2,1)
#ax[0].imshow(long_phases[:,0], **img_kwargs)
#ax[1].imshow(long_phases[:,1], **img_kwargs)
#plt.show()

fin_time_vec = (time_vec-2)*1000
time_lims = [-1000,2500]
dat_inds = (fin_time_vec > time_lims[0]) & (fin_time_vec < time_lims[1])

trial = 2
wanted_phases = long_phases[trial]
phase_diff_raw = np.exp(-1j*np.diff(wanted_phases,axis=0))
phase_diff = np.angle(phase_diff_raw)
fig,ax = plt.subplots(3,1, sharex=True, 
        gridspec_kw = dict(height_ratios = [1,1,0.25]),
        figsize = (5,3))
for name, this_sig, this_phase in \
        zip(sorted_region_names, long_sig[trial], long_phases[trial]):
    ax[0].plot(fin_time_vec[dat_inds], this_sig[...,dat_inds].T, label = name)
    ax[1].plot(fin_time_vec[dat_inds], this_phase[...,dat_inds].T, label = name)
im = ax[2].pcolormesh(
        fin_time_vec[dat_inds],
        [0,1],
        np.tile(phase_diff[:,dat_inds], (2,1)), 
            cmap = 'twilight')
cax = plt.axes([0.91,0.1,0.02,0.1])
cbar = fig.colorbar(im, cax = cax)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Radians', rotation = 270)
ax[0].set_yticks([])
ax[2].set_yticks([])
#ax[1].set_ylabel('Phase (Radians)')
#ax[2].set_ylabel('Phase Difference') 
#ax[0].set_ylabel('Filtered LFP') 
ax[0].axvline(0, color = 'black', linestyle = '--')
ax[1].axvline(0, color = 'black', linestyle = '--')
ax[2].axvline(0, color = 'black', linestyle = '--')
ax[-1].set_xlabel('Time post-stimulus (ms)')
ax[0].legend(bbox_to_anchor = (1.1,1.1))
fig.savefig(os.path.join(plot_dir,'single_trial_phase_diff.png'), dpi = 300,
        bbox_inches = 'tight')
plt.close(fig)
#plt.show()

############################################################
############################################################
single_bin_phases = long_phases[:,:,fin_time_vec==-500]
single_bin_phase_diff_raw = np.exp(-1j*np.diff(single_bin_phases,axis=1))
single_bin_phase_diff = np.angle(single_bin_phase_diff_raw)
single_mean_vec = np.mean(single_bin_phase_diff_raw)
shuffle_bin_phases = np.stack([
        single_bin_phases[:,0],
        np.random.permutation(single_bin_phases[:,1])]
        ).swapaxes(1,0)
shuffle_phase_diff_raw = np.exp(-1j*np.diff(shuffle_bin_phases,axis=1))
shuffle_phase_diff = np.angle(shuffle_phase_diff_raw)
shuffle_mean_vec = np.mean(shuffle_phase_diff_raw)

#ax = plt.subplots(1,2,projection = 'polar')
scale = 3
bins = 30
fig, ax = plt.subplots(1,2, subplot_kw = dict(projection = 'polar'),
        figsize = (5,3))
rose_plot(ax[0], single_bin_phase_diff, 
        density= True , lab_unit = 'radians', bins = bins)
ax[0].quiver(
        np.zeros(phase_diff_raw.shape),
        np.zeros(phase_diff_raw.shape),
        np.real(single_mean_vec) *scale, 
        np.imag(single_mean_vec) *scale,
        scale = 5,
        linewidth = 15,
        headwidth = 10)
rose_plot(ax[1], shuffle_phase_diff, 
        density= True , lab_unit = 'radians', bins = bins)
ax[1].quiver(
        np.zeros(phase_diff_raw.shape),
        np.zeros(phase_diff_raw.shape),
        np.real(shuffle_mean_vec) *scale, 
        np.imag(shuffle_mean_vec) *scale,
        scale = 5,
        linewidth = 5,
        headwidth = 10)
ax[1].set_ylim(ax[0].get_ylim())
ax[0].set_title('Actual Data')
ax[1].set_title('Trial Shuffled Data')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir,'phase_diff_polar_hists.png'), dpi = 300)
plt.close(fig)
#plt.show()
