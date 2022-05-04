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
import visualize as vz
from scipy.stats import zscore
from scipy.signal import savgol_filter
from pathlib import Path
from joblib import Parallel, delayed, cpu_count
import matplotlib as mpl
import pandas as pd
import pingouin as pg
import seaborn as sns

# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(
                os.path.dirname(path_to_node),os.path.basename(path_to_node))

##################################################
## Read in data 
##################################################

plot_dir='/media/bigdata/firing_space_plot/lfp_analyses/lfp_phase_coherence/plots'

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in dir_list]
file_list = [str(list(Path(x).glob('*.h5'))[0]) for x in dir_list] 

coherence_save_path = '/stft/analyses/phase_coherence/diff_phase_coherence_array'
shuffle_save_path = '/stft/analyses/phase_coherence/diff_shuffle_phase_coherence_array'
intra_save_path = '/stft/analyses/phase_coherence/diff_intra_phase_coherence_array'

# Write out final phase channels and channel numbers 
coherence_list = []
shuffle_list = []
intra_list = []

for this_file in tqdm(file_list):
    with tables.open_file(this_file,'r') as hf5:
        coherence_list.append(hf5.get_node(coherence_save_path)[:])
        shuffle_list.append(hf5.get_node(shuffle_save_path)[:])
        intra_list.append(hf5.get_node(intra_save_path)[:])

coherence_array = np.stack(coherence_list)
shuffle_array = np.stack(shuffle_list)
intra_array = np.stack(intra_list)

## Smooth out by convolving with flat kernel
#kern_len = 1000
#kern = np.ones(kern_len)/kern_len
#inds = list(np.ndindex(coherence_array.shape[:-1]))
#for this_ind in tqdm(inds):
#    coherence_array[this_ind] = np.convolve(kern, coherence_array[this_ind], mode = 'same')

# Get time and freq_vecs
with tables.open_file(this_file,'r') as hf5:
    time_vec = hf5.get_node('/stft/time_vec')[:]
    freq_vec = hf5.get_node('/stft/freq_vec')[:]

# Convert to bands
freq_bands_lims = [[3,8] , [8,13], [13, 19]] 
freq_bands = [np.arange(*x) for x in freq_bands_lims]
freq_inds = [np.array([i for i,val in enumerate(freq_vec) if int(val) in band]) \
                    for band in freq_bands]

coherence_band_list = np.stack([coherence_array[:,:,x].mean(axis=2) for x in freq_inds])
shuffle_band_list = np.stack([shuffle_array[:,:,x].mean(axis=2) for x in freq_inds])
intra_band_list = np.stack([intra_array[:,:,:,x].mean(axis=3) for x in freq_inds])
#mean_band_coherence = np.stack([x.mean(axis=2) for x in coherence_band_list])

####################################### 
# Comparison of shuffle vs actual for evoked response 
####################################### 
evoked_t_lims = [2,4]
evoked_inds = np.logical_and(time_vec >= evoked_t_lims[0],
                            time_vec <= evoked_t_lims[1])

evoked_coherence = coherence_band_list[..., evoked_inds].mean(axis = (2,3))
evoked_shuffle = shuffle_band_list[..., evoked_inds].mean(axis = (2,3))
evoked_intra = intra_band_list[..., evoked_inds].mean(axis = (2,4))
inds = np.array(list(np.ndindex(evoked_coherence.shape)))
evoked_frame = pd.DataFrame(dict(
                    band = inds[:,0],
                    session = inds[:,1],
                    comp_type = 'Inter-Region',
                    values = evoked_coherence.flatten()))
shuffle_frame = pd.DataFrame(dict(
                    band = inds[:,0],
                    session = inds[:,1],
                    comp_type = 'Shuffle',
                    values = evoked_shuffle.flatten()))
inds = np.array(list(np.ndindex(evoked_intra.shape)))
intra_frame = pd.DataFrame(dict(
                    band = inds[:,0],
                    session = inds[:,1],
                    region = inds[:,2],
                    comp_type = 'Intra-Region',
                    values = evoked_intra.flatten()))
fin_frame = pd.concat([evoked_frame, shuffle_frame, intra_frame])
freq_map = np.array(['Theta', 'Alpha', 'Beta'])
fin_frame['freq_name'] = freq_map[fin_frame['band']]

g = sns.stripplot(data = fin_frame, x = 'freq_name', y = 'values', 
        hue = 'comp_type', size = 7, alpha = 0.5, dodge = True, color = 'grey',
                    hue_order = ['Inter-Region','Intra-Region','Shuffle'])
sns.boxplot(data = fin_frame, x = 'freq_name', y = 'values', hue = 'comp_type',
                    ax = g, showfliers=False,
                    hue_order = ['Inter-Region','Intra-Region','Shuffle'])
handles, labels = g.get_legend_handles_labels()
handles = handles[:3]
labels = [x.title() for x in labels[:3]]
g.legend(handles,labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
        title = 'Comparison Type')
plt.tight_layout()
plt.xlabel('Frequency Band')
plt.ylabel('Avg. Phase Coherence')
fin_xticklabels = [x.get_text().title() for x in g.get_xticklabels()]
g.set_xticklabels(fin_xticklabels)
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir,'whole_trial_comparison'), dpi = 300, format = 'svg')
plt.close(fig)
#plt.show()

####################################### 
# Difference from baseline
####################################### 
# Pool baseline coherence and conduct tests on non-overlapping bins
## ** time_vec is already defined **
baseline_t = 2 #ms
baseline_range = np.array((1250,1750))/1000
baseline_inds = np.where(
        np.logical_and(time_vec>baseline_range[0], time_vec<baseline_range[1]))[0]

baseline_dat = [x[...,baseline_inds] for x in coherence_band_list]
#baseline_dat = np.stack([x[...,baseline_inds] for x in mean_band_coherence])

ci_interval = [2.5, 97.5]
#ci_array = np.empty((*baseline_dat.shape[:2] , len(ci_interval)))
#inds = list(np.ndindex(ci_array.shape[:-1]))
#for this_ind in inds:
#    ci_array[this_ind] = np.percentile(baseline_dat[this_ind], ci_interval)
ci_array = np.stack([[
    np.percentile(this_session, ci_interval) for this_session in this_band] \
                            for this_band in baseline_dat])

# Find when mean value deviates from baseline bounds
mean_coherence_array = np.stack([x.mean(axis=(1)) for x in coherence_band_list])
std_coherence_array = np.stack([x.std(axis=(1)) for x in coherence_band_list])
dev_array = \
        (mean_coherence_array < ci_array[...,0][...,np.newaxis]) * -1 +\
        (mean_coherence_array > ci_array[...,1][...,np.newaxis]) * +1


#fig, ax = plt.subplots(len(mean_coherence_array))
#for num, this_dat in enumerate(mean_coherence_array):
#    this_mean = this_dat.mean(axis=0)
#    this_std = this_dat.std(axis=0)
#    ax[num].plot(time_vec, this_mean, linewidth = 2, zorder = 3)
#    ax[num].fill_between(x = time_vec,
#                            y1 = this_mean + this_std,
#                            y2 = this_mean - this_std,
#                            alpha = 0.7, zorder = 2)
#    for x in this_dat:
#        ax[num].plot(time_vec, x, color = 'k', alpha = 0.3, 
#                linewidth = 0.5, zorder = 1)
#plt.show()

##############################
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
##############################
# Create plots for:
#   1) Average coherence for different bands per animal
#   2) Aggregate deviation of coherence from baseline

def extract_contiguous_chunks(x):
    chunks = []
    this_chunk = []
    last_true = 0
    for num,i in enumerate(x):
        if i != 0:
            this_chunk.append(num)
            last_true = 1
        else:
            if last_true == 1:
                chunks.append(this_chunk)
                last_true = 0
            this_chunk = []
    return [np.array(x) for x in chunks]

# Get rid of deviations less than threshold
dev_len_thresh = 100
inds = list(np.ndindex(dev_array.shape[:-1]))
all_lens = []
for this_ind in tqdm(inds):
    this_dev = dev_array[this_ind]
    chunks = extract_contiguous_chunks(this_dev)
    all_lens.append([len(x) for x in chunks])
all_lens = [x for y in all_lens for x in y]

for this_ind in tqdm(inds):
    this_dev = dev_array[this_ind]
    chunks = extract_contiguous_chunks(this_dev)
    chunk_bool = [len(x) < dev_len_thresh for x in chunks]
    for this_bool, chunk_inds in zip(chunk_bool, chunks):
        if this_bool:
            this_dev[np.array(chunk_inds)] = 0
    dev_array[this_ind] = this_dev

########################################
## Coherence Plots
########################################

for band_num, this_band_lims in tqdm(enumerate(freq_bands_lims)):

    band_str = '-'.join([str(x) for x in this_band_lims])

    for session_num in range(mean_coherence_array.shape[1]): 
        this_basename = basenames[session_num]

        this_mean_coherence = mean_coherence_array[band_num, session_num]
        this_std_coherence = std_coherence_array[band_num, session_num]
        this_dev = dev_array[band_num, session_num]
        highlight_chunks = extract_contiguous_chunks(this_dev) 

        title_str = this_basename + '\n' + f'Freq : {band_str}'

        fig,ax = plt.subplots()
        ax.plot(time_vec, this_mean_coherence)
        ax.fill_between(x = time_vec,
                y1 = this_mean_coherence + this_std_coherence,
                y2 = this_mean_coherence - this_std_coherence,
                alpha = 0.5)
        for this_lim in ci_array[band_num, session_num]:
            ax.axhline(this_lim, color = 'red')
        if len(highlight_chunks):
            for this_chunk in highlight_chunks:
                ax.axvspan(time_vec[this_chunk.min()], 
                        time_vec[this_chunk.max()], 
                        color = 'yellow', alpha = 0.5)
        ax.axvline(time_vec[baseline_inds[-1]], color = 'red',
                        linestyle = '--')
        ax.axvline(time_vec[baseline_inds[0]], color = 'red',
                        linestyle = '--')
        fig.suptitle(title_str)
        fig.savefig(os.path.join(plot_dir,
                        "_".join([f'freq_{band_str}', this_basename])))
        plt.close(fig)
        #plt.show()

########################################
## Deviance Plots
########################################
#fig, ax = plt.subplots(2, len(freq_bands[:2]), sharex=True)
#for col, dat in enumerate(dev_array):
#    im = ax[0,col].imshow(dat, interpolation = 'nearest', aspect = 'auto')
#    ax[1,col].plot(np.mean(np.abs(dat) > 0, axis=0))
#    ax[0,col].set_title('-'.join([str(x) for x in freq_bands_lims[col]]))
#plt.show()
#
#epoch_lims = [[2000,2300],[2300,2750],[2750,3250]]
#colors = ['pink','orange','purple']
## To compare timeseries, plot on top of eachother
#for col, dat in enumerate(dev_array[:2]):
#    wanted_data = np.mean(np.abs(dat) > 0, axis=0)
#    filtered_data = savgol_filter(wanted_data, 101, 2)
#    plt.plot(filtered_data, 
#        label = '-'.join([str(x) for x in freq_bands_lims[col]]),
#        linewidth = 2)
#for epoch_num, epoch_lims in enumerate(epoch_lims):
#    plt.axvspan(epoch_lims[0], epoch_lims[1], color=colors[epoch_num], alpha = 0.3)
#plt.xlim([1000, 4500])
#plt.legend()
#plt.show()

############################################################

cmap = mpl.cm.viridis
bounds = [-1, 0, 1, 2] 
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
epoch_lims = (np.stack([[2000,2300],[2300,2850],[2850,3300]]) - 2000)

epoch_colors = ['pink','orange','purple']

#x,y = np.meshgrid(np.arange(dev_array.shape[1]), np.arange(dev_array.shape[2]))
x = (time_vec*1000) - 2000 #np.arange(dev_array.shape[-1])
y = np.arange(dev_array.shape[1])
xlims = [-1000, 2500]

fig, ax = plt.subplots(2, len(freq_bands[:2]), 
        sharex=True, figsize = (15,10))
for col, dat in enumerate(dev_array[:2]):
    #im = ax[0,col].imshow(dat, interpolation = 'nearest', aspect = 'auto',
    #                cmap = cmap)#, norm = norm)
    im = ax[0,col].pcolormesh(x,y,dat,cmap = cmap, shading = 'nearest')
    #ax[1,col].plot(np.mean(np.abs(dat) > 0, axis=0))
    title_str = '-'.join([str(x) for x in freq_bands_lims[col]])
    ax[0,col].set_title(f'Freq : {title_str}')
    wanted_data = np.mean(np.abs(dat) > 0, axis=0)
    filtered_data = savgol_filter(wanted_data, 101, 2)
    ax[1, col].plot(x, filtered_data, 
            linewidth = 3, color = 'k', zorder = 2)
    for epoch_num, this_epoch_lims in enumerate(epoch_lims):
        ax[1, col].axvspan(this_epoch_lims[0], this_epoch_lims[1], 
                color=epoch_colors[epoch_num], alpha = 0.5, zorder = 1)
    ax[0, col].set_xlim(xlims)
    ax[1, col].set_xlim(xlims)
    ax[1, col].set_xlabel('Time post-stim (ms)')
ax[0, 0].set_ylabel('Coherence change across all session')
ax[1, 0].set_ylabel('Fraction of session with change')
cax = fig.add_axes([0.82, 0.5, 0.02, 0.45])
#cbar = plt.colorbar(im, cax=cax, ticks = [-1, 0, 1]) 
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax,
        ticks = [-0.5, 0.5, 1.5]) 
cbar.set_ticklabels(['Decrease', 'No Change', 'Increase'])  
plt.subplots_adjust(right=0.8)
fig.suptitle('BLA-GC Phase Coherence Dynamics')
fig.savefig(os.path.join(plot_dir,'aggregate_phase_coherence'),
                dpi = 300, format = 'png')
plt.close(fig)
#plt.show()
