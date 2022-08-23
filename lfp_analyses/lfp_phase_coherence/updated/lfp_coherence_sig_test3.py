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

#mean_amp_list = []
#std_amp_list = []
#sorted_regions = []
#for this_dir in tqdm(dir_list):
#    dat = ephys_data(this_dir)
#    dat.get_stft(recalculate = False, dat_type = ['amplitude'])
#    dat.get_lfp_electrodes()
#
#    #stft_array = dat.stft_array
#    amplitude_array = dat.amplitude_array
#    mean_amplitude = amplitude_array.mean(axis=(0,2))
#    #stft_array = np.concatenate(np.swapaxes(stft_array, 1,2))
#    time_vec = dat.time_vec
#    freq_vec = dat.freq_vec
#    region_electrodes = dat.lfp_region_electrodes
#    region_amplitude = [mean_amplitude[x] for x in region_electrodes]
#    mean_region_amp = [x.mean(axis=0) for x in region_amplitude]
#    region_amp_diff = [np.abs(x-y).mean(axis=(1,2)) \
#            for x,y in zip(region_amplitude, mean_region_amp)]
#    closest_inds = [region_elecs[np.argmin(x)] \
#            for region_elecs,x in zip(region_electrodes, region_amp_diff)]
#
#    region_order = np.argsort(dat.region_names)
#    sorted_regions.append(np.array(dat.region_names)[region_order])
#
#    fin_region_dat = np.stack([amplitude_array[:,x] for x in closest_inds])
#    fin_region_dat = fin_region_dat[region_order]
#    fin_region_long = np.reshape(fin_region_dat, 
#            (fin_region_dat.shape[0], -1, *fin_region_dat.shape[3:]))
#
#    region_mean_amp = fin_region_long.mean(axis=1)
#    region_std_amp = fin_region_long.std(axis=1)
#
#    mean_amp_list.append(region_mean_amp)
#    std_amp_list.append(region_std_amp)
#
#mean_amp_array = np.stack(mean_amp_list)

# Get region orders so intra-region coherence can be sorted by region
region_order_list = []
for this_dir in tqdm(dir_list):
    dat = ephys_data(this_dir)
    dat.get_region_electrodes()
    region_order = np.argsort(dat.region_names)
    region_order_list.append(region_order)

# Write out final phase channels and channel numbers 
coherence_list = []
shuffle_list = []
intra_list = []

for this_file in tqdm(file_list):
    with tables.open_file(this_file,'r') as hf5:
        coherence_list.append(hf5.get_node(coherence_save_path)[:])
        shuffle_list.append(hf5.get_node(shuffle_save_path)[:])
        intra_list.append(hf5.get_node(intra_save_path)[:])

# Sort intra_list
intra_list = [x[:,y] for x,y in zip(intra_list, region_order_list)]

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

######################################## 
## Plot amplitude for all sessions by band 
######################################## 
#band_mean_amp = np.stack([mean_amp_array[:,:,x].mean(axis=2) for x in freq_inds])
#zscore_band_mean_amp = zscore(band_mean_amp, axis=-1)
#
#imshow_kwargs = dict(
#        vmin = -3,#band_mean_amp.min(),
#        vmax = 3,#band_mean_amp.max(),
#        interpolation = 'nearest',
#        aspect = 'auto'
#        )
#fig,ax = plt.subplots(len(band_mean_amp),2, sharex=True, sharey=True)
#for num, (this_ax, this_dat) in enumerate(zip(ax, zscore_band_mean_amp)):
#    this_ax[0].imshow(this_dat[:,0], **imshow_kwargs) 
#    this_ax[1].imshow(this_dat[:,1], **imshow_kwargs) 
#    this_ax[0].set_title(f'Freq : {freq_bands_lims[num]}')
#    this_ax[0].set_ylabel('Sessions')
#plt.tight_layout()
#plt.show()
#
#mean_zscore_band_mean = zscore_band_mean_amp.mean(axis=1)
#std_zscore_band_mean = zscore_band_mean_amp.std(axis=1)
#
#stim_t = 2
#post_stim_time = (time_vec - stim_t)*1000
#fig,ax = plt.subplots(len(band_mean_amp),2, sharex=True, sharey=True)
#for num, this_ax in enumerate(ax):
#    this_mean = mean_zscore_band_mean[num]
#    this_std = std_zscore_band_mean[num]
#    for fin_ax, fin_mean, fin_std in zip(this_ax, this_mean, this_std):
#        fin_ax.plot(post_stim_time, fin_mean)
#        fin_ax.fill_between(
#                x = post_stim_time, 
#                y1 = fin_mean + fin_std,
#                y2 = fin_mean - fin_std,
#                alpha = 0.5)
#        fin_ax.axvline(0, color = 'red', linestyle = '--')
#    this_ax[0].set_ylabel('Normalized power')
#    this_ax[1].set_ylabel(f'Freq : {freq_bands_lims[num]}')
#ax[-1,0].set_xlabel('Time post-stim (ms)')
#ax[-1,1].set_xlabel('Time post-stim (ms)')
#ax[0,0].set_title(sorted_regions[0][0])
#ax[0,1].set_title(sorted_regions[0][1])
#plt.show()


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


# Calculate summaries for intra and shuffle datasets
mean_shuffle_array = np.stack([x.mean(axis=(1)) for x in shuffle_band_list])
std_shuffle_array = np.stack([x.std(axis=(1)) for x in shuffle_band_list])

mean_intra_array = np.stack([x.mean(axis=(1)) for x in intra_band_list])
std_intra_array = np.stack([x.std(axis=(1)) for x in intra_band_list])

####################################### 
# Coherence is higher than shuffle 
####################################### 
# Mean coherence is higher than mean_shuffle + 3*std_shuffle
coherence_bool = mean_coherence_array > (mean_shuffle_array + 3*std_shuffle_array)
frac_coherence_bool = coherence_bool.mean(axis=1)

fig,ax = plt.subplots(frac_coherence_bool.shape[0],1, sharex=True, sharey=True)
for num, (this_dat, this_ax) in enumerate(zip(frac_coherence_bool, ax)):
    this_ax.plot((time_vec - baseline_t)*1000, this_dat)
    this_ax.set_title(f'Freq range : {freq_bands_lims[num]}')
    this_ax.set_ylim([0,1.1])
    this_ax.axvline(0, color = 'red', linestyle = '--', linewidth = 2,
            label = 'Stimulus delivery')
ax[1].set_ylabel('Fraction of sessions')
ax[-1].set_xlabel("Time post-stimulus delivery (ms)")
ax[-1].legend()
plt.suptitle('Fraction of sessions with mean_coherence > mean_shuffle + 3*shuffle_std' + '\n' +\
        f'Total sessions : {coherence_bool.shape[1]}')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'deviance_from_shuffle_agg.png'))
plt.close(fig)
#plt.show()

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

stim_t = 2
post_stim_time = (time_vec - stim_t)*1000

# Plot dynamics of intra-region coherence
fig,ax = plt.subplots(len(mean_intra_array),2, sharex=True, sharey=True)
inds = list(np.ndindex(ax.shape))
for this_ind in inds:
    this_dat = mean_intra_array[this_ind[0],:,this_ind[1]]
    zscore_dat = zscore(this_dat,axis=-1)
    ax[this_ind].plot(
            post_stim_time, 
            zscore_dat.mean(axis=0),
            color = 'k',
            label = 'Mean Change',
            zorder = 10)
    ax[this_ind].plot(
            post_stim_time,
            zscore_dat.T,
            alpha = 0.5,
            color = 'grey',
            zorder = 1)
    ax[this_ind].axvline(0, color = 'red', linestyle = '--', linewidth = 2)
    if this_ind[1] == 0:
        ax[this_ind].set_ylabel('Norm. Coherence')
    if this_ind[1] == 1:
        ax[this_ind].set_ylabel(f'Freq : {freq_bands_lims[this_ind[0]]}')
ax[0,0].set_title(np.sort(dat.region_names)[0])
ax[0,1].set_title(np.sort(dat.region_names)[1])
ax[-1,0].set_xlabel('Time post-stim (ms')
ax[-1,1].set_xlabel('Time post-stim (ms')
plt.suptitle('Mean intra-region coherence')
fig.savefig(os.path.join(plot_dir, 'mean_intraregion_coherence'))
plt.close(fig)
#plt.show()

# Plot mean coherence by epoch
epoch_lims = (np.stack([[1500,2000],[2000,2300],[2300,2850],[2850,3300]]) - 2000)
epoch_names = np.array(['pre_stim','stim','iden','pal'])
inds = np.array(list(np.ndindex(mean_intra_array.shape)))
intra_coherence_frame = pd.DataFrame(
        dict(
            time = post_stim_time[inds[:,-1]],
            freq_band = inds[:,0],
            session = inds[:,1],
            region = inds[:,2],
            coh = zscore(mean_intra_array,axis=-1).flatten(),
            )
        )
intra_coherence_frame = intra_coherence_frame[
        np.logical_and(
            intra_coherence_frame.time > epoch_lims.min(),
            intra_coherence_frame.time < epoch_lims.max(),
            )
        ]

intra_coherence_frame['epoch'] = pd.cut(intra_coherence_frame.time, 
                        np.unique(epoch_lims), 
                        labels = np.arange(len(np.unique(epoch_lims))-1))
intra_coherence_frame = intra_coherence_frame.\
                                groupby(['session','epoch', 'region','freq_band']).\
                                mean().\
                                reset_index(drop=False)
intra_coherence_frame['epoch_name'] = epoch_names[intra_coherence_frame.epoch]
intra_coherence_frame['region_name'] = np.sort(dat.region_names)[intra_coherence_frame.region]
freq_band_array = np.stack([str(x) for x in freq_bands_lims])
intra_coherence_frame['band_freqs'] = freq_band_array[intra_coherence_frame.freq_band]

g = sns.catplot(
        data = intra_coherence_frame,
        x = 'epoch_name',
        y = 'coh',
        col = 'region_name',
        row = 'band_freqs',
        kind = 'box'
        )
ax_inds = list(np.ndindex(g.axes.shape))
for this_ind in ax_inds:
    g.axes[this_ind].axhline(0, color = 'k', linestyle = '--', alpha = 0.5, zorder = -1)
    if this_ind[1] == 0:
        g.axes[this_ind].set_ylabel('Coherence')
    if this_ind[0] == len(freq_bands_lims)-1:
        g.axes[this_ind].set_xlabel('Epoch')
fig = plt.gcf()
plt.suptitle('Mean intra-region epoch coherence')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'mean_intraregion_epoch_coherence'))
plt.close(fig)
#plt.show()

# Plot everything
for band_num, this_band_lims in tqdm(enumerate(freq_bands_lims)):

    band_str = '-'.join([str(x) for x in this_band_lims])

    for session_num in range(mean_coherence_array.shape[1]): 
        this_basename = basenames[session_num]

        this_mean_coherence = mean_coherence_array[band_num, session_num]
        this_std_coherence = std_coherence_array[band_num, session_num]
        this_dev = dev_array[band_num, session_num]
        highlight_chunks = extract_contiguous_chunks(this_dev) 

        this_mean_shuffle = mean_shuffle_array[band_num, session_num]
        this_std_shuffle = std_shuffle_array[band_num, session_num]

        this_mean_intra = mean_intra_array[band_num, session_num]
        this_std_intra = std_intra_array[band_num, session_num]

        title_str = this_basename + '\n' + f'Freq : {band_str}'

        fig,ax = plt.subplots()
        ax.plot(time_vec, this_mean_coherence, label = 'Inter')
        ax.fill_between(x = time_vec,
                y1 = this_mean_coherence + 3*this_std_coherence,
                y2 = this_mean_coherence - 3*this_std_coherence,
                alpha = 0.5)
        ax.plot(time_vec, this_mean_shuffle, label = 'Shuffle')
        ax.fill_between(x = time_vec,
                y1 = this_mean_shuffle + 3*this_std_shuffle,
                y2 = this_mean_shuffle - 3*this_std_shuffle,
                alpha = 0.5)
        for num, (region_mean, region_std) \
                in enumerate(zip(this_mean_intra, this_std_intra)):
            ax.plot(time_vec, region_mean, label = f'Intra{num}')
            ax.fill_between(x = time_vec,
                    y1 = region_mean + region_std,
                    y2 = region_mean - region_std,
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
        ax.axvline(baseline_t, color = 'black',
                        linestyle = '--', linewidth = 2,
                        label = 'Stim Delivery')
        ax.set_ylim([0,1])
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Coherence (0-1)")
        plt.legend(loc='lower left')
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
