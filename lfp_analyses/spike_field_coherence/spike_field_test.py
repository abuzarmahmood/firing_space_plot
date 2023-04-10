# Import required modules
from pycircstat.tests import rayleigh
from scipy.stats import mannwhitneyu
import seaborn as sns
from astropy.stats import rayleightest
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
from scipy.stats import zscore
from visualize import *
from ephys_data import ephys_data
import os
import matplotlib.pyplot as plt
import tables
import numpy as np
import numpy.ma as ma
from tqdm import tqdm, trange
import shutil
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')


#import dask
#from dask.dataframe import from_pandas
#from dask.distributed import Client
#client = Client()


def parallelize(func, iterator):
    return Parallel(n_jobs=cpu_count()-2)(delayed(func)(this_iter) for this_iter in tqdm(iterator))


def custom_rayleigh(x):
    return rayleigh(x.phases)
# return rayleightest(x.phases)


def rolling_rayleigh(x, start=-1000, stop=2500, window=250, step=25):
    window_starts = np.arange(start, stop-window+step, step)
    window_ends = window_starts + window
    windows = list(zip(window_starts, window_ends))
    p_val_list = []
    for this_window in windows:
        this_dat = x.loc[(x.time >= this_window[0]) &
                         (x.time < this_window[1])]
        p_val = custom_rayleigh(this_dat)
        p_val_list.append(p_val)
    p_val_frame = np.array(p_val_list)
    # Window ends is taken as time for window, otherwise -100 will get
    # data for post-stim if window > 100
    return_frame = pd.DataFrame(dict(time=window_ends,
                                p_val=p_val_frame[:, 0],
                                z_stat=p_val_frame[:, 1]))
    z_stat = return_frame['z_stat']
    return_frame['scaled_z_stat'] = (
        z_stat - z_stat.min()) / (z_stat.max() - z_stat.min())
    return return_frame
    # return pd.DataFrame(dict(time = window_starts, p_val = p_val_list))


##################################################
## Read in data
##################################################
plot_dir = '/media/bigdata/firing_space_plot/lfp_analyses/spike_field_coherence/plots'

dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
dir_list = [x.strip() for x in open(dir_list_path, 'r').readlines()]

frame_save_path = '/stft/analyses/spike_phase_coherence'

dat = ephys_data(dir_list[0])
dat.get_stft()
freq_vec = dat.freq_vec

phase_list = []
for this_dir in tqdm(dir_list):
    dat = ephys_data(this_dir)
    phase_coherence_frame = pd.read_hdf(dat.hdf5_path, frame_save_path)
    phase_coherence_frame['basename'] = os.path.basename(dat.data_dir)
    phase_coherence_frame = phase_coherence_frame.reset_index(drop=False)
    index_list = ['basename', 'spikes_region', 'phase_region',
                  'nrn_num', 'trials', 'freq', 'time']
    phase_coherence_frame = phase_coherence_frame.set_index(index_list)
    phase_list.append(phase_coherence_frame)
fin_phase_frame = pd.concat(phase_list)

#fin_phase_frame = fin_phase_frame.reset_index()
#fin_phase_frame = from_pandas(fin_phase_frame, npartitions = 16)

#fin_phase_frame = fin_phase_frame.reset_index(level = 'time')
fin_phase_frame['time'] = fin_phase_frame['time'] - 2000
time_bool = np.logical_and(
    fin_phase_frame['time'] > -1000,
    fin_phase_frame['time'] < 2500)
fin_phase_frame = fin_phase_frame.loc[time_bool.values]


#test = fin_phase_frame.head(2000)
#bin_width = 200
#bins = np.arange(-1000, 2500, step = bin_width)
#labels = np.arange(len(bins)-1)
# def custom_cut(x):#, bins, labels):
#    return pd.cut(x['time'], bins = bins, labels = labels)

#test2 = pd.DataFrame(dict(time = np.arange(-1000,2500)))
#fin_phae_frame['bin'] = pd.cut(fin_phase_frame['time'])
#fin_phase_frame['bin'] = fin_phase_frame.map_partitions(custom_cut).compute()

# For each neuron, roll over time to generate histograms,
# aggregate across trials for tastes separately (we might merge tastes later)
#group_name_list = ['basename','spikes_region','phase_region','nrn_num','freq','bin']
group_name_list = ['basename', 'spikes_region',
                   'phase_region', 'nrn_num', 'freq']
group_obj = fin_phase_frame.groupby(group_name_list)
group_list = list(group_obj)

########################################
#fin_phase_frame_pd = fin_phase_frame.compute()
#group_list = list(fin_phase_frame_pd.groupby(group_name_list))

#test  = group_list[0][1]
#test = test[['time','phases']].reset_index(drop=True)


def temp_rolling_rayleigh(x): return rolling_rayleigh(x[1])


meta_list = [x[0] for x in group_list]
group_rolling_rayleigh = parallelize(temp_rolling_rayleigh, group_list)

for meta, dat in tqdm(zip(meta_list, group_rolling_rayleigh)):
    dat[group_name_list] = meta

group_rayleigh = pd.concat(group_rolling_rayleigh)
group_rayleigh = group_rayleigh.dropna()

########################################

#group_rayleigh = group_obj.apply(custom_rayleigh, meta = ('float')).compute()
#group_rayleigh = group_obj.apply(rolling_rayleigh).compute()

#alpha = 0.05
#sig_rayleigh = group_rayleigh.loc[group_rayleigh.p_val < alpha]
#sig_rayleigh = group_rayleigh.loc[group_rayleigh < alpha]
#sig_rayleigh.name = 'p_val'
#sig_rayleigh = sig_rayleigh.reset_index()
#sig_rayleigh['time'] = bins[sig_rayleigh.bin]

# Create unique identifier for each neuron
group_rayleigh['nrn_id'] = group_rayleigh['basename'] + '_' + \
    group_rayleigh['nrn_num'].astype(str) + group_rayleigh['spikes_region']
# Convert nrn_id to categorical number
group_rayleigh['nrn_id'] = pd.Categorical(group_rayleigh['nrn_id'])
group_rayleigh['nrn_id'] = group_rayleigh['nrn_id'].cat.codes

# Count number of unique neurons by spike region
nrn_count = group_rayleigh.groupby(
    ['spikes_region', 'nrn_id']).size().reset_index()
nrn_count = nrn_count.groupby('spikes_region').size().reset_index()
nrn_count.columns = ['spikes_region', 'nrn_count']

#imp_dat = sig_rayleigh.copy()
imp_dat = group_rayleigh[['spikes_region', 'freq', 'time', 'scaled_z_stat']]
imp_dat = imp_dat.groupby(
    ['spikes_region', 'freq', 'time']).mean().reset_index()
gc_dat = imp_dat.loc[imp_dat['spikes_region'] == 'gc']
bla_dat = imp_dat.loc[imp_dat['spikes_region'] == 'bla']

#gc_dat = gc_dat[['freq','time','p_val']]
#bla_dat = bla_dat[['freq','time','p_val']]
gc_dat = gc_dat[['freq', 'time', 'scaled_z_stat']]
bla_dat = bla_dat[['freq', 'time', 'scaled_z_stat']]
gc_dat['freq'] = freq_vec[gc_dat['freq']]
bla_dat['freq'] = freq_vec[bla_dat['freq']]

pivot_gc = gc_dat.pivot(index='freq', columns='time', values='scaled_z_stat')
pivot_bla = bla_dat.pivot(index='freq', columns='time', values='scaled_z_stat')


# sns.heatmap(pivot_gc);plt.show()
# sns.heatmap(pivot_bla);plt.show()
#
# plt.imshow(pivot_gc,
#        interpolation = 'gaussian',
#        aspect='auto', origin = 'lower');
# plt.show()

vmin = imp_dat['scaled_z_stat'].min()
vmax = imp_dat['scaled_z_stat'].max()
levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)
fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
im = ax[0].contourf(
    # im = ax[0].pcolormesh(
    pivot_gc.columns,
    pivot_gc.index,
    pivot_gc.values,
    #levels = levels,
    #vmin = vmin, vmax=vmax
)
fig.colorbar(im, ax=ax[0])
im = ax[1].contourf(
    # im = ax[1].pcolormesh(
    pivot_bla.columns,
    pivot_bla.index,
    pivot_bla.values,
    #levels = levels,
    #vmin = vmin, vmax=vmax
)
fig.colorbar(im, ax=ax[1])
ax[0].set_title('GC_spike-->BLA_field')
ax[1].set_title('BLA_spike-->GC_field')
ax[0].set_ylim([0, 15])
ax[1].set_xlabel('Time post-stim (ms)')
ax[0].set_ylabel('Frequency (Hz)')
ax[1].set_ylabel('Frequency (Hz)')
ax[0].axvline(0, color='red', linestyle='--')
ax[1].axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.suptitle('BLA-GC Spike Field Coherence')
plt.subplots_adjust(top=0.8)
# plt.show()
fig.savefig(os.path.join(plot_dir, 'raw_spike_field_coherence.png'))
plt.close(fig)

time_vals = pivot_gc.columns.values
norm_pivot_gc = pivot_gc.values
#norm_pivot_gc = norm_pivot_gc / (norm_pivot_gc[:,time_vals<-500].mean(axis=-1))[:,np.newaxis]
norm_pivot_gc = zscore(norm_pivot_gc, axis=-1)
norm_pivot_bla = pivot_bla.values
#norm_pivot_bla = norm_pivot_bla / (norm_pivot_bla[:,time_vals<-500].mean(axis=-1))[:,np.newaxis]
norm_pivot_bla = zscore(norm_pivot_bla, axis=-1)
norm_stack = np.stack([norm_pivot_gc, norm_pivot_bla])

#vmin = np.nanmin(norm_stack, axis=None)
#vmax = np.nanmax(norm_stack, axis=None)
#levels = np.logspace(np.log10(vmin),np.log10(vmax),10)
cmap = plt.get_cmap('viridis')
level_colors = cmap(levels)
fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
im = ax[0].contourf(
    # im = ax[0].pcolormesh(
    time_vals,
    freq_vec,
    norm_pivot_gc,
    #levels = levels,
    #vmin = vmin, vmax=vmax,
    #alpha = 0.3
)
# im = ax[0].contourf(
#        time_vals,
#        freq_vec,
#        ma.masked_array(norm_pivot_gc, mask = norm_pivot_gc < 1.5),
#        levels = levels,
#        vmin = vmin, vmax=vmax,
#        )
fig.colorbar(im, ax=ax[0])
im = ax[1].contourf(
    # im = ax[1].pcolormesh(
    time_vals,
    freq_vec,
    norm_pivot_bla,
    #levels = levels,
    #vmin = vmin, vmax=vmax,
    #alpha = 0.1
)
# im = ax[1].contourf(
#        time_vals,
#        freq_vec,
#        ma.masked_array(norm_pivot_bla, mask = norm_pivot_bla < 1.5),
#        levels = levels,
#        vmin = vmin, vmax=vmax,
#        )
fig.colorbar(im, ax=ax[1])
ax[0].set_title('GC_spike-->BLA_field')
ax[1].set_title('BLA_spike-->GC_field')
ax[0].set_ylim([0, 15])
ax[1].set_xlabel('Time post-stim (ms)')
ax[0].set_ylabel('Frequency (Hz)')
ax[1].set_ylabel('Frequency (Hz)')
plt.tight_layout()
plt.suptitle('BLA-GC Spike Field Coherence')
plt.subplots_adjust(top=0.8)
ax[0].axvline(0, color='red', linestyle='--')
ax[1].axvline(0, color='red', linestyle='--')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'norm_spike_field_coherence.png'))
plt.close(fig)

############################################################
# Only neurons and frequencies with significant changes
############################################################
# Select only specific timeseries by whether there is a significant
# change in the z_stat before and after stim
pre_stim_lims = [-1000, 0]
post_stim_lims = [0, 1000]

sig_pval_list = []
pre_post_list = []
for x in tqdm(group_rolling_rayleigh):
    pre_bool = (x.time >= pre_stim_lims[0]) & (x.time < pre_stim_lims[1])
    post_bool = (x.time >= post_stim_lims[0]) & (x.time < post_stim_lims[1])
    pre, post = x[pre_bool].z_stat, x[post_bool].z_stat
    _, p_val = mannwhitneyu(pre, post)
    sig_pval_list.append(p_val)
    pre_post_list.append((np.mean(pre), np.mean(post)))

pre_post_array = np.array(pre_post_list)
region_name = np.array([x.spikes_region[0] for x in group_rolling_rayleigh])
locking_diff = pre_post_array[:, 1] - pre_post_array[:, 0]

# Plot GC and BLA separately + also have a joint plot
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
ax[0].hist(locking_diff[region_name == 'gc'], bins=50)
ax[0].set_title('GC Neurons')
ax[0].set_yscale('log')
ax[0].ylabel('Frequency (log)')
ax[1].hist(locking_diff[region_name == 'bla'], bins=50)
ax[1].set_yscale('log')
ax[1].set_title('BLA Neurons')
ax[2].hist(locking_diff, bins=50)
ax[2].set_yscale('log')
ax[2].set_title('All Neurons')
ax[-1].set_xlabel('Post - Pre ...   <-- Pre Higher, Post Higher -->')
fig.suptitle("Difference in phase locking value (Rayleigh's Z)")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'post_pre_phase_locking.png'))
plt.close('all')
# plt.show()

alpha = 0.05
sig_bool_list = [x < alpha for x in sig_pval_list]

sig_group_rayleigh_list = [dat for dat, this_bool
                           in tqdm(zip(group_rolling_rayleigh, sig_bool_list)) if this_bool]
sig_group_rayleigh = pd.concat(sig_group_rayleigh_list)

imp_dat = sig_group_rayleigh[[
    'spikes_region', 'freq', 'time', 'scaled_z_stat']]
imp_dat = imp_dat.groupby(
    ['spikes_region', 'freq', 'time']).mean().reset_index()
gc_dat = imp_dat.loc[imp_dat['spikes_region'] == 'gc']
bla_dat = imp_dat.loc[imp_dat['spikes_region'] == 'bla']

gc_dat = gc_dat[['freq', 'time', 'scaled_z_stat']]
bla_dat = bla_dat[['freq', 'time', 'scaled_z_stat']]
gc_dat['freq'] = freq_vec[gc_dat['freq']]
bla_dat['freq'] = freq_vec[bla_dat['freq']]

pivot_gc = gc_dat.pivot(index='freq', columns='time', values='scaled_z_stat')
pivot_bla = bla_dat.pivot(index='freq', columns='time', values='scaled_z_stat')

pivot_gc = pivot_gc.values
pivot_bla = pivot_bla.values

# Plots by Epoch
time_vals = pivot_gc.columns.values
epochs = [[-500, 0], [0, 150], [0, 250], [250, 750], [750, 1250]]

gc_epoch_list = []
bla_epoch_list = []
for this_epoch in epochs:
    inds = (time_vals >= this_epoch[0]) & (time_vals < this_epoch[1])
    gc_epoch_list.append(pivot_gc[:, inds])
    bla_epoch_list.append(pivot_bla[:, inds])
region_list = list(zip(gc_epoch_list, bla_epoch_list))
region_names = ['gc', 'bla']

baseline_vals = region_list[0]
mean_baseline_per_region = [x.mean(axis=-1) for x in baseline_vals]
mean_baseline = np.mean(mean_baseline_per_region, axis=0)

cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(len(epochs)-1, figsize=(4, 7),
                       sharey=True, sharex=True)
for num, (dat, this_ax) in enumerate(zip(region_list[1:], ax)):
    mean_vals = [x.mean(axis=-1)-mean_baseline for x in dat]
    std_vals = [x.std(axis=-1) for x in dat]
    for region_num, (this_mean, this_std) in enumerate(zip(mean_vals, std_vals)):
        this_ax.plot(freq_vec, this_mean,
                     label=region_names[region_num].upper(),
                     color=cmap(region_num),
                     linewidth=2)
        this_ax.axhline(color = 'k', linestyle = '--',
                        linewidth = 2)
        # Plot mean (across both regions) values for baseline
        # on all plots for comparison
        #if num != 0:
        #    this_ax.plot(freq_vec, mean_baseline,
        #                 color='r', linestyle='--',
        #                 linewidth=2,
        #                 label='Baseline')
        this_ax.scatter(freq_vec, this_mean,
                        color=cmap(region_num))
        this_ax.fill_between(
            x=freq_vec,
            y1=this_mean - this_std,
            y2=this_mean + this_std,
            alpha=0.5
        )
    this_ax.set_xlim([0, 15])
    this_ax.set_ylabel('Phase Locking' + '\n' + '(A.U.)')
    #this_ax.set_title(f'Epoch : {epochs[num]}')
    # Mark epoch as text on right side of plot
    # Add box around text
    this_ax.text(1.02, 0.5, f'{epochs[num+1][0]} - {epochs[num+1][1]} ms',
                 rotation=270, transform=this_ax.transAxes,
                 verticalalignment='center', horizontalalignment='left',
                 bbox=dict(facecolor='white',
                           edgecolor='black', boxstyle='round'),
                 )
# Add legend with title
ax[0].legend(title='Spike Region',
             facecolor='white', framealpha=0.1,
             loc='upper right',)
## Get handler for legend from axes
#handles = ax[1].get_legend_handles_labels()[0]
## Pull out handle for baseline
#baseline_handle = handles[-1]
## Plot legend on ax[1]
#ax[1].legend([baseline_handle], ['Baseline'], loc='upper left',
#             facecolor = 'white', framealpha = 0.1,)
ax[-1].set_xlabel('Frequency (Hz)')
fig.suptitle('Phase Locking by epoch')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'phase_locking_by_epoch'),
            dpi=300)
plt.close(fig)
# plt.show()


########################################
# 2D Plots
########################################
vmin = imp_dat['scaled_z_stat'].min()
vmax = imp_dat['scaled_z_stat'].max()
levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)
fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
im = ax[0].contourf(
    # im = ax[0].pcolormesh(
    time_vals,
    freq_vec,
    pivot_gc,
    #levels = levels,
    #vmin = vmin, vmax=vmax
)
fig.colorbar(im, ax=ax[0])
im = ax[1].contourf(
    # im = ax[1].pcolormesh(
    time_vals,
    freq_vec,
    pivot_bla,
    #levels = levels,
    #vmin = vmin, vmax=vmax
)
fig.colorbar(im, ax=ax[1])
ax[0].set_title('GC_spike-->BLA_field')
ax[1].set_title('BLA_spike-->GC_field')
ax[0].set_ylim([0, 15])
ax[1].set_xlabel('Time post-stim (ms)')
ax[0].set_ylabel('Frequency (Hz)')
ax[1].set_ylabel('Frequency (Hz)')
ax[0].axvline(0, color='red', linestyle='--')
ax[1].axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.suptitle('BLA-GC Spike Field Coherence')
plt.subplots_adjust(top=0.8)
# plt.show()
fig.savefig(os.path.join(plot_dir, 'sig_only_raw_spike_field_coherence.png'))
plt.close(fig)

norm_pivot_gc = pivot_gc.values
norm_pivot_gc = zscore(norm_pivot_gc, axis=-1)
norm_pivot_bla = pivot_bla.values
norm_pivot_bla = zscore(norm_pivot_bla, axis=-1)
norm_stack = np.stack([norm_pivot_gc, norm_pivot_bla])

#vmin = np.nanmin(norm_stack, axis=None)
#vmax = np.nanmax(norm_stack, axis=None)
#levels = np.logspace(np.log10(vmin),np.log10(vmax),10)
cmap = plt.get_cmap('viridis')
level_colors = cmap(levels)
fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
im = ax[0].contourf(
    # im = ax[0].pcolormesh(
    time_vals,
    freq_vec,
    norm_pivot_gc,
    #levels = levels,
    #vmin = vmin, vmax=vmax,
    #alpha = 0.3
)
fig.colorbar(im, ax=ax[0])
im = ax[1].contourf(
    # im = ax[1].pcolormesh(
    time_vals,
    freq_vec,
    norm_pivot_bla,
    #levels = levels,
    #vmin = vmin, vmax=vmax,
    #alpha = 0.1
)
fig.colorbar(im, ax=ax[1])
ax[0].set_title('GC_spike-->BLA_field')
ax[1].set_title('BLA_spike-->GC_field')
ax[0].set_ylim([0, 15])
ax[1].set_xlabel('Time post-stim (ms)')
ax[0].set_ylabel('Frequency (Hz)')
ax[1].set_ylabel('Frequency (Hz)')
plt.tight_layout()
plt.suptitle('BLA-GC Spike Field Coherence')
plt.subplots_adjust(top=0.8)
ax[0].axvline(0, color='red', linestyle='--')
ax[1].axvline(0, color='red', linestyle='--')
# plt.show()
fig.savefig(os.path.join(plot_dir, 'sig_only_norm_spike_field_coherence.png'))
plt.close(fig)
