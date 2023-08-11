"""
Use Rayleigh test to generate plots of:
    1) zscored rayleigh statistic
    2) zscored p-value

"""

# Import required modules
from pycircstat.tests import rayleigh
from scipy.stats import mannwhitneyu
import seaborn as sns
# from astropy.stats import rayleightest
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import tables
import numpy as np
import numpy.ma as ma
from tqdm import tqdm, trange
import shutil
import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *
import pingouin as pg
from scipy.interpolate import interp1d
from scipy.optimize import minimize

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
    return_frame = pd.DataFrame(
            dict(
                time=window_ends,
                p_val=p_val_frame[:, 0],
                z_stat=p_val_frame[:, 1],
                time_bins = list(zip(window_starts, window_ends))
                     )
    )
    z_stat = return_frame['z_stat']
    return_frame['scaled_z_stat'] = (
        z_stat - z_stat.min()) / (z_stat.max() - z_stat.min())
    return return_frame
    # return pd.DataFrame(dict(time = window_starts, p_val = p_val_list))

def interp_phase(df, t_vec):
    t = df.time.values/1000
    phi = np.unwrap(df.phases.values)
    f = interp1d(t, phi, kind='linear', fill_value='extrapolate')
    phi_vec = f(t_vec)
    return phi_vec

def wrap_phase(phase_vec):
    """
    Wrap phase vector to -pi to pi
    """
    phase_vec = np.mod(phase_vec, 2*np.pi)
    phase_vec[phase_vec > np.pi] = phase_vec[phase_vec > np.pi] - 2*np.pi
    return phase_vec

# Align phases per trial
def return_alignment_times(phase_array, t_vec, freq):
    """
    Find most common start and end points of phase across trials
    Stretch everything to fit in that window
    """
    unwrap_phase = np.unwrap(phase_array, axis = 1)
    start_phi = unwrap_phase[:,0]
    set_start_phi = np.median(start_phi)
    delta_phi = set_start_phi - start_phi 

    delta_t = delta_phi / (2*np.pi*freq)
    new_t_vecs = np.stack([t_vec - x for x in delta_t])

    # Now stretch or shrink each trial to fit in the window
    set_end_phi = np.median(unwrap_phase[:,-1])
    set_delta_phi = set_end_phi - set_start_phi
    set_time_taken = set_delta_phi / (2*np.pi*freq)

    ind_at_set_end = np.argmin(np.abs(unwrap_phase - set_end_phi), axis = 1)
    t_at_set_end = np.array([x[y] for x,y in zip(new_t_vecs, ind_at_set_end)])
    new_t_vecs = np.stack([(this_t / this_end)*set_time_taken \
            for this_t, this_end in zip(new_t_vecs, t_at_set_end)])

    return new_t_vecs

def remove_outliers(X, thresh = 3):
    """
    Remove outliers using distance from median

    Input:
        X: samples x features
        thresh: number of median absolute deviations from median to consider outlier

    Output:
        outlier_inds: indices of outliers
        non_outlier_inds: indices of non-outliers
    """
    median = np.median(X, axis = 0)
    dist_from_median = np.abs(X - median)
    sum_dist_from_median = dist_from_median.sum(axis=-1)
    median_dist_from_median = np.median(sum_dist_from_median)
    MAD = np.median(np.abs(sum_dist_from_median - median_dist_from_median))
    outlier_inds = np.where(sum_dist_from_median > median_dist_from_median + thresh*MAD)[0]
    non_outlier_inds = np.where(sum_dist_from_median <= median_dist_from_median + thresh*MAD)[0]
    return outlier_inds, non_outlier_inds

# Make array with aligned phases
def register_phases(phase_array, aligned_t_vec):
    min_t = aligned_t_vec.min(axis=None)
    max_t = aligned_t_vec.max(axis=None)
    new_t_vec = np.linspace(min_t, max_t, 100)
    new_phase_array = np.stack([np.interp(new_t_vec, this_t, this_phase) \
            for this_t, this_phase in zip(aligned_t_vec, phase_array)])
    return new_phase_array, new_t_vec

def register_spikes(df, aligned_t_vec, t_vec):
    """
    Take positions of spikes according to old t_vec and put them in aligned_t_vec
    """
    spike_times = df.time.values / 1000
    old_t_vec_inds = np.argmin(np.abs(t_vec - spike_times[:,None]), axis = 1)
    new_t_vec_times = aligned_t_vec[old_t_vec_inds]
    return new_t_vec_times

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

fin_phase_frame = fin_phase_frame.reset_index(level = 'time')
fin_phase_frame['time'] = fin_phase_frame['time'] - 2000
time_bool = np.logical_and(
    fin_phase_frame['time'] > -1000,
    fin_phase_frame['time'] < 2500)
fin_phase_frame = fin_phase_frame.loc[time_bool.values]

# Convert freq_ind to actual freq
fin_phase_frame.reset_index(inplace=True, drop=False, level = 'freq')
fin_phase_frame['freq'] = freq_vec[fin_phase_frame['freq']]
fin_phase_frame['freq'] = fin_phase_frame['freq'].astype(int)
fin_phase_frame.set_index('freq', append=True, inplace=True)

# Plot histogram of spike counts
spike_counts = fin_phase_frame.reset_index(level=['spikes_region', 'freq'])
spike_counts = spike_counts[spike_counts['freq'] == 0]
spike_counts.reset_index(inplace=True, drop=False)
spike_counts = spike_counts.groupby(['spikes_region','basename','nrn_num']).count().freq.reset_index()
spike_counts = spike_counts.rename(columns = {'freq':'spike_count'})

# Plot histogram with x-axis on log scale
pval = pg.mwu(*[x[1].spike_count for x in list(spike_counts.groupby('spikes_region'))])['p-val'].values[0]
sns.histplot(data = spike_counts, x = 'spike_count', hue = 'spikes_region', bins = 50, log_scale = True)
plt.title('Total spikes per neuron per region\n' + f'MWU p = {pval:.3f}')
plt.savefig(os.path.join(plot_dir, 'spike_rate_distributions.png'))
plt.close()

# For each neuron, roll over time to generate histograms,
# aggregate across trials for tastes separately (we might merge tastes later)
group_name_list = ['basename', 'spikes_region',
                   'phase_region', 'nrn_num', 'freq']
group_obj = fin_phase_frame.groupby(group_name_list)
group_list = list(group_obj)

########################################
# Run rayleigh test on each neuron

window_size = 250
step_size = 250
def temp_rolling_rayleigh(x): 
    return rolling_rayleigh(x[1], window = window_size, step = step_size)

meta_list = [x[0] for x in group_list]
group_rolling_rayleigh = parallelize(temp_rolling_rayleigh, group_list)

for meta, dat in tqdm(zip(meta_list, group_rolling_rayleigh)):
    dat[group_name_list] = meta

group_rayleigh = pd.concat(group_rolling_rayleigh)
group_rayleigh = group_rayleigh.dropna()

########################################
# Create unique identifier for each neuron
group_rayleigh['nrn_id'] = group_rayleigh['basename'] + '_' + \
    group_rayleigh['nrn_num'].astype(str) + group_rayleigh['spikes_region']
# Convert nrn_id to categorical number
group_rayleigh['nrn_id'] = pd.Categorical(group_rayleigh['nrn_id'])

nrn_id_map = dict(enumerate(group_rayleigh['nrn_id'].cat.categories))
group_rayleigh['nrn_id'] = group_rayleigh['nrn_id'].cat.codes

# Convert freq (which is inds) to freq_vec
#group_rayleigh['freq'] = freq_vec[group_rayleigh['freq']]
#group_rayleigh['freq'] = group_rayleigh['freq'].astype(int)

###############
# Significant p-values
alpha = 0.05
group_rayleigh['p_val_sig'] = group_rayleigh['p_val'] < alpha

###############
# Drop irrelevant columns
group_rayleigh.drop(columns=['basename', 'nrn_num'], inplace=True)

# Count number of unique neurons by spike region
nrn_count = group_rayleigh.groupby(
    ['spikes_region', 'nrn_id']).size().reset_index()
nrn_count = nrn_count.groupby('spikes_region').size().reset_index()
nrn_count.columns = ['spikes_region', 'nrn_count']

# Histogram of average Rayleigh Z-statistic per neuron per region
mean_rayleigh = group_rayleigh.groupby(
    ['spikes_region', 'nrn_id']).mean().reset_index()
pval = pg.mwu(*[x[1].z_stat for x in list(mean_rayleigh.groupby('spikes_region'))])['p-val'].values[0]
sns.histplot(data=mean_rayleigh, x='z_stat', hue='spikes_region', bins=50, log_scale=True)
plt.title('Average Rayleigh Z-statistic per neuron per region\n' + f'MWU p = {pval:.3f}')
plt.savefig(os.path.join(plot_dir, 'rayleigh_z_stat_distributions.png'))
plt.close()

########################################
# Further analysis using scaled_z_stat
subset_cols = ['spikes_region', 'phase_region','freq', 
               'time', 'scaled_z_stat', 'p_val_sig']
imp_dat = group_rayleigh[subset_cols]
imp_dat = imp_dat.groupby(subset_cols[:-2]).mean().reset_index()
region_dat = [imp_dat[imp_dat['spikes_region'] == x] for x in imp_dat['spikes_region'].unique()]

# Generate timeseries of scaled_z_stat
region_z_stat_pivot = [x.pivot(index='freq', columns='time', values='scaled_z_stat') for x in region_dat]
region_pval_sig_pivot = [x.pivot(index='freq', columns='time', values='p_val_sig') for x in region_dat]
direction_names = [f'{x.spikes_region.unique()[0].upper()} spikes --> {x.phase_region.unique()[0].upper()} field' for x in region_dat]

########################################
# Plot heatmap of scaled_z_stat
time_vec = group_rayleigh.time_bins.unique()
time_vec = np.array([x[1] for x in time_vec]) + window_size/2
vmin = imp_dat['scaled_z_stat'].min()
vmax = imp_dat['scaled_z_stat'].max()
fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
for num, (this_dat, this_ax) in enumerate(zip(region_z_stat_pivot, ax)):
    im = this_ax.pcolormesh(time_vec, freq_vec, this_dat,
                            edgecolors = 'k', linewidths = 0.05, 
                            vmin = vmin, vmax = vmax,
                            cmap = 'jet') 
    this_ax.set_ylim([-0.5, 15.5])
    this_ax.axvline(0, color='red', linestyle='--', linewidth = 4)
    this_ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=this_ax)
    this_ax.set_title(direction_names[num])
ax[1].set_xlabel('Time post-stim (ms)')
plt.tight_layout()
plt.suptitle('BLA-GC Spike Field Coherence')
plt.subplots_adjust(top=0.8)
# plt.show()
fig.savefig(os.path.join(plot_dir, 'raw_spike_field_coherence.png'))
plt.close(fig)

# Genreate a plot with baseline subtracted
time_vec = group_rayleigh.time_bins.unique()
time_vec = np.array([x[1] for x in time_vec]) + window_size/2
fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
for num, (this_dat, this_ax) in enumerate(zip(region_z_stat_pivot, ax)):
    baseline_dat = this_dat.values[:, time_vec < 0].mean(axis=1)
    baseline_sub_dat = this_dat.values - baseline_dat[:, np.newaxis]
    im = this_ax.pcolormesh(time_vec, freq_vec, baseline_sub_dat,
                            edgecolors = 'k', linewidths = 0.05, 
                            cmap = 'jet',
                            vmin = 0, vmax = 0.5) 
    this_ax.set_ylim([-0.5, 15.5])
    this_ax.axvline(0, color='red', linestyle='--', linewidth = 4)
    this_ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=this_ax)
    this_ax.set_title(direction_names[num])
ax[1].set_xlabel('Time post-stim (ms)')
plt.tight_layout()
plt.suptitle('BLA-GC Spike Field Coherence\nNote: Color limits slightly different from those shown')
plt.subplots_adjust(top=0.8)
# plt.show()
fig.savefig(os.path.join(plot_dir, 'baseline_sub_spike_field_coh.png'))
plt.close(fig)

# Plot heatmap of p_val_sig
time_vec = group_rayleigh.time_bins.unique()
time_vec = np.array([x[1] for x in time_vec]) + window_size/2
vmin = imp_dat['p_val_sig'].min()
vmax = imp_dat['p_val_sig'].max()
fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
for num, (this_dat, this_ax) in enumerate(zip(region_pval_sig_pivot, ax)):
    im = this_ax.pcolormesh(time_vec, freq_vec, this_dat,
                            edgecolors = 'k', linewidths = 0.05, 
                            vmin = vmin, vmax = vmax,
                            cmap = 'jet') 
    this_ax.set_ylim([-0.5, 15.5])
    this_ax.axvline(0, color='red', linestyle='--', linewidth = 4)
    this_ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=this_ax)
    this_ax.set_title(direction_names[num])
ax[1].set_xlabel('Time post-stim (ms)')
plt.tight_layout()
plt.suptitle('BLA-GC Spike Field Coherence Significance')
plt.subplots_adjust(top=0.8)
# plt.show()
fig.savefig(os.path.join(plot_dir, 'spike_field_sig_pval.png'))
plt.close(fig)

##############################
# Single Neuron Examples of phase locking per frequency
# (collapse across time)

# Find neurons with highest phase locking
# The ids can be cross references to the nrn_id_map
mean_nrn_rayleigh = group_rayleigh.groupby('nrn_id').mean()
mean_nrn_rayleigh.reset_index(inplace=True, drop=False)
mean_nrn_rayleigh.sort_values('z_stat', ascending=False, inplace=True)

# The nrn_id's we want
wanted_raw_nrn_id = [nrn_id_map[x] for x in mean_nrn_rayleigh['nrn_id'].values]

# Will have to generate similar nrn_id raw to cross-reference to nrn_id_map
fin_phase_frame.reset_index(inplace=True, drop=False)
nrn_group_list = list(fin_phase_frame.groupby(['basename','spikes_region','phase_region','nrn_num']))
nrn_group_list_inds = [x[0] for x in nrn_group_list]
nrn_group_list_nrn_id = [x[0]+'_'+str(x[-1])+x[1] for x in nrn_group_list_inds]
nrn_group_list_dat = [x[1] for x in nrn_group_list]
for this_dat in tqdm(nrn_group_list_dat):
    this_dat['nrn_id'] = this_dat['basename'] + '_' + \
        this_dat['nrn_num'].astype(str) + this_dat['spikes_region']

# These are the inds in the nrn_group_list_dat in descending order of mean z_stat
wanted_nrn_group_inds = \
        [[i for i,x in enumerate(nrn_group_list_nrn_id) if x == y] \
        for y in tqdm(wanted_raw_nrn_id)]
wanted_nrn_group_inds = [x for y in wanted_nrn_group_inds for x in y]

for this_ind in wanted_nrn_group_inds[:10]:
    this_dat = nrn_group_list_dat[this_ind].loc[:, ['phases','freq']]
    # Drop 0 hz
    this_dat = this_dat[this_dat['freq'] != 0]

    plt.hist2d(this_dat['phases'], this_dat['freq'], bins = 50)
    plt.title(nrn_group_list_nrn_id[this_ind])

    # 2D polar histogram with phase as angle, and freq as radius
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    im = ax.hist2d(
            this_dat['phases'], 
            this_dat['freq'], 
            )
    ax.set_title(nrn_group_list_nrn_id[this_ind])
    fig.colorbar(im[3], ax=ax)
    plt.show()


###############
test_nrn = group_rayleigh[group_rayleigh.nrn_id == 54]
test_nrn_pivot = test_nrn.pivot(index='freq', columns='time_bins', values='z_stat')

plt.imshow(np.log10(test_nrn_pivot), aspect='auto', cmap='jet' ,origin='lower')
plt.colorbar()
plt.show()

test_nrn_raw_dat = nrn_group_list_dat[wanted_nrn_group_inds[0]]
test_nrn_raw_dat = test_nrn_raw_dat.loc[:, ['time','freq','phases','trials']]
# Drop 0 Hz
test_nrn_raw_dat = test_nrn_raw_dat[test_nrn_raw_dat.freq != 0]

# Time between 0 and 250
# Freq between 0 and 10
max_t = 500
wanted_freq = 4
test_nrn_raw_dat = test_nrn_raw_dat[(test_nrn_raw_dat.time > 0) & (test_nrn_raw_dat.time < max_t)]
test_nrn_raw_dat = test_nrn_raw_dat[test_nrn_raw_dat.freq == wanted_freq]

# Generate twilight colormap from -pi to pi
cmap = plt.get_cmap('twilight')
norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

# Plot raw dat for a frequency across trials
plot_dat = test_nrn_raw_dat[test_nrn_raw_dat.trials < 30]

# For each trial, interpolate phases
# Every time phase decreases, add 2pi
group_plot_dat = [x[1] for x in list(plot_dat.groupby('trials'))]

t_vec = np.linspace(0,max_t/1000)
group_trial_phase = np.stack([interp_phase(x,t_vec) for x in group_plot_dat])
group_trial_phase = np.stack([wrap_phase(x) for x in group_trial_phase])

aligned_t_vec = return_alignment_times(group_trial_phase, t_vec, 2)

out_inds, nonout_inds = remove_outliers(aligned_t_vec)
aligned_t_vec = aligned_t_vec[nonout_inds]
phase_array = group_trial_phase[nonout_inds]
fin_group_plot_dat = [group_plot_dat[i] for i in nonout_inds]
fin_plot_dat = plot_dat[plot_dat.trials.isin(nonout_inds)]
trial_map = dict(zip(nonout_inds, np.arange(len(nonout_inds))))
fin_plot_dat['cont_trials'] = [trial_map[x] for x in fin_plot_dat.trials] 

new_phase_array, new_t_vec = register_phases(phase_array, aligned_t_vec)
new_spike_times = [register_spikes(x, aligned_t_vec[i], t_vec) for i,x in enumerate(fin_group_plot_dat)]
flat_new_spike_times = [x for y in new_spike_times for x in y]

fig,ax = plt.subplots(2,2, sharex=True, sharey=False)
ax[0,0].hist(fin_plot_dat.time.values.flatten()/1000, bins = 20)
ax[0,1].hist(flat_new_spike_times, bins = 20)
im = ax[1,0].pcolormesh(t_vec, 
               np.arange(len(nonout_inds)), 
               phase_array,
               cmap = cmap.cmap)
ax[1,0].scatter(fin_plot_dat['time'].values/1000, fin_plot_dat['cont_trials'], 
            marker = "|", color = 'k', linewidth = 2)
ax[1,0].colorbar(im)
ax[1,1].pcolor(new_t_vec, np.arange(new_phase_array.shape[0]), new_phase_array,
             cmap = cmap.cmap)
for i,dat in enumerate(new_spike_times):
    ax[1,1].scatter(dat, [i]*len(dat), color = 'k', marker = '|', linewidth = 2)
ax[1,1].colorbar()
plt.show()

#ind = 3
#plt.plot(t_vec, phase_array[ind])
#plt.plot(new_t_vecs[ind], phase_array[ind])
#plt.axhline(set_start_phi)
#plt.axvline(0, color = 'r')
#plt.show()

fig,ax = plt.subplots(2,1, sharex=True)
ax[0].plot(t_vec, phase_array.T) 
for this_t, this_dat in zip(aligned_t_vec, phase_array):
    ax[1].plot(this_t, this_dat)
ax[1].axhline(set_start_phi)
ax[1].axvline(0, color = 'r')
plt.show()


plt.imshow(phase_array, aspect='auto')
plt.colorbar()
plt.show()



trial0 = plot_dat[plot_dat.trials == 0]
trial0_phase = interp_phase(trial0, t_vec)
plt.plot(trial0['time']/1000, trial0['phases'], 'x', label = 'actual')
plt.plot(t_vec, wrap_phase(trial0_phase), 'o', label = 'interp', alpha = 0.1)
plt.scatter(trial0['time']/1000, np.ones(len(trial0)))
plt.legend()
plt.show()

# Plot histogram of phases for each frequency
wanted_freqs = test_nrn_raw_dat.freq.unique()
phi_bins = np.linspace(-np.pi, np.pi, 30)
hist_list = []
for num, (this_freq, this_ax) in enumerate(zip(wanted_freqs, ax)):
    this_dat = test_nrn_raw_dat[test_nrn_raw_dat.freq == this_freq]
    hist_out = np.histogram(this_dat['phases'], bins=phi_bins)
    hist_list.append(hist_out[0])

# Min max scaling
hist_array = np.array(hist_list)
hist_array = (hist_array - hist_array.min(axis=-1, keepdims=True)) / (hist_array.max(axis=-1, keepdims=True) - hist_array.min(axis=-1, keepdims=True))
# Smoothen hist array with Savitzky-Golay filter
from scipy.signal import savgol_filter
smooth_hist_array = savgol_filter(hist_array, 7, 2, axis=-1)

fig,ax = plt.subplots(len(wanted_freqs), 1, sharey=True, sharex=True)
for num, (this_freq, this_ax) in enumerate(zip(wanted_freqs, ax)):
    this_ax.plot(phi_bins[:-1], smooth_hist_array[num,:], '-x')
    this_ax.set_title(this_freq)
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].pcolormesh(phi_bins[:-1], wanted_freqs, hist_array,
               cmap='viridis', shading='auto')
ax[1].pcolormesh(phi_bins[:-1], wanted_freqs, smooth_hist_array,
                 cmap='viridis', shading='auto')
ax[0].set_title('Raw')
ax[1].set_title('Smooth')
ax[0].set_ylabel('Frequency (Hz)')
ax[0].set_xlabel('Phase (radians)')
ax[1].set_xlabel('Phase (radians)')
plt.show()

cmap = plt.get_cmap('viridis')
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
for num, this_dat in enumerate(smooth_hist_array):
    ax.fill(phi_bins[:-1], this_dat, '-x', 
            alpha = 0.5, c = cmap(num/len(smooth_hist_array)),
            label = wanted_freqs[num])
ax.set_title('Smooth')
ax.legend(wanted_freqs)
plt.show()

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

# Normalize by baseline
# Then scale from 0-1
baseline_lims = [-500, 0]
baseline_inds = (time_vals >= baseline_lims[0]) & (time_vals < baseline_lims[1])
baseline_vals_gc = pivot_gc[:, baseline_inds]
mean_baseline_gc = baseline_vals_gc.mean(axis=-1)
norm_pivot_gc = pivot_gc - mean_baseline_gc[:, None]
norm_pivot_gc = norm_pivot_gc - norm_pivot_gc.min()
norm_pivot_gc = norm_pivot_gc / norm_pivot_gc.max()

baseline_vals_bla = pivot_bla[:, baseline_inds]
mean_baseline_bla = baseline_vals_bla.mean(axis=-1)
norm_pivot_bla = pivot_bla - mean_baseline_bla[:, None]
norm_pivot_bla = norm_pivot_bla - norm_pivot_bla.min()
norm_pivot_bla = norm_pivot_bla / norm_pivot_bla.max()

# plot both
plot_time_lims = [-500, 2500]
plot_freq_lims = [0, 15]
plot_time_inds = (time_vals >= plot_time_lims[0]) & (time_vals < plot_time_lims[1])
plot_freq_inds = (freq_vec >= plot_freq_lims[0]) & (freq_vec < plot_freq_lims[1])
plot_gc_dat = norm_pivot_gc[plot_freq_inds, :][:, plot_time_inds]
plot_bla_dat = norm_pivot_bla[plot_freq_inds, :][:, plot_time_inds]
plot_time_vals = time_vals[plot_time_inds]
plot_freq_vals = freq_vec[plot_freq_inds]

epoch_markers = [300, 850, 1450]
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True,
                       figsize = (7*0.75,7*0.75))
im = ax[0].contourf(plot_time_vals, plot_freq_vals,
                 plot_gc_dat, 
                 cmap='viridis', vmin=0, vmax=1,
                    levels = 10)
# Colorbar limit from  0-1
#cbar = plt.colorbar(im, ax = ax[0],
#                    ticks = np.arange(0,1.2,0.2))
im = ax[1].contourf(plot_time_vals, plot_freq_vals,
                 plot_bla_dat, 
                 cmap='viridis', vmin=0, vmax=1,
                    levels = 10)
# Mark stimulus delivery at 0
for this_ax in ax:
    this_ax.axvline(color = 'red', linestyle = '--', linewidth = 2)
    for this_marker in epoch_markers:
        this_ax.axvline(this_marker, color = 'red', linewidth = 1)
# Create shared colorbar for both plots
cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Normalized Phase Locking Value')
ax[0].set_title('GC Spike --> BLA Field')
ax[1].set_title('BLA Spike --> GC Field')
ax[0].set_ylabel('Frequency (Hz)')
ax[1].set_ylabel('Frequency (Hz)')
ax[1].set_xlabel('Time post-stimulus (ms)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'phase_locking_heatmap.png'),
            bbox_inches='tight', dpi = 300)
plt.close('all')
# plt.show()


# Plots by Epoch
#time_vals = pivot_gc.columns.values
epochs = [[-500,0], [0, 300], [300, 850], [850, 1450]]

gc_epoch_list = []
bla_epoch_list = []
for this_epoch in epochs:
    inds = (plot_time_vals >= this_epoch[0]) & (plot_time_vals < this_epoch[1])
    gc_epoch_list.append(plot_gc_dat[:, inds])
    bla_epoch_list.append(plot_bla_dat[:, inds])
region_list = list(zip(gc_epoch_list, bla_epoch_list))
region_names = ['gc', 'bla']
baseline_vals = region_list.pop(0)
mean_baseline_per_region = [x.mean(axis=-1) for x in baseline_vals]
mean_baseline = np.mean(mean_baseline_per_region, axis=0)

epochs.pop(0)

cmap = plt.get_cmap('tab10')
fig, ax = plt.subplots(len(region_list), figsize=(4*0.9, 7*0.9),
                       sharey=True, sharex=True)
for num, (dat, this_ax) in enumerate(zip(region_list, ax)):
    mean_vals = [x.mean(axis=-1) for x in dat]
    std_vals = [x.std(axis=-1) for x in dat]
    this_ax.plot(plot_freq_vals, mean_baseline,
                 color='r', linestyle='--',
                 linewidth=2,
                 label='Baseline')
    for region_num, (this_mean, this_std) in enumerate(zip(mean_vals, std_vals)):
        this_ax.plot(plot_freq_vals, this_mean,
                     #label=region_names[region_num].upper(),
                     color=cmap(region_num),
                     linewidth=2)
        #this_ax.axhline(color = 'k', linestyle = '--',
        #                linewidth = 2)
        # Plot mean (across both regions) values for baseline
        # on all plots for comparison
        this_ax.scatter(plot_freq_vals, this_mean,
                        color=cmap(region_num))
        this_ax.fill_between(
            x=plot_freq_vals,
            y1=this_mean - this_std,
            y2=this_mean + this_std,
            alpha=0.5,
            label = region_names[region_num].upper()
        )
    this_ax.set_xlim([0, 15])
    this_ax.set_ylabel('Scaled Phase' + '\n' + 'Locking (A.U.)')
    #this_ax.set_title(f'Epoch : {epochs[num]}')
    # Mark epoch as text on right side of plot
    # Add box around text
    this_ax.text(1.02, 0.5, f'{epochs[num][0]} - {epochs[num][1]} ms',
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
# Temporarily remove these plots below 
########################################


#########################################
## 2D Plots
#########################################
#vmin = imp_dat['scaled_z_stat'].min()
#vmax = imp_dat['scaled_z_stat'].max()
#levels = np.logspace(np.log10(vmin), np.log10(vmax), 10)
#fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
#im = ax[0].contourf(
#    # im = ax[0].pcolormesh(
#    time_vals,
#    freq_vec,
#    pivot_gc,
#    #levels = levels,
#    #vmin = vmin, vmax=vmax
#)
#fig.colorbar(im, ax=ax[0])
#im = ax[1].contourf(
#    # im = ax[1].pcolormesh(
#    time_vals,
#    freq_vec,
#    pivot_bla,
#    #levels = levels,
#    #vmin = vmin, vmax=vmax
#)
#fig.colorbar(im, ax=ax[1])
#ax[0].set_title('GC_spike-->BLA_field')
#ax[1].set_title('BLA_spike-->GC_field')
#ax[0].set_ylim([0, 15])
#ax[1].set_xlabel('Time post-stim (ms)')
#ax[0].set_ylabel('Frequency (Hz)')
#ax[1].set_ylabel('Frequency (Hz)')
#ax[0].axvline(0, color='red', linestyle='--')
#ax[1].axvline(0, color='red', linestyle='--')
#plt.tight_layout()
#plt.suptitle('BLA-GC Spike Field Coherence')
#plt.subplots_adjust(top=0.8)
## plt.show()
#fig.savefig(os.path.join(plot_dir, 'sig_only_raw_spike_field_coherence.png'))
#plt.close(fig)
#
#norm_pivot_gc = pivot_gc.values
#norm_pivot_gc = zscore(norm_pivot_gc, axis=-1)
#norm_pivot_bla = pivot_bla.values
#norm_pivot_bla = zscore(norm_pivot_bla, axis=-1)
#norm_stack = np.stack([norm_pivot_gc, norm_pivot_bla])
#
##vmin = np.nanmin(norm_stack, axis=None)
##vmax = np.nanmax(norm_stack, axis=None)
##levels = np.logspace(np.log10(vmin),np.log10(vmax),10)
#cmap = plt.get_cmap('viridis')
#level_colors = cmap(levels)
#fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
#im = ax[0].contourf(
#    # im = ax[0].pcolormesh(
#    time_vals,
#    freq_vec,
#    norm_pivot_gc,
#    #levels = levels,
#    #vmin = vmin, vmax=vmax,
#    #alpha = 0.3
#)
#fig.colorbar(im, ax=ax[0])
#im = ax[1].contourf(
#    # im = ax[1].pcolormesh(
#    time_vals,
#    freq_vec,
#    norm_pivot_bla,
#    #levels = levels,
#    #vmin = vmin, vmax=vmax,
#    #alpha = 0.1
#)
#fig.colorbar(im, ax=ax[1])
#ax[0].set_title('GC_spike-->BLA_field')
#ax[1].set_title('BLA_spike-->GC_field')
#ax[0].set_ylim([0, 15])
#ax[1].set_xlabel('Time post-stim (ms)')
#ax[0].set_ylabel('Frequency (Hz)')
#ax[1].set_ylabel('Frequency (Hz)')
#plt.tight_layout()
#plt.suptitle('BLA-GC Spike Field Coherence')
#plt.subplots_adjust(top=0.8)
#ax[0].axvline(0, color='red', linestyle='--')
#ax[1].axvline(0, color='red', linestyle='--')
## plt.show()
#fig.savefig(os.path.join(plot_dir, 'sig_only_norm_spike_field_coherence.png'))
#plt.close(fig)
