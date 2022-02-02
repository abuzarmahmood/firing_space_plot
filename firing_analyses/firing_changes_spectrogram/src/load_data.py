"""
Pull data from HDF5 files and store locally
"""
#   _  _     ___                            _       
# _| || |_  |_ _|_ __ ___  _ __   ___  _ __| |_ ___ 
#|_  ..  _|  | || '_ ` _ \| '_ \ / _ \| '__| __/ __|
#|_      _|  | || | | | | | |_) | (_) | |  | |_\__ \
#  |_||_|   |___|_| |_| |_| .__/ \___/|_|   \__|___/
#                         |_|                       

import os
import pandas as pd
import json
import sys
import pandas as pd
from tqdm import tqdm
import tables
import numpy as np
import xarray as xr
import pylab as plt
import scipy, scipy.signal
from scipy.stats import zscore
from scipy.stats import ks_2samp, mannwhitneyu
from scipy.ndimage import gaussian_filter1d as gf1d

proj_path = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'firing_changes_spectrogram'
os.chdir(proj_path)
ephys_path = os.path.join(proj_path, "conf/ephys_data.path") 
sys.path.append(open(ephys_path,'r').read().strip())

from ephys_data import ephys_data
import visualize as vz

def calc_stft(timeseries, max_freq,time_range_tuple,\
            Fs,signal_window,window_overlap):
    """
    timeseries : 1D array
    max_freq : where to lob off the transform
    time_range_tuple : (start,end) in seconds, time_lims of spectrogram
                            from start of trial snippet`
    """
    f,t,this_stft = scipy.signal.stft(
                scipy.signal.detrend(timeseries),
                fs=Fs,
                window='hanning',
                nperseg=signal_window,
                noverlap=signal_window-(signal_window-window_overlap))
    this_stft =  this_stft[np.where(f<max_freq)[0]]
    this_stft = this_stft[:,np.where((t>=time_range_tuple[0])*\
                                            (t<time_range_tuple[1]))[0]]
    fin_freq = f[f<max_freq]
    fin_t = t[np.where((t>=time_range_tuple[0])*(t<time_range_tuple[1]))]
    return  fin_freq, fin_t, this_stft

stft_params = {
        'Fs' : 1000, 
        'signal_window' : 250,
        'window_overlap' : 249,
        'max_freq' : 100,
        'time_range_tuple' : (0,2)
        }

#   _  _     ____  _                 _       _       
# _| || |_  / ___|(_)_ __ ___  _   _| | __ _| |_ ___ 
#|_  ..  _| \___ \| | '_ ` _ \| | | | |/ _` | __/ _ \
#|_      _|  ___) | | | | | | | |_| | | (_| | ||  __/
#  |_||_|   |____/|_|_| |_| |_|\__,_|_|\__,_|\__\___|
# Perform analysis on simulated data with sharp transition
# and compare with trial shuffled response
time = np.arange(2000)
trials = 10
sd = 500
trans_time_vec = 1000 + np.random.randint(-sd,sd,trials)

def gen_spikes(time, trans_time, f_low = 10, f_high = 100, dt = 0.001):
    f_rate = np.ones(time.shape)
    f_rate[:trans_time] *= dt*f_low
    f_rate[trans_time:] *= dt*f_high
    rand_vec = np.random.random(f_rate.shape)
    return rand_vec < f_rate

spike_array = np.stack([gen_spikes(time, this_trans) \
                                for this_trans in trans_time_vec])
# Generate shuffle by permuting spikes across trials for every timepoint
shuffle_spikes = np.stack([np.random.permutation(x) for x in spike_array.T]).T

fig, ax = plt.subplots(2,1)
ax[0] = vz.raster(ax[0],spike_array, "|")
ax[1] = vz.raster(ax[1],shuffle_spikes, "|")
plt.show()

fig, ax = plt.subplots(2,1)
ax[0].imshow(spike_array, interpolation='gaussian', aspect='auto')
#ax[0].imshow(spike_array, interpolation='gaussian', aspect='auto')
ax[1].imshow(shuffle_spikes, interpolation='gaussian', aspect='auto')
#ax[1].imshow(shuffle_spikes, interpolation='gaussian', aspect='auto')
plt.show()

xr_true = xr.DataArray(
        data = spike_array,
        dims = ['trials','time'],
        coords = {'time' : time})
xr_shuffle = xr.DataArray(
        data = shuffle_spikes,
        dims = ['trials','time'],
        coords = {'time' : time})

spikes_dataset = xr.Dataset()
spikes_dataset['true'] = xr_true
spikes_dataset['shuffle'] = xr_shuffle

#rate_dataset = spikes_dataset.rolling(time = 100).mean().dropna('time')
#rate_dataset['true'].plot(x='time',hue='trials', alpha = 0.7)
#plt.figure()
#rate_dataset['shuffle'].plot(x='time',hue='trials', alpha = 0.7);plt.show()

def return_gauss_kern(length, sd):
    assert length%2 == 1, 'Length must be an odd number'
    x = np.zeros((length))
    x[int(length/2+0.5)] = 1
    return gf1d(x,sd)

kern_len = 101
gauss_kern = xr.DataArray(return_gauss_kern(kern_len,25), dims=["window"])
def spike_conv(array, kern):
    radius = int(len(kern)/2-0.5)
    array[:,:radius] = 0
    array[:,-radius:] = 0
    inds = np.where(array)
    rate_array = np.zeros(array.shape)
    for x,y in zip(*inds):
        rate_array[x, y-radius : y+radius] += kern[:-1]
    return rate_array
        

rate_dataset = xr.Dataset()
rate_dataset['true'] = xr.DataArray(
        data = spike_conv(spikes_dataset['true'].values, gauss_kern),
        dims = ['trials','time'],
        coords = dict(time=time))
rate_dataset['shuffle'] = xr.DataArray(
        data = spike_conv(spikes_dataset['shuffle'].values, gauss_kern),
        dims = ['trials','time'],
        coords = dict(time=time))

rate_dataset['true'].plot(x='time',hue='trials', alpha = 0.7)
plt.figure()
rate_dataset['shuffle'].plot(x='time',hue='trials', alpha = 0.7);plt.show()

freq, time, stft = calc_stft(rate_dataset['true'][0].values, **stft_params)
true_stft = np.stack(\
        [calc_stft(x.values, **stft_params)[2] for x in rate_dataset['true']])
shuffle_stft = np.stack(\
        [calc_stft(x.values, **stft_params)[2] for x in rate_dataset['shuffle']])

stft_dataset = xr.Dataset(
        data_vars = dict(
            true_complex = (['trials','freq','time'], true_stft),
            shuffle_complex = (['trials','freq','time'], shuffle_stft),
           ),
        coords = dict(
            freq = freq,
            time = time)
        )
stft_dataset['true_abs'] = np.abs(stft_dataset['true_complex'])
stft_dataset['shuffle_abs'] = np.abs(stft_dataset['shuffle_complex'])
stft_dataset['true_zscore'] = (['trials','freq','time'],
                        zscore(stft_dataset['true_abs'],axis=-1))
stft_dataset['shuffle_zscore'] = (['trials','freq','time'],
                        zscore(stft_dataset['shuffle_abs'],axis=-1))

#stft_dataset['true_zscore'].plot(col = 'trials', col_wrap = 4)
#stft_dataset['shuffle_zscore'].plot(col = 'trials', col_wrap = 4)
#plt.show()

# Since averaging will cause the jittered transitions to smoothen
# Instead calculate distributions across time for each frequency band
true_abs = stft_dataset['true_abs'].values.swapaxes(0,1)
shuffle_abs = stft_dataset['shuffle_abs'].values.swapaxes(0,1)

bins = 50
fig,ax = vz.gen_square_subplots(true_abs.shape[0])
for ind in range(true_abs.shape[0]):
    #ind = 5
    this_ax = ax.flatten()[ind]
    cat_dat = np.concatenate((true_abs[ind].flatten(),shuffle_abs[ind].flatten()))
    bin_lims = np.linspace(np.min(cat_dat), np.max(cat_dat), bins+1)
    this_ax.hist(true_abs[ind].flatten(), bin_lims, alpha = 0.7, label = 'True')
    this_ax.hist(shuffle_abs[ind].flatten(), bin_lims, alpha = 0.7, label = 'Shuffle')
    this_ax.set_title(f'Freq : {freq[ind]}') 
this_ax.legend()
plt.show()

# Perform KS test between histograms with Bonferroni correction
alpha = 0.05
corrected_alpha = alpha/true_abs.shape[0]
ks_pvals = [ks_2samp(x.flatten(),y.flatten())[1] \
        for x,y, in zip(true_abs, shuffle_abs)]
mann_pvals = [mannwhitneyu(x.flatten(),y.flatten())[1] \
        for x,y, in zip(true_abs, shuffle_abs)]

plt.plot(freq, mann_pvals, '-x')
plt.yscale('log')
plt.show()


#mean_stft = stft_dataset[['true_abs','shuffle_abs']].median(dim='trials')
#mean_stft['mean_diff'] = mean_stft['true_abs'] - mean_stft['shuffle_abs']
#mean_stft['true_zscore'] = (['freq','time'],zscore(mean_stft['true_abs'],axis=-1))
#mean_stft['shuffle_zscore'] = (['freq','time'],zscore(mean_stft['shuffle_abs'],axis=-1))
#mean_stft['diff_zscore'] = (['freq','time'],zscore(mean_stft['mean_diff'],axis=-1))
#
#mean_stft['true_zscore'].plot()
#plt.figure()
#mean_stft['shuffle_zscore'].plot()
#plt.figure()
#mean_stft['diff_zscore'].plot()
#plt.show()
#
#mean_stft['true_abs'].plot()
#plt.figure()
#mean_stft['shuffle_abs'].plot()
#plt.show()

#   _  _     _                    _   ____        _        
# _| || |_  | |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#|_  ..  _| | |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#|_      _| | |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#  |_||_|   |_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
                                                          

# Iterate over info files and pull out recordings which have BLA and GC
# Make sure no laser in files
info_list_path = 'data/info_file_paths.txt'
info_list = open(os.path.join(proj_path,info_list_path),'r').readlines()
info_list = [x.strip() for x in info_list]

gc_files = []
bla_files = []

for this_file in info_list:
    with open(this_file,'r') as info_file:
        info_dict = json.load(info_file)
    if info_dict['laser_params']['dig_in'] == []:
        if 'gc' in info_dict['regions']:
            gc_files.append(this_file)
        if 'bla' in info_dict['regions']:
            bla_files.append(this_file)
    
# Extract spike trains per region and save to local HDF5
concat_files = concat_files
region_name = ['gc']*len(gc_files) + ['bla']*len(bla_files)
basenames = [os.path.basename(x).split('.')[0] for x in concat_files]
dirnames = [os.path.dirname(x) for x in concat_files] 
spike_frame = pd.DataFrame(
        {'basename' : basenames,
        'dirnames' : dirnames,
        'region_name' : region_name,
        'path' : concat_files,
        'spikes' : None
            }) 

for rownum, this_row in tqdm(spike_frame.iterrows()):
    spikes_exist = False
    dat = ephys_data(this_row['dirnames'])
    with tables.open_file(dat.hdf5_path,'r') as hf5:
        if '/spike_trains' in hf5:
            spikes_exist = True
    if spikes_exist:
        dat.check_laser()
        if not dat.laser_exists:
            region_name = this_row['region_name']
            spikes = dat.return_region_spikes(region_name)
            spike_frame.iloc[rownum]['spikes'] = spikes

spike_frame = spike_frame.dropna()

## Calculate firing rates
# Remove outlier to allow concatenation
trial_bool = [x.shape[1] == 30 for x in spike_frame.spikes]
taste_bool = [x.shape[0] == 4 for x in spike_frame.spikes]
time_bool = [x.shape[-1] == 7000 for x in spike_frame.spikes]
fin_bool = [all(x) for x in zip(trial_bool, taste_bool, time_bool)]
spike_frame = spike_frame.loc[trial_bool]

#spikes = np.concatenate([x.swapaxes(2,0) \
#        for x in spike_frame['spikes']],axis=0)

xr_spikes_list = [xr.DataArray(
                data = row['spikes'],
                dims = ['tastes','trials','neurons','time'],
                coords = {'basename' : row['basename'],
                            'region' : row['region_name'],
                            'time' : np.arange(7000)}
                ) \
                        for num, row in tqdm(spike_frame.iterrows())]

xr_spikes_array = xr.concat(xr_spikes_list, dim = 'neurons') 
xr_spikes_array = xr_spikes_array.loc[...,1500:4500]
xr_spikes_array = xr_spikes_array.assign_coords(
        {'neurons' : xr_spikes_array.coords['region']})

gc_array = xr_spikes_array.sel({'neurons' : 'gc'})
bla_array = xr_spikes_array.sel({'neurons' : 'bla'})

test = gc_array[:,:,:4]
test.rolling(time = 50).mean().dropna('time')
