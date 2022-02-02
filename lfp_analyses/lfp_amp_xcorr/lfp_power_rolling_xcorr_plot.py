"""
1) Plot median sttt amplitude per region to make sure the split was correct
2) Plot XCorr analyses:
    a) AVERAGE Inter-region + Intra-region xcorrs with shuffles
    b) AVERAGE Binned Inter-Region XCorr
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
from scipy.stats import zscore
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
import pylab as plt
from glob import glob
import xarray as xr
import json

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import firing_overview, gen_square_subplots

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

save_dir = '/media/bigdata/firing_space_plot/lfp_analyses/lfp_amp_xcorr/data'
plot_dir = '/media/bigdata/firing_space_plot/'\
                    'lfp_analyses/lfp_amp_xcorr/Plots/rolling'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

#data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM11/AM11_4Tastes_191030_114043_copy'
data_dir = '/media/bigdata/Abuzar_Data/bla_gc/AM12/AM12_4Tastes_191105_083246'
#data_dir = sys.argv[1]
if data_dir[-1] != '/':
    data_dir += '/'
dat = ephys_data(data_dir)
dat.get_lfp_electrodes()
dat.get_stft()

json_path = glob(os.path.join(dat.data_dir,"*.info"))[0]
with open(json_path, 'r') as params_file:
    info_dict = json.load(params_file)
tastant_names = info_dict['taste_params']['tastes']

name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]
# Somehow messed up naming
basename = os.path.basename(data_dir[:-1])[:-1]
fin_plot_dir = os.path.join(plot_dir, basename)

if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

median_amplitude = np.median(dat.amplitude_array,axis=(0,2))


########################################
## Load XCorr frames
########################################
file_list = glob(os.path.join(save_dir, basename+"*"))

#save_path = '/stft/analyses/amplitude_xcorr'
#for frame_name in ['inter_region_frame',
#                            'binned_inter_region_frame',
#                            'intra_region_frame',
#                            'binned_intra_region_frame',
#                            'base_inter_region_frame',
#                            'base_intra_region_frame']:
#    # Save transformed array to HDF5
#    globals()[frame_name] = pd.read_hdf(dat.hdf5_name,  
#            os.path.join(save_path, frame_name))

def return_band(val):
    if val < 4:
        return '1_delta'
    elif 4 <= val <= 7:
        return '2_theta'
    elif 7 < val <= 12:
        return '3_alpha'
    elif val > 12:
        return '4_beta'

xr_datasets = [xr.load_dataarray(path) for path in file_list]
bands = [return_band(x) for x in xr_datasets[0].coords['freqs'].values]
xr_datasets = [x.assign_coords(bands = ('freqs',bands)) for x in xr_datasets]
#xr_datasets = [x.assign_coords(tastant_names = ('tastes',tastant_names)) \
#        for x in xr_datasets]
xr_datasets = [x.assign_coords(tastes = tastant_names) for x in xr_datasets]

##################################################
# ____  _       _   _   _             
#|  _ \| | ___ | |_| |_(_)_ __   __ _ 
#| |_) | |/ _ \| __| __| | '_ \ / _` |
#|  __/| | (_) | |_| |_| | | | | (_| |
#|_|   |_|\___/ \__|\__|_|_| |_|\__, |
#                               |___/ 
##################################################

########################################
## Regional Spectrogram Plots
########################################

for region,region_name in zip(dat.lfp_region_electrodes,dat.region_names):
    fig,ax = gen_square_subplots(len(region), sharex=True, sharey=True)
    for this_dat, this_ax in \
            zip(zscore(median_amplitude[region],axis=-1),ax.flatten()):
        this_ax.imshow(this_dat,aspect='auto',cmap='jet',origin='lower')
    title_str = fin_name + "_" + region_name 
    plt.suptitle(title_str + '\nZscore Median Spectrogram')
    fig.savefig(os.path.join(fin_plot_dir,title_str+'_spectrogram'),dpi=300)

fig2,ax2 = plt.subplots(1,len(dat.region_names), figsize=(10,3))
for num,(region,region_name) in \
        enumerate(zip(dat.lfp_region_electrodes,dat.region_names)):
    this_region = median_amplitude[region]
    ax2[num].imshow(zscore(np.median(this_region,axis=0),axis=-1),
                aspect='auto',cmap='jet',origin='lower')
    ax2[num].set_title(region_name)
plt.subplots_adjust(top = 0.7)
plt.suptitle(title_str + '\nZscore Median Region Spectrogram')
fig2.savefig(os.path.join(fin_plot_dir,fin_name + '_region_spectrogram'),dpi=300)

    #plt.show()

########################################
## Whole Trial XCorr Plots
########################################
# Means across pairs and trials
agg_datasets = [x.mean(dim=['pairs','trials']) for x in xr_datasets]
agg_concat = xr.concat(agg_datasets, dim = 'region_type')

inds = np.array(list(np.ndindex(agg_concat.shape[:2])))
fig, ax = plt.subplots(np.prod(agg_concat.shape[:2],axis=-1), 
        agg_concat.shape[2], sharex=True, sharey=True)
for num, this_ind in enumerate(inds):
    order_type = str(agg_concat[tuple(this_ind)]['order_type'].values)
    region_type = str(agg_concat[tuple(this_ind)]['region_type'].values)
    ylabel = '\n'.join([order_type,region_type])
    ax[num,0].set_ylabel(ylabel)
    for this_taste in range(agg_concat.shape[2]):
        this_dat = agg_concat[tuple([*this_ind, this_taste])]
        x = this_dat['bins'].values 
        y = this_dat['freqs'].values 
        pcm = ax[num,this_taste].\
                pcolormesh(x,y,this_dat, vmin = 0, vmax = 1)
plt.subplots_adjust(right = 0.85)
#PCM=ax[-1,-1].get_children()[2]
cax = plt.axes([0.9, 0.1, 0.05, 0.8])
fig.colorbar(pcm, cax=cax)
plt.suptitle(fin_name + '_median_xcorr')
fig.savefig(os.path.join(fin_plot_dir,fin_name + '_median_xcorr'),dpi=300)
plt.close(fig)

#sem = lambda x: np.std(x) / np.sqrt(len(x))
mean_dims = ['pairs','trials', 'tastes','freqs']
band_mean_datasets = [x.groupby('bands').mean(dim=mean_dims) \
                            for x in xr_datasets]
band_std_datasets = [x.groupby('bands').quantile([0.25,0.75],dim=mean_dims) \
                            for x in xr_datasets]
#band_std_datasets = [x.groupby('bands').std(dim=mean_dims) \
#                            for x in xr_datasets]
band_mean_concat = xr.concat(band_mean_datasets, dim = 'region_type')
band_std_concat = xr.concat(band_std_datasets, dim = 'region_type')

agg_concat_ds = band_mean_concat.to_dataset(name = 'agg')
agg_concat_ds['std'] = band_std_concat

stim_t = 2000
fig,ax = plt.subplots(len(agg_concat_ds['region_type'].values),
        len(agg_concat_ds['bands'].values), sharex=True, sharey=True)
for region_num, (region_key, region_ds) \
        in enumerate(agg_concat_ds.groupby('region_type')):
    ax[region_num,0].set_ylabel(str(region_ds.region_type.values))
    for band_num, (band_key, band_ds) in enumerate(region_ds.groupby('bands')):
        ax[0,band_num].set_title(str(band_ds.bands.values))
        for order_key, order_ds in band_ds.groupby('order_type'):
            agg_values = np.squeeze(order_ds['agg'].values)
            std_values = np.squeeze(order_ds['std'].values)
            ax[region_num, band_num].plot(order_ds.bins.values, 
                    agg_values)
            ax[region_num, band_num].fill_between(
                    x = order_ds.bins.values, 
                    y1 = std_values[:,0], 
                    y2 = std_values[:,1],
                    #y1 = agg_values + std_values, 
                    #y2 = agg_values - std_values,
                    alpha = 0.3,
                    label = str(order_ds.order_type.values))
            ax[region_num, band_num].axvline(
                    stim_t, color = 'red', alpha = 0.2, linestyle = 'dashed')
#ax[0,0].legend()
ax[0,-1].legend(bbox_to_anchor=(1.1, 1.05))
plt.suptitle(fin_name + ": Bandwise XCorr")
fig.savefig(os.path.join(fin_plot_dir,fin_name + '_band_xcorr'),
        dpi=300, bbox_inches = 'tight')
plt.close(fig)

fig,ax = plt.subplots(len(agg_concat_ds['region_type'].values),
        len(agg_concat_ds['bands'].values), sharex=True, sharey='row')
for region_num, (region_key, region_ds) \
        in enumerate(agg_concat_ds.groupby('region_type')):
    ax[region_num,0].set_ylabel(str(region_ds.region_type.values))
    for band_num, (band_key, band_ds) in enumerate(region_ds.groupby('bands')):
        ax[0,band_num].set_title(str(band_ds.bands.values))
        for order_key, order_ds in band_ds.groupby('order_type'):
            if order_key == 'actual':
                ax[region_num, band_num].plot(order_ds.bins.values, 
                        np.squeeze(order_ds['agg'].values),
                        label = str(order_ds.order_type.values))
                ax[region_num, band_num].axvline(
                        stim_t, color = 'red', alpha = 0.2, linestyle = 'dashed')
#ax[0,0].legend()
ax[0,-1].legend(bbox_to_anchor=(1.1, 1.05))
plt.suptitle(fin_name + ": Bandwise XCorr")
fig.savefig(os.path.join(fin_plot_dir,fin_name + '_mean_band_xcorr'),
        dpi=300, bbox_inches = 'tight')
plt.close(fig)

inter_agg = agg_concat[agg_concat["region_type"] == 'inter'].squeeze()
inter_agg = inter_agg[inter_agg["order_type"] == 'actual'].squeeze()
xr.plot.pcolormesh(inter_agg, x = 'bins',y = 'freqs',
        col = 'tastes', figsize = (15,3), aspect='auto',
        cmap = 'viridis')
fig = plt.gcf()
plt.suptitle(fin_name + ": Inter XCorr")
plt.subplots_adjust(top = 0.8, right = 0.8)
fig.savefig(os.path.join(fin_plot_dir,fin_name + '_inter_xcorr'),dpi=300)
plt.close(fig)

#plt.show()
#concat_frame = pd.concat([inter_region_frame,intra_region_frame,
#                        base_inter_region_frame, base_intra_region_frame])
## Make new column to distinguish baseline from post-stim
#concat_frame['baseline'] = concat_frame.label.str.contains('base')
#
#plot_frame = concat_frame.groupby(['label','baseline','pair','freq']).mean().\
#                            reset_index()
#
#title_str = fin_name + "\nLFP AMP XCorr (mean +/- SD)" 
#sns.relplot(x='freq',y='xcorr',hue='label', col = 'baseline',data=plot_frame, 
#        kind = 'line', ci='sd',  markers = True, style = 'label')
#plt.suptitle(title_str)
#fig = plt.gcf()
#fig.savefig(os.path.join(fin_plot_dir,fin_name+'_LFP_AMP_XCorr'),dpi=300)
##plt.show()
#
### Taste-specific
#plot_frame = concat_frame.\
#                        groupby(['label','baseline','pair','freq','taste']).mean().\
#                        reset_index()
#
#title_str = fin_name + "\nLFP AMP Taste XCorr (mean +/- SD)" 
#sns.relplot(x='freq',y='xcorr',hue='label',data=plot_frame, 
#        kind = 'line', ci='sd',  markers = True, style = 'label',col='taste', row='baseline')
#plt.suptitle(title_str)
#fig = plt.gcf()
#fig.savefig(os.path.join(fin_plot_dir,fin_name+'_LFP_AMP_Taste_XCorr'),dpi=300)
##plt.show()
