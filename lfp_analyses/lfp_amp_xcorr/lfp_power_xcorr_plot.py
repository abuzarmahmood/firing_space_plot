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

plot_dir = '/media/bigdata/firing_space_plot/lfp_analyses/lfp_amp_xcorr/Plots'

#data_dir = '/media/bigdata/Abuzar_Data/AM12/AM12_4Tastes_191106_085215/'
data_dir = sys.argv[1]
if data_dir[-1] != '/':
    data_dir += '/'
dat = ephys_data(data_dir)
dat.get_lfp_electrodes()
dat.get_stft()

name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]
fin_plot_dir = os.path.join(plot_dir, fin_name)

if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

median_amplitude = np.median(dat.amplitude_array,axis=(0,2))


########################################
## Load XCorr frames
########################################
save_path = '/stft/analyses/amplitude_xcorr'
for frame_name in ['inter_region_frame',
                    'binned_inter_region_frame',
                    'intra_region_frame',
                    'binned_intra_region_frame']:
    # Save transformed array to HDF5
    globals()[frame_name] = pd.read_hdf(dat.hdf5_name,  
            os.path.join(save_path, frame_name))

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
    #plt.show()

########################################
## Whole Trial XCorr Plots
########################################

concat_frame = pd.concat([inter_region_frame,intra_region_frame])
plot_frame = concat_frame.groupby(['label','pair','freq']).mean().\
                            reset_index()

title_str = fin_name + "\nLFP AMP XCorr (mean +/- SD)" 
sns.relplot(x='freq',y='xcorr',hue='label',data=plot_frame, 
        kind = 'line', ci='sd',  markers = True, style = 'label')
plt.suptitle(title_str)
fig = plt.gcf()
fig.savefig(os.path.join(fin_plot_dir,fin_name+'_LFP_AMP_XCorr'),dpi=300)
#plt.show()

## Taste-specific
plot_frame = concat_frame.\
                        groupby(['label','pair','freq','taste']).mean().\
                        reset_inde()

title_str = fin_name + "\nLFP AMP Taste XCorr (mean +/- SD)" 
sns.relplot(x='freq',y='xcorr',hue='label',data=plot_frame, 
        kind = 'line', ci='sd',  markers = True, style = 'label',col='taste')
plt.suptitle(title_str)
fig = plt.gcf()
fig.savefig(os.path.join(fin_plot_dir,fin_name+'_LFP_AMP_Taste_XCorr'),dpi=300)
#plt.show()

########################################
## Binned Corr Plots
########################################
# Hardcoded for now but fix in future
concat_frame = pd.concat([binned_inter_region_frame,binned_intra_region_frame])
plot_frame = concat_frame.\
        groupby(['bin','label','pair','freq']).mean().reset_index()
bin_width = 250
bin_starts = np.arange(np.max(plot_frame.bin)+1)*bin_width
bin_lims = list(zip(bin_starts,bin_starts+bin_width))
plot_frame['bin_lims'] = [bin_lims[x] for x in plot_frame['bin'].values]

sns.relplot(x='freq',y='xcorr',hue='label',col = 'bin_lims', 
                    data=plot_frame, kind = 'line', 
                    ci='sd',col_wrap=4, markers = True, style = 'label')
title_str = fin_name + "\nBinned LFP AMP XCorr (mean +/- SD)" 
plt.suptitle(title_str)
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(fin_plot_dir,fin_name+'_Binned_LFP_AMP_XCorr'),dpi=300)
#plt.show()

## Taste-specific
plot_frame = binned_inter_region_frame.\
        groupby(['bin','label','pair','freq','taste']).mean().reset_index()
plot_frame['bin_lims'] = [bin_lims[x] for x in plot_frame['bin'].values]

sns.relplot(x='freq',y='xcorr',hue='label',col = 'bin_lims',row = 'taste', 
                    data=plot_frame, kind = 'line', 
                    ci='sd', markers = True, style = 'label')
title_str = fin_name + "\nBinned LFP AMP Taste XCorr (mean +/- SD)" 
plt.suptitle(title_str)
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(fin_plot_dir,fin_name+'_Binned_LFP_AMP_Taste_XCorr'),dpi=300)
#plt.show()
