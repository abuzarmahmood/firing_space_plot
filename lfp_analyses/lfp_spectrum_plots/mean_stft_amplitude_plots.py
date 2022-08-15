import sys
import os
from glob import glob
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from tqdm import tqdm,trange
import pylab as plt
from scipy.stats import zscore
import numpy as np
import tables

# Get file list
base_datadir = '/media/bigdata/Abuzar_Data/bla_gc'
file_list = glob(os.path.join(base_datadir,'*','*','*.h5'))
dir_list = [os.path.dirname(x) for x in file_list]
basename_list = [os.path.basename(x).split('.')[0] for x in file_list]

stft_list =  []
fin_inds = []
for ind in trange(len(dir_list)):
    this_h5_path = file_list[ind]
    with tables.open_file(this_h5_path,'r') as h5:
        if '/unit_descriptor' not in h5:
            continue
    #ind = 0
    this_dir = dir_list[ind]
    fin_inds.append(ind)
    dat = ephys_data(this_dir)
    dat.stft_params['signal_window'] = 1000
    dat.stft_params['window_overlap'] = 975
    dat.get_stft(recalculate=True)
    #dat.get_stft()
    time_vec = dat.time_vec
    freq_vec = dat.freq_vec
    mean_stft = dat.get_mean_stft_amplitude()
    stft_list.append(mean_stft)

# Get Region names
region_list = []
for ind in trange(len(dir_list)):
    this_h5_path = file_list[ind]
    with tables.open_file(this_h5_path,'r') as h5:
        if '/unit_descriptor' not in h5:
            continue
    this_dir = dir_list[ind]
    dat = ephys_data(this_dir)
    dat.get_region_units()
    region_list.append(dat.region_names)

sorted_region_inds = [np.argsort(x) for x in region_list]
stft_list = [x[inds] for x,inds in zip(stft_list,sorted_region_inds)]


stim_t = 2
mean_baseline = [np.mean(x[...,time_vec<stim_t],axis=-1) \
        for x in stft_list]
base_sub_stft = [x - y[...,None] for x,y in zip(stft_list, mean_baseline)]
base_div_stft = [x/y[...,None] for x,y in zip(stft_list, mean_baseline)]
base_div_stft = np.stack(base_div_stft)

mean_base_div = np.median(base_div_stft,axis=0)

##imshow_kwargs = {'interpolation' : 'spline36', 
#imshow_kwargs = {'interpolation' : 'nearest', 
#                'aspect' : 'auto',
#                'origin' : 'lower',
#                'cmap' : 'jet'}
#t_lims = [1.5,4]
#inds = np.logical_and(t_lims[0]<time_vec, time_vec<t_lims[1])
#fig,ax = plt.subplots(2,1, sharex=True)
#ax[0].imshow(mean_base_div[0][...,inds], **imshow_kwargs)
#ax[1].imshow(mean_base_div[1][...,inds], **imshow_kwargs)
##ticks = np.arange(len(time_vec),step=10)[inds]
##plt.xticks(ticks,time_vec[ticks], rotation=45)
#plt.show()

#ind = 6
tmarks = [300,750]
t_lims = [1.5,4]
region_labels = np.array(region_list[0])[sorted_region_inds[0]]
inds = np.logical_and(t_lims[0]<time_vec, time_vec<t_lims[1])
fin_t = time_vec[inds] - stim_t
fin_t = fin_t*1000
#fin_dat = base_div_stft[ind][...,inds]
fin_dat = mean_base_div[...,inds]
fin_dat = np.stack([zscore(x, axis=-1) for x in fin_dat])
contourf_kwargs = {'levels' : 100, 'cmap' : 'jet',
        'antialiased' : True, 'linewidths' : 0,
        'vmin' : -2, 'vmax' : 2}
fig,ax = plt.subplots(1,2, sharex=True, sharey=True,figsize=(5,3))
cnt0 = ax[0].contourf(fin_t, freq_vec, fin_dat[0], 
        **contourf_kwargs)
cnt1 = ax[1].contourf(fin_t, freq_vec, fin_dat[1], 
        **contourf_kwargs)
for x in [cnt0,cnt1]:
    for c in x.collections:
        c.set_edgecolor("face")
for this_ax in ax:
    this_ax.axvline(0, lw = 2, color = 'k')
    for this_mark in tmarks:
        this_ax.axvline(this_mark, lw = 2, color='red', linestyle = '--')
    this_ax.set_xticks(np.arange(-500,2500,500))
    this_ax.set_xticklabels(np.arange(-500,2500,500),rotation = 90)
for this_ax, title in zip(ax, region_labels):
    this_ax.set_title(title.upper())
ax[0].set_ylabel('Frequency (Hz)')
ax[1].set_xlabel('Time post-stimulus delivery (ms)')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'mean_lfp_spectrogram2.svg'),format = 'svg')
plt.close(fig)
#plt.show()

#########################################
imshow_kwargs = {'interpolation' : 'bilinear', 
                'aspect' : 'auto',
                'origin' : 'lower'}

# Plot
plot_classes = ['raw','zscore','back_sub']
plot_dir = '/media/bigdata/firing_space_plot/lfp_analyses/lfp_spectrum_plots/plots'
fin_paths = [os.path.join(plot_dir,x) for x in plot_classes]

for x in fin_paths:
    if not os.path.exists(x):
        os.makedirs(x)

# Raw
for session, name in zip(stft_list, basename_list):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(session[0], **imshow_kwargs)
    ax[1].imshow(session[1], **imshow_kwargs)
plt.show()

    fig.savefig(os.path.join(fin_paths[0], name))
    plt.close(fig)

# Zscore
for session, name in zip(stft_list, basename_list):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(zscore(session[0],axis=-1), **imshow_kwargs)
    ax[1].imshow(zscore(session[1],axis=-1), **imshow_kwargs)
plt.show()

    fig.savefig(os.path.join(fin_paths[1], name))
    plt.close(fig)

# Baseline subtracted
for session, name in zip(base_sub_stft, basename_list):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(session[0], **imshow_kwargs)
    ax[1].imshow(session[1], **imshow_kwargs)
plt.show()

for session, name in zip(base_sub_stft, basename_list):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(zscore(session[0],axis=-1), **imshow_kwargs)
    ax[1].imshow(zscore(session[1],axis=-1), **imshow_kwargs)
plt.show()

    fig.savefig(os.path.join(fin_paths[2], name))
    plt.close(fig)

# Baseline fold-change
for session, name in zip(base_div_stft, basename_list):
    fig,ax = plt.subplots(2,1)
    ax[0].imshow(session[0], **imshow_kwargs)
    ax[1].imshow(session[1], **imshow_kwargs)
plt.show()


