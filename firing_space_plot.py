# 1) Open single file for now
# 2) extract spiking for all tastes for on&off conditions
# 3) Compute firing rate for ENTIRE 7s (or whatever length of recording)
# 3.5) Maybe smooth firing rate
# 4) Project into 'n' dim space and reduce dimensions

# Normalize firing rate, use only responsive neurons (probs not needed)
# Try projecting all tastes at same time
######################### Import dat ish #########################
import tables
import easygui
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.animation as animation

from scipy.ndimage.filters import gaussian_filter1d
import scipy.signal as sig
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
from mpl_toolkits.mplot3d import Axes3D
import time

###################### Open file and extract data ################
dir_name = "/media/sf_shared_folder/jian_you_data/tastes_separately/file_1"
os.chdir(dir_name)
file_list = os.listdir('./')
hdf5_name = ''
for files in file_list:
    if files[-2:] == 'h5':
        hdf5_name = files

# Open the hdf5 file
hf5 = tables.open_file(hdf5_name, 'r') 

## For extracting spike trains
dig_in = hf5.list_nodes('/spike_trains')
dig_in = [dig_in[i] if dig_in[i].__str__().find('dig_in')!=-1 else None for i in range(len(dig_in))]
dig_in = list(filter(None, dig_in))

## Number of trials per taste for indexing when concatenating spike trains
taste_n = [dig_in[i].spike_array[:].shape[0] for i in range(len(dig_in))]
if np.std(taste_n) == 0:
    taste_n = taste_n[0]
else:
    taste_n = int(easygui.multenterbox('How many trails per taste??',fields = ['# of trials'])[0])

# Which trials did and did not have laser
off_trials = [np.where(dig_in[i].laser_durations[:] == 0)[0] for i in range(len(dig_in))]
on_trials = [np.where(dig_in[i].laser_durations[:] > 0)[0] for i in range(len(dig_in))]

# Get the spike array for each individual taste and put in list:
spikes = [dig_in[i].spike_array[:] for i in range(len(dig_in))]
off_spikes = [spikes[i][off_trials[i],:,:] for i in range(len(dig_in))] #Index trials with no laser
on_spikes = [spikes[i][on_trials[i],:,:] for i in range(len(dig_in))] #Index trials with laser

################### Convert spikes to firing rates ##################
step_size = 25
window_size = 250
tot_time = 7000

off_firing = []

for l in range(len(off_spikes)): # How TF do you get nan's from means?
    # [trials, nrns, time]
    this_off_firing = np.zeros((off_spikes[0].shape[0],off_spikes[0].shape[1],int((tot_time-window_size)/step_size)-1))
    for i in range(this_off_firing.shape[0]):
        for j in range(this_off_firing.shape[1]):
            for k in range(this_off_firing.shape[2]):
                this_off_firing[i, j, k] = np.mean(off_spikes[l][i, j, step_size*k:step_size*k + window_size])
    this_off_firing = this_off_firing.reshape((this_off_firing.shape[1],this_off_firing.shape[0]*this_off_firing.shape[2]))
    off_firing.append(this_off_firing)
    
# Normalize firing
for l in range(len(off_firing)):
    for m in range(off_firing[0].shape[0]):
        min_val = np.min(off_firing[l][m,:])
        max_val = np.max(off_firing[l][m,:])
        off_firing[l][m,:] = (off_firing[l][m,:] - min_val)/(max_val-min_val)

all_off_f = np.asarray(off_firing)
all_off_f = all_off_f.reshape((all_off_f.shape[1],all_off_f.shape[0]*all_off_f.shape[2]))
#all_off_f = all_off_f[~np.isnan(all_off_f)]
## Make sure there are not problems with reshaping    
## Maybe smooth firing rates

## Test plots for spikes to firing rate  
# =============================================================================
# inds = [0,3,1]
# spikes = off_spikes[inds[0]][inds[1],inds[2],:]
# rate = off_firing[inds[0]][inds[1],inds[2],:]
# rate2 = sig.resample(rate,spikes.shape[0])
# rate2 = rate2/np.max(rate2)
# plt.plot(spikes)
# plt.plot(rate2)
# =============================================================================

## All da neurons, all da tastes, and ALL DA PLOTS!
# =============================================================================
# rows = len(train_rate)
# rows = len(train_rate)
# cols = train_rate[0].shape[1]
# count = 1
# x = np.linspace(0,10,100)
# fig, axes = plt.subplots(rows,cols,sharex = 'all',sharey='all')
# for i in range(rows):
#     for j in range(cols):
#         axes[i,j].hist(np.ndarray.flatten(train_rate[i][:,j,:]),density=1)
#         params = train_params[i][j]
#         #y = pdf_plot(x,params)
#         #axes[i,j].plot(x,y,colors[int(params[2])])
#         count +=1
#         print(count)
#    
# fig.tight_layout()
# =============================================================================

################### Reduce dimensions ########################
off_f_red = LLE(n_neighbors = 50,n_components=3).fit_transform(np.transpose(all_off_f))
off_f_red = LLE(n_neighbors = 50,n_components=3).fit_transform(np.transpose(off_firing[0]))
off_f_red = TSNE(n_components=3).fit_transform(np.transpose(this_off_firing))

## 3D Plot for single trajectory
fig = plt.figure()
ax = Axes3D(fig)
i = 2
trial_len = int((tot_time-window_size)/step_size)-1
ran_inds = np.arange((trial_len*i),(trial_len*(i+1)))
this_cmap = Colormap('hsv')
p = ax.scatter(off_f_red[ran_inds,0],off_f_red[ran_inds,1],off_f_red[ran_inds,2],
               c =np.linspace(1,255,len(ran_inds)),cmap='hsv')
ax.plot(off_f_red[ran_inds,0],off_f_red[ran_inds,1],off_f_red[ran_inds,2])
fig.colorbar(p)

## 3D animated scatter plot (DOESN'T WORK)
fig = plt.figure()
ax = Axes3D(fig)
scat = ax.scatter(off_f_red[0,0],off_f_red[0,1],off_f_red[0,2])

def animate(i):
    scat.set_xdata(off_f_red[i,0])
    scat.set_ydata(off_f_red[i,1])
    scat.set_zdata(off_f_red[i,2])
    
anim = animation.FuncAnimation(fig, animate, interval = 100, frames = len(ran_inds)-1)
plt.draw()
plt.show()

def main():
    numframes = 100

    fig = plt.figure()
    scat = ax.scatter(off_f_red[i,0],off_f_red[i,1],off_f_red[i,2])

    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
                                  fargs=(color_data, scat))
    plt.show()

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,

main()