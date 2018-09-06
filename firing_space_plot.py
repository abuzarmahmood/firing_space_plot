#  ______ _      _               _______        _           _                   
# |  ____(_)    (_)             |__   __|      (_)         | |                  
# | |__   _ _ __ _ _ __   __ _     | |_ __ __ _ _  ___  ___| |_ ___  _ __ _   _ 
# |  __| | | '__| | '_ \ / _` |    | | '__/ _` | |/ _ \/ __| __/ _ \| '__| | | |
# | |    | | |  | | | | | (_| |    | | | | (_| | |  __/ (__| || (_) | |  | |_| |
# |_|    |_|_|  |_|_| |_|\__, |    |_|_|  \__,_| |\___|\___|\__\___/|_|   \__, |
#                         __/ |               _/ |                         __/ |
#                        |___/               |__/                         |___/

# 1) Open single file for now
# 2) extract spiking for all tastes for on&off conditions
# 3) Compute firing rate for ENTIRE 7s (or whatever length of recording)
# 3.5) Maybe smooth firing rate
# 4) Project into 'n' dim space and reduce dimensions

# Use only responsive neurons (probs not needed)
# Try projecting all tastes at same time
######################### Import dat ish #########################
import tables
#import easygui
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.animation as animation
from scipy.spatial import distance_matrix as dist_mat
from scipy import stats

from scipy.ndimage.filters import gaussian_filter1d
import scipy.signal as sig
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D
import time

#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#

#dir_name = "/media/sf_shared_folder/jian_you_data/tastes_separately/file_1"
dir_name = "/media/bigdata/jian_you_data/all_tastes/file_1/"
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
firing_len = int((tot_time-window_size)/step_size)-1
off_firing = []

## Moving Window
for l in range(len(off_spikes)): # How TF do you get nan's from means?
    # [trials, nrns, time]
    this_off_firing = np.zeros((off_spikes[0].shape[0],off_spikes[0].shape[1],firing_len))
    for i in range(this_off_firing.shape[0]):
        for j in range(this_off_firing.shape[1]):
            for k in range(this_off_firing.shape[2]):
                this_off_firing[i, j, k] = np.mean(off_spikes[l][i, j, step_size*k:step_size*k + window_size])
    #this_off_firing = this_off_firing.reshape((this_off_firing.shape[1],this_off_firing.shape[0]*this_off_firing.shape[2]))
    off_firing.append(this_off_firing)
    
## Inter-spike interval
# =============================================================================
# off_firing = []
# for l in range(len(off_spikes)):
#     this_off_firing = np.zeros(off_spikes[0].shape)
#     for i in range(this_off_firing.shape[0]):
#         for j in range(this_off_firing.shape[1]):
#             this_trial = off_spikes[l][m,n,:]
#             spike_inds = np.where(off_spikes[l][i,j,:]>0)[0]
#             f_rate = np.reciprocal(np.diff(spike_inds).astype(float))
#             for k in range(len(f_rate)):
#                 this_off_firing[i,j,spike_inds[k]:] = f_rate[k]
#     off_firing.append(this_off_firing)  
# =============================================================================
    
    
    # Normalize firing (over every trial of every neuron)
# =============================================================================
#     for l in range(len(off_firing)):
#         for m in range(off_firing[0].shape[0]):
#             for n in range(off_firing[0].shape[1]):
#                 min_val = np.min(off_firing[l][m,n,:])
#                 max_val = np.max(off_firing[l][m,n,:])
#                 off_firing[l][m,n,:] = (off_firing[l][m,n,:] - min_val)/(max_val-min_val)
#     
# =============================================================================
    # Normalized firing of every neuron over entire dataset
    off_firing_array = np.asarray(off_firing) #(taste x nrn x trial x time)
    for m in range(off_firing_array.shape[1]): # nrn
        min_val = np.min(off_firing_array[:,m,:,:])
        max_val = np.max(off_firing_array[:,m,:,:])
        for l in range(len(off_firing)): #taste
            for n in range(off_firing[0].shape[1]): # trial
                off_firing[l][m,n,:] = (off_firing[l][m,n,:] - min_val)/(max_val-min_val)


all_off_f_temp = np.asarray(off_firing)
all_off_f = np.zeros((all_off_f_temp.shape[2], int(all_off_f_temp.size / all_off_f_temp.shape[2])))
count = 0
for i in range(all_off_f_temp.shape[0]):
    for j in range(all_off_f_temp.shape[1]):
        for k in range(all_off_f_temp.shape[3]):
            all_off_f[:,count] = all_off_f_temp[i,j,:,k]
            count += 1

#
#  ______           _ _     _ _               _____  _     _   
# |  ____|         | (_)   | (_)             |  __ \(_)   | |  
# | |__  _   _  ___| |_  __| |_  __ _ _ __   | |  | |_ ___| |_ 
# |  __|| | | |/ __| | |/ _` | |/ _` | '_ \  | |  | | / __| __|
# | |___| |_| | (__| | | (_| | | (_| | | | | | |__| | \__ \ |_ 
# |______\__,_|\___|_|_|\__,_|_|\__,_|_| |_| |_____/|_|___/\__|
#


############ Euclidean distance from starting point of each trial ############
# First attempt -> straight up subtraction from starting point
# Part A -> Run a loop over all trials
dist_list = []
for taste in off_firing:
    dist_temp = np.empty((taste.shape[1]))
    for i in range(taste.shape[1]):
        dist_temp[i] = np.linalg.norm(taste[:,i] - taste[:,0])
    dist_list.append(dist_temp)
    
fig=plt.figure(figsize=(21, 6))
columns = 7
rows = 2
count = 0
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.plot(dist_list[0][count*firing_len:(count+1)*firing_len])
    count +=1
plt.show()
    
# Part B -> Run over all trials but use respective basline for every trial

# Part C -> Calculate a distance matrix...patterning will show regularity of trajectory
# Normalize and sum dist matrices for all trials
# Assuming state transition are bounded in time, there should be a trend
# Observation: Trials for different tastes show different structure
taste = 0
#dist_array = np.empty((firing_len,firing_len,np.int(off_firing[0].shape[1]/firing_len)))
dist_array = np.empty((firing_len,firing_len,off_firing[0].shape[0]))
for trial in range(dist_array.shape[2]):
    #dat = np.transpose(off_firing[taste][:,firing_len*trial:firing_len*(trial+1)])
    dat = np.transpose(off_firing[taste][trial,:,:])
    dist_array[:,:,trial] = dist_mat(dat,dat)

fig=plt.figure(figsize=(21, 6))
columns = 7
rows = 2
count = 0
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(stats.zscore(dist_array[:,:,count]))
    count +=1
plt.show()
#fig.savefig('taste%i.png' % taste)



for i in range(15):
    fig = plt.figure()
    #p = plt.imshow(stats.zscore(np.sum(dist_array,axis=2)))
    p = plt.imshow(stats.zscore(dist_array[:,:,i]))
    fig.colorbar(p)

# Second attempt -> subtract means of chunks of time-points to avert noise effects

# Third attempt
# Use PCA find relevant parts of trajectory
# See if 95% variance can be explained by significantly lower dimensions
pca = PCA(n_components=12)
pca.fit(off_firing[0])
np.sum(pca.explained_variance_ratio_)


###############################################################################
def lin_interp(seq, new_len, normalize=True):
    # Do linear interpolation with however many multiple you can fit
    # With remaining ones, just insert values randomly
    interp_len = new_len//len(seq)
    new_seq = []
    try:
        [new_seq.append(np.linspace(seq[i],seq[i+1],interp_len)) for i in range(len(seq))]
    except:
        pass
    new_seq = np.asarray(new_seq)
    new_seq = np.reshape(new_seq, ((new_seq.shape[0]*new_seq.shape[1]),1))
    
    if normalize:
        new_seq = new_seq/np.max(new_seq)
    
    return new_seq

#all_off_f = all_off_f[~np.isnan(all_off_f)]
## Make sure there are not problems with reshaping    
## Maybe smooth firing rates

## Test plots for spikes to firing rate  
inds = [1,3,4] # Taste, trial, nrn
spikes = off_spikes[inds[0]][inds[1],inds[2],:]
rate = off_firing[inds[0]][inds[1],inds[2],:]
#rate2 = sig.resample(rate,spikes.shape[0])
#rate2 = rate2/np.max(rate2)
plt.plot(spikes)
plt.plot(lin_interp(rate,len(spikes)))

## All da neurons, all da tastes, and ALL DA PLOTS!
# =============================================================================
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

#  _____  _             _____          _            _   _             
# |  __ \(_)           |  __ \        | |          | | (_)            
# | |  | |_ _ __ ___   | |__) |___  __| |_   _  ___| |_ _  ___  _ __  
# | |  | | | '_ ` _ \  |  _  // _ \/ _` | | | |/ __| __| |/ _ \| '_ \ 
# | |__| | | | | | | | | | \ \  __/ (_| | |_| | (__| |_| | (_) | | | |
# |_____/|_|_| |_| |_| |_|  \_\___|\__,_|\__,_|\___|\__|_|\___/|_| |_|
#

################### Reduce dimensions ########################
off_f_red = LLE(n_neighbors = 50,n_components=2).fit_transform(np.transpose(all_off_f))
#off_f_red = LLE(n_neighbors = 50,n_components=3).fit_transform(np.transpose(off_firing[0]))
#off_f_red = TSNE(n_components=3).fit_transform(np.transpose(all_off_f))

## 3D Plot for single trajectory
fig = plt.figure()
ax = Axes3D(fig)
i = 3
trial_len = int((tot_time-window_size)/step_size)-1
ran_inds = np.arange((trial_len*i),(trial_len*(i+1)))
this_cmap = Colormap('hsv')
p = ax.scatter(off_f_red[ran_inds,0],off_f_red[ran_inds,1],off_f_red[ran_inds,2],
               c =np.linspace(1,255,len(ran_inds)),cmap='hsv')
ax.plot(off_f_red[ran_inds,0],off_f_red[ran_inds,1],off_f_red[ran_inds,2])
fig.colorbar(p)

## 2D animated scatter plot
###########################
fig, ax = plt.subplots()
x, y = off_f_red[:,0],off_f_red[:,1]
sc = ax.scatter([],[])
plt.xlim(min(x),max(x))
plt.ylim(min(y),max(y))

def animate(i):
    #x.append(np.random.rand(1)*10)
    #y.append(np.random.rand(1)*10)
    #sc.set_offsets(np.c_[x,y])
    sc.set_offsets([x[i],y[i]])

ani = animation.FuncAnimation(fig, animate, 
                frames=range(len(x)), interval=10, repeat=True) 
plt.show()

###############################
x, y = off_f_red[:,0],off_f_red[:,1]
plt.scatter(x,y,c=np.floor(np.linspace(0,4,len(x))))
plt.colorbar()

x, y = off_f_red[:,0],off_f_red[:,1]
for i in range(4):
    inds = range(int(i*(off_f_red.shape[0]/4)), int((i+1)*(off_f_red.shape[0]/4)))
    plt.figure()
    plt.scatter(x[inds],y[inds])
    
## Plot all trajectories to see tastewise effect
################################################
x, y = off_f_red[:,0],off_f_red[:,1]
all_trajs = [[x[(i*firing_len):(i+1)*firing_len], y[(i*firing_len):(i+1)*firing_len]] for i in range(int(x.size/firing_len))]
for traj in all_trajs:
    plt.figure()
    plt.scatter(traj[0],traj[1],c = np.linspace(0,1,traj[0].size))

rows = 4
cols = 15
count = 1
fig, axes = plt.subplots(rows,cols,sharex = 'all',sharey='all')
for i in range(rows):
    for j in range(cols):
        axes[i,j].scatter(all_trajs[count][0],all_trajs[count][1],c = np.linspace(0,1,all_trajs[0][0].size),s=1)
        #y = pdf_plot(x,params)
        #axes[i,j].plot(x,y,colors[int(params[2])])
        count +=1
        print(count)
   
fig.tight_layout()