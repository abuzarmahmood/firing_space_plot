
## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from itertools import product
from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from scipy.signal import fftconvolve
from scipy.stats import zscore
from scipy.signal import savgol_filter as sk_ft

########################################################
# Define functions and relevant variables to extract data
########################################################

def remove_node(path_to_node, hf5):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),os.path.basename(path_to_node))

def firing_overview(data, t_vec = None, y_values_vec = None,
                    interpolation = 'nearest',
                    cmap = 'jet',
                    #min_val = None, max_val=None, 
                    cmap_lims = 'individual',
                    subplot_labels = None):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron
    """

    #if min_val is None:
    #    min_val = np.min(data,axis=None)
    #if max_val is None:
    #    max_val = np.max(data,axis=None)
    if cmap_lims == 'shared':
        min_val, max_val = np.repeat(np.min(data,axis=None),data.shape[0]),\
                                np.repeat(np.max(data,axis=None),data.shape[0])
    else:
        min_val,max_val = np.min(data,axis=tuple(list(np.arange(data.ndim)[1:]))),\
                            np.max(data,axis=tuple(list(np.arange(data.ndim)[1:])))
    if t_vec is None:
        t_vec = np.arange(data.shape[-1])
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])

    if data.shape[-1] != len(t_vec):
        raise Exception('Time dimension in data needs to be'\
            'equal to length of time_vec')
    num_nrns = data.shape[0]

    # Plot firing rates
    square_len = np.int(np.ceil(np.sqrt(num_nrns)))
    fig, ax = plt.subplots(square_len,square_len, sharex='all',sharey='all')
    
    nd_idx_objs = []
    for dim in range(ax.ndim):
        this_shape = np.ones(len(ax.shape))
        this_shape[dim] = ax.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to( 
                    np.reshape(
                        np.arange(ax.shape[dim]),
                        this_shape.astype('int')), ax.shape).flatten())
    
    if subplot_labels is None:
        subplot_labels = np.zeros(num_nrns)
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])
    for nrn in range(num_nrns):
        plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
        plt.gca().set_title('{}:{}'.format(int(subplot_labels[nrn]),nrn))
        plt.gca().pcolormesh(t_vec, y_values_vec,
                data[nrn,:,:],cmap=cmap,
                vmin = min_val[nrn], vmax = max_val[nrn])
    return ax



# ___       _ _   _       _ _          _   _             
#|_ _|_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __  
# | || '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
# | || | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
#|___|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|
#                                                        

###########################################################
### Extract Data
###########################################################

# Define middle channels in board
middle_channels = np.arange(8,24)

# Create file to store data + analyses in
data_hdf5_name = 'AM_LFP_analyses.h5'
data_folder = '/media/bigdata/Abuzar_Data/lfp_analysis'
data_hdf5_path = os.path.join(data_folder, data_hdf5_name)

with tables.open_file(data_hdf5_path,'r') as hf5:
    firing_node_list  = [x for x in hf5.root.firing._f_walknodes() \
            if 'normalized_firing_array' in x.__str__()]
    print('Extracting firing info')
    firing_array_list = [x[:] for x in tqdm(firing_node_list )]
    # Extract all node names with firing array
    node_path_list = [os.path.dirname(x.__str__().split(" ")[0]) for x in firing_node_list]
    # Pull electrode numbers from each array
    electrode_num_list  = [hf5.get_node(path,'unit_electrode')[:] for path in node_path_list]
    spike_array_list =  [hf5.get_node(path,'spike_array')[:] for path in node_path_list]

print('Calculating firing correlations')

this_node_num = 0

firing_array = firing_array_list[this_node_num].swapaxes(0,1)
# Should be a better wap to reshape spike_array
spike_array = spike_array_list[this_node_num].swapaxes(0,2).swapaxes(1,2)
electrode_nums = electrode_num_list[this_node_num]
middle_channels_bool = np.array([True if channel in middle_channels else False \
        for channel in electrode_nums ])
firing_array_split = [firing_array[middle_channels_bool], firing_array[~middle_channels_bool]]
spike_array_split = [spike_array[middle_channels_bool], spike_array[~middle_channels_bool]]
channel_num_split = \
        [electrode_nums[middle_channels_bool], electrode_nums[~middle_channels_bool]]
relative_channel_num_split = \
        [np.arange(len(electrode_nums))[middle_channels_bool], 
                np.arange(len(electrode_nums))[~middle_channels_bool]]

#firing_array_long = np.array([np.reshape(x,tuple((np.prod(x.shape[:2]),x.shape[-1])))\
#        for x in firing_array])

###########################################################
### Perform Correlations 
###########################################################

# Calc firing rates with half-gaussian filter
smooth_kern_sd = 200 # seconds; For plotting
gaussian_func = lambda x,sd : (1/(sd*np.sqrt(2*np.pi)))*\
        np.exp(-0.5*((x/sd)**2))
smooth_kern_temp = gaussian_func(np.arange(-3*smooth_kern_sd,3*smooth_kern_sd),smooth_kern_sd)
smooth_kern = smooth_kern_temp[:smooth_kern_temp.shape[0]//2]


#spike_array_long = np.array([np.reshape(x,tuple((np.prod(x.shape[:2]),x.shape[-1])))\
#        for x in spike_array])
#spike_array_long += np.random.random(spike_array_long.shape)*1e-9
#smooth_firing_long = np.apply_along_axis(
#                lambda x : np.convolve(x,smooth_kern,mode='same'),
#                axis = -1,
#                arr = spike_array_long )
#
#firing_overview(firing_array_long)
#firing_overview(smooth_firing_long, nime_step = 1);plt.show()

# Take fixed window around every spike for 'master' neuron
# Split trial for other neuron same way
# Perform convolution and save mean value for window
fin_conv_values = np.zeros(array1.shape)
window_radius = 250
array1 = spike_array_split[0][0]
array2 = spike_array_split[1][0]
array1[array1<0.1] = 0
array2[array2<0.1] = 0
trial_iters = list(np.ndindex(array1.shape[:2]))
#this_iter = trial_iters[0]
for this_iter in trial_iters:
    nrn1_trial = array1[this_iter]
    nrn2_trial = array2[this_iter]
    spike_inds = np.where(nrn1_trial)[0]
    spike_inds = [x for x in spike_inds if x-window_radius>0]
    spike_windows = [(x-window_radius,x+window_radius) for x in spike_inds] 
    # Remove any windows with t<0
    #spike_windows = [x for x in spike_windows if x[0]>0 and x[1]>0] 
    conv_arrays = np.array([[trial[window[0]:window[1]] for window in spike_windows] \
            for trial in [nrn1_trial,nrn2_trial]])
    # Mean cross correlation per spike
    out = np.mean(fftconvolve(conv_arrays[0],conv_arrays[1],mode = 'same',axes=-1),axis=-1)
    fin_conv_values[(*this_iter,spike_inds)] = out
plot_image(out);plt.show()

# Reshape spike_arrays to be total_steps x window_size arrays
def step_break_array(array, window_size, step_size, zscore_bool=True):
    total_time = array.shape[-1] - (array.shape[-1] % step_size)
    bin_inds = (0,window_size)
    total_bins = int((total_time - window_size + 1) // step_size) + 1 
    bin_list = [(bin_inds[0]+step,bin_inds[1]+step) \
            for step in np.arange(total_bins)*step_size ]
    array_steps = np.empty(tuple((*array.shape[:-1],total_bins,window_size)))
    for bin_num,bin_inds in enumerate(bin_list):
        array_steps[...,bin_num, :] = \
                array[...,bin_inds[0]:bin_inds[1]]
    if zscore_bool:
        #array_steps += np.random.random(array_steps.shape)*1e-9
        array_steps  = zscore(array_steps,axis=-1) 
    return array_steps

def step_convolve(array1,array2,window_size,step_size,zscore_bool=True):
    # Arrays will be convolved along the last axis
    # Require that both arrays have the same size
    if array1.shape != array2.shape:
        raise Exception('Arrays need to have same shape')
    step_break_arrays = step_break_array(np.array([array1,array2]),
            window_size,step_size,zscore_bool=zscore_bool)
    out = fftconvolve(step_break_arrays[0],step_break_arrays[1],mode = 'same',axes=-1)
    return np.squeeze(out)

def plot_image(array):
    plt.imshow(array, interpolation='nearest',aspect='auto',origin='lower',cmap='jet')

spike_array_split = [x + np.random.random(x.shape)*1e-9 for x in spike_array_split]
smooth_firing_split = [np.apply_along_axis(
                lambda x : np.convolve(x,smooth_kern,mode='same'),
                axis = -1,
                arr = spike_array) for spike_array in spike_array_split]
# Generate pairs of neurons to correlate
from itertools import product
nrn_pair_list = list(product(*[np.arange(x.shape[0]) for x in smooth_firing_split]))

pair_num = 100
this_pair = nrn_pair_list[pair_num]

window_size = 250
step_size = 25
total_bins = int((smooth_firing_split[0].shape[-1] - window_size + 1) // step_size) + 1 

#firing_conv_array = np.zeros((len(nrn_pairs),
#                    *smooth_firing_split[0].shape[1:3],
#                    total_bins, window_size))
#spiking_conv_array = np.zeros((len(nrn_pairs),
#                    *smooth_firing_split[0].shape[1:3],
#                    total_bins, window_size))

# Wrapper functions to calculation firing and spiking convolutions
def spike_convolutions(nrn_pair):
    this_pair_spiking = \
            [x[nrn_pair[ind]] for ind,x in enumerate(spike_array_split)]
    spike_conv = step_convolve(this_pair_spiking[0],this_pair_spiking[1],
                    window_size = window_size, step_size = step_size,zscore_bool=False)
    return spike_conv

def firing_convolutions(nrn_pair):
    this_pair_firing = \
            [x[ind] for ind,x in zip(nrn_pair,smooth_firing_split)]
            #[x[nrn_pair[ind]] for ind,x in enumerate(smooth_firing_split)]
    firing_conv = step_convolve(this_pair_firing[0],this_pair_firing[1],
                    window_size = window_size, step_size = step_size,
                    zscore_bool=False)
    return firing_conv

def firing_convolutions_shuffle(nrn_pair, shuffle_count = 100):
    this_pair_firing = \
            [x[ind] for ind,x in zip(nrn_pair,smooth_firing_split)]
    choices = [np.random.choice(range(this_pair_firing[0].shape[1]),shuffle_count) \
            for x in range(len(nrn_pair))]
    firing_conv = step_convolve(
                this_pair_firing[0][:,choices[0]],
                this_pair_firing[1][:,choices[1]],
                window_size = window_size, step_size = step_size,
                zscore_bool=False)
    return np.mean(firing_conv,axis=1)

from scipy.stats import gamma
from scipy.stats import zscore

alpha = np.linspace(0.1,2,100)
gamma_mat = zscore(np.array([gamma.rvs(a,size=10000) for a in alpha]),axis=-1)
bins = np.linspace(np.min(gamma_mat,axis=None),np.max(gamma_mat,axis=None),100)
hist_out = np.array([np.histogram(gamma_vec,bins)[0] for gamma_vec in gamma_mat])
mads = asymmetric_mad(gamma_mat)
med_val = np.median(gamma_mat,axis=-1)
plt.pcolormesh(bins,np.arange(hist_out.shape[0]),hist_out)
for dist_num in range(gamma_mat.shape[0]):
    plt.vlines([med_val[dist_num] - 2*mads[0][dist_num], med_val[dist_num] + 2*mads[1][dist_num]] ,
            dist_num-0.5,dist_num+0.5)
    plt.vlines(med_val[dist_num],dist_num-0.5,dist_num+0.5,color='red')
plt.show()

def asymmetric_mad(array,axis=-1):
    median_val = np.median(array,axis=axis)
    med_diff = array - np.expand_dims(median_val,axis=axis)
    lower_vals, higher_vals = np.copy(med_diff),np.copy(med_diff)
    lower_vals[lower_vals > 0] = np.nan
    higher_vals[higher_vals < 0] = np.nan
    lower_mad = np.sqrt(np.nanmean(lower_vals,axis=axis)**2)
    higher_mad = np.sqrt(np.nanmean(higher_vals,axis=axis)**2)
    return lower_mad, higher_mad

spike_conv_array = np.array( 
        Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(spike_convolutions)(nrn_pair) for nrn_pair in tqdm(nrn_pair_list)))
spike_conv_array = np.mean(spike_conv_array,axis=2)
firing_conv_array = np.array(
        Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(firing_convolutions)(nrn_pair) for nrn_pair in tqdm(nrn_pair_list)))
#firing_conv_array = np.mean(firing_conv_array,axis=2)
firing_shuffle_conv_array = np.array(
        Parallel(n_jobs = mp.cpu_count()-2)\
        (delayed(firing_convolutions_shuffle)(nrn_pair) for nrn_pair in tqdm(nrn_pair_list)))
firing_shuffle_conv_array = np.broadcast_to(firing_shuffle_conv_array[:,:,np.newaxis],
        firing_conv_array.shape)
#firing_shuffle_conv_array = np.mean(firing_shuffle_conv_array,axis=2)
shuffle_corrected_firing_conv = firing_conv_array - firing_shuffle_conv_array
mean_shuffle_corrected_conv = np.mean(shuffle_corrected_firing_conv,axis=-1)
mean2_shuffle_corrected_conv = np.mean(mean_shuffle_corrected_conv,axis=2)
std_shuffle_corrected_conv = np.std(mean_shuffle_corrected_conv,axis=2)

plot_pair = 10
plt.plot(mean2_shuffle_corrected_conv[plot_pair].T)
for mean,std in zip(mean2_shuffle_corrected_conv[plot_pair],std_shuffle_corrected_conv[plot_pair]):
    plt.fill_between(x = range(mean.shape[-1]),
            y1 = mean-2*std, y2 = mean+2*std, alpha = 0.5)
plt.show()

firing_overview(shuffle_corrected_firing_conv[0][0],cmap_lims='shared');plt.show()

############################################################
## Shuffled correlations
############################################################
# Shuffle trials for every taste within a neuron pair
# Save only mean shuffled correlations

#this_pair_firing = [x[this_pair[ind]] for ind,x in enumerate(smooth_firing_split)]

#this_pair_firing_long = [np.reshape(x,tuple((np.prod(x.shape[:2]),x.shape[-1])))\
#        for x in this_pair_firing]
#plt.subplot(121)
#plot_image(this_pair_firing_long[0])
#plt.subplot(122)
#plot_image(this_pair_firing_long[1])
#plt.show()

#firing_conv_array[pair_num] = step_convolve(this_pair_firing[0],this_pair_firing[1],
#                window_size = window_size, step_size = step_size, zscore_bool=False)
#firing_overview(np.mean(this_conv[:,:],axis=1))
#plt.show()

#firing_overview(np.mean(this_conv[:,:],axis=1),interpolation='Gaussian', time_step = 1)
#plt.show()

#for repeats in range(10):
#    fig = plt.gcf()
#    fig.savefig('/home/abuzarmahmood/Pictures/test{}.png'.format(repeats))
#    plt.close(fig)
#plt.show()

# _____         _      ____               
#|_   _|__  ___| |_   / ___|__ _ ___  ___ 
#  | |/ _ \/ __| __| | |   / _` / __|/ _ \
#  | |  __/\__ \ |_  | |__| (_| \__ \  __/
#  |_|\___||___/\__|  \____\__,_|___/\___|
#                                         
# Generate test cases for correlations
# Pair of spike trains with varying correlations
# 1/4 : No correlation
# 2/4 : Very strong periodic correlation
# 3/4 : Half as strong periodic previous section
# 3/4 : Aperiodic correlation ## Needs to happen at a slow enough timescale for there to be correlations
# Poisson approximation
smooth_kern_sd = 50e-3 # seconds; For plotting
gaussian_func = lambda x,sd : (1/(sd*np.sqrt(2*np.pi)))*\
        np.exp(-0.5*((x/sd)**2))
smooth_kern = gaussian_func(np.arange(-3*smooth_kern_sd,3*smooth_kern_sd,step=dt),smooth_kern_sd)
smooth_kern_narrow = gaussian_func(np.arange(-3*smooth_kern_sd,3*smooth_kern_sd,step=dt),smooth_kern_sd/10)
smooth_kern /= np.sum(smooth_kern)
smooth_kern_narrow /= np.sum(smooth_kern_narrow)
dt = 1e-3
t_epoch = 20 # 10 second epochs
max_rate = 100
noise_ratio = 0.5

epochs = [np.concatenate([
    np.random.random(int(t_epoch/dt)) * max_rate,
        (np.sin(2*np.pi*np.arange(t_epoch, step = dt))**2 + \
                noise_ratio*np.random.random(int(t_epoch/dt)))*max_rate,
        ((np.sin(2*np.pi*np.arange(t_epoch, step = dt)))**2 + \
                noise_ratio*np.random.random(int(t_epoch/dt)))*max_rate/2]) for x in range(2)]
common_component = np.convolve(np.random.random(int(t_epoch/dt)) * max_rate,smooth_kern_narrow,'same')
epochs = [np.concatenate((x,common_component),axis=-1) for x in epochs]
spike_probs = [np.random.random(epochs[0].shape) for trials in range(2)]
spike_train_list = [((x)<epoch*dt)*1 for x,epoch in zip(spike_probs,epochs)]
firing_rate_list = [np.convolve(x,smooth_kern, 'same') for x in spike_train_list]
fig,ax = plt.subplots(3,1,sharex=True)
ax[0].plot(np.arange(len(epochs[0]))*dt,epochs[0])
ax[0].plot(np.arange(len(epochs[0]))*dt,epochs[1])
ax[1].scatter(np.where(spike_train_list[0])[0]*dt,np.ones(sum(spike_train_list[0])),alpha=0.1)
ax[1].scatter(np.where(spike_train_list[1])[0]*dt,np.ones(sum(spike_train_list[1]))*2,alpha=0.1)
ax[2].plot(np.arange(len(epochs[0]))*dt, firing_rate_list[0])
ax[2].plot(np.arange(len(epochs[0]))*dt, firing_rate_list[1])
plt.show()

# Correlation parameters
window_size = 2 #sec
step_size = 2 #sec

# Spike train correlation
out = step_convolve(spike_train_list[0],spike_train_list[1],
        int(window_size/dt), int(step_size/dt),zscore_bool=False)
plt.pcolormesh(np.linspace(-window_size/2,window_size/2,out.shape[-1]),
        np.linspace(0,len(epochs[0])*dt,out.shape[0]),out,cmap='jet')
plt.xlabel('Time lag (sec)')
plt.ylabel('Window center (sec)')
plt.show()

# Firing rate correlations
out = step_convolve(firing_rate_list[0],firing_rate_list[1],
        int(window_size/dt), int(step_size/dt),zscore_bool=False)
# Test histogram
bin_num = 10
percentiles = np.arange(0,100,100//bin_num)
abs_out = np.abs(out)
bins_edges = np.percentile(abs_out,percentiles,axis=None)
window_centers = np.linspace(0,len(epochs[0])*dt,abs_out.shape[0])
time_hist = np.array([np.histogram(x,bins_edges)[0] for x in abs_out])
fig, ax = plt.subplots(1,4)
ax[0].pcolormesh(np.linspace(-window_size/2,window_size/2,out.shape[-1]),
        window_centers,out, cmap='jet')
ax[0].set_xlabel('Time lag (sec)')
ax[0].set_ylabel('Window center (sec)')
ax[1].plot(np.mean(abs_out,axis=-1),np.linspace(0,len(epochs[0])*dt,out.shape[0]))
ax[1].plot(sk_ft(np.mean(abs_out,axis=-1),21,1),np.linspace(0,len(epochs[0])*dt,out.shape[0]))
ax[2].pcolormesh(percentiles,
        window_centers ,time_hist, cmap='jet')
ax[2].set_xlabel('Percentile Coherence')
ax[2].set_ylabel('Window center (sec)')
ax[2].set_title('Coherence Percentile count')
ax[3].plot(np.abs(out[:,out.shape[-1]//2]), window_centers)

# Firing rate correlations
out = step_convolve(firing_rate_list[0],firing_rate_list[1],
        int(window_size/dt), int(step_size/dt),zscore_bool=True)
# Test histogram
bin_num = 10
percentiles = np.arange(0,100,100//bin_num)
abs_out = np.abs(out)
bins_edges = np.percentile(abs_out,percentiles,axis=None)
window_centers = np.linspace(0,len(epochs[0])*dt,abs_out.shape[0])
time_hist = np.array([np.histogram(x,bins_edges)[0] for x in abs_out])
fig, ax = plt.subplots(1,4)
ax[0].pcolormesh(np.linspace(-window_size/2,window_size/2,out.shape[-1]),
        window_centers,out, cmap='jet')
ax[0].set_xlabel('Time lag (sec)')
ax[0].set_ylabel('Window center (sec)')
ax[1].plot(np.mean(abs_out,axis=-1),np.linspace(0,len(epochs[0])*dt,out.shape[0]))
ax[1].plot(sk_ft(np.mean(abs_out,axis=-1),21,1),np.linspace(0,len(epochs[0])*dt,out.shape[0]))
ax[2].pcolormesh(percentiles,
        window_centers ,time_hist, cmap='jet')
ax[2].set_xlabel('Percentile Coherence')
ax[2].set_ylabel('Window center (sec)')
ax[2].set_title('Coherence Percentile count')
ax[3].plot(np.abs(out[:,out.shape[-1]//2]), window_centers)
plt.show()

