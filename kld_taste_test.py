"""
Attempt to determine Taste Discriminability of neuron
using KL divergence on firing rate distributions for each
neuron conditions on time
"""

## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import lspopt
import tables
import easygui
import scipy
from scipy.signal import spectrogram
from lspopt import spectrogram_lspopt
import numpy as np
from scipy.signal import hilbert, butter, filtfilt,freqs 
from tqdm import tqdm, trange
from sklearn.utils import resample
from itertools import product
import scipy.signal as signal
import itertools
from scipy.special import gamma

def kl_divergence(vec_a,vec_b):
    """
    Both vectors are 1D arrays
    Vectors will be renormalized
    Order of divergence is D(A||B)
    """
    dat1 = vec_a/np.sum(vec_a)
    dat2 = vec_b/np.sum(vec_b)
    kl_div = np.sum(dat1*np.log(dat1/dat2))
    return kl_div

def gauss_filt(data,window_size):
    """
    data : 1D array
    """
    std = int(window_size/2)
    window = signal.gaussian(window_size, std=std)
    window = window/window.sum()
    filt_data = np.convolve(data,window,mode='same')
    return filt_data

def BAKS(SpikeTimes, Time):
    
    N = len(SpikeTimes)
    a = 4
    b = N**0.8
    sumnum = 0; sumdenum = 0
    
    for i in range(N):
        numerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a)
        denumerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a-0.5)
        sumnum = sumnum + numerator
        sumdenum = sumdenum + denumerator
    h = (gamma(a)/gamma(a+0.5))*(sumnum/sumdenum)
    
    FiringRate = np.zeros((len(Time)))
    for j in range(N):
        K = (1/(np.sqrt(2*np.pi)*h))*np.exp(-((Time-SpikeTimes[j])**2)/((2*h)**2))
        FiringRate = FiringRate + K
        
    return FiringRate


os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extract data
dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM12/AM12_extracted/AM12_4Tastes_191106_085215')
    #ephys_data('/media/bigdata/Abuzar_Data/AM17/AM17_extracted/AM17_4Tastes_191126_084934')

dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()
dat.firing_overview(dat.all_normalized_firing);plt.show()

# Generate firing rate distribution for each taste

# Calculate firing rates using BAKS
nrn_num = 18
this_nrn_spikes = np.array(dat.spikes)[:,:,nrn_num]
# If the neuron doesn't spike in a trial, throw a single spike somewhere
for inds in zip(*np.where(np.sum(this_nrn_spikes,axis=-1)==0)):
    this_nrn_spikes[(*inds,np.random.choice(np.arange(this_nrn_spikes.shape[-1]),1))] = 1
this_nrn = np.array([[BAKS(np.where(trial)[0]/1000,np.arange(len(trial),step=25)/1000) for trial in taste]\
        for taste in this_nrn_spikes])
#dat.imshow(zscore(this_nrn.reshape(120,-1),axis=0));plt.show()
dat.imshow(this_nrn.reshape(120,-1));plt.show()

mean_vals = np.mean(this_nrn,axis=1)
std_vals = np.std(this_nrn,axis=1)
for num,(mean_f,std_f) in enumerate(zip(mean_vals,std_vals)):
    plt.fill_between(x = np.arange(len(mean_f)),
            y1 = mean_f - std_f, y2 = mean_f + std_f,alpha=0.5)
    plt.plot(mean_vals[num])
plt.show()


# Chop firing rates in quartile and time bins for each taste
symbols = 5
quartiles = np.linspace(0,100,symbols+1)
percentiles = np.linspace(0,100,100+1)
quart_vals = np.percentile(this_nrn.flatten(),quartiles)
percentile_vals = np.percentile(this_nrn.flatten(),percentiles)
time_bin_count = 40
time_bins = list(map(int,np.floor(np.linspace(0,this_nrn.shape[2],time_bin_count+1))))

plt.subplot(211)
this_hist = plt.hist(this_nrn.flatten(),100,density=True)
this_ecdf = np.cumsum(this_hist[0])
this_ecdf /= np.max(this_ecdf)
plt.vlines(quart_vals,ymin=0,ymax = np.max(this_hist[0]))
plt.subplot(212)
plt.plot(this_hist[1][1:],this_ecdf,'x')
plt.vlines(quart_vals,ymin=0,ymax = np.max(this_ecdf))
plt.show()

cpd = np.empty((this_nrn.shape[0],symbols,time_bin_count))
for taste in range(this_nrn.shape[0]):
    for time_bin_num in range(1,len(time_bins)):
        cpd[taste,:,time_bin_num-1] = \
                np.histogram(this_nrn[taste,:,time_bins[time_bin_num-1]:time_bins[time_bin_num]],bins = quart_vals)[0]

# Add some noise to CPD to avoid zero errors
cpd += np.random.random(cpd.shape)*1e-9
dat.firing_overview(cpd,cmap='viridis',time_step=1);plt.show()


# Normalize CPD within each bin
#norm_cpd = np.empty(cpd.shape)
#for time_bin in range(cpd.shape[-1]):
#    norm_cpd[...,time_bin] = cpd[...,time_bin] / np.sum(cpd[...,time_bin],axis=None)
norm_cpd = cpd / np.sum(cpd,axis=(1))[:,np.newaxis]

dat.firing_overview(norm_cpd,cmap='viridis',time_step=1);plt.show()

# List all possible combinations of pairwise taste comparisons
taste_comparisons = list(itertools.combinations(range(this_nrn.shape[0]),2))

kld_array = np.empty((len(taste_comparisons),time_bin_count))
for pair_num, pair in enumerate(taste_comparisons):
    for time_bin in range(cpd.shape[-1]):
        kld_array[pair_num,time_bin] = \
            kl_divergence(norm_cpd[pair[0],:,time_bin], norm_cpd[pair[1],:,time_bin])

#dat.imshow(zscore(np.log2(kld_array),axis=-1));plt.show()
plt.subplot(211)
for num,(mean_f,std_f) in enumerate(zip(mean_vals,std_vals)):
    plt.fill_between(x = np.arange(len(mean_f)),
            y1 = mean_f - std_f, y2 = mean_f + std_f,alpha=0.5)
    plt.plot(mean_vals[num])
ax = plt.subplot(212)
ax.imshow(kld_array,interpolation='nearest',aspect='auto')
#plt.pcolormesh(np.arange(kld_array.shape[1]),np.arange(kld_array.shape[0])+1,kld_array)
plt.gca().set_yticks((-0.5,*np.arange(kld_array.shape[0]),kld_array.shape[0]+0.5))
plt.gca().set_yticklabels((None,*taste_comparisons,None))
plt.gca().tight_layout()
plt.show()

# Total taste discriminability is sum KLD
taste_discrim_time = np.sum(kld_array,axis=0)
plt.plot(taste_discrim_time);plt.show()
taste_discrim = np.sum(kld_array,axis=None)

#def calc_taste_discrim(this_nrn):
#    """
#    this_nrn : (taste,trials,time)
#    """
#    # Chop firing rates in quartile and time bins for each taste
#    symbols = 5
#    quartiles = np.linspace(0,100,symbols+1)
#    quart_vals = np.percentile(this_nrn.flatten(),quartiles)
#    time_bin_count = 40
#    time_bins = list(map(int,np.floor(np.linspace(0,this_nrn.shape[2],time_bin_count+1))))
#
#    cpd = np.empty((this_nrn.shape[0],symbols,time_bin_count))
#    for taste in range(this_nrn.shape[0]):
#        for time_bin_num in range(1,len(time_bins)):
#            cpd[taste,:,time_bin_num-1] = \
#                    np.histogram(this_nrn[taste,:,time_bins[time_bin_num-1]:time_bins[time_bin_num]],bins = quart_vals)[0]
#
#    # Add some noise to CPD to avoid zero errors
#    cpd += np.random.random(cpd.shape)*1e-9
#
#    # Normalize CPD within each bin
#    norm_cpd = cpd / np.sum(cpd,axis=(1))[:,np.newaxis]
#
#    # List all possible combinations of pairwise taste comparisons
#    taste_comparisons = list(itertools.combinations(range(this_nrn.shape[0]),2))
#
#    kld_array = np.empty((len(taste_comparisons),time_bin_count))
#    for pair_num, pair in enumerate(taste_comparisons):
#        for time_bin in range(cpd.shape[-1]):
#            kld_array[pair_num,time_bin] = \
#                kl_divergence(cpd[pair[0],:,time_bin], cpd[pair[1],:,time_bin])
#
#    return kld_array

def calc_taste_discrim2(this_nrn, taste_labels):
    """
    this_nrn : (taste,trials,time)
    """
    # Chop firing rates in quartile and time bins for each taste
    symbols = 5
    quartiles = np.linspace(0,100,symbols+1)
    quart_vals = np.percentile(this_nrn.flatten(),quartiles)
    time_bin_count = 40
    time_bins = list(map(int,np.floor(np.linspace(0,this_nrn.shape[-1],time_bin_count+1))))

    cpd = np.empty((len(np.unique(taste_labels)),symbols,time_bin_count))
    for taste in np.sort(np.unique(taste_labels)):
        for time_bin_num in range(1,len(time_bins)):
            cpd[taste,:,time_bin_num-1] = \
                    np.histogram(this_nrn[taste_labels == taste,\
                        time_bins[time_bin_num-1]:time_bins[time_bin_num]],bins = quart_vals)[0]

    # Add some noise to CPD to avoid zero errors
    cpd += np.random.random(cpd.shape)*1e-9

    # Normalize CPD within each bin
    norm_cpd = cpd / np.sum(cpd,axis=(1))[:,np.newaxis]

    # List all possible combinations of pairwise taste comparisons
    taste_comparisons = list(
            itertools.combinations(
                range(len(np.unique(taste_labels))),2))

    kld_array = np.empty((len(taste_comparisons),time_bin_count))
    for pair_num, pair in enumerate(taste_comparisons):
        for time_bin in range(cpd.shape[-1]):
            kld_array[pair_num,time_bin] = \
                kl_divergence(cpd[pair[0],:,time_bin], cpd[pair[1],:,time_bin])

    return kld_array

# Calculate taste_discrim for all neurons
taste_labels = np.sort(np.array([0,1,2,3]*30))
all_spikes = np.array(dat.spikes)
all_spikes = np.rollaxis(all_spikes,2,0)
all_firing = np.zeros((*all_spikes.shape[:-1],all_spikes.shape[-1]//25))
for nrn_num,this_nrn_spikes in enumerate(np.array(all_spikes)):
    for inds in zip(*np.where(np.sum(this_nrn_spikes,axis=-1)==0)):
        this_nrn_spikes[(*inds,np.random.choice(np.arange(this_nrn_spikes.shape[-1]),1))] = 1
    all_firing[nrn_num] = np.array([[BAKS(np.where(trial)[0]/1000,np.arange(len(trial),step=25)/1000) for trial in taste]\
            for taste in this_nrn_spikes])
all_firing_long = np.reshape(all_firing,(all_firing.shape[0],-1,all_firing.shape[-1]))

#test1 = calc_taste_discrim(all_firing[0])
#test2 = calc_taste_discrim2(all_firing_long[0],taste_labels)
#
#plt.subplot(211)
#dat.imshow(test1)
#plt.subplot(212)
#dat.imshow(test2)
#plt.show()

taste_discrim_vec = np.array([calc_taste_discrim(nrn,taste_labels) \
        for nrn in all_firing_long[...,26:]])

# Plot firing of neurons sorted by taste discriminability
dat.firing_overview(dat.all_normalized_firing[np.argsort(np.max(taste_discrim_vec,axis=(1,2)))]);plt.show()
