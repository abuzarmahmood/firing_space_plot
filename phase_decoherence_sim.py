"""
1) Simulate phase reset in GC and absent/weak phase-reset in BLA
    - Check whether this can describe the sudden decoherence on taste delivery
"""

########################################
## Setup
########################################

import numpy as np
import matplotlib as mpl
import pylab as plt
import os
from scipy.signal import hilbert
os.chdir('/media/bigdata/firing_space_plot/ephys_data')
import visualize

def gauss_kern(size):
    x = np.arange(-size,size+1)
    kern = np.exp(-(x**2)/float(size))
    return kern / sum(kern)
def gauss_filt(vector, size):
    kern = gauss_kern(size)
    return np.convolve(vector, kern, mode='same')


########################################
## Simulation 
########################################
band = (2,4) # Frequency range
dt = 1e-3
t_range = (0,5)
t_vec = np.arange(t_range[0],t_range[1],dt)

# ** Don't know how kosher this is but should work fine :p **
# Simulate frequency band by dividing power into bins across frequency band

#freq_bins = 10
#freqs = np.linspace(band[0],band[1],freq_bins)
#x_mat = np.broadcast_to(t_vec[np.newaxis,:], (freq_bins,t_vec.shape[0]))
#x_mat = x_mat * freqs[:,np.newaxis] * 2 * np.pi
#y_mat = np.sin(x_mat)
#mean_y = np.mean(y_mat, axis = 0)
#plt.plot(t_vec, mean_y);plt.show()


## ** OK, scratch that, just use mean frequency
mean_band_freq = np.mean(band)
x_vec = t_vec * 2 * np.pi * mean_band_freq
y_vec = np.sin(x_vec)

# Test plot for single trial, one region
fig, ax = plt.subplots(2,1)
ax[0].plot(t_vec, y_vec)
ax[1].plot(t_vec, x_vec % (2*np.pi))
plt.show()

# Generate trials for both regions
# Baseline : Phase aligned
# At t=2, GC undergoes phase-reset and BLA shows no change
# To create this, concatenate baseline and post-stim phases
trial_num = 10
stim_time = 2
x_mat = np.broadcast_to(x_vec[np.newaxis,:],(trial_num, len(x_vec)))
# Add random start time to every trial
x_mat = x_mat +  np.random.random((trial_num,1))* 2 * np.pi
x_mat = x_mat % (2 * np.pi)
bla_phase = x_mat
gc_post_stim = np.broadcast_to(x_mat[0,t_vec > stim_time],(trial_num, sum(t_vec>stim_time))) 
gc_pre_stim = x_mat[:,t_vec <= stim_time]
gc_phase = np.concatenate((gc_pre_stim, gc_post_stim),axis=-1)

# Test plot for phases
fig, ax = plt.subplots(2,1)
plt.sca(ax[0])
visualize.imshow(gc_phase)
plt.sca(ax[1])
visualize.imshow(bla_phase)
plt.show()

# Calculate coherence
coherence = np.abs(np.mean( np.exp(-1.j * (gc_phase-bla_phase)), axis = 0))
plt.plot(t_vec, coherence)
plt.show()

# Repeat but with smoothened phase reset
visualize.imshow(gc_phase);plt.show()
gc_signal = np.sin(gc_phase)
bla_signal = np.sin(bla_phase)
visualize.imshow(gc_signal);plt.show()

smooth_gc_signal = np.array([ gauss_filt(x, 1000) for x in gc_signal])
# Apply same process to BLA to maintain consistency
smooth_bla_signal = np.array([ gauss_filt(x, 1000) for x in bla_signal])
gc_hilbert = hilbert(smooth_gc_signal)
gc_smooth_phase = np.angle(gc_hilbert)
bla_hilbert = hilbert(smooth_bla_signal)
bla_smooth_phase = np.angle(bla_hilbert)

# Calculate coherence
smooth_coherence = np.abs(np.mean( np.exp(-1.j * \
        (gc_smooth_phase-bla_smooth_phase)), axis = 0))

# Plot results
fig = plt.figure()
ax0 = fig.add_subplot(3,2,1)
ax1 = fig.add_subplot(3,2,2, sharex = ax0, sharey = ax0)
ax2 = fig.add_subplot(3,2,3, sharex = ax0, sharey = ax0)
ax3 = fig.add_subplot(3,2,4, sharex = ax0, sharey = ax0)
ax4 = fig.add_subplot(3,1,3)
ax_list = [ax0,ax1,ax2,ax3]
title_list = ['GC Signal', 'GC Phase', 'BLA Signal', 'BLA Phase']
for title, this_ax in zip(title_list,ax_list):
    this_ax.set_title(title)
    this_ax.set_xlabel('Time (s)')
    this_ax.set_ylabel('Trial #')
ax4.set_title('Phase coherence')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Coherence')
plt.suptitle('Phase reset coherence test \n Freq = {} Hz'\
        .format(int(mean_band_freq)))
plt.sca(ax0)
visualize.imshow(smooth_gc_signal)
plt.sca(ax1)
visualize.imshow(gc_smooth_phase)
plt.sca(ax2)
visualize.imshow(smooth_bla_signal)
plt.sca(ax3)
visualize.imshow(bla_smooth_phase)
plt.sca(ax4)
plt.plot(t_vec, smooth_coherence)
plt.tight_layout()
plt.show()
