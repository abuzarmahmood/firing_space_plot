import numpy as np

step_size = 25 
window_size = 250
total_time = 7000

spikes = np.asarray(data.off_spikes)

bin_inds = (0,window_size)
total_bins = int((total_time - window_size + 1) / step_size) + 1
bin_list = [(taste,trial,neuron,\
        (bin_inds[0]+step,bin_inds[1]+step)) \
        for step in np.arange(total_bins)*step_size \
        for taste in range(spikes.shape[0]) \
        for trial in range(spikes.shape[1]) \
        for neuron in range(spikes.shape[2])]

firing_rate = np.empty((spikes.shape[0],spikes.shape[1],spikes.shape[2],total_bins))
for bin_inds in bin_list:
    firing_rate[bin_inds[0],bin_inds[1],bin_inds[2],bin_inds[3][0]//step_size] = \
        np.sum(spikes[bin_inds[0],bin_inds[1],bin_inds[2],bin_inds[3][0]:bin_inds[3][1]])

#================================================#
# Version with 3D array (single taste) 
#================================================#

spikes = data.off_spikes[0]

bin_inds = (0,window_size)
total_bins = int((total_time - window_size + 1) / step_size) + 1
bin_list = [(trial,neuron,\
        (bin_inds[0]+step,bin_inds[1]+step)) \
        for trial in range(spikes.shape[0]) \
        for neuron in range(spikes.shape[1])
        for step in np.arange(total_bins)*step_size ]

firing_rate = np.empty((spikes.shape[0],spikes.shape[1],total_bins))
for bin_inds in bin_list:
    firing_rate[bin_inds[0],bin_inds[1],bin_inds[2][0]//step_size] = \
        np.sum(spikes[bin_inds[0],bin_inds[1],bin_inds[2][0]:bin_inds[2][1]])

#================================================#
# SHELVED -- Incorrect calculation
#================================================#

import numpy as np
from numba import jit

start = time.time()
step_size = 25 
window_size = 250
total_time = 7000

spikes = np.asarray(data.off_spikes)

bin_inds = (0,window_size)
total_bins = int((total_time - window_size + 1) / step_size) + 1
bin_list = [(taste,trial,\
        (bin_inds[0]+step,bin_inds[1]+step)) \
        for step in np.arange(total_bins)*step_size \
        for taste in range(spikes.shape[0]) \
        for trial in range(spikes.shape[1])] \

@jit(nopython = True)
def bin_sum(array, bin_list):
    firing_rate = np.empty((spikes.shape[0],spikes.shape[1],total_bins))
    for bin_inds in bin_list:
        firing_rate[bin_inds[0],bin_inds[1],bin_inds[2][0]//step_size] = \
            np.sum(spikes[bin_inds[0],bin_inds[1],bin_inds[2][0]:bin_inds[2][1]])
    return firing_rate


firing_rate = np.empty((spikes.shape[0],spikes.shape[1],spikes.shape[2],total_bins))
for nrn in range(spikes.shape[2]):
    firing_rate[:,:,nrn,:] = bin_sum(spikes[:,:,nrn,:], bin_list)
        

