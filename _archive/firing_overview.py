import tables
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import convolve

os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data

os.chdir('/media/bigdata/PyHMM/PyHMM/')
from fake_firing import *

dir_list = ['/media/bigdata/jian_you_data/des_ic']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*.h5',recursive=True)
    
file = 4
data_dir = os.path.dirname(file_list[file])
data = ephys_data(data_dir = data_dir ,file_id = file, use_chosen_units = True)
data.firing_rate_params = dict(zip(['step_size','window_size','total_time','calc_type','baks_len'],
                               [25,250,7000,'baks',100]))
data.get_data()
data.get_firing_rates()    
data.get_normalized_firing()
data.firing_overview('off')

num_nrns = data.all_normal_off_firing.shape[0]

# Plot firing rates
square_len = np.int(np.ceil(np.sqrt(num_nrns)))
fig, ax = plt.subplots(square_len,square_len)
plt.title(file)

nd_idx_objs = []
for dim in range(ax.ndim):
    this_shape = np.ones(len(ax.shape))
    this_shape[dim] = ax.shape[dim]
    nd_idx_objs.append(np.broadcast_to( np.reshape(np.arange(ax.shape[dim]),this_shape.astype('int')), ax.shape).flatten())

for nrn in range(num_nrns):
    plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
    data.imshow(data.all_normal_off_firing[nrn,:,:])
    
# Check spike counts

spikes = np.asarray(data.off_spikes)
sum_spikes = np.sum(spikes,axis=2)
np.sum(spikes[:,:,:,2000:5000],axis=(2,3)).T
