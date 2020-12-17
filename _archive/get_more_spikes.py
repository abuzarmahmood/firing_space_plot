# ==============================
# Setup
# ==============================

#Import necessary tools
import numpy as np
import tables
import easygui
import os
import glob
import matplotlib.pyplot as plt
import re
import sys
from tqdm import tqdm, trange
#Import specific functions in order to filter the data file
from scipy.signal import butter
from scipy.signal import filtfilt
from joblib import Parallel,delayed

# ==============================
# Define Functions 
# ==============================

def get_filtered_electrode(data, low_pass, high_pass, sampling_rate):
    el = 0.195*(data)
    m, n = butter(
            2, 
            [2.0*int(low_pass)/sampling_rate, 2.0*int(high_pass)/sampling_rate], 
            btype = 'bandpass')
    filt_el = filtfilt(m, n, el)
    return filt_el

# ==============================
# Collect user input needed for later processing 
# ==============================

#Get name of directory where the data files and hdf5 file sits, 
#and change to that directory for processing
if len(sys.argv)>1:
    dir_name = sys.argv[1]
else:
    dir_name = easygui.diropenbox()

param_file_name = glob.glob(os.path.join(dir_name,'**.params'))[0]
with open(param_file_name,'r') as file:
    params = file.readlines()
before_snapshot = float(params[-3][:-1])
sampling_rate = float(params[-1][:-1])
before_inds = int(before_snapshot*sampling_rate/1000)



# Low pass, high pass, sampling rate
freqparam = [500, 3000, 30000]

# ==============================
# Open HDF5 File 
# ==============================

#Look for the hdf5 file in the directory
hdf5_name = glob.glob(os.path.join(dir_name,'**.h5'))[0]

#Open the hdf5 file
with tables.open_file(hdf5_name, 'r') as hf5:
    unit_descriptor = hf5.get_node('/','unit_descriptor')[:]
    units = [(os.path.basename(x.__str__().split(" ")[0]),
            x.times[:], x.waveforms[:])\
            for x in hf5.root.sorted_units.__iter__()]
    

Raw_Electrodefiles = np.sort(glob.glob(os.path.join(dir_name,'*amp*dat*')))

unit_num = 3
this_unit = units[unit_num] 
this_mean_waveform_upsampled = np.mean(this_unit[-1],axis=0)
this_mean_waveform = np.mean(
        this_mean_waveform_upsampled.reshape((-1,10)),axis=1)
file_name = Raw_Electrodefiles\
        [unit_descriptor['electrode_number'][unit_num]]
data = np.fromfile(os.path.join(dir_name, file_name), 
            dtype = np.dtype('int16'))

filtered_data = get_filtered_electrode(data = data,
                        low_pass = freqparam[0],
                        high_pass = freqparam[1],
                        sampling_rate = freqparam[-1])

spike_num = 100
test_data = filtered_data[:this_unit[1][spike_num]] 

#plt.plot(test_data)
#plt.vlines(this_unit[1][:spike_num],min(test_data),max(test_data))
#plt.show()

def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(\
            a, shape=(nrows,L), strides=(S*n,n))

#test_conv = np.convolve(test_data,np.flip(this_mean_waveform), mode='valid')
#test_conv = np.convolve(test_data,this_mean_waveform, mode='valid')

# Zero-pad data to have the convolution have same length as origin
rolling_data = strided_app(test_data,len(this_mean_waveform),1) 
test_conv = np.matmul(rolling_data,this_mean_waveform) 

# Use parallel to calculate magnitude in chunks
rolling_data_split = np.array_split(rolling_data,100)

rolling_mag = np.concatenate([np.sqrt(np.sum(x**2,axis=-1)) for x in tqdm(rolling_data_split)])

#rolling_data_mag = Parallel(n_jobs = 10)\
#        (delayed(np.sum(x**2,axis=-1)) for x in tqdm(rolling_data_split))


#rolling_mag = np.linalg.norm(rolling_data,axis=-1)
waveform_mag = np.linalg.norm(this_mean_waveform)

normalized_conv = test_conv/(rolling_mag*waveform_mag)

#plt.plot(normalized_conv,'-x')
#plt.vlines(this_unit[1][:spike_num] - before_inds,
#         0, 1.5*max(normalized_conv))
#plt.show()

# As a reference, find the values of the normed
# convolution of the detected spikes with the reference
down_waveforms = np.mean(np.reshape(this_unit[-1],
            (this_unit[-1].shape[0],-1,10)),axis=-1)
ref_convs = np.matmul(down_waveforms,this_mean_waveform[:,np.newaxis])
ref_norms = np.linalg.norm(down_waveforms,axis=-1)
normalized_ref_convs = ref_convs.flatten() / \
        (ref_norms.flatten() * waveform_mag)

plt.hist(normalized_conv,100, density = True)
plt.hist(normalized_ref_convs,100, density = True)
plt.show()

# Use 5th percentile normalized_ref_convs as threshold
percentile_thresh = 5
conv_thresh = np.percentile(normalized_ref_convs,percentile_thresh)

def img_plot(array):
    plt.imshow(array, interpolation='nearest',aspect='auto', cmap='jet')

# Extract all windows above threshold
thresh_windows = normalized_conv > conv_thresh 
thresh_times = np.where(thresh_windows)[0]

# Find which spikes are new 
from scipy.spatial import distance_matrix as distmat
spike_dists = distmat((thresh_times+before_inds)[:,np.newaxis] , 
        this_unit[1][:spike_num,np.newaxis])
# Distance of <0.5ms
dist_thresh = 0.1
new_spikes = np.where(np.sum(spike_dists < dist_thresh*30,axis=-1)<1)[0]
old_spikes = np.where(np.sum(spike_dists < dist_thresh*30,axis=-1)>=1)[0]

# Plot waveforms
fig, ax  = plt.subplots(1,2, sharey=True, sharex=True)
ax[0].plot(rolling_data[thresh_windows][old_spikes].T,c='r', alpha = 0.2);
ax[0].plot(this_mean_waveform, c = 'k', linewidth = 5)
ax[0].set_title('Old detected {}'.format(len(old_spikes)))
ax[1].plot(rolling_data[thresh_windows][new_spikes].T,c='blue', alpha = 0.2);
ax[1].plot(this_mean_waveform, c = 'k', linewidth = 5)
ax[1].set_title('New detected {}'.format(len(new_spikes)))
plt.suptitle('Original spike count {}'.format(len(this_unit[1])))
plt.show()


# Plot data with old and new spiketimes overlayed
plt.plot(test_data)
plt.vlines(this_unit[1][:spike_num],min(test_data),0)
plt.vlines(thresh_times+before_inds,0,1.5*max(test_data),colors='r')
plt.vlines(thresh_times[new_spikes]+before_inds,0,1.5*max(test_data),colors='b')
plt.show()
