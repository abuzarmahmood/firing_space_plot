"""
Generate waveforms with different frequencies and test how quickly BSA changes
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
# rpy.common got deprecated in newer versions of pandas. So we use rpy2 instead
#import pandas.rpy.common as com
from rpy2.robjects import r

import scipy.ndimage

# Fire up BaSAR on R
basar = importr('BaSAR')

def gen_waveform(freq, fs):
    dur = 0.5/freq # We only need the first half of the wave
    t = np.linspace(0, dur, int(dur*fs))
    return t, np.sin(2*np.pi*freq*t), np.ones(len(t))*freq

def gen_waveform_string(freq_list, fs):
    t = np.array([])
    x = np.array([])
    f = np.array([])
    for freq in freq_list:
        t1, x1, f1 = gen_waveform(freq, fs)
        if len(t) == 0:
            t = t1
            x = x1
            f = f1
        else:
            t = np.concatenate((t, t[-1]+t1))
            x = np.concatenate((x, x1))
            f = np.concatenate((f, f1))
    return t, x, f

##############################
freq_lims = [1,10]
fs = 1000

# Generate waveform wtih given frequency

freq_list = np.random.uniform(freq_lims[0], freq_lims[1], 20)
t, x, f = gen_waveform_string(freq_list, fs)

# Gaussian smooth
# x = scipy.ndimage.gaussian_filter1d(x, 10)

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(t, x)
ax[1].plot(t, f, color = 'r', linestyle = '--')
plt.show()

##############################
# Run BSA on the generated data

# Make the time array and assign it to t on R
# T = (np.arange(7000) + 1)/1000.0
T = (np.arange(len(t)) + 1)/1000.0
t_r = ro.r.matrix(T, nrow = 1, ncol = len(t)) 
ro.r.assign('t_r', t_r)
ro.r('t = c(t_r)')

input_data = x.copy() 
# Check that trial is non-zero, if it isn't, don't try to run BSA

Br = ro.r.matrix(input_data, nrow = 1, ncol = len(t))
ro.r.assign('B', Br)
ro.r('x = c(B[1,])')

# x is the data, 
# we scan periods from 0.1s (10 Hz) to 1s (1 Hz) in 20 steps. 
# Window size is 300ms. 
# There are no background functions (=0)
ro.r('r_local = BaSAR.local(x, 0.1, 1, 20, t, 0, 300)') 
p_r = r['r_local']
# r_local is returned as a length 2 object, 
# with the first element being omega and the second being the 
# posterior probabilities. These need to be recast as floats
p = np.array(p_r[1]).astype('float')
omega = np.array(p_r[0]).astype('float')/(2.0*np.pi) 

fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(t, x)
ax[1].pcolormesh(t, omega, p.T)
ax[1].plot(t, f, color = 'r', linestyle = '--')
plt.show()
