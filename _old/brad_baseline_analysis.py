import os
import numpy as np
import seaborn as sns
import pandas as pd
import pylab as plt
from scipy.signal import savgol_filter as sgf
os.chdir('/media/bigdata/firing_space_plot/')
from ephys_data import ephys_data
from baseline_divergence_funcs import firing_rates

file = 4
data_dir = '/media/bigdata/brads_data/file%i/' % file
data = ephys_data(data_dir,1)
data.get_data()
spikes = data.spikes
spikes = spikes[0][:,0,0:20000]
spikes = spikes[:,np.newaxis,:]

firing, normal_firing = firing_rates(spikes, 25, 250)

fig, (ax0,ax1) = plt.subplots(nrows=2,ncols=1)
ax0.plot(np.mean(normal_firing[:,0,:],axis=0))
filt_normal = sgf(np.mean(normal_firing[:,0,:],axis=0),51,2)
ax0.plot(filt_normal)
plt.xlabel('20 seconds')
ax0.legend(['Mean Normalized Firing', 'SG Filtered Mean Normal'])
#plt.figure()
ax1.imshow(normal_firing[:,0,:],interpolation='hermite',aspect='auto')
plt.suptitle('Brad File %i' % file)
plt.savefig('/media/bigdata/firing_space_plot/plots/baseline_activity/brad_file%i_small' % file)
plt.close(fig)

time_series = pd.DataFrame(data = normal_firing[:,0,:].T,index = range(normal_firing.shape[-1]))
time_val = pd.DataFrame(data={'time' : range(normal_firing.shape[-1])})
time_series = pd.concat([time_series, time_val],axis=1)

time_series = pd.melt(frame = time_series, 
                      id_vars = 'time',
                      var_name = 'nrn', 
                      value_name = 'firing')

#time_series['time'] = time_series.groupby('nrn').size().reset_index(name='counts')
#mean_time_series = time_series.groupby(by = 'time').mean()
sns.regplot(x = 'time', y= 'firing', data = time_series, x_estimator=np.mean)