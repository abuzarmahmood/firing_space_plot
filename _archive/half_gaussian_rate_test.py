os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
from visualize import *


data = ephys_data('/media/bigdata/Abuzar_Data/AS18/AS18_4Tastes_200229_154608') 
data.firing_rate_params = dict(zip(\
    ('type', 'step_size','window_size','dt', 'baks_resolution', 'baks_dt'),
    ('conv',25,250,1,1e-3,1e-3)))
data.extract_and_process()

spike_array = np.array(data.spikes)
firing_overview(data.all_normalized_firing);plt.show()

def half_gauss_kern(size):
    x = np.arange(-3*size,3*size+1)
    exponential = np.exp(-(x**2)/((2*float(size)**2)))
    coefficient = 1/(size * np.sqrt(np.pi *2))
    kern = (coefficient * exponential)[:len(x)//2]
    return kern / sum(kern)

plt.plot(half_gauss_kern(100));plt.show()

def half_gauss_filt(vector, size):
    kern = half_gauss_kern(size)
    return np.convolve(vector, kern, mode='same')

test_spikes = spike_array[0,0,0]
test_firing = half_gauss_filt(test_spikes,250)
plt.scatter(np.arange(len(test_spikes)), test_spikes)
plt.plot(test_firing / np.max(test_firing))
plt.show()

from scipy.signal import fftconvolve
kern_sd = 50
kern = half_gauss_kern(kern_sd)
firing_rate = fftconvolve(spike_array, np.reshape(kern,(1,1,1,-1)),
        axes = -1, mode='same')
firing_rate_long = np.reshape(np.moveaxis(firing_rate,2,0),
        (firing_rate.shape[2],-1,firing_rate.shape[-1]))
#firing_overview(firing_rate_long);plt.show()
imshow(firing_rate_long[18]);plt.show()
#plt.plot(firing_rate[0,0,0]);plt.show()
