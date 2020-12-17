# ___                            _   
#|_ _|_ __ ___  _ __   ___  _ __| |_ 
# | || '_ ` _ \| '_ \ / _ \| '__| __|
# | || | | | | | |_) | (_) | |  | |_ 
#|___|_| |_| |_| .__/ \___/|_|   \__|
#              |_|                   



import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import tables

from scipy.signal import hilbert, butter, filtfilt, spectrogram 

from sklearn.decomposition import PCA as pca
from scipy.stats import zscore
import itertools

dat_imshow = lambda x : plt.imshow(x,interpolation='nearest',aspect='auto')

#   _____      _     _____        _        
#  / ____|    | |   |  __ \      | |       
# | |  __  ___| |_  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ __| | |  | |/ _` | __/ _` |
# | |__| |  __/ |_  | |__| | (_| | || (_| |
#  \_____|\___|\__| |_____/ \__,_|\__\__,_|
#

dir_list = ['/media/bigdata/brads_data']
file_list = []
for x in dir_list:
    file_list = file_list + glob.glob(x + '/**/' + '*Taste*.h5',recursive=True)
    
# ____  _       _                          _     _____ ____  
#|  _ \| | ___ | |_   _ __ __ ___      __ | |   |  ___|  _ \ 
#| |_) | |/ _ \| __| | '__/ _` \ \ /\ / / | |   | |_  | |_) |
#|  __/| | (_) | |_  | | | (_| |\ V  V /  | |___|  _| |  __/ 
#|_|   |_|\___/ \__| |_|  \__,_| \_/\_/   |_____|_|   |_|    
#                                                            

fix, ax = plt.subplots(1,len(file_list))

for file_num,file_name in enumerate(file_list):
    hf5 = tables.open_file(file_name)
    try:
        parsed_LFP = np.mean(\
                np.array([x[:] for x in hf5.root.Parsed_LFP])[:,:,:,2000:2500],
                axis = 1)

        hf5.close()

        parsed_LFP_long = parsed_LFP.reshape(
                (parsed_LFP.shape[0]*parsed_LFP.shape[1],parsed_LFP.shape[2]))

        plt.sca(ax[file_num])
        dat_imshow(parsed_LFP_long)
    except:
       print('File {} could not be processed'.format(file_num)) 
plt.show()


# _____           _   _                  __ _ _ _            _             
#|  ___|   _ _ __| |_| |__   ___ _ __   / _(_) | |_ ___ _ __(_)_ __   __ _ 
#| |_ | | | | '__| __| '_ \ / _ \ '__| | |_| | | __/ _ \ '__| | '_ \ / _` |
#|  _|| |_| | |  | |_| | | |  __/ |    |  _| | | ||  __/ |  | | | | | (_| |
#|_|   \__,_|_|   \__|_| |_|\___|_|    |_| |_|_|\__\___|_|  |_|_| |_|\__, |
#                                                                    |___/ 

for file_num in 
    hf5 = tables.open_file(file_list[file_num])
    parsed_LFP = np.mean(\
            np.array([x[:] for x in hf5.root.Parsed_LFP]),
            axis = 1)
    hf5.close()
    parsed_LFP_long = parsed_LFP.reshape(
        (parsed_LFP.shape[0]*parsed_LFP.shape[1],parsed_LFP.shape[2]))


    #define bandpass filter parameters to parse out frequencies
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut /nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandpass')
        y = filtfilt(b, a, data)
        return y

    lowcut = 7
    highcut = 12

    filtered_signal_long = butter_bandpass_filter(
            data = parsed_LFP_long,
            lowcut = lowcut,
            highcut = highcut,
            fs = 1000)

    # ____                  _                                       
    #/ ___| _ __   ___  ___| |_ _ __ ___   __ _ _ __ __ _ _ __ ___  
    #\___ \| '_ \ / _ \/ __| __| '__/ _ \ / _` | '__/ _` | '_ ` _ \ 
    # ___) | |_) |  __/ (__| |_| | | (_) | (_| | | | (_| | | | | | |
    #|____/| .__/ \___|\___|\__|_|  \___/ \__, |_|  \__,_|_| |_| |_|
    #      |_|                            |___/                     

    fs = 1000
    f, t, Sxx = spectrogram(
            parsed_LFP,
            fs = fs,
            window = 'hanning',
            nperseg = 1000,
            noverlap = 900,
            mode = 'psd')

    mean_spectrogram = np.mean(Sxx,axis=1)

    low_ind = np.argmin(np.abs(f - lowcut))
    high_ind = np.argmin(np.abs(f - highcut))

    band_f, band_spectrogram = f[low_ind:high_ind], mean_spectrogram[:,low_ind:high_ind,:]

    mean_band_power = np.mean(band_spectrogram,axis=1)
    plt.plot(t,mean_band_power.T);plt.show()


plt.pcolormesh(t,band_f,band_spectrogram[0,:,:]);plt.show()
plt.pcolormesh(t,f,np.mean(mean_spectrogram,axis=0));plt.show()

#    _                _           _     
#   / \   _ __   __ _| |_   _ ___(_)___ 
#  / _ \ | '_ \ / _` | | | | / __| / __|
# / ___ \| | | | (_| | | |_| \__ \ \__ \
#/_/   \_\_| |_|\__,_|_|\__, |___/_|___/
#                       |___/           


zscore_long = zscore(filtered_signal_long, axis=None)
fig, axs = plt.subplots(1,3)
plt.sca(axs[0])
dat_imshow(filtered_signal_long[:,:500])
plt.sca(axs[1])
dat_imshow(zscore_long[:,:500])
plt.sca(axs[2])
dat_imshow(parsed_LFP_long[:,:500])
plt.show()

pca_dat = pca(n_components = 3).fit_transform(filtered_long)
#plt.imshow(pca_dat,interpolation='nearest',aspect='auto');plt.show()
trial_labels = np.sort(list(range(4))*30)

dim_pairs = list(itertools.combinations(range(pca_dat.shape[1]),2))
fig,axs = plt.subplots(1,3)
for ax,dim_pair in zip(axs,dim_pairs):
    ax.scatter(pca_dat[:,dim_pair[0]],pca_dat[:,dim_pair[1]],c=trial_labels)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_dat[:,0],pca_dat[:,1],pca_dat[:,2],c=trial_labels)
plt.show()
