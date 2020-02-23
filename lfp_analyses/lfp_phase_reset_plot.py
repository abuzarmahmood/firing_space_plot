
##################################################
# Generate phase reset plots for both regions
##################################################

# Signal projection using IFFT
# Get stft from file
stft_list = [hf5.get_node(os.path.join(this_path,'stft_array')) \
        for this_path in node_path_list] 
# Pull out min err channels
channel_list = [hf5.get_node(os.path.join(this_path,'relative_region_channel_nums'))[:] \
        for this_path in node_path_list] 
selected_stft_list = [stft[:,channels] for stft,channels in zip(stft_list,channel_list)]
# Pull out mean power in pre-delivery time
window_len = 0.5 # sec
padding = 0.25 # sec
stim_delivery_t = 2 # sec
pre_stim_stft = [ stft[...,(time_vec > (stim_delivery_t - window_len - padding)) \
        * (time_vec < (stim_delivery_t - padding))] for stft in selected_stft_list]
pre_stim_power = [np.abs(x) for x in pre_stim_stft]

########################################
# Reconstructing test signal
Fs = 1000 
signal_window = 500 
window_overlap = 499
max_freq = 25
time_range_tuple = (1,5)

session_num = 0
trial_ind = (0,0,0)
test_trial_stft = selected_stft_list[session_num][trial_ind]

# Temp stft to find size of origin stft
# Add to setup to save dimensions
f,t,temp_stft = scipy.signal.stft(
            scipy.signal.detrend(actual_test_lfp), 
            fs=Fs, 
            window='hanning', 
            nperseg=signal_window, 
            noverlap=signal_window-(signal_window-window_overlap)) 

# Define function to return ISTFT on single frequency band data

temp_trial_stft = np.zeros((temp_stft.shape[0],test_trial_stft.shape[-1]),
        dtype = np.dtype('complex'))
temp_trial_stft[:test_trial_stft.shape[0]] = test_trial_stft 
test_istft = scipy.signal.istft(
                temp_trial_stft, 
                fs=Fs, 
                window='hanning', 
                nperseg=signal_window, 
                noverlap=signal_window-(signal_window-window_overlap))

dat = ephys_data('/media/bigdata/Abuzar_Data/AM17/AM17_extracted/'\
        'AM17_4Tastes_191125_084206')
dat.get_lfps()
actual_selected_lfp = dat.lfp_array[:,channel_list[0]]
actual_test_lfp = actual_selected_lfp[trial_ind]

# Check overlay of ISTFT with original LFP
plt.plot(time_vec[1:],test_istft[1],linewidth = 5);
plt.plot(np.arange(len(actual_test_lfp))/Fs,actual_test_lfp);plt.show()

test_pre = pre_stim_power[session_num][trial_ind]

########################################
#test_array = final_phases_long[0] 
## Reshape so mean period can be calculated
#tmp_array = test_array.swapaxes(1,2)
#test_long = np.reshape(tmp_array,(tmp_array.shape[0],tmp_array.shape[1],-1))
#period_arrays = np.where(np.abs(np.diff(test_long,axis=-1)) > 6)
#freq_period_arrays = [period_arrays[2][period_arrays[1] == freq]\
#        for freq in np.sort(np.unique(period_arrays[1]))]
#freq_periods = [np.diff(x) for x in freq_period_arrays]
#freq_periods_mean = [np.mean(x) for x in freq_periods]
#freq_periods_std = [np.std(x) for x in freq_periods]
#plt.errorbar(np.arange(len(freq_periods)),freq_periods_mean, freq_periods_std)
#plt.show()
#plt.plot(freq_periods_mean,'-x');plt.show()

# Unroll phase for each band
# Perform linear regression on pre-stimulus phase
# Check deviation at stimulus delivery

session = 0
trial_ind = (0,0,0)
band_num = 1
test_trial = final_phases[session_num][trial_ind]
this_band = test_trial[band_num]

from sklearn.linear_model import LinearRegression as LR
reg = LR().fit(time_vec.reshape(-1,1),np.unwrap(this_band).reshape(-1,1))

plt.plot(time_vec, np.unwrap(this_band))
plt.plot(time_vec, np.squeeze(reg.predict(time_vec.reshape(-1,1))))
plt.show()
