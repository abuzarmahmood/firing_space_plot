#Better way to do all the try/except?
# Store sigmoid parameters and derivative result in file

######################### Import dat ish #########################
import numpy as np
import tables
import easygui
import os
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import kruskal
import shutil

plot_dir = easygui.diropenbox(msg = 'Choose directory to store the plots in')

try: 
    shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)
except:
    print('Could not delete plot folder!')  

# Ask the user for the hdf5 files that need to be plotted together
if easygui.ynbox('Load directories from file?', 'Title', ('Yes', 'No')):
    params_file = 'all_correlation_params'
    correlation_dir = easygui.diropenbox(msg = 'Choose a directory with correlation_params')
    f = open(correlation_dir + '/' + params_file, 'r')
    dirs = []
    for line in f.readlines():
        dirs.append(line.strip())
    f.close()
else:
    dirs = []
    while True:
        dir_name = easygui.diropenbox(msg = 'Choose a directory with a hdf5 file, hit cancel to stop choosing')
        try:
            if len(dir_name) > 0:    
                dirs.append(dir_name)
        except:
            break
        
    if easygui.ynbox('Save directories to folder?', 'Title', ('Yes', 'No')):
        all_dirs_dir = easygui.diropenbox(msg = 'Choose a directory to store directory list in')
        if os.path.isfile(all_dirs_dir + '/' + 'all_correlation_params'):
            if easygui.ynbox('Overwrite old parameters file?', 'Title', ('Yes', 'No')):
                f = open(plot_dir + '/' + 'all_correlation_params', 'w')
                for dir in dirs:
                    	print(dir, file=f)
                f.close()
        else:
                f = open(plot_dir + '/' + 'all_correlation_params', 'w')
                for dir in dirs:
                    	print(dir, file=f)
                f.close()

unique_states = []
for dir_name in dirs:
        
    # Change to the directory
    os.chdir(dir_name)
    # Locate the hdf5 file
    file_list = os.listdir('./')
    hdf5_name = ''
    for files in file_list:
        if files[-2:] == 'h5':
            hdf5_name = files

    # Open the hdf5 file
    hf5 = tables.open_file(hdf5_name, 'r') 
    states = hf5.list_nodes('/spike_trains/multinomial_hmm_results/laser_off')
    states = re.findall('\d',states.__str__())
    for i in range(len(states)):
        states[i] = int(states[i])
    unique_states.append(states)

hf5.close()
unique_states = np.unique(unique_states)

#Specify params or load from file
if easygui.ynbox('Load params from lining up file?', 'Title', ('Yes', 'No')):
    params_folder = easygui.diropenbox('Select parameters folder for lining up')
    params_file = 'correlation_lining_params'
    f = open(params_folder+'/'+params_file, 'r')
    params = []
    for line in f.readlines():
        params.append(line)
    f.close()
    pre_stim_t = int(params[7]) #2000   #time before stimulus
    data_window_pre = int(params[8]) #500   #window before transition time
    data_window_post = int(params[9]) #750  #window after transition time
    bin_window_size = int(params[10]) #250
    step_size = int(params[11]) #25
    
    
else:   #Request for parameters and save to file in correlation plot folder
    
    #Params for making PSTH's
    psth_params = easygui.multenterbox('Parameters for making PSTHs',fields = ['Pre-stimulus time (2000)','Time wanted before transition (500)','Time watned after transition (750)', 'Binning window size (250)', 'Window step size (25)'])
    pre_stim_t = int(psth_params[0]) #2000   #time before stimulus
    data_window_pre = int(psth_params[1]) #500   #window before transition time
    data_window_post = int(psth_params[2]) #750  #window after transition time
    bin_window_size = int(psth_params[3]) #250
    step_size = int(psth_params[4]) #25
 

# Make indexing vector to pull out correaltion from ancillary analysis array
params = easygui.multenterbox(fields = ['Unaligned correlation centering time (950)','Time length (7000)'])
middle_time = int(params[0]) #(700+1200)/2
t_length = int(params[1]) #7000

flank_window = int((data_window_pre + data_window_post - bin_window_size)/2)
time_vec = np.arange(0,t_length-bin_window_size+step_size,step_size)
index_vec = np.where((time_vec>=pre_stim_t+middle_time-flank_window)*(time_vec<=pre_stim_t+middle_time+flank_window))[0]

# Now run through the directories, and pull out the data
# Going to loop through each file and each model of the HMM (different number 
# of states) with on and off states separately


for num_state in unique_states:
    
    r_pearson_off = []
    r_pearson_off_scram = []
    del_transition_off = []
    #p_pearson_off = []
    
    r_pearson_on = []
    r_pearson_on_scram = []
    del_transition_on = []
    #p_pearson_on = []
    
    num_units_on = 0
    num_units_off = 0
    
    for dir_name in dirs:
        # Change to the directory
        os.chdir(dir_name)
        # Locate the hdf5 file
        file_list = os.listdir('./')
        hdf5_name = ''
        for files in file_list:
            if files[-2:] == 'h5':
                hdf5_name = files
    
        # Open the hdf5 file
        hf5 = tables.open_file(hdf5_name, 'r')
    
        # Pull the data from the /ancillary_analysis node
        try:
            exec('r_pearson_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.r_pearson[:])' % num_state)
        except:
            pass
        #exec('p_pearson_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.p_pearson[:])' % num_state)
        try:
            exec('r_pearson_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.r_pearson[:])' % num_state)
        except:
            pass
        #exec('p_pearson_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.p_pearson[:])' % num_state)
        try:
            exec('del_transition_off.append(hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.palatability_state_transition_t[:])' % num_state)
        except:
            pass
        try:
            exec('del_transition_on.append(hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.palatability_state_transition_t[:])' % num_state)
        except:
            pass
        
        # Append r_pearson from the ancillary analysis of all files
        r_pearson_off_scram.append(np.ndarray.transpose(hf5.root.ancillary_analysis.r_pearson[0,:]))
        r_pearson_on_scram.append(np.ndarray.transpose(hf5.root.ancillary_analysis.r_pearson[1,:]))
        
        ## Also maintain a counter of the number of units in the analysis
        #exec('num_units_on += hf5.root.spike_trains.multinomial_hmm_results.laser_on.states_%i.p_pearson.shape[0]' % num_state)
        #exec('num_units_off += hf5.root.spike_trains.multinomial_hmm_results.laser_off.states_%i.p_pearson.shape[0]' % num_state)
        
        # Close the hdf5 file
        hf5.close()
            
    ## Error catching and cleaning - should be better way to do this
    # Delete any rows which have zero correlation (because they will be averaged later)
    # Delete any arrays which then become empty
    rm_list_off = []
    for i in range(len(r_pearson_off)):
        rm_row_off = []
        for j in range(r_pearson_off[i].shape[0]):
            if np.mean(r_pearson_off[i][j,:]) == 0:
                rm_row_off.append(j)
        r_pearson_off[i] = np.delete(r_pearson_off[i],rm_row_off,0)
        if (np.sum(r_pearson_off[i])==0):
            rm_list_off.append(i)
    
    r_pearson_off = [r_pearson_off[e] for e in range(len(r_pearson_off)) if e not in rm_list_off]

    rm_list_on = []
    for i in range(len(r_pearson_on)):
        rm_row_on = []
        for j in range(r_pearson_on[i].shape[0]):
            if np.mean(r_pearson_on[i][j,:]) == 0:
                rm_row_on.append(j)
        r_pearson_on[i] = np.delete(r_pearson_on[i],rm_row_on,0)
        if (np.sum(r_pearson_on[i])==0):
            r_pearson_on.append(i)      
            
    r_pearson_on = [r_pearson_on[e] for e in range(len(r_pearson_on)) if e not in rm_list_on]

        
    if len(unique_states)==1:
        r_pearson_off = r_pearson_off[0]
        #p_pearson_off = p_pearson_off[0]
        r_pearson_on = r_pearson_on[0]
        #p_pearson_on = p_pearson_on[0]
        r_pearson_off_scram = r_pearson_off_scram[0]
        r_pearson_on_scram = r_pearson_on_scram[0]
        del_transition_off = del_transition_off[0]
        del_transition_on = del_transition_on[0]
        
    else:
        r_pearson_off = np.concatenate(r_pearson_off,0)
        #p_pearson_off = np.concatenate(p_pearson_off,0)
        r_pearson_on = np.concatenate(r_pearson_on,0)
        #p_pearson_on = np.concatenate(p_pearson_on,0)
        r_pearson_off_scram = np.concatenate(r_pearson_off_scram,0)
        r_pearson_on_scram = np.concatenate(r_pearson_on_scram,0)
        del_transition_off = np.concatenate(del_transition_off,0)
        del_transition_on = np.concatenate(del_transition_on,0)
        
    #Index the unalgined correaltions by the index_vec
    r_pearson_off_scram = r_pearson_off_scram[:,index_vec]
    r_pearson_on_scram = r_pearson_on_scram[:,index_vec]
    
######################### Sigmoid fitting #########################
    # Use sigmoid from Sadacca paper
    #def sigmoid(x, a, b, x_0, d):
    #    return (a/(1 + np.exp(-b*(x-x_0)))) + d
    
    def sigmoid(x, a, b, x_0, d): #Sadacca sigmoid
        return ((a/b)/(1 + np.exp(-b*(x-x_0)))) + d
    
    #Returns x value at index corresponding to closest value in y
    # Was needed to provide initial estimate for the fit...otherwise it wasn't working.
    def take_closest(val,y,x):
        est_median = np.min(y) + (np.max(y) - np.min(y))/2
        abs_dist = abs(y - est_median)
        closest_val_ind = np.where(abs_dist == np.min(abs_dist))[0]
        return closest_val_ind[0], x[closest_val_ind][0]
    
    # Generates fit and outputs parameters
    def custom_sig_fit(x_data,y_data, function):
        try:
            dat_range = np.max(y_data)-np.min(y_data)
            mid_ind,mid_time = take_closest(dat_range/2,y_data,x_data)
            mid_grad = np.mean(np.gradient(y_data[np.where((x_data>(mid_time-100))*(x_data<(mid_time+100)))]))
            param_est = [4*mid_grad,2*mid_grad/y_data[mid_ind],mid_time,np.min(y_data)]
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0 = param_est)
            return popt, pcov
        except:
            return [0,0,0,0],0
    
    x_data = np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size)
    
    popt_on, pcov_on = custom_sig_fit(x_data, np.mean(r_pearson_on[:, :]**2, axis = 0), sigmoid)
    popt_off, pcov_off = custom_sig_fit(x_data, np.mean(r_pearson_off[:, :]**2, axis = 0), sigmoid)
    popt_on_scram, pcov_on_scram = custom_sig_fit(x_data, np.mean(r_pearson_on_scram[:, :]**2, axis = 0), sigmoid)
    popt_off_scram, pcov_off_scram = custom_sig_fit(x_data, np.mean(r_pearson_off_scram[:, :]**2, axis = 0), sigmoid)
    
    on_smooth = gaussian_filter1d(np.mean(r_pearson_on[:, :]**2, axis = 0),2)
    off_smooth = gaussian_filter1d(np.mean(r_pearson_off[:, :]**2, axis = 0),2)
    on_scram_smooth = gaussian_filter1d(np.mean(r_pearson_on_scram[:, :]**2, axis = 0),2)
    off_scram_smooth = gaussian_filter1d(np.mean(r_pearson_off_scram[:, :]**2, axis = 0),2)

######################### PLOTS!!! #########################    

    # Instead of adding mean +/- sd to the image, make a separate image with histogram
    #ax = fig.add_subplot(111)
    #ax.text(-750,pearson_min+(pearson_min_max*0.9),'Mean state transition \n %i +\- %i msec' % (np.mean(del_transition_off),np.std(del_transition_off)))
    
    fig2 = plt.figure()
    plt.hist(del_transition_off, 30, alpha=0.5, label='Off_trials')
    plt.hist(del_transition_on, 30, alpha=0.5, label='On_trials')
    plt.legend(loc='upper right')
    plt.title('State = '+str(num_state)+', p_val = %0.2f' % kruskal(del_transition_off,del_transition_on)[1])
    plt.xlabel('Interstate transition times')
    fig2.savefig(plot_dir + '/' + 'transitions_states%i.png' % num_state)
    
## Correlation plots and sigmoid fits ##
    
    all_pearson = np.concatenate((np.mean(r_pearson_off[:, :]**2, axis = 0), np.mean(r_pearson_off_scram[:, :]**2, axis = 0) \
                    ,np.mean(r_pearson_on[:, :]**2, axis = 0), np.mean(r_pearson_on_scram[:, :]**2, axis = 0)))
    pearson_max = np.max(all_pearson)
    pearson_min = np.min(all_pearson)
    pearson_min_max = pearson_max - pearson_min  
    
    fig = plt.figure(figsize=(16,12))
    ax0 = fig.add_subplot(2,2,1)
    plt.title('Correlation, Aligned, Off trials , Model states =' + str(num_state))
    #Plot aligned correlation
    ax0.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),
                 np.mean(r_pearson_off[:, :]**2, axis = 0),
                 yerr=np.std(r_pearson_off[:,:]**2,axis = 0)/np.sqrt(r_pearson_off.shape[0]), 
                 label = 'Aligned')
    try:
        ax0.plot(x_data, sigmoid(x_data, *popt_off), 'r-')
        ax0.text(-750,pearson_min+(pearson_min_max*0.9),'A = %f\nB = %f' % (popt_off[0],popt_off[1]))
    except:
            pass
    plt.ylim(pearson_min, pearson_max)
    plt.xlabel('State transition time')
    plt.ylabel('Pearson Correlation Coefficient')
    
    ax1 = fig.add_subplot(2,2,2)
    plt.title('Correlation, Unaligned, Off trials , Model states =' + str(num_state))
    #Plot unaligned correlation
    ax1.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),
                 np.mean(r_pearson_off_scram[:, :]**2, axis = 0),
                 yerr=np.std(r_pearson_off_scram[:, :]**2,axis = 0)/np.sqrt(r_pearson_off_scram.shape[0]), 
                 label = 'Unaligned')
    try:
        plt.plot(x_data, sigmoid(x_data, *popt_off_scram), 'r-')
        ax1.text(-750,pearson_min+(pearson_min_max*0.9),'A = %f\nB = %f' % (popt_off_scram[0],popt_off_scram[1]))
    except:
        pass
    plt.ylim(pearson_min, pearson_max)
    plt.xlabel('State transition time')
    plt.ylabel('Pearson Correlation Coefficient')
    
    
    ax2 = fig.add_subplot(2,2,3)
    plt.title('Correlation, Aligned, On trials , Model states =' + str(num_state))
    #Plot aligned correlation
    ax2.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),
                 np.mean(r_pearson_on[:, :]**2, axis = 0),
                 yerr=np.std(r_pearson_on[:,:]**2,axis = 0)/np.sqrt(r_pearson_on.shape[0]), 
                 label = 'Aligned')
    plt.ylim(pearson_min, pearson_max)
    plt.xlabel('State transition time')
    plt.ylabel('Pearson Correlation Coefficient')
    try:
        plt.plot(x_data, sigmoid(x_data, *popt_on), 'r-')
        ax2.text(-750,pearson_min+(pearson_min_max*0.9),'A = %f\nB = %f' % (popt_on[0],popt_on[1]))
    except:
        pass
    
    ax3 = fig.add_subplot(2,2,4)
    plt.title('Correlation, Unaligned, On trials , Model states =' + str(num_state))
    #Plot unaligned correlation
    ax3.errorbar(np.arange(-data_window_pre,data_window_post-bin_window_size+step_size,step_size),
                 np.mean(r_pearson_on_scram[:, :]**2, axis = 0),
                 yerr=np.std(r_pearson_on_scram[:, :]**2,axis = 0)/np.sqrt(r_pearson_on_scram.shape[0]), 
                 label = 'Unaligned')
    plt.ylim(pearson_min, pearson_max)
    plt.xlabel('State transition time')
    plt.ylabel('Pearson Correlation Coefficient')
    try:
        plt.plot(x_data, sigmoid(x_data, *popt_on_scram), 'r-')
        ax3.text(-750,pearson_min+(pearson_min_max*0.9),'A = %f\nB = %f' % (popt_on_scram[0],popt_on_scram[1]))
    except:
        pass
    
    plt.subplots_adjust(hspace = 0.3)
    #plt.tight_layout()
    fig.savefig(plot_dir + '/' + 'correlations_state%i.png' % num_state)
    plt.close('all')
       

