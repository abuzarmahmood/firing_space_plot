"""
Code to analyse poking bouts
"""

# Import stuff!
import tables
import os
import numpy as np
import tqdm
import glob
import pandas as pd
import numpy as np
import pylab as plt

############################################################
#|  _ \  ___ / _(_)_ __   ___  |  ___|   _ _ __   ___ ___ 
#| | | |/ _ \ |_| | '_ \ / _ \ | |_ | | | | '_ \ / __/ __|
#| |_| |  __/  _| | | | |  __/ |  _|| |_| | | | | (__\__ \
#|____/ \___|_| |_|_| |_|\___| |_|   \__,_|_| |_|\___|___/
############################################################

########################################
## Load data 
########################################
def return_sampling_rate(dir_name):
    # Extract sampling frequency from info.rhd
    info_rhd_read = np.fromfile(os.path.join(dir_name, 'info.rhd'), 
                        dtype = np.dtype('float32'))
    sampling_rate = int(info_rhd_read[2])
    return sampling_rate

def return_digin_array(dir_name, fin_sampling_rate):
    """
    Read dig-ins and place into array
    DIN-00 -> IOC Delivery
    DIN-04 -> Laser On
    DIN-07 -> Nosepoke
    """

    file_list = np.sort(glob.glob(os.path.join(dir_name,"*DIN*")) )
    digin_array = np.array([np.fromfile(x, dtype = np.dtype('uint16')) for x in file_list])
    # Downsample to 100Hz (there is no way the rat can react faster than that)
    skip_samples = sampling_rate//fin_sampling_rate
    digin_array = digin_array[:,::int(skip_samples)]
    # Convert to int (otherwise bad things happen when you do diff)
    digin_array = np.vectorize(np.int)(digin_array)

    # Original nosepoke output: 0->Beam Broken, 1->Beam intact
    # Flip to have 1 mean poking
    digin_array[-1] = 1 - digin_array[-1]
    return digin_array

########################################
# Extract session statistics:
########################################
def extract_poke_stats(digin_array, fin_sampling_rate):
    """
    # Per (Arieli, Moran 2020):
    #   Bout = 5+ pokes (without widthrawal?) in 10 secs 
    # Total pokes
    # Successful pokes
    # Inter-poke intervals
    # In-poke duration distribution
    """
    successful_pokes = np.sum(np.diff(digin_array[0]) == 1)
    total_pokes = np.sum(np.diff(digin_array[-1]) == 1)
    total_poke_time = np.sum(digin_array[-1])/fin_sampling_rate

    # Find periods of poking
    poke_up = np.where(np.diff(digin_array[-1]) == 1)[0]
    poke_down = np.where(np.diff(digin_array[-1]) == -1)[0]
    # Pair together poke ups and downs
    poke_bounds = list(zip(poke_up,poke_down))
    poke_durations = np.diff(poke_bounds).flatten()

    # Calculate inter-poke intervals
    # Will likely have to plot log(t) for comparison
    interpoke_intervals = np.diff(list(zip(poke_down[:-1],poke_up[1:]))).flatten()

    # For every 10s period, mark burst as:
    # 1) Either 5+ pokes in 10s
    # 2) 90% of the 10s window was spent poking 
    window_kern = np.ones(10*100)
    only_poke_entry = np.diff(digin_array[-1])
    only_poke_entry[only_poke_entry<0] = 0
    poke_condition = np.convolve(only_poke_entry,window_kern,'valid') >= 5
    continuous_condition = \
            (np.convolve(digin_array[-1][:-1], window_kern,'valid')/len(window_kern)) >= 0.9 
    bout_bool = (poke_condition + continuous_condition) > 0

    total_bout_time = np.sum(bout_bool)/fin_sampling_rate

    return total_pokes, successful_pokes, total_poke_time, total_bout_time

############################################################
#|  _ \ _   _ _ __     __ _ _ __   __ _| |_   _ ___(_)___ 
#| |_) | | | | '_ \   / _` | '_ \ / _` | | | | / __| / __|
#|  _ <| |_| | | | | | (_| | | | | (_| | | |_| \__ \ \__ \
#|_| \_\\__,_|_| |_|  \__,_|_| |_|\__,_|_|\__, |___/_|___/
#                                         |___/           
############################################################
########################################
## Code to read file list
########################################
file_csv_path = '/media/bigdata/Tom_Data/TJM_file_list.csv'
file_frame = pd.read_csv(file_csv_path)

# Find files
#dir_name = '/media/bigdata/Tom_Data/TJM2/TJM2_H2O1_200828_173613'

data_cols =[ "animal_name", "expt_day",  "total_pokes", "successful_pokes",
        "total_poke_time","total_bout_time"]
data_df = pd.DataFrame(columns = data_cols)

########################################
## Load data 
########################################
for file_ind,path in enumerate(file_frame.path):
    dir_name = file_frame.path[file_ind]

    fin_sampling_rate = 100
    sampling_rate = return_sampling_rate(dir_name)
    digin_array = return_digin_array(dir_name, fin_sampling_rate)

    ########################################
    ## Extract statistics
    ########################################
    total_pokes,successful_pokes,total_poke_time,total_bout_time = \
            extract_poke_stats (digin_array, fin_sampling_rate)
    data_dict = {"animal_name":file_frame.animal_name[file_ind],
                "expt_day" : file_frame.expt_day[file_ind],
                "total_pokes" : total_pokes,
                "successful_pokes" : successful_pokes,
                "total_poke_time" : total_poke_time,
                "total_bout_time" : total_bout_time}
    data_df = data_df.append(data_dict, ignore_index = True)


#cut_digin_array = digin_array[-1,len(window_kern):]
#time_vec = np.arange(len(cut_digin_array))/100
#fig, ax = plt.subplots(3,1, sharex=True)
#ax[0].plot(time_vec, cut_digin_array)
#ax[1].plot(time_vec,poke_condition)
#ax[1].plot(time_vec,continuous_condition)
#ax[2].plot(time_vec,bout_bool)
#plt.show()
