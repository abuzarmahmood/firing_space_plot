"""
Code to analyse poking bouts
"""

# Import stuff!
import tables
import os
import numpy as np
import tqdm
import glob

# Find files
dir_name = '/media/bigdata/Tom_Data/TJM2/TJM2_H2O1_200828_173613'
file_list = np.sort(glob.glob(os.path.join(dir_name,"*DIN*")) )

# Extract sampling frequency from info.rhd
info_rhd_read = np.fromfile(os.path.join(dir_name, 'info.rhd'), 
        dtype = np.dtype('float32'))
sampling_rate = int(info_rhd_read[2])

# Read dig-ins and place into array
# DIN-00 -> IOC Delivery
# DIN-04 -> Laser On
# DIN-07 -> Nosepoke

digin_array = np.array([np.fromfile(x, dtype = np.dtype('uint16')) for x in file_list])
# Downsample to 100Hz (there is no way the rat can react faster than that)
digin_array = digin_array[:,::int(sampling_rate//100)]
# Convert to int (otherwise bad things happen when you do diff)
digin_array = np.vectorize(np.int)(digin_array)

# Original nosepoke output: 0->Beam Broken, 1->Beam intact
# Flip to have 1 mean poking
digin_array[-1] = 1 - digin_array[-1]

# Extract session statistics:
# Per (Arieli, Moran 2020):
#   Bout = 5+ pokes (without widthrawal?) in 10 secs 
# Total pokes
# Successful pokes
# Inter-poke intervals
# In-poke duration distribution
successful_pokes = np.sum(np.diff(digin_array[0]) == 1)
total_pokes = np.sum(np.diff(digin_array[-1]) == 1)
total_poke_time = np.sum(digin_array[-1])
# Find periods of poking
poke_up = np.where(np.diff(digin_array[-1]) == 1)[0]
poke_down = np.where(np.diff(digin_array[-1]) == -1)[0]
# Pair together poke ups and downs
poke_bounds = list(zip(poke_up,poke_down))
poke_durations = np.diff(poke_bounds).flatten()

# Calculate inter-poke intervals
# Will likely have to plot log(t) for comparison
interpoke_intervals = np.diff(list(zip(poke_down[:-1],poke_up[1:]))).flatten()
