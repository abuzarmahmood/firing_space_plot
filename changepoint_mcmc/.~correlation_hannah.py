### This file runs correlation on changepoint fits of split datasets.
#WARNING: This code is specific to splitting data into 2 pieces. It needs to be 
#optimized for any other number of splits by updating the pickle file to contain
#the number of splits and using those to generate model paths iteratively.

#When calling this file, make sure to provide the path to the model for split0
#this will allow the code to find split1.


#####IMPORTS######
import numpy as np
import os
import sys
import pickle
import argparse
import re
import scipy.stats as stats

#####LOAD DATA#####

#parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
#parser.add_argument('model_path',  help = 'Path to model pkl file')
#args = parser.parse_args()
#model_path0 = args.model_path 
#model_path1 = model_path0.replace('split0','split1')

#Hardcoded for tests: REMOVE LATER
model_path0 = '/media/bigdata/Abuzar_Data/AS18/AS18_4Tastes_200228_151511/saved_models/vi_4_states/split_0_0_vi_4states_40000fit_2000_4000time_50bin.pkl'
model_path1 = '/media/bigdata/Abuzar_Data/AS18/AS18_4Tastes_200228_151511/saved_models/vi_4_states/split_1_0_vi_4states_40000fit_2000_4000time_50bin.pkl'


# Extract model params from basename
model_name0 = os.path.basename(model_path0).split('.')[0]
model_name1 = os.path.basename(model_path1).split('.')[0]
states0 = int(re.findall("\d+states",model_name0)[0][:-6])
states1 = int(re.findall("\d+states",model_name1)[0][:-6])
time_lims0 = [int(x) for x in \
        re.findall("\d+_\d+time",model_name0)[0][:-4].split('_')]
time_lims1 = [int(x) for x in \
        re.findall("\d+_\d+time",model_name1)[0][:-4].split('_')]
bin_width0 = int(re.findall("\d+bin",model_name0)[0][:-3])
bin_width1 = int(re.findall("\d+bin",model_name1)[0][:-3]) 
# Extract data_dir from model_path
data_dir0 = "/".join(model_path0.split('/')[:-3])
data_dir1 = "/".join(model_path1.split('/')[:-3])

######IMPORT MODEL TAUS#######
if os.path.exists(model_path0):
    print('Trace loaded from cache')
    with open(model_path0, 'rb') as buff:
        data = pickle.load(buff)
    tau_samples_0 = data['tau']
    # Remove pickled data to conserve memory
    del data 

if os.path.exists(model_path1):
    print('Trace loaded from cache')
    with open(model_path1, 'rb') as buff:
        data = pickle.load(buff)
    tau_samples_1 = data['tau']
    # Remove pickled data to conserve memory
    del data
#note, the shape of the tau datasets is (20000,120,3)
[tau_samp_num, tau_trials, tau_changepoints] = tau_samples_0.shape

#Convert all tau values to integers and find the mode
tau_0_modes = np.zeros([tau_trials,tau_changepoints])
tau_1_modes = np.zeros([tau_trials,tau_changepoints])
for i in range(tau_trials):
    taus_0 = np.array(tau_samples_0[:,i,:],dtype='int')
    taus_1 = np.array(tau_samples_1[:,i,:],dtype='int')
    for j in range(tau_changepoints):
        mode0 = stats.mode(taus_0[:,j])
        tau_0_modes[i,j] = mode0.mode[0]
        mode1 = stats.mode(taus_1[:,j])
        tau_1_modes[i,j] = mode1.mode[0]
del i, j, taus_0, taus_1, mode0, mode1 

#Calculate correlation
correlation_coef = np.corrcoef(tau_0_modes,tau_1_modes)
average_correlation = np.mean(correlation_coef) #this is augmented by a diagonal of 1s



