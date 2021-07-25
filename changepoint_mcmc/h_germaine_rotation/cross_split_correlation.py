#####IMPORTS######
import numpy as np
import os
import sys
import pickle
import argparse
import re
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
import pandas as pd
import ast
from numpy.random import default_rng
rng = default_rng()
from matplotlib import cm
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
sys.path.append('/media/bigdata/firing_space_plot/changepoint_mcmc')
from ephys_data import ephys_data
print('Imports Done')


#####LOAD DATA#####
####Load basic stats####

split_num = 10 #set the number of splits here manually

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('model_path',  help = 'Path to model pkl file')
args = parser.parse_args()
base_path = args.model_path #'...../split_analyses/'

average_0 = []
average_1 = []
average_0_taste = []
average_1_taste = []
variance_0 = []
variance_1 = []
variance_0_taste = []
variance_1_taste = []
mse = []
mse_taste = []

#Run through all splits
for i in range(split_num):
    split_folder = base_path + "/split_" + str(i) + "/"
    if os.path.exists(split_folder): #pull needed data
        all_data = [] #pull out csv values
        with open(split_folder + 'analysis_results.csv', newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                first_val = row[0]
                isint = 0
                try:
                    int(first_val[0])
                    isint = 1
                except:
                    isint = 0
                if first_val[0] == '[' or isint == 1: #numeric row
                    all_data.append(row)
        #separate csv data into variables
        average_0.append(ast.literal_eval(all_data[0][0]))
        average_1.append(ast.literal_eval(all_data[0][1]))
        average_0_taste.append(ast.literal_eval(all_data[1][0]))
        average_1_taste.append(ast.literal_eval(all_data[1][1]))
        variance_0.append(ast.literal_eval(all_data[2][0]))
        variance_1.append(ast.literal_eval(all_data[2][1]))
        variance_0_taste.append(ast.literal_eval(all_data[3][0]))
        variance_1_taste.append(ast.literal_eval(all_data[3][1]))
        mse.append(ast.literal_eval(all_data[4][0]))
        mse_taste.append(ast.literal_eval(all_data[5][0]))
    else: #error message printout
        print("Folder " + split_folder + " doesn't exist. Data not loaded.")
    
#Create save path for split analyses
analysis_save_path = base_path + '/combined_splits/'
if not os.path.exists(analysis_save_path):
        os.makedirs(analysis_save_path)
        
#Store basic size data
tau_changepoints = len(np.array(average_0).T)
taste_num = len(np.array(average_0_taste[0][0]))

print('Data Loaded.')


####Load correlation stats####
split_num = 10 #set the number of splits here manually

parser = argparse.ArgumentParser(description = 'Script to fit changepoint model')
parser.add_argument('model_path',  help = 'Path to model pkl file')
args = parser.parse_args()
base_path = args.model_path #'...../split_analyses/'

s_correlation_coef = [] #index 1
p_correlation_coef = [] #index 3
s_correlation_all_taste = [] #index 5
p_correlation_all_taste = [] #index 7

#Run through all splits
for i in range(split_num):
    split_folder = base_path + "split_" + str(i) + "/"
    if os.path.exists(split_folder): #pull needed data
        all_data = [] #pull out csv values
        with open(split_folder + 'regular_correlations.csv', newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            for row in datareader:
                first_val = row[0]
                isint = 0
                try:
                    int(first_val[0])
                    isint = 1
                except:
                    isint = 0
                if first_val[0] == '[' or first_val[0] == '-' or isint == 1: #numeric row
                    all_data.append(row)
        #separate csv data into variables
        s_correlation_coef.append([float(all_data[0][i]) for i in range(len(all_data[0]))])
        p_correlation_coef.append([ast.literal_eval(all_data[1][i]) for i in range(len(all_data[1]))])
        s_correlation_all_taste.append([ast.literal_eval(all_data[2][i]) for i in range(len(all_data[2]))])
        p_correlation_all_taste.append([ast.literal_eval(all_data[3][i]) for i in range(len(all_data[3]))])
    else: #error message printout
        "Folder " + split_folder + " doesn't exist. Data not loaded."

print('Correlation Data Loaded.')

#####PLOT#####

#Average Values
fig, all_ax = plt.subplots(1,tau_changepoints,sharey=False, figsize=(10,5))
for i in range(tau_changepoints):
    all_average_taus_0 = np.array(average_0).T[i]
    all_average_taus_1 = np.array(average_1).T[i]
    plt.sca(all_ax[i])
    plt.boxplot([all_average_taus_0,all_average_taus_1])
    x_1 = 1*np.ones(all_average_taus_0.shape)+np.random.normal(loc=0.0, scale=0.01, size=all_average_taus_0.shape)
    x_2 = 2*np.ones(all_average_taus_1.shape)+np.random.normal(loc=0.0, scale=0.01, size=all_average_taus_1.shape)
    plt.scatter(x_1, all_average_taus_0, alpha=0.4)
    plt.scatter(x_2, all_average_taus_1, alpha=0.4)
    plt.title('Tau '+str(i+1))
plt.suptitle('Box Plot of Average Tau Values Per Split') 
plt.savefig(os.path.join(analysis_save_path,'avg_taus'))

#Average Values By Taste (Don't use for combined script)
fig, all_ax = plt.subplots(tau_changepoints,taste_num,sharey=True, figsize=(10,5))
for i in range(tau_changepoints):
    for j in range(taste_num):
        all_average_taus_0 = np.array(average_0_taste).T[j][i]
        avg_val_0 = np.mean(all_average_taus_0)
        all_average_taus_1 = np.array(average_1_taste).T[j][i]
        avg_val_1 = np.mean(all_average_taus_1)
        plt.sca(all_ax[i][j])
        _ = plt.hist(all_average_taus_0)
        _ = plt.hist(all_average_taus_1)
        plt.title('Taste '+str(j+1) + '; Tau '+str(i+1))
#fig.suptitle('Average Tau Values Per Taste') 
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'avg_taus'))

#Average Tau Value Distances By Taste
fig, all_ax = plt.subplots(tau_changepoints,taste_num,sharey=True,sharex=True,figsize=(20,10))
all_diff = []
for i in range(tau_changepoints):
    for j in range(taste_num):
        all_average_taus_0 = np.array(average_0_taste).T[j][i]
        all_average_taus_1 = np.array(average_1_taste).T[j][i]
        diff = [np.abs(x - y) for x in all_average_taus_0 for y in all_average_taus_1]
        all_diff.extend(diff)
        plt.sca(all_ax[i][j])
        _ = plt.hist(diff)
        plt.title('Taste '+str(j+1) + '; Tau '+str(i+1) +'; Avg = ' + str(np.round(np.mean(diff),2)) + '\n Absolute Split Distance Histogram')
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'tau_diff_hist'))
print('Overall Average: ' + str(np.round(np.mean(all_diff),2)))

#Plot Tau Correlations Across Splits

#Spearman
#fig, all_ax = plt.subplots(1,tau_changepoints,sharey=True, figsize=(10,5))
#fig = plt.figure(figsize=(10,5))
#for i in range(tau_changepoints):
#    corr_vals = np.array(s_correlation_coef).T[i]
#    #plt.sca(all_ax[i])
#    _ = plt.hist(corr_vals, label='Tau '+str(i+1))
#plt.title('Histograms of Correlation Per Tau') 
#plt.legend()
#plt.savefig(os.path.join(analysis_save_path,'tau_corr_hist'))

#Pearson
#fig, all_ax = plt.subplots(1,tau_changepoints,sharey=True, figsize=(10,5))
#fig = plt.figure(figsize=(10,5))
#for i in range(tau_changepoints):
#    corr_vals = np.array(p_correlation_coef).T[i]
#    #plt.sca(all_ax[i])
#    _ = plt.hist(corr_vals, label='Tau '+str(i+1))
#plt.title('Histograms of Correlation Per Tau') 
#plt.legend()
#plt.savefig(os.path.join(analysis_save_path,'tau_corr_hist'))

#Plot Tau Correlations Across Splits and Tastes

#Spearman
#fig, all_ax = plt.subplots(1,taste_num,sharey=True, figsize=(10,5))
#for j in range(taste_num):
#    plt.sca(all_ax[j])
#    for i in range(tau_changepoints):
#        #correlation_coeff_taste is split_num x tau x taste
#        corr_vals = np.array(s_correlation_all_taste).T[i][j]
#        _ = plt.hist(corr_vals, label='Tau '+str(i+1))
#    plt.title('Taste '+str(j)) 
#    plt.legend()
#fig.suptitle('Tau Correlation Histograms')
#plt.savefig(os.path.join(analysis_save_path,'tau_corr_hist_taste'))

#Pearson
#fig, all_ax = plt.subplots(1,taste_num,sharey=True, figsize=(10,5))
#for j in range(taste_num):
#    plt.sca(all_ax[j])
#    for i in range(tau_changepoints):
#        #correlation_coeff_taste is split_num x tau x taste
#        corr_vals = np.array(p_correlation_all_taste).T[i][j]
#        _ = plt.hist(corr_vals, label='Tau '+str(i+1))
#    plt.title('Taste '+str(j)) 
#    plt.legend()
#fig.suptitle('Tau Correlation Histograms')
#plt.savefig(os.path.join(analysis_save_path,'tau_corr_hist_taste'))

#####LOAD DATA#####

print('Now loading split percentiles compared to shuffled.')

#Want to plot histograms of real data percentiles so need to load:
real_p_percentiles = [] #10
real_s_percentiles = [] #11
real_p_percentiles_taste = [] #12
real_s_percentiles_taste = [] #13
real_mse_percentiles = [] #14
real_mse_percentiles_taste = [] #15

#Run through all splits
for i in range(split_num):
    split_folder = base_path + "/split_" + str(i) + "/"
    if os.path.exists(split_folder): #pull needed data
        all_data = [] #pull out csv values
        with open(split_folder + 'analysis_results_shuffle.csv', newline='\n') as csvfile:
            datareader = csv.reader(csvfile, delimiter=' ')
            for row in datareader:
                first_val = row[0]
                isint = 0
                try:
                    int(first_val[0])
                    isint = 1
                except:
                    isint = 0
                if first_val[0] == '[' or first_val[0] == '-' or isint == 1: #numeric row
                    all_data.append(row)
        #separate csv data into variables
        real_p_percentiles.append(all_data[10])
        real_s_percentiles.append(all_data[11])
        real_p_percentiles_taste.append(all_data[12])
        real_s_percentiles_taste.append(all_data[13])
        real_mse_percentiles.append(all_data[14])
        real_mse_percentiles_taste.append(all_data[15])
    else: #error message printout
        "Folder " + split_folder + " doesn't exist. Data not loaded."

#Reformat data properly
real_p_percentiles = [[((real_p_percentiles[j][i]).replace('[[','')).replace(']]','') for i in range(len(real_p_percentiles[j]))] for j in range(len(real_p_percentiles))]
for j in range(len(real_p_percentiles)):
    try:
        while '' in real_p_percentiles[j]: real_p_percentiles[j].remove('')
        real_p_percentiles[j] = [ast.literal_eval(real_p_percentiles[j][i]) for i in range(len(real_p_percentiles[j]))]
    except:
        real_p_percentiles[j] = [ast.literal_eval(real_p_percentiles[j][i]) for i in range(len(real_p_percentiles[j]))]
real_s_percentiles = [[((real_s_percentiles[j][i]).replace('[[','')).replace(']]','') for i in range(len(real_s_percentiles[j]))] for j in range(len(real_s_percentiles))]
for j in range(len(real_s_percentiles)):
    try:
        while '' in real_s_percentiles[j]: real_s_percentiles[j].remove('')
        real_s_percentiles[j] = [ast.literal_eval(real_s_percentiles[j][i]) for i in range(len(real_s_percentiles[j]))]
    except:
        real_s_percentiles[j] = [ast.literal_eval(real_s_percentiles[j][i]) for i in range(len(real_s_percentiles[j]))]
real_p_percentiles_taste = [ast.literal_eval(((((((((real_p_percentiles_taste[i][0]).replace('\n ',',')).replace('. ','.0')).replace('[  ','[')).replace('[ ','[')).replace('    ',',')).replace('   ',',')).replace('  ',',')).replace(' ',',')) for i in range(len(real_p_percentiles_taste))]
real_s_percentiles_taste = [ast.literal_eval(((((((((real_s_percentiles_taste[i][0]).replace('\n ',',')).replace('. ','.0')).replace('[  ','[')).replace('[ ','[')).replace('    ',',')).replace('   ',',')).replace('  ',',')).replace(' ',',')) for i in range(len(real_s_percentiles_taste))]
real_mse_percentiles = [[((real_mse_percentiles[j][i]).replace('[[','')).replace(']]','') for i in range(len(real_mse_percentiles[j]))] for j in range(len(real_mse_percentiles))]
for j in range(len(real_mse_percentiles)):
    try:
        while '' in real_mse_percentiles[j]: real_mse_percentiles[j].remove('')
        real_mse_percentiles[j] = [ast.literal_eval(real_mse_percentiles[j][i]) for i in range(len(real_mse_percentiles[j]))]
    except:
        real_mse_percentiles[j] = [ast.literal_eval(real_mse_percentiles[j][i]) for i in range(len(real_mse_percentiles[j]))]
real_mse_percentiles_taste = [ast.literal_eval(((((((((real_mse_percentiles_taste[i][0]).replace('\n ',',')).replace('. ','.0')).replace('[  ','[')).replace('[ ','[')).replace('    ',',')).replace('   ',',')).replace('  ',',')).replace(' ',',')) for i in range(len(real_mse_percentiles_taste))]

print('Shuffled Data Loaded.')

#Resave percentiles as numpy binary files for easy upload to compare across all BLA and all GC datasets

np.save(os.path.join(analysis_save_path,'real_p_percentiles'), np.array(real_p_percentiles), allow_pickle=False, fix_imports=False)
np.save(os.path.join(analysis_save_path,'real_s_percentiles'), np.array(real_s_percentiles), allow_pickle=False, fix_imports=False)
np.save(os.path.join(analysis_save_path,'real_p_percentiles_taste'), np.array(real_p_percentiles_taste), allow_pickle=False, fix_imports=False)
np.save(os.path.join(analysis_save_path,'real_s_percentiles_taste'), np.array(real_s_percentiles_taste), allow_pickle=False, fix_imports=False)
np.save(os.path.join(analysis_save_path,'real_mse_percentiles'), np.array(real_mse_percentiles), allow_pickle=False, fix_imports=False)
np.save(os.path.join(analysis_save_path,'real_mse_percentiles_taste'), np.array(real_mse_percentiles_taste), allow_pickle=False, fix_imports=False)

#Across all tastes (real_p_percentiles, real_s_percentiles)

#Pearson's
fig, all_ax = plt.subplots(1, tau_changepoints, sharey = True, figsize=(10,5))
for j in range(tau_changepoints):
    plt.sca(all_ax[j])
    _ = plt.hist(np.array(real_p_percentiles).T[j])
    plt.ylabel('Number of Percentiles')
    plt.xlabel('Percentile Value')
    plt.title('Tau '+str(j+1))
fig.suptitle('Split Real Data Pearsons Correlation Percentiles')
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'splits_pearsons_percentiles'))

#Spearman's
fig, all_ax = plt.subplots(1, tau_changepoints, sharey = True, figsize=(10,5))
for j in range(tau_changepoints):
    plt.sca(all_ax[j])
    _ = plt.hist(np.array(real_s_percentiles).T[j])
    plt.ylabel('Number of Percentiles')
    plt.xlabel('Percentile Value')
    plt.title('Tau '+str(j+1))
fig.suptitle('Split Real Data Spearmans Correlation Percentiles')
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'splits_pearsons_percentiles'))


#By taste (real_p_percentiles_taste, real_s_percentiles_taste)

#Pearson's
fig, all_ax = plt.subplots(tau_changepoints, taste_num, sharey = True, figsize=(10,5))
for i in range(tau_changepoints):
    for j in range(taste_num):
        plt.sca(all_ax[i][j])
        _ = plt.hist(np.array(real_p_percentiles_taste).T[i][j])
        #plt.ylabel('Number of Percentiles')
        #plt.xlabel('Percentile Value')
        plt.title('Taste '+str(j+1)+'\nTau '+str(i+1))
#fig.supylabel('Number of Percentiles')
#fig.supxlabel('Percentile Value')
fig.suptitle('Split Real Data Pearsons Taste Correlation Percentiles')
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'splits_pearsons_percentiles_taste'))

#Spearman's
fig, all_ax = plt.subplots(tau_changepoints, taste_num, sharey = True, figsize=(10,5))
for i in range(tau_changepoints):
    for j in range(taste_num):
        plt.sca(all_ax[i][j])
        _ = plt.hist(np.array(real_s_percentiles_taste).T[i][j])
        #plt.ylabel('Number of Percentiles')
        #plt.xlabel('Percentile Value')
        plt.title('Taste '+str(j+1)+'\nTau '+str(i+1))
#fig.supylabel('Number of Percentiles')
#fig.supxlabel('Percentile Value')
fig.suptitle('Split Real Data Spearmans Taste Correlation Percentiles')
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'splits_spearmans_percentiles_taste'))


#Across all tastes (real_p_percentiles, real_s_percentiles)

#All MSE
fig, all_ax = plt.subplots(1, tau_changepoints, sharey = True, figsize=(10,5))
for j in range(tau_changepoints):
    plt.sca(all_ax[j])
    _ = plt.hist(np.array(real_mse_percentiles).T[j])
    plt.title('Tau '+str(j+1))
#fig.supylabel('Number of Percentiles')
#fig.supxlabel('MSE Value')
fig.suptitle('Split Real Data MSE Percentiles')
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'splits_mse_percentiles'))

#Taste MSE
fig, all_ax = plt.subplots(tau_changepoints, taste_num, sharey = True, figsize=(10,5))
for i in range(tau_changepoints):
    for j in range(taste_num):
        plt.sca(all_ax[i][j])
        _ = plt.hist(np.array(real_mse_percentiles_taste).T[i][j])
        #plt.ylabel('Number of Percentiles')
        #plt.xlabel('Percentile Value')
        plt.title('Taste '+str(j+1)+'\nTau '+str(i+1))
#fig.supylabel('Number of Percentiles')
#fig.supxlabel('MSE Value')
fig.suptitle('Split Real Data MSE Taste Percentiles')
fig.tight_layout()
plt.savefig(os.path.join(analysis_save_path,'splits_mse_percentiles_taste'))

print("Plots Saved - File Complete.")