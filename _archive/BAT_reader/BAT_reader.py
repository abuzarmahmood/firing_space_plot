#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:27:01 2019

@author: bradly
"""
# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import easygui
import pandas as pd
import itertools
import glob
from datetime import date

# =============================================================================
# =============================================================================
# # #DEFINE ALL FUNCTIONS
# =============================================================================
# =============================================================================
#Define a padding function
def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def MedMS8_reader_stone(file_name,file_check):
# =============================================================================
#     Input: File Name (with directory) from MedAssociates Davis Rig
#           (e.g. .ms8.text)
#
#     Output: Dictionary containing a dataframe (all lick data categorized), file
#             information (animal name, date of recording, etc), and a matrix
#             with all latencies between licks by trial
# =============================================================================

    print(f"Processing file {os.path.basename(file_name)}")
    with open(file_name, 'r') as file_input:
        lines = file_input.readlines()

    #Create dictionary for desired file into
    Detail_Dict = {'FileName': None,\
                   'StartDate': None,\
                   'StartTime': None,\
                   'Animal': None,\
                   'Condition': None,\
                   'MAXFLick': None,\
                   'Trials': None,\
                   'LickDF': None,\
                   'LatencyMatrix': None}

    #Extract file name and store
    Detail_Dict['FileName'] = file_name[file_name.rfind('/')+1:]

    #Store details in dictionary and construct dataframe
    for i in range(len(lines)):
        if "Start Date" in lines[i]:
            Detail_Dict['StartDate'] = lines[i].split(',')[-1][:-1].strip()
        if "Start Time" in lines[i]:
            Detail_Dict['StartTime'] = lines[i].split(',')[-1][:-1]
        if "Animal ID" in lines[i]:
            Detail_Dict['Animal'] = lines[i].split(',')[-1][:-1]
        if "Max Wait" in lines[i]:
            Detail_Dict['MAXFLick'] = lines[i].split(',')[-1][:-1]
        if "Max Number" in lines[i]:
            Detail_Dict['Trials'] = lines[i].split(',')[-1][:-1]
        if "PRESENTATION" and "TUBE" in lines[i]:
            ID_line = i
        if len(lines[i].strip()) == 0:
            Trial_data_stop = i
            if ID_line > 0 and Trial_data_stop > 0:
                #Create dataframe
                df = pd.DataFrame(columns=lines[ID_line].split(','),\
                              data=[row.split(',') for row in \
                             lines[ID_line+1:Trial_data_stop]])

                #Remove spaces in column headers (caused by split)
                df.columns = df.columns.str.replace(' ', '')

                #Set concentrations to 0 if concentration column blank
                df['CONCENTRATION']=df['CONCENTRATION'].str.strip()
                df['CONCENTRATION'] = df['CONCENTRATION'].apply(lambda x: 0 if x == '' else x)

                #Convert specific columns to numeric
                df["SOLUTION"] = df["SOLUTION"].str.strip()
                numeric_cols = ["PRESENTATION","TUBE","CONCENTRATION","LICKS","Latency"]
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

                #Add in identifier columns
                df.insert(loc=0, column='Animal', value=Detail_Dict['Animal'])
                df.insert(loc=3, column='Trial_num', value='')
                df['Trial_num'] = df.groupby('TUBE').cumcount()+1

                #Store in dataframe
                Detail_Dict['LickDF'] = df

                #Grab all ILI data, pad with NaNs to make even matrix
                #Store in dictionary (TrialXILI count)
                Detail_Dict['LatencyMatrix'] = boolean_indexing([row.split(',')\
                                      for row in lines[Trial_data_stop+1:]])

    #Add column if 'Retries' Column does not exist
    if 'Retries' not in df:
        df.insert(df.columns.get_loc("Latency")+1,'Retries', '      0')


    #Check if user has data sheet of study details to add to dataframe
    if len(file_check) != 0:

        detail_df=pd.read_csv(file_check[0], header=0,sep='\t')

        #Check data with detail sheeet
        detail_row = np.array(np.where(detail_df.Date==Detail_Dict['StartDate'].strip()))
        for case in range(detail_row.shape[1]):
            if detail_df.Notes[detail_row[:,case][0]].lower() in \
                Detail_Dict['FileName'][Detail_Dict['FileName'].rfind('_')+1:].lower()\
                and detail_df.Animal[detail_row[:,case][0]] in Detail_Dict['Animal']:


                #Add details to dataframe
                df.insert(loc=1, column='Notes', \
                    value=detail_df.Notes[detail_row[:,case][0]].lower())
                df.insert(loc=2, column='Condition', \
                    value=detail_df.Condition[detail_row[:,case][0]].lower())
                break

    if len(file_check) == 0:
        #Add blank columns
        df.insert(loc=1, column='Notes', value='')
        df.insert(loc=2, column='Condition', value='')

    return Detail_Dict

def LickMicroStructure_stone(dFrame_lick,latency_array,bout_crit):
# =============================================================================
#     Function takes in the dataframe and latency matrix pertaining to all
#     licking data obtained from MedMS8_reader_stone as the data sources. This
#     requires a bout_crit
#
#     Input: 1) Dataframe and Licking Matrix (obtained from MedMS8_reader_stone)
#            2) Bout_crit; variable which is the time (ms) needed to pause between
#               licks to count as a bout (details in: Davis 1996 & Spector et al. 1998).
#
#     Output: Appended dataframe with the licks within a bout/trial, latency to
#             to first lick within trial
# =============================================================================
    #Find where the last lick occured in each trial
    last_lick = list(map(lambda x: [i for i, x_ in enumerate(x) if not \
                                    np.isnan(x_)][-1], latency_array))

    #Create function to search rows of matrix avoiding 'runtime error' caused
    #by Nans
    crit_nan_search = np.frompyfunc(lambda x: (~np.isnan(x)) & (x >=bout_crit), 1, 1)

    #Create empty list to store number of bouts by trial
    bouts = []; ILIs_win_bouts = []
    for i in range(latency_array.shape[0]):
        #Create condition if animal never licks within trial
        if last_lick[i] == 0:
            bouts.append(last_lick[i])
            ILIs_win_bouts.append(last_lick[i])

        else:
            bout_pos = np.where(np.array(crit_nan_search(latency_array[i,:])).astype(int))

            #Insert the start number or row to get accurate count
            bout_pos = np.insert(bout_pos,0,1)

            #Caclulate bout duration
            bout_dur = np.diff(bout_pos)

            #Flip through all bouts and calculate licks between and store
            if last_lick[i] != bout_pos[-1]:
                #Insert the last lick row to get accurate count
                bout_pos = np.insert(bout_pos,len(bout_pos),last_lick[i])

                #Calculate bout duration
                bout_dur = np.diff(bout_pos)

            #Append the time diff between bouts to list (each number symbolizes a lick)
            bouts.append(np.array(bout_dur))

            #Grab all ILIs within bouts and store
            trial_ILIs = []

            #append list if only one lick occurs post intial lick
            if len(bout_pos) ==1:
                trial_ILIs.append(latency_array[i,1])

            if len(bout_pos) !=1:
                for lick in range(len(bout_pos)-1):
                    if lick ==0:
                        trial_ILIs.append(latency_array[i,1:bout_pos[lick+1]])
                    else:
                        trial_ILIs.append(latency_array[i,bout_pos[lick]:1+bout_pos[lick+1]])

            #Append for trial total
            ILIs_win_bouts.append(trial_ILIs)

    #Store bout count into dframe
    dFrame_lick["Bouts"] = bouts
    dFrame_lick["ILIs"] = ILIs_win_bouts
    dFrame_lick["Lat_First"] = latency_array[:,1]

    #Return the appended dataframe
    return dFrame_lick

# =============================================================================
# =============================================================================
# # #BEGIN PROCESSING
# =============================================================================
# =============================================================================

#Get name of directory where the data files sit, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Ask user if they will be using a detailed sheet
msg   = "Do you have a datasheet with animal details?"
detail_check = easygui.buttonbox(msg,choices = ["Yes","No"])

if detail_check == 'Yes':
    #Ask user for experimental data sheet if they want to include additional details
    detail_name = easygui.diropenbox(msg='Where is the ".txt" file?')
    file_check= glob.glob(detail_name+'/*.txt')

else:
    file_check = []

#Set bout_pause criteria
bout_pause = 500

#Initiate a list to store individual file dataframes
merged_data = []

#Look for the ms8 files in the directory
file_list = os.listdir('./')
med_name = ''
for files in file_list:
    if files[-3:] == 'txt':
        med_name = files

        file_name = dir_name+'/'+med_name

        #Run functions to extract trial data
        out_put_dict = MedMS8_reader_stone(file_name,file_check)
        dfFull = LickMicroStructure_stone(out_put_dict['LickDF'],
                    out_put_dict['LatencyMatrix'],bout_pause)

        #Merge the data into a list
        merged_data.append(dfFull)

df = out_put_dict

#Append dataframe with animal's details
merged_df = pd.concat(merged_data)

#Format to capitalize first letter of labels
merged_df['Condition'] = merged_df.Condition.str.title()

#Extract dataframe for ease of handling
df = merged_df

#Untack all the ILIs across all bouts to performa math
df_lists = df[['Bouts']].unstack().apply(pd.Series)
df['bout_count'] = np.array(df_lists.count(axis='columns'))
df['Bouts_mean']=np.array(df_lists.mean(axis = 1, skipna = True))

#Work on ILI means
df_lists = df[['ILIs']].unstack().apply(pd.Series)

all_trials = []
for row in range(df_lists.shape[0]):

    trial_ILI = []
    trial_ILI = [np.insert(trial_ILI,len(trial_ILI),df_lists.iloc[row][i]) for i in \
                 range(0,int(np.array(df_lists.iloc[row].shape)))]
    flatt_trial = list(itertools.chain(*trial_ILI))
    all_trials.append(np.array(flatt_trial))

#Store ILIs extended into dataframe
df['ILI_all'] = all_trials

#Save dataframe for later use/plotting/analyses
#timestamped with date
df.to_pickle(dir_name+'/%s_grouped_dframe.df' %(date.today().strftime("%d_%m_%Y")))
