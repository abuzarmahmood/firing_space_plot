
# =============================================================================
# Import stuff
# =============================================================================

# import Libraries
import os # functions for interacting w operating system
import numpy as np # module for low-level scientific computing
import easygui
import pandas as pd
import itertools
import glob
from datetime import date
import argparse

# =============================================================================
# Define workhorse function
# =============================================================================

def MedMS8_reader(file_name, data_sheet = None):
# =============================================================================
#     Input: File Name (with directory) from MedAssociates Davis Rig
#           (e.g. .ms8.text)
#
#     Output: Dictionary containing a dataframe (all lick data categorized), file
#             information (animal name, date of recording, etc), and a matrix
#             with all latencies between licks by trial
# =============================================================================

    print(f"Processing file {os.path.basename(file_name)}", end = '\t : ')
    with open(file_name, 'r') as file_input:
        lines = file_input.readlines()

    #Create dictionary for desired file into
    Detail_Dict_keys = ['FileName', 'StartDate', 'StartTime',
                   'Animal','Condition', 'MAXFLick', 'Trials',
                   'LickDF', 'LatencyMatrix']
    Detail_Dict = dict(zip(Detail_Dict_keys, [None] * len(Detail_Dict_keys)))

    #Extract file name and store
    Detail_Dict['FileName'] = file_name[file_name.rfind('/')+1:]

    ##Store details in dictionary and construct dataframe
    search_words = ['Start Date', 'Start Time', 'Animal ID', 'Max Wait', 'Max Number']
    save_keys = ['StartDate', 'StartTime', 'Animal', 'MAXFLick', 'Trials']

    # Pull out details in header
    for i in range(len(lines)):
        for num, detail in enumerate(search_words):
            if detail in lines[i]:
                Detail_Dict[save_keys[num]] = lines[i].split(',')[-1][:-1].strip()
        # Pull out trial lick COUNTS (not lick intervals)
        if "PRESENTATION" and "TUBE" in lines[i]:
            ID_line = i
        if len(lines[i].strip()) == 0:
            Trial_data_stop = i

    # Create dataframe to keep count data
    if ID_line > 0 and Trial_data_stop > 0:
        #Create dataframe
        df = pd.DataFrame(columns= [x.strip() for x in lines[ID_line].split(',')],
                      data=[[x.strip() for x in row.split(',')] \
                              for row in lines[ID_line+1:Trial_data_stop]])

    #Set concentrations to 0 if concentration column blank
    df.CONCENTRATION.replace('',0)

    #Convert specific columns to numeric
    numeric_cols = ["PRESENTATION","TUBE","CONCENTRATION","LICKS","Latency"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    #Add in identifier columns
    df.insert(loc=0, column='Animal', value=Detail_Dict['Animal'])
    df.insert(loc=3, column='Trial_num', value='')
    df['Trial_num'] = df.groupby('TUBE').cumcount()+1

    #Add column if 'Retries' Column does not exist
    if 'Retries' not in df:
        df.insert(df.columns.get_loc("Latency")+1,'Retries', '      0')

    #Check if user has data sheet of study details to add to dataframe
    if data_sheet is not None: 

        detail_df=pd.read_csv(data_sheet, header=0,sep='\t')
        # Make sure everything is stripped
        for col_name in detail_df.columns:
            detail_df[col_name] = [x.strip() for x in detail_df[col_name]]

        #Match data with detail sheeet and pull out potential matches
        date_match = np.where(detail_df.Date == Detail_Dict['StartDate'])[0]
        name_match = np.where(np.array([x.lower() for x in detail_df.Animal]) == \
                Detail_Dict['FileName'].split('_')[0].lower())[0] 
        fin_match = list(set(date_match).intersection(set(name_match)))

        if len(fin_match) > 0:
            #print(f"{Detail_Dict['FileName']},{Detail_Dict['StartDate']} Matched")
            print("Matched")
            fin_match = fin_match[0]
            #Add details to dataframe
            df.insert(loc=1, column='Notes', \
                value=detail_df.Notes[fin_match].lower())
            df.insert(loc=2, column='Condition', \
                value=detail_df.Condition[fin_match].lower())

        else:
            print("NOT Matched")
            #Add blank columns
            df.insert(loc=1, column='Notes', value='')
            df.insert(loc=2, column='Condition', value='')

    return Detail_Dict, df

# =============================================================================
# =============================================================================
# # #BEGIN PROCESSING
# =============================================================================
# =============================================================================

# Create argument parser
parser = argparse.ArgumentParser(description = 'Extracts lick counts from ms8.txt files')
parser.add_argument('--dir_name', '-d', help = 'Directory containing ms8.txt files')
parser.add_argument('--data_sheet', '-s', help = 'File containing experiment details')
args = parser.parse_args()

#Get name of directory where the data files sit, and change to that directory for processing
if args.dir_name: 
    dir_name = args.dir_name 
else:
    dir_name = easygui.diropenbox()

if args.data_sheet: 
    data_sheet = args.data_sheet 
else:
    #Ask user if they will be using a detailed sheet
    msg   = "Do you have a datasheet with animal details?"
    detail_check = easygui.buttonbox(msg,choices = ["Yes","No"])

    if detail_check == 'Yes':
        #Ask user for experimental data sheet if they want to include additional details
        detail_name = easygui.diropenbox(msg='Where is the ".txt" file?')
        data_sheet = glob.glob(detail_name+'/*.txt')
    else:
        data_sheet = None

# Print what files are being used
print('\n')
print(f'Data dir : {dir_name}')
print(f'Data sheet : {data_sheet}')


#Initiate a list to store individual file dataframes
merged_data = []

#Look for the ms8 files in the directory
file_list = np.sort(glob.glob(os.path.join(dir_name, "**.ms8.txt")))
dict_list, df_list = zip(*[MedMS8_reader(file_name, data_sheet) \
        for file_name in file_list])

#Append dataframe with animal's details
merged_df = pd.concat(df_list)

#Save dataframe for later use/plotting/analyses
#timestamped with date
merged_df.to_pickle(dir_name+'/%s_grouped_dframe.df' %(date.today().strftime("%d_%m_%Y")))
