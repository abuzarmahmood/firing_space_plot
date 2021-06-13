"""
Given a data directory list
Descend into directories and find:
    1) Whether data is sorted
    2) Region names
    3) If directory contains changepoint pkl files
    4) If directory contains split pkl files
"""

import numpy as np
import json
from glob import glob
import os
import pandas as pd
import sys

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs')
from check_data import check_data 

dir_list_path = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_data_dirs.txt'

with open(dir_list_path, 'r') as dir_file:
    dir_list =  dir_file.readlines()
dir_list = [x[:-1] for x in dir_list]
basename_list = [os.path.basename(x) for x in dir_list]

test = check_data(dir_list[27])
test.run_all()
test.print()

all_dir_info = []
for this_dir in dir_list:
    this_info = check_data(this_dir)
    this_info.run_all()
    all_dir_info.append(this_info.print())

all_dir_info_frame = pd.DataFrame(data = all_dir_info,
        columns = ['sorted','regions','non_split_pkl','split_pkl'])
all_dir_info_frame['name'] = basename_list
all_dir_info_frame['path'] = dir_list

## Take out single-region recordings with split pkls
region_bool = all_dir_info_frame.regions.map(len) == 1
split_pkl_bool = all_dir_info_frame.split_pkl == 1
fin_frame = all_dir_info_frame.loc[region_bool * split_pkl_bool]

## Write out
all_dir_info_frame.to_pickle(os.path.join(os.path.dirname(dir_list_path),
                        'all_dir_info_frame.pkl'))
fin_frame.to_pickle(os.path.join(os.path.dirname(dir_list_path),
                        'single_region_split_frame.pkl'))
