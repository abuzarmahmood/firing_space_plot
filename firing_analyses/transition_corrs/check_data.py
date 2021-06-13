import numpy as np
import json
from glob import glob
import os
import pandas as pd

## Define class

class check_data():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def check_sorted(self):
        self.is_sorted = len(glob(os.path.join(self.data_dir, "*.h5")))>0

    def check_info_file(self):
        self.info_path = glob(os.path.join(self.data_dir, "*.info"))
        self.info_present = len(self.info_path)>0 

    def check_region_names(self):
        if 'info_present' not in dir(self):
            self.check_info_file()

        if self.info_present:
            with open(self.info_path[0], 'r') as params_file:
                self.info_dict = json.load(params_file)
            self.regions = self.info_dict['regions']

        else:
            self.regions = []

    def get_pkl_file_paths(self):
       self.pkl_file_paths = glob(os.path.join(self.data_dir, 'saved_models',
                        '**', '*pkl'))

    def check_pkl_counts(self):
        if 'pkl_file_paths' not in dir(self):
            self.get_pkl_file_paths()

        if len(self.pkl_file_paths):
            self.pkl_basenames = [os.path.basename(x) for x in self.pkl_file_paths]
            self.split_inds = [num for num,path in enumerate(self.pkl_basenames) \
                    if 'split_' in path]
            self.non_split_inds = [num for num in range(len(self.pkl_basenames)) \
                    if num not in self.split_inds]

        else:
            self.split_inds = []
            self.non_split_inds = []

        self.split_pkl_present = len(self.split_inds)>0
        self.non_split_pkl_present = len(self.non_split_inds)>0

    def run_all(self):
        self.check_sorted()
        self.check_region_names()
        self.check_pkl_counts()

    def print(self):
        return (self.is_sorted, self.regions, 
                self.non_split_pkl_present, self.split_pkl_present)

####
