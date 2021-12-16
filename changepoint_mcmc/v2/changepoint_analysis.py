import sys
import os
import pandas as pd
import numpy as np
import pickle as pkl

class pkl_handler():
    def __init__(self, file_path):
        self.dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        self.file_name_base = file_name.split('.')[0]
        self.pkl_file_path = \
                os.path.join(self.dir_name, self.file_name_base + ".pkl")
        with open(self.pkl_file_path, 'rb') as f:
                self.data = pkl.load(f)

        model_keys = ['model', 'approx', 'lambda', 'tau', 'data']
        key_savenames = ['_model_structure','_fit_model',
                            'lambda','tau','processed_spikes']
        data_map = dict(zip(model_keys, key_savenames))

        for key, var_name in data_map.items():
            #locals()[var_name] = self.data['model_data'][key]
            setattr(self, var_name, self.data['model_data'][key])

        self.metadata = self.data['metadata']
        self.pretty_metadata = pd.json_normalize(self.data['metadata']).T
        
###########################################################################
## LOAD DATA
###########################################################################

FIT_PKL = '/media/bigdata/firing_space_plot/changepoint_mcmc/'\
        'saved_models/natasha_gc_binary/natasha_gc_binary_0296f33c.info'

model_dat = pkl_handler(FIT_PKL)
