import sys
import os
from tqdm import tqdm
import pandas as pd
blech_clust_dir = os.path.expanduser('~/Desktop/blech_clust')
sys.path.append(blech_clust_dir)
from utils.ephys_data.ephys_data import ephys_data


data_dir_file_path = '/media/fastdata/Thomas_Data/all_data_dirs.txt'
data_dir_list = [x.strip() for x in open(data_dir_file_path, 'r').readlines()]

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/inter_region_rate_regression'
artifact_dir =  os.path.join(base_dir,'artifacts')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

all_firing_path = os.path.join(artifact_dir,'all_firing_frame.pkl')

if not os.path.exists(all_firing_path):
    seq_firing_list = []
    region_units_list = []
    basename_list = []
    for data_dir in tqdm(data_dir_list):
        try:
            this_dat = ephys_data(data_dir)
            this_dat.get_spikes()
            this_dat.get_sequestered_firing()
            this_dat.get_region_units()
            region_dict = dict(
                    zip(
                        this_dat.region_names,
                        this_dat.region_units,
                        )
                    )
            seq_firing = this_dat.sequestered_firing
            seq_firing_list.append(seq_firing)
            region_units_list.append(region_dict)
            basename_list.append(os.path.basename(data_dir))
        except:
            print(f'Error with {data_dir}')

    all_firing_frame = pd.DataFrame(
            dict(
                basename = basename_list,
                region_units = region_units_list,
                seq_firing = seq_firing_list,
                )
            )
    all_firing_frame.to_pickle(all_firing_path)
else:
    all_firing_frame = pd.read_pickle(all_firing_path)


############################################################
############################################################
