import sys
import os
from tqdm import tqdm, trange
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# blech_clust_dir = os.path.expanduser('~/Desktop/blech_clust')
blech_clust_dir = '/home/abuzarmahmood/projects/blech_clust'
sys.path.append(blech_clust_dir)
from utils.ephys_data.ephys_data import ephys_data


data_dir_file_path = '/media/fastdata/Thomas_Data/all_data_dirs.txt'
data_dir_list = [x.strip() for x in open(data_dir_file_path, 'r').readlines()]

# base_dir = '/media/bigdata/firing_space_plot/firing_analyses/inter_region_rate_regression'
base_dir = '/home/abuzarmahmood/projects/firing_space_plot/firing_analyses/inter_region_rate_regression'
artifact_dir =  os.path.join(base_dir,'artifacts')
plot_dir = os.path.join(base_dir,'plots')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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
# Perform regression
############################################################

time_lims = np.array([2000, 4000])
step_size = 25
time_inds = time_lims // step_size

unit_counts = []
n_repeats = 10
train_test_split = 0.75
for i, this_row in tqdm(all_firing_frame.iterrows()):
    this_basename = this_row['basename']
    this_region_units = this_row['region_units']
    this_seq_firing = this_row['seq_firing']
    unit_counts_dict = {x:len(this_region_units[x]) for x in this_region_units.keys()}
    unit_counts_dict['basename'] = this_basename
    unit_counts.append(unit_counts_dict)

    # Split firing by region
    for taste_ind, this_taste_dat in enumerate(tqdm(this_seq_firing)):
        region_firing = [this_taste_dat[:, x] for x in this_region_units.values()]
        # Cut down to time window
        region_firing = [x[...,time_inds[0]:time_inds[1]] for x in region_firing]
        for this_repeat in trange(n_repeats):
            n_trials = region_firing[0].shape[0]
            train_trials = np.random.choice(n_trials, int(n_trials*train_test_split), replace = False)
            test_trials = np.setdiff1d(np.arange(n_trials),train_trials)
            train_firing = [x[train_trials] for x in region_firing]
            test_firing = [x[test_trials] for x in region_firing]
            train_firing_long = []
            for region in train_firing:
                region_long = region.swapaxes(0,1)
                region_long = region_long.reshape(region_long.shape[0],-1)
                # Scale
                scaled_region = StandardScaler().fit_transform(region_long.T)
                train_firing_long.append(scaled_region)
            test_firing_long = []
            for region in test_firing:
                region_long = region.swapaxes(0,1)
                region_long = region_long.reshape(region_long.shape[0],-1)
                # Scale
                scaled_region = StandardScaler().fit_transform(region_long.T)
                test_firing_long.append(scaled_region)

            reg = MLPRegressor(
                    hidden_layer_sizes = (100,100,100), 
                    max_iter = 1000,
                    verbose = False,
                    )
            reg.fit(train_firing_long[0],train_firing_long[1])
            pred = reg.predict(test_firing_long[0])
            r2 = r2_score(test_firing_long[1],pred)
            
            save_dict = dict(
                    basename = this_basename,
                    region_units = this_region_units,
                    taste_ind = taste_ind,
                    repeat = this_repeat,
                    train_firing_long = train_firing_long,
                    test_firing_long = test_firing_long,
                    pred = pred,
                    r2 = r2,
                    )
            # Write to file
            save_path = os.path.join(artifact_dir,f'{this_basename}_taste{taste_ind}_repeat{this_repeat}.pkl')
            pd.to_pickle(save_dict,save_path)

            img_kwargs = dict(
                    aspect = 'auto',
                    interpolation = 'none',
                    )
            fig, ax = plt.subplots(3,2, sharey = True,
                                   figsize = (5,10))
            ax[0,0].imshow(train_firing_long[0], **img_kwargs)
            ax[0,1].imshow(train_firing_long[1], **img_kwargs)
            ax[1,0].imshow(test_firing_long[0], **img_kwargs)
            ax[1,1].imshow(test_firing_long[1], **img_kwargs)
            ax[2,1].imshow(pred, **img_kwargs)
            ax[0,0].set_title('Train firing')
            ax[0,1].set_title('Train target')
            ax[1,0].set_title('Test firing')
            ax[1,1].set_title('Test target')
            ax[2,1].set_title('Predicted')
            ax[2,0].set_title(f'R2: {r2:.2f}')
            plt.suptitle(f'{this_basename} - {this_taste_dat.shape}')
            # plt.show()
            plt.savefig(os.path.join(plot_dir,f'{this_basename}_taste{taste_ind}_repeat{this_repeat}.png'))
            plt.close()


unit_counts_frame = pd.DataFrame(unit_counts)
