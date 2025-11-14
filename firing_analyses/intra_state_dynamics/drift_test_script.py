from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data import visualize as vz
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from pprint import pprint as pp

data_dir = '/media/storage/NM_resorted_data/laser_2500ms/NM51_2500ms_161030_130155'

this_data = ephys_data.ephys_data(data_dir)
this_data.get_firing_rates()
this_data.separate_laser_firing()

firing_array = this_data.off_firing
stacked_firing = np.concatenate(firing_array, axis=0)

vz.firing_overview(stacked_firing.swapaxes(0,1))
plt.show()

NBT_KM_paths_list_path = '/media/bigdata/firing_space_plot/NBT_EMB_Classifier_Analyses/src/GC_EMG_changepoints/data_dir_list.txt'
NM_paths_list_path = '/media/storage/NM_resorted_data/dir_paths.txt'

NBT_KM_paths = open(NBT_KM_paths_list_path).read().splitlines()
NM_paths = open(NM_paths_list_path).read().splitlines()

all_paths = NBT_KM_paths + NM_paths

loaded_paths = []
spike_list = []
firing_rate_list = []
#ind = 3
for this_path in tqdm(all_paths): 
    try:
        dat = ephys_data.ephys_data(this_path)
        dat.get_spikes()
        dat.get_firing_rates()
        loaded_paths.append(this_path)

        dat.check_laser()
        # If laser exists, only get non-laser trials
        if dat.laser_exists:
            dat.separate_laser_spikes()
            dat.separate_laser_firing()
            spike_list.append(np.array(dat.off_spikes))
            firing_rate_list.append(np.array(dat.off_firing))
        else:
            spike_list.append(np.array(dat.spikes))
            firing_rate_list.append(np.array(dat.firing_list))

    except Exception as e:
        print(f"Could not load data from {this_path}: {e}")
        continue

time_lims = [-500, 2000]
firing_time_inds = np.where((this_data.time_vector >= time_lims[0]) & (this_data.time_vector <= time_lims[1]))[0]

for i, this_firing in enumerate(firing_rate_list):
    # fig, ax = plt.subplots(len(this_firing),1, figsize=(10,15), sharex=True)
    stacked_firing = np.concatenate(this_firing, axis=0)  # Shape: (total_trials, num_neurons, num_time_bins)
    fig, ax = vz.firing_overview(
            stacked_firing.swapaxes(0,1)[..., firing_time_inds]  # Shape: (num_neurons, total_trials, selected_time_bins)
            # stacked_firing.swapaxes(0,1)
            )
    this_base_name = loaded_paths[i].split('/')[-1]
    fig.suptitle(this_base_name)
    plt.show()
    # plt.savefig(os.path.join(plot_dir, f'{this_base_name}_mean_firing_rates.png'))
    # plt.close()

wanted_basename_pattern = '210620'
wanted_paths = [p for p in loaded_paths if wanted_basename_pattern in p]

wanted_data_indices = [loaded_paths.index(p) for p in wanted_paths]
wanted_data_firing = [firing_rate_list[i] for i in wanted_data_indices]
wanted_firing = wanted_data_firing[0]  # Assuming we want the first dataset matching the pattern

vz.firing_overview(np.concatenate(wanted_firing, axis=0).swapaxes(0,1))
plt.show()

# long_firing = np.concatenate(wanted_firing, axis=0)  # Shape: (total_trials, num_neurons, num_time_bins)
wanted_taste_firing = wanted_firing[1]  
vz.firing_overview(wanted_taste_firing.swapaxes(0,1))
plt.show()

trial_breaks = [0,3,10,15,25]
trial_chunks = [(trial_breaks[i], trial_breaks[i+1]) for i in range(len(trial_breaks)-1)]

for i in trial_chunks:
    this_trials = wanted_taste_firing[trial_breaks[i]:trial_breaks[i+1]]
    vz.firing_overview(this_trials.swapaxes(0,1))
plt.show()

test_unit = 22
test_chunk = 3
test_data = wanted_taste_firing[:, test_unit, :][trial_chunks[test_chunk][0]:trial_chunks[test_chunk][1]]
test_data = test_data[..., firing_time_inds]

down_inds = np.linspace(0, basis_funcs.shape[1]-1, test_data.shape[1]).astype(int)
down_template = basis_funcs[:, down_inds]

fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax[0].imshow(test_data, aspect='auto')
ax[0].set_title('Test Data Firing Rates')
ax[1].imshow(down_template, aspect='auto')
ax[1].set_title('Downsampled Basis Functions Template')
plt.show()

estim_weights = estimate_weights(
        test_data,
        down_template
        )

mean_estim_weights = estim_weights.mean(axis=0)
# Tile
mean_estim_weights = np.tile(mean_estim_weights, (test_data.shape[0],1))

fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax[0].imshow(estim_weights, aspect='auto')
ax[0].set_title('Estimated Weights per Trial')
ax[1].imshow(mean_estim_weights, aspect='auto')
ax[1].set_title('Mean Estimated Weights across Trials')
plt.show()

projected_firing = mean_estim_weights.dot(down_template)

fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax[0].imshow(test_data, aspect='auto')
ax[0].set_title('Original Test Data Firing Rates')
ax[1].imshow(projected_firing, aspect='auto')
ax[1].set_title('Projected Firing Rates from Estimated Weights')
plt.show()

# Invert projected firing to try and recover template to check similarity
recov_template = np.linalg.pinv(mean_estim_weights).dot(test_data)

# masked_recov_template = recov_template.copy()
# masked_recov_template[down_template == 0] = 0

vmin = min(down_template.min(), recov_template.min())
vmax = max(down_template.max(), recov_template.max())
fig, ax = plt.subplots(3,1, figsize=(10,6), sharex=True)
ax[0].imshow(down_template, aspect='auto', vmin=vmin, vmax=vmax)
ax[0].set_title('Original Downsampled Basis Functions Template')
ax[1].imshow(recov_template, aspect='auto', vmin=vmin, vmax=vmax)
ax[1].set_title('Recovered Template from Inverted Weights')
ax[2].imshow(masked_recov_template, aspect='auto', vmin=vmin, vmax=vmax)
ax[2].set_title('Masked Recovered Template (Zeroed Outside Original Template)')
plt.show()

template_recov_similarity = np.corrcoef(
        down_template.flatten(),
        recov_template.flatten()
        )[0,1]

def calc_unit_template_dynamics(
        unit_firing,
        template,
        ):
    """Calculate the dynamics of a single unit's firing with respect to a template.
    Args:
        unit_firing (np.ndarray): Shape (num_trials, num_time_bins)
        template (np.ndarray): Shape (num_states, num_time_bins)
    Returns:
        estim_weights (np.ndarray): Shape (num_trials, num_states)
        projected_firing (np.ndarray): Shape (num_trials, num_time_bins)
        template_similarity (float): Correlation between original and recovered template.
    """
    estim_weights = estimate_weights(
            unit_firing,
            template
            )
    mean_estim_weights = estim_weights.mean(axis=0)
    mean_estim_weights = np.tile(mean_estim_weights, (unit_firing.shape[0],1))
    projected_firing = mean_estim_weights.dot(template)
    recov_template = np.linalg.pinv(mean_estim_weights).dot(unit_firing)
    template_similarity = np.corrcoef(
            template.flatten(),
            recov_template.flatten()
            )[0,1]
    return estim_weights, projected_firing, template_similarity

# Find most dynamic units
test_chunk_data = wanted_taste_firing[trial_chunks[test_chunk][0]:trial_chunks[test_chunk][1]]

unit_dynamics = []
for unit_idx in range(test_chunk_data.shape[1]):
    unit_firing = test_chunk_data[:, unit_idx, :][..., firing_time_inds]
    estim_weights, projected_firing, template_similarity = calc_unit_template_dynamics(
            unit_firing,
            down_template
            )
    unit_dynamics.append({
        unit_idx: template_similarity
        })


similarity_scores = [value for d in unit_dynamics for value in d.values()]
sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
sorted_scores = np.round(np.array(similarity_scores)[sorted_indices],2)

sorted_data = test_chunk_data[:, sorted_indices, :][..., firing_time_inds]
fig,ax = vz.firing_overview(sorted_data.swapaxes(0,1))
for this_ax, this_score in zip(ax.flatten(), sorted_scores):
    this_ax.set_title(str(this_score))
plt.show()


##############################

epoch_lims = [
        [-500, 0],
        [0,200],
        [200,850],
        [850,1450],
        [1450,2000]
        ]
epoch_lims = np.array(epoch_lims)
epoch_lims -= epoch_lims.min()
states = len(epoch_lims)
epoch_lens = np.array([np.abs(np.diff(x)[0]) for x in epoch_lims])
basis_funcs = np.stack([np.zeros(epoch_lims.max()) for i in range(states)] )
for this_func, this_lims in zip(basis_funcs, epoch_lims):
    this_func[this_lims[0]:this_lims[1]] = 1
basis_funcs = basis_funcs / norm(basis_funcs,axis=-1)[:,np.newaxis] 
vz.imshow(basis_funcs);plt.show()

assert np.abs(np.diff(time_lims)) == len(basis_funcs.T)

def estimate_weights(firing, template):
    """Estimate weights for each neuron to match the template.

    Args:
        firing (np.ndarray): Shape (num_neurons, num_time_bins)
        template (np.ndarray): Shape (num_states, num_time_bins)
    Returns:
        estim_weight (np.ndarray): Shape (num_neurons, num_states)
    """
    estim_weight = firing.dot(np.linalg.pinv(template))
    return estim_weight
