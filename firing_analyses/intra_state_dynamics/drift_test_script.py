"""
Filters to improve "dynamicity" of population
1- Firing rate threshold
2- Stability threshold
3- Single-neuron max-dynamicity threshold
"""

from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data import visualize as vz
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from numpy.linalg import norm
from pprint import pprint as pp
from itertools import combinations, product
import os
from matplotlib import colors
import json
from glob import glob
from scipy.stats import zscore
import pingouin as pg
from statsmodels.tsa.stattools import adfuller

class NumpyTypeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):  # Handle NumPy scalar types
                return obj.item()
            return json.JSONEncoder.default(self, obj)

class spike_time_converter:
    def __init__(self, spike_obj, max_time = 7000): 
        if isinstance(spike_obj, list):
            self.spike_times = spike_obj
        elif isinstance(spike_obj, np.ndarray):
            self.spike_array = spike_obj

        if hasattr(self, 'spike_array'):
            self.spike_times = np.where(self.spike_array) 
        else:
            self.spike_array = self.convert_to_array(max_time)

    def convert_to_array(self, max_time):
        num_units = len(self.spike_times)
        spike_array = np.zeros((num_units, max_time), dtype=bool)
        for unit_idx, unit_spikes in enumerate(self.spike_times):
            spike_array[unit_idx, unit_spikes] = True
        return spike_array

def calc_chunk_template_dynamics2(
        chunk_data,
        template,
        ):
    """Calculate the dynamics of all units in a chunk with respect to a template.
    Args:
        chunk_data (np.ndarray): Shape (num_trials, num_units, num_time_bins)
        template (np.ndarray): Shape (num_states, num_time_bins)
    Returns:
        estim_weights (np.ndarray): Shape (num_units, num_states)
        projected_firing (np.ndarray): Shape (num_trials, num_units, num_time_bins)
        template_similarity (float): Correlation between original and recovered template.
    """
    long_chunk = np.concatenate(chunk_data, axis=1)
    long_template = np.tile(template, (1, chunk_data.shape[0]))
    # Make sure template is 0-mean and normed
    long_template -= long_template.mean(axis=-1)[:,None]
    long_template /= norm(long_template, axis=-1)[:,None]
    # Make sure long_chunk is normed
    long_chunk /= norm(long_chunk, axis=-1)[:,None]
    similarity = long_chunk.dot(long_template.T)

    abs_sim = np.abs(similarity)
    max_abs_sim_ind = np.argmax(abs_sim,axis=0)
    max_abs_sim = np.max(abs_sim, axis=0)

    # Recreate long_template from projection
    recov_template = similarity.T.dot(long_chunk) 
    recov_template_normed = recov_template / norm(recov_template,axis=1)[:,None] 

    template_self_sim = recov_template_normed.dot(long_template.T) 
    recov_similarity = np.diag(template_self_sim)

    return max_abs_sim, similarity, recov_similarity, recov_template_normed


load_artifacts_bool = True
artifacts_dir = '/media/bigdata/firing_space_plot/firing_analyses/intra_state_dynamics/artifacts'
# artifacts_dir = '/home/abuzarmahmood/projects/firing_space_plot/firing_analyses/intra_state_dynamics/artifacts'

if not load_artifacts_bool:
    data_dir = '/media/storage/NM_resorted_data/laser_2500ms/NM51_2500ms_161030_130155'

    this_data = ephys_data.ephys_data(data_dir)
    this_data.firing_rate_params = this_data.default_firing_params
    this_data.firing_rate_params['window_size'] = 250
    this_data.firing_rate_params['step_size'] = 10
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
            dat.firing_rate_params = dat.default_firing_params
            dat.firing_rate_params['window_size'] = 250
            dat.firing_rate_params['step_size'] = 10
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

    # Save
    spike_time_lists = [spike_time_converter(spikes).spike_times for spikes in spike_list]
    firing_time_vector = this_data.time_vector

    np.savez(
            os.path.join(artifacts_dir, 'loaded_firing_data.npz'),
            paths = np.array(loaded_paths),
            spikes = np.array(spike_time_lists, dtype=object), 
            firing_rates = np.array(firing_rate_list, dtype=object),
            firing_time_vector = firing_time_vector 
            )

else:
    # Load
    loaded_artifacts = np.load(
            os.path.join(artifacts_dir, 'loaded_firing_data.npz'),
            allow_pickle=True
            )
    loaded_paths = loaded_artifacts['paths'].tolist()
    spike_list = loaded_artifacts['spikes'].tolist()
    firing_rate_list = loaded_artifacts['firing_rates'].tolist()
    firing_time_vector = loaded_artifacts['firing_time_vector']


time_lims = [-500, 2000]
firing_time_inds = np.where((firing_time_vector >= time_lims[0]) & (firing_time_vector <= time_lims[1]))[0]

plot_dir = os.path.expanduser('~/Desktop/template_dynamics/population_dynamics_plots')
os.makedirs(plot_dir, exist_ok=True)

flat_firing = [x for sublist in firing_rate_list for x in sublist]
session_nums = np.concatenate([np.ones(len(x))*i for i, x in enumerate(firing_rate_list)])

max_sims_list = []
recov_sim_list = []
unit_firing_list = []
unit_sim_list = []
for chunk_idx in trange(len(flat_firing)):
    chunk_data = flat_firing[chunk_idx][:, :, firing_time_inds]
    
    max_sims, sim_mat, recov_sim, recov_template = calc_chunk_template_dynamics2(chunk_data, template)
    max_sims_list.append(max_sims)
    recov_sim_list.append(recov_sim)

    max_abs_sim_inds = np.argmax(np.abs(sim_mat),axis=0)

    for unit_ind, (unit_firing, unit_sims) in enumerate(zip(chunk_data.swapaxes(0,1), sim_mat)):
        unit_firing_list.append(unit_firing)
        unit_sim_list.append(unit_sims)

    fig, ax = vz.firing_overview(chunk_data.swapaxes(0,1), figsize=(20,12))
    for ax_ind, this_ax in enumerate(ax.flatten()):
        if not ax_ind < len(sim_mat):
            continue
        this_ax.set_title(np.round(np.abs(sim_mat[ax_ind]),2))
    fig.suptitle(f"Reconstruction similarity: {np.round(recov_sim,2)}")
    fig.savefig(os.path.join(plot_dir, f'chunk_{chunk_idx}_firing_rates.png'))
    plt.close(fig)

    fig,ax = plt.subplots(2,1, sharex=True)
    ax[0].imshow(np.tile(template, (1, len(chunk_data))), aspect='auto', interpolation='nearest')
    ax[1].imshow(recov_template, aspect='auto', interpolation='nearest')
    fig.suptitle(f"Reconstruction similarity: {np.round(recov_sim,2)}")
    fig.savefig(os.path.join(plot_dir, f'chunk_{chunk_idx}_template_reconstruction.png'))
    plt.close(fig)

    # Plot template, mean and std of recovered template
    norm_recov_sim = norm(recov_sim)
    recov_template_trials = recov_template.reshape(
            recov_template.shape[0],
            chunk_data.shape[0],
            -1
            )
    mean_recov_template = recov_template_trials.mean(axis=1)
    std_recov_template = recov_template_trials.std(axis=1)
    fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
    ax[0].imshow(template, aspect='auto', interpolation='nearest')
    ax[0].set_title('Original Template')
    ax[1].imshow(mean_recov_template, aspect='auto', interpolation='nearest')
    ax[1].set_title('Mean Recovered Template across Trials')
    ax[2].imshow(std_recov_template, aspect='auto', interpolation='nearest')
    ax[2].set_title('STD of Recovered Template across Trials')
    fig.suptitle(f'Norm Recov Sim: {norm_recov_sim:.2}')
    fig.savefig(os.path.join(plot_dir, f'chunk_{chunk_idx}_template_recovery_stats.png'))
    plt.close(fig)


##############################
# Relationship between mean firing rate, response stability, and max dynamicity
mean_firing_rate = [x.mean(axis=None) for x in unit_firing_list]
max_sim_list = [np.max(x) for x in unit_sim_list]
trial_pca = [PCA(1).fit_transform(x) for x in unit_firing_list]

stable_list = [adfuller(x.flatten()+(np.random.randn(len(x))*1e-3))[1] for x in trial_pca]

fig, ax = plt.subplots(1,3)
sc = ax[0].scatter(mean_firing_rate, max_sim_list, c=stable_list, cmap='viridis', norm=colors.LogNorm(),
                   alpha = 0.3)
ax[0].set_xlabel('Mean Firing Rate')
ax[0].set_ylabel('Max Similarity to Template')
ax[1].scatter(mean_firing_rate, stable_list, c=max_sim_list, cmap='plasma', alpha=0.3)
ax[1].set_xlabel('Mean Firing Rate')
ax[1].set_ylabel('ADF p-value (Stability)')
ax[2].scatter(stable_list, max_sim_list, c=mean_firing_rate, cmap='cividis', alpha=0.3)
ax[2].set_xlabel('ADF p-value (Stability) - Low p-values indicate stability')
ax[2].set_ylabel('Max Similarity to Template')
plt.show()

# Plot units sorted by their similarity for a template
unit_plot_dir = os.path.expanduser('~/Desktop/template_dynamics/population_dynamics_plots/unit_sim_sorted')
os.makedirs(unit_plot_dir, exist_ok=True)

mean_unit_firing = np.stack([x.mean(axis=0) for x in unit_firing_list])
recov_sim_array = np.stack(recov_sim_list)

for i in range(recov_sim_array.shape[1]):
    this_sim = recov_sim_array[:,i]
    sim_sort_inds = np.argsort(this_sim)[::-1] 
    sorted_firing = [unit_firing_list[ind] for ind in sim_sort_inds]
    fig,ax = vz.gen_square_subplots(len(sorted_firing), figsize=(12,12))
    for this_dat, this_ax in zip(sorted_firing, ax.flatten()):
        this_ax.imshow(this_dat, aspect='auto', interpolation='None', cmap='jet')
    fig.savefig(os.path.join(unit_plot_dir, f'template_{i}_sorted_firing.png'))
    plt.close(fig)

# Plot session with max and mean reconstruction similarity
norm_max_sims = [norm(x) for x in max_sims_list]
norm_recov_sims = [norm(x) for x in recov_sim_list]

fig, ax = plt.subplots()
ax.scatter(norm_recov_sims, norm_max_sims)
ax.set_xlabel('Norm of Reconstruction Similarities')
ax.set_ylabel('Norm of Max Similarities')
ax.set_title('Norm of Similarities Across Sessions')
ax.set_aspect('equal', 'box')
fig.savefig(os.path.join(plot_dir, 'norm_recov_vs_max_similarities.png'))
plt.close(fig)
# plt.show()

# Overlay with axvspan for session_inds
fig, ax = plt.subplots()
for this_num in np.unique(session_nums):
    wanted_inds = np.where(session_nums == this_num)[0]
    ax.axvspan(
            wanted_inds[0],
            wanted_inds[-1],
            alpha=0.2,
            color='C{}'.format(int(this_num) % 10)
            )
    ax.plot(
        wanted_inds,
        np.array(norm_max_sims)[wanted_inds],
        '-o',
        )
ax.set_xlabel('Session Index')
ax.set_ylabel('Norm of Max Similarities')
ax.set_title('Norm of Max Similarities Across Sessions')
fig.savefig(os.path.join(plot_dir, 'norm_max_similarities_across_sessions.png'))
plt.close(fig)
# plt.show()

plt.hist(norm_max_sims, bins=30)
plt.xlabel('Norm of Max Similarities')
plt.ylabel('Count')
plt.title('Distribution of Norms of Max Similarities Across Sessions')
plt.show()

max_session_ind = np.nanargmax(norm_max_sims)
min_session_ind = np.nanargmin(norm_max_sims)

for session_label, session_ind in zip(
        ['best', 'worst'],
        [max_session_ind, min_session_ind]
        ):
    chunk_data = flat_firing[session_ind][:, :, firing_time_inds]
    
    max_sims, sim_mat, recov_sim, recov_template = calc_chunk_template_dynamics2(chunk_data, template)


    max_abs_sim_inds = np.argmax(np.abs(sim_mat),axis=0)

    fig, ax = vz.firing_overview(chunk_data.swapaxes(0,1), figsize=(20,12))
    for ax_ind, this_ax in enumerate(ax.flatten()):
        if not ax_ind < len(sim_mat):
            continue
        this_ax.set_title(np.round(np.abs(sim_mat[ax_ind]),2))
    fig.suptitle(f"Reconstruction similarity: {np.round(recov_sim,2)}")
    fig.savefig(os.path.join(plot_dir, f'session_{session_label}_firing_rates.png'))
    plt.close(fig)

    fig,ax = plt.subplots(2,1, sharex=True)
    ax[0].imshow(np.tile(template, (1, len(chunk_data))), aspect='auto', interpolation='nearest')
    ax[1].imshow(recov_template, aspect='auto', interpolation='nearest')
    fig.suptitle(f"Reconstruction similarity: {np.round(recov_sim,2)}")
    fig.savefig(os.path.join(plot_dir, f'session_{session_label}_template_reconstruction.png'))
    plt.close(fig)

    # Plot template, mean and std of recovered template
    recov_template_trials = recov_template.reshape(
            recov_template.shape[0],
            chunk_data.shape[0],
            -1
            )
    mean_recov_template = recov_template_trials.mean(axis=1)
    std_recov_template = recov_template_trials.std(axis=1)
    fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
    ax[0].imshow(template, aspect='auto', interpolation='nearest')
    ax[0].set_title('Original Template')
    ax[1].imshow(mean_recov_template, aspect='auto', interpolation='nearest')
    ax[1].set_title('Mean Recovered Template across Trials')
    ax[2].imshow(std_recov_template, aspect='auto', interpolation='nearest')
    ax[2].set_title('STD of Recovered Template across Trials')
    fig.savefig(os.path.join(plot_dir, f'session_{session_label}_template_recovery_stats.png'))
    plt.close(fig)


# Plot all datasets' mean firing rates
# for this_firing in firing_rate_list:
#     plot_firing = this_firing[..., firing_time_inds]
#     vz.firing_overview(np.concatenate(plot_firing, axis=0).swapaxes(0,1))
# plt.show()

#
# for i, this_firing in enumerate(firing_rate_list):
#     # fig, ax = plt.subplots(len(this_firing),1, figsize=(10,15), sharex=True)
#     stacked_firing = np.concatenate(this_firing, axis=0)  # Shape: (total_trials, num_neurons, num_time_bins)
#     fig, ax = vz.firing_overview(
#             stacked_firing.swapaxes(0,1)[..., firing_time_inds]  # Shape: (num_neurons, total_trials, selected_time_bins)
#             # stacked_firing.swapaxes(0,1)
#             )
#     this_base_name = loaded_paths[i].split('/')[-1]
#     fig.suptitle(this_base_name)
#     plt.show()
#     # plt.savefig(os.path.join(plot_dir, f'{this_base_name}_mean_firing_rates.png'))
#     # plt.close()

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
    this_trials = wanted_taste_firing[i[0]:i[1], :, :][..., firing_time_inds]
    vz.firing_overview(this_trials.swapaxes(0,1))
plt.show()


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

# Check orthonormality of basis functions
dot_product = basis_funcs.dot(basis_funcs.T)
print("Dot Product Matrix of Basis Functions (Should be close to Identity):")
pp(dot_product)

##############################

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
# Loop over all units and all chunks and remove data incrementally based on template similarity
# Check population similarity after each removal, not just single unit


for chunk_idx in range(len(trial_chunks)):
    chunk_data = wanted_taste_firing[:, :, firing_time_inds][
            trial_chunks[chunk_idx][0]:trial_chunks[chunk_idx][1]
            ]
    
    # vz.firing_overview(chunk_data.swapaxes(0,1));plt.show()

    max_sims, sim_mat, recov_sim, recov_template = calc_chunk_template_dynamics2(chunk_data, template)

    # # If sim_mat has any nans, drop those units
    # nan_inds = np.unique(np.where(np.isnan(sim_mat))[0])
    #
    # sim_mat = np.delete(sim_mat, nan_inds, axis=0)

    max_abs_sim_inds = np.argmax(np.abs(sim_mat),axis=0)
    mean_sims = np.abs(sim_mat).mean(axis=0)

    # # Boxplot for abs_sims
    # abs_sim = np.abs(sim_mat)
    # inds = np.array(list(np.ndindex(abs_sim.shape)))
    # # plt.boxplot(inds[:,1], abs_sim.flatten())
    # # plt.show()
    # plt.boxplot(abs_sim)

    fig, ax = vz.firing_overview(chunk_data.swapaxes(0,1))
    for ax_ind, this_ax in enumerate(ax.flatten()):
        if not ax_ind < len(sim_mat):
            continue
        this_ax.set_title(np.round(np.abs(sim_mat[ax_ind]),2))
    fig.suptitle(f"Reconstruction similarity: {np.round(recov_sim,2)}")

    fig,ax = plt.subplots(2,1, sharex=True)
    ax[0].imshow(np.tile(template, (1, len(chunk_data))), aspect='auto', interpolation='nearest')
    ax[1].imshow(recov_template, aspect='auto', interpolation='nearest')
    fig.suptitle(f"Reconstruction similarity: {np.round(recov_sim,2)}")

plt.show()

def calc_chunk_template_dynamics(
        chunk_data,
        template,
        ):
    """Calculate the dynamics of all units in a chunk with respect to a template.
    Args:
        chunk_data (np.ndarray): Shape (num_trials, num_units, num_time_bins)
        template (np.ndarray): Shape (num_states, num_time_bins)
    Returns:
        estim_weights (np.ndarray): Shape (num_units, num_states)
        projected_firing (np.ndarray): Shape (num_trials, num_units, num_time_bins)
        template_similarity (float): Correlation between original and recovered template.
    """
    long_chunk = np.concatenate(chunk_data, axis=(1))
    long_template = np.tile(template, (1, chunk_data.shape[0]))
    estim_weights = estimate_weights(
            long_chunk,
            long_template
            )
    projected_firing = estim_weights.dot(long_template)
    recov_template = np.linalg.pinv(estim_weights).dot(long_chunk)

    # fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    # ax[0].imshow(long_chunk, aspect='auto',interpolation='none')
    # ax[0].set_title('Long Chunk Firing Data')
    # ax[1].imshow(projected_firing, aspect='auto',interpolation='none')
    # ax[1].set_title('Projected Firing Data from Estimated Weights')
    # plt.show()


    # # Check dot similarity
    # dot_similarity = np.dot(
    #     long_template,
    #     recov_template.T,
    #     ) / (
    #             norm(long_template, axis=0)[:, np.newaxis] *
    #             norm(recov_template, axis=0)[np.newaxis, :]
    #             )
    # 
    # plt.matshow(dot_similarity, cmap='viridis')
    # plt.title('Dot Product Similarity between Original and Recovered Templates')
    # plt.colorbar(label='Dot Product Similarity')
    # plt.show()

    # Check recovered template per trial
    recov_template_trials = recov_template.reshape(
            recov_template.shape[0],
            chunk_data.shape[0],
            -1
            )

    # vz.firing_overview(recov_template_trials.swapaxes(0,1), cmap='viridis')
    # plt.show()

    mean_recov_template = recov_template.reshape(
            recov_template.shape[0],
            chunk_data.shape[0],
            -1
            ).mean(axis=1)

    # fig, ax = plt.subplots(2,2, figsize=(10,6), sharex='col', sharey='row')
    # ax[0,0].imshow(long_template, aspect='auto',interpolation='none')
    # ax[0,0].set_title('Long Template')
    # ax[1,0].imshow(recov_template, aspect='auto',interpolation='none')
    # ax[1,0].set_title('Recovered Template from Chunk Data')
    # ax[0,1].imshow(template, aspect='auto',interpolation='none')
    # ax[0,1].set_title('Original Template')
    # ax[1,1].imshow(mean_recov_template, aspect='auto',interpolation='none')
    # ax[1,1].set_title('Mean Recovered Template across Trials')
    # plt.show()
   
    # template_similarity = np.corrcoef(
    #         long_template.flatten(),
    #         recov_template.flatten()
    #         )[0,1]

    all_template_similarity = [
            np.corrcoef(
                template.flatten(),
                this_recov_template.flatten() 
                )[0,1]
            for this_recov_template in recov_template_trials.swapaxes(0,1)
            ]

    # Also calculate reconstruction accuracy as R2 for the firing
    ss_total = np.sum((long_template - np.mean(long_template))**2)
    ss_residual = np.sum((long_template - recov_template)**2)
    r_squared = 1 - (ss_residual / ss_total)
    # ss_total = np.sum((long_chunk - np.mean(long_chunk))**2)
    # ss_residual = np.sum((long_chunk - projected_firing)**2)
    # r_squared = 1 - (ss_residual / ss_total)

    # min_val = min(long_template.min(), recov_template.min())
    # max_val = max(long_template.max(), recov_template.max())
    # plt.scatter(
    #         long_template.flatten(),
    #         recov_template.flatten(),
    #         alpha=0.1
    #         )
    # plt.plot(
    #         [min_val, max_val],
    #         [min_val, max_val],
    #         color='r',
    #         linestyle='--',
    #         label='Unity Line'
    #         )
    # plt.show()
    # 

    return estim_weights, np.mean(all_template_similarity), all_template_similarity, recov_template_trials, projected_firing, r_squared

# Chunk by single trials
trial_chunks = [(i, i+1) for i in range(wanted_taste_firing.shape[0])]

chunk_dynamics = []
var_similarities = []
all_recov_templates = []
all_r_squared = []
all_estim_weights = []
for chunk_idx in range(len(trial_chunks)):
    chunk_data = wanted_taste_firing[:, :, :][
            trial_chunks[chunk_idx][0]:trial_chunks[chunk_idx][1]
            ]
    chunk_data = chunk_data[..., firing_time_inds]
    (
            estim_weights, 
            template_similarity, 
            all_similarities,
            recov_template_trials,
            projected_firing,
            r_squared
                )= calc_chunk_template_dynamics(
            chunk_data,
            down_template
            )
    chunk_dynamics.append({
        chunk_idx: template_similarity
        })
    var_similarities.append({
        chunk_idx: np.var(all_similarities)
        })
    all_recov_templates.append(recov_template_trials)
    all_r_squared.append(r_squared)
    all_estim_weights.append(estim_weights)

test_data = chunk_data[0]
norm_chunk_data = test_data / norm(test_data,axis=1)[:,None] 
this_template_similarity = down_template.dot(norm_chunk_data.T)

plt.matshow(np.abs(this_template_similarity).T)
plt.colorbar()
plt.show()

vz.imshow(norm_chunk_data);plt.show()

max_abs_similarity = np.max(np.abs(this_template_similarity).T,axis=0)

# Plot all estim weights
fig, ax = vz.gen_square_subplots(len(all_estim_weights), figsize=(12,12))
for i in range(len(all_estim_weights)):
    ax.flatten()[i].imshow(all_estim_weights[i], aspect='auto', interpolation='none')
    ax.flatten()[i].set_title(f'Chunk {i} Estim Weights')
plt.show()

r2_thresh = 0.1
plt.plot(all_r_squared, '-o')
plt.axhline(r2_thresh, color='r', linestyle='--')
plt.show()

wanted_r2_inds = np.array(all_r_squared) > r2_thresh

vz.firing_overview(wanted_taste_firing.swapaxes(0,1)[..., firing_time_inds][:, wanted_r2_inds, :], cmap='viridis')
plt.show()

cat_recov_templates = np.concatenate(all_recov_templates, axis=1)

fig, ax = vz.gen_square_subplots(cat_recov_templates.shape[1])
for this_ax, this_dat in zip(ax.flatten(), cat_recov_templates.swapaxes(0,1)):
    im = this_ax.imshow(this_dat, interpolation='nearest', aspect = 'auto',
                        vmin = vmin, vmax = vmax)
plt.show()


##############################
# See if dynamics can be recovered from white noise
noise_data = np.random.rand(
        1,
        wanted_taste_firing.shape[1],
        len(firing_time_inds)
        )
# Smooth noise data with same params as firing rates
# kernel = np.ones(this_data.firing_rate_params['window_size']) / this_data.firing_rate_params['window_size']
kernel = np.ones(25)
kernel = kernel / norm(kernel,1)
noise_data = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=2,
        arr=noise_data
        )

data_with_bias = np.concatenate(
        [
            noise_data,
            np.ones(noise_data.shape[2])[None,None,:]
            ],
        axis=1,
        )


(
        estim_weights, 
        template_similarity, 
        all_similarities,
        recov_template_trials,
        projected_firing,
        r_squared
            )= calc_chunk_template_dynamics(
        data_with_bias,
        down_template
        )

plt.matshow(estim_weights, aspect='auto', interpolation='nearest')

fig,ax = plt.subplots(2,1)
ax[0].imshow(noise_data[0], aspect='auto', interpolation='nearest')
ax[1].imshow(projected_firing, aspect='auto', interpolation='nearest')

fig,ax = plt.subplots(2,1)
ax[0].imshow(down_template, aspect='auto', interpolation='nearest')
ax[1].imshow(recov_template_trials[:,0], aspect='auto', interpolation='nearest')
plt.show()

# Calculate prinicpal components of data
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(noise_data[0])
explained_variance = pca.explained_variance_ratio_
pcs = pca.components_

plt.imshow(pcs, aspect='auto', interpolation='none')
plt.title('Principal Components of White Noise Data')
plt.show()

# from sklearn.metrics.pairwise import cosine_similarity

# norm_template = down_template / norm(down_template, axis=1)[:, np.newaxis]

# Have template such that the bumps are positive and everything else is negative
# template_sums = down_template.sum(axis=1)
# zero_lens = np.sum(down_template == 0,axis=1)
# neg_vals = template_sums / zero_lens
# pos_neg_template = down_template.copy()
# for state_idx in range(down_template.shape[0]):
#     pos_neg_template[state_idx, down_template[state_idx,:] == 0] = -neg_vals[state_idx]

pos_neg_template = down_template.copy()
# Subtract mean to make positive and negative parts
pos_neg_template = pos_neg_template - pos_neg_template.mean(axis=1)[:, np.newaxis]

norm_pos_neg_template = pos_neg_template / norm(pos_neg_template, axis=1)[:, np.newaxis]

# Check orthonormality
norm_pos_neg_template.dot(norm_pos_neg_template.T)

template_pc_similarity = np.dot(pcs, norm_pos_neg_template.T)
scaled_template_pc_similarity = template_pc_similarity * explained_variance[:, np.newaxis]

fig, ax = plt.subplots(1,5, figsize=(10,6)) 
ax[0].matshow(np.abs(template_pc_similarity), cmap='viridis')
plt.colorbar(ax=ax[0], label='Dot Product Similarity')
ax[1].imshow(explained_variance[:, np.newaxis], aspect='auto', cmap='viridis')
ax[2].imshow(pcs, aspect='auto', cmap='viridis', interpolation='none')
ax[3].imshow(norm_pos_neg_template, aspect='auto', cmap='viridis', interpolation='none')
ax[4].matshow(np.abs(scaled_template_pc_similarity), cmap='viridis')
fig.suptitle('Dot Product Similarity between PCs and Template')
plt.show()

plt.matshow(estim_weights, cmap='viridis')
plt.show()

plt.plot(norm_pos_neg_template.T)
plt.show()

fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax[0].imshow(noise_data[0], aspect='auto', interpolation='none')
ax[0].set_title('White Noise Data')
ax[1].imshow(recov_template_trials[:,0], aspect='auto', interpolation='none')
ax[1].set_title('Recovered Template from White Noise Data')
fig.suptitle(f'R² of Template Reconstruction from White Noise: {r_squared:.4f}')
plt.show()

abs_max_template_similarity = np.max(np.abs(scaled_template_pc_similarity), axis=0)
similarity_norm = norm(abs_max_template_similarity)

abs_sum_template_similarity = np.sum(np.abs(scaled_template_pc_similarity), axis=0)

# Do find how much of the population activity can be explained by a single template
# We project the data onto the template

def template_pca_similarity(
        data_array,
        template,
        ):
    """
    Calculate template similarity weighted by variance explained of principal component

    data_array = 3D numpy array
        - shape: trials, neurons, time
    template = 2D numpy array
        - shape: "states", time
    """
    # Make sure template is 0-mean and normed
    template -= template.mean(axis=1)[:, None]
    # template /= norm(template, axis=1)[:, None]

    # Perform PCA on long data
    n_trials = len(data_array)
    long_data = data_array.swapaxes(0,1).reshape((data_array.shape[1], -1))
    long_template = np.tile(template, (1, n_trials)) 
    long_template /= norm(long_template, axis=1)[:,None]

    pca_obj = PCA().fit(long_data)
    pcs = pca_obj.components_
    explained_variance = pca_obj.explained_variance_ratio_

    template_pc_similarity = np.dot(pcs, long_template.T)
    scaled_template_pc_similarity = template_pc_similarity * explained_variance[:, np.newaxis]

    abs_max_template_similarity = np.max(np.abs(scaled_template_pc_similarity), axis=0)
    similarity_norm = norm(abs_max_template_similarity)

##############################
# Perform pca on recovered templates
from sklearn.decomposition import PCA

recov_templates_long = cat_recov_templates.reshape(
        cat_recov_templates.shape[0],
        -1
        ).T  # Shape: (num_trials * num_time_bins, num_states)
vz.imshow(recov_templates_long.T) 
# plt.show()

pca = PCA(n_components=3)
pca.fit(recov_templates_long)
pca_scores = pca.transform(recov_templates_long)
print("Explained Variance Ratios of PCA Components:")
pp(pca.explained_variance_ratio_)
fig, ax = plt.subplots(3,1, figsize=(8,10))
for i in range(3):
    ax[i].plot(pca_scores[:, i])
    ax[i].set_title(f'PCA Component {i+1} Scores')
plt.show()

pca_scores_trials = pca_scores.reshape(
        cat_recov_templates.shape[1],
        -1,
        3
        )  # Shape: (num_trials, num_time_bins, num_components)
pca_scores_trials = np.moveaxis(pca_scores_trials, -1, 0)  # Shape: (num_components, num_trials, num_time_bins)

vz.firing_overview(pca_scores_trials.swapaxes(0,1), cmap='viridis')
plt.show()

# Plot all trajectories in 3d space
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for trial_idx in range(pca_scores_trials.shape[1]):
    ax.plot(
            pca_scores_trials[0, trial_idx, :],
            pca_scores_trials[1, trial_idx, :],
            pca_scores_trials[2, trial_idx, :],
            alpha=0.5
            )
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('Neural Trajectories in PCA Space of Recovered Templates')
plt.show()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for trial_idx in range(pca_scores_trials.shape[1]):
    ax.plot(
            cat_recov_templates[0, trial_idx, :],
            cat_recov_templates[1, trial_idx, :],
            cat_recov_templates[2, trial_idx, :],
            alpha=0.5
            )
plt.show()

vz.firing_overview(cat_recov_templates.swapaxes(0,1), cmap='viridis', cmap_lims='shared')
plt.show()


##############################

vz.firing_overview(cat_recov_templates.swapaxes(0,1)[wanted_r2_inds],
                   cmap_lims='shared', cmap='viridis')
plt.show()

vz.firing_overview(wanted_taste_firing.swapaxes(0,1)[..., firing_time_inds], cmap='viridis') 
fig, ax = vz.gen_square_subplots(len(cat_recov_templates), figsize=(12,12))
for i in range(len(cat_recov_templates)):
    ax.flatten()[i].plot(cat_recov_templates[i, :, :].T, alpha=0.3, color='gray')
vz.firing_overview(cat_recov_templates, cmap_lims='shared', cmap='jet')
plt.show()

# Write out recovered templates
if not load_artifacts_bool: 
    np.save(
            os.path.join(artifacts_dir, 'recovered_templates.npy'),
            cat_recov_templates
            )
else:
    cat_recov_templates = np.load(
            os.path.join(artifacts_dir, 'recovered_templates.npy')
            )

# all_mean_recov_templates = [
#         recov_templates.mean(axis=1)
#         for recov_templates in all_recov_templates
#         ]
#
# vz.firing_overview(np.stack(all_mean_recov_templates), cmap_lims='shared', cmap='viridis')
# plt.show()

# Check mean orthogonality of recovered templates for each trial
trial_mean_similarity = []
for trial_idx in range(cat_recov_templates.shape[1]):
    this_trial_templates = cat_recov_templates[:, trial_idx, :]
    norm_trial_templates = this_trial_templates / norm(this_trial_templates, axis=1)[:, np.newaxis]
    dot_similarity = np.dot(
        norm_trial_templates,
        norm_trial_templates.T,
        ) 
    off_diag_inds = np.triu_indices(dot_similarity.shape[0], k=1)
    mean_off_diag_abs_similarity = np.mean(np.abs(dot_similarity[off_diag_inds]))
    trial_mean_similarity.append(mean_off_diag_abs_similarity)

# Plot orthogonality of recovered templates against r2
plt.scatter(
        all_r_squared,
        trial_mean_similarity
        )
plt.xlabel('R² of Template Reconstruction')
plt.ylabel('Mean Off-Diagonal Template Similarity')
plt.title('Template Reconstruction R² vs Orthogonality')
plt.show()

fig, ax = vz.gen_square_subplots(cat_recov_templates.shape[1], figsize=(12,12))
for i in range(cat_recov_templates.shape[1]):
    ax.flatten()[i].plot(cat_recov_templates[:, i, :].T)
plt.show()

# Plot only for trials with high r2
fig, ax = vz.gen_square_subplots(np.sum(wanted_r2_inds), figsize=(12,12))
for ax_ind, i in enumerate(np.where(wanted_r2_inds)[0]):
    ax.flatten()[ax_ind].plot(cat_recov_templates[:, i, :].T)
plt.show()

# "Infer" states
inferred_states = np.argmax(cat_recov_templates, axis=0)

# get transition times by finding the first time in each trial when a state comes on
# First smooth states with a n-window median filter
from scipy.signal import medfilt, convolve
smoothed_states = medfilt(inferred_states, kernel_size=(1,15))
# kernel_size = 11
# kernel = np.ones(kernel_size) / kernel_size
# smoothed_states = np.apply_along_axis(
#         lambda m: convolve(m, kernel, mode='same'),
#         axis=1,
#         arr=inferred_states
#         )

fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
ax[0].imshow(inferred_states, aspect='auto', interpolation='none')
ax[0].set_title('Inferred States across Trials and Time')
ax[1].imshow(smoothed_states, aspect='auto', interpolation='none')
ax[1].set_title('Smoothed Inferred States across Trials and Time')
plt.show()

transition_times = []
for trial_idx in range(inferred_states.shape[0]):
    this_trial_states = smoothed_states[trial_idx, :]
    this_trial_transitions = [np.where(this_trial_states >= state)[0][0] for state in range(states)]
    transition_times.append(this_trial_transitions)

# Plot transition time histograms
transition_times = np.array(transition_times)
transition_times_scaled = transition_times * 25
bins = np.arange(0,2000,50)
for state_idx in range(states):
    plt.hist(transition_times_scaled[:, state_idx], 
             bins=bins, alpha=0.5, label=f'State {state_idx}')
plt.xlabel('Time Bins')
plt.ylabel('Count')
plt.title('Histogram of Transition Times per State')
plt.legend()
plt.show()

plt.imshow(inferred_states, aspect='auto', interpolation='none')
plt.title('Inferred States across Trials and Time')
plt.xlabel('Time Bins')
plt.ylabel('Trials')
plt.colorbar(label='Inferred State')
# Plot transition times
for trial_idx, this_trial_transitions in enumerate(transition_times):
    plt.scatter(
            this_trial_transitions,
            [trial_idx]*len(this_trial_transitions),
            color='r',
            marker='x'
            )
plt.show()

##############################
# Align neural activity to transition times 
dat_to_align = wanted_taste_firing[..., firing_time_inds]
window_radius = 15

# shape: states x units x trials x window*2
aligned_data = np.empty((states-1, dat_to_align.shape[1], dat_to_align.shape[0], window_radius*2)) 
for trial_idx in range(dat_to_align.shape[0]):
    trial_changes = transition_times[trial_idx]
    trial_firing = dat_to_align[trial_idx, :, :]
    # Drop first state
    trial_changes = trial_changes[1:]
    for change_idx, this_change in enumerate(trial_changes):
        pre_window = max(0, this_change - window_radius)
        post_window = min(trial_firing.shape[1], this_change + window_radius)
        pre_data = trial_firing[:, pre_window:this_change] 
        post_data = trial_firing[:, this_change:post_window]
        aligned_data[change_idx, :, trial_idx, window_radius - pre_data.shape[1]:window_radius] = pre_data 
        aligned_data[change_idx, :, trial_idx, window_radius:window_radius+post_data.shape[1]] = post_data


mean_transitions = np.mean(transition_times, axis=0).astype(int)[1:]
unaligned_data = np.empty(aligned_data.shape)
for state_idx in range(len(mean_transitions)):
    pre_data = dat_to_align[:, :, max(0, mean_transitions[state_idx] - window_radius):mean_transitions[state_idx]]
    post_data = dat_to_align[:, :, mean_transitions[state_idx]:min(dat_to_align.shape[2], mean_transitions[state_idx] + window_radius)]
    unaligned_data[state_idx, :, :, window_radius - pre_data.shape[2]:window_radius] = pre_data.swapaxes(0,1)
    unaligned_data[state_idx, :, :, window_radius:window_radius+post_data.shape[2]] = post_data.swapaxes(0,1)

align_plot_dir = os.path.expanduser('~/Desktop/template_dynamics_aligned/plots/transition_aligned/')
os.makedirs(align_plot_dir, exist_ok=True)

num_units = aligned_data.shape[1]
for this_unit in range(num_units):
    fig, ax = plt.subplots(2, states - 1, figsize=(15,5), sharey=True)
    for state_idx in range(states - 1):
        ax[0,state_idx].imshow(
                aligned_data[state_idx, this_unit, :, :],
                aspect='auto',
                interpolation='none'
                )
        ax[0,state_idx].set_title(f'State {state_idx+1} Transition Aligned - Unit {this_unit}')
        ax[1,state_idx].imshow(
                unaligned_data[state_idx, this_unit, :, :],
                aspect='auto',
                interpolation='none'
                )
    plt.savefig(os.path.join(align_plot_dir, f'unit_{this_unit}_aligned.png'))
    plt.close(fig)

mean_aligned_data = np.mean(aligned_data, axis=2)
mean_unaligned_data = np.mean(unaligned_data, axis=2)

# fig,ax = vz.firing_overview(mean_aligned_data, cmap='viridis')
# fig.suptitle('Mean Transition Aligned Data')
# fig,ax = vz.firing_overview(mean_unaligned_data, cmap='viridis')
# fig.suptitle('Mean Unaligned Data')
# plt.show()

fig, ax = plt.subplots(2, states - 1, figsize=(15,5), sharey=True)
for state_idx in range(states - 1):
    ax[0,state_idx].imshow(
            zscore(mean_aligned_data[state_idx, :, :], axis=1),
            aspect='auto',
            interpolation='none'
            )
    ax[0,state_idx].set_title(f'State {state_idx+1} Aligned') 
    ax[1,state_idx].imshow(
            zscore(mean_unaligned_data[state_idx, :, :], axis=1),
            aspect='auto',
            interpolation='none'
            )
    ax[1,state_idx].set_title(f'State {state_idx+1} Unaligned') 
    ax[0,state_idx].axvline(window_radius, color='r', linestyle='--')
    ax[1,state_idx].axvline(window_radius, color='r', linestyle='--')
plt.show()

##############################
# Repeat with alignment of projected templates
aligned_recov_templates = np.empty((states-1, cat_recov_templates.shape[0], cat_recov_templates.shape[1], window_radius*2))
for trial_idx in range(cat_recov_templates.shape[1]):
    trial_changes = transition_times[trial_idx]
    trial_templates = cat_recov_templates[:, trial_idx, :]
    # Drop first state
    trial_changes = trial_changes[1:]
    for change_idx, this_change in enumerate(trial_changes):
        pre_window = max(0, this_change - window_radius)
        post_window = min(trial_templates.shape[1], this_change + window_radius)
        pre_data = trial_templates[:, pre_window:this_change] 
        post_data = trial_templates[:, this_change:post_window]
        aligned_recov_templates[change_idx, :, trial_idx, window_radius - pre_data.shape[1]:window_radius] = pre_data 
        aligned_recov_templates[change_idx, :, trial_idx, window_radius:window_radius+post_data.shape[1]] = post_data
mean_aligned_recov_templates = np.mean(aligned_recov_templates, axis=2)

unaligned_recov_templates = np.empty(aligned_recov_templates.shape)
for state_idx in range(len(mean_transitions)):
    pre_data = cat_recov_templates[:, :, max(0, mean_transitions[state_idx] - window_radius):mean_transitions[state_idx]]
    post_data = cat_recov_templates[:, :, mean_transitions[state_idx]:min(cat_recov_templates.shape[2], mean_transitions[state_idx] + window_radius)]
    unaligned_recov_templates[state_idx, :, :, window_radius - pre_data.shape[2]:window_radius] = pre_data
    unaligned_recov_templates[state_idx, :, :, window_radius:window_radius+post_data.shape[2]] = post_data
mean_unaligned_recov_templates = np.mean(unaligned_recov_templates, axis=2)

plot_aligned_recov = aligned_recov_templates[:,:, wanted_r2_inds, :]
plot_unaligned_recov = unaligned_recov_templates[:,:, wanted_r2_inds, :]
plot_mean_aligned_recov = plot_aligned_recov.mean(axis=2)
plot_mean_unaligned_recov = plot_unaligned_recov.mean(axis=2)

fig, ax = plt.subplots(2, states - 1, figsize=(15,5), sharey=True)
for state_idx in range(states - 1):
    ax[0,state_idx].imshow(
            zscore(plot_mean_aligned_recov[state_idx, :, :], axis=1),
            aspect='auto',
            interpolation='none'
            )
    ax[0,state_idx].set_title(f'State {state_idx+1} Aligned Recovered Templates') 
    ax[1,state_idx].imshow(
            zscore(plot_mean_unaligned_recov[state_idx, :, :], axis=1),
            aspect='auto',
            interpolation='none'
            )
    ax[1,state_idx].set_title(f'State {state_idx+1} Unaligned Recovered Templates') 
    ax[0,state_idx].axvline(window_radius, color='r', linestyle='--')
    ax[1,state_idx].axvline(window_radius, color='r', linestyle='--')
plt.show()

# Plot overlay of aligned and unaligned recovered templates for each state
fig, ax = plt.subplots(states - 1,1, figsize=(8,12))
for state_idx in range(states - 1):
    ax[state_idx].plot(
            plot_mean_aligned_recov[state_idx, :, :].T,
            color='b',
            alpha=0.3,
            label='Aligned' if state_idx==0 else ""
            )
    ax[state_idx].plot(
            plot_mean_unaligned_recov[state_idx, :, :].T,
            color='g',
            alpha=0.3,
            label='Unaligned' if state_idx==0 else ""
            )
    ax[state_idx].set_title(f'State {state_idx+1} Recovered Templates')
    if state_idx==0:
        ax[state_idx].legend()
plt.show()



##############################

similarity_thresh = 0.15
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(trial_mean_similarity, '-o')
ax.axhline(similarity_thresh, color='r', linestyle='--', label='Similarity Threshold')
ax.set_xlabel('Trial Index')
ax.set_ylabel('Mean Off-Diagonal Template Similarity')
ax.set_title('Mean Template Similarity across Chunks per Trial')
plt.show()

wanted_similarity_inds = np.array(trial_mean_similarity) < similarity_thresh

orthogonal_dynamics_trials = cat_recov_templates[:, wanted_similarity_inds, :]

vz.firing_overview(orthogonal_dynamics_trials, cmap_lims='shared', cmap='jet')
vz.firing_overview(orthogonal_dynamics_trials.swapaxes(0,1), cmap_lims='shared', cmap='viridis')
plt.show()

fig, ax = vz.gen_square_subplots(orthogonal_dynamics_trials.shape[1], figsize=(12,12))
for i in range(orthogonal_dynamics_trials.shape[1]):
    ax.flatten()[i].plot(orthogonal_dynamics_trials[:, i, :].T)
plt.show()

# Find 


##############################

# Plot all chunks with their template similarities
chunk_similarity_scores = [value for d in chunk_dynamics for value in d.values()]
chunk_variance_scores = [value for d in var_similarities for value in d.values()]

for chunk_idx in range(len(trial_chunks)):
    chunk_data = wanted_taste_firing[:, :, :][
            trial_chunks[chunk_idx][0]:trial_chunks[chunk_idx][1]
            ]
    fig,ax = vz.firing_overview(chunk_data.swapaxes(0,1)[..., firing_time_inds])
    fig.suptitle(f'Chunk {chunk_idx} - Template Similarity: {chunk_similarity_scores[chunk_idx]:.3f}')
plt.show()

# Plot mean similarity vs variance
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.scatter(
        chunk_similarity_scores,
        chunk_variance_scores
        )
ax.set_xlabel('Template Similarity')
ax.set_ylabel('Variance of Template Similarity across Trials')
for i, (sim, var) in enumerate(zip(chunk_similarity_scores, chunk_variance_scores)):
    ax.text(sim, var, str(i))
plt.show()

# fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
# ax[0].imshow(long_chunk, aspect='auto', interpolation='none')
# ax[0].set_title('Long Chunk Firing Data')
# ax[1].imshow(projected_firing, aspect='auto', interpolation='none')
# ax[1].set_title('Projected Firing Data from Estimated Weights')
# plt.show()

############################################################
# Run loop over all chunks and units, removing the least dynamic units incrementally 

orig_data = wanted_taste_firing[..., firing_time_inds]
orig_trial_chunks = trial_chunks.copy()
num_units = orig_data.shape[1]
num_chunks = len(orig_trial_chunks)
orig_trial_inds = np.arange(orig_data.shape[0])
orig_nrn_inds = np.arange(orig_data.shape[1])


# plt.show()

removal_mode = 'weighted'
probabilistic_removal = True
bias_mode = 'unit'
bias_strength = 0.5
max_iters = 20
n_runs = 200
plot_bool = False

for this_run in range(n_runs):

    this_plot_dir = os.path.join(plot_dir, f'run_{this_run}')
    os.makedirs(this_plot_dir, exist_ok=True)

    if plot_bool:
        fig, ax = vz.firing_overview(orig_data.swapaxes(0,1))
        fig.suptitle('Original Data Before Any Removals')
        fig.savefig(os.path.join(this_plot_dir, 'iter_0.png'))
        plt.close()

    iteration = 1
    removal_list = []
    current_data = orig_data.copy()
    current_trial_chunks = orig_trial_chunks.copy()
    current_similarity = orig_similarity
    running_orig_trial_inds = orig_trial_inds.copy()
    running_orig_nrn_inds = orig_nrn_inds.copy()

    removal_list.append(
            {
                'orig_unit': np.array([]),  
                'orig_trials': np.array([]), 
                'current_similarity': current_similarity,
                'current_data_fraction': current_data.size / orig_data.size
            }
            )

    # Initialize array to hold similarity scores after each removal
    # Account for the fact that there should be a no-units-removed and no-chunks-removed case
    while iteration <= max_iters:
        try:
            similarity_rm_array = np.zeros((num_units+1, num_chunks+1)) * np.nan
            dataset_loss_array = np.zeros((num_units+1, num_chunks+1)) * np.nan
            num_chunks = len(current_trial_chunks)
            num_units = current_data.shape[1]

            for rm_unit_idx in [np.nan, *np.arange(num_units)]:
                for rm_chunk_idx in [np.nan, *np.arange(num_chunks)]:

                    if not np.isnan(rm_chunk_idx):
                        rm_chunk_bounds = current_trial_chunks[rm_chunk_idx]
                        rm_trial_inds = np.arange(
                                rm_chunk_bounds[0],
                                rm_chunk_bounds[1]
                                )
                        test_data = np.delete(
                                current_data,
                                rm_trial_inds,
                                axis=0
                                )
                    else:
                        test_data = current_data.copy()
                    # Remove unit
                    if not np.isnan(rm_unit_idx):
                        test_data = np.delete(
                                test_data,
                                int(rm_unit_idx),
                                axis=1
                                )

                    # Calculate dynamics for current data
                    estim_weights, template_similarity = calc_chunk_template_dynamics(
                            test_data,
                            down_template
                            )

                    array_inds = (
                            int(rm_unit_idx)+1 if not np.isnan(rm_unit_idx) else 0,
                            int(rm_chunk_idx)+1 if not np.isnan(rm_chunk_idx) else 0
                            )
                    similarity_rm_array[array_inds] = template_similarity

                    dataset_loss = 1 - (test_data.size / current_data.size)
                    dataset_loss_array[array_inds] = dataset_loss
                    # print((rm_unit_idx, rm_chunk_idx, dataset_loss, template_similarity))

            delta_similarity = similarity_rm_array - current_similarity

            improvement_loss_ratio = delta_similarity / dataset_loss_array
            # Set (0,0) entry to nan since no removal happened there which makes ratio meaningless
            improvement_loss_ratio[0,0] = np.nan

            if removal_mode == 'weighted':
                metric_array = improvement_loss_ratio
            elif removal_mode == 'max_delta':
                metric_array = delta_similarity
            if not probabilistic_removal:
                max_delta_ind = np.unravel_index(
                    np.nanargmax(metric_array),
                    delta_similarity.shape
                    )
            else:
                # Min-max scale metric array to [0,1]
                scaled_metric_array = (metric_array - np.nanmin(metric_array)) / (np.nanmax(metric_array) - np.nanmin(metric_array))
                non_nan_inds = np.where(~np.isnan(scaled_metric_array))
                non_nan_values = scaled_metric_array[non_nan_inds]
                sampled_value = np.random.choice(
                    non_nan_values,
                    size = 1,
                    p = non_nan_values / np.nansum(non_nan_values)
                    )
                max_delta_ind_raw = np.where(scaled_metric_array == sampled_value)
                max_delta_ind = (max_delta_ind_raw[0][0], max_delta_ind_raw[1][0])

            # Check for biasing
            if bias_mode == 'unit':
                # Bias towards removing units
                # Set chunk removal indices to nan with probability bias_strength
                if (not np.isnan(max_delta_ind[1])) and (np.random.rand() < bias_strength):
                    max_delta_ind = (max_delta_ind[0], 0)
            elif bias_mode == 'chunk':
                # Bias towards removing chunks
                # Set unit removal indices to nan with probability bias_strength
                if (not np.isnan(max_delta_ind[0])) and (np.random.rand() < bias_strength):
                    max_delta_ind = (0, max_delta_ind[1])

            # If both indices are zero (no removal), skip this iteration
            if max_delta_ind == (0,0):
                print("No removal selected, ending iterations.")
                break

            rm_nrn_ind = max_delta_ind[0]-1 if max_delta_ind[0] !=0 else None
            rm_chunk_ind = max_delta_ind[1]-1 if max_delta_ind[1] !=0 else None

            updated_trial_chunks = current_trial_chunks.copy()
            if rm_chunk_ind is not None:
                del updated_trial_chunks[rm_chunk_ind]
            # Make sure remaining chunks are contiguous
            contig_chunks = []
            start_trial = 0
            for chunk_bounds in updated_trial_chunks:
                num_trials_in_chunk = chunk_bounds[1] - chunk_bounds[0]
                contig_chunks.append((start_trial, start_trial + num_trials_in_chunk))
                start_trial += num_trials_in_chunk
            updated_trial_chunks = contig_chunks

            # Remove the identified unit and chunk from current data for next iteration
            if rm_chunk_ind is not None:
                rm_chunk_bounds = current_trial_chunks[rm_chunk_ind]

                rm_trial_inds = np.arange(
                        rm_chunk_bounds[0],
                        rm_chunk_bounds[1]
                        )
                test_data = np.delete(
                        current_data,
                        rm_trial_inds,
                        axis=0
                        )
                updated_running_orig_trial_inds = np.delete(
                        running_orig_trial_inds,
                        rm_trial_inds,
                        axis=0
                        )
            else:
                test_data = current_data.copy()
            # Remove unit
            if rm_nrn_ind is not None:
                test_data = np.delete(
                        test_data,
                        int(rm_nrn_ind),
                        axis=1
                        )
                updated_running_orig_nrn_inds = np.delete(
                        running_orig_nrn_inds,
                        int(rm_nrn_ind),
                        axis=0
                        )

            current_trial_chunks = updated_trial_chunks
            current_data = test_data.copy()
            current_similarity = similarity_rm_array[max_delta_ind]

            # divnorm1 = colors.TwoSlopeNorm(vmin=-np.nanmax(np.abs(delta_similarity)), vcenter=0, vmax=np.nanmax(np.abs(delta_similarity)))
            # divnorm2 = colors.TwoSlopeNorm(vmin=-np.nanmax(np.abs(improvement_loss_ratio)), vcenter=0, vmax=np.nanmax(np.abs(improvement_loss_ratio)))
            # 
            # fig, ax = plt.subplots(1,3, figsize=(15,5))
            # ax[0].matshow(delta_similarity, origin='lower', aspect='auto', cmap='bwr', 
            #             vmin=-np.nanmax(np.abs(delta_similarity)), vmax=np.nanmax(np.abs(delta_similarity)),
            #               norm=divnorm1)
            # ax[0].set_title('Delta Template Similarity after Unit/Chunk Removal')
            # plt.colorbar(ax[0].images[0], ax=ax[0], label='Delta Similarity')
            # ax[1].matshow(dataset_loss_array, origin='lower', aspect='auto', cmap='viridis')
            # ax[1].set_title('Dataset Loss after Unit/Chunk Removal')
            # plt.colorbar(ax[1].images[0], ax=ax[1], label='Dataset Loss')
            # ax[2].matshow(improvement_loss_ratio, origin='lower', aspect='auto', cmap='bwr',
            #               vmin=-np.nanmax(np.abs(improvement_loss_ratio)), vmax=np.nanmax(np.abs(improvement_loss_ratio)),
            #               norm=divnorm2)
            # ax[2].set_title('Improvement to Loss Ratio after Unit/Chunk Removal')
            # ax[2].scatter(*max_delta_ind[::-1], color='k', s=10, label='Max Ratio Point')
            # plt.colorbar(ax[2].images[0], ax=ax[2], label='Improvement/Loss Ratio')
            # plt.show()
            #
            removal_list.append(
                    {
                        'orig_unit': np.setdiff1d(running_orig_nrn_inds, updated_running_orig_nrn_inds),
                        'orig_trials': np.setdiff1d(running_orig_trial_inds, updated_running_orig_trial_inds),
                        'current_similarity': current_similarity,
                        'current_data_fraction': current_data.size / orig_data.size
                    }
                    )
            print(f'Removed orig unit: {removal_list[-1]["orig_unit"]}')
            running_orig_nrn_inds = updated_running_orig_nrn_inds
            print(f'Removed orig trials: {removal_list[-1]["orig_trials"]}')
            running_orig_trial_inds = updated_running_orig_trial_inds

            if plot_bool:
                fig, ax = vz.firing_overview(current_data.swapaxes(0,1))
                fig.suptitle(f'Data after Iteration {iteration} Removals')
                fig.savefig(os.path.join(this_plot_dir, f'iter_{iteration}.png'))
                plt.close()

            iteration += 1
        except Exception as e:
            print(f"Error during iteration {iteration}: {e}")
            break

    # Plot details of removal list
    # Plot as scatter of data_fraction vs similarity
    if plot_bool:
        frac_vector = [removal['current_data_fraction'] for removal in removal_list]
        similarity_vector = [removal['current_similarity'] for removal in removal_list]
        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.plot(
                similarity_vector,
                frac_vector,
                '-o'
                )
        for i, removal in enumerate(removal_list):
            ax.text(
                    removal['current_data_fraction'],
                    removal['current_similarity'],
                    f"Trials removed: {removal['orig_trials']}\nUnits removed: {removal['orig_unit']}",
                    fontsize=8,
                    transform=ax.transData,
                    )
        ax.set_ylabel('Fraction of Original Data Remaining')
        ax.set_xlabel('Template Similarity')
        ax.set_title('Template Similarity vs Data Fraction after Each Removal')
        # plt.show()
        fig.savefig(os.path.join(this_plot_dir, 'similarity_vs_data_fraction.png'))
        plt.close()

    # Write out removal list to text file
    removal_list_path = os.path.join(this_plot_dir, 'removal_list.json')
    with open(removal_list_path, 'w') as f:
        json.dump(removal_list, f, indent=4, cls=NumpyTypeEncoder)


############################################################
# Load all removal lists and plot summary
removal_summary_dir = plot_dir
all_removal_lists = glob(os.path.join(removal_summary_dir, 'run_*', 'removal_list.json'))
summary_data = []
for removal_list_path in all_removal_lists:
    with open(removal_list_path, 'r') as f:
        removal_list = json.load(f)
        summary_data.append(removal_list)

# Plot all runs on single figure
fig, ax = plt.subplots(1,1, figsize=(6,4))
for run_idx, removal_list in enumerate(summary_data):
    frac_vector = [removal['current_data_fraction'] for removal in removal_list]
    similarity_vector = [removal['current_similarity'] for removal in removal_list]
    ax.plot(
            similarity_vector,
            frac_vector,
            '-o',
            label=f'Run {run_idx}',
            alpha=0.5
            )
ax.set_ylabel('Fraction of Original Data Remaining')
ax.set_xlabel('Template Similarity')
ax.set_title('Template Similarity vs Data Fraction after Each Removal')
plt.show()

