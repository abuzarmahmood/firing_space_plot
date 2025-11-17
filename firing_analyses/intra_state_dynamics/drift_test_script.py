from blech_clust.utils.ephys_data import ephys_data
from blech_clust.utils.ephys_data import visualize as vz
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from pprint import pprint as pp
from itertools import combinations, product
import os
from matplotlib import colors
import json
from glob import glob

class NumpyTypeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):  # Handle NumPy scalar types
                return obj.item()
            return json.JSONEncoder.default(self, obj)

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

# Save
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


spike_time_lists = [spike_time_converter(spikes).spike_times for spikes in spike_list]

artifacts_dir = '/media/bigdata/firing_space_plot/firing_analyses/intra_state_dynamics/artifacts'
np.savez(
        os.path.join(artifacts_dir, 'loaded_firing_data.npz'),
        paths = np.array(loaded_paths),
        spikes = np.array(spike_time_lists, dtype=object), 
        firing_rates = np.array(firing_rate_list, dtype=object)
        )


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
# Loop over all units and all chunks and remove data incrementally based on template similarity
# Check population similarity after each removal, not just single unit

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

    return estim_weights, np.mean(all_template_similarity), all_template_similarity, recov_template_trials

# Chunk by single trials
trial_chunks = [(i, i+1) for i in range(wanted_taste_firing.shape[0])]

chunk_dynamics = []
var_similarities = []
all_recov_templates = []
for chunk_idx in range(len(trial_chunks)):
    chunk_data = wanted_taste_firing[:, :, :][
            trial_chunks[chunk_idx][0]:trial_chunks[chunk_idx][1]
            ]
    chunk_data = chunk_data[..., firing_time_inds]
    (
            estim_weights, 
            template_similarity, 
            all_similarities,
            recov_template_trials
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

cat_recov_templates = np.concatenate(all_recov_templates, axis=1)

vz.firing_overview(cat_recov_templates.swapaxes(0,1), cmap_lims='shared', cmap='viridis')
plt.show()

vz.firing_overview(wanted_taste_firing.swapaxes(0,1)[..., firing_time_inds], cmap='viridis') 
fig, ax = vz.gen_square_subplots(len(cat_recov_templates), figsize=(12,12))
for i in range(len(cat_recov_templates)):
    ax.flatten()[i].plot(cat_recov_templates[i, :, :].T, alpha=0.3, color='gray')
vz.firing_overview(cat_recov_templates, cmap_lims='shared', cmap='jet')
plt.show()

all_mean_recov_templates = [
        recov_templates.mean(axis=1)
        for recov_templates in all_recov_templates
        ]

vz.firing_overview(np.stack(all_mean_recov_templates), cmap_lims='shared', cmap='viridis')
plt.show()

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

fig, ax = vz.gen_square_subplots(cat_recov_templates.shape[1], figsize=(12,12))
for i in range(cat_recov_templates.shape[1]):
    ax.flatten()[i].plot(cat_recov_templates[:, i, :].T)
plt.show()

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

plot_dir = os.path.expanduser('~/Desktop/template_dynamics_removals/')
os.makedirs(plot_dir, exist_ok=True)

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
