"""
How do we know the templates defined in the template_regression
are valid, and that there aren't better templates?

One way to test would be to do a random sampling test (maybe no enumeration over:
    1) Number of states
    2) State positions
And calculate reconstruction error for templates from data.
The parameters with the best reconstruction error should best resemble the
dynamics in the data

The sampling can run in parallel
"""

import numpy as np
from tqdm import tqdm, trange
from numpy.linalg import norm
import pandas as pd
import pylab as plt
#from scipy.stats import percentileofscore

############################################################
# Create sampler
#def get_rand_int(n_max):
#    return np.random.randint(0, high = n_max, size = size)
#
#def vec_rand_int(n_max):
#    return np.vectorize(get_rand_int)(n_max)

def return_transition_pos(
        n_states,
        max_len = 2000,
        min_state_dur = 50,
        n_samples = 1000
        ):
    #n_states = 4
    #max_len = 2000
    #min_state_dur = 50
    #n_samples = 1000 

    grid = np.arange(0, max_len, min_state_dur)
    # Iteratively select transitions
    n_transitions = n_states - 1
    # We can select next transition using randint,
    # Need to know min and max
    # First max needs to allow "n_transitions" more transitions

    #first_max = len(grid) - n_transitions
    #second_max = len(grid) - n_transitions + 1
    #third_max = len(grid) - n_transitions + 2
    #first_transition = np.random.randint(0, np.ones(n_samples)*first_max)
    #second_transition = np.random.randint(first_transition + 1, second_max) 
    #third_transition = np.random.randint(second_transition + 1, third_max) 

    trans_max_list = [len(grid) - n_transitions + i for i in range(n_transitions)]

    transition_ind_list = []
    for i in range(n_transitions):
        if i == 0:
            temp_trans = np.random.randint(0, np.ones(n_samples)*trans_max_list[i])
        else:
            temp_trans = np.random.randint(
                    transition_ind_list[i-1] + 1,
                    trans_max_list[i])
        transition_ind_list.append(temp_trans)

    transition_inds_array = np.stack(transition_ind_list).T
    transition_array = grid[transition_inds_array]
    return transition_array

# Convert transition positions to template vectors
def return_template_mat(trans_points, max_len):
    #trans_points = transition_list[3][0]
    #max_len = 2000
    trans_points_fin = [0, *trans_points.flatten(), max_len]
    state_lims = [(trans_points_fin[x], trans_points_fin[x+1]) \
                        for x in range(len(trans_points_fin) - 1)]
    template_mat = np.zeros((len(state_lims), max_len))
    for i, vals in enumerate(state_lims):
        template_mat[i, vals[0]:vals[1]] = 1
    return template_mat

############################################################
state_range = np.arange(2,10)
repeats = 1000
fin_state_range = np.repeat(state_range, repeats)
transition_list = [return_transition_pos(x, n_samples = 1) \
        for x in tqdm(fin_state_range)]
max_len = 2000
template_mat_list = [return_template_mat(x, max_len) \
        for x in tqdm(transition_list)]

############################################################
class template_projection():
    def __init__(
            self,
            data,
            templates):
        """
        data : nrns x trials x time
        templates : n_templates x time
        """
        self.data = data
        #self.templates = templates
        self.templates = templates / norm(templates,axis=-1)[:,np.newaxis] 
        # Elongate data for later use
        self.long_data = np.reshape(data, (len(data),-1)) 
        temp_long_templates = np.reshape(templates, (len(templates),-1)) 
        self.long_templates = np.tile(
                temp_long_templates,
                (1, data.shape[1])
                ) 

    def estimate_weights(self):
        """Estimate weights for regression from data to templates
        """
        estim_weight = self.long_data.dot(np.linalg.pinv(self.long_templates))
        self.estim_weight = estim_weight

    def project_to_template(self):
        """Project data into template space
        """
        self.proj = np.linalg.pinv(self.estim_weight).dot(self.long_data)

    def reconstruct_data(self):
        """Reconstruct data from the template
        """
        self.reconstruction = self.estim_weight.dot(self.long_templates)

    def calculate_reconstruction_similarity(self):
        """Calculate raw and normalzied similarity for data reconstruction
        """
        data_norm = norm(self.long_data,axis=-1)[:,np.newaxis]
        reconstruction_norm = norm(self.reconstruction,axis=-1)[:,np.newaxis]
        # Numberator : Matrix of all to all comparison
        reconstruction_sim = np.diag(self.long_data.dot(self.reconstruction.T)) \
                / (data_norm * reconstruction_norm).flatten()
        self.recon_similarity_raw = np.round(norm(np.diag(reconstruction_sim)),4)
        self.recon_similarity_normalized = np.round(
                self.recon_similarity_raw / np.sqrt(len(self.data)),
                4)
    
    def calculate_projection_similarity(self):
        """Calculate similarity of template with data projected into template space
        """
        template_norm = norm(self.long_templates,axis=-1)[:,np.newaxis]
        proj_norm = norm(self.proj,axis=-1)[:,np.newaxis]
        # Numberator : Matrix of all to all comparison
        proj_sim = self.long_templates.dot(self.proj.T) \
                / (template_norm.dot(proj_norm.T))
        self.proj_similarity_raw = np.round(norm(np.diag(proj_sim)),4)
        self.proj_similarity_normalized = np.round(
                self.proj_similarity_raw / np.sqrt(len(self.templates)),
                4)

####################
# Test template projection
############################################################
## Simulation
############################################################
# Time-lims : 0-2000ms
# 4 States
states = 4
epoch_lims = [
        [0,200],
        [200,850],
        [850,1450],
        [1450,2000]
        ]
epoch_lens = np.array([np.abs(np.diff(x)[0]) for x in epoch_lims])
basis_funcs = np.stack([np.zeros(2000) for i in range(4)] )
for this_func, this_lims in zip(basis_funcs, epoch_lims):
    this_func[this_lims[0]:this_lims[1]] = 1
basis_funcs = basis_funcs / norm(basis_funcs,axis=-1)[:,np.newaxis] 

nrns = 10
trials = 15
sim_w = np.random.random(size = (nrns,states))
firing = np.matmul(sim_w, basis_funcs)*10
firing_array = np.tile(firing[:,np.newaxis], (1,trials,1))
firing_array = firing_array + np.random.randn(*firing_array.shape)*0.1

test = template_projection(
        data = firing_array,
        templates = basis_funcs)
test.estimate_weights()
test.project_to_template()
test.calculate_projection_similarity()
test.reconstruct_data()
test.calculate_reconstruction_similarity()
print(f'Proj similarity : {test.proj_similarity_normalized}')
print(f'Reconstruction similarity : {test.recon_similarity_normalized}')

############################################################
# Estimate states and positions for simulated data
all_normalized_proj_sims = []
all_normalized_recon_sims = []
for this_template in tqdm(template_mat_list):
    proj_handler = template_projection(
            data = firing_array,
            templates = this_template
            )
    proj_handler.estimate_weights()
    proj_handler.project_to_template()
    proj_handler.calculate_projection_similarity()
    proj_handler.reconstruct_data()
    proj_handler.calculate_reconstruction_similarity()
    all_normalized_proj_sims.append(proj_handler.proj_similarity_normalized)
    all_normalized_recon_sims.append(proj_handler.recon_similarity_normalized)

all_normalized_proj_sims = np.array(all_normalized_proj_sims)
all_normalized_recon_sims = np.array(all_normalized_recon_sims)

sample_frame = pd.DataFrame(
        dict(
            states = fin_state_range,
            transitions = transition_list,
            proj_similarity = all_normalized_proj_sims,
            recon_similarity = all_normalized_recon_sims,
            )
        )
sample_frame.dropna(inplace=True)
sample_frame['total_sim'] = sample_frame['proj_similarity'] + sample_frame['recon_similarity']

#mean_state_frame = sample_frame.groupby('states').mean()
#
#plt.plot(mean_state_frame.index, mean_state_frame.recon_similarity,
#        '-x', label = 'Recon Sim') 
#plt.plot(mean_state_frame.index, mean_state_frame.proj_similarity,
#        '-x', label = 'Proj Sim')
#plt.legend()
#plt.show()

# Check where top 5th percentile of similarities reside
critical_val = np.percentile(sample_frame.total_sim, 95)
pass_bool = sample_frame.total_sim >= critical_val

critical_frame = sample_frame[pass_bool]

np.histogram(critical_frame.states, bins = state_range)
