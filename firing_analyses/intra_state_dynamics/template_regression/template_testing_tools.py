import numpy as np
from tqdm import tqdm, trange
from numpy.linalg import norm
from joblib import Parallel, delayed, cpu_count

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def return_transition_pos(
        n_states,
        max_len = 2000,
        min_state_dur = 50,
        n_samples = 1000
        ):

    grid = np.arange(0, max_len, min_state_dur)
    # Iteratively select transitions
    n_transitions = n_states - 1
    # We can select next transition using randint,
    # Need to know min and max
    # First max needs to allow "n_transitions" more transitions

    #trans_max_list = [len(grid) - n_transitions + i for i in range(n_transitions)]
    trans_max_list = [len(grid) - (n_transitions - i)*min_state_dur \
            for i in range(n_transitions)]

    transition_ind_list = []
    for i in range(n_transitions):
        if i == 0:
            temp_trans = np.random.randint(min_state_dur, np.ones(n_samples)*trans_max_list[i])
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
    trans_points_fin = [0, *trans_points.flatten(), max_len]
    state_lims = [(trans_points_fin[x], trans_points_fin[x+1]) \
                        for x in range(len(trans_points_fin) - 1)]
    template_mat = np.zeros((len(state_lims), max_len))
    for i, vals in enumerate(state_lims):
        template_mat[i, vals[0]:vals[1]] = 1
    return template_mat



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

def return_similarities(this_template, firing_array):
    proj_handler = template_projection(
            data = firing_array,
            templates = this_template
            )
    proj_handler.estimate_weights()
    proj_handler.project_to_template()
    proj_handler.calculate_projection_similarity()
    proj_handler.reconstruct_data()
    proj_handler.calculate_reconstruction_similarity()
    return (proj_handler.proj_similarity_normalized,
            proj_handler.recon_similarity_normalized)
