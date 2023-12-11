import numpy as np
import statsmodels.api as sm

def build_design_matrix(spike_trains, history_order, coupling_order):
    """
    Build the design matrix for the Poisson GLM with history and coupling filters.
    
    Args:
    - spike_trains: numpy array of shape (neurons, time) representing spike trains.
    - history_order: int, order of the history filter.
    - coupling_order: int, order of the coupling filter.
    
    Returns:
    - design_matrix: numpy array of shape (time, neurons*(history_order + coupling_order + 1))
                     representing the design matrix.
    """
    neurons, time = spike_trains.shape
    design_matrix = np.zeros((time, neurons*(history_order + coupling_order + 1)))
    
    # History filter
    for t in range(history_order, time):
        for n in range(neurons):
            design_matrix[t, n] = np.sum(spike_trains[n, t-history_order:t])
    
    # Coupling filter
    for t in range(coupling_order, time):
        for n1 in range(neurons):
            for n2 in range(neurons):
                design_matrix[t, neurons*(history_order+1) + n1*neurons + n2] = spike_trains[n2, t-coupling_order:t].mean()
    
    return design_matrix

# Spike trains data (example)
neurons = 2
time = 1000



spike_trains = np.random.rand(neurons, time) < 0.05  # Assuming spike probability of 0.05

import matplotlib.pyplot as plt
plt.scatter(*np.where(spike_trains.T), s=1)
plt.show()

# Model parameters
history_order = 20
coupling_order = 10

# Build design matrix
design_matrix = build_design_matrix(spike_trains, history_order, coupling_order)

plt.imshow(design_matrix, aspect='auto')
plt.show()

# Target variable (spike counts)
y = spike_trains.sum(axis=0)

# Fit Poisson GLM
poisson_glm = sm.GLM(y, design_matrix, family=sm.families.Poisson())
poisson_results = poisson_glm.fit()

# Print model summary
print(poisson_results.summary())

############################################################
# Extract estimated coefficients
coefficients = poisson_results.params

# Extract filter dimensions
neurons = spike_trains.shape[0]
history_dim = neurons * (history_order + 1)
coupling_dim = neurons * neurons * coupling_order

# Visualize history filters
history_filters = coefficients[:history_dim]
fig, axs = plt.subplots(neurons) 
fig.suptitle('History Filters')
for n in range(neurons):
    ax = axs[n]
    filter_idx = n * (history_order + 1) + h
    filter_vals = history_filters[filter_idx::history_dim]
    ax.plot(filter_vals)
    ax.set_title(f'Neuron {n+1}, Lag {h}')
plt.tight_layout()
plt.show()

# Visualize coupling filters
coupling_filters = coefficients[history_dim:history_dim + coupling_dim]
fig, axs = plt.subplots(neurons, neurons, figsize=(12, 10))
fig.suptitle('Coupling Filters')
for n1 in range(neurons):
    for n2 in range(neurons):
        ax = axs[n1, n2]
        filter_idx = history_dim + (n1 * neurons + n2) * coupling_order
        filter_vals = coupling_filters[filter_idx::coupling_dim]
        ax.plot(filter_vals)
        ax.set_title(f'Coupling Filter: Neuron {n2+1} to {n1+1}')
plt.tight_layout()
plt.show()

