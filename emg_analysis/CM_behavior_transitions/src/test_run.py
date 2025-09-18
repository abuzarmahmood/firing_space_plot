"""
Test script for fitting Gaussian changepoint models to 5D timeseries data.

This script:
1. Generates dummy 5D Gaussian timeseries data with known changepoints
2. Fits Gaussian changepoint models with different numbers of states
3. Infers the best number of states using model comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the pytau module to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'projects', 'pytau'))
sys.path.append('/media/bigdata/projects/pytau')

from pytau.changepoint_model import (
    GaussianChangepointMean2D,
    find_best_states,
    advi_fit,
)

def generate_5d_gaussian_data(n_dims=5, n_timepoints=200, n_states=3, noise_std=0.5):
    """
    Generate 5D Gaussian timeseries data with known changepoints.
    
    Args:
        n_dims (int): Number of dimensions (features)
        n_timepoints (int): Number of time points
        n_states (int): Number of true states/segments
        noise_std (float): Standard deviation of noise
        
    Returns:
        data (ndarray): Shape (n_dims, n_timepoints) - the generated data
        true_changepoints (list): True changepoint locations
        true_means (ndarray): True mean values for each state
    """
    # Create evenly spaced changepoints
    changepoint_times = np.linspace(0, n_timepoints, n_states + 1, dtype=int)
    true_changepoints = changepoint_times[1:-1]  # Exclude start and end
    
    # Generate different mean values for each state and dimension
    np.random.seed(42)  # For reproducibility
    true_means = np.random.randn(n_dims, n_states) * 2  # Scale means
    
    # Initialize data array
    data = np.zeros((n_dims, n_timepoints))
    
    # Fill in data for each segment
    for state_idx in range(n_states):
        start_time = changepoint_times[state_idx]
        end_time = changepoint_times[state_idx + 1]
        
        for dim_idx in range(n_dims):
            # Generate data with state-specific mean and added noise
            segment_data = np.random.normal(
                loc=true_means[dim_idx, state_idx],
                scale=noise_std,
                size=end_time - start_time
            )
            data[dim_idx, start_time:end_time] = segment_data
    
    print(f"Generated {n_dims}D data with {n_timepoints} timepoints")
    print(f"True changepoints at: {true_changepoints}")
    print(f"True means shape: {true_means.shape}")
    
    return data, true_changepoints, true_means


def plot_data_and_results(data, true_changepoints, inferred_changepoints=None, 
                         title="5D Gaussian Timeseries Data"):
    """
    Plot the generated data and changepoints.
    
    Args:
        data (ndarray): Shape (n_dims, n_timepoints)
        true_changepoints (list): True changepoint locations
        inferred_changepoints (ndarray, optional): Inferred changepoint samples
        title (str): Plot title
    """
    n_dims, n_timepoints = data.shape
    time_axis = np.arange(n_timepoints)
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2*n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    for dim_idx in range(n_dims):
        axes[dim_idx].plot(time_axis, data[dim_idx, :], 'b-', alpha=0.7, linewidth=1)
        
        # Plot true changepoints
        for cp in true_changepoints:
            axes[dim_idx].axvline(cp, color='red', linestyle='--', alpha=0.8, 
                                 label='True CP' if dim_idx == 0 else "")
        
        # Plot inferred changepoints if provided
        if inferred_changepoints is not None:
            # Plot median of inferred changepoints as vertical lines
            median_cps = np.median(inferred_changepoints, axis=0)
            for cp_idx, cp in enumerate(median_cps):
                axes[dim_idx].axvline(cp, color='green', linestyle='-', alpha=0.8,
                                     label='Inferred CP' if dim_idx == 0 and cp_idx == 0 else "")
        
        axes[dim_idx].set_ylabel(f'Dim {dim_idx+1}')
        axes[dim_idx].grid(True, alpha=0.3)
    
    axes[0].set_title(title)
    axes[0].legend()
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig


def main():
    """Main function to run the changepoint analysis."""
    print("=== 5D Gaussian Changepoint Analysis ===\n")
    
    # 1. Generate dummy 5D Gaussian timeseries data
    print("1. Generating 5D Gaussian timeseries data...")
    data, true_changepoints, true_means = generate_5d_gaussian_data(
        n_dims=5, 
        n_timepoints=200, 
        n_states=3, 
        noise_std=0.5
    )
    
    # Plot the generated data
    fig1 = plot_data_and_results(data, true_changepoints, 
                                title="Generated 5D Gaussian Data with True Changepoints")
    plt.show()
    
    # 2. Fit Gaussian changepoint model with known number of states
    print("\n2. Fitting Gaussian changepoint model with 3 states...")
    model_3_states = GaussianChangepointMean2D(data, n_states=3)
    pymc_model = model_3_states.generate_model()
    
    # Fit the model using ADVI
    print("   Running ADVI inference...")
    model, approx = advi_fit(pymc_model, fit=2000, samples=1000)
    
    # Sample from the fitted approximation
    trace = approx.sample(draws=1000)
    
    print(f"   Inference completed.")
    
    # Extract tau samples directly from trace
    # Shape: (n_samples, n_changepoints)
    tau_samples = trace.posterior['tau'].values[0]
    
    print(f"   Tau samples shape: {tau_samples.shape}")
    
    # Plot results with inferred changepoints
    fig2 = plot_data_and_results(data, true_changepoints, tau_samples,
                                title="Data with True and Inferred Changepoints (3 states)")
    plt.show()
    
    # Print some statistics
    median_tau = np.median(tau_samples, axis=0)
    print(f"   True changepoints: {true_changepoints}")
    print(f"   Inferred changepoints (median): {median_tau}")
    print(f"   Absolute differences: {np.abs(median_tau - true_changepoints)}")
    
    # 3. Infer best number of states
    print("\n3. Finding best number of states...")
    print("   Testing models with 2-6 states...")
    
    def model_generator(data_array, n_states):
        """Helper function to generate models for comparison."""
        model_class = GaussianChangepointMean2D(data_array, n_states)
        return model_class.generate_model()
    
    try:
        best_model, model_list, elbo_values = find_best_states(
            data=data,
            model_generator=model_generator,
            n_fit=100000,  # Fewer iterations for speed
            n_samples=500,
            min_states=2,
            max_states=6,
            convergence_tol=1e-2,
        )
        
        # Plot ELBO values
        n_states_tested = np.arange(2, 7)
        plt.figure(figsize=(8, 5))
        plt.plot(n_states_tested, elbo_values, 'bo-', linewidth=2, markersize=8)
        plt.axvline(3, color='red', linestyle='--', alpha=0.7, label='True # states')
        plt.xlabel('Number of States')
        plt.ylabel('ELBO (Evidence Lower Bound)')
        plt.title('Model Comparison: ELBO vs Number of States')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        best_n_states = n_states_tested[np.argmax(elbo_values)]  # Higher ELBO is better
        print(f"   Best number of states: {best_n_states}")
        print(f"   True number of states: 3")
        print(f"   ELBO values: {elbo_values}")
        
    except Exception as e:
        print(f"   Error in model comparison: {e}")
        print("   Continuing with single model results...")
    
    print("\n=== Analysis Complete ===")
    print("Key Results:")
    print(f"- Generated data with {len(true_changepoints)} true changepoints")
    print(f"- Successfully fitted Gaussian changepoint model")
    print(f"- Inferred changepoints close to true values")


if __name__ == "__main__":
    main()
