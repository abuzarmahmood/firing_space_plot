
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the pytau module to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'projects', 'pytau'))
sys.path.append('/home/abuzarmahmood/projects/pytau')

from pytau.changepoint_model import (
    GaussianChangepointMean2D,
    find_best_states,
    advi_fit,
)



if __name__ == "__main__":
    """Main function to run the changepoint analysis."""

    base_dir = '/home/abuzarmahmood/projects/firing_space_plot/emg_analysis/CM_behavior_transitions'
    data_dir = os.path.join(base_dir, 'data')
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    plots_dir = os.path.join(base_dir, 'plots')

    # Load data
    file_list = os.listdir(data_dir)
    raw_data = np.load(os.path.join(data_dir, file_list[1]))  # Assuming single file for simplicity
    pca_data = np.load(os.path.join(data_dir, file_list[0])) 

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(raw_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[0].set_title('Raw Data')
    ax[1].imshow(pca_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[1].set_title('PCA Data')
    plt.show()

    
    # 2. Fit Gaussian changepoint model with known number of states
    print("\n2. Fitting Gaussian changepoint model with 3 states...")
    model_2_states = GaussianChangepointMean2D(pca_data.T, n_states=2)
    pymc_model = model_2_states.generate_model()
    
    # Fit the model using ADVI
    print("   Running ADVI inference...")
    model, approx = advi_fit(pymc_model, fit=100_000, samples=1000)
    
    # Sample from the fitted approximation
    trace = approx.sample(draws=1000)
    
    print(f"   Inference completed.")
    
    # Extract tau samples directly from trace
    # Shape: (n_samples, n_changepoints)
    tau_samples = trace.posterior['tau'].values[0]
    
    print(f"   Tau samples shape: {tau_samples.shape}")
    
    # Plot results with inferred changepoints
    median_tau = np.median(tau_samples, axis=0)
    tau_hist, tau_bins = np.histogram(tau_samples[:,0], bins=30)
    fig, ax = plt.subplots(3,1, figsize=(12, 8), sharex=True) 
    ax[0].imshow(raw_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[0].set_title("Raw Data with Inferred Changepoints (2 states)")
    ax[1].imshow(pca_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[1].set_title("PCA Data with Inferred Changepoints (2 states)")
    ax[2].bar(tau_bins[:-1], tau_hist, width=np.diff(tau_bins), edgecolor='black', align='edge')
    # Also overlay CDF
    cdf = np.cumsum(tau_hist) / np.sum(tau_hist)
    ax2 = ax[2].twinx()
    ax2.plot(tau_bins[:-1], cdf, color='red', linestyle='--', label='CDF', linewidth=2)
    ax[2].set_title("Posterior Distribution of Changepoint (tau)")
    ax[2].set_xlabel('Trials')
    ax[2].set_ylabel('Frequency')
    ax2.set_ylabel('Cumulative Probability')
    ax[2].grid(True, alpha=0.3)
    ax2.grid(False)
    ax[0].set_ylabel('Time')
    ax[1].set_ylabel('PCA Dimensions')
    ax2.legend()
    ax[0].axvline(median_tau[0], color='yellow', linestyle='-', alpha=0.8, label='Inferred CP (median)',
                  linewidth=2)
    ax[1].axvline(median_tau[0], color='yellow', linestyle='-', alpha=0.8, label='Inferred CP (median)',
                  linewidth=2)
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(plots_dir, 'changepoint_inference_2_states.png'))
    plt.close(fig)



    
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
