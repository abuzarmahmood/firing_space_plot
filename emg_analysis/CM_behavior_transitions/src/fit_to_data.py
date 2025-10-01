<<<<<<< HEAD
"""
Script for fitting Gaussian changepoint models to real EMG/behavior data.

This script:
1. Loads real multidimensional timeseries data from files
2. Fits Gaussian changepoint models with different numbers of states
3. Infers the best number of states using model comparison
4. Saves results and generates plots
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
import argparse
from pathlib import Path
import pandas as pd

# Add the pytau module to the path
sys.path.append('/media/bigdata/projects/pytau')

from pytau.changepoint_model import (
    GaussianChangepointMean2D,
    GaussianChangepointMeanDirichlet,
    PoissonChangepoint1D,
    find_best_states,
    advi_fit,
    dpp_fit
)


def load_data(data_path):
    """
    Load multidimensional timeseries data from file.
    
    Args:
        data_path (str): Path to data file (supports .npy, .pkl, .npz)
        
    Returns:
        data (ndarray): Shape (n_dims, n_timepoints) - the loaded data
        metadata (dict): Any additional metadata from the file
    """
    data_path = Path(data_path)
    metadata = {}
    
    if data_path.suffix == '.npy':
        data = np.load(data_path)
    elif data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            loaded = pickle.load(f)
            if isinstance(loaded, dict):
                data = loaded.get('data', loaded.get('timeseries', None))
                metadata = {k: v for k, v in loaded.items() if k not in ['data', 'timeseries']}
            else:
                data = loaded
    elif data_path.suffix == '.npz':
        loaded = np.load(data_path)
        data = loaded.get('data', loaded.get('timeseries', loaded[loaded.files[0]]))
        metadata = {k: loaded[k] for k in loaded.files if k not in ['data', 'timeseries']}
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    if data is None:
        raise ValueError("Could not find data in the loaded file")
    
    # Ensure data is in the correct shape (n_dims, n_timepoints)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim == 2:
        # If n_timepoints > n_dims, assume data needs to be transposed
        if data.shape[0] > data.shape[1]:
            print(f"Transposing data from shape {data.shape} to {data.shape[::-1]}")
            data = data.T
    else:
        raise ValueError(f"Data must be 1D or 2D, got shape {data.shape}")
    
    n_dims, n_timepoints = data.shape
    print(f"Loaded data with {n_dims} dimensions and {n_timepoints} timepoints")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    return data, metadata


def preprocess_data(data, normalize=True, detrend=False):
    """
    Preprocess the data before fitting changepoint models.
    
    Args:
        data (ndarray): Shape (n_dims, n_timepoints)
        normalize (bool): Whether to z-score normalize each dimension
        detrend (bool): Whether to remove linear trends
        
    Returns:
        processed_data (ndarray): Preprocessed data
    """
    processed_data = data.copy()
    
    if detrend:
        print("Detrending data...")
        from scipy import signal
        for dim_idx in range(processed_data.shape[0]):
            processed_data[dim_idx, :] = signal.detrend(processed_data[dim_idx, :])
    
    if normalize:
        print("Normalizing data...")
        for dim_idx in range(processed_data.shape[0]):
            mean_val = np.mean(processed_data[dim_idx, :])
            std_val = np.std(processed_data[dim_idx, :])
            if std_val > 0:
                processed_data[dim_idx, :] = (processed_data[dim_idx, :] - mean_val) / std_val
    
    return processed_data


def plot_data_and_results(data, inferred_changepoints=None, 
                         title="Multidimensional Timeseries Data", 
                         save_path=None):
    """
    Plot the data and changepoints.
    
    Args:
        data (ndarray): Shape (n_dims, n_timepoints)
        inferred_changepoints (ndarray, optional): Inferred changepoint samples
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    n_dims, n_timepoints = data.shape
    time_axis = np.arange(n_timepoints)
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2*n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    for dim_idx in range(n_dims):
        axes[dim_idx].plot(time_axis, data[dim_idx, :], 'b-', alpha=0.7, linewidth=1)
        
        # Plot inferred changepoints if provided
        if inferred_changepoints is not None:
            # Plot median of inferred changepoints as vertical lines
            median_cps = np.median(inferred_changepoints, axis=0)
            # Also plot confidence intervals
            ci_lower = np.percentile(inferred_changepoints, 5, axis=0)
            ci_upper = np.percentile(inferred_changepoints, 95, axis=0)
            
            for cp_idx, (cp, lower, upper) in enumerate(zip(median_cps, ci_lower, ci_upper)):
                axes[dim_idx].axvline(cp, color='red', linestyle='-', alpha=0.8,
                                     label='Inferred CP' if dim_idx == 0 and cp_idx == 0 else "")
                axes[dim_idx].axvspan(lower, upper, color='red', alpha=0.2)
        
        axes[dim_idx].set_ylabel(f'Dim {dim_idx+1}')
        axes[dim_idx].grid(True, alpha=0.3)
    
    axes[0].set_title(title)
    if inferred_changepoints is not None:
        axes[0].legend()
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def save_results(results, save_path):
    """
    Save analysis results to file.
    
    Args:
        results (dict): Dictionary containing analysis results
        save_path (str): Path to save results
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {save_path}")

# import pymc as pm
# def dpp_fit(model, n_chains=24, n_cores=1, tune=500, draws=500, use_numpyro=False):
#     """Convenience function to fit DPP model"""
#     if not use_numpyro:
#         with model:
#             dpp_trace = pm.sample(
#                 tune=tune,
#                 draws=draws,
#                 target_accept=0.95,
#                 chains=n_chains,
#                 cores=n_cores,
#                 return_inferencedata=False,
#             )
#     else:
#         with model:
#             dpp_trace = pm.sample(
#                 nuts_sampler="numpyro",
#                 tune=tune,
#                 draws=draws,
#                 target_accept=0.95,
#                 chains=n_chains,
#                 cores=n_cores,
#                 return_inferencedata=False,
#             )
#     return dpp_trace


def main():
    print("=== Changepoint Analysis on Real Data ===\n")
    
    base_dir = '/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions'
    artifacts_dir = os.path.join(base_dir, 'artifacts')
    plot_dir = os.path.join(base_dir, 'plots')
    data_dir = os.path.join(base_dir, 'data')


    plot_dir = Path(plot_dir)
    artifacts_dir = Path(artifacts_dir)

    # 1. Load data
    data_list = os.listdir(data_dir)
    # Raw_data shape: trials x time
    raw_data, raw_metadata = load_data(
            os.path.join(data_dir,
            [x for x in data_list if 'gape' in x][0]
                         )
            )

    # shape: PCA components x trials
    pca_data, pca_metadata = load_data(
            os.path.join(data_dir,
            [x for x in data_list if 'pca' in x][0]
                         )
            )

    # Add noise to avoid singularities
    pca_data_range = np.max(pca_data) - np.min(pca_data)
    pca_data += np.random.normal(0, 0.05 * pca_data_range, size=pca_data.shape) 

    # Summed counts
    summed_data = np.sum(raw_data, axis=1, keepdims=True)


    fig, ax = plt.subplots(3,1) 
    ax[0].imshow(raw_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[1].imshow(pca_data, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[2].plot(np.arange(len(summed_data)), summed_data)
    plt.show()

    ##############################
    # Fit poisson changepoint model to summed data
    n_states_tested = np.arange(1, 7)
    n_repeats = 5
    n_states_tested = np.repeat(n_states_tested, n_repeats)
    elbo_list = []
    for n_states in n_states_tested:
        model = PoissonChangepoint1D(summed_data.flatten(), n_states=3).generate_model()
        model, approx = advi_fit(model, fit=50_000, samples=2000)
        this_elbo = approx.hist[-1]
        elbo_list.append(this_elbo)

    elbo_df = pd.DataFrame({
        'n_states': n_states_tested,
        'elbo': elbo_list
        })

    # Mean min
    elbo_df_mean = elbo_df.groupby('n_states').mean().reset_index()
    min_state = elbo_df_mean.loc[elbo_df_mean['elbo'].idxmin(), 'n_states']

    plt.plot(n_states_tested, elbo_list, 'o')
    plt.show()

    # Sample from the fitted approximation
    trace = approx.sample(draws=2000)
    
    # Extract tau samples
    tau_samples = trace.posterior['tau'].values[0]
    print(f"   Tau samples shape: {tau_samples.shape}")
    
    # Plot final results
    median_tau = np.median(tau_samples, axis=0)
    print(f"   Inferred changepoints (median): {median_tau}")

    fig,ax = plt.subplots(3,1, sharex=True)
    ax[0].imshow(raw_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[1].imshow(pca_data, aspect='auto', cmap='viridis', interpolation='nearest')
    for i, this_tau in enumerate(tau_samples.T):
        ax[2].hist(this_tau, bins=50, alpha=0.3, label=f'CP {i+1}' if i==0 else "")
        ax[2].axvline(median_tau[i], color='red', linestyle='-', alpha=0.8)
    ax[2].legend()
    plt.show()

    def model_generator(data_array, n_states):
        """Helper function to generate models for comparison."""
        model_class = PoissonChangepoint1D(data_array, n_states)
        return model_class.generate_model()

    n_states_tested = np.arange(1, 7)
    repeats = 5
    all_elbo_lists = []
    for repeat_idx in range(repeats):
        best_model, model_list, elbo_values = find_best_states(
            data=summed_data.flatten(),
            model_generator=model_generator,
            n_fit=100_000,
            n_samples=1000,
            min_states=np.min(n_states_tested),
            max_states=np.max(n_states_tested),
            convergence_tol=5e-3,
        )
        all_elbo_lists.append(elbo_values)

    all_state_vector = np.tile(n_states_tested, repeats)
    all_elbo_vector = np.concatenate(all_elbo_lists)

    plt.plot(all_state_vector, all_elbo_vector, 'o', alpha=0.5)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(n_states_tested, elbo_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of States')
    plt.ylabel('ELBO (Evidence Lower Bound)')
    plt.title('Model Comparison: ELBO vs Number of States')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    best_n_states = all_state_vector[np.argmin(all_elbo_vector)]  
    
    # Fit the model using ADVI
    print("   Running ADVI inference...")
    best_model = PoissonChangepoint1D(summed_data.flatten(), n_states=int(best_n_states)).generate_model()
    model, approx = advi_fit(best_model, fit=50_000, samples=2000)
    
    # Sample from the fitted approximation
    trace = approx.sample(draws=2000)
    
    # Extract tau samples
    tau_samples = trace.posterior['tau'].values[0]
    print(f"   Tau samples shape: {tau_samples.shape}")
    
    # Plot final results
    median_tau = np.median(tau_samples, axis=0)
    print(f"   Inferred changepoints (median): {median_tau}")

    
    # 2. Preprocess data
    # print("\n2. Preprocessing data...")
    # processed_data = preprocess_data(data, normalize=args.normalize, detrend=args.detrend)
    
    # Plot the original and processed data
    # fig1 = plot_data_and_results(pca_data, title="Original Data",
    #                             save_path= plot_dir / "original_data.png")
    # plt.show()
    
    # if not np.array_equal(data, processed_data):
    #     fig2 = plot_data_and_results(processed_data, title="Preprocessed Data",
    #                                 save_path=output_dir / "preprocessed_data.png")
    #     plt.show()
    
    # 3. Find best number of states
    # First try Dirichlet model
    model = GaussianChangepointMeanDirichlet(pca_data, max_states = 10).generate_model()
    dpp_trace = dpp_fit(model, use_numpyro=True, n_cores=24, n_chains=24, tune=500, draws=500)
    # with model:
    #     dpp_trace = pm.sample(
    #         nuts_sampler="nutpie",
    #         tune=500,
    #         draws=500,
    #         target_accept=0.95,
    #         chains=24,
    #         cores=25,
    #         return_inferencedata=False,
    #     )
    
=======

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


>>>>>>> refs/remotes/origin/master
    def model_generator(data_array, n_states):
        """Helper function to generate models for comparison."""
        model_class = GaussianChangepointMean2D(data_array, n_states)
        return model_class.generate_model()
<<<<<<< HEAD

    plt.plot(pca_data[0])
    plt.show()
    
    try:
        best_model, model_list, elbo_values = find_best_states(
            # data=pca_data[0][np.newaxis, :],  # Use first PCA component for model selection
            data=pca_data,
            model_generator=model_generator,
            n_fit=100_000,
            n_samples=1000,
            min_states=1,
            max_states=6,
            convergence_tol=5e-2,
        )
        
        # Plot ELBO values
        # n_states_tested = np.arange(args.min_states, args.max_states + 1)
=======
    
    try:
        best_model, model_list, approx_list, elbo_values = find_best_states(
            data=pca_data.T,
            model_generator=model_generator,
            n_fit=100000,  # Fewer iterations for speed
            n_samples=500,
            min_states=1,
            max_states=6,
            convergence_tol=1e-2,
        )
        
        # Plot ELBO values
>>>>>>> refs/remotes/origin/master
        n_states_tested = np.arange(1, 7)
        plt.figure(figsize=(8, 5))
        plt.plot(n_states_tested, elbo_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of States')
        plt.ylabel('ELBO (Evidence Lower Bound)')
        plt.title('Model Comparison: ELBO vs Number of States')
        plt.grid(True, alpha=0.3)
<<<<<<< HEAD
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        best_n_states = n_states_tested[np.argmin(elbo_values)]  # Higher ELBO is better
        print(f"   Best number of states: {best_n_states}")
        print(f"   ELBO values: {dict(zip(n_states_tested, elbo_values))}")
        
        # 4. Fit final model with best number of states
        print(f"\n4. Fitting final model with {best_n_states} states...")
        # final_model = GaussianChangepointMean2D(processed_data, n_states=best_n_states)
        # pymc_model = final_model.generate_model()
        
        # Fit the model using ADVI
        print("   Running ADVI inference...")
        model, approx = advi_fit(best_model, fit=50_000, samples=2000)
        
        # Sample from the fitted approximation
        trace = approx.sample(draws=2000)
        
        # Extract tau samples
        tau_samples = trace.posterior['tau'].values[0]
        print(f"   Tau samples shape: {tau_samples.shape}")
        
        # Plot final results
        median_tau = np.median(tau_samples, axis=0)
        print(f"   Inferred changepoints (median): {median_tau}")

        fig,ax = plt.subplots(3,1, sharex=True)
        ax[0].imshow(raw_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax[1].imshow(pca_data, aspect='auto', cmap='viridis', interpolation='nearest')
        for i, this_tau in enumerate(tau_samples.T):
            ax[2].hist(this_tau, bins=50, alpha=0.3, label=f'CP {i+1}' if i==0 else "")
            ax[2].axvline(median_tau[i], color='red', linestyle='-', alpha=0.8)
        ax[2].legend()
        plt.show()

        
        fig3 = plot_data_and_results(pca_data, inferred_changepoints=tau_samples,
                                    title=f"Final Results ({best_n_states} states)",
                                    save_path=output_dir / "final_results.png")
        plt.show()
        
        # 5. Save results
        print("\n5. Saving results...")
        results = {
            'data_path': args.data_path,
            'original_data': data,
            'processed_data': processed_data,
            'metadata': metadata,
            'preprocessing': {
                'normalize': args.normalize,
                'detrend': args.detrend
            },
            'model_comparison': {
                'n_states_tested': n_states_tested,
                'elbo_values': elbo_values,
                'best_n_states': best_n_states
            },
            'final_model': {
                'n_states': best_n_states,
                'tau_samples': tau_samples,
                'median_changepoints': median_tau,
                'changepoint_ci': {
                    'lower': np.percentile(tau_samples, 5, axis=0),
                    'upper': np.percentile(tau_samples, 95, axis=0)
                }
            },
            'parameters': {
                'n_fit': args.n_fit,
                'n_samples': args.n_samples,
                'min_states': args.min_states,
                'max_states': args.max_states
            }
        }
        
        save_results(results, output_dir / "changepoint_results.pkl")
        
    except Exception as e:
        print(f"   Error in model fitting: {e}")
        print("   Please check your data and try again.")
        return
    
    print("\n=== Analysis Complete ===")
    print("Key Results:")
    print(f"- Loaded data with {processed_data.shape[0]} dimensions and {processed_data.shape[1]} timepoints")
    print(f"- Best number of states: {best_n_states}")
    print(f"- Inferred changepoints: {median_tau}")
    print(f"- Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
=======
        plt.legend()
        plt.show()
        
    best_n_states = n_states_tested[np.argmin(elbo_values)]  # Higher ELBO is better
    best_approx = approx_list[np.argmin(elbo_values)]

    samples = best_approx.sample(1000) 
    tau_samples = samples.posterior['tau'].values[0]


    median_tau = np.median(tau_samples, axis=0)
    fig, ax = plt.subplots(3,1, figsize=(12, 8), sharex=True) 
    ax[0].imshow(raw_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[0].set_title("Raw Data with Inferred Changepoints (2 states)")
    ax[1].imshow(pca_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax[1].set_title("PCA Data with Inferred Changepoints (2 states)")
    ax[2].set_title("Posterior Distribution of Changepoint (tau)")
    ax[2].set_xlabel('Trials')
    ax[2].set_ylabel('Frequency')
    ax2.set_ylabel('Cumulative Probability')
    ax[2].grid(True, alpha=0.3)
    ax2.grid(False)
    ax[0].set_ylabel('Time')
    ax[1].set_ylabel('PCA Dimensions')
    ax2.legend()
    for this_change in tau_samples.T:
        tau_hist, tau_bins = np.histogram(this_change, bins=30)
        ax[2].bar(tau_bins[:-1], tau_hist, width=np.diff(tau_bins), edgecolor='black', align='edge', alpha=0.3)
        cdf = np.cumsum(tau_hist) / np.sum(tau_hist)
        ax2 = ax[2].twinx()
        ax2.plot(tau_bins[:-1], cdf, linestyle='--', label='CDF', linewidth=2)
        median_tau = np.median(this_change)
        ax[0].axvline(median_tau, color='yellow', linestyle='-', alpha=0.8, label='Inferred CP (median)',
                      linewidth=2)
        ax[1].axvline(median_tau, color='yellow', linestyle='-', alpha=0.8, label='Inferred CP (median)',
                      linewidth=2)
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

>>>>>>> refs/remotes/origin/master
