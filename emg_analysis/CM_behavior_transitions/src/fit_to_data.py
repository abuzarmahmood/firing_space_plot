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

# Add the pytau module to the path
sys.path.append('/media/bigdata/projects/pytau')

from pytau.changepoint_model import (
    GaussianChangepointMean2D,
    find_best_states,
    advi_fit,
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


def main():
    """Main function to run the changepoint analysis on real data."""
    parser = argparse.ArgumentParser(description='Fit changepoint models to real data')
    parser.add_argument('data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='./results', 
                       help='Directory to save results')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data before fitting')
    parser.add_argument('--detrend', action='store_true', default=False,
                       help='Detrend data before fitting')
    parser.add_argument('--min_states', type=int, default=2,
                       help='Minimum number of states to test')
    parser.add_argument('--max_states', type=int, default=8,
                       help='Maximum number of states to test')
    parser.add_argument('--n_fit', type=int, default=5000,
                       help='Number of ADVI iterations')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to draw')
    
    args = parser.parse_args()
    
    print("=== Changepoint Analysis on Real Data ===\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print(f"1. Loading data from: {args.data_path}")
    try:
        data, metadata = load_data(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    processed_data = preprocess_data(data, normalize=args.normalize, detrend=args.detrend)
    
    # Plot the original and processed data
    fig1 = plot_data_and_results(data, title="Original Data",
                                save_path=output_dir / "original_data.png")
    plt.show()
    
    if not np.array_equal(data, processed_data):
        fig2 = plot_data_and_results(processed_data, title="Preprocessed Data",
                                    save_path=output_dir / "preprocessed_data.png")
        plt.show()
    
    # 3. Find best number of states
    print(f"\n3. Finding best number of states (testing {args.min_states}-{args.max_states} states)...")
    
    def model_generator(data_array, n_states):
        """Helper function to generate models for comparison."""
        model_class = GaussianChangepointMean2D(data_array, n_states)
        return model_class.generate_model()
    
    try:
        best_model, model_list, elbo_values = find_best_states(
            data=processed_data,
            model_generator=model_generator,
            n_fit=args.n_fit,
            n_samples=args.n_samples,
            min_states=args.min_states,
            max_states=args.max_states,
            convergence_tol=1e-3,
        )
        
        # Plot ELBO values
        n_states_tested = np.arange(args.min_states, args.max_states + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(n_states_tested, elbo_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of States')
        plt.ylabel('ELBO (Evidence Lower Bound)')
        plt.title('Model Comparison: ELBO vs Number of States')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        best_n_states = n_states_tested[np.argmax(elbo_values)]  # Higher ELBO is better
        print(f"   Best number of states: {best_n_states}")
        print(f"   ELBO values: {dict(zip(n_states_tested, elbo_values))}")
        
        # 4. Fit final model with best number of states
        print(f"\n4. Fitting final model with {best_n_states} states...")
        final_model = GaussianChangepointMean2D(processed_data, n_states=best_n_states)
        pymc_model = final_model.generate_model()
        
        # Fit the model using ADVI
        print("   Running ADVI inference...")
        model, approx = advi_fit(pymc_model, fit=args.n_fit, samples=args.n_samples)
        
        # Sample from the fitted approximation
        trace = approx.sample(draws=args.n_samples)
        
        # Extract tau samples
        tau_samples = trace.posterior['tau'].values[0]
        print(f"   Tau samples shape: {tau_samples.shape}")
        
        # Plot final results
        median_tau = np.median(tau_samples, axis=0)
        print(f"   Inferred changepoints (median): {median_tau}")
        
        fig3 = plot_data_and_results(processed_data, tau_samples,
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
