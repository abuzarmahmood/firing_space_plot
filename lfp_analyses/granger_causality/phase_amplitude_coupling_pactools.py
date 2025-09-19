"""
Phase-Amplitude Coupling Analysis using pactools

This script performs phase-amplitude coupling (PAC) analysis using the pactools library on:
    1. Raw LFP
    2. Granger preprocessed LFP

at different time periods:
    1. Baseline
    2. Stimulus

Uses the pactools.Comodulogram class for robust PAC estimation with multiple methods.
"""

from glob import glob
import tables
import os
import numpy as np
import sys
ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
granger_causality_path = \
    '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality'
process_scripts_path = os.path.join(granger_causality_path,'process_scripts')
sys.path.append(ephys_data_dir)
sys.path.append(granger_causality_path)
sys.path.append(process_scripts_path)
import granger_utils as gu
from ephys_data import ephys_data
import multiprocessing as mp
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import pactools
try:
    from pactools import Comodulogram, REFERENCES
    print("Successfully imported pactools")
except ImportError:
    print("ERROR: pactools not found. Please install with: pip install pactools")
    sys.exit(1)

def compute_pac_comodulogram(signal, fs, low_fq_range, high_fq_range, 
                           low_fq_width=2.0, high_fq_width=10.0, 
                           method='tort', progress_bar=False):
    """
    Compute PAC comodulogram using pactools.
    
    Parameters:
    -----------
    signal : array
        1D time series signal
    fs : float
        Sampling frequency
    low_fq_range : array
        Range of low frequencies for phase
    high_fq_range : array
        Range of high frequencies for amplitude
    low_fq_width : float
        Width of low frequency bands
    high_fq_width : float
        Width of high frequency bands
    method : str
        PAC method to use ('tort', 'ozkurt', 'canolty', etc.)
    progress_bar : bool
        Whether to show progress bar
        
    Returns:
    --------
    comodulogram : array
        2D comodulogram matrix
    estimator : Comodulogram
        Fitted comodulogram estimator
    """
    estimator = Comodulogram(
        fs=fs,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_range=high_fq_range,
        high_fq_width=high_fq_width,
        method=method,
        progress_bar=progress_bar
    )
    
    estimator.fit(signal)
    return estimator.comod_, estimator

def process_session_pac(dir_name, low_fq_range, high_fq_range, 
                       baseline_samples, stimulus_samples, 
                       pac_methods=['tort']):
    """
    Process PAC for a single session.
    
    Parameters:
    -----------
    dir_name : str
        Path to session directory
    low_fq_range : array
        Low frequency range for phase
    high_fq_range : array
        High frequency range for amplitude
    baseline_samples : list
        [start, end] sample indices for baseline
    stimulus_samples : list
        [start, end] sample indices for stimulus
    pac_methods : list
        List of PAC methods to compute
        
    Returns:
    --------
    results : dict
        Dictionary containing PAC results for each condition and method
    """
    basename = dir_name.split('/')[-1]
    print(f'Processing session: {basename}')
    
    try:
        dat = ephys_data(dir_name)
        dat.get_info_dict()
        
        # Get LFP data
        lfp_channel_inds, region_lfps, region_names = \
            dat.return_representative_lfp_channels()
        
        # Flatten data for processing
        flat_region_lfps = np.reshape(
            region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))
        
        # Remove trials with artifacts
        good_lfp_trials_bool = \
            dat.lfp_processing.return_good_lfp_trial_inds(flat_region_lfps)
        good_lfp_trials = flat_region_lfps[:, good_lfp_trials_bool]
        
        # Get sampling frequency
        try:
            fs = dat.info_dict['sampling_rate']
        except KeyError:
            fs = 1000  # Default
            
        # Initialize results dictionary
        results = {
            'region_names': region_names,
            'basename': basename,
            'fs': fs,
            'methods': {}
        }
        
        # Process PAC method
        for method in pac_methods:
            print(f'  Computing {method} PAC...')
            
            method_results = {
                'raw_baseline': [],
                'raw_stimulus': [],
                'processed_baseline': [],
                'processed_stimulus': []
            }
            
            # Process each region
            for region_idx in range(good_lfp_trials.shape[0]):
                region_data = good_lfp_trials[region_idx]
                
                # Initialize storage for this region
                region_raw_baseline = []
                region_raw_stimulus = []
                region_processed_baseline = []
                region_processed_stimulus = []
                
                # Process raw data
                for trial_idx in range(region_data.shape[0]):
                    trial_data = region_data[trial_idx]
                    
                    # Baseline period
                    baseline_data = trial_data[baseline_samples[0]:baseline_samples[1]]
                    if len(baseline_data) > 100:  # Minimum length check
                        comod, _ = compute_pac_comodulogram(
                            baseline_data, fs, low_fq_range, high_fq_range, method=method)
                        region_raw_baseline.append(comod)
                    
                    # Stimulus period
                    stimulus_data = trial_data[stimulus_samples[0]:stimulus_samples[1]]
                    if len(stimulus_data) > 100:  # Minimum length check
                        comod, _ = compute_pac_comodulogram(
                            stimulus_data, fs, low_fq_range, high_fq_range, method=method)
                        region_raw_stimulus.append(comod)
                
                # Process with Granger preprocessing
                this_granger = gu.granger_handler(region_data[np.newaxis, :, :])
                this_granger.preprocess_data()
                preprocessed_data = this_granger.preprocessed_data[0]
                
                for trial_idx in range(preprocessed_data.shape[0]):
                    trial_data = preprocessed_data[trial_idx]
                    
                    # Baseline period
                    baseline_data = trial_data[baseline_samples[0]:baseline_samples[1]]
                    if len(baseline_data) > 100:
                        comod, _ = compute_pac_comodulogram(
                            baseline_data, fs, low_fq_range, high_fq_range, method=method)
                        region_processed_baseline.append(comod)
                    
                    # Stimulus period
                    stimulus_data = trial_data[stimulus_samples[0]:stimulus_samples[1]]
                    if len(stimulus_data) > 100:
                        comod, _ = compute_pac_comodulogram(
                            stimulus_data, fs, low_fq_range, high_fq_range, method=method)
                        region_processed_stimulus.append(comod)
                
                # Store results for this region
                method_results['raw_baseline'].append(np.array(region_raw_baseline))
                method_results['raw_stimulus'].append(np.array(region_raw_stimulus))
                method_results['processed_baseline'].append(np.array(region_processed_baseline))
                method_results['processed_stimulus'].append(np.array(region_processed_stimulus))
            
            results['methods'][method] = method_results
        
        return results
        
    except Exception as e:
        print(f'ERROR processing {basename}: {e}')
        return None

# Set up directories
plot_dir = os.path.join(granger_causality_path, 'plots', 'phase_amplitude_coupling_pactools')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

artifacts_dir = os.path.join(granger_causality_path, 'artifacts')
if not os.path.isdir(artifacts_dir):
    os.makedirs(artifacts_dir)

# Load directory list
dir_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
print(f"Loading directory list from: {dir_list_path}")
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()
print(f"Found {len(dir_list)} directories to process")

# Define frequency ranges for PAC analysis
low_fq_range = np.linspace(2, 20, 19)  # 2-20 Hz for phase
high_fq_range = np.linspace(30, 100, 15)  # 30-100 Hz for amplitude
pac_methods = ['tort']  # Use only tort method

print(f"Phase frequencies: {low_fq_range[0]:.1f}-{low_fq_range[-1]:.1f} Hz ({len(low_fq_range)} bands)")
print(f"Amplitude frequencies: {high_fq_range[0]:.1f}-{high_fq_range[-1]:.1f} Hz ({len(high_fq_range)} bands)")
print(f"PAC method: {pac_methods[0]}")

# Time windows for analysis
baseline_start, baseline_end = 0, 2  # seconds
stimulus_start, stimulus_end = 2, 4  # seconds
print(f"Analysis time windows:")
print(f"  Baseline: {baseline_start}-{baseline_end} seconds")
print(f"  Stimulus: {stimulus_start}-{stimulus_end} seconds")

# Check for existing artifacts
pac_results_path = os.path.join(artifacts_dir, 'pac_pactools_results.npy')

if os.path.exists(pac_results_path):
    print("PAC results already exist. Loading existing data...")
    all_results = np.load(pac_results_path, allow_pickle=True).item()
    print(f"Loaded existing results for {len(all_results)} sessions")
else:
    print("Starting PAC analysis with pactools...")
    
    # Convert time windows to sample indices (assuming 1000 Hz sampling rate)
    fs_default = 1000
    baseline_samples = [int(baseline_start * fs_default), int(baseline_end * fs_default)]
    stimulus_samples = [int(stimulus_start * fs_default), int(stimulus_end * fs_default)]
    
    all_results = {}
    
    # Process each session
    for session_idx, dir_name in enumerate(tqdm(dir_list, desc="Processing sessions")):
        basename = dir_name.split('/')[-1]
        
        # Convert time windows to sample indices for this session
        try:
            dat = ephys_data(dir_name)
            dat.get_info_dict()
            fs = dat.info_dict.get('sampling_rate', fs_default)
            baseline_samples = [int(baseline_start * fs), int(baseline_end * fs)]
            stimulus_samples = [int(stimulus_start * fs), int(stimulus_end * fs)]
        except:
            fs = fs_default
            baseline_samples = [int(baseline_start * fs), int(baseline_end * fs)]
            stimulus_samples = [int(stimulus_start * fs), int(stimulus_end * fs)]
        
        results = process_session_pac(
            dir_name, low_fq_range, high_fq_range,
            baseline_samples, stimulus_samples, pac_methods
        )
        
        if results is not None:
            all_results[basename] = results
            
            # Save intermediate results
            if (session_idx + 1) % 5 == 0:  # Save every 5 sessions
                print(f"Saving intermediate results after {session_idx + 1} sessions...")
                np.save(pac_results_path, all_results, allow_pickle=True)
    
    # Final save
    print(f"Saving final results for {len(all_results)} sessions...")
    np.save(pac_results_path, all_results, allow_pickle=True)

############################################################
# Generate plots
############################################################

print(f'\n=== Starting plot generation ===')
print(f'Using data for {len(all_results)} sessions')

# Average results across sessions
def average_pac_results(all_results, method):
    """Average PAC results across sessions for a specific method"""
    conditions = ['raw_baseline', 'raw_stimulus', 'processed_baseline', 'processed_stimulus']
    averaged_results = {}
    
    # Get region names from first session
    first_session = list(all_results.values())[0]
    region_names = first_session['region_names']
    
    for condition in conditions:
        condition_data = []
        
        for session_name, session_data in all_results.items():
            if method in session_data['methods']:
                method_data = session_data['methods'][method][condition]
                
                # Average across trials for each region
                session_avg = []
                for region_data in method_data:
                    if len(region_data) > 0:
                        region_avg = np.nanmean(region_data, axis=0)
                        session_avg.append(region_avg)
                    else:
                        # Handle empty regions
                        session_avg.append(np.full((len(low_fq_range), len(high_fq_range)), np.nan))
                
                condition_data.append(session_avg)
        
        # Average across sessions
        if condition_data:
            condition_data = np.array(condition_data)
            averaged_results[condition] = np.nanmean(condition_data, axis=0)
        else:
            averaged_results[condition] = np.full((len(region_names), len(low_fq_range), len(high_fq_range)), np.nan)
    
    return averaged_results, region_names

# Generate plots for PAC method
for method in pac_methods:
    print(f'\nGenerating plots for {method} method...')
    
    try:
        averaged_data, region_names = average_pac_results(all_results, method)
        
        # Plot comodulograms for each region
        conditions = ['raw_baseline', 'raw_stimulus', 'processed_baseline', 'processed_stimulus']
        condition_labels = ['Raw Baseline', 'Raw Stimulus', 'Processed Baseline', 'Processed Stimulus']
        
        for region_idx, region_name in enumerate(region_names):
            print(f'  Creating plots for region: {region_name}')
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for cond_idx, (condition, label) in enumerate(zip(conditions, condition_labels)):
                ax = axes[cond_idx]
                
                # Get data for this condition and region
                data = averaged_data[condition][region_idx]
                
                # Plot comodulogram
                im = ax.imshow(data, 
                              extent=[high_fq_range[0], high_fq_range[-1], 
                                     low_fq_range[0], low_fq_range[-1]],
                              aspect='auto', origin='lower', cmap='viridis')
                
                ax.set_xlabel('Amplitude Frequency (Hz)')
                ax.set_ylabel('Phase Frequency (Hz)')
                ax.set_title(f'{label} - {region_name}\n({method.upper()} method)')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='PAC Strength')
            
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f'pac_{method}_{region_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'    Saved: {plot_path}')
        
        # Create method comparison plot (average across regions)
        print(f'  Creating method comparison plot...')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for cond_idx, (condition, label) in enumerate(zip(conditions, condition_labels)):
            ax = axes[cond_idx]
            
            # Average across all regions
            data = np.nanmean(averaged_data[condition], axis=0)
            
            im = ax.imshow(data, 
                          extent=[high_fq_range[0], high_fq_range[-1], 
                                 low_fq_range[0], low_fq_range[-1]],
                          aspect='auto', origin='lower', cmap='viridis')
            
            ax.set_xlabel('Amplitude Frequency (Hz)')
            ax.set_ylabel('Phase Frequency (Hz)')
            ax.set_title(f'{label} - All Regions\n({method.upper()} method)')
            
            plt.colorbar(im, ax=ax, label='PAC Strength')
        
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f'pac_{method}_all_regions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    Saved: {plot_path}')
        
    except Exception as e:
        print(f'  ERROR generating plots for {method}: {e}')
        continue

# Create method comparison plot
print('\nCreating method comparison plot...')
try:
    fig, axes = plt.subplots(len(pac_methods), 4, figsize=(16, 4*len(pac_methods)))
    if len(pac_methods) == 1:
        axes = axes.reshape(1, -1)
    
    conditions = ['raw_baseline', 'raw_stimulus', 'processed_baseline', 'processed_stimulus']
    condition_labels = ['Raw Baseline', 'Raw Stimulus', 'Processed Baseline', 'Processed Stimulus']
    
    for method_idx, method in enumerate(pac_methods):
        averaged_data, region_names = average_pac_results(all_results, method)
        
        for cond_idx, (condition, label) in enumerate(zip(conditions, condition_labels)):
            ax = axes[method_idx, cond_idx]
            
            # Average across all regions
            data = np.nanmean(averaged_data[condition], axis=0)
            
            im = ax.imshow(data, 
                          extent=[high_fq_range[0], high_fq_range[-1], 
                                 low_fq_range[0], low_fq_range[-1]],
                          aspect='auto', origin='lower', cmap='viridis')
            
            if method_idx == len(pac_methods) - 1:
                ax.set_xlabel('Amplitude Frequency (Hz)')
            if cond_idx == 0:
                ax.set_ylabel(f'{method.upper()}\nPhase Frequency (Hz)')
            if method_idx == 0:
                ax.set_title(label)
            
            # Add colorbar only for rightmost plots
            if cond_idx == len(conditions) - 1:
                plt.colorbar(im, ax=ax, label='PAC')
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, 'pac_methods_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved method comparison: {plot_path}')
    
except Exception as e:
    print(f'ERROR creating method comparison plot: {e}')

print(f"\nPAC analysis with pactools complete!")
print(f"Results saved to: {pac_results_path}")
print(f"Plots saved to: {plot_dir}")
print(f"Processed {len(all_results)} sessions with {pac_methods[0]} PAC method")
