"""
Phase-Amplitude Coupling Analysis for Granger Causality LFP Data

This script performs phase-amplitude coupling (PAC) analysis on:
    1. Raw LFP
    2. Granger preprocessed LFP

at different time periods:
    1. Baseline
    2. Stimulus

PAC measures the coupling between the phase of low-frequency oscillations
and the amplitude of high-frequency oscillations.
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
from scipy import signal
from scipy.stats import circmean, circstd
import seaborn as sns

def extract_phase_amplitude(lfp_data, fs, phase_freq_range, amp_freq_range):
    """
    Extract phase and amplitude from LFP data for PAC analysis.
    
    Parameters:
    -----------
    lfp_data : array
        LFP time series data
    fs : float
        Sampling frequency
    phase_freq_range : tuple
        (low, high) frequency range for phase extraction
    amp_freq_range : tuple
        (low, high) frequency range for amplitude extraction
        
    Returns:
    --------
    phase : array
        Instantaneous phase of low-frequency component
    amplitude : array
        Instantaneous amplitude of high-frequency component
    """
    # Filter for phase frequency
    phase_filt = signal.butter(4, phase_freq_range, btype='band', fs=fs)
    phase_signal = signal.filtfilt(phase_filt[0], phase_filt[1], lfp_data)
    
    # Filter for amplitude frequency
    amp_filt = signal.butter(4, amp_freq_range, btype='band', fs=fs)
    amp_signal = signal.filtfilt(amp_filt[0], amp_filt[1], lfp_data)
    
    # Extract phase using Hilbert transform
    analytic_phase = signal.hilbert(phase_signal)
    phase = np.angle(analytic_phase)
    
    # Extract amplitude envelope using Hilbert transform
    analytic_amp = signal.hilbert(amp_signal)
    amplitude = np.abs(analytic_amp)
    
    return phase, amplitude

def calculate_modulation_index(phase, amplitude, n_bins=18):
    """
    Calculate Modulation Index (MI) for phase-amplitude coupling.
    
    Parameters:
    -----------
    phase : array
        Instantaneous phase values
    amplitude : array
        Instantaneous amplitude values
    n_bins : int
        Number of phase bins for MI calculation
        
    Returns:
    --------
    mi : float
        Modulation Index value
    mean_amp_per_phase : array
        Mean amplitude for each phase bin
    phase_bins : array
        Phase bin centers
    """
    # Create phase bins
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    phase_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    
    # Calculate mean amplitude for each phase bin
    mean_amp_per_phase = np.zeros(n_bins)
    for i in range(n_bins):
        phase_mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if np.sum(phase_mask) > 0:
            mean_amp_per_phase[i] = np.mean(amplitude[phase_mask])
    
    # Normalize to create probability distribution
    p = mean_amp_per_phase / np.sum(mean_amp_per_phase)
    
    # Calculate Modulation Index (KL divergence from uniform distribution)
    uniform_p = np.ones(n_bins) / n_bins
    # Add small epsilon to avoid log(0)
    p = p + 1e-10
    mi = np.sum(p * np.log(p / uniform_p))
    
    return mi, mean_amp_per_phase, phase_centers

def calculate_pac_comodulogram(lfp_data, fs, phase_freqs, amp_freqs):
    """
    Calculate PAC comodulogram across multiple frequency pairs.
    
    Parameters:
    -----------
    lfp_data : array
        LFP time series data
    fs : float
        Sampling frequency
    phase_freqs : array
        Center frequencies for phase extraction
    amp_freqs : array
        Center frequencies for amplitude extraction
        
    Returns:
    --------
    comodulogram : array
        2D array of MI values (phase_freqs x amp_freqs)
    """
    comodulogram = np.zeros((len(phase_freqs), len(amp_freqs)))
    
    for i, phase_freq in enumerate(phase_freqs):
        for j, amp_freq in enumerate(amp_freqs):
            # Define frequency ranges (±2 Hz around center frequency)
            phase_range = (max(1, phase_freq - 2), phase_freq + 2)
            amp_range = (max(1, amp_freq - 5), amp_freq + 5)
            
            # Skip if phase frequency overlaps with amplitude frequency
            if phase_range[1] >= amp_range[0]:
                comodulogram[i, j] = np.nan
                continue
                
            try:
                phase, amplitude = extract_phase_amplitude(
                    lfp_data, fs, phase_range, amp_range)
                mi, _, _ = calculate_modulation_index(phase, amplitude)
                comodulogram[i, j] = mi
            except:
                comodulogram[i, j] = np.nan
                
    return comodulogram

# Set up directories
plot_dir = os.path.join(granger_causality_path, 'plots', 'phase_amplitude_coupling')
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
phase_freqs = np.arange(2, 20, 2)  # Low frequencies for phase (2-18 Hz)
amp_freqs = np.arange(30, 100, 5)  # High frequencies for amplitude (30-95 Hz)
print(f"Phase frequencies: {phase_freqs} Hz")
print(f"Amplitude frequencies: {amp_freqs} Hz")
print(f"Total frequency pairs to analyze: {len(phase_freqs)} x {len(amp_freqs)} = {len(phase_freqs) * len(amp_freqs)}")

# Check for existing artifacts
pac_raw_baseline_path = os.path.join(artifacts_dir, 'pac_raw_baseline.npy')
pac_raw_stimulus_path = os.path.join(artifacts_dir, 'pac_raw_stimulus.npy')
pac_processed_baseline_path = os.path.join(artifacts_dir, 'pac_processed_baseline.npy')
pac_processed_stimulus_path = os.path.join(artifacts_dir, 'pac_processed_stimulus.npy')
pac_region_names_path = os.path.join(artifacts_dir, 'pac_region_names.npy')
pac_phase_freqs_path = os.path.join(artifacts_dir, 'pac_phase_freqs.npy')
pac_amp_freqs_path = os.path.join(artifacts_dir, 'pac_amp_freqs.npy')

# Check if all artifacts exist
all_artifacts_exist = all([
    os.path.exists(pac_raw_baseline_path),
    os.path.exists(pac_raw_stimulus_path),
    os.path.exists(pac_processed_baseline_path),
    os.path.exists(pac_processed_stimulus_path),
    os.path.exists(pac_region_names_path),
    os.path.exists(pac_phase_freqs_path),
    os.path.exists(pac_amp_freqs_path)
])

if all_artifacts_exist:
    print("All PAC artifacts already exist. Skipping processing and loading existing data...")
    all_raw_pac_baseline = np.load(pac_raw_baseline_path, allow_pickle=True)
    all_raw_pac_stimulus = np.load(pac_raw_stimulus_path, allow_pickle=True)
    all_processed_pac_baseline = np.load(pac_processed_baseline_path, allow_pickle=True)
    all_processed_pac_stimulus = np.load(pac_processed_stimulus_path, allow_pickle=True)
    region_names_list = np.load(pac_region_names_path, allow_pickle=True)
    phase_freqs = np.load(pac_phase_freqs_path)
    amp_freqs = np.load(pac_amp_freqs_path)
    print(f"Loaded existing results for {len(all_raw_pac_baseline)} sessions")
else:
    print("Some or all PAC artifacts missing. Starting processing...")
    
    # Storage for results
    all_raw_pac_baseline = []
    all_raw_pac_stimulus = []
    all_processed_pac_baseline = []
    all_processed_pac_stimulus = []
    region_names_list = []

# Time windows for analysis
baseline_start, baseline_end = 8, 10  # seconds
stimulus_start, stimulus_end = 10, 12  # seconds
print(f"Analysis time windows:")
print(f"  Baseline: {baseline_start}-{baseline_end} seconds")
print(f"  Stimulus: {stimulus_start}-{stimulus_end} seconds")
print(f"Starting PAC analysis across {len(dir_list)} sessions...")

for session_idx, dir_name in enumerate(tqdm(dir_list, desc="Processing sessions")):
        basename = dir_name.split('/')[-1]
        h5_path = glob(dir_name + '/*.h5')[0]
        
        print(f'\n=== Session {session_idx + 1}/{len(dir_list)}: {basename} ===')
        print(f'H5 file: {h5_path}')
        
        try:
            dat = ephys_data(dir_name)
            dat.get_info_dict()
            
            # Get LFP data
            lfp_channel_inds, region_lfps, region_names = \
                dat.return_representative_lfp_channels()
            region_names_list.append(region_names)
            print(f'Found {len(region_names)} regions: {region_names}')
            print(f'LFP data shape: {region_lfps.shape} (regions, trials, time)')
            
            # Flatten data for processing
            flat_region_lfps = np.reshape(
                region_lfps, (region_lfps.shape[0], -1, region_lfps.shape[-1]))
            
            # Remove trials with artifacts
            good_lfp_trials_bool = \
                dat.lfp_processing.return_good_lfp_trial_inds(flat_region_lfps)
            good_lfp_trials = flat_region_lfps[:, good_lfp_trials_bool]
            print(f'Artifact removal: {np.sum(good_lfp_trials_bool)}/{len(good_lfp_trials_bool)} trials retained')
            print(f'Clean LFP data shape: {good_lfp_trials.shape}')
            
            # Get sampling frequency with fallback
            try:
                fs = dat.info_dict['sampling_rate']
                print(f'Sampling rate: {fs} Hz')
            except KeyError:
                # Try alternative key structures or use default
                if 'sampling_rate' in dat.info_dict:
                    fs = dat.info_dict['sampling_rate']
                    print(f'Sampling rate: {fs} Hz')
                else:
                    print(f'Warning: Could not find sampling rate for {basename}, using default 1000 Hz')
                    fs = 1000  # Default sampling rate
            
            # Convert time windows to sample indices
            baseline_samples = [int(baseline_start * fs), int(baseline_end * fs)]
            stimulus_samples = [int(stimulus_start * fs), int(stimulus_end * fs)]
            print(f'Sample indices - Baseline: {baseline_samples}, Stimulus: {stimulus_samples}')
            
            # Initialize storage for this session
            session_raw_pac_baseline = []
            session_raw_pac_stimulus = []
            session_processed_pac_baseline = []
            session_processed_pac_stimulus = []
            
            # Process each region
            print(f'Processing PAC for each region...')
            for region_idx in range(good_lfp_trials.shape[0]):
                region_data = good_lfp_trials[region_idx]
                print(f'  Region {region_idx + 1}/{good_lfp_trials.shape[0]} ({region_names[region_idx]}): {region_data.shape[0]} trials')
                
                # Calculate PAC for raw data
                print(f'    Computing raw PAC...')
                region_raw_pac_baseline = []
                region_raw_pac_stimulus = []
                
                for trial_idx in range(region_data.shape[0]):
                    trial_data = region_data[trial_idx]
                    
                    # Baseline period
                    baseline_data = trial_data[baseline_samples[0]:baseline_samples[1]]
                    baseline_comod = calculate_pac_comodulogram(
                        baseline_data, fs, phase_freqs, amp_freqs)
                    region_raw_pac_baseline.append(baseline_comod)
                    
                    # Stimulus period
                    stimulus_data = trial_data[stimulus_samples[0]:stimulus_samples[1]]
                    stimulus_comod = calculate_pac_comodulogram(
                        stimulus_data, fs, phase_freqs, amp_freqs)
                    region_raw_pac_stimulus.append(stimulus_comod)
                
                session_raw_pac_baseline.append(np.array(region_raw_pac_baseline))
                session_raw_pac_stimulus.append(np.array(region_raw_pac_stimulus))
                
                # Process with Granger preprocessing
                print(f'    Applying Granger preprocessing...')
                this_granger = gu.granger_handler(region_data[np.newaxis, :, :])
                this_granger.preprocess_data()
                preprocessed_data = this_granger.preprocessed_data[0]
                print(f'    Computing preprocessed PAC...')
                
                region_processed_pac_baseline = []
                region_processed_pac_stimulus = []
                
                for trial_idx in range(preprocessed_data.shape[0]):
                    trial_data = preprocessed_data[trial_idx]
                    
                    # Baseline period
                    baseline_data = trial_data[baseline_samples[0]:baseline_samples[1]]
                    baseline_comod = calculate_pac_comodulogram(
                        baseline_data, fs, phase_freqs, amp_freqs)
                    region_processed_pac_baseline.append(baseline_comod)
                    
                    # Stimulus period
                    stimulus_data = trial_data[stimulus_samples[0]:stimulus_samples[1]]
                    stimulus_comod = calculate_pac_comodulogram(
                        stimulus_data, fs, phase_freqs, amp_freqs)
                    region_processed_pac_stimulus.append(stimulus_comod)
                
                session_processed_pac_baseline.append(np.array(region_processed_pac_baseline))
                session_processed_pac_stimulus.append(np.array(region_processed_pac_stimulus))
            
            all_raw_pac_baseline.append(session_raw_pac_baseline)
            all_raw_pac_stimulus.append(session_raw_pac_stimulus)
            all_processed_pac_baseline.append(session_processed_pac_baseline)
            all_processed_pac_stimulus.append(session_processed_pac_stimulus)
            
            # Save intermediate results after each session
            print(f'    Saving intermediate results after session {session_idx + 1}...')
            np.save(pac_raw_baseline_path, all_raw_pac_baseline, allow_pickle=True)
            np.save(pac_raw_stimulus_path, all_raw_pac_stimulus, allow_pickle=True)
            np.save(pac_processed_baseline_path, all_processed_pac_baseline, allow_pickle=True)
            np.save(pac_processed_stimulus_path, all_processed_pac_stimulus, allow_pickle=True)
            np.save(pac_region_names_path, region_names_list, allow_pickle=True)
            np.save(pac_phase_freqs_path, phase_freqs)
            np.save(pac_amp_freqs_path, amp_freqs)
            
        except Exception as e:
            print(f'ERROR processing {basename}: {e}')
            continue

    print(f'\nCompleted processing {len(all_raw_pac_baseline)} sessions successfully')

    # Final save of results
    print(f'\nSaving final results to {artifacts_dir}...')
    np.save(pac_raw_baseline_path, all_raw_pac_baseline, allow_pickle=True)
    np.save(pac_raw_stimulus_path, all_raw_pac_stimulus, allow_pickle=True)
    np.save(pac_processed_baseline_path, all_processed_pac_baseline, allow_pickle=True)
    np.save(pac_processed_stimulus_path, all_processed_pac_stimulus, allow_pickle=True)
    np.save(pac_region_names_path, region_names_list, allow_pickle=True)
    np.save(pac_phase_freqs_path, phase_freqs)
    np.save(pac_amp_freqs_path, amp_freqs)

    print("PAC analysis complete. Results saved to artifacts directory.")
    print(f"Saved files:")
    print(f"  - pac_raw_baseline.npy: {len(all_raw_pac_baseline)} sessions")
    print(f"  - pac_raw_stimulus.npy: {len(all_raw_pac_stimulus)} sessions") 
    print(f"  - pac_processed_baseline.npy: {len(all_processed_pac_baseline)} sessions")
    print(f"  - pac_processed_stimulus.npy: {len(all_processed_pac_stimulus)} sessions")
    print(f"  - pac_region_names.npy, pac_phase_freqs.npy, pac_amp_freqs.npy")

############################################################
# Generate plots
############################################################

print(f'\n=== Starting plot generation ===')

# Data is already loaded from either processing or loading existing artifacts
print(f'Using data for {len(all_raw_pac_baseline)} sessions')

# Average across sessions and trials for each condition
def average_pac_data(pac_data_list):
    """Average PAC data across sessions and trials"""
    all_sessions = []
    for session_data in pac_data_list:
        session_avg = []
        for region_data in session_data:
            # Average across trials
            region_avg = np.nanmean(region_data, axis=0)
            session_avg.append(region_avg)
        all_sessions.append(session_avg)
    
    # Convert to array and average across sessions
    all_sessions = np.array(all_sessions)
    return np.nanmean(all_sessions, axis=0)

print('Averaging PAC data across sessions and trials...')
avg_raw_baseline = average_pac_data(all_raw_pac_baseline)
avg_raw_stimulus = average_pac_data(all_raw_pac_stimulus)
avg_processed_baseline = average_pac_data(all_processed_pac_baseline)
avg_processed_stimulus = average_pac_data(all_processed_pac_stimulus)
print(f'Averaged data shape: {avg_raw_baseline.shape} (regions, phase_freqs, amp_freqs)')

# Get region names (assuming consistent across sessions)
region_names = region_names_list[0]
print(f'Generating plots for {len(region_names)} regions: {region_names}')

# Plot comodulograms
conditions = ['Raw Baseline', 'Raw Stimulus', 'Processed Baseline', 'Processed Stimulus']
data_arrays = [avg_raw_baseline, avg_raw_stimulus, avg_processed_baseline, avg_processed_stimulus]

for region_idx, region_name in enumerate(region_names):
    print(f'Creating comodulogram plots for region {region_idx + 1}/{len(region_names)}: {region_name}')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for cond_idx, (condition, data_array) in enumerate(zip(conditions, data_arrays)):
        ax = axes[cond_idx]
        
        # Plot comodulogram
        im = ax.imshow(data_array[region_idx], 
                      extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
                      aspect='auto', origin='lower', cmap='viridis')
        
        ax.set_xlabel('Amplitude Frequency (Hz)')
        ax.set_ylabel('Phase Frequency (Hz)')
        ax.set_title(f'{condition} - {region_name}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Modulation Index')
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f'pac_comodulogram_{region_name}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'  Saved: {plot_path}')

# Plot comparison between raw and processed data
print('Creating comparison plots between raw and processed data...')
for region_idx, region_name in enumerate(region_names):
    print(f'Creating comparison plot for region {region_idx + 1}/{len(region_names)}: {region_name}')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Baseline comparison
    axes[0, 0].imshow(avg_raw_baseline[region_idx], 
                     extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
                     aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title(f'Raw Baseline - {region_name}')
    axes[0, 0].set_ylabel('Phase Frequency (Hz)')
    
    axes[0, 1].imshow(avg_processed_baseline[region_idx], 
                     extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
                     aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title(f'Processed Baseline - {region_name}')
    
    # Stimulus comparison
    axes[1, 0].imshow(avg_raw_stimulus[region_idx], 
                     extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
                     aspect='auto', origin='lower', cmap='viridis')
    axes[1, 0].set_title(f'Raw Stimulus - {region_name}')
    axes[1, 0].set_xlabel('Amplitude Frequency (Hz)')
    axes[1, 0].set_ylabel('Phase Frequency (Hz)')
    
    im = axes[1, 1].imshow(avg_processed_stimulus[region_idx], 
                          extent=[amp_freqs[0], amp_freqs[-1], phase_freqs[0], phase_freqs[-1]],
                          aspect='auto', origin='lower', cmap='viridis')
    axes[1, 1].set_title(f'Processed Stimulus - {region_name}')
    axes[1, 1].set_xlabel('Amplitude Frequency (Hz)')
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Modulation Index')
    
    plot_path = os.path.join(plot_dir, f'pac_comparison_{region_name}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'  Saved: {plot_path}')

print(f"\nPAC plotting complete. All plots saved to: {plot_dir}")
print(f"Generated {len(region_names) * 2} plot files total")
