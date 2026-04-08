# Dependencies and I/O Documentation for multibehavior_transition.py

## External Dependencies

### Python Packages
- `pytau.changepoint_model` - Custom changepoint modeling module
  - Used functions: `GaussianChangepointMeanVar2D`, `advi_fit`
- `pickle` - Standard library for serialization
- `os` - Standard library for file system operations
- `matplotlib.pyplot` - Plotting library
- `pandas` - Data manipulation library
- `scipy.stats.zscore` - Statistical functions for z-score normalization
- `numpy` - Numerical computing library
- `tqdm` - Progress bar library

### Internal Module Dependencies
- `pytau.changepoint_model.GaussianChangepointMeanVar2D` - Gaussian changepoint model class
- `pytau.changepoint_model.advi_fit` - ADVI (Automatic Differentiation Variational Inference) fitting function

## Input Files

### Primary Input
**File**: `behavior_dict_df_all_two_test_animals.pkl`
**Location**: `/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions/data/`
**Format**: Pickled pandas DataFrame
**Contents**: 
- DataFrame with columns:
  - `session_name` - String identifier for the session
  - `behavior_dict` - Dictionary containing:
    - `behavior_array` - numpy array of shape (trials × taste) × behavior
    - `global_trial_map` - Dictionary mapping (taste, rel_trial) tuples to absolute trial numbers

### Alternative Input (Commented Out)
**File**: `CM74_CTATest2_h2o_nacl_lowqhcl_highqhcl_250614_120546_behavior_score_dict.pkl`
**Location**: Same data directory
**Format**: Pickled dictionary (legacy format)

## Output Files

### Plots
**Directory**: `/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions/plots/two_test_changepoint_plots/`
**Files Generated** (per session):
1. `{basename}_behavior_changepoint_plots.png`
   - 3 × n_tastes subplot grid
   - Row 1: Raw behavior scores by relative trial number
   - Row 2: Z-scored behavior scores by relative trial number
   - Row 3: Changepoint sample histograms with mode marked

2. `{basename}_behavior_changepoint_plots_absolute_trial.png`
   - 3 × n_tastes subplot grid
   - Row 1: Raw behavior scores by absolute trial number
   - Row 2: Z-scored behavior scores by absolute trial number
   - Row 3: Changepoint sample histograms with mode marked (absolute trial numbers)

### Data Output
**File**: `behavior_changepoint_out_df.pkl`
**Location**: `/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions/data/`
**Format**: Pickled pandas DataFrame
**Contents**:
- Columns:
  - `basename` - Session name
  - `taste` - Taste identifier
  - `taste_trials` - Array of absolute trial numbers for this taste
  - `taste_behavior_array` - Raw behavior scores for this taste
  - `zscored_taste_behavior_array` - Z-scored behavior scores for this taste
  - `tau_samples` - Changepoint samples from ADVI inference (1000 samples)
  - `mode_changepoint` - Mode of the changepoint distribution (most likely changepoint trial)

## Processing Pipeline

1. **Data Loading**: Load behavior score DataFrame from pickle file
2. **Per-Session Processing**:
   - Extract behavior array and trial mapping
   - Split data by taste
   - Z-score normalization (with small noise addition to avoid zero variance)
3. **Changepoint Detection**:
   - Model: `GaussianChangepointMeanVar2D` with 2 states
   - Inference: ADVI with 50,000 fit iterations and 1,000 samples
   - Output: Posterior samples of changepoint locations (tau)
4. **Visualization**: Generate plots showing behavior scores and changepoint distributions
5. **Results Compilation**: Aggregate all results into output DataFrame with mode changepoints

## Model Parameters

- **Model Type**: Gaussian Changepoint Mean and Variance (2D)
- **Number of States**: 2
- **ADVI Fit Iterations**: 50,000
- **Posterior Samples**: 1,000
- **Data Preprocessing**: Z-score normalization + 0.1 * random noise

## Notes for Repository Migration

- The `pytau` module is a custom dependency that must be migrated or replaced
- Hard-coded paths should be parameterized:
  - `/media/bigdata/firing_space_plot/emg_analysis/CM_behavior_transitions/`
- Consider making model parameters (n_states, fit iterations, samples) configurable
- The script processes all sessions in a loop - consider parallelization for large datasets
