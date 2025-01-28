"""
HMM:
    - Fit HMM over min - max states
    - Calc inferred number of states using cross-validation
    - Calculate state transitions per trial
    - Calculate error in state transitions

Non-switch changepoint:
    - Fit non-switch changepoint model over min - max states
    - Calculate inferred number of states using ELBO
    - Calculate error in state transitions
        - Each trial will have same number of transitions

Switch changepoint:
    - Fit switch changepoint model over fixed number of states
        - Variable number of states in other models is to account
            for trial variability
    - Fit with both MCMC and VI
    - Calculate error in state transitions
        - Each trial will have different number of transitions

For each model:
    - Record time taken to fit model (and across all models if for multiple states)

For changepoint models:
    - Save variables as histograms with 1 percentile bins

"""
