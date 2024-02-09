Perform granger causality on transition aligned activity
Since granger is performed on all trials together, we will have to repeat the 
process for each transition.
Instead of aligning the whole trial, might be computationally faster
to take snippets of time around each transition.

Can break process down:
1. Get changepoint positions for datasets
2. Get LFP data for datasets
3. Generate aligned snippets of LFP (keep track of changepoints before and after current one)
4. Perform granger
5. **Still figuring out how to re-align / merge granger post-hoc
