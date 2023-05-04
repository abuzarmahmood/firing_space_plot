"""
On single trials, does strength of granger-causal interactions
predict accuracy of prediction?

Parts:
    1- Causality:
        a- Directions : Forward, Backward
        b- Epochs : Middle, Late
        c- Frequencies : All, Low, High, Individual Bands
    2- Prediction:
        a- Epochs : Middle, Late
        b- Metrics : Entropy, Binary Prediction, Probability

Causality should only predict prediction accuracy in the same or later epoch,
i.e.    causality [middle] -> prediction [middle, late]
        causality [middle, late] -> prediction [late]

Might be best to do 2 runs of the analysis broken up by causality epoch,
and then break each run by prediction metric.

First pass:
    Simply look at changes in causality given high or low prediction metric.
"""

epoch_lims = [[300,800],[800,1300]]
epoch_names = ['middle','late']

frequency_lims = \
        [[0,100],[0,30],[30,100],[4,8],[8,12],[12,30],[30,60],[60,100]]
frequency_names = \
        ['all','low','high','theta','alpha','beta','low_gamma','high_gamma']
