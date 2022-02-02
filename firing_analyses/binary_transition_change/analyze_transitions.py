"""
Calculate change of activity across palatability transition
Determine transition on a per-fit basis using timing of transition

Determine:
    1) Fraction of taste-responsive neurons
    2) Taste descriminative neurons
    3) Raw and conditional fractions of neurons with the following patterns:
        a) Neurons which transition for any taste
            - Plot bar plot for how many tastes the neurons respond to
            - Each bar can be broken down by what tastes are involved
        b) Transition for palatable but not unpalatable (and vice versa)
        c) Positive transition for palatable but negative for unpalatable
            (and vice versa)
        ** Evaluate conditional fraction of b) and c) relative to a)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import itertools as it
