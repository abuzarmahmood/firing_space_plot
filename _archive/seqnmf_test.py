#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:30:43 2019

@author: abuzarmahmood
"""

from seqnmf import seqnmf, plot, example_data
import pylab as plt


[W, H, cost, loadings, power] = seqnmf(example_data, K=5, L=100, Lambda=0.001, plot_it=False)

plot(W, H).show()