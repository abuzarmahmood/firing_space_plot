"""
% Bayesian Adaptive Kernel Smoother (BAKS)
% BAKS is a method for estimating firing rate from spike train data that uses kernel smoothing technique 
% with adaptive bandwidth determined using a Bayesian approach
% ---------------INPUT---------------
% - SpikeTimes : spike event times [nSpikes x 1]
% - Time : time at which the firing rate is estimated [nTime x 1]
% - a : shape parameter (alpha) -> Current value taken from paper
% - b : scale paramter (beta)-> Current value taken from paper
% ---------------INPUT---------------
% - h : adaptive bandwidth [nTime x 1]
% - FiringRate : estimated firing rate [nTime x 1]
% More information, please refer to "Estimation of neuronal firing rate using Bayesian adaptive kernel smoother (BAKS)"

Python adaptation of BAKS
"""
import numpy as np
from scipy.special import gamma

def BAKS(SpikeTimes, Time):
    
    
    N = len(SpikeTimes)
    a = 4
    b = N**0.8
    sumnum = 0; sumdenum = 0
    
    for i in range(N):
        numerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a)
        denumerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a-0.5)
        sumnum = sumnum + numerator
        sumdenum = sumdenum + denumerator
    h = (gamma(a)/gamma(a+0.5))*(sumnum/sumdenum)
    
    FiringRate = np.zeros((len(Time)))
    for j in range(N):
        K = (1/(np.sqrt(2*np.pi)*h))*np.exp(-((Time-SpikeTimes[j])**2)/((2*h)**2))
        FiringRate = FiringRate + K
        
    return FiringRate