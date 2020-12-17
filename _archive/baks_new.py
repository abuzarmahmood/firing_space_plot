import numpy as np
from scipy.special import gamma

def BAKS_b(SpikeTimes, Time):
    
    N = len(SpikeTimes)
    a = 4
    b = N**0.8

    common_array = np.array([Time - spike for spike in SpikeTimes])

    num_array = (common_array/2) + 1/b
    sumnum_b = np.sum(num_array**(-a), axis = 0) 
    sumdenum_b = np.sum(num_array**(-a-0.5), axis = 0) 
    h_b = (gamma(a)/gamma(a+0.5))*(sumnum_b/sumdenum_b)

    C = (1/(np.sqrt(2*np.pi)*h_b))
    D = ((2*h_b)**2)
    K_b = C *np.exp(-common_array/D)
    FiringRate_b = np.sum(K_b, axis=0)

    return FiringRate_b
