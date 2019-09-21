# Import libraries
import numpy as np
import pylab as plt
from scipy.special import factorial

# Generate single neuron poisson firing with 1 changepoint
dt = 0.01
tmax = 10
tvec = np.arange(0,tmax,dt)
t_changepoint = np.floor(7/dt).astype('int')
firing_rates = [5,10]
spikes = np.concatenate( ( (np.random.rand(t_changepoint))< firing_rates[0]*dt, \
                       (np.random.rand(len(tvec)-t_changepoint))< firing_rates[1]*dt ) )

plt.plot(tvec,spikes)
plt.title('Changepoint at %.2f seconds' % (t_changepoint*dt))

"""
Loop through timepoints splitting data to before and after changepoint, and calculate ratio of likelihood for
data split into two parts vs no split
Assuming spikes are poisson processes probability given by "https://en.wikipedia.org/wiki/Poisson_distribution" 
--> section on "Probability of events for a Poisson distribution"
"""

log_prob_change = np.zeros(tvec.shape)
log_prob_nochange = np.zeros(tvec.shape)

for time in range(1,len(tvec)):
    # Sum of spikes before curren time
    n1 = sum(spikes[0:time])
    t1 = time*dt
    # Sum of spikes after current time
    n2 = sum(spikes[time:len(tvec)])
    t2 = (len(tvec)-time)*dt
    # Sum of spikes for entire trial
    nt = sum(spikes)
    tt = len(tvec)*dt
    
    expected_n = t1*nt/tt
    log_prob_change[time] = -expected_n + n1*np.log(expected_n) - np.log(factorial(n1))

# Mark maximum of ratio
changepoint_ind = np.nanargmin(log_prob_change)
changepoint_est = tvec[changepoint_ind] 
    
plt.title('Log-likelihood ratio assuming constant firing rate')
plt.xlabel('Time (s)')
plt.ylabel('Log-likelihood ratio')
plt.plot(tvec,log_prob_change)
plt.vlines(changepoint_est,np.nanmin(log_prob_change),np.nanmax(log_prob_change))  

# Estimate firing rates before and after changepoint and compare to original

f_prior = sum(spikes[0:changepoint_ind])/changepoint_est
f_post = sum(spikes[changepoint_ind:len(tvec)])/changepoint_est
print('Actual transition = %.2f \n Estimated transition = %.2f \n Actual prior = %.2f \n Estimated prior = %.2f \n Actual post = %.2f \n Estimated post = %.2f' \
     % (t_changepoint*dt, changepoint_est,firing_rates[0],f_prior,firing_rates[1],f_post))
