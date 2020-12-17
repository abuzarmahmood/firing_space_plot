# =============================================================================
# Import stuff
# =============================================================================
# import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import numpy as np # module for low-level scientific computing
import pandas as pd
import glob
import os

#Import plotting utilities
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Packages for mcmc
import pymc3 as pm
import theano.tensor as tt
import theano

# =============================================================================
# =============================================================================
# #Read in file
# =============================================================================
# Get name of directory where the data files and pickle file sits, 
# and change to that directory for processing
dir_name = '/media/bigdata/brads_data/' 
pkl_file = glob.glob(os.path.join(dir_name,'./*df'))

#Read in dataframe from CSV
df = pd.read_pickle(pkl_file[0])

#remove unnecessary index column
df.drop(['index'], axis=1,inplace=True)

#create a minute bin identifier column
df['min_bin'] = df['Start_time_ms']//60000

#read event types
all_events = sorted(list(df.Event_1.unique()))

#Query positive (flagged as '1') and negative events (flagged as '0')
# Rename quadrant moves to 'move'
affective_df = df
str_to_rename = ['Q1','Q2','Q3','Q4']
affective_df.loc[affective_df['Event_1']\
        .str.contains("|".join(str_to_rename)),'Event_1'] = 'move'
#affective_df = affective_df.loc[affective_df['Event_1'].\
#        str.contains('move|Rearing')]

# Remove unwanted columns
wanted_cols = ['Animal','Condition','Experimental_group','Event_1',\
        'code','min_bin']
affective_df = affective_df[wanted_cols]

# Plot data to make sure things look good
# Group by Animal and Condition
grouped_df = list(affective_df.groupby(['Animal','Condition']))
sns.scatterplot(data = grouped_df[3][1], x = 'min_bin', y = 'Event_1')
plt.show()

# Pull out LiCl datagroups
licl_groups = [x[1] for x in grouped_df if x[1].Condition.unique()[0]=='LiCl']
fig,ax = plt.subplots(5,1, sharex=True)
for dat,this_ax in zip(licl_groups,ax):
    plt.sca(this_ax)
    sns.scatterplot(data = dat, x = 'min_bin', y = 'Event_1')
plt.show()

# Convert dataframe events to timeseries
# Give each event type a row
licl_timeseries = []
mins = 20
for dat in licl_groups:
    unique_events = dat.Event_1.unique()
    unique_event_dict = dict(zip(unique_events,range(len(unique_events))))
    event_array = np.zeros((len(unique_events),mins))
    for row in range(dat.shape[0]):
        row_ind = unique_event_dict[dat.iloc[row].Event_1]
        col_ind = int(dat.iloc[row].min_bin)
        event_array[row_ind, col_ind] = 1
    licl_timeseries.append(event_array)

########################################
## MCMC Switchpoint
########################################

dat = licl_timeseries[3]
plt.imshow(dat,origin='lower');plt.show()
idx = np.arange(dat.shape[-1])

# Define number of states and mean values for each state
states = 2
split_list = np.array_split(dat,states,axis=-1)
# Cut all to the same size
mean_vals = np.squeeze(np.array([np.mean(x,axis=-1) for x in split_list]))
mean_vals += 0.01 # To avoid zero starting prob

# Add lambda_latent to model
with pm.Model() as model:
        
    # SAME LAMBDAS ACROSS ALL TRIALS
    # Finite, but somewhere on the lower end, Beta prior
    a_lambda = 2
    b_lambda = 5
        
    lambda_latent = pm.Beta('lambda_latent', a_lambda, b_lambda, testval = mean_vals, 
                               shape = (states,mean_vals.shape[1]))

    #print(lambda_latent.tag.test_value.shape)
    #plt.imshow(lambda_latent.tag.test_value.T,aspect='auto');plt.show();

# Find midpoints for swithpoints
even_switches = np.linspace(0,idx.max(),states+1)
even_switches_normal = even_switches/np.max(even_switches)
#print(even_switches)
#print(even_switches_normal[1:(states)])

# Add switchpoint to model
with model:
        
    # INDEPENDENT TAU FOR EVERY TRIAL
    a_tau = pm.HalfNormal('a_tau', 3.)
    b_tau = pm.HalfNormal('b_tau', 3.) 
    tau_latent = pm.Beta('tau_latent', a_tau, b_tau) 
    tau = pm.Deterministic('tau', idx.min() + (idx.max() - idx.min()) * tau_latent)

# Add lambda to model
with model:
        
    # Assign lambdas to time_bin indices using sigmoids centered on switchpoints
    # Refer to https://www.desmos.com/calculator/yisbydv2cq

    weight_1_stack = tt.nnet.sigmoid(2 * (idx - tau))
    weight_2_stack = tt.nnet.sigmoid(2 * (idx - tau))

    lambda_ = np.multiply(1 - weight_1_stack, lambda_latent[0][:,np.newaxis]) + \
            np.multiply(weight_2_stack, lambda_latent[1][:,np.newaxis])

    #print(weight_1_stack.tag.test_value.shape)
    #print(lambda_.tag.test_value.shape)
    #plt.imshow(lambda_.tag.test_value,aspect='auto');plt.colorbar();plt.show()

# Using Bernoulli likelihood for count data
with model:
        observation = pm.Bernoulli("obs", lambda_, observed=dat)

# Show model graph
#g = pm.model_to_graphviz(model)
#g.view()

# Run sampling
# Sample for longer but thin trace at end
with model:
    #step= pm.NUTS()
    step= pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step = step, chains = 4, cores = 4)

# Make sure traces converged and are not autocorrelated
#trace_summary = pm.summary(trace)
#trace_summary.r_hat

#pm.autocorrplot(trace['tau_latent'],max_lag = 400)
#pm.autocorrplot(trace['tau_latent'][::50],max_lag = 400)
#plt.show()

# Perform 10x pruning
trace = trace[::50]

# Create forestplot
pm.forestplot(trace, var_names = ['tau']);plt.show()

# Pull out tau samples
tau_samples = trace['tau']

# Plot output
tau_ecdf = np.cumsum(np.histogram(tau_samples,mins)[0])/tau_samples.shape[0]
plt.imshow(dat)
plt.plot(tau_ecdf*dat.shape[0])
plt.show()
