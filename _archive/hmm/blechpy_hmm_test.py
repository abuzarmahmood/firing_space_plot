import blechpy
from blechpy.analysis import poissonHMM as phmm
import pylab as plt
import numpy as np

dat_dir =    '/media/bigdata/Abuzar_Data/AM34/AM34_4Tastes_201219_130532'

blech_dat = blechpy.load_dataset(dat_dir)

#blech_dat = blechpy.dataset(dat_dir)
#blech_dat.initParams()
#blech_dat.make_unit_arrays()
#blech_dat.create_trial_list()
#blechpy.dio.h5io.write_electrode_map_to_h5(blech_dat.h5_file, blech_dat.electrode_mapping)


#din_channel = 0 # Channel of the digital input that the trials you wish to fit are on
#unit_type = 'single' # This can be single (for all single units),
#                     # pyramidal (only single unit regular-spiking cells)
#                     # or interneuron (only single unit fast-spiking cells)
#
##The parameters below are optional
#time_start = 0  # Time start in ms
#time_end = 2000 # Time end in ms
#dt = 0.01 # desired bin size for the return spike array in seconds, default is 0.001 seconds
#
#spike_array, dt, time = \
#        phmm.get_hmm_spike_data(dat_dir, unit_type, din_channel, time_start, time_end, dt)
#
#
#n_states = 3  # Number of predicted states in your data
#cost_window = 0.25 # Size of the window with which to compute the cost function in seconds
#                   # The cost of a model is computed by predicting the firing rate of each
#                   # neuron in each bin and comparing it to the actual data. The cost is
#                   # currently the average euclidean distance of the prediction from the actual.
#                   # Averaged over trials
#                   # This is optional: default is 0.25 seconds
#                   # Right now the cost is only used for your own evaluation of how good your model fits the data
#
## Initializing the model
#model = phmm.PoissonHMM(n_states)#, cost_window=cost_window) # Notice you're not giving it the data yet
#
## Fitting the model
#convergence_threshold = 1e-4  # This is optional and this number is based on nothing
#                              # So far this works well for fitting simulated data,
#                              # But I have not yet seen actual data meet this criteria
#max_iter = 300  # This is also optional, the default is 1000
#
#model.randomize(spike_array, dt, time)
#model.fit(spike_array, dt, time, max_iter=max_iter)#, convergence_thresh=convergence_threshold)
#
#plt.plot(model.ll_hist);plt.show()

############################
h = phmm.HmmHandler(dat_dir)
h.add_params({'n_states' : 4,
                'area':None, 
                'dt':0.05,
                'time_start' : -250,
                'time_end' : 2000,
                'notes':'sequential'})
h.run(constraint_func = phmm.sequential_constraint)
h.plot_saved_models()

############################

# Initializing
#handler = phmm.HmmHandler(dat_dir)
## Save directory is automatically made inside the recording directory,
## but you can also specificy another place witht eh save_dir keyword argument.
## You can also pass the params directly when initializing the handler, but
## I just split it here you can see how to add new parameters later.
#handler.add_params(phmm.HMM_PARAMS)
#
## Running the handler
#handler.run() # to overwrite existing models pass overwrite=True
#
## Looking at the parameters already in the handler
#parameter_overview = handler.get_parameter_overview() # this is a pandas DataFrame
#
## Looking at the parameters and fitted model stats
#data_overview = handler.get_data_overview() # also a pandas DataFrame with extra info such as cost and BIC
#
## The matrices defining each HMM and the best sequences can be access from teh HDF5 store directly. They can also be access programatically with:
#hdf5_file = handler.h5_file
#model, model_params = phmm.load_hmm_from_hdf5(hdf5_file, hmm_id)
## Now you have the PoissonHMM object with the fitted model parameters and can do anything with it.
## Only information lost is the model history, every model is set to the best model in the history before saving
