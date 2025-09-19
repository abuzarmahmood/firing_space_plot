from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_spd_matrix
from pomegranate import *
from time import time
from scipy.spatial import distance_matrix as distmat 
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

import sys
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from visualize import gen_square_subplots, imshow

sys.path.append('/home/abuzarmahmood/Desktop/blech_clust')
from clustering import *

numpy.random.seed(0)


def gen_samples(n_components, samples,dims):
    mean_array = np.random.random((n_components, dims)) * 100
    cov_array = np.array([make_spd_matrix(dims)*100 \
            for x in range(n_components)])
    #cov_array = np.array(cov_list)
    dat_array = \
            np.concatenate([np.random.multivariate_normal(this_mean,this_cov,samples) \
                            for this_mean,this_cov in zip(mean_array,cov_array)])
    labels = np.concatenate([[x]*samples for x in range(n_components)])
    return mean_array, cov_array, dat_array, labels

# Use greedy assignment, just in case there are overlaps
def get_greedy_label(dist_array):
    sorted_order = []
    for row_num in range(dist_array.shape[0]):
        this_order = np.argsort(dist_array[row_num])
        for this_label in this_order:
            if this_label in sorted_order:
                pass
            else:
                sorted_order.append(this_label)
                break
    return sorted_order

def run_fit_comparison(n_components, samples, dims):
    mean_array, cov_array, dat_array, labels = \
            gen_samples(n_components, samples, dims)

    ########################################
    ## Plot Data 
    ########################################
    #downsampling = 1
    #plt.scatter(*dat_array.T[:2][:,::downsampling], c=labels[::downsampling])
    #plt.show()

    ########################################
    ## Compare fits 
    ########################################
    tol = 1e-6
    n_iter = 1000
    n_init = 10

    start2 = time()
    pomgmm = GeneralMixtureModel.\
            from_samples(MultivariateGaussianDistribution, 
                    X = dat_array,
                    n_components = n_components,
                    stop_threshold = tol,
                    n_init = n_init,
                    batch_size = None,
                    batches_per_epoch = 20)
                    #max_iterations = n_iter,
    stop2 = time()

    start1 = time()
    skgmm = GaussianMixture(
            n_components = n_components, 
            covariance_type = 'full', 
            tol = tol, 
            max_iter = n_iter, 
            n_init = n_init)
    skgmm.fit(dat_array)
    stop1 = time()

    sk_pred_labels = skgmm.predict(dat_array)
    pom_pred_labels = pomgmm.predict(dat_array)

    ## Sort predicted labels by matching to original data
    sk_mean_dists = distmat(skgmm.means_ , mean_array)
    pom_means = np.array([x.parameters[0] for x in pomgmm.distributions])
    pom_mean_dists = distmat(pom_means , mean_array)

    sk_sorted_order = get_greedy_label(sk_mean_dists) 
    pom_sorted_order = get_greedy_label(pom_mean_dists) 

    sk_sorted_labels = np.zeros(sk_pred_labels.shape)
    for num,val in enumerate(sk_sorted_order):
        sk_sorted_labels[np.where(sk_pred_labels == num)[0]] = val

    pom_sorted_labels = np.zeros(pom_pred_labels.shape)
    for num,val in enumerate(pom_sorted_order):
        pom_sorted_labels[np.where(pom_pred_labels == num)[0]] = val

    sk_accuracy = np.mean(sk_sorted_labels==labels)
    pom_accuracy = np.mean(pom_sorted_labels==labels)
    sk_fit_time = stop1-start1
    pom_fit_time = stop2-start2

    return [sk_fit_time,pom_fit_time],[sk_accuracy, pom_accuracy]

########################################
## Create Data
########################################

sample_size_list = np.vectorize(np.int)(np.logspace(2,4,8))

n_components = 7
#samples = 10000//n_components
dims = 8
repeats = 5

# Serialize arguments so all iterations can be run in parallel
args_array = np.zeros((len(sample_size_list)*repeats,3))
args_array[:,0] = n_components
args_array[:,1] = np.repeat(sample_size_list, repeats)
args_array[:,2] = dims
args_array = np.vectorize(np.int)(args_array)

outs = Parallel(n_jobs = cpu_count() - 2)\
        (delayed(run_fit_comparison)(*args) for args in tqdm(args_array))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

time_list, accuracy_list = [np.array(x) for x in list(zip(*outs))]
chunked_time = list(chunks(time_list,repeats))
chunked_accuracy = list(chunks(accuracy_list,repeats))

#time_pair, accuracy_pair = [np.array(x) for x in list(zip(*outs))]
#time_list.append(time_pair)
#accuracy_list.append(accuracy_pair)

#time_list = []
#accuracy_list = []
#for samples in tqdm(sample_size_list):
#    #outs = [run_fit_comparison(n_components, samples, dims) for x in range(repeats)]
#    outs = Parallel(n_jobs = repeats)\
#            (delayed(run_fit_comparison)(n_components,samples,dims) \
#        for x in range(repeats))
#    time_pair, accuracy_pair = [np.array(x) for x in list(zip(*outs))]
#    time_list.append(time_pair)
#    accuracy_list.append(accuracy_pair)

#time_list.append([sk_fit_time,pom_fit_time])
#accuracy_list.append([sk_accuracy, pom_accuracy])

#print(f'SK Fit  Time : {sk_fit_time}')
#print(f'Pom Fit  Time : {pom_fit_time}')
#print(f'SK Accuracy : {sk_accuracy}')
#print(f'Pom Accuracy : {pom_accuracy}')

########################################
## Plot Predicted Data 
########################################
#fig,ax = plt.subplots(1,2)
#ax[0].scatter(*dat_array.T[:2][:,::downsampling], c=sk_pred_labels[::downsampling])
#ax[1].scatter(*dat_array.T[:2][:,::downsampling], c=pom_pred_labels[::downsampling])
#plt.show()


########################################
## Plot comparison results 
########################################
#time_array = np.array(time_list).T
#accuracy_array = np.array(accuracy_list).T
time_array = np.array(chunked_time).T
accuracy_array = np.array(chunked_accuracy).T
broadcasted_sample_size = np.broadcast_to(sample_size_list, time_array.shape)

########################################
## Comparison Overview 
########################################
mean_time = np.mean(time_array,axis=1)
sd_time = np.std(time_array,axis=1)
mean_accuracy = np.mean(accuracy_array,axis=1)
sd_accuracy = np.std(accuracy_array,axis=1)

time_ratio = time_array[0]/time_array[1]
mean_ratio = np.mean(time_ratio,axis=0)
sd_ratio = np.std(time_ratio,axis=0)

cmap = plt.get_cmap("tab10")
fig,ax = plt.subplots(4,1,sharex=True, figsize=(7,20))
plt.xscale('log')
ax[0].plot(sample_size_list, mean_time[0], '-x', label = 'SKLearn', c = cmap(0))
ax[0].plot(sample_size_list, mean_time[1], '-x', label = 'Pomegranate', c = cmap(1))
ax[0].legend()
#ax[0].fill_between(sample_size_list, 
#        mean_time[0] - sd_time[0], mean_time[0] + sd_time[0], 
#        alpha = 0.5, color = cmap(0))
#ax[0].fill_between(sample_size_list, 
#        mean_time[1] - sd_time[1], mean_time[1] + sd_time[1], 
#        alpha = 0.5, color = cmap(1))
ax[0].scatter(broadcasted_sample_size[0], time_array[0], c = cmap(0), edgecolor='k')
ax[0].scatter(broadcasted_sample_size[1], time_array[1], c = cmap(1), edgecolor='k')
ax[0].set_yscale('log')
ax[0].set_ylabel('Run time (s)')
ax[1].plot(sample_size_list, mean_time[0]/mean_time[1])
ax[1].fill_between(sample_size_list, mean_ratio - sd_ratio,
        mean_ratio + sd_ratio, alpha = 0.5)
ax[1].scatter(broadcasted_sample_size[0], time_ratio, edgecolor = 'k')
ax[1].axhline(1, linewidth = 2, color = 'red')
ax[1].set_ylabel('Run time ratio')
#ax[1].set_yscale('log')
ax[2].plot(sample_size_list, mean_accuracy[0], '-x', label = 'SKLearn')
ax[2].plot(sample_size_list, mean_accuracy[1], '-x', label = 'Pomegranate')
ax[2].fill_between(sample_size_list, 
        mean_accuracy[0] - sd_accuracy[0], mean_accuracy[0] + sd_accuracy[0], 
        alpha = 0.5)
ax[2].fill_between(sample_size_list, 
        mean_accuracy[1] - sd_accuracy[1], mean_accuracy[1] + sd_accuracy[1], 
        alpha = 0.5)
ax[2].set_ylabel('Prediction Accuracy')
ax[-1].plot(sample_size_list, np.max(accuracy_array,axis=1).T, '-x')
ax[-1].set_ylabel('Best Prediction Accuracy')
ax[-1].set_xlabel('Training set size')
#plt.tight_layout()
plt.suptitle(f'{n_components} Components, {dims} Dims, {repeats} Repeats')
plt.show()

# Plot scatters of accuracy vs time for all sample_size
fig,ax = gen_square_subplots(len(sample_size_list))
for this_ax, this_acc, this_time \
        in zip(ax.flatten(), accuracy_array.T, time_array.T):
    this_ax.scatter(this_time.T[0], this_acc.T[0], 
            label = 'SKLearn', edgecolor = 'k')
    this_ax.scatter(this_time.T[1], this_acc.T[1], 
            label = 'Pomegranate', edgecolor = 'k')
for this_ax in ax[:,0]:
    this_ax.set_ylabel('Accuracy')
for this_ax in ax[-1,:]:
    this_ax.set_xlabel('Time Elapsed (s)')
for this_ax, this_val in zip(ax.flatten(),sample_size_list):
    this_ax.set_title(f'Sample size : {this_val}')
ax[0,0].legend()
plt.tight_layout()
plt.suptitle(f'{n_components} Components, {dims} Dims, {repeats} Repeats')
plt.show()

##################################################
## POM Model Selection
##################################################
mean_array, cov_array, dat_array, labels = \
        gen_samples(n_components = 7, samples = 10000, dims = 8)

plt.scatter(*dat_array.T[:2],c=labels)
plt.show()

model_list = []
accuracy_list = []
logl_list = []
for this_repeat in tqdm(range(repeats)):
    pomgmm = GeneralMixtureModel.\
            from_samples(MultivariateGaussianDistribution, 
                    X = dat_array,
                    n_components = n_components,
                    stop_threshold = 1e-6,
                    n_init = 10,
                    batch_size = None,
                    batches_per_epoch = 10)

    pom_pred_labels = pomgmm.predict(dat_array)

    ## Sort predicted labels by matching to original data
    pom_means = np.array([x.parameters[0] for x in pomgmm.distributions])
    pom_mean_dists = distmat(pom_means , mean_array)

    pom_sorted_order = get_greedy_label(pom_mean_dists) 

    pom_sorted_labels = np.zeros(pom_pred_labels.shape)
    for num,val in enumerate(pom_sorted_order):
        pom_sorted_labels[np.where(pom_pred_labels == num)[0]] = val

    pom_accuracy = np.mean(pom_sorted_labels==labels)
    logl = np.sum(pomgmm.log_probability(dat_array))

    accuracy_list.append(pom_accuracy)
    logl_list.append(logl)


# Plot relationship between accuracy and sum log_likelihood
plt.hist2d(accuracy_list, logl_list)
plt.colorbar()
plt.show()

########################################
# ____            _   ____        _        
#|  _ \ ___  __ _| | |  _ \  __ _| |_ __ _ 
#| |_) / _ \/ _` | | | | | |/ _` | __/ _` |
#|  _ <  __/ (_| | | | |_| | (_| | || (_| |
#|_| \_\___|\__,_|_| |____/ \__,_|\__\__,_|
#                                          
########################################
dir_name = '/media/fastdata/BS72_Test1_210210_105712/'
electrode_num = 0
os.chdir(dir_name)

file_list = os.listdir('./')
hdf5_name = ''
params_file = ''
for files in file_list:
	if files[-2:] == 'h5':
		hdf5_name = files
	if files[-6:] == 'params':
		params_file = files

# Read the .params file
f = open(params_file, 'r')
params = []
for line in f.readlines():
	params.append(line)
f.close()

# Assign the parameters to variables
max_clusters = int(params[0])
num_iter = int(params[1])
thresh = float(params[2])
num_restarts = int(params[3])
voltage_cutoff = float(params[4])
max_breach_rate = float(params[5])
max_secs_above_cutoff = int(params[6])
max_mean_breach_rate_persec = float(params[7])
wf_amplitude_sd_cutoff = int(params[8])
bandpass_lower_cutoff = float(params[9])
bandpass_upper_cutoff = float(params[10])
spike_snapshot_before = float(params[11])
spike_snapshot_after = float(params[12])
sampling_rate = float(params[13])

# Open up hdf5 file, and load this electrode number
hf5 = tables.open_file(hdf5_name, 'r')
exec("raw_el = hf5.root.raw.electrode"+str(electrode_num)+"[:]")
hf5.close()

filt_el = get_filtered_electrode(raw_el, freq = [bandpass_lower_cutoff, bandpass_upper_cutoff], sampling_rate = sampling_rate)

# Delete raw electrode recording from memory
del raw_el

# Calculate the 3 voltage parameters
breach_rate = float(len(np.where(filt_el>voltage_cutoff)[0])*int(sampling_rate))/len(filt_el)
test_el = np.reshape(filt_el[:int(sampling_rate)*int(len(filt_el)/sampling_rate)], (-1, int(sampling_rate)))
breaches_per_sec = [len(np.where(test_el[i] > voltage_cutoff)[0]) for i in range(len(test_el))]
breaches_per_sec = np.array(breaches_per_sec)
secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
if secs_above_cutoff == 0:
	mean_breach_rate_persec = 0
else:
	mean_breach_rate_persec = np.mean(breaches_per_sec[np.where(breaches_per_sec > 0)[0]])

# And if they all exceed the cutoffs, assume that the headstage fell off mid-experiment
recording_cutoff = int(len(filt_el)/sampling_rate)
if breach_rate >= max_breach_rate and secs_above_cutoff >= max_secs_above_cutoff and mean_breach_rate_persec >= max_mean_breach_rate_persec:
	# Find the first 1 second epoch where the number of cutoff breaches is higher than the maximum allowed mean breach rate 
	recording_cutoff = np.where(breaches_per_sec > max_mean_breach_rate_persec)[0][0]

# Then cut the recording accordingly
filt_el = filt_el[:recording_cutoff*int(sampling_rate)]	

# Slice waveforms out of the filtered electrode recordings
slices, spike_times = extract_waveforms(filt_el, spike_snapshot = [spike_snapshot_before, spike_snapshot_after], sampling_rate = sampling_rate)

# Delete filtered electrode from memory
del filt_el, test_el

# Dejitter these spike waveforms, and get their maximum amplitudes
slices_dejittered, times_dejittered = dejitter(slices, spike_times, spike_snapshot = [spike_snapshot_before, spike_snapshot_after], sampling_rate = sampling_rate)
amplitudes = np.min(slices_dejittered, axis = 1)

# Delete the original slices and times now that dejittering is complete
del slices; del spike_times

# Scale the dejittered slices by the energy of the waveforms
scaled_slices, energy = scale_waveforms(slices_dejittered)

# Run PCA on the scaled waveforms
pca_slices, explained_variance_ratio = implement_pca(scaled_slices)

# Make an array of the data to be used for clustering, and delete pca_slices, scaled_slices, energy and amplitudes
n_pc = 3
data = np.zeros((len(pca_slices), n_pc + 2))
data[:,2:] = pca_slices[:,:n_pc]
data[:,0] = energy[:]/np.max(energy)
data[:,1] = np.abs(amplitudes)/np.max(np.abs(amplitudes))
del pca_slices; del scaled_slices; del energy

def run_fit_comparison(dat_array, n_components):

    ########################################
    ## Compare fits 
    ########################################
    tol = 1e-6
    n_iter = 1000
    n_init = 10

    start2 = time()
    pomgmm = GeneralMixtureModel.\
            from_samples(MultivariateGaussianDistribution, 
                    X = dat_array,
                    n_components = n_components,
                    stop_threshold = tol,
                    n_init = n_init,
                    batch_size = None,
                    batches_per_epoch = 20)
                    #max_iterations = n_iter,
    stop2 = time()

    start1 = time()
    skgmm = GaussianMixture(
            n_components = n_components, 
            covariance_type = 'full', 
            tol = tol, 
            max_iter = n_iter, 
            n_init = n_init)
    skgmm.fit(dat_array)
    stop1 = time()

    sk_pred_labels = skgmm.predict(dat_array)
    pom_pred_labels = pomgmm.predict(dat_array)

    sk_fit_time = stop1-start1
    pom_fit_time = stop2-start2

    return [sk_fit_time,pom_fit_time],[sk_pred_labels, pom_pred_labels]

run_times, [sk_pred_labels, pom_pred_labels] = \
        run_fit_comparison(data,7)

def square_plot(data, labels):
    fig,ax = plt.subplots(data.shape[1],data.shape[1])
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            ax[i,j].scatter(data[:,i],data[:,j],c=labels)

square_plot(data, sk_pred_labels)
square_plot(data, pom_pred_labels)
plt.show()

#fig,ax = plt.subplots(1,2)
#ax[0].scatter(*data[:,:2].T, c = sk_pred_labels)
#ax[1].scatter(*data[:,:2].T, c = pom_pred_labels)
#plt.show()

# Serialize arguments so all iterations can be run in parallel
components_list = np.arange(2,8)
args_array = np.repeat(components_list, repeats)

outs = Parallel(n_jobs = cpu_count() - 2)\
        (delayed(run_fit_comparison)(data,args) for args in tqdm(args_array))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

time_list, labels_list = [np.array(x) for x in list(zip(*outs))]
chunked_time = list(chunks(time_list,repeats))
chunked_labels = list(chunks(accuracy_list,repeats))

time_array = np.array(chunked_time)
broadcast_components = np.broadcast_to(\
        components_list[:,np.newaxis], time_array[...,0].shape)
mean_time = np.mean(time_array,axis=1)
sd_time = np.std(time_array,axis=1)

plt.scatter(broadcast_components, time_array[...,0], 
        label = 'SKLearn', c = cmap(0))
plt.scatter(broadcast_components, time_array[...,1], 
        label = 'Pomegranate', c = cmap(1))
plt.fill_between(components_list, mean_time[:,0] + sd_time[:,0], 
        mean_time[:,0] - sd_time[:,0], alpha = 0.5, color = cmap(0))
plt.plot(components_list, mean_time[:,0], c=cmap(0))
plt.fill_between(components_list, mean_time[:,1] + sd_time[:,1], 
        mean_time[:,1] - sd_time[:,1], alpha = 0.5, color = cmap(1))
plt.plot(components_list, mean_time[:,1], c=cmap(1))
plt.xlabel('Components fit')
plt.ylabel('Run time (s)')
plt.legend()
plt.title(f'Data size : {data.shape}')
plt.show()

################################################################################
################################################################################
################################################################################
tol = 1e-6
n_iter = 1000
n_init = 10

# Test with separate funcs for fitting
def run_pom_gmm(dat_array, n_components, repeats):

    tol = 1e-6
    n_iter = 1000
    n_init = 10
    
    model_list = []

    start2 = time()
    for i in range(repeats):
        pomgmm = GeneralMixtureModel.\
                from_samples(MultivariateGaussianDistribution, 
                        X = dat_array,
                        n_components = n_components,
                        stop_threshold = tol,
                        n_init = n_init,
                        batch_size = 10,
                        batches_per_epoch = None)
                        #max_iterations = n_iter,
        model_list.append(pomgmm)
    stop2 = time()

    pom_pred_labels = pomgmm.predict(dat_array)
    pom_fit_time = stop2-start2

    return pom_fit_time, model_list 

def run_sk_gmm(dat_array, n_components, repeats):

    tol = 1e-6
    n_iter = 1000
    n_init = 10

    model_list = []

    start1 = time()

    for i in range(repeats):
        skgmm = GaussianMixture(
                n_components = n_components, 
                covariance_type = 'full', 
                tol = tol, 
                max_iter = n_iter) 
        skgmm.fit(dat_array)
        model_list.append(skgmm)
    stop1 = time()

    sk_fit_time = stop1-start1
    sk_pred_labels = skgmm.predict(dat_array)

    return sk_fit_time, model_list

components_list = np.arange(2,8)
args_array = np.repeat(components_list, repeats)

pom_outs = Parallel(n_jobs = cpu_count() - 2)\
        (delayed(run_pom_gmm)(data,args,repeats) for args in tqdm(args_array))
sk_outs = Parallel(n_jobs = cpu_count() - 2)\
        (delayed(run_sk_gmm)(data,args,repeats) for args in tqdm(args_array))

pom_time, pom_model_list = list(zip(*pom_outs))
sk_time, sk_model_list = list(zip(*sk_outs))

plt.scatter(args_array,sk_time, label = "SK")
plt.scatter(args_array,pom_time, label = "POM")
plt.legend()
plt.show()

## Test with clustering funcs
def this_pom_gmm(data, clusters):
        start = time()
        out =  pom_clusterGMM(
                    data, 
                    n_clusters = clusters, 
                    n_iter = n_iter, 
                    restarts = repeats, 
                    threshold = tol)
        end = time()
        return out, end-start

def this_sk_gmm(data, clusters):
        start = time()
        out =  clusterGMM(
                    data, 
                    n_clusters = clusters, 
                    n_iter = n_iter, 
                    restarts = repeats, 
                    threshold = tol)
        end = time()
        return out, end-start

pom_outs = Parallel(n_jobs = cpu_count() - 2)\
        (delayed(this_pom_gmm)(data,args) for args in tqdm(args_array))
sk_outs = Parallel(n_jobs = cpu_count() - 2)\
        (delayed(this_sk_gmm)(data,args) for args in tqdm(args_array))

pom_other_outs, pom_times = list(zip(*pom_outs))
sk_other_outs, sk_times  = list(zip(*sk_outs))

plt.figure()
plt.scatter(args_array,sk_times, label = "SK")
plt.scatter(args_array,pom_times, label = "POM")
plt.legend()
plt.show()
