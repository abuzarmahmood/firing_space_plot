## Import required modules
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import tables
import easygui
import scipy
import numpy as np
from tqdm import tqdm, trange
from sklearn.utils import resample
from itertools import product
import scipy.signal as signal
import itertools
from scipy.special import gamma
from joblib import Parallel,delayed
import multiprocessing as mp
from sklearn.decomposition import PCA as pca
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from scipy.stats import ttest_ind
from statsmodels.distributions.empirical_distribution import ECDF


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

def firing_overview(data, t_vec = None, y_values_vec = None,
                    interpolation = 'nearest',
                    cmap = 'jet',
                    min_val = None, max_val=None, 
                    subplot_labels = None):
    """
    Takes 3D numpy array as input and rolls over first dimension
    to generate images over last 2 dimensions
    E.g. (neuron x trial x time) will generate heatmaps of firing
        for every neuron
    """

    if min_val is None:
        min_val = np.min(data,axis=None)
    if max_val is None:
        max_val = np.max(data,axis=None)
    if t_vec is None:
        t_vec = np.arange(data.shape[-1])
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])

    if data.shape[-1] != len(t_vec):
        raise Exception('Time dimension in data needs to be'\
            'equal to length of time_vec')

    num_nrns = data.shape[0]
    # Plot firing rates
    square_len = np.int(np.ceil(np.sqrt(num_nrns)))
    fig, ax = plt.subplots(square_len,square_len, sharex='all',sharey='all')
    
    nd_idx_objs = []
    for dim in range(ax.ndim):
        this_shape = np.ones(len(ax.shape))
        this_shape[dim] = ax.shape[dim]
        nd_idx_objs.append(
                np.broadcast_to( 
                    np.reshape(
                        np.arange(ax.shape[dim]),
                        this_shape.astype('int')), ax.shape).flatten())
    
    if subplot_labels is None:
        subplot_labels = np.zeros(num_nrns)
    if y_values_vec is None:
        y_values_vec = np.arange(data.shape[1])
    for nrn in range(num_nrns):
        plt.sca(ax[nd_idx_objs[0][nrn],nd_idx_objs[1][nrn]])
        plt.gca().set_title('{}:{}'.format(int(subplot_labels[nrn]),nrn))
        plt.gca().pcolormesh(t_vec, y_values_vec,
                data[nrn,:,:],cmap=cmap)#,
                #vmin = min_val, vmax = max_val)
    return ax

os.chdir('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data

# Extract data
dat = \
    ephys_data('/media/bigdata/Abuzar_Data/AM12/AM12_extracted/AM12_4Tastes_191106_085215')
    #ephys_data('/media/bigdata/Abuzar_Data/AM17/AM17_extracted/AM17_4Tastes_191126_084934')

dat.firing_rate_params = dict(zip(('step_size','window_size','dt'),
                                    (25,250,1)))

dat.extract_and_process()
dat.firing_overview(dat.all_normalized_firing);plt.show()

# Generate firing rate distribution for each taste

# Calculate firing rates using BAKS
all_spikes = np.array(dat.spikes)
all_spikes_long = np.reshape(all_spikes,(-1,*all_spikes.shape[2:])).swapaxes(0,1)
# Any trial without a spike gets a random spike to avoid zero-division 
for inds in zip(*np.where(np.sum(all_spikes_long,axis=-1)==0)):
    all_spikes_long[(*inds,np.random.choice(np.arange(all_spikes_long.shape[-1]),1))] = 1

all_firing_long = np.array(\
        [[BAKS(np.where(trial)[0]/1000,np.arange(len(trial),step=25)/1000) \
        for trial in nrn]\
        for nrn in all_spikes_long])
dat.firing_overview(all_firing_long);plt.show()

# Find PC's for all neurons 
really_long_firing = np.reshape(all_firing_long,(all_spikes_long.shape[0],-1))
really_long_zscore_firing = zscore(really_long_firing,axis=-1)
pca_object = pca(n_components = 10).fit(really_long_firing.T)
explained_variance_threshold = 0.8
needed_components = np.sum(
                        np.cumsum(
                            pca_object.explained_variance_ratio_) < explained_variance_threshold)+1
pca_object = pca(n_components = needed_components).fit(really_long_firing.T)
pca_firing = np.array([pca_object.transform(trial.T).T for trial in all_firing_long.swapaxes(0,1)]) 

dat.firing_overview(pca_firing.swapaxes(0,1));plt.show()

# Pairwise logistic regression on PCA firing

# List all possible combinations of pairwise taste comparisons
#taste_comparisons = list(itertools.combinations(range(this_nrn.shape[0]),2))
#taste_labels = np.sort([0,1,2,3]*30)
#
#taste_list = [pca_firing[taste_labels == taste] for taste in np.sort(np.unique(taste_labels))]
#taste_label_list = [taste_labels[taste_labels == taste] \
#        for taste in np.sort(np.unique(taste_labels))]
#
#shuffle_repeats = 100
#pca_firing_shuffle = np.array([np.random.permutation(pca_firing) for x in range(shuffle_repeats)])
#taste_list_shuffle = [pca_firing_shuffle[:,taste_labels == taste] \
#        for taste in np.sort(np.unique(taste_labels))]
#
#taste_comparison_scores = np.zeros((len(taste_comparisons),pca_firing.shape[-1]))
#for comparison_num, this_comparison in enumerate(taste_comparisons):
#    this_comparison_tastes = \
#            np.concatenate([taste_list[x] for x in this_comparison])
#    this_comparison_labels = np.concatenate([taste_label_list[x] \
#            for x in this_comparison])
#    X_train, X_test, y_train, y_test = train_test_split(
#                    this_comparison_tastes, this_comparison_labels, test_size=0.33, random_state=42)
#    clf_list = [LogisticRegression(random_state=0).fit(
#        X_train[...,time_bin], y_train) \
#                for time_bin in range(X_train.shape[-1])]
#    taste_comparison_scores[comparison_num] = np.array(
#            [clf_list[time_bin].score(X_test[...,time_bin], y_test) \
#                    for time_bin in range(X_test.shape[-1])])

# Define function to calculate classification accuracy
def calc_classification_accuracy(taste_array, taste_labels, test_ratio = 0.33):
    """
    taste_array = (trials, features , time)
    taste_labels = Vector containing trial labels
    """
    taste_list = [taste_array[taste_labels == taste] for taste in np.sort(np.unique(taste_labels))]
    taste_label_list = [taste_labels[taste_labels == taste] \
            for taste in np.sort(np.unique(taste_labels))]
    taste_comparisons = list(itertools.combinations(np.sort(np.unique(taste_labels)),2))
    taste_comparison_scores = np.zeros((len(taste_comparisons),taste_array.shape[-1]))
    for comparison_num, this_comparison in enumerate(taste_comparisons):
        this_comparison_tastes = \
                np.concatenate([taste_list[x] for x in this_comparison])
        this_comparison_labels = np.concatenate([taste_label_list[x] \
                for x in this_comparison])
        X_train, X_test, y_train, y_test = train_test_split(
                this_comparison_tastes, this_comparison_labels, test_size=test_ratio, 
                random_state=np.random.randint(1000))
        clf_list = [LogisticRegression(random_state=0).fit(
            X_train[...,time_bin], y_train) \
                    for time_bin in range(X_train.shape[-1])]
        taste_comparison_scores[comparison_num] = np.array(
                [clf_list[time_bin].score(X_test[...,time_bin], y_test) \
                        for time_bin in range(X_test.shape[-1])])
    return taste_comparison_scores

taste_comparison_scores = np.array(
            Parallel(n_jobs = -1)(delayed(calc_classification_accuracy)\
            (pca_firing,taste_labels,0.5) for repeat in tqdm(range(shuffle_repeats))))
taste_comparison_shuffle_scores = np.array(
        Parallel(n_jobs = -1)(delayed(calc_classification_accuracy)\
        (repeat,taste_labels,0.5) for repeat in tqdm(pca_firing_shuffle)))
comparison_mean = np.mean(taste_comparison_scores,axis=0)
comparison_std = np.std(taste_comparison_scores,axis=0)
shuffle_mean = np.mean(taste_comparison_shuffle_scores,axis=0)
shuffle_std = np.std(taste_comparison_shuffle_scores,axis=0)

#fig,ax = plt.subplots(1,1)
#im = ax.imshow(taste_comparison_scores,interpolation='nearest',aspect='auto',origin='lower')
#plt.colorbar(im)
#ax.set_yticklabels(taste_comparisons)
#plt.show()

# Plot all comparisons vs their respective shuffles
fig, ax = plt.subplots(taste_comparison_scores.shape[1],sharex=True,sharey=True)
for ax_num, this_ax in enumerate(ax):
    this_ax.fill_between( x = range(len(comparison_mean[ax_num])), 
            y1 = comparison_mean[ax_num] - 2*comparison_std[ax_num],
            y2 = comparison_mean[ax_num] + 2*comparison_std[ax_num],
            color = 'orange',alpha = 0.5)
    this_ax.plot(comparison_mean[ax_num], color = 'orange')
    this_ax.fill_between( x = range(len(shuffle_mean[ax_num])), 
            y1 = shuffle_mean[ax_num] - 2*shuffle_std[ax_num],
            y2 = shuffle_mean[ax_num] + 2*shuffle_std[ax_num],
            color = 'blue',alpha = 0.5)
    this_ax.plot(shuffle_mean[ax_num], color = 'blue')
    this_ax.set_title(taste_comparisons[ax_num])
firing_overview(pca_firing.swapaxes(0,1))
plt.show()

# _   _                     _____         _   
#| | | |_   _ _ __   ___   |_   _|__  ___| |_ 
#| |_| | | | | '_ \ / _ \    | |/ _ \/ __| __|
#|  _  | |_| | |_) | (_) |   | |  __/\__ \ |_ 
#|_| |_|\__, | .__/ \___/    |_|\___||___/\__|
#       |___/|_|                              

# Using T-Test
p_vals = np.zeros(taste_comparison_scores.shape[1:])
for this_iter in np.ndindex(taste_comparison_scores.shape[1:]):
    p_vals[this_iter] = ttest_ind(
            taste_comparison_scores[...,this_iter[0],this_iter[1]],
            taste_comparison_shuffle_scores[...,this_iter[0],this_iter[1]])[-1]

# Plot all comparisons vs their respective shuffles
fig, ax = plt.subplots(taste_comparison_scores.shape[1],sharex=True,sharey=True)
for ax_num, this_ax in enumerate(ax):
    this_ax.fill_between( x = range(len(comparison_mean[ax_num])), 
            y1 = comparison_mean[ax_num] - 2*comparison_std[ax_num],
            y2 = comparison_mean[ax_num] + 2*comparison_std[ax_num],
            color = 'orange',alpha = 0.5)
    # Mark significant values with Bonferroni correction
    sig_inds = p_vals[ax_num] < 0.05/p_vals.shape[-1]
    this_ax.plot(comparison_mean[ax_num], color = 'orange')
    this_ax.plot(np.arange(len(comparison_mean[ax_num]))[sig_inds],
            comparison_mean[ax_num][sig_inds], 'x',color = 'red')
    this_ax.fill_between( x = range(len(shuffle_mean[ax_num])), 
            y1 = shuffle_mean[ax_num] - 2*shuffle_std[ax_num],
            y2 = shuffle_mean[ax_num] + 2*shuffle_std[ax_num],
            color = 'blue',alpha = 0.5)
    this_ax.plot(shuffle_mean[ax_num], color = 'blue')
    this_ax.set_title(taste_comparisons[ax_num])
plt.show()



# Significance calculation using bootstrap
mean_difference = comparison_mean - shuffle_mean 

# Bootstrap correction for multiple comparisons
merged_data = np.concatenate((taste_comparison_scores,taste_comparison_shuffle_scores),axis=0)

bootstrap_repeats = 100
alpha = 0.05
#random_p_vals = np.zeros((*merged_data.shape[1:],bootstrap_repeats))
random_mean_difference = np.zeros((*merged_data.shape[1:],bootstrap_repeats))

for this_bootstrap_repeat in tqdm(range(bootstrap_repeats)):
    random_split = np.array_split(np.random.permutation(merged_data),2)
    random_split = [x.T for x in random_split]
    random_mean_difference[...,this_bootstrap_repeat] = \
            (np.mean(random_split[0],axis=-1) - np.mean(random_split[1],axis=-1)).T
    # Compare both splits for every comparison, for every timepoint
    #null_comparison_array = np.zeros(merged_data.T.shape[:2]) 
    #for this_iter in np.ndindex(random_split[0].shape[:2]):
    #    null_comparison_array[this_iter] = \
    #            ttest_ind(random_split[0][this_iter], random_split[1][this_iter])[1]
    #random_p_vals[...,this_bootstrap_repeat] = null_comparison_array.T

random_mean_difference_mean = np.mean(random_mean_difference,axis=-1)
random_mean_difference_std = np.std(random_mean_difference,axis=-1)

# Plot all comparisons vs their respective shuffles
fig, ax = plt.subplots(mean_difference.shape[0],sharex=True,sharey=True)
for ax_num, this_ax in enumerate(ax):
    this_ax.plot(mean_difference[ax_num],
            color = 'orange',alpha = 0.5)
    this_ax.fill_between( x = range(len(random_mean_difference_mean[ax_num])), 
            y1 = random_mean_difference_mean[ax_num] - 2*random_mean_difference_std[ax_num],
            y2 = random_mean_difference_mean[ax_num] + 2*random_mean_difference_std[ax_num],
            color = 'blue',alpha = 0.5)
    this_ax.plot(random_mean_difference_mean[ax_num], color = 'blue')
    this_ax.set_title(taste_comparisons[ax_num])
plt.show()


## Generate ECDF using NULL difference distribution and calculate distance of true mean
difference_ecdf = np.zeros( random_mean_difference.shape[:-1])
for this_iter in np.ndindex(random_mean_difference.shape[:-1]):
    difference_ecdf[this_iter] = ECDF(random_mean_difference[this_iter])\
            (mean_difference[this_iter])

# Transform ECDF to p-values
difference_p_val = 0.5 - np.abs(difference_ecdf - 0.5)

# Plot all comparisons vs their respective shuffles
fig, ax = plt.subplots(taste_comparison_scores.shape[1],sharex=True,sharey=True)
for ax_num, this_ax in enumerate(ax):
    this_ax.fill_between( x = range(len(comparison_mean[ax_num])), 
            y1 = comparison_mean[ax_num] - 2*comparison_std[ax_num],
            y2 = comparison_mean[ax_num] + 2*comparison_std[ax_num],
            color = 'orange',alpha = 0.5)
    # Mark points where the comparison is significant
    sig_inds = difference_p_val[ax_num] < 0.05
    this_ax.plot(len(np.arange(comparison_mean[ax_num])),
            comparison_mean[ax_num], color = 'orange')
    this_ax.fill_between( x = range(len(shuffle_mean[ax_num])), 
            y1 = shuffle_mean[ax_num] - 2*shuffle_std[ax_num],
            y2 = shuffle_mean[ax_num] + 2*shuffle_std[ax_num],
            color = 'blue',alpha = 0.5)
    this_ax.plot(shuffle_mean[ax_num], color = 'blue')
    this_ax.set_title(taste_comparisons[ax_num])
firing_overview(pca_firing.swapaxes(0,1))
plt.show()

random_p_vals_long = np.reshape(random_p_vals, (random_p_vals.shape[0],-1)) 
# Find what p-value gives significant results at true alpha
corrected_alpha = [np.percentile(x,alpha*100) for x in random_p_vals_long]


#random_means = [np.mean(x,axis=0) for x in random_split]
#random_std = [np.std(x,axis=0) for x in random_split]
#
#fig, ax = plt.subplots(random_split[0].shape[1],sharex=True,sharey=True)
#for ax_num, this_ax in enumerate(ax):
#    this_ax.fill_between( x = range(len(random_means[0][ax_num])), 
#            y1 = random_means[0][ax_num] - 2*random_std[0][ax_num],
#            y2 = random_means[0][ax_num] + 2*random_std[0][ax_num],
#            color = 'orange',alpha = 0.5)
#    this_ax.plot(random_means[0][ax_num], color = 'orange')
#    this_ax.fill_between( x = range(len(random_means[1][ax_num])), 
#            y1 = random_means[1][ax_num] - 2*random_std[1][ax_num],
#            y2 = random_means[1][ax_num] + 2*random_std[1][ax_num],
#            color = 'blue',alpha = 0.5)
#    this_ax.plot(random_means[1][ax_num], color = 'blue')
#    this_ax.set_title(taste_comparisons[ax_num])
#firing_overview(pca_firing.swapaxes(0,1))
#plt.show()
