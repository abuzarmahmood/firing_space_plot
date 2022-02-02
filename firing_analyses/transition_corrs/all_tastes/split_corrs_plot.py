"""
Generate plots for split region correlations
1) Raster plots for each split for each trial with changepoints overlayed
2) Scatterplots of the changepoint positions

=== 3 Way Models ===
3) Intra-session plots across splits comparing percentile to 
    median neuron firing
4) Inter-session plots comparing:
        a) Median percentile (across splits) to median neuron firing
        b) Median percentile (across splits) to neuron count
        c) Median percentile (across splits) to median of variance of tau distribution
"""

##################################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
##################################################

import numpy as np
import json
from glob import glob
import os
import pandas as pd
import pickle 
import sys
from scipy import stats
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm,trange 
import tables
import re
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler as ss
from sklearn.mixture import GaussianMixture as gaussmix
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs')
from check_data import check_data 

def load_model_data(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as buff:
            data = pickle.load(buff)
        print('Trace loaded from cache')
        tau_samples = data['tau']
        int_tau = np.vectorize(int)(tau_samples)
        # Convert to int first, then take mode
        int_mode_tau = stats.mode(int_tau,axis=0)[0][0]
        spike_array = data['fulldata']
        # Remove pickled data to conserve memory
        del data
    return int_mode_tau, spike_array, tau_samples

def parallelize_shuffles(func, args, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(*args) for this_iter in tqdm(iterator))

def calc_mse(x,y):
    return np.mean(np.abs(x-y))

def gen_shuffle(func, x, y):
    return func(np.random.permutation(x),y)

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

class params_from_path:
    def __init__(self, path):
        # Extract model params from basename
        self.path = path
        self.model_name = os.path.basename(self.path).split('.')[0]
        self.states = int(re.findall("\d+states",self.model_name)[0][:-6])
        self.time_lims = [int(x) for x in \
                re.findall("\d+_\d+time",self.model_name)[0][:-4].split('_')]
        self.bin_width = int(re.findall("\d+bin",self.model_name)[0][:-3])
        #self.fit_type = re.findall("type_.+",self.model_name)[0].split('_')[1]
        # Exctract data_dir from model_path
        self.data_dir = "/".join(self.path.split('/')[:-3])
        self.session_name = self.data_dir.split('/')[-1]
        self.animal_name = self.session_name.split('_')[0]

vline_kwargs = {'color': 'red', 'linewidth' :3, 'alpha' : 0.7}

def create_split_changepoint_plots(array1, array2, tau1, tau2, 
                                trial_inds_list, params, split_ind,
                                spike_lims):

    """
    array1, array2 : (trials x neuron x time) spike arrays
    tau1, tau2 : (trials x changepoints)
    trial_inds : list of lists/arrays for which trials to plot per figure
    """

    fin_spike_array = np.concatenate([array1,array2],axis=1)
    #fin_spike_array = fin_spike_array[...,spike_lims[0]:spike_lims[1]]
    div_line = array1.shape[1] - 0.5
    max_line = fin_spike_array.shape[1]-0.5
    cmap = plt.get_cmap("Set1")
    tau1,tau2 = tau1.T, tau2.T

    state_inds1 = np.concatenate([np.zeros((tau1.shape[0],1)),
                    tau1, 
                    np.ones((tau1.shape[0],1))*array1.shape[-1]],
                    axis=-1)
    state_inds2 = np.concatenate([np.zeros((tau1.shape[0],1)),
                    tau2, 
                    np.ones((tau1.shape[0],1))*array1.shape[-1]],
                    axis=-1)

    for fig_num in tqdm(np.arange(len(trial_inds_list))):
        trial_inds = trial_inds_list[fig_num]
        trial_count = len(trial_inds)
        
        fig, ax = plt.subplots(trial_count,1, 
                sharex = True, figsize = (5,trial_count))
        for num,trial in enumerate(trial_inds):
            ax[num].scatter(
                    *np.where(fin_spike_array[trial])[::-1], 
                    marker = "|", color = 'k')
            ax[num].set_ylabel(trial)
            ax[num].vlines(tau1[trial],-0.5,div_line, **vline_kwargs)
            ax[num].vlines(tau2[trial],div_line, max_line, **vline_kwargs)
            ax[num].set_xlim(spike_lims)
            ax[num].fill_betweenx(div_line, tau1[trial])
            for state in range(tau1.shape[1]+1):
                ax[num].axvspan(
                    state_inds1[trial,state], state_inds1[trial,state+1],
                    ymax = div_line/max_line, 
                    alpha = 0.2, color = cmap(state))
                ax[num].axvspan(
                    state_inds2[trial,state], state_inds2[trial,state+1],
                    ymin = div_line/max_line, 
                    alpha = 0.2, color = cmap(state))
        plt.suptitle(f'{params.session_name} {split_ind} : Set{fig_num}') 
        #fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(fin_plot_dir,
                f'{params.session_name}_split{split_ind}_set{fig_num}'))
        plt.close(fig)
        #plt.show()

save_names = ['tau_corrs','tau_mse','tau_list',
        'rho_shuffles','mse_shuffles',
        'rho_percentiles','mse_percentiles']

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/split_region'

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/plots/single_region_split'

# Load pkl detailing which recordings have split changepoints
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/single_region_split_frame.pkl'

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################
                                               

split_frame = pd.read_pickle(data_dir_pkl)

all_counts = []
all_firing = []
all_percentiles = []
all_tau_vars = []

for data_dir in tqdm(split_frame.path.iloc):
    #data_dir = split_frame.path.iloc[0]

    dat = ephys_data(data_dir)
    with tables.open_file(dat.hdf5_path,'r+') as hf5:
        if save_path not in hf5:
            hf5.create_group(os.path.dirname(save_path),
                    os.path.basename(save_path),
                    createparents = True)

    ## Find and load split pickle files
    this_info = check_data(data_dir)
    this_info.run_all()
    split_paths = [path for  num,path in enumerate(this_info.pkl_file_paths) \
                    if num in this_info.split_inds]
    state4_models = [path for path in split_paths if '4state' in path]
    split_basenames = [os.path.basename(x) for x in state4_models]

    # Match splits
    # Extract split nums
    split_basenames_split = [x.split('_') for x in split_basenames]
    split_nums = np.vectorize(int)\
            (np.array([[x[1],x[2]] for x in split_basenames_split]))
    split_inds = np.array([np.where(split_nums[:,1] == num)[0] \
            for num in range(split_nums[:,1].max()+1)])

    session_counts = []
    session_firing = []
    session_percentiles = []
    session_tau_vars = []
    ##############################
    # ____  _       _       
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    ##############################
    #for split_ind in range(len(split_inds)):
    split_ind = 0
    this_split = split_inds[split_ind]
    split_str = "_".join([str(x) for x in this_split])

    #with tables.open_file(dat.hdf5_path,'r+') as hf5:
    #    # Will only remove if array already there
    #    remove_node(fin_save_path, hf5, recursive=True)
    #    hf5.create_group(save_path, terminal_dir, createparents = True)

    model_paths = [split_paths[i] for i in this_split]

    #terminal_dir = f'split_{"_".join([str(x) for x in this_split])}'
    params = params_from_path(model_paths[0])
    fin_save_path = os.path.join(save_path,f'split{split_ind}')
    fin_plot_dir = os.path.join(
            plot_dir,params.session_name, f'split{split_ind}') 
    if not os.path.exists(fin_plot_dir):
        os.makedirs(fin_plot_dir)

    data_list = [load_model_data(this_path) for this_path in model_paths]
    tau_list, spike_list, full_tau_list = list(zip(*data_list)) 
    tau_vars = np.array([np.var(x,axis=0) for x in full_tau_list])
    spike_long_list = [np.reshape(x,(-1,*x.shape[2:])) for x in spike_list]
    trial_inds_list = np.array_split(np.arange(spike_long_list[0].shape[0]),8)

    h5_path = glob(os.path.join(params.data_dir,"*.h5"))[0]
    with tables.open_file(h5_path, 'r') as h5:
        if fin_save_path in h5:
            for this_name in save_names:
                globals()[this_name] = \
                        h5.get_node(os.path.join(fin_save_path, this_name))[:]

    ## Info for 3 way Models
    split_firing_rates = \
        [np.mean(x[...,params.time_lims[0]:params.time_lims[1]],axis=(0,2)) 
                for x in spike_long_list]
    session_firing.append(split_firing_rates)
    session_percentiles.append(rho_percentiles)
    session_counts.append(spike_long_list[0].shape[1])
    session_tau_vars.append(tau_vars)
     
    ## =====================

    stim_t = 2000
    tau_list = np.array(tau_list) 
    rho_shuffles = np.array(rho_shuffles)
    # Scale tau values
    tau_list = (tau_list * params.bin_width) + params.time_lims[0]
    tau_list -= stim_t
    tau_corr_dat = [stats.spearmanr(x,y) for x,y in zip(*tau_list)] 
    tau_corrs, tau_ps = list(zip(*tau_corr_dat))

    shuffle_count = 1000
    tau_shuffle = np.array([[np.array(this_tau)[np.random.choice(
                            np.arange(len(this_tau)), shuffle_count)] \
                    for this_tau in this_split] for this_split in tau_list])

    x = np.linspace(*params.time_lims) - stim_t

    # Plot scatter plots for each tau and tau shuffled
    fig, ax = plt.subplots(len(tau_list[0]),2, 
            sharex = True, sharey=True, figsize = (7,10))
    for change_num in range(len(tau_list[0])):
        rho_str = f'Rho : {np.round(tau_corrs[change_num],3)}'
        percentile_str = f'Perc : {rho_percentiles[change_num]}'
        p_val_str = f'p_val : {np.round(tau_ps[change_num],3)}'

        shuffle_data = tau_shuffle[:,change_num]
        actual_data = tau_list[:,change_num]

        ax[change_num, 0].scatter(*actual_data, alpha = 0.5,
                label = rho_str )#+ "\n" + percentile_str + "\n" + p_val_str)
        ax[change_num, 0].legend(loc = 'upper left')
        ax[change_num, 0].set_ylabel(f'Change {change_num}')
        rho_str_sh = f'Rho : {np.round(np.mean(rho_shuffles[change_num]),3)}'# +\
                #f'+/- {np.round(np.std(rho_shuffles[change_num]),3)}'
        ax[change_num, 1].scatter(*shuffle_data, alpha = 0.2,label = rho_str_sh)
        ax[change_num, 1].legend(loc = 'upper left')
        ax[change_num, 0].set(adjustable='box', aspect='equal')
        ax[change_num, 1].set(adjustable='box', aspect='equal')
        ax[change_num, 0].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        ax[change_num, 1].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        #ax[change_num, 0].text(0,0,f'Rho : {np.round(tau_corrs[change_num],3)}')
    plt.suptitle(params.session_name + ": Tau Corrs" + "\n" +\
            f'Split {split_ind} : {str(this_split)}')
    ax[0,0].set_title('Actual')
    ax[0,1].set_title('Shuffle')
    fig.savefig(os.path.join(
        fin_plot_dir,f'{params.session_name}_split{split_ind}' +\
                            "_tau_scatter"))
    plt.close(fig)
    #plt.show()

    ## Scatter plots with contours
    fig, ax = plt.subplots(len(tau_list[0]),2, 
            sharex = True, sharey=True, figsize = (7,10))
    for change_num in range(len(tau_list[0])):
        rho_str = f'Rho : {np.round(tau_corrs[change_num],3)}'
        percentile_str = f'Perc : {rho_percentiles[change_num]}'
        p_val_str = f'p_val : {np.round(tau_ps[change_num],3)}'

        shuffle_data = tau_shuffle[:,change_num]
        actual_data = tau_list[:,change_num]

        X,Y = np.meshgrid(x,x)
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([*shuffle_data])
        clf = gaussmix(n_components=1, covariance_type='full')
        clf.fit(values.T)
        Z = -clf.score_samples(positions.T)
        Z_shuff = Z.reshape(X.shape)
        #kernel = stats.gaussian_kde(values)
        #Z_shuff = np.reshape(kernel(positions).T, X.shape)

        values = np.vstack([*actual_data])
        clf = gaussmix(n_components=1, covariance_type='full')
        clf.fit(values.T)
        Z = -clf.score_samples(positions.T)
        Z_actual = Z.reshape(X.shape)
        #kernel = stats.gaussian_kde(values)
        #Z_actual = np.reshape(kernel(positions).T, X.shape)

        ax[change_num, 0].scatter(*actual_data, alpha = 0.5,
                label = rho_str )#+ "\n" + percentile_str + "\n" + p_val_str)
        ax[change_num, 0].legend(loc = 'upper left')
        ax[change_num, 0].set_ylabel(f'Change {change_num}')
        rho_str_sh = f'Rho : {np.round(np.mean(rho_shuffles[change_num]),3)}'# +\
                #f'+/- {np.round(np.std(rho_shuffles[change_num]),3)}'
        ax[change_num, 1].scatter(*shuffle_data, alpha = 0.2,label = rho_str_sh)
        ax[change_num, 1].contour(X, Y, Z_shuff, alpha = 0.7)
        ax[change_num, 0].contour(X, Y, Z_actual, alpha = 0.7)
        ax[change_num, 1].legend(loc = 'upper left')
        ax[change_num, 0].set(adjustable='box', aspect='equal')
        ax[change_num, 1].set(adjustable='box', aspect='equal')
        ax[change_num, 0].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        ax[change_num, 1].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        #ax[change_num, 0].text(0,0,f'Rho : {np.round(tau_corrs[change_num],3)}')
    plt.suptitle(params.session_name + ": Tau Corrs" + "\n" +\
            f'Split {split_ind} : {str(this_split)}')
    ax[0,0].set_title('Actual')
    ax[0,1].set_title('Shuffle')
    fig.savefig(os.path.join(
        fin_plot_dir,f'{params.session_name}_split{split_ind}' +\
                            "_tau_scatter_scatter"))
    plt.close(fig)
    #plt.show()

    spike_lims = [1500,4000]
    #create_split_changepoint_plots(
    #                    *spike_long_list, *tau_list,
    #                    trial_inds_list, params, split_ind, spike_lims)

    ####################################### 
    all_counts.append(session_counts)
    all_firing.append(session_firing)
    all_percentiles.append(session_percentiles)
    all_tau_vars.append(session_tau_vars)

        #tau_list = [x.T for x in tau_list]
        ## Calculate spearman rho
        #tau_corrs = [stats.spearmanr(x,y)[0] for x,y in zip(*tau_list)] 
        ## Calculate MSE
        #tau_mse = [np.mean(np.abs(x-y)) for x,y in zip(*tau_list)]

all_firing_mean = [[[np.mean(x),np.median(y)] for x,y in this_split] \
        for this_split in all_firing]

all_counts = np.array(all_counts)
all_firing_mean = np.array(all_firing_mean)
all_percentiles = np.array(all_percentiles)
all_tau_vars = np.array(all_tau_vars)

## Intra-session comparison of firing rates to percentiles

all_firing_fit_frames = []

for session_ind in range(len(all_counts)):
    this_firing = all_firing_mean[session_ind]
    this_percentiles = all_percentiles[session_ind]

    for change_ind in range(this_percentiles.shape[1]):
    #change_ind = 0
        this_dat = np.concatenate(
                (this_firing, this_percentiles[:,change_ind][:,np.newaxis]),
                axis = -1)

        dat_scaled = ss().fit_transform(this_dat)

        dat_frame = pd.DataFrame(dict(zip(['f1','f2','perc'],dat_scaled.T)))
        mod = smf.ols(formula='perc ~ f1 * f2', data=dat_frame)
        res = mod.fit()
        #print(res.summary())

        data_dir_splits = split_frame.path.iloc[session_ind].split('/')
        fit_details = pd.DataFrame({'coeffs' : res.params})
        fit_details['pvals'] = res.pvalues
        fit_details['change_ind'] = change_ind
        fit_details['session'] = data_dir_splits[-1] 
        fit_details['animal_name'] = data_dir_splits[-2] 
        all_firing_fit_frames.append(fit_details)

        fin_plot_dir = os.path.join(plot_dir, data_dir_splits[-1])

        this_str = fit_details[['coeffs','pvals','change_ind']].to_string()
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(121, projection = '3d')
        ax.scatter(*dat_scaled.T, s = 80, color = 'k')
        ax.set_xlabel('Firing1')
        ax.set_ylabel('Firing2')
        ax.set_zlabel('Percentile')
        ax2 = fig.add_subplot(122)
        ax2.text(0.1,0.5,this_str)
        plt.suptitle(data_dir_splits[-1] + ": Firing percentile")
        fig.savefig(os.path.join(fin_plot_dir, 
            f'firing_percentile_plot_change{change_ind}'))
        plt.close(fig)
        #plt.show()

fin_firing_fit_frame = pd.concat(all_firing_fit_frames)
fin_firing_fit_frame[fin_firing_fit_frame.pvals < 0.1]

## Inter-session comparison of neuron count to percentiles

animal_names = [x.split('/')[-2] for x in split_frame.path]
animal_encoding, unique_names = pd.factorize(animal_names)
animal_encoding = animal_encoding / np.max(animal_encoding)
fin_animal_encoding = np.repeat(animal_encoding, all_percentiles.shape[1], axis=0)

cmap = plt.get_cmap('tab20')
colors = cmap(animal_encoding)
fin_colors = np.repeat(colors, all_percentiles.shape[1], axis=0)

## Intra-session comparison of counts firing rates to percentiles
fig,ax = plt.subplots(1,all_percentiles.shape[-1], 
        sharex= True, sharey=True, figsize = (15,5))
for change_ind in range(all_percentiles.shape[-1]):
    #change_ind = 0
    this_percentiles = all_percentiles[:,:,change_ind]
    zscore_perc = stats.zscore(this_percentiles.flatten())
    zscore_counts = stats.zscore(all_counts.flatten())
    corr, p = stats.spearmanr(zscore_perc, zscore_counts)
    legend_str = f'Corr : {np.round(corr,3)}' + "\n" + f'p_val : {np.round(p,3)}'
    ax[change_ind].scatter(zscore_counts, zscore_perc, 
            label = legend_str, c = fin_colors)
    ax[change_ind].set_xlabel('Counts')
    ax[change_ind].legend(loc = 'lower right')
    ax[change_ind].set_title(f'Change {change_ind}')
    ax[change_ind].set(adjustable='box', aspect='equal')
ax[0].set_ylabel('Percentile')
plt.suptitle('Split region Count Percentile Correlation')
fig.savefig(os.path.join(plot_dir,'split_region_count_percentile'))
plt.close(fig)
#plt.show()

## Inter-session comparison of firing rates to percentiles
long_percentiles = np.reshape(all_percentiles, (-1, all_percentiles.shape[-1]))
long_firing = np.reshape(all_firing_mean, (-1, all_firing_mean.shape[-1]))
inds = np.array(list(np.ndindex(all_percentiles.shape[:2])))
firing_perc_frame = pd.DataFrame({
                'animal' : fin_animal_encoding,
                'session' : inds[:,0],
                'split' : inds[:,1],
                'c0' : long_percentiles[:,0],
                'c1' : long_percentiles[:,1],
                'c2' : long_percentiles[:,2],
                'f0' : long_firing[:,0],
                'f1' : long_firing[:,1],
                    })

fig,ax = plt.subplots(1,all_percentiles.shape[-1], 
        sharex= True, sharey=True, figsize = (15,5))
for change_ind in range(all_percentiles.shape[-1]):
    #change_ind = 0
    this_percentiles = all_percentiles[:,:,change_ind]
    zscore_perc = stats.zscore(this_percentiles.flatten())
    zscore_counts = stats.zscore(all_counts.flatten())
    corr, p = stats.spearmanr(zscore_perc, zscore_counts)
    legend_str = f'Corr : {np.round(corr,3)}' + "\n" + f'p_val : {np.round(p,3)}'
    ax[change_ind].scatter(zscore_counts, zscore_perc, 
            label = legend_str, c = fin_colors)
    ax[change_ind].set_xlabel('Counts')
    ax[change_ind].legend(loc = 'lower right')
    ax[change_ind].set_title(f'Change {change_ind}')
    ax[change_ind].set(adjustable='box', aspect='equal')
ax[0].set_ylabel('Percentile')
plt.suptitle('Split region Count Percentile Correlation')
fig.savefig(os.path.join(plot_dir,'split_region_count_percentile'))
plt.close(fig)
#plt.show()

scaled_frame_values = ss().fit_transform(firing_perc_frame.values)
for ind in range(scaled_frame_values.shape[1]):
    firing_perc_frame.iloc[:,ind] = scaled_frame_values[:,ind]

mods = [smf.ols(formula=f'c{ind} ~ f0 * f1', data=firing_perc_frame)\
                        for ind in range(all_percentiles.shape[-1])]
res = [x.fit() for x in mods]
pvals = [x.pvalues for x in res]
coeffs = [x.params for x in res] 

fit_params_frame_list = []
for ind in range(all_percentiles.shape[-1]):
    this_frame = pd.DataFrame({
        'pvals' : pvals[ind],
        'coeffs' : coeffs[ind],
        'change_ind' : ind
        })
    fit_params_frame_list.append(this_frame)
fin_fit_params_frame = pd.concat(fit_params_frame_list)
fin_fit_params_frame['sig'] = fin_fit_params_frame['pvals'] < 0.05

for change_ind in range(all_percentiles.shape[-1]):
    this_fit_frame = \
            fin_fit_params_frame[fin_fit_params_frame.change_ind == change_ind]
    this_str = this_fit_frame[['coeffs','pvals','sig']].to_string()
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(121, projection = '3d')
    ax.scatter(*firing_perc_frame[['f0','f1',f'c{change_ind}']].values.T, 
            s = 80, color = fin_colors)
    ax.set_xlabel('Firing1')
    ax.set_ylabel('Firing2')
    ax.set_zlabel('Percentile')
    ax2 = fig.add_subplot(122)
    ax2.text(0.1,0.5,this_str)
    plt.suptitle(f'Split region Firing Percentile Correlation : Change {change_ind}')
    fig.savefig(
            os.path.join(plot_dir,
                f'split_region_firing_percentile_change{change_ind}'))
    plt.close(fig)

## Intra-session comparison of median changepoint variance to percentiles
#session_ind = 0
all_tau_var_frames = []
all_fit_frames = []

var_names = ['var1','var2','perc']

for session_ind in range(all_tau_vars.shape[0]):
    session_basename = os.path.basename(split_frame.path.iloc[session_ind])
    this_tau_vars = all_tau_vars[session_ind]
    median_tau_vars = np.median(this_tau_vars,axis=2).swapaxes(1,2)
    median_tau_vars_long = \
            np.reshape(median_tau_vars, (-1, median_tau_vars.shape[-1]))
    this_percentiles = all_percentiles[session_ind]
    inds = np.array(list(np.ndindex(this_percentiles.shape)))

    this_frame = pd.DataFrame({
                    'animal_num' : animal_names[session_ind],
                    'split' : inds[:,0],
                    'change_ind' : inds[:,1],
                    'perc' : this_percentiles.flatten(),
                    'var1' : median_tau_vars_long[:,0],
                    'var2' : median_tau_vars_long[:,1]})
    all_tau_var_frames.append(this_frame)

    for change_ind in range(all_percentiles.shape[-1]):
        change_frame = this_frame.query(f'change_ind == {change_ind}')
        for this_var in var_names:
            change_frame[this_var] = \
                    ss().fit_transform(change_frame[this_var][:,np.newaxis])

        mods = smf.ols(formula=f'perc ~ var1 * var2', data=change_frame)
        res = mods.fit()
        pvals = res.pvalues 
        coeffs = res.params
        fit_frame = pd.DataFrame(dict(pvals = pvals, coeffs = coeffs, 
                            sig = pvals<0.05))

        fit_frame['animal_num'] = animal_names[session_ind]
        fit_frame['change_ind'] = change_ind
        all_fit_frames.append(fit_frame)

        this_str = fit_frame[['coeffs','pvals','sig']].to_string()
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(121, projection = '3d')
        ax.scatter(*change_frame[['var1','var2','perc']].values.T, s = 80)
        ax.set_xlabel('Var1')
        ax.set_ylabel('Var2')
        ax.set_zlabel('Percentile')
        ax2 = fig.add_subplot(122)
        ax2.text(0.1,0.5,this_str)
        plt.suptitle('Split region Tau Var Percentile Correlation : '\
                f'Change {change_ind}')
        fig.savefig(
                os.path.join(plot_dir,session_basename,
                    f'split_region_TauVar_percentile_change{change_ind}'))
        plt.close(fig)

fin_tau_var_frame = pd.concat(all_tau_var_frames)


agg_fit_frames = []

for change_ind in range(all_percentiles.shape[-1]):
    change_frame = fin_tau_var_frame.query(f'change_ind == {change_ind}')
    for this_var in var_names:
        change_frame.loc[:,this_var] = \
                ss().fit_transform(change_frame.loc[:,this_var][:,np.newaxis])

    mods = smf.ols(formula=f'perc ~ var1 * var2', data=change_frame)
    res = mods.fit()
    pvals = res.pvalues 
    coeffs = res.params
    fit_frame = pd.DataFrame(dict(pvals = pvals, coeffs = coeffs, 
                        sig = pvals<0.05))
    fit_frame['change_ind'] = change_ind
    agg_fit_frames.append(fit_frame)

    animal_encoding, unique_names = pd.factorize(change_frame.animal_num)
    colors = cmap(animal_encoding)

    this_str = fit_frame[['coeffs','pvals','sig']].to_string()
    fig = plt.figure(figsize = (10,5))
    ax = fig.add_subplot(121, projection = '3d')
    ax.scatter(*change_frame[['var1','var2','perc']].values.T, s = 80,
            color = colors)
    ax.set_xlabel('Var1')
    ax.set_ylabel('Var2')
    ax.set_zlabel('Percentile')
    ax2 = fig.add_subplot(122)
    ax2.text(0.1,0.5,this_str)
    plt.suptitle('agg_Split region Tau Var Percentile Correlation : '\
            f'Change {change_ind}')
    fig.savefig(
            os.path.join(plot_dir,
                f'split_region_agg_TauVar_percentile_change{change_ind}'))
    plt.close(fig)

fin_agg_fit_frame = pd.concat(agg_fit_frames)
