"""
Correlation plots for actual and shuffle data per dataset
Plots:
    1) All trials correlations
    2) Per taste correlations
    3) Changepoint distributions for both datasets to make sure we're comparing
        the right order of transitions between regions
    4) Changepoint distributions per taste (as above)
    5) Raster plots overlayed with states/changepoints
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
from scipy.stats import zscore
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture as gaussmix
from sklearn.preprocessing import StandardScaler as ss
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

sys.path.append('/media/bigdata/firing_space_plot/'\
        'firing_analyses/transition_corrs/all_tastes')
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
    def to_dict(self):
        return dict(zip(['states','time_lims','bin_width','session_name'],
            [self.states,self.time_lims,self.bin_width,self.session_name]))

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

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################

plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/plots/multi_region'

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/changepoint_alignment/inter_region'
wanted_names = ['rho_percentiles','mode_tau','rho_shuffles',
        'tau_corrs','tau_list'] 

# Load pkl detailing which recordings have split changepoints
#data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
#        'transition_corrs/multi_region_frame.pkl'
data_dir_pkl = '/media/bigdata/firing_space_plot/firing_analyses/'\
        'transition_corrs/all_tastes/inter_region/multi_region_frame.pkl'
inter_frame = pd.read_pickle(data_dir_pkl)
inter_frame['animal_name'] = [x.split('_')[0] for x in inter_frame['name']]

#all_tau_ps = []

for num, data_dir in tqdm(enumerate(inter_frame.path)):
    #num = 0
    stim_t = 2000
    data_dir = inter_frame.path.iloc[num]
    dat = ephys_data(data_dir)


    this_info = check_data(data_dir)
    this_info.run_all()
    inter_region_paths = [path for  num,path in enumerate(this_info.pkl_file_paths) \
                    if num in this_info.region_fit_inds]
    state4_models = [path for path in inter_region_paths if '4state' in path]
    #split_basenames = [os.path.basename(x) for x in state4_models]
    # Check params for both fits add up
    if not len(state4_models):
        print(f'No models for {data_dir}')
        continue

    check_params_bool = params_from_path(state4_models[0]).to_dict() ==\
                        params_from_path(state4_models[1]).to_dict()
    region_check = all([any([region_name in params_from_path(x).model_name \
                        for region_name in ['gc','bla']])\
                        for x in state4_models])
    if not (check_params_bool and region_check):
        raise Exception('Fit params dont match up')
    params = params_from_path(inter_region_paths[0]) 

    data_list = [load_model_data(this_path) for this_path in state4_models]
    tau_list, spike_list, full_tau_list = list(zip(*data_list)) 
    tau_vars = np.array([np.var(x,axis=0) for x in full_tau_list])
    spike_long_list = [np.reshape(x,(-1,*x.shape[2:])) for x in spike_list]
    trial_inds_list = np.array_split(np.arange(spike_long_list[0].shape[0]),8)

    session_name = inter_frame.name.iloc[num]
    #animal_name = session_name.split("_")[0]
    this_plot_dir = os.path.join(plot_dir, session_name)
    if not os.path.exists(this_plot_dir):
        os.makedirs(this_plot_dir)

    with tables.open_file(dat.hdf5_path,'r') as hf5:
        for this_name in wanted_names:
            globals()[this_name] = hf5.get_node(save_path, this_name)[:] 

    # Convert tau's to real-values
    tau_list = np.array(tau_list)
    tau_list = (tau_list * params.bin_width) + params.time_lims[0]
    tau_list = tau_list - stim_t
    tau_corr_dat = [stats.spearmanr(x,y) for x,y in zip(*tau_list)] 
    tau_corrs, tau_ps = list(zip(*tau_corr_dat))
    #all_tau_ps.append(tau_ps)

    #tau_shuffle = [stats.spearmanr(np.random.permutation(x),y) for x,y in zip(*tau_list)] 
    tau_shuffle = np.array([(np.random.permutation(x),y) for x,y in zip(*tau_list)]).swapaxes(0,1)
    rho_shuffles, tau_sh_ps = list(zip(*[stats.spearmanr(*x) for x in tau_shuffle.swapaxes(0,1)]))
    #shuffle_count = 1000
    #tau_shuffle = np.array([[np.array(this_tau)[np.random.choice(
    #                        np.arange(len(this_tau)), shuffle_count)] \
    #                for this_tau in this_split] for this_split in tau_list])


    ########################################
    #|  _ \| | ___ | |_ ___ 
    #| |_) | |/ _ \| __/ __|
    #|  __/| | (_) | |_\__ \
    #|_|   |_|\___/ \__|___/
    ########################################

    ########################################
    ## Scatter plot for each transition
    ########################################
    fin_plot_dir = os.path.join(plot_dir,params.session_name)

    x = np.linspace(*params.time_lims) - stim_t
    # Plot scatter plots for each tau and tau shuffled
    fig, ax = plt.subplots(len(tau_list[0]),2, 
            sharex = True, sharey=True, figsize = (7,10))
    for change_num in range(len(tau_list[0])):
        rho_str = f'Rho : {np.round(tau_corrs[change_num],3)}'
        percentile_str = f'Perc : {rho_percentiles[change_num]}'
        p_val_str = f'p_val : {np.round(tau_ps[change_num],3)}'
        ax[change_num, 0].scatter(*tau_list[:,change_num], alpha = 0.5,
                label = rho_str )#+ "\n" + percentile_str + "\n" + p_val_str)
        ax[change_num, 0].legend(loc = 'upper left')
        ax[change_num, 0].set_ylabel(f'Change {change_num}')
        rho_str_sh = f'Rho : {np.round(np.mean(rho_shuffles[change_num]),3)}'# +\
                #f'+/- {np.round(np.std(rho_shuffles[change_num]),3)}'
        ax[change_num, 1].scatter(*tau_shuffle[:,change_num], 
                alpha = 0.2,label = rho_str_sh)
        ax[change_num, 1].legend(loc = 'upper left')
        ax[change_num, 0].set(adjustable='box', aspect='equal')
        ax[change_num, 1].set(adjustable='box', aspect='equal')
        ax[change_num, 0].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        ax[change_num, 1].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        #ax[change_num, 0].text(0,0,f'Rho : {np.round(tau_corrs[change_num],3)}')
    plt.suptitle(params.session_name + ": Tau Corrs")
    ax[0,0].set_title('Actual')
    ax[0,1].set_title('Shuffle')
    #plt.show()
    fig.savefig(os.path.join(fin_plot_dir,f'{params.session_name}_tau_scatter'))
    plt.close(fig)
    #plt.show()

    ## Scatter plots with contours
    fig, ax = plt.subplots(2, len(tau_list[0]),
            sharex = True, 
            #sharey=True, 
            figsize = (10,7))
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

        ax[0, change_num].scatter(*actual_data, alpha = 0.5,)
        ax[0, change_num].set_title(rho_str + "\n" + p_val_str)
        #ax[0, change_num].legend(loc = 'upper left')
        ax[0, change_num].set_ylabel(f'Change {change_num}')
        rho_str_sh = f'Rho : {np.round(np.mean(rho_shuffles[change_num]),3)}'# +\
        p_val_str_sh = f'p_val : {np.round(tau_sh_ps[change_num],3)}'
                #f'+/- {np.round(np.std(rho_shuffles[change_num]),3)}'
        #ax[1, change_num].scatter(*shuffle_data, alpha = 0.2,label = rho_str_sh)
        ax[1, change_num].scatter(*shuffle_data, alpha = 0.5)
        ax[1, change_num].set_title(rho_str_sh + "\n" + p_val_str_sh)
        ax[1, change_num].contour(X, Y, Z_shuff, alpha = 0.7)
        ax[0, change_num].contour(X, Y, Z_actual, alpha = 0.7)
        #ax[1, change_num].legend(loc = 'upper left')
        ax[0, change_num].set(adjustable='box', aspect='equal')
        ax[1, change_num].set(adjustable='box', aspect='equal')
        ax[0, change_num].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        ax[1, change_num].plot(x,x, alpha = 0.5, 
                linestyle = 'dashed', color = 'red')
        #ax[0, change_num].text(0,0,f'Rho : {np.round(tau_corrs[change_num],3)}')
    plt.suptitle(params.session_name + ": Tau Corrs")
    plt.tight_layout()
    ax[0,0].set_title('Actual')
    ax[1,0].set_title('Shuffle')
    fig.savefig(os.path.join(fin_plot_dir,
        f'{params.session_name}_tau_scatter_contour'), dpi = 300)
    plt.close(fig)
    #plt.show()

    spike_lims = [1500,4000]
    create_split_changepoint_plots(
                        *spike_long_list, *tau_list,
                        trial_inds_list, params, 0, spike_lims)

    ########################################
    ## Distribution of Tau for each region 
    ########################################
    cmap = plt.get_cmap('tab10')
    bins = np.linspace(*params.time_lims,50)
    fig, ax = plt.subplots(2,1, sharex=True, sharey = True)
    for num, region in enumerate(tau_list + np.random.random(tau_list.shape)*50):
        for change_num, this_change in enumerate(region):
            ax[num].hist(this_change, bins, 
                    label = f'Trans {change_num}', alpha = 0.7, 
                    color = cmap(change_num))
            ax[num].hist(this_change, bins, histtype = 'step', 
                    color = cmap(change_num)) 
        ax[num].set_title(f'Region {num}')
    ax[-1].legend()
    plt.suptitle(params.session_name + ": Tau Dists")
    fig.savefig(os.path.join(fin_plot_dir,f'{params.session_name}_tau_dists'))
    plt.close(fig)
    #plt.show()
