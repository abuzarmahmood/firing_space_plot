"""
Noise correlations between:
    1) Whole trial BLA and GC neurons (and shuffle)
    2) Intra region BLA and GC neurons
    3) Binned for 1) and 2)
    4*) Repeat above analyses specifically for TASTE RESPONSIVE/DISCRIMINATIVE
        neurons

Shuffles:
    1) Trial shuffle to show trial-to-trial variability
    - Do STRINGENT trial shuffle
        - Shuffle pairs of trials for same taste in CHRONOLOGICAL ORDER


    ** Actually this second control should not be necessary
    ** If the first shuffle shows lower R than the actual data,
    ** then there should be no need for this
    2) Random chance of 2 neurons with same firing characteristics 
        generating that R value
        - Take Mean + Variance of both neurons to GENERATE trials
            with similar firing rate characteristics and calculate R
"""

########################################
# ____       _               
#/ ___|  ___| |_ _   _ _ __  
#\___ \ / _ \ __| | | | '_ \ 
# ___) |  __/ |_| |_| | |_) |
#|____/ \___|\__|\__,_| .__/ 
#                     |_|    
########################################

########################################
# Import modules
########################################

import os
import sys
import scipy.stats as stats
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tables
from joblib import Parallel, delayed, cpu_count
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare
import pylab as plt

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

def parallelize(func, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

data_dir = sys.argv[1]
#data_dir = '/media/bigdata/Abuzar_Data/AM12/AM12_4Tastes_191106_085215/'
plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/spike_noise_corrs/Plots'
name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]
fin_plot_dir = os.path.join(plot_dir, fin_name)

if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()
#dat.get_firing_rates()
#dat.firing_rate_params = dat.default_firing_params 
spikes = np.array(dat.spikes)

# Path to save noise corrs in HDF5
save_path = '/ancillary_analysis/spike_noise_corrs'

with tables.open_file(dat.hdf5_name,'r+') as hf5:
    if save_path not in hf5:
        hf5.create_group(os.path.dirname(save_path),os.path.basename(save_path),
                createparents = True)

########################
# / ___|___  _ __ _ __ 
#| |   / _ \| '__| '__|
#| |__| (_) | |  | |   
# \____\___/|_|  |_|   
########################

# If array not present, then perform calculation
with tables.open_file(dat.hdf5_name,'r+') as hf5:
    if os.path.join(save_path, 'corr_array') not in hf5:
        present_bool = False 
    else:
        present_bool = True

present_bool = False
if not present_bool: 

    ##################################################
    ## Whole Trial
    ##################################################
    time_lims = [2000,4000]
    temp_spikes = spikes[...,time_lims[0]:time_lims[1]]
    region_spikes = [temp_spikes.swapaxes(0,2)[region_inds]\
            for region_inds in dat.region_units]

    unit_count = [len(x) for x in region_spikes]
    wanted_order = np.argsort(unit_count)[::-1]
    temp_region_spikes = [region_spikes[x] for x in wanted_order]
    sorted_unit_count = [len(x) for x in temp_region_spikes]
    all_pairs = np.arange(1,1+sorted_unit_count[0])[:,np.newaxis].\
            dot(np.arange(1,1+sorted_unit_count[1])[np.newaxis,:])
    pair_inds = list(zip(*np.where(np.tril(all_pairs))))

    sum_spikes = [np.sum(x,axis=-1) for x in temp_region_spikes]
    # Try detrending with 1st order difference before corr
    diff_sum_spikes = [np.diff(region,axis=1) for region in sum_spikes]
    # Zscore along trial axis to normalize values across neurons
    diff_sum_spikes = [stats.zscore(region,axis=1) for region in diff_sum_spikes]

    # Perform correlation over all pairs for each taste separately
    # Compare values to corresponding shuffles
    repeats = 1000

    def calc_corrs(this_ind):
        this_pair = np.array([diff_sum_spikes[0][this_ind[0]], 
                                diff_sum_spikes[1][this_ind[1]]])
        #corrs = np.abs([spearmanr(this_taste)[0] for this_taste in this_pair.T])
        out = \
                list(zip(*[spearmanr(this_taste) for this_taste in this_pair.T]))
        corrs, p_vals = [np.array(x) for x in out] 

        # Random ok for now but replace with STRINGENT shuffle
        #shuffled_corrs = np.abs(np.array(\
        out =  \
                        [[spearmanr(np.random.permutation(this_taste[:,0]), 
                                this_taste[:,1]) \
                            for this_taste in this_pair.T]\
                            for x in np.arange(repeats)]
        trans_out = [list(zip(*x)) for x in out]
        shuffled_corrs, shuffled_p_vals = [np.array(x) for x in list(zip(*trans_out))]
        percentile_list = [percentileofscore(shuffle,actual) \
                                for shuffle, actual in \
                                zip(shuffled_corrs.T,corrs)]        

        # Shuffle trials in pairs (adjacent trials) to remove
        # correlations created by long-term state changes
        #stringent_shuffle_inds = np.arange(this_pair[0].shape[0]).\
        #                            reshape(-1,2)[:,::-1].flatten()
        #out =  [spearmanr(this_taste[stringent_shuffle_inds,0], this_taste[:,1]) \
        #                    for this_taste in this_pair.T]
        #trans_out = list(zip(*out))
        #stringent_shuffled_corrs, stringent_shuffled_p_vals = \
        #        [np.array(x) for x in trans_out]

        return corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentile_list#,\
                    #stringent_shuffled_corrs, stringent_shuffled_p_vals

    # Check corrs are significant individually aswell
    out = parallelize(calc_corrs,pair_inds)
    [corr_array, p_val_array, 
    shuffled_corrs, shuffled_p_vals, percentile_array] = \
            [np.array(x) for x in list(zip(*out))]
            #stringent_shuffled_corrs, stringent_shuffled_p_vals] = \

    names = ['corr_array', 'p_val_array', 
            'shuffled_corrs', 'shuffled_p_vals',
            'percentile_array']#, 
            #'stringent_shuffled_corrs', 'stringent_shuffled_p_vals'] 

    # Remove any nans
    # Assuming nans are shared across arrays
    # Take out entire pair if nan is present
    nan_inds = np.where(np.isnan(corr_array))[0]
    keep_inds = [x for x in np.arange(corr_array.shape[0]) \
                        if x not in nan_inds]
    for array in names:
        globals()[array] = eval(array)[keep_inds]

    with tables.open_file(dat.hdf5_name,'r+') as hf5:
        for array in names:
            remove_node(os.path.join(save_path, array),hf5) 
            hf5.create_array(save_path, array, eval(array))

    #mean_shuffled_corrs = np.mean(shuffled_corrs,axis=1)
    #mean_shuffled_p_vals = np.mean(shuffled_p_vals,axis=1)


    ##################################################
    # ____  _       _   _   _             
    #|  _ \| | ___ | |_| |_(_)_ __   __ _ 
    #| |_) | |/ _ \| __| __| | '_ \ / _` |
    #|  __/| | (_) | |_| |_| | | | | (_| |
    #|_|   |_|\___/ \__|\__|_|_| |_|\__, |
    #                               |___/ 
    ##################################################

    # Perform bonferroni correction
    #alpha = 0.05/mat_inds.shape[0]
    alpha = 0.05

    #========================================
    # Side-by-side significance matrices
    #fig,ax = plt.subplots(1,2)
    #for this_ax, this_dat in zip(ax,[p_val_array, mean_shuffled_p_vals]):
    #plt.imshow(p_val_array <= alpha,aspect='auto', vmin = 0, vmax = 1)
    #plt.xlabel('Taste')
    #plt.ylabel('All Neuron Pair Combinations')
    #plt.suptitle('Noise Correlation Significance')
    #plt.show()

    # Side-by-side corrleation matrices
    #fig,ax = plt.subplots(1,2)
    #for this_ax, this_dat in zip(ax,[corr_array, mean_shuffled_corrs]):
    #    this_ax.imshow(this_dat,aspect='auto', vmin = 0, vmax = 1)
    #plt.show()

    #========================================
    # Same plot as above but histogram with sides that are nrns
    # and bins counting how many significant correlations
    mat_inds = np.array(list(np.ndindex(p_val_array.shape)))
    inds = np.array(pair_inds)
    sig_hist_array = np.zeros(inds[-1]+1)
    for this_mat_ind,this_val in zip(mat_inds[:,0],(p_val_array<alpha).flatten()):
        sig_hist_array[inds[this_mat_ind,0], inds[this_mat_ind,1]] += \
                                        this_val
    sorted_region_names = [dat.region_names[x] for x in wanted_order]

    fig = plt.figure()
    plt.imshow(sig_hist_array,aspect='auto',cmap='viridis');
    # ROWS are region0, COLS are region1
    plt.xlabel(str(sorted_region_names[1])+' Neuron #');
    plt.ylabel(str(sorted_region_names[0])+' Neuron #')
    plt.suptitle('Count of Significant\nNoise Correlations across all comparisons')
    plt.colorbar()
    fig.savefig(os.path.join(fin_plot_dir,fin_name+'_sig_nrn_table'),dpi=300)
    plt.close(fig)
    #plt.show()

    #========================================
    # histogram of corr percentile relative to respective shuffle
    fig = plt.figure()
    freq_hist = np.histogram(percentile_array.flatten(),percentile_array.size//20)
    chi_test = chisquare(freq_hist[0])
    plt.hist(percentile_array.flatten(),percentile_array.size//20)
    plt.suptitle('Percentile relative to respective shuffles\n' +\
            'Chi_sq vs. Uniform Discrete Dist \n p_val :' \
            + str(np.format_float_scientific(chi_test[1],3)))
    plt.xlabel('Percentile of Corr Relative to respective shuffle distribution')
    plt.ylabel('Frequency')
    fig.savefig(os.path.join(\
            fin_plot_dir,fin_name+'_random_shuffle_percentiles'),
            dpi=300)
    plt.close(fig)
    #plt.show()

    #========================================
    # heatmap of corr percentile relative to respective shuffle
    #plt.imshow(percentile_array,aspect='auto',cmap='viridis');plt.show()

    #========================================
    # histogram of corr percentile relative to respective shuffle
    # Plot scatter plots to show correlation of actual data and a shuffle
    # Find MAX corr
    corr_mat_inds = np.where(corr_array == np.max(corr_array,axis=None))
    nrn_inds = pair_inds[corr_mat_inds[0][0]]
    this_pair = np.array([diff_sum_spikes[0][nrn_inds[0],...,corr_mat_inds[1][0]], 
                            diff_sum_spikes[1][nrn_inds[1],...,corr_mat_inds[1][0]]])

    #shuffled_pair = np.array([[np.random.permutation(this_pair[0]),this_pair[1]]\
    #            for x in np.arange(repeats)])
    #shuffled_pair = np.reshape(\
    #        np.moveaxis(shuffled_pair,0,-1),(shuffled_pair.shape[1],-1))

    fig, ax = plt.subplots(2,2)
    fig.suptitle('Firing Rate Scatterplots')
    ax[0,0].set_title('Actual Data - Pair : ' + str(nrn_inds) +\
                            '\nTaste :' + str(corr_mat_inds[1][0]))
    ax[1,0].set_title('Shuffle') 
    ax[0,0].scatter(this_pair[0],this_pair[1])
    #ax[1,0].scatter(shuffled_pair[0],shuffled_pair[1]);
    ax[1,0].plot(this_pair[0]);
    ax[1,0].plot(this_pair[1]);

    # Find MIN corr
    corr_mat_inds = np.where(corr_array == np.min(corr_array,axis=None))
    nrn_inds = pair_inds[corr_mat_inds[0][0]]
    this_pair = np.array([diff_sum_spikes[0][nrn_inds[0],...,corr_mat_inds[1][0]], 
                            diff_sum_spikes[1][nrn_inds[1],...,corr_mat_inds[1][0]]])

    #shuffled_pair = np.array([[np.random.permutation(this_pair[0]),this_pair[1]]\
    #            for x in np.arange(repeats)])
    #shuffled_pair = np.reshape(\
    #        np.moveaxis(shuffled_pair,0,-1),(shuffled_pair.shape[1],-1))

    ax[0,1].set_title('Actual Data - Pair : ' + str(nrn_inds) +\
                            '\nTaste :' + str(corr_mat_inds[1][0]))
    ax[1,1].set_title('Shuffle') 
    ax[0,1].scatter(this_pair[0],this_pair[1])
    #ax[1,0].scatter(shuffled_pair[0],shuffled_pair[1]);
    ax[1,1].plot(this_pair[0]);
    ax[1,1].plot(this_pair[1]);

    for this_ax in ax.flatten():
        this_ax.set_xlabel('Nrn 1 Firing')
        this_ax.set_ylabel('Nrn 0 Firing')
    plt.tight_layout()
    fig.savefig(os.path.join(fin_plot_dir,fin_name+'_example_corrs'),dpi=300)
    #plt.show()

    ##################################################
    ## WITH STRINGENT SHUFFLE
    ##################################################

    sig_array = p_val_array <= alpha
    #stringent_shuffle_sig_array = stringent_shuffled_p_vals <= alpha
    #overlay_array = sig_array*stringent_shuffle_sig_array 
    #net_sig_array = sig_array.copy()
    #net_sig_array[np.where(overlay_array)] = 0

    ## Net mean significant fraction
    #net_mean_sig_frac = np.mean((sig_array*1) - overlay_array,axis=None)

    #========================================
    # Side-by-side significance matrices
    #titles = ['Actual','Stringent_Shuffle','Intersection']
    #fig,ax = plt.subplots(1,3)
    #for num,(this_ax, this_dat) in enumerate(zip(
    #                    ax,[sig_array, stringent_shuffle_sig_array, overlay_array])):
    #    this_ax.imshow(this_dat, aspect='auto')
    #    this_ax.set_title(titles[num] + "\n{} Total".format(this_dat.sum()))
    fig,ax = plt.subplots(1,1)
    ax.imshow(sig_array,origin='lower',aspect='auto')
    ax.set_xlabel('Taste')
    ax.set_ylabel('All Neuron Pair Combinations')
    #ax[0].set_xlabel('Taste')
    #ax[0].set_ylabel('All Neuron Pair Combinations')
    plt.suptitle('Noise Correlation Significance\n{:.2f} % net significant corrs'\
                            .format(np.mean(sig_array,axis=None) * 100))
                            #.format(net_mean_sig_frac * 100))
    plt.tight_layout(rect=[0, 0.0, 1, 0.9])
    fig.savefig(os.path.join(fin_plot_dir,fin_name+'_sig_array'),dpi=300)

    #========================================
    # histogram of corr percentile relative to stringent shuffle
    #cat_array = np.concatenate([corr_array.flatten(),
    #                        stringent_shuffled_corrs.flatten()],
    #                        axis = -1)
    #min_val = np.min(cat_array)
    #max_val = np.max(cat_array)
    #bins = np.linspace(min_val,max_val,percentile_array.size//20)
    #freq_hist = np.histogram(corr_array.flatten(),bins)
    #stringent_shuffle_freq_hist = \
    #        np.histogram(stringent_shuffled_corrs.flatten(),bins)
    #chi_test = chisquare(freq_hist[0],stringent_shuffle_freq_hist[0])

    #fig = plt.figure()
    #plt.hist(corr_array.flatten(),label='Actual')
    #plt.hist(stringent_shuffled_corrs.flatten(), label = 'Shuffled')
    #plt.legend()
    #plt.suptitle('Actual vs Stringent Shuffled corrs\n'\
    #        + 'Chi-sq p_val= ' + str(np.format_float_scientific(chi_test[1],3)))
    #plt.xlabel('Spearman R value')
    #plt.ylabel('Frequency')
    #fig.savefig(os.path.join(fin_plot_dir,fin_name+'_corr_hist_comparison'),dpi=300)
    ##plt.show()

    #========================================
    # For significant correlations, plot summed spikes in chornological order
    # to see whether there is a clear trend
    sig_nrns = inds[np.where(sig_array)[0]]
    sig_tastes = np.where(sig_array)[1]
    sig_comparisons = np.concatenate([sig_nrns,sig_tastes[:,np.newaxis]],axis=-1)

    # How many pairs in one plot
    # This will double because of line and corr plots
    plot_thresh = 8

    num_figs = int(np.ceil(sig_comparisons.shape[0]/plot_thresh))

    for this_fig_num in np.arange(num_figs):
        fig,ax = visualize.gen_square_subplots(int(plot_thresh*2))
        ax_inds = np.array(list(np.ndindex(ax.shape)))
        #fig,ax = plt.subplots(len(sig_comparisons),2)
        for num, this_comp in enumerate(sig_comparisons\
                [(this_fig_num*plot_thresh):((this_fig_num+1)*plot_thresh)]):
            #ax[num,0].plot(sum_spikes[0][this_comp[0],:,this_comp[-1]])
            #ax[num,0].plot(sum_spikes[1][this_comp[1],:,this_comp[-1]])
            #ax[num,-1].scatter(sum_spikes[0][this_comp[0],:,this_comp[-1]],
            #            diff_sum_spikes[1][this_comp[1],:,this_comp[-1]])
            region0 = diff_sum_spikes[0][this_comp[0],:,this_comp[-1]]
            region1 = diff_sum_spikes[1][this_comp[1],:,this_comp[-1]]
            line_plot_ind = tuple(np.split(ax_inds[2*num],2))
            scatter_plot_ind = tuple(np.split(ax_inds[2*num+1],2))
            ax[line_plot_ind][0].plot(region0)
            ax[line_plot_ind][0].plot(region1)
            ax[scatter_plot_ind][0].scatter(region0,region1,s=5)
            #this_ax.title(str(this_comp[:2]))
        plt.suptitle('Net significant comparisons')
        fig.savefig(os.path.join(\
                fin_plot_dir,fin_name+'_net_sig_comps_{}'.format(this_fig_num)),
                dpi=300)
        plt.close(fig)
    #plt.show()

    # Create same plots with pairs that were affected by stringent shuffling
    #sig_nrns = inds[np.where(overlay_array)[0]]
    #sig_tastes = np.where(overlay_array)[1]
    #sig_comparisons = np.concatenate([sig_nrns,sig_tastes[:,np.newaxis]],axis=-1)

    #fig,ax = visualize.gen_square_subplots(int(len(sig_comparisons)*2))
    #ax_inds = np.array(list(np.ndindex(ax.shape)))
    ##fig,ax = plt.subplots(len(sig_comparisons),2)
    #for num, this_comp in enumerate(sig_comparisons):
    #    #ax[num,0].plot(sum_spikes[0][this_comp[0],:,this_comp[-1]])
    #    #ax[num,0].plot(sum_spikes[1][this_comp[1],:,this_comp[-1]])
    #    #ax[num,-1].scatter(sum_spikes[0][this_comp[0],:,this_comp[-1]],
    #    #            sum_spikes[1][this_comp[1],:,this_comp[-1]])
    #    region0 = diff_sum_spikes[0][this_comp[0],:,this_comp[-1]]
    #    region1 = diff_sum_spikes[1][this_comp[1],:,this_comp[-1]]
    #    line_plot_ind = tuple(np.split(ax_inds[2*num],2))
    #    scatter_plot_ind = tuple(np.split(ax_inds[2*num+1],2))
    #    ax[line_plot_ind][0].plot(region0)
    #    ax[line_plot_ind][0].plot(region1)
    #    ax[scatter_plot_ind][0].scatter(region0,region1, s=5)
    #plt.suptitle('Shuffle damagable comparisons')
    #fig.savefig(os.path.join(fin_plot_dir,fin_name+'_shuffle_damage_comps'),dpi=300)
    #plt.close(fig)
    ##plt.show()
