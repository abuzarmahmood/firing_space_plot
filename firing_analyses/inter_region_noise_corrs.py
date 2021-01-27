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

    3) Calculation of correlation for baseline
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

def parallelize(func, fixed_args, iterator):
    return Parallel(n_jobs = cpu_count()-2)\
            (delayed(func)(*fixed_args,this_iter) for this_iter in tqdm(iterator))

def remove_node(path_to_node, hf5, recursive = False):
    if path_to_node in hf5:
        hf5.remove_node(os.path.dirname(path_to_node),
                    os.path.basename(path_to_node), 
                    recursive = recursive)

#========================================#
## Corr calculation functions
#========================================#

def calc_corrs(array1,array2,ind_tuple, repeats = 1000):
    """
    Calculate correlations and shuffled correlations between given arrays
    inputs ::
        array1,array2 : nrns x trials
        ind_tuple : tuple for indexing neurons from array1 and array2
    """
    corr, p_val  = spearmanr(array1[ind_tuple[0]],array2[ind_tuple[1]])
    out = [spearmanr(\
                        np.random.permutation(array1[ind_tuple[0]]),
                        array2[ind_tuple[1]]) \
                for x in np.arange(repeats)]
    shuffled_corrs, shuffled_p_vals = np.array(out).T
    percentile = percentileofscore(shuffled_corrs,corr)

    return corr, p_val, shuffled_corrs, shuffled_p_vals, percentile

def taste_calc_corrs(array1,array2,ind_tuple):
    """
    Convenience function wrapping calc_corrs to extend to arrays with
    a taste dimension
    inputs ::
        array1,array2 : taste x nrns x trials
        ind_tuple : tuple for indexing neurons from array1 and array2
    """
    outs = list(zip(*[calc_corrs(this1,this2,ind_tuple) \
                    for this1,this2 in zip(array1,array2)]))
    outs = [np.array(x) for x in outs]
    corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentiles = outs
    return  corrs, p_vals, shuffled_corrs, shuffled_p_vals, percentiles

def gen_df(corr_array, p_val_array, percentiles, pair_list, label):
    """
    Helper function to generate dataframes for analyses
    """
    inds = np.array(list(np.ndindex(corr_array.shape)))
    if percentiles is None:
        percentiles = np.array([np.nan]*inds.shape[0])
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'pair_ind' : inds[:,0],
            'pair' : [pair_list[x] for x in inds[:,0]],
            'taste' : inds[:,1],
            'corr' : corr_array.flatten(),
            'p_vals' : p_val_array.flatten(),
            'percentiles' : percentiles.flatten()})

def gen_bin_df(corr_array, p_val_array, percentiles, pair_list, label):
    """
    Helper function to generate dataframes for analyses
    """
    inds = np.array(list(np.ndindex(corr_array.shape)))
    if percentiles is None:
        percentiles = np.array([np.nan]*inds.shape[0])
    return pd.DataFrame({
            'label' : [label] * inds.shape[0],
            'bin_num' : inds[:,0],
            'pair_ind' : inds[:,1],
            'pair' : [pair_list[x] for x in inds[:,1]],
            'taste' : inds[:,2],
            'corr' : corr_array.flatten(),
            'p_vals' : p_val_array.flatten(),
            'percentiles' : percentiles.flatten()})


################################################### 
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
#                                               
################################################### 

data_dir = sys.argv[1]
#data_dir = '/media/bigdata/Abuzar_Data/AM26/AM26_4Tastes_200829_100535'
plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/spike_noise_corrs/Plots'
name_splits = os.path.basename(data_dir[:-1]).split('_')
fin_name = name_splits[0]+'_'+name_splits[2]
fin_plot_dir = os.path.join(plot_dir, fin_name)

if not os.path.exists(fin_plot_dir):
    os.makedirs(fin_plot_dir)

dat = ephys_data(data_dir)
dat.get_spikes()
dat.get_region_units()
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
    # Pre-processing
    ##################################################

    time_lims = [2000,4000]
    temp_spikes = spikes[...,time_lims[0]:time_lims[1]]
    region_spikes = [temp_spikes.swapaxes(0,2)[region_inds]\
            for region_inds in dat.region_units]

    unit_count = [len(x) for x in region_spikes]
    wanted_order = np.argsort(unit_count)[::-1]
    sorted_region_names = [dat.region_names[x] for x in wanted_order]
    temp_region_spikes = [region_spikes[x] for x in wanted_order]
    sorted_unit_count = [len(x) for x in temp_region_spikes]
    all_pairs = np.arange(1,1+sorted_unit_count[0])[:,np.newaxis].\
            dot(np.arange(1,1+sorted_unit_count[1])[np.newaxis,:])
    #pair_inds = list(zip(*np.where(np.tril(all_pairs))))
    pair_inds = list(zip(*np.where(all_pairs)))

    sum_spikes = [np.sum(x,axis=-1) for x in temp_region_spikes]
    # Try detrending with 1st order difference before corr
    diff_sum_spikes = [np.diff(region,axis=1) for region in sum_spikes]
    # Zscore along trial axis to normalize values across neurons
    diff_sum_spikes = [stats.zscore(region,axis=1) for region in diff_sum_spikes]
    diff_sum_spikes = [np.moveaxis(x,-1,0) for x in diff_sum_spikes]

    ##################################################
    ## Inter-Region Whole Trial
    ##################################################

    # Perform correlation over all pairs for each taste separately
    # Compare values to corresponding shuffles

    out = list(zip(*parallelize(taste_calc_corrs,diff_sum_spikes,pair_inds)))
    out = [np.array(x) for x in out]

    inter_region_frame = gen_df(out[0], out[1], out[-1],
                                pair_inds,'inter_region')

    inter_region_shuffle_frame = gen_df(out[2], out[3], None,
                                pair_inds,'shuffle_inter_region')

    ##################################################
    ## BINNED Inter-Region 
    ##################################################
    # Sama pair_inds as above, add additional loop for bins
    bin_width = 500
    bin_count = np.diff(time_lims)[0]//bin_width 
    binned_spikes = [np.array(np.split(x,bin_count,axis=-1)) for x in temp_region_spikes]
    binned_sum_spikes = [np.sum(x,axis=-1) for x in binned_spikes]
    # Try detrending with 1st order difference before corr
    diff_bin_sum_spikes = [np.diff(region,axis=2) for region in binned_sum_spikes]
    # Zscore along trial axis to normalize values across neurons
    diff_bin_sum_spikes = [stats.zscore(region,axis=2) for region in diff_bin_sum_spikes]
    diff_bin_sum_spikes = [np.moveaxis(x,-1,1) for x in diff_bin_sum_spikes]

    # Better way to handle this list-nesting???
    out = [parallelize(taste_calc_corrs,this_bin,pair_inds) \
            for this_bin in zip(*diff_bin_sum_spikes)]
    out= [list(zip(*x)) for x in out]
    out= [[np.array(array) for array in this_bin] for this_bin in out]
    out= list(zip(*out))
    out= [np.array(x) for x in out]

    bin_inter_region_frame = gen_bin_df(out[0],out[1],out[-1],
                                        pair_inds, 'inter_region')
    shuffle_bin_inter_region_frame = gen_bin_df(out[2],out[3],None,
                                        pair_inds, 'shuffle_inter_region')

    ##################################################
    ## INTRA-Region Whole Trial
    ##################################################

    #this_save_path = os.path.join(save_path,'intra_region')

    pair_list = [list(it.combinations(np.arange(x.shape[1]),2)) \
                            for x in diff_sum_spikes]

    out0 = list(zip(*parallelize(taste_calc_corrs,\
                        [diff_sum_spikes[0],diff_sum_spikes[0]],pair_list[0])))
    out1 = list(zip(*parallelize(taste_calc_corrs,\
                        [diff_sum_spikes[1],diff_sum_spikes[1]],pair_list[1])))

    out0 = [np.array(x) for x in out0]
    out1 = [np.array(x) for x in out1]

    intra_region_frame = pd.concat([\
        gen_df(this_dat[0],this_dat[1],this_dat[-1],this_inds,region_name) \
        for this_dat,this_inds,region_name in \
        zip([out0,out1],pair_list,sorted_region_names)])

    shuffle_intra_region_frame = pd.concat([\
        gen_df(this_dat[2],this_dat[3],None,this_inds,'shuffle_'+region_name) \
        for this_dat,this_inds,region_name in \
        zip([out0,out1],pair_list,sorted_region_names)])

    ########################################
    ## Save arrays 
    ########################################
    frame_name_list = ['inter_region_frame',
                            'shuffle_intra_region_frame',
                            'intra_region_frame',
                            'shuffle_intra_region_frame']
    with tables.open_file(dat.hdf5_name,'r+') as hf5:
        for frame_name in frame_name_list:
            # Will only remove if array already there
            remove_node(os.path.join(save_path, frame_name),hf5, recursive=True)

    for frame_name in frame_name_list:
        # Save transformed array to HDF5
        eval(frame_name).to_hdf(dat.hdf5_name,  
                os.path.join(save_path, frame_name))

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
    # Same plot as above but histogram with sides that are nrns
    # and bins counting how many significant correlations

    # Create function to pull out significant pairs
    def gen_sig_mat(pd_frame, index_label):
        label_cond = pd_frame['label'] == index_label
        sig_cond = pd_frame['p_vals'] <= alpha
        sig_frame = pd_frame[label_cond & sig_cond]
        sig_pairs = sig_frame['pair']

        sig_hist_array = np.zeros([x+1 for x in pd_frame[label_cond]['pair'].max()])
        for this_pair in sig_pairs:
            sig_hist_array[this_pair] += 1
        return sig_hist_array, sig_frame

    inter_sig_hist_array,_ = gen_sig_mat(inter_region_frame.dropna(),'inter_region')
    intra_sig_hist_arrays = [gen_sig_mat(intra_region_frame.dropna(), region_name)[0]\
                            for region_name in sorted_region_names]
    #mat_inds = np.array(list(np.ndindex(p_val_array.shape)))
    #inds = np.array(pair_inds)
    #for this_mat_ind,this_val in zip(mat_inds[:,0],(p_val_array<alpha).flatten()):
    #    sig_hist_array[inds[this_mat_ind,0], inds[this_mat_ind,1]] += \
    #                                    this_val

    fig,ax = plt.subplots(1,3, figsize = (15,5))
    im = ax[0].imshow(inter_sig_hist_array,aspect='equal',cmap='viridis',vmin = 0,vmax = 4);
    ax[1].imshow(intra_sig_hist_arrays[0].T,aspect='equal',cmap='viridis',vmin = 0,vmax = 4);
    ax[2].imshow(intra_sig_hist_arrays[1].T,aspect='equal',cmap='viridis',vmin = 0,vmax = 4);
    # ROWS are region0, COLS are region1
    ax[0].set_xlabel(str(sorted_region_names[1])+' Neuron #');
    ax[0].set_ylabel(str(sorted_region_names[0])+' Neuron #')
    # ROWS are region0, COLS are region1
    ax[0].set_xlabel(str(sorted_region_names[1])+' Neuron #');
    ax[0].set_ylabel(str(sorted_region_names[0])+' Neuron #')
    ax[0].set_title('Inter-region')
    ax[1].set_title(sorted_region_names[0])
    ax[2].set_title(sorted_region_names[1])
    plt.suptitle('Count of Significant\nNoise Correlations across all comparisons')
    plt.colorbar(im)
    fig.savefig(os.path.join(fin_plot_dir,fin_name+'_sig_nrn_table'),dpi=300)
    plt.close(fig)
    #plt.show()

    #========================================
    # histogram of corr percentile relative to respective shuffle
    #========================================
    all_frame = pd.concat([inter_region_frame,intra_region_frame])
    percentile_list = [[x[0],x[1]['percentiles']] for x in list(all_frame.groupby('label'))]
    label_list, percentile_list = list(zip(*percentile_list))
    percentile_list = [x.dropna() for x in percentile_list]

    fig,ax = plt.subplots(1,len(label_list), figsize=(15,5))
    for this_ax,this_name, this_percentile in zip(ax,label_list,percentile_list):

        percentile_array = np.array(this_percentile).flatten()
        #freq_hist = np.histogram(percentile_array,percentile_array.size//20)
        # Use default binning (which tends to be more conservative)
        counts, bins, patches = this_ax.hist(percentile_array.flatten(),bins='auto')
        #chi_test = chisquare(freq_hist[0])
        chi_test = chisquare(counts)
        this_ax.set_title(this_name.upper() + ': p_val :' \
                + str(np.format_float_scientific(chi_test[1],3)))
        this_ax.set_xlabel('Percentile Relative to shuffle ')
        this_ax.set_ylabel('Frequency')
    plt.suptitle('Percentile relative to respective shuffles\n' +\
                'Chi_sq vs. Uniform Discrete Dist\n')
    plt.tight_layout(rect=[0, 0.0, 1, 0.9])
    fig.savefig(os.path.join(\
            fin_plot_dir,fin_name+'_random_shuffle_percentiles'),
            dpi=300)
    plt.close(fig)
    #plt.show()

    #========================================
    # Plot scatter plots to show correlation of actual data and a shuffle
    # Find MAX corr
    #corr_mat_inds = np.where(corr_array == np.max(corr_array,axis=None))
    lowest = inter_region_frame.sort_values('corr').iloc[0]
    nrn_inds,taste_ind,corr_val, p_val = lowest[['pair','taste', 'corr','p_vals']] 
    #nrn_inds = pair_inds[corr_mat_inds[0][0]]
    #this_pair = np.array([diff_sum_spikes[0][nrn_inds[0],...,corr_mat_inds[1][0]], 
    #                        diff_sum_spikes[1][nrn_inds[1],...,corr_mat_inds[1][0]]])
    this_pair = np.array([diff_sum_spikes[0][taste_ind,nrn_inds[0]], 
                            diff_sum_spikes[1][taste_ind,nrn_inds[1]]])

    fig, ax = plt.subplots(2,2)
    fig.suptitle('Firing Rate Scatterplots')
    ax[0,0].set_title('Pair : {}, Taste : {}\nCorr : {:.3f}, p_val : {:.3f}'.\
            format(nrn_inds,taste_ind,corr_val,p_val))
    ax[1,0].set_title('Shuffle') 
    ax[0,0].scatter(this_pair[0],this_pair[1])
    #ax[1,0].scatter(shuffled_pair[0],shuffled_pair[1]);
    ax[1,0].plot(this_pair[0]);
    ax[1,0].plot(this_pair[1]);

    # Find MIN corr
    #corr_mat_inds = np.where(corr_array == np.max(corr_array,axis=None))
    highest = inter_region_frame.sort_values('corr',ascending=False).iloc[0]
    nrn_inds,taste_ind,corr_val, p_val = highest[['pair','taste', 'corr','p_vals']] 
    #nrn_inds = pair_inds[corr_mat_inds[0][0]]
    this_pair = np.array([diff_sum_spikes[0][taste_ind,nrn_inds[0]], 
                            diff_sum_spikes[1][taste_ind,nrn_inds[1]]])

    ax[0,1].set_title('Pair : {}, Taste : {}\nCorr : {:.3f}, p_val : {:.3f}'.\
            format(nrn_inds,taste_ind,corr_val,p_val))
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

    #========================================
    # Matrix of significant correlations
    #========================================
    name_list,grouped_frames = list(zip(*list(all_frame.groupby('label'))))
    grouped_frames = [x.dropna() for x in grouped_frames]
    max_inds = [x['pair_ind'].max() for x in grouped_frames]
    sig_frames = [x[x['p_vals'] <= alpha][['pair_ind','taste']] for x in grouped_frames]
    sig_mat_list = [np.zeros((x+1,4)) for x in max_inds]
    for this_mat,this_inds in zip(sig_mat_list,sig_frames):
        inds_array = np.array(this_inds)
        this_mat[inds_array[:,0],inds_array[:,1]] = 1


    fig,ax = plt.subplots(1,len(sig_mat_list),figsize=(15,10))
    for this_ax, this_mat,this_name in zip(ax,sig_mat_list,name_list):
        this_ax.imshow(this_mat,origin='lower',aspect='auto')
        this_ax.set_xlabel('Taste')
        this_ax.set_ylabel('All Neuron Pair Combinations')
        this_ax.set_title(this_name.upper() + '\n{:.2f} % net significant corrs'\
                            .format(np.mean(this_mat,axis=None) * 100))
    #ax[0].set_xlabel('Taste')
    #ax[0].set_ylabel('All Neuron Pair Combinations')
    plt.suptitle('Noise Correlation Significance')
                            #.format(net_mean_sig_frac * 100))
    #plt.tight_layout(rect=[0, 0.0, 1, 0.9])
    #plt.show()
    fig.savefig(os.path.join(fin_plot_dir,fin_name+'_sig_array'),dpi=300)

    #========================================
    # For significant correlations, plot summed spikes in chornological order
    # to see whether there is a clear trend

    #sig_nrns = inds[np.where(sig_array)[0]]
    #sig_tastes = np.where(sig_array)[1]
    #sig_comparisons = np.concatenate([sig_nrns,sig_tastes[:,np.newaxis]],axis=-1)

    sig_frames = [x[x['p_vals'] <= alpha] for x in grouped_frames]
    # How many pairs in one plot
    # This will double because of line and corr plots
    plot_thresh = 8

    for this_name,this_frame in zip(name_list, sig_frames):
        if this_name == 'inter_region':
            this_plot_dir = os.path.join(fin_plot_dir, 'inter_region')
            dat0,dat1 = diff_sum_spikes
        else:
            this_plot_dir = os.path.join(fin_plot_dir, 'intra_region')
            this_ind = [num for num,name in enumerate(sorted_region_names) \
                            if name==this_name][0]
            dat0,dat1 = [diff_sum_spikes[this_ind]]*2

        if not os.path.exists(this_plot_dir):
            os.makedirs(this_plot_dir)

        num_figs = int(np.ceil(this_frame.shape[0]/plot_thresh))
        pairs = this_frame['pair']
        tastes = this_frame['taste']

        for this_fig_num in np.arange(num_figs):
            fig,ax = visualize.gen_square_subplots(int(plot_thresh*2))
            ax_inds = np.array(list(np.ndindex(ax.shape)))
            cut_comps = pairs[plot_thresh*this_fig_num : plot_thresh*(this_fig_num+1)]
            cut_tastes = tastes[plot_thresh*this_fig_num : plot_thresh*(this_fig_num+1)]

            # Reshape axes to pass into loop via zip
            reshaped_axes = np.reshape(ax,(-1,2))
            for num, (this_ax, this_comp,this_taste) in \
                    enumerate(zip(reshaped_axes,cut_comps,cut_tastes)):
                region0 = dat0[this_taste,this_comp[0]]
                region1 = dat1[this_taste,this_comp[1]]
                #line_plot_ind = tuple(np.split(ax_inds[2*num-2],2))
                #scatter_plot_ind = tuple(np.split(ax_inds[2*num-1],2))
                this_ax[0].plot(region0)
                this_ax[0].plot(region1)
                this_ax[1].scatter(region0,region1,s=5)
            plt.suptitle('Net significant comparisons')
            fig.savefig(os.path.join(\
                    this_plot_dir,fin_name+'_{}_{}'.format(this_name,this_fig_num)),
                    dpi=300)
            plt.close(fig)
        #plt.show()

    #========================================
    # Plot sum_spikes before and after detrending just to confirm
    this_plot_dir = os.path.join(fin_plot_dir,'detrend_plots')
    if not os.path.exists(this_plot_dir):
        os.makedirs(this_plot_dir)
    sum_spikes = [np.moveaxis(x,-1,0) for x in sum_spikes]
    flat_sum_spikes = np.concatenate(sum_spikes,axis=1)
    flat_sum_spikes = flat_sum_spikes.reshape(-1,flat_sum_spikes.shape[-1])
    flat_sum_spikes = stats.zscore(flat_sum_spikes,axis=-1)

    flat_diff_sum_spikes = np.concatenate(diff_sum_spikes,axis=1)
    flat_diff_sum_spikes = flat_diff_sum_spikes\
            .reshape(-1,flat_diff_sum_spikes.shape[-1])

    plot_thresh = 16

    num_figs = int(np.ceil(flat_sum_spikes.shape[0]/plot_thresh))

    for this_fig_num in np.arange(num_figs):
        fig,ax = visualize.gen_square_subplots(int(plot_thresh))
        ax_inds = np.array(list(np.ndindex(ax.shape)))
        #fig,ax = plt.subplots(len(sig_comparisons),2)
        dat_ind_range = np.arange(this_fig_num*plot_thresh,(this_fig_num+1)*plot_thresh)
        dat_ind_range = np.array([x for x in dat_ind_range if x < flat_sum_spikes.shape[0]])
        for this_ax_ind, dat_ind in enumerate(dat_ind_range):
            plot_ind = tuple(np.split(ax_inds[this_ax_ind],2))
            ax[plot_ind][0].plot(flat_sum_spikes[dat_ind])
            ax[plot_ind][0].plot(flat_diff_sum_spikes[dat_ind],alpha = 0.6)
        plt.suptitle('Net significant comparisons')
        fig.savefig(os.path.join(\
                this_plot_dir,fin_name+'_detrend_comps_{}'.format(this_fig_num)),
                dpi=300)
        plt.close(fig)
    #plt.show()

