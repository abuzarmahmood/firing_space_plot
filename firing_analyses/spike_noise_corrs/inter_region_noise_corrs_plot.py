"""
Testing significance of corr R percentile distribution:
    1) Converting to percentiles is a TRANSFORMATION
        - We can't directly compare it to a shuffle distribution because:
            a) There is no single shuffle distribution
            b) If we want to do a shuffle test, we'll need a STATISTIC
    2) the STATISTIC can be : Summed distance from 50th percentile
    3) Since we already have a bunch of shuffles, we can calculate
        this statistic for each shuffle distribution to generate
        a null-distribution for the statistic...to which we'll then 
        compare the statistic for the actual data
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
import tables
import itertools as it
import ast
from scipy.stats import spearmanr, percentileofscore, chisquare
import pylab as plt
import seaborn as sns

sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize

##################################################
# _                    _   ____        _        
#| |    ___   __ _  __| | |  _ \  __ _| |_ __ _ 
#| |   / _ \ / _` |/ _` | | | | |/ _` | __/ _` |
#| |__| (_) | (_| | (_| | | |_| | (_| | || (_| |
#|_____\___/ \__,_|\__,_| |____/ \__,_|\__\__,_|
##################################################

data_dir = sys.argv[1]
#data_dir = '/media/bigdata/Abuzar_Data/AM35/AM35_4Tastes_201231_105700'
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

# Load frames
save_path = '/ancillary_analysis/spike_noise_corrs'
frame_name_list = ['inter_region_frame',
                        'shuffle_inter_region_frame',
                        'intra_region_frame',
                        'bin_inter_region_frame',
                        'baseline_inter_region_frame',
                        'baseline_intra_region_frame']

frame_list = [pd.read_hdf(dat.hdf5_path,
                    os.path.join(save_path,frame_name)) \
                            for frame_name in frame_name_list]
for this_dat,this_name in zip(frame_list,frame_name_list):
    globals()[this_name] = this_dat

#for frame_name in frame_name_list:
#    # Save transformed array to HDF5
#    pd.read_hdf(save_path,frame_name)
#    eval(frame_name).to_hdf(dat.hdf5_name,  
#            os.path.join(save_path, frame_name))
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

fig,ax = plt.subplots(1,3, figsize = (15,5))
imshow_kwargs = {'aspect':'equal','cmap':'viridis','vmin':0,'vmax':4}
im = ax[0].imshow(inter_sig_hist_array,**imshow_kwargs);
ax[1].imshow(intra_sig_hist_arrays[0].T,**imshow_kwargs);
ax[2].imshow(intra_sig_hist_arrays[1].T,**imshow_kwargs);
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
# Histograms of significant number of interactions per neuron 
#========================================
fig,ax = plt.subplots(2,1)
for num, region_name in enumerate(sorted_region_names):
    summed_interactions = np.mean(inter_sig_hist_array,axis=num)
    #bin_num = np.arange(int(np.max(summed_interactions))+2)
    #bin_num = np.linspace(0,np.max(summed_interactions),
    #                            len(np.unique(summed_interactions))+1)
    #ax[num].hist(summed_interactions-0.5, bins = bin_num-0.5)
    ax[num].hist(summed_interactions, bins = np.linspace(0,1,11))
    #ax[num].set_xticks(bin_num[:-1])
    #ax[num].set_xticklabels(bin_num[:-1])
    ax[num].set_title(region_name)
    ax[num].set_xlabel('Total number of significant interactions')
    ax[num].set_ylabel('Frequency')
plt.tight_layout()
fig.savefig(os.path.join(fin_plot_dir,
    fin_name+'_significant_interactions_per_unit'),dpi=300)
plt.close(fig)
#plt.show()


#========================================
# Comparison of strength of actual data correlations to shuffle 
#========================================

vector_percentile = lambda v,e : np.array(\
        list(map(lambda e: percentileofscore(v, e), v)))

grouped_shuffle_inter_region_frame = list(shuffle_inter_region_frame.groupby(\
                                        ['pair_ind','taste']))

for num,this_frame in grouped_shuffle_inter_region_frame:
    corrs = this_frame['corr']
    this_frame['percentiles'] = vector_percentile(corrs,corrs)
    this_frame['repeat_num'] = np.arange(this_frame.shape[0])

shuffle_inter_region_frame = \
        pd.concat([x[1] for x in grouped_shuffle_inter_region_frame])

temp_xvals = shuffle_inter_region_frame\
        [['pair_ind','taste','repeat_num','percentiles']]\
        .to_xarray().to_array().values

# Convert to array
# ** This might not be the best way to generate shuffles since
# ** the trial shuffle order for each neuron pair will be different
# ** But going ahead for now
vector_int = np.vectorize(np.int)

shuffle_percentile_array = np.zeros(vector_int(np.max(temp_xvals[:3],axis=1)+1))

shuffle_percentile_array[[vector_int(temp_xvals[0]),
                            vector_int(temp_xvals[1]),
                            vector_int(temp_xvals[2])]] = temp_xvals[-1]

# Flatten across first 2 dims to have shape be : dataset x repeats
shuffle_percentile_array = np.reshape(shuffle_percentile_array,
                                (-1,shuffle_percentile_array.shape[-1]))

# Calculate test statistic
shuffle_dist_vals = np.sum(np.abs(shuffle_percentile_array-50),axis=0)
actual_dist_val = np.sum(np.abs(inter_region_frame['percentiles'] - 50))
actual_data_percentile = percentileofscore(shuffle_dist_vals, actual_dist_val)

plt.hist(shuffle_dist_vals,label = 'Shuffle values')
plt.axvline(actual_dist_val, linewidth = 2, color = 'red', label = 'Actual Data')
plt.legend()
plt.title('Symmetric Distance Metric \n' +\
        f'Statistic percentile : {actual_data_percentile:.2f}' + '\n'+\
        f'2-tailed p-value : {(1-(actual_data_percentile/100))*2:.2E}')
plt.gcf().savefig(os.path.join(\
        fin_plot_dir,fin_name+'_distance_metric_percentile'))
plt.close()
#plt.show()


#========================================
# histogram of corr percentile relative to respective shuffle
#========================================

all_frame = pd.concat([inter_region_frame,intra_region_frame])
percentile_list = [[x[0],x[1]['percentiles']] \
        for x in list(all_frame.groupby('label'))]
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
# Histogramo of RAW correlations and shuffled correlations 
#========================================
# For shock value, plot WORST shuffle
shuffle_ind = np.argmin(shuffle_dist_vals)
this_shuffle_frame = \
        shuffle_inter_region_frame.loc\
        [shuffle_inter_region_frame['repeat_num'] == shuffle_ind]
inter_dat = np.abs(inter_region_frame['corr'])
intra_dat = np.abs(this_shuffle_frame['corr'])
cmap = plt.get_cmap("Set1")

fig,ax = plt.subplots()
hist_aes = {'alpha' : 0.7}
hist_props = {'bins':20, 'density' : False}
n,bins,patches = \
    ax.hist(inter_dat, **hist_props, **hist_aes, color = cmap(0),label='Actual')
ax.hist(inter_dat, **hist_props, histtype='step', color = cmap(0))
ax.hist(intra_dat, bins = bins, 
         **hist_aes, color = cmap(5),label='Shuffle')
ax.hist(intra_dat, bins = bins, 
         histtype='step',color = cmap(5))
ax.set_xlabel('Absolute Correlation |R|')
ax.set_ylabel('Frequency')
#ax.set_xlim([-1,1])
plt.legend()
#ax.set_yscale('log')
fig.savefig(os.path.join(\
        fin_plot_dir,fin_name+'_r_comparison'),
        dpi=300)
plt.close(fig)
#plt.show()

#========================================
# Bar plot of significant interactions in actual data vs shuffle 
#========================================
actual_sig_count = np.mean(inter_region_frame['p_vals'] < alpha)
#shuffle_sig_count = np.sum(this_shuffle_frame['p_vals'] < alpha)

shuffle_inter_region_frame['sig_val'] = shuffle_inter_region_frame['p_vals'] < alpha
shuffle_sig_counts = shuffle_inter_region_frame.\
        groupby('repeat_num').mean()['sig_val']

#plt.bar(['Shuffle','Actual'],[np.mean(shuffle_sig_counts),actual_sig_count])
#plt.boxplot([shuffle_sig_counts,actual_sig_count])
plt.hist(shuffle_sig_counts,bins=50, color = cmap(0),alpha=0.7, label = 'Shuffle')
plt.hist(shuffle_sig_counts,bins=50, color = cmap(0), histtype = 'step')
plt.axvline(actual_sig_count, linewidth = 5, color = 'black', 
                    alpha = 0.7, linestyle = '--', label = 'Actual')
plt.xlabel('Fraction of significant correlations')
plt.ylabel('Frequency')
#plt.legend()
plt.gcf().savefig(os.path.join(\
        fin_plot_dir,fin_name+'actual_sig_count_vs_shuffle_hist'),
        dpi=300)
plt.close()
#plt.show()

inter_region_frame['repeat_num'] = 0
full_frame = pd.concat([inter_region_frame, shuffle_inter_region_frame])
full_frame['sig_val'] = full_frame['p_vals'] < alpha
sig_count_frame = full_frame.groupby(['label','repeat_num']).mean()
sig_count_frame.reset_index(inplace=True)

sns.barplot(x='label',y='sig_val',data=sig_count_frame, ci='sd')
#sns.violinplot(x='label',y='sig_val',data=sig_count_frame, linewidth=2)
#sns.swarmplot(x='label',y='sig_val',data=sig_count_frame)
plt.show()

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
max_trial_num = 7
ax[1,0].plot(this_pair[0][:max_trial_num],'-o',linewidth = 2);
ax[1,0].plot(this_pair[1][:max_trial_num],'-o',linewidth = 2);

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
ax[1,1].plot(this_pair[0][:max_trial_num],'-o',linewidth = 2);
ax[1,1].plot(this_pair[1][:max_trial_num],'-o',linewidth = 2);

for this_ax in ax.flatten():
    this_ax.set_xlabel('Nrn 1 Firing')
    this_ax.set_ylabel('Nrn 0 Firing')
plt.tight_layout()
plt.show()

fig.savefig(os.path.join(fin_plot_dir,fin_name+'_example_corrs'),dpi=300)

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
#========================================
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
    dat_ind_range \
            = np.array([x for x in dat_ind_range if x < flat_sum_spikes.shape[0]])
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

#========================================
# BASELINE PLOTS 
#========================================
baseline_plot_dir = os.path.join(fin_plot_dir,'baseline_plots')
if not os.path.exists(baseline_plot_dir):
    os.makedirs(baseline_plot_dir)

time_lims = [0,2000]

base_temp_spikes = spikes[...,time_lims[0]:time_lims[1]]
base_region_spikes = [base_temp_spikes.swapaxes(0,2)[region_inds]\
        for region_inds in dat.region_units]
base_temp_region_spikes = [base_region_spikes[x] for x in wanted_order]
base_sum_spikes = [np.sum(x,axis=-1) for x in base_temp_region_spikes]
# Try detrending with 1st order difference before corr
base_diff_sum_spikes = [np.diff(region,axis=1) for region in base_sum_spikes]
# Zscore along trial axis to normalize values across neurons
base_diff_sum_spikes = [stats.zscore(region,axis=1) for region in base_diff_sum_spikes]
base_diff_sum_spikes = [np.moveaxis(x,-1,0) for x in base_diff_sum_spikes]
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

baseline_inter_sig_hist_array,_ = gen_sig_mat(\
                    baseline_inter_region_frame.dropna(),'inter_region')
baseline_intra_sig_hist_arrays = [gen_sig_mat(\
                                    baseline_intra_region_frame.dropna(), 
                                    region_name)[0]\
                        for region_name in sorted_region_names]

fig,ax = plt.subplots(1,3, figsize = (15,5))
im = ax[0].imshow(baseline_inter_sig_hist_array,
                aspect='equal',cmap='viridis',vmin = 0,vmax = 4);
ax[1].imshow(baseline_intra_sig_hist_arrays[0].T,
                aspect='equal',cmap='viridis',vmin = 0,vmax = 4);
ax[2].imshow(baseline_intra_sig_hist_arrays[1].T,
                aspect='equal',cmap='viridis',vmin = 0,vmax = 4);
# ROWS are region0, COLS are region1
ax[0].set_xlabel(str(sorted_region_names[1])+' Neuron #');
ax[0].set_ylabel(str(sorted_region_names[0])+' Neuron #')
# ROWS are region0, COLS are region1
ax[0].set_xlabel(str(sorted_region_names[1])+' Neuron #');
ax[0].set_ylabel(str(sorted_region_names[0])+' Neuron #')
ax[0].set_title('Inter-region')
ax[1].set_title(sorted_region_names[0])
ax[2].set_title(sorted_region_names[1])
plt.suptitle('BASELINE\nCount of Significant\n'\
        'Noise Correlations across all comparisons')
plt.colorbar(im)
fig.savefig(os.path.join(baseline_plot_dir,fin_name+'_base_sig_nrn_table'),dpi=300)
plt.close(fig)
#plt.show()


#========================================
# histogram of corr percentile relative to respective shuffle
#========================================
all_frame = pd.concat([baseline_inter_region_frame,baseline_intra_region_frame])
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
plt.suptitle('BASELINE\n'\
            'Percentile relative to respective shuffles\n' +\
            'Chi_sq vs. Uniform Discrete Dist\n')
plt.tight_layout(rect=[0, 0.0, 1, 0.9])
fig.savefig(os.path.join(\
        baseline_plot_dir,fin_name+'_base_random_shuffle_percentiles'),
        dpi=300)
plt.close(fig)
#plt.show()

#========================================
# Plot scatter plots to show correlation of actual data and a shuffle
# Find MAX corr
#corr_mat_inds = np.where(corr_array == np.max(corr_array,axis=None))
lowest = baseline_inter_region_frame.sort_values('corr').iloc[0]
nrn_inds,taste_ind,corr_val, p_val = lowest[['pair','taste', 'corr','p_vals']] 
#nrn_inds = pair_inds[corr_mat_inds[0][0]]
#this_pair = np.array([diff_sum_spikes[0][nrn_inds[0],...,corr_mat_inds[1][0]], 
#                        diff_sum_spikes[1][nrn_inds[1],...,corr_mat_inds[1][0]]])
this_pair = np.array([base_diff_sum_spikes[0][taste_ind,nrn_inds[0]], 
                        base_diff_sum_spikes[1][taste_ind,nrn_inds[1]]])

fig, ax = plt.subplots(2,2)
fig.suptitle('BASELINE Firing Rate Scatterplots')
ax[0,0].set_title('Pair : {}, Taste : {}\nCorr : {:.3f}, p_val : {:.3f}'.\
        format(nrn_inds,taste_ind,corr_val,p_val))
ax[1,0].set_title('Shuffle') 
ax[0,0].scatter(this_pair[0],this_pair[1])
#ax[1,0].scatter(shuffled_pair[0],shuffled_pair[1]);
ax[1,0].plot(this_pair[0]);
ax[1,0].plot(this_pair[1]);

# Find MIN corr
#corr_mat_inds = np.where(corr_array == np.max(corr_array,axis=None))
highest = baseline_inter_region_frame.sort_values('corr',ascending=False).iloc[0]
nrn_inds,taste_ind,corr_val, p_val = highest[['pair','taste', 'corr','p_vals']] 
#nrn_inds = pair_inds[corr_mat_inds[0][0]]
this_pair = np.array([base_diff_sum_spikes[0][taste_ind,nrn_inds[0]], 
                        base_diff_sum_spikes[1][taste_ind,nrn_inds[1]]])

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

fig.savefig(os.path.join(baseline_plot_dir,fin_name+'_base_example_corrs'),dpi=300)
plt.close(fig)
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
fig.savefig(os.path.join(baseline_plot_dir,fin_name+'_base_sig_array'),dpi=300)
plt.close(fig)

#========================================
# BINNED PLOTS 
#========================================
#========================================
# Same plot as above but histogram with sides that are nrns
# and bins counting how many significant correlations

# Create function to pull out significant pairs
def gen_sig_mat(pd_frame, index_label):
    label_cond = pd_frame['label'] == index_label
    sig_cond = pd_frame['p_vals'] <= alpha
    sig_frame = pd_frame[label_cond & sig_cond]
    bin_nums, frame_list = list(zip(*list(sig_frame.groupby('bin_num'))))
    sig_pair_list = [x['pair'] for x in frame_list]

    sig_hist_array = np.zeros((4,*[x+1 for x in pd_frame[label_cond]['pair'].max()]))
    for num, (this_sig_pair_list, this_frame) in \
            enumerate(zip(sig_pair_list,frame_list)):
        for this_pair in this_sig_pair_list:
            sig_hist_array[(num,*this_pair)] += 1
    return sig_hist_array, frame_list

bin_inter_sig_hist_array,_ = \
        gen_sig_mat(bin_inter_region_frame.dropna(),'inter_region')
#mat_inds = np.array(list(np.ndindex(p_val_array.shape)))
#inds = np.array(pair_inds)
#for this_mat_ind,this_val in zip(mat_inds[:,0],(p_val_array<alpha).flatten()):
#    sig_hist_array[inds[this_mat_ind,0], inds[this_mat_ind,1]] += \
#                                    this_val

bin_width = 500
bin_count = np.diff(time_lims)[0]//bin_width 
bin_starts = bin_width*np.arange(bin_count)
bin_lims = list(zip(*[bin_starts,bin_starts+bin_width]))
fig,ax = visualize.gen_square_subplots(len(bin_inter_sig_hist_array)) 
for num, (this_ax,this_dat) in enumerate(zip(ax.flatten(),bin_inter_sig_hist_array)):
    im=this_ax.imshow(this_dat,aspect='auto',cmap='viridis',vmin = 0,vmax = 4);
    # ROWS are region0, COLS are region1
    this_ax.set_xlabel(str(sorted_region_names[1])+' Neuron #');
    this_ax.set_ylabel(str(sorted_region_names[0])+' Neuron #')
    this_ax.set_title(bin_lims[num])
plt.suptitle('Count of Significant\nNoise Correlations across all comparisons')
cbaxes = fig.add_axes([0.9, 0.3, 0.03, 0.3])
cb = plt.colorbar(im, cax = cbaxes)
plt.tight_layout(rect=[0, 0.0, 0.9, 0.9])
fig.savefig(os.path.join(fin_plot_dir,fin_name+'_binned_sig_nrn_table'),dpi=300)
plt.close(fig)
#plt.show()

#========================================
# histogram of corr percentile relative to respective shuffle
#========================================
percentile_list = [[x[0],x[1]['percentiles']] for x in \
                list(bin_inter_region_frame.groupby('bin_num'))]
label_list, percentile_list = list(zip(*percentile_list))
percentile_list = [x.dropna() for x in percentile_list]

fig,ax = visualize.gen_square_subplots(len(label_list)) 
for this_ax,this_name, this_percentile in zip(ax.flatten(),bin_lims,percentile_list):
    percentile_array = np.array(this_percentile).flatten()
    # Use default binning (which tends to be more conservative)
    counts, bins, patches = this_ax.hist(percentile_array.flatten(),bins='auto')
    chi_test = chisquare(counts)
    this_ax.set_title(str(this_name) + ': p_val :' \
            + str(np.format_float_scientific(chi_test[1],3)))
    this_ax.set_xlabel('Percentile Relative to shuffle ')
    this_ax.set_ylabel('Frequency')
plt.suptitle('Percentile relative to respective shuffles\n' +\
            'Chi_sq vs. Uniform Discrete Dist\n')
plt.tight_layout(rect=[0, 0.0, 1, 0.9])
fig.savefig(os.path.join(\
        fin_plot_dir,fin_name+'_binned_random_shuffle_percentiles'),
        dpi=300)
plt.close(fig)
#plt.show()

#========================================
# Matrix of significant correlations
#========================================
name_list,grouped_frames = list(zip(*list(bin_inter_region_frame.groupby('bin_num'))))
grouped_frames = [x.dropna() for x in grouped_frames]
max_inds = [x['pair_ind'].max() for x in grouped_frames]
sig_frames = [x[x['p_vals'] <= alpha][['pair_ind','taste']] for x in grouped_frames]
sig_mat_list = [np.zeros((x+1,4)) for x in max_inds]
for this_mat,this_inds in zip(sig_mat_list,sig_frames):
    inds_array = np.array(this_inds)
    this_mat[inds_array[:,0],inds_array[:,1]] = 1


fig,ax = plt.subplots(1,len(sig_mat_list),figsize=(15,10))
for this_ax, this_mat,this_name in zip(ax,sig_mat_list,bin_lims):
    this_ax.imshow(this_mat,origin='lower',aspect='auto')
    this_ax.set_xlabel('Taste')
    this_ax.set_title(str(this_name) + '\n{:.2f} % net significant corrs'\
                        .format(np.mean(this_mat,axis=None) * 100))
ax[0].set_ylabel('All Neuron Pair Combinations')
plt.suptitle('Noise Correlation Significance')
                        #.format(net_mean_sig_frac * 100))
#plt.tight_layout(rect=[0, 0.0, 1, 0.9])
#plt.show()
fig.savefig(os.path.join(fin_plot_dir,fin_name+'_binned_sig_array'),dpi=300)
plt.close(fig)
