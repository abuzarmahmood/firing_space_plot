"""
On single trials, does strength of granger-causal interactions
predict accuracy of prediction?

Parts:
    1- Causality:
        a- Directions : Forward, Backward
        b- Epochs : Middle, Late
        c- Frequencies : All, Low, High, Individual Bands
    2- Prediction:
        a- Epochs : Middle, Late
        b- Metrics : Entropy, Binary Prediction, Probability

Causality should only predict prediction accuracy in the same or later epoch,
i.e.    causality [middle] -> prediction [middle, late]
        causality [middle, late] -> prediction [late]

Might be best to do 2 runs of the analysis broken up by causality epoch,
and then break each run by prediction metric.

First pass:
    Simply look at changes in causality given high or low prediction metric.
"""

import os
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
from itertools import product
import tables
import pylab as plt
#discrim_path = '/media/bigdata/firing_space_plot/firing_analyses/single_trial_discrimination'
#sys.path.append(discrim_path)
#from single_trial_discrim_test import template_classifier
import pingouin as pg
import seaborn as sns

ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
sys.path.append(ephys_data_dir)
from ephys_data import ephys_data
granger_path = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/discrim_vs_interaction'
sys.path.append(granger_path)
from discrimination_process import discrim_handler

def test_machine(df, between, dv, test_var):
    """
    Given between, iterate over groups to perform one-way anovas
    """
    grouped_dat = list(df.groupby(between))
    out_list = []
    for name, group in grouped_dat:
        #print(f'Group : {name}')
        #subgroups = list(group.groupby(test_var))
        #subgroups = [x[1][dv] for x in subgroups]
        #pval = pg.mwu(subgroups[0], subgroups[1])['p-val'].values[0]
        try:
            pval = pg.anova(
                    data = group,
                    dv = dv,
                    between = [test_var],
                    detailed = True)['p-unc'].values[0]
        except:
            pval = np.nan
        out_dict = dict(zip(between, name)) 
        out_dict['pval'] = pval
        out_list.append(out_dict)
    return pd.DataFrame(out_list) 


epoch_lims = [[300,800],[800,1300]]
epoch_names = ['middle','late']

frequency_lims = \
        [[0,100],[0,30],[30,100],[4,8],[8,12],[12,30],[30,60],[60,100]]
frequency_names = \
        ['all','low','high','theta','alpha','beta','low_gamma','high_gamma']

############################################################
# Load in data
############################################################

dir_list_path = \
        '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
with open(dir_list_path, 'r') as f:
    dir_list = f.read().splitlines()

epoch_inds = np.arange(len(epoch_lims))
dir_inds = np.arange(len(dir_list))

iter_inds = list(product(epoch_inds, dir_inds))

taste_anova_list = []
entropy_anova_list = []
prediction_anova_list = []
############################################################

#this_iter = iter_inds[0]

for this_iter in tqdm(iter_inds):

    epoch_ind = this_iter[0]
    dir_ind = this_iter[1]

    this_epoch = epoch_lims[epoch_ind]
    dir_name = dir_list[dir_ind]

    dat = ephys_data(dir_name)

    print(f'Processing {dir_name}')
    print(f'Epoch : {this_epoch}')

    this_epoch_name = epoch_names[epoch_ind]
    basename = dir_name.split('/')[-1]

    ############################################################
    # Discrimination processing
    ############################################################

    this_discrim_handler = discrim_handler(
            dir_name, this_epoch)

    epoch_flat_rates = this_discrim_handler.get_epoch_flat_rates()

    pred, pred_proba, pred_entropy = \
            this_discrim_handler.return_pred_proba_and_entropy()
    y = this_discrim_handler.y
    taste_names = this_discrim_handler.taste_names

    prediction_bool = pred == y

    ############################################################
    # Granger processing
    ############################################################
    save_path = '/ancillary_analysis/granger_causality/single_trial'
    df = pd.read_hdf(dat.hdf5_path, save_path)

    df = df[df.epoch_name == this_epoch_name]

    wanted_trials = df.trial.unique()

    # Add discrimination data to granger dataframe
    df['pred'] = pred[df.trial]
    #df['pred_proba'] = pred_proba[df.trial]
    df['pred_entropy'] = pred_entropy[df.trial]
    df['y'] = y[df.trial]
    df['prediction_bool'] = prediction_bool[df.trial]

    # Chop entropy into high and low
    df['entropy_bin'] = pd.cut(df.pred_entropy,2,labels=['low','high'])

    # Also chop entropy into quartiles
    df['entropy_quartile'] = pd.cut(df.pred_entropy,4,labels=['q1','q2','q3','q4'])

    # Some issue with generating causality, fix direction labels
    direction_strs = df.direction.unique()
    split_strs = [x.split('--') for x in direction_strs]
    import re
    # Replace '<' or '>' with '-'
    split_strs = [[re.sub('<|>','-',x) for x in y] for y in split_strs]
    # Make region with '-' first region
    split_strs = [[y[1],y[0]] if '-' in y[1] else y for y in split_strs]
    # Drop '-' and join with '<--'
    split_strs = [[re.sub('-','',x) for x in y] for y in split_strs]
    split_strs = ['<--'.join(x) for x in split_strs]

    direction_map = dict(zip(direction_strs, split_strs))
    df['direction'] = df.direction.map(direction_map)

    ############################################################
    # Analyses 
    ############################################################

    # 1) Difference between tastants in causality
    #anova_out = pg.anova(
    #        data = df,
    #        dv = 'f_stat',
    #        between = ['taste','direction', 'frequency'],
    #        )
    taste_out = test_machine(df, ['direction','frequency'], 'f_stat', 'taste')
    taste_out['epoch'] = str(this_epoch)
    taste_out['basename'] = basename
    taste_anova_list.append(taste_out)

    # 2) Difference in causality between high and low entropy, per taste and direction
    entropy_out = test_machine(df = df, 
                               between = ['direction','frequency'], 
                               dv = 'f_stat', 
                               test_var = 'entropy_quartile')
    entropy_out['epoch'] = str(this_epoch)
    entropy_out['basename'] = basename
    entropy_anova_list.append(entropy_out)

    # 3) Difference in causality between correct and incorrect predictions, per taste and direction
    prediction_out = test_machine(df = df, 
                               between = ['direction','frequency'], 
                               dv = 'f_stat', 
                               test_var = 'prediction_bool')
    prediction_out['epoch'] = str(this_epoch)
    prediction_out['basename'] = basename
    prediction_anova_list.append(prediction_out)

############################################################
# Plotting 
############################################################
# 1) Difference between tastants in causality
taste_anova_frame = pd.concat(taste_anova_list)
alpha = 0.05
taste_anova_frame['sig'] = taste_anova_frame.pval < alpha

taste_anova_frac = taste_anova_frame.groupby(['direction','frequency','epoch']).sig.mean()
taste_anova_frac = taste_anova_frac.reset_index()
taste_anova_frac['comparison'] = 'taste'

sns.catplot(data = taste_anova_frac,
        x = 'frequency',
        y = 'sig',
        hue = 'direction',
            row = 'epoch',
        kind = 'bar',
        )
fig = plt.gcf()
fig.suptitle('Fraction of significant taste difference anovas')
fig.subplots_adjust(top=0.9)
plt.show()

# 2) Difference in causality between high and low entropy, per direction
entropy_anova_frame = pd.concat(entropy_anova_list)
alpha = 0.05
entropy_anova_frame['sig'] = entropy_anova_frame.pval < alpha

entropy_anova_frac = entropy_anova_frame.groupby(['direction','frequency','epoch']).sig.mean()
entropy_anova_frac = entropy_anova_frac.reset_index()
entropy_anova_frac['comparison'] = 'entropy'

sns.catplot(data = entropy_anova_frac,
        x = 'frequency',
        y = 'sig',
        hue = 'direction',
            row = 'epoch',
        kind = 'bar',
        )
fig = plt.gcf()
fig.suptitle('Fraction of significant entropy difference anovas')
fig.subplots_adjust(top=0.9)
plt.show()

# 3) Difference in causality between correct and incorrect predictions, per direction
prediction_anova_frame = pd.concat(prediction_anova_list)
alpha = 0.05
prediction_anova_frame['sig'] = prediction_anova_frame.pval < alpha

prediction_anova_frac = prediction_anova_frame.groupby(['direction','frequency','epoch']).sig.mean()
prediction_anova_frac = prediction_anova_frac.reset_index()
prediction_anova_frac['comparison'] = 'prediction'

sns.catplot(data = prediction_anova_frac,
        x = 'frequency',
        y = 'sig',
        hue = 'direction',
            row = 'epoch',
        kind = 'bar',
        )
fig = plt.gcf()
fig.suptitle('Fraction of significant prediction difference anovas')
fig.subplots_adjust(top=0.9)
plt.show()

# Plot all three together
anova_frac = pd.concat([taste_anova_frac, entropy_anova_frac, prediction_anova_frac])

sns.catplot(data = anova_frac,
        x = 'frequency',
        y = 'sig',
        hue = 'direction',
            row = 'epoch',
        col = 'comparison',
        kind = 'bar',
        )
fig = plt.gcf()
fig.suptitle('Fraction of significant difference anovas')
fig.subplots_adjust(top=0.9)
plt.show()

