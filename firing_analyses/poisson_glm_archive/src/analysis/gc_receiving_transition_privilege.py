"""
Are GC neurons receiving input from BLA privileged in 
transition timing or coherence?
"""

import pandas as pd
if 'pytau_env' not in pd.__file__:
    env_name = 'base'
else:
    env_name = 'pytau_env'

import numpy as np

artifact_dir = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/artifacts'

##############################
# Load transition snippets from changepoint models
# Needs to be done in pytau_env

## Import modules
import sys
import pylab as plt
from tqdm import tqdm
import os
import pandas as pd
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_1samp, wilcoxon, percentileofscore
import pingouin as pg
blech_clust_path = '/home/abuzarmahmood/Desktop/blech_clust/utils/'
sys.path.append(blech_clust_path)
from ephys_data import visualize as vz
cp_path = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/src/analysis'
sys.path.append(cp_path)
from single_poisson_changepoint import *

if env_name == 'pytau_env':
    base_dir = '/media/bigdata/projects/pytau'
    sys.path.append(base_dir)
    from pytau.changepoint_io import FitHandler
    from pytau.utils import plotting
    from pytau.utils import ephys_data
    from pytau.changepoint_io import DatabaseHandler
    from pytau.changepoint_analysis import PklHandler

    fit_database = DatabaseHandler()
    fit_database.drop_duplicates()
    fit_database.clear_mismatched_paths()

    dframe = fit_database.fit_database
    wanted_exp_name = 'pretty_gc_trans'
    wanted_n_states = 4
    exp_bool = dframe['exp.exp_name'] == wanted_exp_name
    states_bool = dframe['model.states'] == wanted_n_states
    basename_bool = dframe['data.basename'].isin(basenames)
    final_bool = exp_bool & states_bool & basename_bool
    wanted_frame = dframe.loc[final_bool]

    assert wanted_frame['data.region_name'].unique() == 'gc'
else:
    import pymc as pm
    import arviz

##############################
file_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
file_list = [x.strip() for x in open(file_list_path,'r').readlines()]
basenames = [os.path.basename(x) for x in file_list]

snip_pkl_path = os.path.join(artifact_dir, 'gc_transition_snips.pkl')

if not os.path.exists(snip_pkl_path) and env_name == 'pytau_env':
    basename_list = []
    taste_num_list = []
    transition_snips_list = []
    error_list = []
    for i, this_row in tqdm(wanted_frame.iterrows()):
        try:
            pkl_path = this_row['exp.save_path']
            basename = this_row['data.basename']
            taste_num = this_row['data.taste_num']

            # From saved pkl file
            this_handler = PklHandler(pkl_path)
            # this_handler.pretty_metadata

            # scaled_mode_tau = this_handler.tau.scaled_mode_tau
            # scaled_mode_tau_list.append(scaled_mode_tau)
            transition_snips = this_handler.firing.transition_snips
            transition_snips_list.append(transition_snips)
            basename_list.append(basename)
            taste_num_list.append(taste_num)
        except Exception as e:
            print(f"Error in {basename}")
            print(e)
            error_list.append(basename + '_' + str(taste_num))

    # Convert outputs to dataframe and save
    transition_snips_df = pd.DataFrame(
            {'basename' : basename_list,
             'taste_num' : taste_num_list,
             'transition_snips' : transition_snips_list})
    transition_snips_df.to_pickle(snip_pkl_path)
else:
    transition_snips_df = pd.read_pickle(snip_pkl_path)

##############################
# Load gc_neurons
gc_neurons = pd.read_csv(artifact_dir + '/gc_neurons_all.csv')

# Neuron inds are in order of sorting, but models contain only GC neurons
# so absolute indices need to be converted to relative indices
unit_region_frame = pd.read_csv(artifact_dir + '/unit_region_frame.csv',
                                index_col = 0)
unit_gc_frame = unit_region_frame[unit_region_frame['region'] == 'gc']
_, unit_gc_groups = zip(*list(unit_gc_frame.groupby('session')))
fin_unit_gc_list = []
for session in unit_gc_groups:
    session['rel_inds'] = np.arange(session.shape[0])
    fin_unit_gc_list.append(session)
fin_unit_gc_frame = pd.concat(fin_unit_gc_list)

# Add relative indices to gc_neurons
gc_neurons = gc_neurons.merge(
        fin_unit_gc_frame[['session','unit','rel_inds']],
        left_on = ['session','neuron'], 
        right_on = ['session','unit'])
gc_neurons.drop(columns = ['unit'], inplace=True)
gc_neurons.rename(columns = {'neuron' : 'abs_inds'}, inplace=True)

##############################
base_plot_dir = '/media/bigdata/firing_space_plot/firing_analyses/poisson_glm/plots/run_007'
plot_dir = os.path.join(base_plot_dir, 'gc_transition_privilege')
this_plot_dir = os.path.join(plot_dir, 'transition_rasters')

if not os.path.exists(this_plot_dir):
    os.makedirs(this_plot_dir)

all_cp_df_path = os.path.join(plot_dir, 'all_cp_df.pkl')
make_plots = False

if not os.path.exists(all_cp_df_path) or make_plots:

    all_basename_list = []
    all_taste_num_list = []
    all_nrn_list = []
    all_trans_num_list = []
    all_cp_list = []
    for i, this_row in tqdm(transition_snips_df.iterrows()):
        this_snips = this_row['transition_snips']
        this_basename = this_row['basename']
        this_taste_num = this_row['taste_num']
        this_snips = this_snips.swapaxes(0,1)

        for this_nrn, this_nrn_snips in enumerate(this_snips):
            # Get abs_ind of this neuron
            wanted_row = gc_neurons[
                    (gc_neurons['basename'] == this_basename) & 
                    (gc_neurons['rel_inds'] == this_nrn)]
            if wanted_row.shape[0] == 0:
                continue

            else:
                abs_ind = wanted_row['abs_inds'].values[0]

                if make_plots:
                    fig, ax = plt.subplots(2, this_nrn_snips.shape[-1],
                                           sharex = True, sharey = 'row',)
                for ax_ind in range(this_nrn_snips.shape[-1]):
                    this_snip = this_nrn_snips[:,:,ax_ind]
                    cp_list = [infer_cp(x) for x in this_snip]
                    all_basename_list.append(this_basename)
                    all_taste_num_list.append(this_taste_num)
                    all_nrn_list.append(abs_ind)
                    all_trans_num_list.append(ax_ind)
                    all_cp_list.append(cp_list)

                    if make_plots:
                        no_none_cp = [x for x in cp_list if x is not None]
                        no_none_trials = [x for x in range(len(cp_list)) if cp_list[x] is not None]
                        snip_sum = np.sum(this_snip, axis = 0)
                        bin_size = 25
                        snip_sum_bin = snip_sum.reshape(-1,bin_size).mean(axis = 1)
                        t_bin = np.arange(0,snip_sum.shape[0],bin_size)
                        ax[0, ax_ind].plot(t_bin, snip_sum_bin, color = 'k')
                        ax[0, ax_ind].set_title(f'Transition {ax_ind}')
                        ax2 = ax[0, ax_ind].twinx()
                        ax2.hist(no_none_cp, bins = 20, color = 'r', alpha = 0.5, zorder = -1)
                        ax[1, ax_ind] = vz.raster(ax[1, ax_ind], this_nrn_snips[:,:,ax_ind],
                                                  marker = '|', color = 'k')
                        ax[1, ax_ind].scatter(no_none_cp, no_none_trials,
                                           color = 'r', s = 20)
                        ax[1, ax_ind].set_xlabel('Time (ms)')
                if make_plots:
                    ax[1,0].set_ylabel('Trial')
                    plt.suptitle(f'{this_basename} Taste {this_taste_num} Neuron {abs_ind}')
                    plt.savefig(os.path.join(this_plot_dir, f'{this_basename}_nrn_{abs_ind}_taste_{this_taste_num}.png'))
                    plt.close(fig)

    # Convert to dataframe
    # gc_neuron_ind does not account for BLA neurons
    # They are not the same as neuron numbers in the sorted data!
    all_cp_df = pd.DataFrame(
            {'basename' : all_basename_list,
             'taste_num' : all_taste_num_list,
             'gc_neuron_ind' : all_nrn_list,
             'transition_num' : all_trans_num_list, 
             'cps' : all_cp_list})
    all_cp_df.to_pickle(all_cp_df_path)

else:
    all_cp_df = pd.read_pickle(all_cp_df_path)

# For each transition, plot scatter of mean-cp vs cp-variance
mean_cp_list = []
var_cp_list = []
for i, this_row in tqdm(all_cp_df.iterrows()):
    this_cp_list = this_row['cps']
    # Remove nones
    this_cp_list = [x for x in this_cp_list if x is not None]
    this_mean_cp = np.mean(this_cp_list)
    this_var_cp = np.var(this_cp_list)
    mean_cp_list.append(this_mean_cp)
    var_cp_list.append(this_var_cp)

all_cp_df['mean_cp'] = mean_cp_list
all_cp_df['var_cp'] = var_cp_list

# Plot scatter
snip_mid = transition_snips_df.iloc[0]['transition_snips'].shape[2] // 2 
trans_df_groups = [x[1] for x in all_cp_df.groupby('transition_num')]
fig, ax = plt.subplots(1, len(trans_df_groups), figsize = (20,5),
                       sharex = True, sharey = True)
for i, (this_ax, this_df) in enumerate(zip(ax, trans_df_groups)):
    this_ax.scatter(this_df['mean_cp'], this_df['var_cp'])
    this_ax.set_xlabel('Mean CP')
    this_ax.set_ylabel('Var CP')
    this_ax.set_title(f'Transition {i}')
    this_ax.axvline(snip_mid, color = 'r', linestyle = '--',
                    label = 'Center')
this_ax.legend()
plt.suptitle('Mean vs Var CP')
plt.savefig(os.path.join(plot_dir, 'mean_vs_var_cp.png'))
plt.close(fig)

# Plot histograms of mean and var cp for each transition
fig, ax = plt.subplots(2,1, figsize = (7,7))
for i, this_df in enumerate(trans_df_groups):
    this_ax = ax[0]
    this_ax.hist(this_df['mean_cp'], 
                 bins = 30, 
                 label = f'Transition {i}',
                 histtype = 'step',
                 linewidth = 2,)
    if i == len(trans_df_groups)-1:
        this_ax.axvline(snip_mid, color = 'r', linestyle = '--',
                        label = 'Center')
    this_ax.set_xlabel('Mean CP')
    this_ax.set_ylabel('Count')
    this_ax.legend()
    this_ax = ax[1]
    this_ax.hist(this_df['var_cp'], 
                 bins = 30,
                 histtype = 'step',
                 linewidth = 2,
                 )
    this_ax.set_xlabel('Var CP')
    this_ax.set_ylabel('Count')
    this_ax.legend()
plt.suptitle('Mean and Var CP Histograms')
plt.savefig(os.path.join(plot_dir, 'mean_var_cp_hist.png'))
plt.close(fig)

############################################################
# Transition timing and coherence
# 1) Does one subpopulation transition before the other
# 2) Is inter-trial variance of one group different than the other?
# 3) One a single-trial basis, is one population transition more closely
#   together than the other?

all_cp_df = all_cp_df.merge(
        gc_neurons[['basename','rel_inds','abs_inds','group_label']],
        left_on = ['basename','gc_neuron_ind'],
        right_on = ['basename','rel_inds'])
all_cp_df.drop(columns = ['gc_neuron_ind'], inplace = True)
all_cp_df.dropna(inplace = True)

##############################
# 1) Does one subpopulation transition before the other
# Make histogram of mean_cp by group
sns.displot(
        data = all_cp_df,
        x = 'mean_cp',
        hue = 'group_label',
        row = 'basename',
        col = 'transition_num',
        kind = 'kde',
        facet_kws = dict(sharey = False),
        height = 2,
        aspect = 3,
        )
plt.suptitle('Mean CP by Group')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'mean_cp_by_group_dists.png'))
plt.close(fig)

###############
# For each session, calculate MEDIAN for each group and find differences between them
med_cp_df = all_cp_df.groupby(
        ['basename','transition_num','group_label','taste_num'])['mean_cp'].median()
med_cp_df = pd.DataFrame(med_cp_df).reset_index() 

# Find unique differences
all_groups = med_cp_df['group_label'].unique()
unique_combos = list(combinations(all_groups,2))

name_trans_df_list = list(med_cp_df.groupby(['basename','transition_num','taste_num']))
name_trans_df_ids = [x[0] for x in name_trans_df_list]
name_trans_df_list = [x[1] for x in name_trans_df_list]

ids_list = []
combo_list = []
taste_list = []
diff_list = []
for this_id, this_df in zip(name_trans_df_ids, name_trans_df_list): 
    for this_combo in unique_combos:
        present_bool = this_combo[0] in this_df['group_label'].values and\
                this_combo[1] in this_df['group_label'].values
        if present_bool:
            this_diff = this_df[this_df['group_label'] == this_combo[0]]['mean_cp'].values[0] -\
                    this_df[this_df['group_label'] == this_combo[1]]['mean_cp'].values[0]
            ids_list.append(this_id)
            combo_list.append(this_combo)
            diff_list.append(this_diff)
            taste_list.append(this_df['taste_num'].values[0])

group_diff_df = pd.DataFrame(
        data = {'basename' : [x[0] for x in ids_list], 
                'transition_num' : [x[1] for x in ids_list],
                'group_combo' : combo_list,
                'taste_num' : taste_list,
                'mean_cp_diff' : diff_list})
group_diff_df['group_combo'] = group_diff_df['group_combo'].astype(str)

# For each combo and transition, calculate difference from 0 p-value
trans_combo_group_list = list(group_diff_df.groupby(['transition_num','group_combo']))
trans_combo_ids = [x[0] for x in trans_combo_group_list]
trans_combo_group_list = [x[1] for x in trans_combo_group_list]

ttest_list = [ttest_1samp(x['mean_cp_diff'].values,0) for x in trans_combo_group_list]
p_val_array = np.array([x[1] for x in ttest_list])
p_val_df = pd.DataFrame(
        data = {'transition_num' : [x[0] for x in trans_combo_ids],
                'group_combo' : [x[1] for x in trans_combo_ids],
                'p_val' : p_val_array})
alpha = 0.05
p_val_df['sig'] = p_val_df['p_val'] < alpha
p_val_df.to_csv(os.path.join(plot_dir, 'mean_cp_diff_ttest.csv'))

# Do an anova per transition to see whether deviation from 0
# is significant
trans_group_list = list(group_diff_df.groupby('transition_num'))
trans_group_ids = [x[0] for x in trans_group_list]
trans_group_list = [x[1] for x in trans_group_list]

anova_outs = [
        pg.anova(x, dv = 'mean_cp_diff', between = 'group_combo')
        for x in trans_group_list]
anova_p_vals = [x['p-unc'].values[0] for x in anova_outs]
anova_df = pd.DataFrame(
        data = {'transition_num' : trans_group_ids,
                'p_val' : anova_p_vals})
anova_df.to_csv(os.path.join(plot_dir, 'mean_cp_diff_anova.csv'))


sns.catplot(
        data = group_diff_df,
        x = 'group_combo',
        y = 'mean_cp_diff',
        col = 'transition_num',
        kind = 'swarm',
        height = 7,
        aspect = 0.75,
        )
# Rotate x-axis labels
for ax in plt.gcf().axes:
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    ax.axhline(0, color = 'red', linestyle = '--')
plt.suptitle('Mean CP Differences by Group\nEach point is a taste per session average')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'mean_cp_diff_by_group.png'))
plt.close(fig)

###############
# Instead of looking at differences BETWEEN groups, look at average time for each group
# For each group, calculate p-value for difference from snip_mid
mean_cp_df = all_cp_df.groupby(['basename','group_label','transition_num','taste_num'])['mean_cp'].mean()
mean_cp_df = pd.DataFrame(mean_cp_df).reset_index()
# Correct mean_cp for each group by subtracting snip_mid
mean_cp_df['mean_cp'] = mean_cp_df['mean_cp'] - snip_mid

group_list = list(mean_cp_df.groupby(['group_label','transition_num']))
group_ids = [x[0] for x in group_list]
group_list = [x[1] for x in group_list]

#####
# Model each group's mean_cp as a t-distribution and find if mean is different from snip_mid
# due to presence of outliers
# Data is also censored (bound) between 0 and 600 so used a censored likelihood

trace_save_dir = os.path.join(artifact_dir, 't_dist_traces')
if not os.path.exists(trace_save_dir):
    os.makedirs(trace_save_dir)

if len(os.listdir(trace_save_dir)) == 0:

    with pm.Model() as model:
        
        data = pm.Data('data', group_list[0]['mean_cp'].values) 

        # Priors
        mu = pm.Normal('mu', mu = 0, sigma = 10)
        sigma = pm.HalfNormal('sigma', sigma = 10)
        nu = pm.Exponential('nu', 1/30)

        # # Likelihood
        # y = pm.StudentT('y', mu = mu, sigma = sigma, nu = nu, observed = data)
        # Censor likelihood between 0 and 600
        y_latent = pm.StudentT.dist(mu = mu, sigma = sigma, nu = nu)
        y = pm.Censored('y', y_latent, lower = -300, upper = 300, observed = data) 

    traces = []
    for this_group in tqdm(group_list):
        with model:
            pm.set_data({'data' : this_group['mean_cp'].values})
            traces.append(pm.sample())

    group_id_strs = [f'{x[0]}_{x[1]}' for x in group_ids]
    save_paths = [os.path.join(trace_save_dir, f't_dist_trace_{x}.nc') for x in group_id_strs]
    trace_df = pd.DataFrame(
            data = {'group_id' : group_ids,
                    'group_id_strs' : group_id_strs,
                    'save_path' : save_paths})
    trace_df.to_csv(os.path.join(trace_save_dir, 'trace_df.csv'))

    for i, this_trace in enumerate(traces):
        this_trace.to_netcdf(save_paths[i])

else:
    trace_df = pd.read_csv(os.path.join(trace_save_dir, 'trace_df.csv'))
    traces = []
    for this_group in group_list:
        this_path = trace_df[
                trace_df['group_id'] == this_group]
        traces.append(arviz.from_netcdf(this_path))


# Extract mu from each trace
mu_list = [x.posterior.mu.values for x in traces]

# Calculate p-values for midline
p_of_score = [percentileofscore(x.flatten(), 0) for x in mu_list]
p_val_2sided = [np.round((2 * min(x, 100-x))/100,4) for x in p_of_score]
alpha = 0.05 # / len(p_val_2sided)
sig_vec = np.array(p_val_2sided) < alpha

t_dist_pval_df = pd.DataFrame(
        data = {'group_label' : [x[0] for x in group_ids],
                'transition_num' : [x[1] for x in group_ids],
                'p_val' : p_val_2sided,
                'sig' : sig_vec})
t_dist_pval_df.sort_values(by = ['transition_num','group_label'], inplace = True)
sig_mark_dict = {0.05 : '*', 0.01 : '**', 0.005 : '***'}
row_ind_list = []
sig_mark_list = []
for i, this_row in t_dist_pval_df.iterrows():
    this_p = this_row['p_val']
    for this_alpha in list(sig_mark_dict.keys())[::-1]:
        if this_p < this_alpha:
            sig_mark_list.append(sig_mark_dict[this_alpha])
            row_ind_list.append(i)
            break
t_dist_pval_df.loc[row_ind_list, 'sig_mark'] = sig_mark_list

t_dist_pval_df.to_csv(os.path.join(plot_dir, 'mean_cp_by_group_tdist.csv'))

# Plot histograms
fig, ax = plt.subplots(len(mu_list), 1, figsize = (5,20),
                       sharex = True, sharey = True)
for i, (this_ax, this_mu) in enumerate(zip(ax, mu_list)):
    this_ax.hist(this_mu.flatten(), bins = 30)
    this_ax.axvline(0, color = 'r', linestyle = '--',
                    label = 'Center')
    this_ax.set_ylabel('Count')
    this_ax.set_title(f'Group {group_ids[i]} - p_val: {p_val_2sided[i]}')
this_ax.legend()
this_ax.set_xlabel('Mean CP')
plt.suptitle('Mean CP by Group - t-dist Inferred Mean')
plt.savefig(os.path.join(plot_dir, 'mean_cp_by_group_inferred_mu.png'))
plt.close(fig)

# #####
# wanted_ppc = pm.sample_posterior_predictive(wanted_group_trace, model = model)
# 
# fig, ax = plt.subplots() 
# ax2 = ax.twinx()
# ax.hist(wanted_ppc.posterior_predictive.y.values.flatten(), bins = np.linspace(0, 600, 50),
#         log = True)
# ax.axvline(snip_mid, color = 'r', linestyle = '--')
# y_random = np.random.random(wanted_group['mean_cp'].shape[0])
# ax2.scatter(wanted_group['mean_cp'], [0.2]*len(wanted_group), 
#             marker = '|', color = 'k', s = 100) 
# ax2.axis('off')
# plt.show()

ttest_list = [ttest_1samp(x['mean_cp'].values,snip_mid) for x in group_list]
p_val_array = np.array([x[1] for x in ttest_list])
effect_size_array = np.array([x[0] for x in ttest_list])
p_val_df = pd.DataFrame(
        data = {'group_label' : [x[0] for x in group_ids],
                'transition_num' : [x[1] for x in group_ids],
                'p_val' : p_val_array,
                'effect_size' : np.abs(effect_size_array)})
alpha = 0.05 
p_val_df['sig'] = p_val_df['p_val'] < alpha
p_val_df.sort_values(by = ['transition_num','group_label'], inplace = True)
p_val_df.to_csv(os.path.join(plot_dir, 'mean_cp_by_group_ttest.csv'))


inferred_mu_df_list = []
g = sns.catplot(
        data = mean_cp_df,
        y = 'group_label',
        x = 'mean_cp',
        col = 'transition_num',
        kind = 'swarm',
        height = 7,
        aspect = 0.75,
        alpha = 0.75,
        )
# Rotate x-axis labels
for i, ax in enumerate(plt.gcf().axes):
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    ax.axvline(0, color = 'red', linestyle = '--')
    wanted_trace_inds = [j for j,x in enumerate(group_ids) if x[1] == i]
    wanted_ids = [group_ids[x] for x in wanted_trace_inds]
    wanted_group_names = [x[0] for x in wanted_ids]
    wanted_mus = [mu_list[x].flatten() for x in wanted_trace_inds]
    inferred_mu_df = pd.DataFrame(
            data = {'group_label' : wanted_group_names,
                    'inferred_mu' : wanted_mus})
    # Explode inferred_mu
    inferred_mu_df = inferred_mu_df.explode('inferred_mu')
    inferred_mu_df['transition_num'] = i
    inferred_mu_df_list.append(inferred_mu_df)
    sns.boxplot(
            data = inferred_mu_df,
            y = 'group_label',
            x = 'inferred_mu',
            ax = ax,
            color = 'r',
            linewidth = 2,
            )
    for this_group_label in wanted_group_names:
        this_sig_mark = t_dist_pval_df[
                (t_dist_pval_df['transition_num'] == i) &
                (t_dist_pval_df['group_label'] == this_group_label)]['sig_mark'].values[0]
        if not this_sig_mark == 'nan':
            ax.text(
                    y = wanted_group_names.index(this_group_label),
                    x = np.max(ax.get_xlim())*0.95,
                    s = this_sig_mark,
                    ha = 'center',
                    va = 'center',
                    fontsize = 20,
                    fontweight = 'bold',
                    color = 'r',
                    )
    ax.set_xlabel('<-- Before mean population | After mean population -->\n(ms)') 
plt.suptitle('Mean CP by Group\nEach point is a taste per session average'+\
        '\nBoxplot is t-dist inferred mean\n' + str(sig_mark_dict)) 
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'mean_cp_by_group.png'))
plt.close(fig)

# Make a boxplot of inferred mu
inferred_mu_df_final = pd.concat(inferred_mu_df_list)
sns.catplot(
        data = inferred_mu_df_final,
        y = 'group_label',
        x = 'inferred_mu',
        col = 'transition_num',
        kind = 'box',
        height = 5,
        aspect = 0.75,
        )
for i, this_ax in enumerate(plt.gcf().axes):
    this_ax.axvline(0, color = 'red', linestyle = '--')
    this_ax.set_xlabel('<--Earlier | Later-->\n(ms)')
    for this_group_label in wanted_group_names:
        this_sig_mark = t_dist_pval_df[
                (t_dist_pval_df['transition_num'] == i) &
                (t_dist_pval_df['group_label'] == this_group_label)]['sig_mark'].values[0]
        if not this_sig_mark == 'nan':
            this_ax.text(
                    y = wanted_group_names.index(this_group_label),
                    x = np.max(this_ax.get_xlim())*0.95,
                    s = this_sig_mark,
                    ha = 'center',
                    va = 'center',
                    fontsize = 20,
                    fontweight = 'bold',
                    color = 'r',
                    )
plt.suptitle('Inferred Mu by Group\n' + str(sig_mark_dict))
plt.tight_layout() 
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'inferred_mu_by_group.png'))
plt.close(fig)

# Also create a box plot with everything in a single plot
fig, ax = plt.subplots(1, 1, figsize = (10,5))
this_region = 'gc'
sns.boxplot(
        data = inferred_mu_df_final,
        x = 'transition_num',
        y = 'inferred_mu',
        hue = 'group_label',
        ax = ax,
        linewidth = 2,
        palette = ['red','orange','green','blue'],
        # order = [
        #     f'{this_region}_inter_receive_only',
        #     f'{this_region}_inter_send_receive',
        #     f'{this_region}_inter_send_only',
        #     f'{this_region}_intra_only',
        #     ],
        hue_order = [
            f'{this_region}_inter_receive_only',
            f'{this_region}_inter_send_receive',
            f'{this_region}_inter_send_only',
            f'{this_region}_intra_only',
            ],
        )
ax.axhline(0, color = 'red', linestyle = '--')
ax.set_ylabel('<--Earlier | Later-->\n(ms)')
# Put legend outside
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('Inferred Mu by Group\n' + str(sig_mark_dict))
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'inferred_mu_by_group_single.png'))
plt.close(fig)

##############################
# 2) Is inter-trial variance of one group different than the other?

# Group by session, transition, taste, and group
# Calculate median of var_cp 
med_var_cp_df = all_cp_df.groupby(
        ['basename','transition_num','group_label','taste_num'])['var_cp'].median()
med_var_cp_df = pd.DataFrame(med_var_cp_df).reset_index()

# Plot
sns.catplot(
        data = med_var_cp_df,
        x = 'group_label',
        y = 'var_cp',
        col = 'transition_num',
        kind = 'swarm',
        height = 7,
        aspect = 0.75,
        )
# Rotate x-axis labels
for i, ax in enumerate(plt.gcf().axes):
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
plt.suptitle('Var CP by Group')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'var_cp_by_group.png'))
plt.close(fig)

##############################
# 3) One a single-trial basis, is one population transition more closely
#   together than the other?
# This can be done by calculating variance of cp across neurons for each group on a
# single-trial basis and comparing mean of variance across groups

group_df_list = list(all_cp_df.groupby(['basename','transition_num','taste_num','group_label']))
group_ids = [x[0] for x in group_df_list]
group_df_list = [x[1] for x in group_df_list]

trial_cp_var_list = []
all_cp_array_list = []
trial_count_list = []
neuron_count_list = []
for this_id, this_df in zip(group_ids, group_df_list):
    all_cp_array = np.stack(this_df['cps'].values.tolist()).T
    # Convert None to nan
    all_cp_array = np.array([[np.nan if x is None else x for x in y] for y in all_cp_array])
    # Drop all trials with nan
    # all_cp_array = all_cp_array[~np.isnan(all_cp_array).any(axis = 1)]
    # Calculate variance across neurons
    all_cp_var = np.nanvar(all_cp_array, axis = 1)
    trial_count_list.append(all_cp_array.shape[0])
    neuron_count_list.append(all_cp_array.shape[1])
    all_cp_array_list.append(all_cp_array)
    trial_cp_var_list.append(all_cp_var)

mean_trial_cp_var_list = [np.nanmean(x) for x in trial_cp_var_list]

# Check relationship between trial_count, neuron_count, and mean_cp_var
fig, ax = plt.subplots(1, 2, figsize = (10,5), sharey = True)
ax[0].scatter(trial_count_list, mean_trial_cp_var_list)
ax[0].set_xlabel('Trial Count')
ax[0].set_ylabel('CP Variance')
ax[1].scatter(neuron_count_list, mean_trial_cp_var_list)
ax[1].set_xlabel('Neuron Count')
ax[1].set_ylabel('CP Variance')
plt.suptitle('CP Variance vs Trial Count and Neuron Count')
plt.savefig(os.path.join(plot_dir, 'mean_cp_var_vs_trial_neuron_count.png'))
plt.close(fig)


trial_cp_var_df = pd.DataFrame(
        data = {'basename' : [x[0] for x in group_ids],
                'transition_num' : [x[1] for x in group_ids],
                'taste_num' : [x[2] for x in group_ids],
                'group_label' : [x[3] for x in group_ids],
                'neuron_count' : neuron_count_list,
                'mean_trial_cp_var' : mean_trial_cp_var_list})
# Drop nans
trial_cp_var_df.dropna(inplace = True)
trial_cp_var_df.to_csv(os.path.join(plot_dir, 'trial_cp_var.csv'))

nonzero_trial_cp_var_df = trial_cp_var_df.loc[trial_cp_var_df['mean_trial_cp_var'] > 0]

# For each transition, is there a difference between mean_trial_cp_var of groups given neuron count
# Model as linear model with group_label as categorical variable
# and neuron_count as continuous variable
# Model each transition separately
import statsmodels.formula.api as smf

model_list = []
for this_trans in nonzero_trial_cp_var_df['transition_num'].unique():
    this_df = nonzero_trial_cp_var_df[nonzero_trial_cp_var_df['transition_num'] == this_trans]
    this_df['group_label'] = this_df['group_label'].astype('category')
    this_df['neuron_count'] = this_df['neuron_count'].astype('float')
    # print(list(this_df.group_label.unique()))
    smf_out = smf.ols('mean_trial_cp_var ~ C(group_label) * neuron_count', data = this_df).fit()
    model_list.append(smf_out)



# Plot
sns.catplot(
        data = nonzero_trial_cp_var_df,
        x = 'neuron_count',
        y = 'mean_trial_cp_var',
        col = 'transition_num',
        kind = 'box',
        hue = 'group_label',
        # s = 50,
        height = 7,
        aspect = 0.75,
        )
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'mean_trial_cp_var_by_group_neuron.png'))
plt.close(fig)

sns.catplot(
        data = nonzero_trial_cp_var_df,
        x = 'group_label',
        y = 'mean_trial_cp_var',
        col = 'transition_num',
        kind = 'swarm',
        hue = 'neuron_count',
        height = 7,
        aspect = 0.75,
        )
# Rotate x-axis labels
for ax in plt.gcf().axes:
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
plt.suptitle('Mean Trial CP Var by Group\nEach point is a taste per session average')
plt.tight_layout()
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'mean_trial_cp_var_by_group.png'))
plt.close(fig)

# For each session
