##############################
# ____  _       _       
#|  _ \| | ___ | |_ ___ 
#| |_) | |/ _ \| __/ __|
#|  __/| | (_) | |_\__ \
#|_|   |_|\___/ \__|___/
##############################

# To extract spike data from model
# model.obs.observations

dat.get_stft(recalculate=False, dat_type = ['amplitude'])
dat.get_region_units()

##################################################
# ELBO Plot
##################################################
fig,ax = plt.subplots()                       
ax.plot(-approx.hist, label='new ADVI', alpha=.3)
plt.legend()
plt.ylabel('ELBO')
plt.xlabel('iteration');
plt.savefig(os.path.join(plot_dir,'ELBO_plot'))
plt.close(fig)

##################################################
# Average Tau Plot 
##################################################
long_tau_samples = tau_samples.reshape((-1, tau_samples.shape[-1]))
fig,ax = plt.subplots()
for switch in range(long_tau_samples.shape[-1]):
     plt.hist(long_tau_samples[...,switch],bins = 100, density = True,alpha = 0.8)
plt.savefig(os.path.join(plot_dir,'cumulated_changepoint_hist'),dpi=300)
plt.close(fig)

##################################################
# Lambda Plot
##################################################
mean_tau = np.mean(tau_samples, axis=0)
mean_lambda = np.mean(lambda_stack,axis=1).swapaxes(1,2)
sd_lambda = np.std(lambda_stack,axis=1).swapaxes(1,2)
zscore_mean_lambda = np.array([stats.zscore(nrn,axis=None) \
        for nrn in mean_lambda.swapaxes(0,1)]).swapaxes(0,1)

fig,ax = plt.subplots(2,mean_lambda.shape[0]);
for ax_num, (this_dat,this_zscore_dat) in \
                    enumerate(zip(mean_lambda,zscore_mean_lambda)):
    ax[0,ax_num].imshow(this_dat, interpolation = 'nearest', 
                aspect = 'auto', cmap = 'viridis',
                vmin = 0, vmax = np.max(mean_lambda,axis=None))
    ax[1,ax_num].imshow(this_zscore_dat, interpolation = 'nearest', 
                aspect = 'auto', cmap = 'viridis')

plt.savefig(os.path.join(plot_dir,'lambda_plot'))
plt.close(fig)

##################################################
# Changepoint Plot
##################################################
# Sort units be region
# Mark change in region using hline
unit_order = np.concatenate(dat.region_units)

mean_mean_tau = np.mean(tau_samples,axis=(0,1))

plot_spikes = dat_binned_long>0
plot_spikes = plot_spikes[:,unit_order]

channel = 0
stft_cut = stats.zscore(dat.amplitude_array[:,:],axis=-1)
stft_cut = stft_cut[:,channel,...,time_lims[0]:time_lims[1]]
stft_cut = np.reshape(stft_cut,(-1,*stft_cut.shape[2:]))
stft_ticks = dat.time_vec[time_lims[0]:time_lims[1]]*1000
stft_tick_inds = np.arange(0,len(stft_ticks),250)

mean_tau_stft = (mean_tau/np.max(mean_tau,axis=None))*stft_cut.shape[-1]

# Overlay raster with CDF of switchpoints
tick_interval = 5
time_tick_count = 10
binned_tick_inds = np.arange(0,len(binned_t_vec),
                        len(binned_t_vec)//time_tick_count)
binned_tick_vals = np.arange(time_lims[0],time_lims[1],
                        np.abs(np.diff(time_lims))//time_tick_count)
vline_kwargs = {'color': 'red', 'linewidth' :3, 'alpha' : 0.7}
hline_kwargs = {'color': 'red', 'linewidth' :1, 'alpha' : 1,
                    'xmin': -0.5, 'xmax' : plot_spikes.shape[-1] -0.5}
imshow_kwargs = {'interpolation':'nearest','aspect':'auto','origin':'lower'}

region_unit_count = dict(zip(dat.region_names,[len(x) for x in dat.region_units]))

for this_taste in np.sort(np.unique(taste_label)):
    trial_inds = np.where(taste_label==this_taste)[0]

    fig, ax = plt.subplots(len(trial_inds),3,sharex='col', figsize = (15,50))
    for num,trial in enumerate(trial_inds):
        #ax[num,0].imshow(plot_spikes[trial], **imshow_kwargs)
        ax[num,0].scatter(*np.where(plot_spikes[trial])[::-1], marker = "|")
        ax[num,0].set_ylabel(taste_label[trial])
        ax[num,1].imshow(stft_cut[trial],**imshow_kwargs)
        ax[num,0].vlines(mean_tau[trial],-0.5,
                            mean_lambda.shape[1]-0.5, **vline_kwargs)
        ax[num,0].hlines(len(dat.region_units[0]) -0.5 ,**hline_kwargs)
        ax[num,1].vlines(mean_tau_stft[trial],
                            -0.5,stft_cut.shape[1]-0.5,**vline_kwargs)
        ax[num,1].set_xticks(stft_tick_inds)
        ax[num,1].set_xticklabels(stft_ticks[stft_tick_inds],rotation='vertical')

        for state in range(tau_samples.shape[-1]):
            ax[num,-1].hist(tau_samples[:,trial,state], bins = 100, density = True)
    ax[-1,0].set_xticks(binned_tick_inds)
    ax[-1,0].set_xticklabels(binned_tick_vals, rotation = 'vertical')
    ax[-1,-1].set_xticks(binned_tick_inds)
    ax[-1,-1].set_xticklabels(binned_tick_vals, rotation = 'vertical')

    fig.suptitle(region_unit_count)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
    plt.savefig(os.path.join(\
            plot_dir,'taste{}_changepoints'.format(this_taste)),dpi=300)
    plt.close(fig)


##################################################
# Good Trial Changepoint Plot
##################################################
# Find trials where the mean tau for one changepoint is outside the 95% interval for other taus 

percentile_array = np.zeros((*mean_tau.shape,mean_tau.shape[-1]))
for trial_num, (this_mean_tau, this_tau_dist) in \
            enumerate(zip(mean_tau, np.moveaxis(tau_samples,0,-1))):
    for tau1_val, this_tau in enumerate(this_mean_tau):
        for tau2_val, this_dist in enumerate(this_tau_dist):
            percentile_array[trial_num, tau1_val, tau2_val] = \
                    percentileofscore(this_dist, this_tau)

# Visually, threshold of <1 percentile seems compelling
# Find all trials where all the upper triangular elements are <1
# and lower triangular elements are >99
lower_thresh = 1
upper_thresh = 100 - lower_thresh
good_trial_list = np.where([all(x[np.triu_indices(states-1,1)] < lower_thresh) \
                  and all(x[np.tril_indices(states-1,-1)] > upper_thresh) \
                  for x in percentile_array])[0]

# Plot only good trials
# Overlay raster with CDF of switchpoints
tick_interval = 5
max_trials = 15
num_plots = int(np.ceil(len(good_trial_list)/max_trials))
trial_inds_list = [good_trial_list[x*max_trials:(x+1)*max_trials] \
                        for x in np.arange(num_plots)]

for fig_num in np.arange(num_plots):
    trial_inds = trial_inds_list[fig_num]
    trial_count = len(trial_inds)
    
    fig, ax = plt.subplots(trial_count,3,sharex='col', figsize = (20,trial_count*3))
    for num,trial in enumerate(trial_inds):
        #ax[num,0].imshow(plot_spikes[trial], **imshow_kwargs)
        ax[num,0].scatter(*np.where(plot_spikes[trial])[::-1], marker = "|")
        ax[num,0].set_ylabel(taste_label[trial])
        ax[num,1].imshow(stft_cut[trial], **imshow_kwargs)
        ax[num,0].hlines(len(dat.region_units[0]) -0.5 ,**hline_kwargs)
        ax[num,0].vlines(mean_tau[trial],-0.5,mean_lambda.shape[1]-0.5,
                            **vline_kwargs)
        ax[num,1].vlines(mean_tau_stft[trial],-0.5,stft_cut.shape[1]-0.5,
                            **vline_kwargs)

        ax[num,1].set_xticks(stft_tick_inds)
        ax[num,1].set_xticklabels(stft_ticks[stft_tick_inds],rotation='vertical')

        for state in range(tau_samples.shape[-1]):
            ax[num,-1].hist(tau_samples[:,trial,state], bins = 100, density = True)
    ax[-1,0].set_xticks(binned_tick_inds)
    ax[-1,0].set_xticklabels(binned_tick_vals, rotation = 'vertical')
    ax[-1,-1].set_xticks(binned_tick_inds)
    ax[-1,-1].set_xticklabels(binned_tick_vals, rotation = 'vertical')

    fig.suptitle(region_unit_count)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(plot_dir,'good_changepoints{}'.format(fig_num)),dpi=300)
    plt.close(fig)

##################################################
# Good Trial Changepoint Plot - Color
##################################################
tick_interval = 5
max_trials = 15
num_plots = int(np.ceil(len(good_trial_list)/max_trials))
trial_inds_list = [good_trial_list[x*max_trials:(x+1)*max_trials] \
                        for x in np.arange(num_plots)]
imshow_kwargs = {'interpolation':'nearest','aspect':'auto','origin':'lower'}

for fig_num in np.arange(num_plots):
    trial_inds = trial_inds_list[fig_num]
    trial_count = len(trial_inds)
    
    fig, ax = plt.subplots(trial_count,3,sharex='col', figsize = (20,trial_count*3))
    for num,trial in enumerate(trial_inds):
        #ax[num,0].imshow(dat_binned_long[trial], cmap='jet',**imshow_kwargs)
        ax[num,0].scatter(*np.where(plot_spikes[trial])[::-1], marker = "|")
        ax[num,0].set_ylabel(taste_label[trial])
        ax[num,1].imshow(stft_cut[trial], **imshow_kwargs)
        ax[num,0].hlines(len(dat.region_units[0]) -0.5 ,**hline_kwargs)
        ax[num,0].vlines(mean_tau[trial],-0.5,mean_lambda.shape[1]-0.5,
                            **vline_kwargs)
        ax[num,1].vlines(mean_tau_stft[trial],-0.5,stft_cut.shape[1]-0.5,
                            **vline_kwargs)

        ax[num,1].set_xticks(stft_tick_inds)
        ax[num,1].set_xticklabels(stft_ticks[stft_tick_inds],rotation='vertical')

        for state in range(tau_samples.shape[-1]):
            ax[num,-1].hist(tau_samples[:,trial,state], bins = 100, density = True)
    ax[-1,0].set_xticks(binned_tick_inds)
    ax[-1,0].set_xticklabels(binned_tick_vals, rotation = 'vertical')
    ax[-1,-1].set_xticks(binned_tick_inds)
    ax[-1,-1].set_xticklabels(binned_tick_vals, rotation = 'vertical')

    fig.suptitle(region_unit_count)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(\
            plot_dir,'good_changepoints{}_color'.format(fig_num)),dpi=300)
    plt.close(fig)
