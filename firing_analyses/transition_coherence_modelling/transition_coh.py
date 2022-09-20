import numpy as np
import pylab as plt
from scipy.io import loadmat
import os
from scipy.signal import savgol_filter as savgol
import pandas as pd
import seaborn as sns

def rose_plot(ax, angles, bins=16, density=None, offset=0, lab_unit="degrees",
              start_zero=False, **param_dict):
    """
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    # Wrap angles to [-pi, pi)
    angles = (angles + np.pi) % (2*np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if bins % 2:
            bins += 1
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    count, bin = np.histogram(angles, bins=bins)

    # Compute width of each bin
    widths = np.diff(bin)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bin
        area = count / angles.size
        # Calculate corresponding bin radius
        radius = (area / np.pi)**.5
    else:
        radius = count

    # Plot data on ax
    ax.bar(bin[:-1], radius, zorder=1, align='center', width=widths,
           edgecolor='C0', 
           fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if lab_unit == "radians":
        #label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
        #          r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']
        label = ['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$',
                  r'$\pi$', r'$-3\pi/4$', r'$-2\pi/2$', r'$-\pi/4$']
        ax.set_xticklabels(label)

############################################################

base_dir = '/media/bigdata/firing_space_plot/firing_analyses/transition_coherence_modelling'
plot_dir = os.path.join(base_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

boot_phase_diff = loadmat(os.path.join(base_dir, 'boot_phase_diff.mat'))['boot_phase_diff']
boot_phase = loadmat(os.path.join(base_dir, 'boot_phase.mat'))['boot_phase']
selected_trials = loadmat(os.path.join(base_dir, 'selected_trials.mat'))['selected_r']

shuffled_phase = np.stack([
    boot_phase[:,0],
    np.random.permutation(boot_phase[:,1])
    ]).swapaxes(0,1)
shuffled_phase_diff_raw = np.squeeze(np.exp(-1j* np.diff(shuffled_phase,axis=1)))
shuffle_coh = np.abs(np.mean(shuffled_phase_diff_raw,axis=1))

# neurons x trials x regions x time
dt = 0.001
time_vec = np.arange(0,2 + dt, dt)
split_trials = np.stack(np.array_split(selected_trials, 2, axis=1))

lfp_array = np.sum(split_trials, axis = 2)


############################################################
# Trial plots
############################################################
trials = 3
change_ind = [
        [[500,2000]],
        [[500,750],[1050,1175]],
        [[275,470],[1100,2000]]
        ]
fig,ax = plt.subplots(trials,1, sharex=True, figsize = (7,3))
for num, this_ax in enumerate(ax): 
    this_ax.plot(lfp_array[:,num].T, alpha = 0.8, zorder = 5)
    this_changes = change_ind[num]
    for x in this_changes:
        this_ax.axvspan(x[0], x[1], color = 'y', alpha = 0.3, zorder = 1)
ax[1].set_ylabel('Firing Rate (Hz)')
ax[-1].set_xlabel('Time (ms)')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'example_trials.png'), dpi = 300)
plt.close(fig)
#plt.show()

W=[1.2,-0.5,0.5,0,-1.5,1.9,0,1.2,-1.1,-0.6,0,-0.8,-1.5,-1.3,-0.8,0],
W = np.reshape(W, (4,4))

fig,ax = plt.subplots(figsize = (3,3))
#im = ax.matshow(W, cmap = 'jet')
im = ax.pcolormesh(W, cmap = 'jet', edgecolor = 'k', linewidth = 2)
ax.set_aspect('equal')
fig.colorbar(im, fraction = 0.046, 
        label = 'Connection Strength')#, pad = 0.04)
plt.subplots_adjust(right = 0.9)
ticks = np.arange(4) + 0.5
ax.set_frame_on(False)
ax.tick_params(bottom=False, left = False)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(['GC.1','GC.2','BLA.1','BLA.2'], rotation = 45)
ax.set_yticklabels(['GC.1','GC.2','BLA.1','BLA.2'])
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'connection_mat.png'), dpi = 300)
plt.close(fig)
#plt.show()

############################################################
# Transition-aligned coherence 
############################################################
time_vec = np.arange(boot_phase_diff.shape[-1]) - boot_phase_diff.shape[-1]//2
boot_coherence = np.abs(np.mean(boot_phase_diff, axis=1))
med_coh = np.median(boot_coherence,axis=0)
std_coh = np.std(boot_coherence,axis=0)

med_coh_sm = savgol(med_coh, 101, 2)
std_coh_sm = savgol(std_coh, 101, 2)

med_shuf_coh = np.median(shuffle_coh, axis=0)
std_shuf_coh = np.std(shuffle_coh, axis=0)

med_shuf_coh_sm = savgol(med_shuf_coh, 101, 2)
std_shuf_coh_sm = savgol(std_shuf_coh, 101, 2)

#kern = np.ones(100)/100
#med_coh = np.convolve(med_coh, kern, mode = 'same')
#std_coh = np.convolve(std_coh, kern, mode = 'same')

fig = plt.figure(figsize = (4,4))
plt.plot(time_vec, med_coh_sm)
plt.fill_between(
        x = time_vec,
        y1 = med_coh_sm + std_coh_sm,
        y2 = med_coh_sm - std_coh_sm,
        alpha = 0.3
        )
plt.plot(time_vec, med_shuf_coh_sm)
plt.fill_between(
        x = time_vec,
        y1 = med_shuf_coh_sm + std_shuf_coh_sm,
        y2 = med_shuf_coh_sm - std_shuf_coh_sm,
        alpha = 0.3
        )
plt.axvline(0, color = 'red', linestyle = '--', linewidth = 2)
plt.xlabel('Time post-transition (ms)')
plt.ylabel('Coherence (0-1)')
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, 'transition_aligned_coherence.png'), dpi = 300)
plt.close(fig)
#plt.show()


############################################################
# Phase hist + Mean Coherence 
############################################################
boot_phase_split = np.array_split(boot_phase_diff, 2, axis=-1)
boot_phase_split = np.stack([x.flatten() for x in boot_phase_split])

ax = [plt.subplot(1,2,1,projection = 'polar'),
        plt.subplot(1,2,2,projection = 'polar')]
for dat, this_ax in zip(boot_phase_diff, ax):
    rose_plot(this_ax, np.angle(dat), bins = 30)
plt.show()


med_coh_split = np.stack(np.split(med_coh,2))
inds = np.array(list(np.ndindex(med_coh_split.shape)))

med_coh_frame = pd.DataFrame(dict(
    State = inds[:,0],
    Time = inds[:,1],
    Coherence = med_coh_split.flatten()
    ))
sh_frame = pd.DataFrame(dict(
    State = 3,
    Time = np.arange(len(med_shuf_coh)),
    Coherence = med_shuf_coh
    ))
med_coh_frame = pd.concat([med_coh_frame, sh_frame])

fig,ax = plt.subplots(figsize = (2,2))
sns.barplot(data = med_coh_frame,
        x = 'State',
        y = 'Coherence',
        ci = 'sd',
        alpha = 0.7,
        capsize = .2,
        linewidth = 3,
        edgecolor = ".5",
        ax=ax)
ax.set_ylabel('Mean Coherence')
ax.set_xlabel('State Coherence')
ax.set_xticklabels(['High','Low','Shuffle'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
#plt.show()
fig.savefig(os.path.join(plot_dir, 'mean_coherence_per_state.svg'), dpi = 300)
plt.close(fig)
