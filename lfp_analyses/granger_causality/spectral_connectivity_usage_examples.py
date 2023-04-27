# From https://github.com/Eden-Kramer-Lab/spectral_connectivity/blob/master/examples/Tutorial_Using_Paper_Examples.ipynb

from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity

from spectral_connectivity.simulate import simulate_MVAR
import numpy as np
import pylab as plt
from tqdm import tqdm, trange


############################################################
# Baccalá, L.A., and Sameshima, K. (2001). Partial directed coherence: a new concept in neural structure determination. Biological Cybernetics 84, 463–474.
############################################################
sampling_frequency = 200
n_time_samples, n_lags, n_signals = 100, 1, 3

coefficients = np.array(
    [[[0.5, 0.3, 0.4], [-0.5, 0.3, 1.0], [0.0, -0.3, -0.2]]])
noise_covariance = np.eye(n_signals)

time_series = simulate_MVAR(
    coefficients,
    noise_covariance=noise_covariance,
    n_time_samples=n_time_samples,
    n_trials=30,
    n_burnin_samples=500,
)

added_noise = (np.random.random(time_series.shape) - 0.5)*10
time_series += added_noise

time_halfbandwidth_product = 2
m = Multitaper(
    time_series,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    start_time=0,
)

c = Connectivity(
    fourier_coefficients=m.fft(), frequencies=m.frequencies, time=m.time
)

# pairwise_spectral_granger=c.pairwise_spectral_granger_prediction()

measures = dict(
    pairwise_spectral_granger=c.pairwise_spectral_granger_prediction()
)

############################################################
# Shuffling
############################################################
shuffles = 500
shuffle_outs = []
for i in trange(shuffles):
    # Shuffle trials (not actual timesteps)
    temp_series = np.stack([np.random.permutation(x) for x in time_series.T]).T
    m = Multitaper(
        temp_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=time_halfbandwidth_product,
        start_time=0,
    )

    shuffle_c = Connectivity(
        fourier_coefficients=m.fft(), frequencies=m.frequencies, time=m.time
    )

    shuffle_outs.append(
        shuffle_c.pairwise_spectral_granger_prediction()
    )

shuffle_array = np.stack(shuffle_outs).squeeze()
mean_shuffle = shuffle_array.mean(axis=0)
std_shuffle = shuffle_array.std(axis=0)

# fig, axes = plt.subplots(
#    n_signals, n_signals, figsize=(n_signals * 3, n_signals * 3), sharex=True
# )
# for ind1, ind2, ax in zip(signal_ind1.ravel(), signal_ind2.ravel(), axes.ravel()):
#    ax.fill_between(
#            x = shuffle_c.frequencies,
#            y1 = mean_shuffle[:,ind1,ind2] + std_shuffle[:,ind1,ind2],
#            y2 = mean_shuffle[:,ind1,ind2] - std_shuffle[:,ind1,ind2],
#            alpha = 0.7
#            )
# plt.show()

############################################################

n_signals = time_series.shape[-1]
signal_ind2, signal_ind1 = np.meshgrid(
    np.arange(n_signals), np.arange(n_signals))

fig, axes = plt.subplots(
    n_signals, n_signals, figsize=(n_signals * 3, n_signals * 3),
    sharex=True, sharey=True
)
for ind1, ind2, ax in zip(signal_ind1.ravel(), signal_ind2.ravel(), axes.ravel()):
    for measure_name, measure in measures.items():
        ax.plot(
            c.frequencies,
            measure[0, :, ind1, ind2],
            label=measure_name,
            linewidth=3,
            alpha=0.8,
        )
        ax.fill_between(
            x=shuffle_c.frequencies,
            y1=mean_shuffle[:, ind1, ind2] + 3*std_shuffle[:, ind1, ind2],
            y2=mean_shuffle[:, ind1, ind2] - 3*std_shuffle[:, ind1, ind2],
            alpha=0.7
        )
    ax.set_title(f"x{ind2+1} → x{ind1+1}" +
                 '\n' + f'coeff : {coefficients.squeeze()[ind1,ind2]}',
                 fontsize=15)
    #ax.set_ylim((0, np.max([np.nanmax(np.stack(list(measures.values()))), 1.05])))
    if ind1 == n_signals-1:
        ax.set_xlabel("Freq ('Hz')")
axes[0, -1].legend()
plt.tight_layout()
plt.show()

#plt.scatter(time_series[:,0,1], time_series[:,0,2])
# plt.show()

#fig, axes = plt.subplots()
#im = axes.imshow(coefficients.squeeze())
#plt.colorbar(im, ax=axes)
# plt.legend()
# plt.title("Power")

#fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
# for i,x in enumerate(c.power().squeeze().T):
#    axes.plot(c.frequencies, x, label = f'x{i+1}')
# plt.legend()
# plt.title("Power")
# plt.show()


############################################################
# Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing information flow in brain networks with nonparametric Granger causality. NeuroImage 41, 354–362.
############################################################

def Dhamala_example2a():
    sampling_frequency = 200
    n_time_samples, n_lags, n_signals = 450, 2, 2
    coefficients = np.zeros((n_lags, n_signals, n_signals))

    coefficients[0, 0, 0] = 0.53
    coefficients[1, 0, 0] = -0.80
    coefficients[0, 0, 1] = 0.50
    coefficients[0, 1, 1] = 0.53
    coefficients[1, 1, 1] = -0.80
    coefficients[0, 1, 0] = 0.00

    noise_covariance = np.eye(n_signals) * [0.25, 0.25]
    return (
        simulate_MVAR(
            coefficients,
            noise_covariance=noise_covariance,
            n_time_samples=n_time_samples,
            n_trials=30,
            n_burnin_samples=1000,
        ),
        sampling_frequency,
    )


def Dhamala_example2b():
    sampling_frequency = 200
    n_time_samples, n_lags, n_signals = 450, 2, 2
    coefficients = np.zeros((n_lags, n_signals, n_signals))

    coefficients[0, 0, 0] = 0.53
    coefficients[1, 0, 0] = -0.80
    coefficients[0, 0, 1] = 0.00
    coefficients[0, 1, 1] = 0.53
    coefficients[1, 1, 1] = -0.80
    coefficients[0, 1, 0] = 0.50

    noise_covariance = np.eye(n_signals) * [0.25, 0.25]
    return (
        simulate_MVAR(
            coefficients,
            noise_covariance=noise_covariance,
            n_time_samples=n_time_samples,
            n_trials=30,
            n_burnin_samples=1000,
        ),
        sampling_frequency,
    )


time_series1, sampling_frequency = Dhamala_example2a()
time_series2, _ = Dhamala_example2b()
time_series = np.concatenate((time_series1, time_series2), axis=0)

# Add noise to timeseries scaled by it's range
val_min, val_max = np.min(time_series), np.max(time_series)
noise_scale = 0.1
val_range = val_max - val_min
time_series += np.random.normal(0, noise_scale * val_range, time_series.shape)

time_halfbandwidth_product = 1

def calc_granger(time_series):
    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=time_halfbandwidth_product,
        start_time=0,
        time_window_duration=0.3,
        time_window_step=0.250,
    )
    c = Connectivity.from_multitaper(m)
    granger = c.pairwise_spectral_granger_prediction()
    return granger, c


granger, c = calc_granger(time_series)

n_shuffles = 1000
temp_series = [np.stack([np.random.permutation(x) for x in time_series.T]).T \
                                for i in trange(n_shuffles)]

# Calc shuffled granger
shuffle_outs = []
for i in trange(shuffles):
    # Shuffle trials (not actual timesteps)
    shuffle_outs.append(calc_granger(temp_series[i])[0])
shuffle_outs = np.stack(shuffle_outs)

mean_shuffle = np.nanmean(shuffle_outs, axis=0)
# Calculate given percentile across all shuffled values 
n_comparisons = shuffle_outs.shape[0] * shuffle_outs.shape[1]
alpha = 0.05
corrected_alpha = alpha / n_comparisons
wanted_percentile = 100 - (corrected_alpha * 100)
percentile_granger = np.percentile(shuffle_outs, wanted_percentile, axis=0)

cat_data = np.concatenate((granger, mean_shuffle), axis=3)
vmin, vmax = np.nanmin(cat_data), np.nanmax(cat_data)

# Create masked array for plotting
masked_granger = np.ma.masked_where(granger < percentile_granger, granger)

# Plot values of granger
# Show values less than shuffle in red
cmap = plt.cm.viridis
cmap.set_bad(color='black')

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

ax[0].pcolormesh(
    c.time, c.frequencies, masked_granger[..., :, 0, 1].T,
    cmap=cmap, shading="auto",
    vmin=vmin, vmax=vmax
)
ax[0].set_title("x1 -> x2")
ax[0].set_ylabel("Frequency")
ax[1].pcolormesh(
    c.time, c.frequencies, masked_granger[..., :, 1, 0].T,
    cmap=cmap, shading="auto",
    vmin=vmin, vmax=vmax
)
ax[1].set_title("x2 -> x1")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Frequency")

plt.show()
