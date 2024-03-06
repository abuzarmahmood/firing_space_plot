"""
Recurrent neural network for firing rate estimation

Inputs:
    - spike trains (with binning) 
    - external input
Outputs:
    - firing rates

Loss:
    - Poisson log-likelihood

Initialization:
    - random weights
    - random biases
    - random initial conditions

Start prior to stim so initial conditions don't matter as much

Use ReLU in output layer to ensure positive firing rates
""" 

############################################################
# Imports
############################################################

import time
import numpy as np
import pylab as plt
from tqdm import tqdm, trange
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import explained_variance_score, r2_score
# import pandas as pd
# import seaborn as sns

############################################################
# Define Model 
############################################################
# Define networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.nn import functional as F
# from model import CTRNN
import math
from scipy.stats import poisson, zscore

class autoencoderRNN(nn.Module):
    """
    Input and output transformations are encoder and decoder architectures
    RNN will learn dynamics of latent space

    Output has to be rectified
    Can add dropout to RNN and autoencoder layers
    """
    def __init__(
            self, 
            input_size, 
            hidden_size,  
            output_size, 
            rnn_layers = 1,
            dropout = 0.2,
            ):
        """
        3 sigmoid layers for input and output each, to project between:
            encoder : input -> latent
            rnn : latent -> latent
            decoder : latent -> output
        """
        super(autoencoderRNN, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size, sum((input_size, hidden_size))//2),
                nn.Sigmoid(),
                nn.Linear(sum((input_size, hidden_size))//2, hidden_size),
                nn.Sigmoid(),
                )
        # self.rnn = nn.RNN(hidden_size, hidden_size, rnn_layers, batch_first=False, bidirectional=True)
        self.rnn = nn.RNN(
                hidden_size, 
                hidden_size, 
                rnn_layers, 
                batch_first=False, 
                bidirectional=False,
                dropout = dropout,
                )
        self.decoder = nn.Sequential(
                # nn.Linear(hidden_size*2, sum((hidden_size, output_size))//2),
                nn.Linear(hidden_size, sum((hidden_size, output_size))//2),
                nn.Sigmoid(),
                nn.Linear(sum((hidden_size, output_size))//2, output_size),
                # nn.Sigmoid(),
                # nn.Tanh(),
                )
        # self.relu = nn.ReLU()
        self.en_dropout = nn.Dropout(p = dropout)
        # self.de_dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        out = self.encoder(x)
        out = self.en_dropout(out)
        latent_out, _ = self.rnn(out)
        out = self.decoder(latent_out)
        # out = self.de_dropout(out)
        # out = self.relu(out)
        return out, latent_out


def train_model(
        net, 
        inputs, 
        labels, 
        train_steps = 1000, 
        lr=0.01, 
        delta_loss = 0.01,
        device = None,
        loss = 'poisson',
        test_inputs = None,
        test_labels = None,
        ):
    """Simple helper function to train the model.

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
        inputs: shape (seq_len, batch, input_size)
        labels: shape (seq_len * batch, output_size)

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # if loss == 'poisson':
    #     criterion = nn.PoissonNLLLoss(
    #             log_input=False, full=True, reduction='mean')
    # elif loss == 'mse':
    #     criterion = nn.MSELoss()
    criterion = nn.MSELoss()
    # criterion = normalizedMSELoss

    cross_val_bool = np.logical_and(
            test_inputs is not None, 
            test_labels is not None
            )

    loss_history = []
    cross_val_loss = {}
    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(train_steps):
        labels = labels.reshape(-1, output_size)


        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # output = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.reshape(-1, output_size)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Only compute cross_val_loss every 100 steps
        # because it's expensive
        if cross_val_bool and (i % 100 == 99): 
            test_out, _ = net(test_inputs)
            test_out = test_out.reshape(-1, output_size)
            test_labels = test_labels.reshape(-1, output_size)
            test_loss = criterion(test_out, test_labels)
            # cross_val_loss.append(test_loss.item())
            cross_val_loss[i] = test_loss.item()
            cross_str = f'Cross Val Loss: {test_loss.item():0.4f}'
        else:
            cross_str = ''

        # Compute the running loss every 100 steps
        current_loss = loss.item()
        loss_history.append(current_loss)
        running_loss += current_loss 
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, {}, Time {:0.1f}s'.format(
                i+1, running_loss, cross_str, time.time() - start_time))
            running_loss = 0
    return net, loss_history, cross_val_loss


############################################################
# Load data
import sys
import os
sys.path.append('/media/bigdata/firing_space_plot/ephys_data')
from ephys_data import ephys_data
import visualize as vz

plot_dir = '/media/bigdata/firing_space_plot/modelling/pytorch_rnn/plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

artifacts_dir = '/media/bigdata/firing_space_plot/modelling/pytorch_rnn/artifacts'
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

data_dir = '/media/storage/gc_only/AM34/AM34_4Tastes_201217_114556' 
dat = ephys_data(data_dir)
dat.firing_rate_params = dat.default_firing_params
dat.get_spikes()
dat.get_firing_rates()

spike_array = np.stack(dat.spikes)
# Drop first n trials
spike_n_trials = 0
spike_array = spike_array[:,spike_n_trials:]

cat_spikes = np.concatenate(spike_array)

# Bin spikes
bin_size = 25
binned_spikes = np.reshape(cat_spikes, 
                           (*cat_spikes.shape[:2], -1, bin_size)).sum(-1)

# vz.firing_overview(binned_spikes.swapaxes(0,1))
# plt.show()

# Reshape to (seq_len, batch, input_size)
inputs = binned_spikes.copy()
inputs = np.moveaxis(inputs, -1, 0)

##############################
# Perform PCA on data
# If PCA is performed on raw data, higher firing neurons will dominate
# the latent space
# Therefore, perform PCA on zscored data

inputs_long = inputs.reshape(-1, inputs.shape[-1])

# Perform standard scaling
scaler = StandardScaler()
# scaler = MinMaxScaler()
inputs_long = scaler.fit_transform(inputs_long)

# Perform PCA and get 95% explained variance
pca_obj = PCA()
inputs_pca = pca_obj.fit_transform(inputs_long)
explained_var = pca_obj.explained_variance_ratio_
cumulative_explained_var = np.cumsum(explained_var)
n_components = np.argmax(cumulative_explained_var > 0.95) + 1

pca_obj = PCA(n_components=n_components)
inputs_pca = pca_obj.fit_transform(inputs_long)

# Scale the PCA outputs
pca_scaler = StandardScaler()
inputs_pca = pca_scaler.fit_transform(inputs_pca)

# explained_var = []
# comp_vec = np.arange(1, inputs_long.shape[-1], 2)
# for i in tqdm(comp_vec): 
#     nmf_obj = NMF(n_components=i)
#     inputs_nmf = nmf_obj.fit_transform(inputs_long)
#     recreated = nmf_obj.inverse_transform(inputs_nmf)
#     score = explained_variance_score(inputs_long, recreated)
#     explained_var.append(score)
# 
# plt.plot(comp_vec, explained_var, '-x')
# plt.show()

# n_components = 20
# nmf_obj = NMF(n_components=n_components)
# inputs_nmf = nmf_obj.fit_transform(inputs_long)
# 
# nmf_scaler = StandardScaler()
# inputs_nmf = nmf_scaler.fit_transform(inputs_nmf)

# Shape (seq_len, trials, n_components)
# inputs_trial_nmf = inputs_nmf.reshape(inputs.shape[0], -1, n_components)
# inputs = inputs_trial_nmf.copy()

inputs_trial_pca = inputs_pca.reshape(inputs.shape[0], -1, n_components)
inputs = inputs_trial_pca.copy()

# plt.imshow(inputs.mean(axis = 1).T, aspect = 'auto', interpolation = 'none')
# plt.colorbar()
# plt.show()

##############################

# Add taste number as external input
taste_number = np.repeat(np.arange(4), inputs.shape[1]//4)
taste_number = np.broadcast_to(taste_number, (inputs.shape[0], inputs.shape[1]))

# Add stim time as external input
stim_time = np.zeros_like(taste_number)
stim_time[:, 2000//bin_size] = 1

inputs_plus_context = np.concatenate(
        [
            inputs, 
            # taste_number[:,:,None],
            stim_time[:,:,None]
            ], 
        axis = -1)

############################################################
# Train model
############################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = inputs.shape[-1] + 1 # Add 2 for taste, and stim-t
output_size = inputs.shape[-1] 

# Instead of predicting activity in the SAME time-bin,
# predict activity in the NEXT time-bin
# Hoping that this will make the model more robust to
# small fluctuations in activity
inputs_plus_context = inputs_plus_context[:-1]
inputs = inputs[1:]

# (seq_len * batch, output_size)
labels = torch.from_numpy(inputs).type(torch.float32)
# (seq_len, batch, input_size)
inputs = torch.from_numpy(inputs_plus_context).type(torch.float)

# Split into train and test
train_test_split = 0.75
train_inds = np.random.choice(
        np.arange(inputs.shape[1]), 
        int(train_test_split * inputs.shape[1]), 
        replace = False)
test_inds = np.setdiff1d(np.arange(inputs.shape[1]), train_inds)

train_inputs = inputs[:,train_inds]
train_labels = labels[:,train_inds]
test_inputs = inputs[:,test_inds]
test_labels = labels[:,test_inds]


train_inputs = train_inputs.to(device)
train_labels = train_labels.to(device)
test_inputs = test_inputs.to(device)
test_labels = test_labels.to(device)

hidden_size_vec = [6, 8]
loss_funcs_vec = ['mse']
repeats = 3

param_product = [[[x,y]]*repeats for x in hidden_size_vec for y in loss_funcs_vec]
param_product = [item for sublist in param_product for item in sublist]

# n_restarts = len(hidden_size_vec)
model_list = []
loss_list = []
cross_val_loss_list = []
# for i in trange(n_restarts):
for i, this_params in enumerate(tqdm(param_product)):
    net = autoencoderRNN( 
            input_size=input_size,
            hidden_size= this_params[0], 
            output_size=output_size,
            dropout = 0.2,
            # rnn_layers = 3,
            )
    net.to(device)
    net, loss, cross_val_loss = train_model(
            net, 
            train_inputs, 
            train_labels, 
            lr = 0.001, 
            train_steps = 15000,
            loss = this_params[1], 
            test_inputs = test_inputs,
            test_labels = test_labels,
            )
    model_list.append(net)
    loss_list.append(loss)
    cross_val_loss_list.append(cross_val_loss)

    model_name = f'hidden_{this_params[0]}_loss_{this_params[1]}_{i}'
    torch.save(net, os.path.join(artifacts_dir, f'{model_name}.pt'))

    fig, ax = plt.subplots()
    # ax.plot(np.stack(loss_list).T, alpha = 0.5)
    # ax.set_title(f'Losses for {i} restarts')
    for ind, loss in enumerate(loss_list):
        ax.plot(loss, label = f'params: {param_product[ind]}') 
    ax.legend(
            bbox_to_anchor=(1.05, 1), 
            loc='upper left', borderaxespad=0.)
    ax.set_title(f'Losses for {i+1}/{len(param_product)} restarts')
    fig.savefig(os.path.join(plot_dir,'run_loss.png'),
                bbox_inches = 'tight')
    plt.close(fig)

    # Plot another figure of cross_val_loss
    fig, ax = plt.subplots()
    for ind, loss in enumerate(cross_val_loss_list):
        ax.plot(loss.keys(), loss.values(),
                label = f'params: {param_product[ind]}')
    ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left', borderaxespad=0.)
    ax.set_title(f'Cross Val Losses for {i+1}/{len(param_product)} restarts')
    fig.savefig(os.path.join(plot_dir,'cross_val_loss.png'),
                bbox_inches = 'tight')
    plt.close(fig)

# ##############################
# # Load models and calculate cross_val_loss
# cross_val_loss_list = []
# model_list = []
# for i, this_params in enumerate(tqdm(param_product)):
#     model_name = f'hidden_{this_params[0]}_loss_{this_params[1]}_{i}'
#     net = torch.load(os.path.join(artifacts_dir, f'{model_name}.pt'))
#     test_out, _ = net(test_inputs)
#     test_out = test_out.reshape(-1, output_size)
#     test_labels = test_labels.reshape(-1, output_size)
#     test_loss = F.mse_loss(test_out, test_labels)
#     cross_val_loss_list.append(test_loss.item())
#     model_list.append(net)

##############################

# # Calculate log-likehood for each model
# def mean_poisson_nll(output, target):
#     temp = output - target * torch.log(output + 1e-10) + \
#             torch.lgamma(target + 1)
#     return temp.mean()
# 
# mean_nll_list = []
# for net in model_list:
#     outs, _ = net(test_inputs.to(device))
#             # torch.from_numpy(test_inputs).type(torch.float).to(device))
#     outs = outs.cpu()
#     outs = np.stack([out.detach().numpy() for out in outs])
#     temp_labels = test_labels.reshape(outs.shape).cpu()
#     mean_nll = mean_poisson_nll(
#             torch.from_numpy(outs), 
#             temp_labels).item()
#     mean_nll_list.append(mean_nll)

# mean_nll_list = cross_val_loss_list.copy()
steps = max([len(x) for x in loss_list])
mean_nll_list = [x[steps-1] for x in cross_val_loss_list] 
fig, ax = plt.subplots()
model_names = np.array([f'hidden_{x[0]}_loss_{x[1]}_{i}' for \
        i, x in enumerate(param_product)])
sort_inds = np.argsort(mean_nll_list)
ax.bar(model_names[sort_inds], np.array(mean_nll_list)[sort_inds])
# ax.bar(model_names, mean_nll_list)
ax.set_title('Mean NLL for each model')
ax.set_xticklabels(model_names[sort_inds], rotation = 90)
fig.savefig(os.path.join(plot_dir,'cross_val_mean_nll.png'),
            bbox_inches = 'tight')
plt.close(fig)

# Pick best model
# net = model_list[np.argmin([loss[-1] for loss in loss_list])]
net = model_list[np.argmin(mean_nll_list)]
best_model_name = model_names[np.argmin(mean_nll_list)]

outs, latent_outs = net(
        torch.from_numpy(inputs_plus_context).type(torch.float).to(device))
# outs = net(
#         torch.from_numpy(inputs_plus_context).type(torch.float).to(device))
outs = outs.cpu()
latent_outs = latent_outs.cpu()
outs = np.stack([out.detach().numpy() for out in outs])
latent_outs = np.stack([out.detach().numpy() for out in latent_outs])

pred_firing = np.moveaxis(outs, 0, -1)

##############################
# Compare outs to inputs_trial_pca

fig, ax = plt.subplots()
ax.scatter(outs.flatten()[::10], inputs_plus_context[...,:-1].flatten()[::10], alpha = 0.1)
fig.savefig(os.path.join(plot_dir, 'actual_vs_pred_scatter_pca.png'))
plt.close(fig)

vmin = min(outs.min(), inputs_plus_context[...,:-1].min())
vmax = max(outs.max(), inputs_plus_context[...,:-1].max())
img_kwargs = {'aspect':'auto', 'interpolation':'none', 'cmap':'viridis',
              'vmin':-2, 'vmax':3}
fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
ax[0].imshow(inputs_plus_context.mean(axis = 1).T, **img_kwargs)
ax[0].set_title('Mean Inputs')
im = ax[1].imshow(pred_firing.mean(axis = 0), **img_kwargs) 
ax[1].set_title('Mean Predicted Firing')
cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax, label = 'Firing Rate (Hz)')
plt.savefig(os.path.join(plot_dir, 'mean_pred_firing_inputs.png'),
            bbox_inches = 'tight')
plt.close()

train_inputs_np = train_inputs.cpu().detach().numpy()
train_labels_np = train_labels.cpu().detach().numpy()
vmin = min(train_inputs_np.min(), train_labels_np.min())
vmax = max(train_inputs_np.max(), train_labels_np.max())

img_kwargs = {'aspect':'auto', 'interpolation':'none', 'cmap':'viridis',
              'vmin':-2, 'vmax':3}
fig, ax = plt.subplots(1,2)
ax[0].imshow(train_inputs_np.mean(axis = 1).T, **img_kwargs) 
ax[0].set_title('Mean Inputs')
im = ax[1].imshow(train_labels_np.mean(axis = 1).T, **img_kwargs) 
ax[1].set_title('Mean Labels')
cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax, label = 'Firing Rate (Hz)')
fig.savefig(os.path.join(plot_dir, 'mean_inputs_labels.png'),
            bbox_inches = 'tight')
plt.close(fig)

##############################

##############################
# Convert back into neuron space
pred_firing = np.moveaxis(pred_firing, 0, -1).T
pred_firing_long = pred_firing.reshape(-1, pred_firing.shape[-1])

# # Reverse NMF scaling
# pred_firing_long = nmf_scaler.inverse_transform(pred_firing_long)
pred_firing_long = pca_scaler.inverse_transform(pred_firing_long)

# Reverse NMF transform
# pred_firing_long = nmf_obj.inverse_transform(pred_firing_long)
pred_firing_long = pca_obj.inverse_transform(pred_firing_long)

# Reverse standard scaling
pred_firing_long = scaler.inverse_transform(pred_firing_long)

pred_firing = pred_firing_long.reshape((*pred_firing.shape[:2], -1))
pred_firing = np.moveaxis(pred_firing, 1,2)

##############################

vz.firing_overview(pred_firing.swapaxes(0,1))
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'firing_pred.png'))
plt.close(fig)
vz.firing_overview(binned_spikes.swapaxes(0,1))
fig = plt.gcf()
fig.savefig(os.path.join(plot_dir, 'firing_actual.png'))
plt.close(fig)
# plt.show()

fig, ax = plt.subplots()
ax.scatter(pred_firing.flatten()[::10], binned_spikes[...,1:].flatten()[::10],
            alpha = 0.1)
fig.savefig(os.path.join(plot_dir, 'actual_vs_pred_scatter.png'))
plt.close(fig)
# plt.show()

fig, ax = plt.subplots(latent_outs.shape[-1], 1, figsize = (5,10))
for i in range(latent_outs.shape[-1]):
    ax[i].imshow(latent_outs[...,i].T, aspect = 'auto')
fig.savefig(os.path.join(plot_dir, 'latent_factors.png'))
plt.close(fig)
# plt.show()

pred_firing_mean = pred_firing.mean(axis = 0)
binned_spikes_mean = binned_spikes.mean(axis = 0)

fig, ax = plt.subplots(1,2)
ax[0].imshow(pred_firing_mean, aspect = 'auto', interpolation = 'none')
ax[1].imshow(binned_spikes_mean, aspect = 'auto', interpolation = 'none')
ax[0].set_title('Pred')
ax[1].set_title('True')
fig.savefig(os.path.join(plot_dir, 'mean_firing.png'))
plt.close(fig)
# plt.show()

fig, ax = plt.subplots(1,2)
ax[0].imshow(zscore(pred_firing_mean,axis=-1), aspect = 'auto', interpolation = 'none')
ax[1].imshow(zscore(binned_spikes_mean,axis=-1), aspect = 'auto', interpolation = 'none')
ax[0].set_title('Pred')
ax[1].set_title('True')
fig.savefig(os.path.join(plot_dir, 'zscore_mean_firing.png'))
plt.close(fig)

fig, ax = vz.gen_square_subplots(len(pred_firing_mean),
                                 figsize = (10,10),
                                 sharex = True,)
for i in range(len(pred_firing)):
    ax.flatten()[i].plot(pred_firing_mean[i], 
                         alpha = 0.7, label = 'pred')
    ax.flatten()[i].plot(binned_spikes_mean[i], 
                         alpha = 0.7, label = 'true')
    ax.flatten()[i].set_ylabel(str(i))
fig.savefig(os.path.join(plot_dir, 'mean_neuron_firing.png'))
plt.close(fig)

# For every neuron, plot 1) spike raster, 2) convolved firing rate , 
# 3) RNN predicted firing rate
ind_plot_dir = os.path.join(plot_dir, 'individual_neurons')
if not os.path.exists(ind_plot_dir):
    os.makedirs(ind_plot_dir)

binned_x = np.arange(0, binned_spikes.shape[-1]*bin_size, bin_size)

conv_kern = np.ones(250) / 250
conv_rate = np.apply_along_axis(
        lambda m: np.convolve(m, conv_kern, mode = 'valid'),
                            axis = -1, arr = cat_spikes)*bin_size
conv_x = np.convolve(
        np.arange(cat_spikes.shape[-1]), conv_kern, mode = 'valid')

for i in range(binned_spikes.shape[1]):
    fig, ax = plt.subplots(3,1, figsize = (10,10),
                           sharex = True, sharey = False)
    ax[0] = vz.raster(ax[0], cat_spikes[:, i], marker = '|')
    ax[1].plot(conv_x, conv_rate[:,i].T, c = 'k', alpha = 0.1)
    # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
    ax[2].plot(binned_x[1:], pred_firing[:,i].T, 
               c = 'k', alpha = 0.1)
    # ax[2].sharey(ax[1])
    for this_ax in ax:
        this_ax.set_xlim([1500, 4000])
        this_ax.axvline(2000, c = 'r', linestyle = '--')
    ax[1].set_title(f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
    ax[2].set_title('RNN Predicted Firing Rate')
    fig.savefig(
            os.path.join(ind_plot_dir, f'neuron_{i}_raster_conv_pred.png'))
    plt.close(fig)

# Make another plot with taste_mean firing rates
cmap = plt.get_cmap('tab10')
for i in range(binned_spikes.shape[1]):
    fig, ax = plt.subplots(3,1, figsize = (10,10),
                           sharex = True, sharey = False)
    ax[0] = vz.raster(ax[0], cat_spikes[:, i], marker = '|', color = 'k')
    # Plot colors behind raster traces
    for j in range(4):
        ax[0].axhspan(len(cat_spikes)*j/4, len(cat_spikes)*(j+1)/4,
                      color = cmap(j), alpha = 0.1, zorder = 0)
    this_conv_rate = conv_rate[:,i]
    this_pred_firing = pred_firing[:,i]
    this_conv_rate = np.stack(np.split(this_conv_rate, 4))
    this_pred_firing = np.stack(np.split(this_pred_firing, 4))
    mean_conv_rate = this_conv_rate.mean(axis = 1)
    mean_pred_firing = this_pred_firing.mean(axis = 1)
    sd_conv_rate = this_conv_rate.std(axis = 1)
    sd_pred_firing = this_pred_firing.std(axis = 1)
    for j in range(mean_conv_rate.shape[0]):
        ax[1].plot(conv_x, mean_conv_rate[j].T, c = cmap(j),
                   linewidth = 2)
        ax[1].fill_between(
                conv_x, 
                mean_conv_rate[j] - sd_conv_rate[j],
                mean_conv_rate[j] + sd_conv_rate[j],
                color = cmap(j), alpha = 0.1)
        # ax[2].plot(binned_x, binned_spikes[:,i].T, label = 'True')
        ax[2].plot(binned_x[1:], mean_pred_firing[j].T,
                   c = cmap(j), linewidth = 2)
        ax[2].fill_between(
                binned_x[1:], 
                mean_pred_firing[j] - sd_pred_firing[j],
                mean_pred_firing[j] + sd_pred_firing[j],
                color = cmap(j), alpha = 0.1)
        # ax[2].sharey(ax[1])
    for this_ax in ax:
        this_ax.set_xlim([1500, 4000])
        this_ax.axvline(2000, c = 'r', linestyle = '--')
    ax[1].set_title(f'Convolved Firing Rate : Kernel Size {len(conv_kern)}')
    ax[2].set_title('RNN Predicted Firing Rate')
    fig.savefig(
            os.path.join(
                ind_plot_dir, 
                f'neuron_{i}_mean_raster_conv_pred.png'))
    plt.close(fig)

# Plot single-trial latent factors
trial_latent_dir = os.path.join(plot_dir, 'trial_latent')
if not os.path.exists(trial_latent_dir):
    os.makedirs(trial_latent_dir)

for i in range(latent_outs.shape[1]):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(latent_outs[1:,i], alpha = 0.5)
    ax[0].set_title(f'Latent factors for trial {i}')
    ax[1].plot(zscore(latent_outs[1:,i], axis = 0), alpha = 0.5)
    fig.savefig(os.path.join(trial_latent_dir, f'trial_{i}_latent.png'))
    plt.close(fig)

# Plot predicted activity vs true activity for every neuron

for i in range(pred_firing.shape[1]):
    fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
    min_val = min(pred_firing[:,i].min(), binned_spikes[:,i].min())
    max_val = max(pred_firing[:,i].max(), binned_spikes[:,i].max())
    img_kwargs = {'aspect':'auto', 'interpolation':'none', 'cmap':'viridis',
                  }
                  #'vmin':min_val, 'vmax':max_val}
    im0 = ax[0].imshow(pred_firing[:,i], **img_kwargs) 
    im1 = ax[1].imshow(binned_spikes[:,i, 1:], **img_kwargs) 
    ax[0].set_title('Pred')
    ax[1].set_title('True')
    # Colorbars under each subplot
    cbar0 = fig.colorbar(im0, ax = ax[0], orientation = 'horizontal')
    cbar1 = fig.colorbar(im1, ax = ax[1], orientation = 'horizontal')
    cbar0.set_label('Firing Rate (Hz)')
    cbar1.set_label('Firing Rate (Hz)')
    fig.savefig(os.path.join(ind_plot_dir, f'neuron_{i}_firing.png'))
    plt.close(fig)

##############################

# Reduce latent outs to 3D and plot
latent_pca_obj = PCA(n_components=3)
latent_outs_long = latent_outs.reshape(-1, latent_outs.shape[-1])
latent_outs_long = zscore(latent_outs_long, axis = 0)
latent_outs_3d = latent_pca_obj.fit_transform(latent_outs_long)
latent_outs_trials_3d = latent_outs_3d.reshape(latent_outs.shape[0],-1,3)
mean_latent_outs_3d = latent_outs_trials_3d.mean(axis = 1)

fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
ax[0].imshow(latent_outs.mean(axis = 1).T, aspect = 'auto', interpolation = 'none')
ax[1].imshow(zscore(latent_outs.mean(axis = 1).T, axis = -1), aspect = 'auto', interpolation = 'none')
ax[0].set_title('Mean')
ax[1].set_title('Zscored Mean')
fig.savefig(os.path.join(plot_dir, 'latent_mean.png'))
plt.close(fig)

taste_latents_3d = np.stack(np.split(latent_outs_trials_3d, 4, axis = 1))
mean_taste_latents_3d = taste_latents_3d.mean(axis = 2)

cmap = plt.get_cmap('viridis')
cat_cmap = plt.get_cmap('tab10')
fig = plt.figure(figsize = (15,5))
ax = fig.add_subplot(121, projection='3d')
# im = ax.scatter(mean_latent_outs_3d[:,0], mean_latent_outs_3d[:,1], mean_latent_outs_3d[:,2], 
#         c = np.arange(mean_latent_outs_3d.shape[0]), cmap = cmap)
ax.plot(mean_latent_outs_3d[:,0], mean_latent_outs_3d[:,1], mean_latent_outs_3d[:,2],
        c = 'k')
for i, this_taste in enumerate(mean_taste_latents_3d):
    ax.plot(this_taste[:,0], this_taste[:,1], this_taste[:,2], c = cat_cmap(i))
    for this_marker in marker_inds:
        ax.scatter(this_taste[this_marker,0], 
                   this_taste[this_marker,1], 
                   this_taste[this_marker,2], c = 'r', s = 100)
# Plot markers at t= 2000, 4000,
marker_times = [2000, 4000]
marker_inds = [x//bin_size for x in marker_times]
ax.scatter(mean_latent_outs_3d[marker_inds,0],
           mean_latent_outs_3d[marker_inds,1],
           mean_latent_outs_3d[marker_inds,2],
           c = 'r', s = 100, zorder = 10)
# fig.colorbar(im, label = 'Time')
# Also plot a 2D projection
ax = fig.add_subplot(122)
ax.scatter(mean_latent_outs_3d[:,0], mean_latent_outs_3d[:,1], 
           c = np.arange(mean_latent_outs_3d.shape[0]), cmap = cmap)
ax.plot(mean_latent_outs_3d[:,0], mean_latent_outs_3d[:,1], c = 'k')
ax.scatter(mean_latent_outs_3d[marker_inds,0],
           mean_latent_outs_3d[marker_inds,1],
           c = 'r', s = 100, zorder = 10)
fig.suptitle('Mean Latent Factors\n' + f'Marker times: {marker_times}')
fig.savefig(os.path.join(plot_dir, 'latent_3d.png'))
plt.close(fig)

# Compare 3D projection of binned spikes and latent factors
# binned_pca_obj = PCA(n_components=3)
# binned_spikes_reshape = binned_spikes.swapaxes(1,2)
# binned_spikes_3d = binned_pca_obj.fit_transform(
#         binned_spikes_reshape.reshape(-1, binned_spikes_reshape.shape[-1]))
# binned_spikes_trials_3d = binned_spikes_3d.reshape(binned_spikes.shape[0],-1,3)
# mean_binned_spikes_3d = binned_spikes_trials_3d.mean(axis = 0)[1:]

# Project conv_rate down to 3D
conv_pca_obj = PCA(n_components=3)
conv_rate_reshape = conv_rate.swapaxes(1,2)
# Zscore each neuron separately
conv_rate_reshape = np.stack([zscore(x, axis=None) for x in conv_rate_reshape.T]).T

conv_rate_3d = conv_pca_obj.fit_transform(
        conv_rate_reshape.reshape(-1, conv_rate_reshape.shape[-1]))
conv_rate_trials_3d = conv_rate_3d.reshape(conv_rate.shape[0],-1,3)
mean_conv_rate_3d = conv_rate_trials_3d.mean(axis = 0)[1:]


# Project latent factors onto binned spikes PCA
from sklearn.linear_model import LinearRegression
# Interpolate mean_latent_outs_3d to match conv_x 
from scipy.interpolate import interp1d
f = interp1d(binned_x[1:], mean_latent_outs_3d, axis = 0)
interp_mean_latent_outs_3d = f(conv_x[:-1])

reg = LinearRegression().fit(interp_mean_latent_outs_3d, mean_conv_rate_3d)
latent_proj = reg.predict(mean_latent_outs_3d)
latent_proj_trials = np.stack([reg.predict(x) for x in latent_outs_trials_3d.swapaxes(0,1)])

# Plot each projection separately, and then aligned
fig = plt.figure(figsize = (15,5))
ax0 = fig.add_subplot(131, projection='3d')
ax0.plot(mean_conv_rate_3d[:,0], mean_conv_rate_3d[:,1], mean_conv_rate_3d[:,2],
        c = 'k')
ax1 = fig.add_subplot(132, projection='3d')
ax1.plot(mean_latent_outs_3d[:,0], mean_latent_outs_3d[:,1], mean_latent_outs_3d[:,2],
        c = 'k')
ax2 = fig.add_subplot(133, projection='3d')
ax2.plot(latent_proj[:,0], latent_proj[:,1], latent_proj[:,2],
        c = 'b', label = 'Latent', alpha = 0.5, linewidth = 3)
ax2.plot(mean_conv_rate_3d[:,0], mean_conv_rate_3d[:,1], mean_conv_rate_3d[:,2],
        c = 'k', label = 'Conv', alpha = 0.5, linewidth = 3)
ax2.legend()
fig.suptitle('3D Projections of Conv Rate and Latent Factors')
fig.savefig(os.path.join(plot_dir, 'conv_latent_3d.png'))
plt.close(fig)

# Recreate above plot for 2D projections
fig = plt.figure(figsize = (15,5))
ax0 = fig.add_subplot(131)
ax0.plot(mean_conv_rate_3d[:,0], mean_conv_rate_3d[:,1],
        c = 'k')
ax1 = fig.add_subplot(132)
ax1.plot(mean_latent_outs_3d[:,0], mean_latent_outs_3d[:,1],
        c = 'k')
ax2 = fig.add_subplot(133)
ax2.plot(latent_proj[:,0], latent_proj[:,1],
        c = 'b', label = 'Latent', alpha = 0.5, linewidth = 3)
ax2.plot(mean_conv_rate_3d[:,0], mean_conv_rate_3d[:,1],
        c = 'k', label = 'Conv', alpha = 0.5, linewidth = 3)
ax2.legend()
fig.suptitle('2D Projections of Conv and Latent Factors')
fig.savefig(os.path.join(plot_dir, 'conv_latent_2d.png'))
plt.close(fig)

# Plot some examples of single trial 2D projections

plot_n = 16
trial_inds = np.random.choice(np.arange(latent_outs_trials_3d.shape[1]), plot_n)
fig, ax = vz.gen_square_subplots(plot_n, figsize = (20,20))
for i, this_ax in enumerate(ax.flatten()):
    this_ind = trial_inds[i]
    this_ax.plot(conv_rate_trials_3d[this_ind,:,0], conv_rate_trials_3d[this_ind,:,1],
            c = 'k', alpha = 0.5, linewidth = 3, label = 'Conv')
    this_ax.plot(latent_proj_trials[this_ind,:,0], latent_proj_trials[this_ind,:,1],
            c = 'b', alpha = 0.5, linewidth = 3, label = 'Latent')
    this_ax.set_title(f'Trial {trial_inds[i]}')
ax.flatten()[-1].legend()
fig.suptitle('Conv and Latent 2D Projections')
fig.savefig(os.path.join(plot_dir, 'conv_latent_2d_trials.png'))
plt.close(fig)


############################################################
# Compare RNN predicted firing to convolved firing rate
# in prediction of binned spikes

fin_bin_inds = np.logical_and(
        binned_x > min(conv_x),
        binned_x < max(conv_x))

fin_binned_x = binned_x[fin_bin_inds]
fin_binned_spikes = binned_spikes[...,fin_bin_inds]
# Move by 1 to align with pred_firing
fin_binned_x = fin_binned_x[1:]
fin_binned_spikes = fin_binned_spikes[...,1:]

# Do same for pred_firing
fin_pred_firing = pred_firing[...,fin_bin_inds[1:]]
fin_pred_firing = fin_pred_firing[...,1:]

# Get convolved firing rate for times closes to binned spikes
wanted_conv_inds = np.array([np.argmin(np.abs(conv_x - x)) for x in fin_binned_x])

fin_conv_x = conv_x[wanted_conv_inds]
fin_conv_rate = conv_rate[...,wanted_conv_inds]

# Plot averages of all 3 to compare
fig, ax = plt.subplots(1,3, figsize = (15,5))
ax[0].imshow(fin_binned_spikes.mean(axis = 1), aspect = 'auto', interpolation = 'none')
ax[0].set_title('Binned Spikes')
ax[1].imshow(fin_conv_rate.mean(axis = 1), aspect = 'auto', interpolation = 'none')
ax[1].set_title('Convolved Firing Rate')
ax[2].imshow(fin_pred_firing.mean(axis = 1), aspect = 'auto', interpolation = 'none')
ax[2].set_title('RNN Predicted Firing Rate')
fig.suptitle('Comparison of Binned Spikes, Convolved Firing Rate, RNN Predicted Firing Rate')
fig.savefig(os.path.join(plot_dir, 'binned_conv_rnn_comparison.png'))
plt.close(fig)

# Also make line plots of each neuron
fig, ax = vz.gen_square_subplots(fin_binned_spikes.shape[1], figsize = (10,10))
for i, this_ax in enumerate(ax.flatten()):
    this_ax.plot(fin_binned_x, fin_binned_spikes[:,i].mean(axis=0), 
                 label = 'True', alpha = 0.5)
    this_ax.plot(fin_conv_x, fin_conv_rate[:,i].mean(axis=0), 
                 label = 'Conv', alpha = 0.8, c = 'r')
    this_ax.plot(fin_binned_x, fin_pred_firing[:,i].mean(axis=0), 
                 label = 'RNN', alpha = 0.8, c = 'k')
    this_ax.set_title(f'Neuron {i}')
ax[0,-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
fig.suptitle('Comparison of Binned Spikes, Convolved Firing Rate, RNN Predicted Firing Rate')
fig.savefig(os.path.join(plot_dir, 'binned_conv_rnn_comparison_line.png'))
plt.close(fig)

# For each neuron, calculate r2 between
# 1) convolved firing rate and binned spikes
# 2) RNN predicted firing rate and binned spikes

conv_r2 = []
rnn_r2 = []
for i in range(fin_binned_spikes.shape[1]):
    conv_r2.append(r2_score(fin_binned_spikes[:,i], fin_conv_rate[:,i]))
    rnn_r2.append(r2_score(fin_binned_spikes[:,i], fin_pred_firing[:,i]))

fig, ax = plt.subplots()
ax.scatter(conv_r2, rnn_r2)
ax.set_xlabel('Conv R2')
ax.set_ylabel('RNN R2')
ax.plot([-1,1],[-1,1], c = 'k', linestyle = '--')
fig.suptitle('R2 between conv, rnn and binned spikes')
fig.savefig(os.path.join(plot_dir, 'conv_rnn_r2.png'))
plt.close(fig)

