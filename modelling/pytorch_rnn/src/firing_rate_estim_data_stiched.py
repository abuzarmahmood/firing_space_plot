"""
Train a firing rate estimator on the stiched data.

Encoder and Decoder are trained separately for each session.
LSTM is shared across sessions.
"""
############################################################
# Imports
############################################################

import time
import numpy as np
import pylab as plt
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.nn import functional as F
import math
from scipy.stats import poisson, zscore

import sys
src_dir = '/media/bigdata/firing_space_plot/modelling/pytorch_rnn/src'
sys.path.append(src_dir)
from model import autoencoderRNN

############################################################
# Define Model 
############################################################
def train_model_single(
        net, 
        inputs, 
        labels, 
        train_steps = 1000, 
        lr=0.01, 
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
    output_size = labels.shape[-1]

    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    cross_val_bool = np.logical_and(
            test_inputs is not None, 
            test_labels is not None
            )

    loss_history = []
    cross_val_loss = {}
    running_loss = 0
    running_acc = 0
    start_time = time.time()
    labels = labels.reshape(-1, output_size)
    # Loop over training batches
    print('Training network...')
    for i in range(train_steps):

        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
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

def train_model(
        net,
        train_inputs_list, 
        train_labels_list, 
        train_steps = 1000, 
        lr=0.01, 
        delta_loss = 0.01,
        device = None,
        test_inputs_list = None,
        test_labels_list = None,
        ):
    """
    For each session, train the network on the inputs and labels
    Pick session randomly and train for steps_per_session steps

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
        inputs: shape (session, seq_len, batch, input_size)
        labels: shape (session, seq_len * batch, output_size)
        test_inputs: shape (session, seq_len, batch, input_size)
        test_labels: shape (session, seq_len * batch, output_size)

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    cross_val_bool = np.logical_and(
            test_inputs_list is not None, 
            test_labels_list is not None
            )

    n_sessions = len(train_inputs_list)
    session_inds = np.random.choice(
            np.arange(n_sessions),
            train_steps, 
            replace = True,
            )

    loss_history = []
    cross_val_loss = {}
    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(train_steps):
        session = session_inds[i]
        session_inputs = train_inputs_list[session]
        session_labels = train_labels_list[session]
        session_test_inputs = test_inputs_list[session]
        session_test_labels = test_labels_list[session]
        this_output_size = session_labels.shape[-1]

        session_labels = session_labels.reshape(-1, this_output_size)

        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(session_inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.reshape(-1, this_output_size)
        loss = criterion(output, session_labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Only compute cross_val_loss every 100 steps
        # because it's expensive
        if cross_val_bool and (i % 100 == 0): 
            test_out, _ = net(session_test_inputs)
            test_out = test_out.reshape(-1, this_output_size)
            session_test_labels = session_test_labels.reshape(-1, this_output_size)
            test_loss = criterion(test_out, session_test_labels)
            # cross_val_loss.append(test_loss.item())
            cross_val_loss[i] = test_loss.item()
            cross_str = f'Cross Val Loss: {test_loss.item():0.4f}'
        else:
            cross_str = ''

        # Compute the running loss every 100 steps
        current_loss = loss.item()
        loss_history.append(current_loss)
        running_loss += current_loss 
        if i % 100 == 0: 
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, {}, Time {:0.1f}s'.format(
                i, running_loss, cross_str, time.time() - start_time))
            running_loss = 0

    return net, loss_history, cross_val_loss, session_inds

############################################################
# Load data
############################################################

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

data_list_path = '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
data_list = np.loadtxt(data_list_path, dtype = str)

spike_train_list = []
for data_dir in data_list:
    try:
        dat = ephys_data(data_dir)
        dat.firing_rate_params = dat.default_firing_params
        dat.get_spikes()
        dat.get_firing_rates()
        spike_array = np.stack(dat.spikes)
        cat_spikes = np.concatenate(spike_array)
        spike_train_list.append(cat_spikes)
    except:
        print(f'Error with {data_dir}')
        continue

bin_size = 25
binned_spikes_list = []
for this_spikes in spike_train_list:
    binned_spikes = np.reshape(this_spikes, 
                               (*this_spikes.shape[:2], -1, bin_size)).sum(-1)
    binned_spikes_list.append(binned_spikes)

min_neuron_num = np.min([x.shape[1] for x in binned_spikes_list])

##############################
# Project all datasets to same PCA space
zscored_binned_spikes = [np.stack(
    [
        zscore(x[:,i], axis=None) for i in range(x.shape[1])
        ], 
    axis = 1) 
                         for x in binned_spikes_list
                         ]

zscored_spikes_all = np.concatenate(zscored_binned_spikes, axis = 1).swapaxes(1,2)
zscored_spikes_all_long = np.reshape(zscored_spikes_all, (-1, zscored_spikes_all.shape[-1]))
all_spikes_pca_obj = PCA(n_components = min_neuron_num)
all_spikes_pca = all_spikes_pca_obj.fit_transform(zscored_spikes_all_long)

# Project individual datasets to all_spikes_pca space
pca_spikes_list = []
for i in range(len(zscored_binned_spikes)): 
    this_zscored_spikes = zscored_binned_spikes[i].swapaxes(1,2)
    this_zscored_spikes_long = np.reshape(this_zscored_spikes, (-1, this_zscored_spikes.shape[-1]))
    lm = LinearRegression().fit(this_zscored_spikes_long, all_spikes_pca)
    pca_spikes = lm.predict(this_zscored_spikes_long)
    pca_spikes_list.append(pca_spikes)

# Reshape back to (trial, time, neuron)
pca_spikes_list = [
        np.reshape(x, (-1, zscored_binned_spikes[0].shape[-1], min_neuron_num)) \
                for x in pca_spikes_list]
fin_pca_data = np.stack(pca_spikes_list)

n_test_datasets = 3
test_dataset_inds = np.random.choice(
        np.arange(len(pca_spikes_list)),
        n_test_datasets,
        replace = False)
train_dataset_inds = np.setdiff1d(
        np.arange(len(pca_spikes_list)),
        test_dataset_inds)

test_pca_data = fin_pca_data[test_dataset_inds]
fin_pca_data = fin_pca_data[train_dataset_inds]

# mean_pca_spikes = [np.mean(x, axis = 0) for x in pca_spikes_list]

# # Plot 3D PCA for all datasets
# n_rows = int(np.ceil(np.sqrt(len(pca_spikes_list))))
# n_cols = int(np.ceil(len(pca_spikes_list) / n_rows))
# 
# fig, ax = plt.subplots(n_rows, n_cols, subplot_kw = {'projection': '3d'})
# ax = ax.ravel()
# for i in range(len(pca_spikes_list)):
#     ax[i].plot(mean_pca_spikes[i][:,0], mean_pca_spikes[i][:,1], mean_pca_spikes[i][:,2])
#     ax[i].set_title(f'Dataset {i}, nrns: {zscored_binned_spikes[i].shape[1]}')
#     ax[i].set_xlabel('PC1')
#     ax[i].set_ylabel('PC2')
#     ax[i].set_zlabel('PC3')
# plt.show()
# 
# # Repeat with 2D PCA
# fig, ax = plt.subplots(n_rows, n_cols, sharex = True, sharey = True)
# ax = ax.ravel()
# for i in range(len(pca_spikes_list)):
#     ax[i].plot(mean_pca_spikes[i][:,0], mean_pca_spikes[i][:,1])
#     ax[i].set_title(f'Dataset {i}, nrns: {zscored_binned_spikes[i].shape[1]}')
#     ax[i].set_xlabel('PC1')
#     ax[i].set_ylabel('PC2')
# plt.show()
# 
# # Plot heatmaps
# fig, ax = vz.gen_square_subplots(len(pca_spikes_list))
# ax = ax.ravel()
# for i in range(len(pca_spikes_list)):
#     ax[i].imshow(mean_pca_spikes[i].T, aspect = 'auto', interpolation = 'none')
#     ax[i].set_title(f'Dataset {i}, nrns: {zscored_binned_spikes[i].shape[1]}')
# plt.show()


##############################

# Make sure shapes are consistent
assert len(set([x.shape[0] for x in binned_spikes_list])) == 1

# Add taste number as external input
# Add stim time as external input
stim_time = np.zeros_like(binned_spikes_list[0])
stim_time[:, :, 2000//bin_size] = 1
stim_time = stim_time[:,0]
stim_time = np.moveaxis(stim_time, -1, 0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reshape to (seq_len, batch, input_size)
# temp_inputs = binned_spikes.copy()
temp_inputs = fin_pca_data.copy()
temp_inputs = np.moveaxis(temp_inputs, 2, 1)

stim_time_broad = np.broadcast_to(
            stim_time[None,:,:,None],
            temp_inputs.shape)

inputs_plus_context = np.concatenate(
        [
            temp_inputs, 
            stim_time_broad[...,0][:,:,:,None]
            ], 
        axis = -1)

# Instead of predicting activity in the SAME time-bin,
# predict activity in the NEXT time-bin
# Hoping that this will make the model more robust to
# small fluctuations in activity
inputs = inputs_plus_context[:,:-1]
labels = temp_inputs[:, 1:]

# (seq_len * batch, output_size)
labels = torch.from_numpy(labels).type(torch.float32)
# (seq_len, batch, input_size)
inputs = torch.from_numpy(inputs).type(torch.float)

# Split into train and test
train_test_split = 0.75
train_inds = np.random.choice(
        np.arange(inputs.shape[2]), 
        int(train_test_split * inputs.shape[2]), 
        replace = False)
test_inds = np.setdiff1d(np.arange(inputs.shape[2]), train_inds)

train_inputs = inputs[:, :, train_inds]
train_labels = labels[:, :, train_inds]
test_inputs = inputs[:, :, test_inds]
test_labels = labels[:, :, test_inds]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_inputs = train_inputs.to(device)
train_labels = train_labels.to(device)
test_inputs = test_inputs.to(device)
test_labels = test_labels.to(device)

############################################################
# Train Model
############################################################
input_size = train_inputs.shape[-1]

hidden_size_vec = [6, 12, 18]
repeats = 3

param_vec = np.repeat(hidden_size_vec, repeats)

model_list = []
loss_list = []
cross_val_loss_list = []
for i, this_params in enumerate(tqdm(param_vec)):

    net = autoencoderRNN( 
            input_size=input_size,
            hidden_size=int(this_params), 
            output_size=input_size-1,
            dropout = 0.2,
            )
    net.to(device)

    net, loss, cross_val_loss, session_inds = train_model(
            net,
            train_inputs, 
            train_labels, 
            lr = 0.001, 
            train_steps = 60000,
            test_inputs_list = test_inputs,
            test_labels_list = test_labels,
            )

    model_list.append(net)
    loss_list.append(loss)
    cross_val_loss_list.append(cross_val_loss)

    ############################## 
    # Plot loss
    ##############################
    fig, ax = plt.subplots()
    for ind, loss in enumerate(loss_list):
        ax.plot(loss, label = f'params: {param_vec[ind]}',
                alpha = 0.5) 
    ax.legend(
            bbox_to_anchor=(1.05, 1), 
            loc='upper left', borderaxespad=0.)
    ax.set_title(f'Losses for {i+1}/{len(param_vec)} restarts')
    fig.savefig(os.path.join(plot_dir,'run_loss.png'),
                bbox_inches = 'tight')
    plt.close(fig)

    # Plot another figure of cross_val_loss
    fig, ax = plt.subplots()
    for ind, loss in enumerate(cross_val_loss_list):
        ax.plot(loss.keys(), loss.values(),
                label = f'params: {param_vec[ind]}',
                alpha = 0.5)
    ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left', borderaxespad=0.)
    ax.set_title(f'Cross Val Losses for {i+1}/{len(param_vec)} restarts')
    fig.savefig(os.path.join(plot_dir,'cross_val_loss.png'),
                bbox_inches = 'tight')
    plt.close(fig)

# Check which model has the lowest cross_val_loss
min_loss_ind = np.argmin([np.mean(np.array(list(x.values()))[-1000:]) \
        for x in cross_val_loss_list])

net = model_list[min_loss_ind]
pred_firing = np.stack([net(x.to(device))[0].cpu().detach().numpy() for x in inputs])
inputs_numpy = inputs.cpu().detach().numpy()

mean_inputs = np.mean(inputs_numpy, axis = 2)[:,:-1]
mean_pred_firing = np.mean(pred_firing, axis = 2)

fig, ax = plt.subplots(len(mean_inputs), 2,
                       sharex = True, sharey = True,
                       figsize = (5,10))
for i in range(len(mean_inputs)):
    ax[i, 0].imshow(mean_inputs[i].T, aspect = 'auto', interpolation = 'none')
    ax[i, 1].imshow(mean_pred_firing[i].T, aspect = 'auto', interpolation = 'none')
ax[0, 0].set_title('Data')
ax[0, 1].set_title('Pred')
fig.suptitle('Mean Data and Pred')
fig.savefig(os.path.join(plot_dir, 'mean_data_pred.png'),
            bbox_inches = 'tight', dpi = 200)
plt.close(fig)

############################################################
# Use RNN for predicting held-out data 
############################################################

class shared_autoencoderRNN(nn.Module):
    """
    Input and output transformations are encoder and decoder architectures
    RNN will learn dynamics of latent space

    Output has to be rectified
    Can add dropout to RNN and autoencoder layers
    """
    def __init__(
            self, 
            input_size, 
            output_size, 
            rnn, 
            hidden_size,
            dropout = 0.2,
            ):
        """
        3 sigmoid layers for input and output each, to project between:
            encoder : input -> latent
            rnn : latent -> latent
            decoder : latent -> output
        """
        super(shared_autoencoderRNN, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size, sum((input_size, hidden_size))//2),
                nn.Sigmoid(),
                nn.Linear(sum((input_size, hidden_size))//2, hidden_size),
                nn.Sigmoid(),
                )
        self.decoder = nn.Sequential(
                nn.Linear(hidden_size, sum((hidden_size, output_size))//2),
                nn.Sigmoid(),
                nn.Linear(sum((hidden_size, output_size))//2, output_size),
                )
        self.en_dropout = nn.Dropout(p = dropout)
        self.rnn = rnn
        for param in self.rnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.encoder(x)
        out = self.en_dropout(out)
        latent_out, _ = self.rnn(out)
        out = self.decoder(latent_out)
        return out, latent_out

from copy import deepcopy
wanted_rnn = deepcopy(net.rnn)

test_binned_spikes = [binned_spikes_list[i] for i in test_dataset_inds]
test_ind = 2
this_test_binned = test_binned_spikes[test_ind].swapaxes(1,2)
vz.firing_overview(np.moveaxis(this_test_binned, -1,0))
plt.show()

# Perform scaling, PCA, scaling
this_binned_long = np.reshape(this_test_binned, (-1, this_test_binned.shape[-1]))
scaler = StandardScaler()
this_binned_long = scaler.fit_transform(this_binned_long)

# Get 95% variance PCA
this_pca_obj = PCA(n_components = 0.95)
this_pca = this_pca_obj.fit_transform(this_binned_long)

# # Rescale to give all neurons equal variance
# this_pca_scaler = StandardScaler()
# this_pca = this_pca_scaler.fit_transform(this_pca)

# Reshape back to (trial, time, neuron)
this_pca_trial = np.reshape(
        this_pca, 
        (this_test_binned.shape[0], -1, this_pca.shape[-1]))
this_pca_trial = np.moveaxis(this_pca_trial, 1, 0)


# Define inputs and labels
this_inputs = this_pca_trial.copy()
this_inputs = np.concatenate(
        [
            this_inputs, 
            stim_time_broad[0,...,0][...,None]
            ], 
        axis = -1)
this_inputs = this_inputs[:-1]
this_labels = this_pca_trial[1:]

this_inputs = torch.from_numpy(this_inputs).type(torch.float).to(device)
this_labels = torch.from_numpy(this_labels).type(torch.float).to(device)

# Get network
input_size = this_inputs.shape[-1] 
shared_bool = True
if shared_bool:
    this_shared_net = shared_autoencoderRNN(
            input_size=input_size,
            output_size=input_size-1,
            rnn = wanted_rnn,
            hidden_size = wanted_rnn.hidden_size,
            dropout = 0.2,
            )
    shared_str = 'shared'
else:
    this_shared_net = autoencoderRNN(
            input_size=input_size,
            hidden_size = wanted_rnn.hidden_size,
            output_size=input_size-1,
            dropout = 0.2,
            )
    shared_str = 'unshared'
this_shared_net.to(device)
this_shared_net, loss, cross_val_loss = train_model_single(
        this_shared_net, 
        this_inputs, 
        this_labels, 
        lr = 0.001, 
        train_steps = 15000,
        )

# Get predictions
this_pred = this_shared_net(this_inputs)[0].cpu().detach().numpy()
this_pred_long = np.reshape(this_pred, (-1, this_pred.shape[-1]))

# # Invert scaling
# this_pred_long = this_pca_scaler.inverse_transform(this_pred_long)

# Invert PCA
this_pred_long = this_pca_obj.inverse_transform(this_pred_long)

# Invert scaling
this_pred_long = scaler.inverse_transform(this_pred_long)

# Reshape back to (trial, time, neuron)
this_pred = np.reshape(this_pred_long, (*this_pred.shape[:2], -1)) 

# Plot mean values of binned spikes and pred
plot_binned_spikes = this_test_binned.swapaxes(0,1)

mean_binned = np.mean(plot_binned_spikes, axis = 1)
mean_pred = np.mean(this_pred, axis = 1)

fig, ax = vz.gen_square_subplots(mean_binned.shape[-1])
for i in range(mean_binned.shape[-1]):
    ax.flatten()[i].plot(mean_binned[:,i], label = 'Data', alpha = 0.5)
    ax.flatten()[i].plot(mean_pred[:,i], label = 'Pred', alpha = 0.5)
ax[0,-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.suptitle('Mean Data and Pred')
fig.savefig(os.path.join(plot_dir, f'mean_data_pred_test_{shared_str}.png'),
            bbox_inches = 'tight', dpi = 200)
plt.close(fig)

# Plot mean values of binned spikes and pred per taste
shared_net_out_dir = os.path.join(plot_dir, 'shared_net_out')
if not os.path.exists(shared_net_out_dir):
    os.makedirs(shared_net_out_dir)

shared_ind_dir = os.path.join(shared_net_out_dir, 'individual_plots')
if not os.path.exists(shared_ind_dir):
    os.makedirs(shared_ind_dir)

for i in range(plot_binned_spikes.shape[-1]):
    fig, ax = plt.subplots(1, 2, figsize = (10,10))
    ax[0].imshow(plot_binned_spikes[:,:, i].T, aspect = 'auto', interpolation = 'none')
    ax[1].imshow(this_pred[:,:, i].T, aspect = 'auto', interpolation = 'none')
    ax[0].set_title('Data')
    ax[1].set_title('Pred')
    fig.suptitle(f'Neuron {i}')
    fig.savefig(os.path.join(shared_ind_dir, f'neuron_{i}_{shared_str}.png'),
                bbox_inches = 'tight', dpi = 200)
    plt.close(fig)
