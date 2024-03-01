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

############################################################
# Define Model 
############################################################
# Define networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.nn import functional as F
from model import CTRNN
import math
from scipy.stats import poisson, zscore


class CTRNN_plus_output(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, dt=None):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, dt) 

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Rectify output
        self.relu = nn.ReLU()

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        out = self.relu(out)
        return out , rnn_output

class rnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=None):
        super(rnnModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.relu(out)
        return out


def train_model(net, inputs, labels, train_steps = 1000, lr=0.01):
    """Simple helper function to train the model.

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # criterion = nn.MSELoss()
    criterion = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')

    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(train_steps):
        # Generate input and target, convert to pytorch tensor
        # inputs, labels = dataset()
        this_inputs = inputs#[i]
        this_labels = labels#[i]
        this_inputs = torch.from_numpy(this_inputs).type(torch.float)
        this_labels = torch.from_numpy(this_labels.flatten()).type(torch.float32)
        this_labels = this_labels.view(-1, output_size)
        # this_labels = torch.from_numpy(this_labels).type(torch.float)

        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        # this_output, _ = net(this_inputs)
        this_output = net(this_inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        # this_output = this_output.view(-1, output_size)
        this_output = this_output.reshape(-1, output_size)
        loss = criterion(this_output, this_labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 10 == 9:
            running_loss /= 10
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i+1, running_loss, time.time() - start_time))
            running_loss = 0
    return net

############################################################
# Generate data 
############################################################
n_batches = 50
n_trials = 1000
dt = 0.01
x = np.arange(0, 2, dt) 
# y = np.abs(np.cumsum(np.random.randn(n_batches, len(x), n_trials), axis=0))
# y = np.broadcast_to(x[None,:,None], (n_batches, len(x), n_trials))
y = np.broadcast_to(x[:,None], (len(x), n_trials))
y = y + (np.random.random(n_trials) * 5)[ None,:]
y = np.abs((np.sin(y * 2 * np.pi ) * 0.5)*500)

# kern_len = 25
# kern = np.ones(kern_len) / kern_len
# y = np.apply_along_axis(lambda m: np.convolve(m, kern, mode='valid'), axis=1, arr=y)
# y *= 50
# 
# y = y - np.min(y, axis=1)[:, None]

# spikes = np.random.rand(*y.shape) < y * dt
spikes = poisson.rvs(y * dt)
# spikes[:50] = 0

# plt.scatter(*np.where(spikes[0]))
# plt.show()
# 
# plt.imshow(y[0], aspect='auto')
# plt.colorbar()
# plt.show()
# 
# plt.plot(y[0], alpha=0.1)
# plt.show()
# 
# ind = 1
# plt.plot(y[:, ind])
# plt.plot(spikes[:, ind])

plt.plot(spikes[:,:2] , alpha=0.1, color='k')
plt.show()

plt.imshow(spikes.T, aspect='auto')
plt.colorbar()
plt.show()


############################################################
# Create and train network 
############################################################
# Inputs and outputs need to be of shape (n_batches, batch_size, seq_len, input_size)
train_test_split = 0.8
n_train = int(spikes.shape[1] * train_test_split)
# Shape: (seq_len, batch_size, input_size)
inputs = spikes[:,:n_train, None]
tests = spikes[:,n_train:, None]
# inputs = y[..., None]

# Instantiate the network and print information
input_size = 1
hidden_size = 128
output_size = 1
# net = CTRNN_plus_output(
#         input_size=input_size, 
#         hidden_size=hidden_size,
#         output_size=output_size, 
#         dt= dt
#         )
# print(net)

net = rnnModel( 
        input_size=input_size,
        hidden_size= 10, 
        output_size=output_size,
        )

net = train_model(net, inputs, inputs, lr = 0.0001, train_steps = 100)
# 
# outs = net(torch.from_numpy(inputs).type(torch.float))
outs = net(torch.from_numpy(tests).type(torch.float))


# action_pred_list = [out[0].detach().numpy() for out in outs]
action_pred_list = [out.detach().numpy() for out in outs]
# rnn_activity_list = [out[1].detach().numpy() for out in outs]
action_pred_array = np.squeeze(np.stack(action_pred_list, axis=0))
# rnn_activity_array = np.stack(rnn_activity_list, axis=0)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].hist(action_pred_array.flatten(), bins=100)
ax[1].hist(inputs.flatten(), bins=100)
ax[0].set_title('Predicted')
ax[1].set_title('Input')
plt.show()

ind = 5
plt.plot(zscore(tests[:, ind], axis=-1), label='pred')
# plt.plot(inputs[:, ind], label='input')
plt.plot(zscore(action_pred_array[:, ind]), label='zscore pred')
plt.legend()
# plt.plot(inputs[inds[0], :, inds[1]])
plt.show()

n_plot = 20
rand_inds = np.random.choice(tests.shape[1], n_plot, replace=False)
fig, ax = plt.subplots(n_plot, 1, sharex=True)
for i, ind in enumerate(rand_inds): 
    ax[i].plot(zscore(action_pred_array[:, ind]))
    ax[i].plot(zscore(tests[:, ind]))
    # ax[i].plot(action_pred_array[:, ind])
    # ax[i].plot(inputs[:, ind])
plt.show()

plt.scatter(action_pred_array.flatten(), tests.flatten(),
            alpha=0.1)
plt.xlabel('Predicted')
plt.ylabel('Input')
plt.show()

fig, ax = plt.subplots(1,3)
ax[0].imshow(zscore(action_pred_array.T,axis=-1), aspect='auto', vmin=-2, vmax=2)
ax[1].imshow(zscore(np.squeeze(tests.T), axis=-1), aspect='auto', vmin=-2, vmax=2)
ax[2].imshow(y[:,n_train:].T, aspect='auto')
ax[0].set_title('Predicted')
ax[1].set_title('Input')
ax[2].set_title('True')
plt.show()
