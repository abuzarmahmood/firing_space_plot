import torch.optim as optim
import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm, trange

from model import CTRNN_plus_output
from task import PerceptualDecisionMaking
import neurogym as ngym
import torch
import torch.nn as nn

def train_model(net, inputs, labels):
    """Simple helper function to train the model.

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(len(inputs)):
        # Generate input and target, convert to pytorch tensor
        # inputs, labels = dataset()
        this_inputs = inputs[i]
        this_labels = labels[i]
        this_inputs = torch.from_numpy(this_inputs).type(torch.float)
        this_labels = torch.from_numpy(this_labels.flatten()).type(torch.float32)
        this_labels = this_labels.view(-1, output_size)
        # this_labels = torch.from_numpy(this_labels).type(torch.float)

        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        this_output, _ = net(this_inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        this_output = this_output.view(-1, output_size)
        loss = criterion(this_output, this_labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i+1, running_loss, time.time() - start_time))
            running_loss = 0
    return net

############################################################
# Generate a dataset
taste_idens = [1, 2, 3, 4, 5] 
taste_iden_ind_map = {iden: i for i, iden in enumerate(taste_idens)}
pals = np.linspace(-1, 1, len(taste_idens))

# Instantiate the network and print information
input_size = len(taste_idens)
hidden_size = 128
output_size = 1
dt = 1


############################################################
# Some units don't go down after stimulus response (hypercritical)
# Having multiple trials in the same train time-series might help with this

baseline_len = 120
seq_len = 60
batch_size = 16
n_batches = 1000
n_trials_cat = 10

inputs = np.zeros((n_batches, batch_size, input_size, seq_len))
input_tastes = np.random.choice(taste_idens, size=(n_batches, batch_size))

# n_batches x batch_size x output_size 
labels = np.zeros((*inputs.shape[:2], inputs.shape[-1]), dtype=float)

# Iterate over input_tastes
for this_ind, this_taste in np.ndenumerate(input_tastes):
    inputs[this_ind[0], this_ind[1], 
          taste_iden_ind_map[this_taste], :(seq_len//2)] = 1
    labels[this_ind[0],this_ind[1], (seq_len//2):] = \
            pals[taste_iden_ind_map[this_taste]] 

# ind = 4
# fig, ax = plt.subplots(2,1, sharex=True)
# ax[0].imshow(inputs[ind,0], aspect='auto', interpolation='none')
# ax[1].plot(labels[ind,0])
# plt.show()

# n_batches x seq_len x batch_size x input_size
inputs = np.moveaxis(inputs, -1,1)
labels = np.moveaxis(labels, -1,1)

# Have a baseline period
inputs_baseline = np.zeros((n_batches, baseline_len, batch_size, input_size))
labels_baseline = np.zeros((n_batches, baseline_len, batch_size))

# Attached baseline to front
inputs = np.concatenate([inputs_baseline, inputs], axis=1)
labels = np.concatenate([labels_baseline, labels], axis=1)

# Also attach baseline to back
inputs = np.concatenate([inputs, inputs_baseline], axis=1)
labels = np.concatenate([labels, labels_baseline], axis=1)

# Add noise to inputs and outputs
inputs += np.random.normal(0, 0.2, inputs.shape)
labels += np.random.normal(0, 0.1, labels.shape)

# ind = 1
# fig, ax = plt.subplots(2,1, sharex=True)
# ax[0].imshow(inputs[ind,:,0].T, aspect='auto', interpolation='none')
# ax[1].plot(labels[ind,:,0])
# plt.show()

# Concatenate multiple trials together

fin_seq_len = inputs.shape[1]
inputs = inputs.swapaxes(1,2)
inputs = np.reshape(inputs, 
                    (-1, batch_size, fin_seq_len*n_trials_cat, input_size)) 
inputs = inputs.swapaxes(1,2)

labels = labels.swapaxes(1,2)
labels = np.reshape(labels,
                    (-1, batch_size, fin_seq_len*n_trials_cat,))
labels = labels.swapaxes(1,2)

# ind = 0
# fig, ax = plt.subplots(2,1, sharex=True)
# ax[0].imshow(inputs[ind,:,0].T, aspect='auto', interpolation='none')
# ax[1].plot(labels[ind,:,0])
# plt.show()



############################################################
# Train Model
net = CTRNN_plus_output(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
print(net)

net = train_model(net, inputs, labels)

############################################################
# Get activit per taste
outs = [net(torch.from_numpy(inputs[i]).type(torch.float)) \
        for i in trange(len(inputs))]


action_pred_list = [out[0].detach().numpy() for out in outs]
rnn_activity_list = [out[1].detach().numpy() for out in outs]
del outs

action_pred_array = np.squeeze(np.stack(action_pred_list, axis=0))
rnn_activity_array = np.stack(rnn_activity_list, axis=0)

plt.plot(action_pred_array[:,:,0].T, color='k', alpha=0.1)
plt.show()

# Generate a trial index that can be used after reshaping
single_seq_inds = np.arange(n_trials_cat).repeat(fin_seq_len)
trial_index_array = np.broadcast_to(single_seq_inds[None,:,None],
                                   action_pred_array.shape) 

action_pred_array = np.reshape(action_pred_array,
                        (n_batches, fin_seq_len, batch_size))
rnn_activity_array = np.reshape(rnn_activity_array,
                        (n_batches, fin_seq_len, batch_size, hidden_size))
trial_index_array = np.reshape(trial_index_array,
                        (n_batches, fin_seq_len, batch_size))

inputs_reshaped = np.reshape(inputs,
                        (n_batches, fin_seq_len, batch_size, input_size))
labels_reshaped = np.reshape(labels,
                        (n_batches, fin_seq_len, batch_size))

# Drop first n trials
n_trials_drop = 2
inds = np.unique(np.where(trial_index_array >= n_trials_drop)[0])

action_pred_array = action_pred_array[inds]
rnn_activity_array = rnn_activity_array[inds]
trial_index_array = trial_index_array[inds]
inputs_reshaped = inputs_reshaped[inds]
labels_reshaped = labels_reshaped[inds]
input_tastes_reshaped = input_tastes[inds]


# # Put everything into a dataframe
# inds = np.array(list(np.ndindex(action_pred_array.shape)))
# df = pd.DataFrame(inds, columns=['batch', 'time', 'trial'])
# df['action_pred'] = action_pred_array.flatten()
# df['trial_index'] = trial_index_array.flatten()
# df['labels'] = labels_reshaped.flatten()
# 
# input_inds = np.array(list(np.ndindex(inputs_reshaped.shape)))


run_ind = 17
trial_ind = 5
fig, ax = plt.subplots(4,1, sharex=True)
ax[0].imshow(inputs_reshaped[run_ind,:,trial_ind].T, 
             aspect='auto', interpolation='none')
ax[1].plot(labels_reshaped[run_ind,:,trial_ind])
ax[2].plot(action_pred_array[run_ind,:,trial_ind])
ax[3].plot(rnn_activity_array[run_ind,:,trial_ind])
plt.show()

# # Drop first trial in every sequence (transient effect)
# rnn_activity_array = rnn_activity_array[:,(2*fin_seq_len):]

# labels_orig = np.reshape(labels,
#                         (n_batches, fin_seq_len, batch_size))

mean_output_action = np.mean(
        action_pred_array[:,baseline_len+(seq_len//2):-baseline_len], 
        axis=1)
# output_true = pals[input_tastes]
output_true = np.mean(
        labels_reshaped[:,baseline_len+(seq_len//2):-baseline_len],
        axis=1)

# Plot mean output action vs true output
plt.scatter(mean_output_action.flatten(), output_true.flatten(),
            alpha=0.5, s=10, c='k')
min_x = np.min(mean_output_action)
max_x = np.max(mean_output_action)
plt.plot([min_x, max_x], [min_x, max_x], 'r--')
plt.xlabel('Mean Output Action')
plt.ylabel('True Output')
plt.show()

# Plot output activity
action_pred_long = np.concatenate(np.swapaxes(action_pred_array,1,2), axis=0)
input_tastes_long = np.concatenate(input_tastes_reshaped, axis=0)
cmap = plt.get_cmap('tab10')

n_thin = 10
for i in np.unique(input_tastes_long):
    this_action = action_pred_long[input_tastes_long==i][::n_thin]
    plt.plot(this_action.T, c= cmap(i), alpha=0.05) 
plt.show()

fig, ax = plt.subplots(3,1, sharex=True)
ax[0].plot(rnn_activity_array.mean(axis=(0,2)), alpha=0.5)
ax[1].plot(zscore(rnn_activity_array.mean(axis=(0,2)),axis=0), alpha=0.5)
ax[2].plot(action_pred_array.mean(axis=(0,2)), alpha=0.5)
plt.show()


# ##############################
# # Get neurons whose mean activity doesn't keep increasing
# mean_rnn_activity = np.mean(rnn_activity_array, axis=(0,2))
# diff_rnn_activity = np.diff(mean_rnn_activity, axis=0).mean(axis=0)
# plt.hist(diff_rnn_activity, bins=100)
# plt.show()
# 
# diff_thresh = 0.0002/2
# neurons_to_keep = np.where(diff_rnn_activity<diff_thresh)[0]
# 
# plt.plot(mean_rnn_activity[:,neurons_to_keep], alpha=0.5)
# plt.show()
# 
# ##############################

rnn_activity_long = np.concatenate(np.swapaxes(rnn_activity_array,1,2), axis=0)
# input_tastes_long = np.concatenate(input_tastes, axis=0)

rnn_activity_list = [rnn_activity_long[input_tastes_long==i] \
        for i in np.unique(input_tastes_long)]

mean_rnn_activity = np.stack(
        [np.mean(rnn_activity, axis=0) for rnn_activity in rnn_activity_list]
        )

plt.plot(mean_rnn_activity[...,2].T, alpha=0.5)
plt.show()

fig, ax = plt.subplots(1,len(mean_rnn_activity), sharex=True, sharey=True)
for i, this_mean_activity in enumerate(mean_rnn_activity):
    ax[i].imshow(this_mean_activity.T, interpolation='nearest', aspect='auto')
plt.show()

plt.plot(mean_rnn_activity[0], color='k', alpha=0.5)
plt.show()

##############################
# PCA
mean_rnn_activity_long = np.concatenate(mean_rnn_activity, axis=0)
pca_obj = PCA(n_components=3)
pca_obj.fit(mean_rnn_activity_long)
# pca_activity = pca_obj.transform(mean_rnn_activity_long)
pca_mean_rnn_activity = np.stack([pca_obj.transform(rnn_activity) \
        for rnn_activity in mean_rnn_activity]
                                 )

fig = plt.figure()
fig.add_subplot(111, projection='3d')
for i, this_pca_activity in enumerate(pca_mean_rnn_activity):
    plt.plot(
            this_pca_activity[:,0], 
            this_pca_activity[:,1], 
            this_pca_activity[:,2], 
            '-o',
            alpha=0.5,
            label = pals[i]
            )
plt.legend()
plt.show()

##############################
# Pull out "responsive" neurons
response_thresh = 0.05
responsive_neuron_inds = np.any(mean_rnn_activity > response_thresh, axis=(0,1))
responsive_neurons = mean_rnn_activity[:,:,responsive_neuron_inds]

fig, ax = plt.subplots(len(mean_rnn_activity), 1,sharex=True, sharey=True)
for i, this_mean_activity in enumerate(responsive_neurons):
    ax[i].imshow(this_mean_activity.T, interpolation='nearest', aspect='auto')
plt.show()

# fig, ax = plt.subplots(responsive_neurons.shape[-1], 1, sharex=True)
# for i in range(responsive_neurons.shape[-1]):
#     ax[i].plot(responsive_neurons[:,:,i].T,) 
# plt.show()
