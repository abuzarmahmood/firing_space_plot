import torch.optim as optim
import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.stats import zscore

from model import CTRNN_plus_output
from task import PerceptualDecisionMaking
import neurogym as ngym
import torch
import torch.nn as nn

def train_model(net, dataset):
    """Simple helper function to train the model.

    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair

    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(1000):
        # Generate input and target, convert to pytorch tensor
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float)
        labels = torch.from_numpy(labels.flatten()).type(torch.long)

        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.view(-1, output_size)
        loss = criterion(output, labels)
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
# Create environment from source code
# Canned environment from neurogym
task_name = 'PerceptualDecisionMaking-v0'
# Importantly, we set discretization time step for the task as well
kwargs = {'dt': 20, 'timing': {'stimulus': 1000}}


env = PerceptualDecisionMaking(**kwargs)
# Visualize the environment with 2 sample trials
_ = ngym.utils.plot_env(env, num_trials=2)
plt.show()

# This is a simple task, the input and output are low-dimensional
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
print('Input size', input_size)
print('Output size', output_size)

# Make supervised dataset, neurogym boilerplate
seq_len = 60
batch_size = 16
dataset = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)

# Generate one batch of data when called
inputs, target = dataset()
print('Input has shape (SeqLen, Batch, Dim) =', inputs.shape)
print('Target has shape (SeqLen, Batch) =', target.shape)
print('Target are the integers, for example target in the first sequence:')
print(target[:, 0])


# Instantiate the network and print information
hidden_size = 128
net = CTRNN_plus_output(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=env.dt)
print(net)


net = train_model(net, dataset)


##############################
# Reset environment
# env = dataset.env
env.reset(no_step=True)

# Initialize variables for logging
perf = 0
activity_dict = {}  # recording activity
trial_infos = {}  # recording trial information

num_trial = 200
for i in range(num_trial):
    # Neurogym boiler plate
    # Sample a new trial
    trial_info = env.new_trial()
    # Observation and groud-truth of this trial
    ob, gt = env.ob, env.gt
    # Convert to numpy, add batch dimension to input
    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)

    # Run the network for one trial
    # inputs (SeqLen, Batch, InputSize)
    # action_pred (SeqLen, Batch, OutputSize)
    action_pred, rnn_activity = net(inputs)

    # Compute performance
    # First convert back to numpy
    action_pred = action_pred.detach().numpy()[:, 0, :]
    # Read out final choice at last time step
    choice = np.argmax(action_pred[-1, :])
    # Compare to ground truth
    correct = choice == gt[-1]

    # Record activity, trial information, choice, correctness
    rnn_activity = rnn_activity[:, 0, :].detach().numpy()
    activity_dict[i] = rnn_activity
    trial_infos[i] = trial_info  # trial_info is a dictionary
    trial_infos[i].update({'correct': correct})

# Print informations for sample trials
for i in range(5):
    print('Trial ', i, trial_infos[i])

print('Average performance', np.mean([val['correct'] for val in trial_infos.values()]))

##############################
# Plot activity of single neurons
activity_df = pd.DataFrame(trial_infos).T
activity_df['activity'] = list(activity_dict.values())

# ind = 0
# this_data = np.stack([activity_df['activity'][i][ind] for i in range(num_trial)])
# 
# plt.plot(this_data)
# plt.xlabel('Time')
# plt.show()
# 
# plt.imshow(activity_df.iloc[1]['activity'].T, aspect='auto')
# plt.show()

type_grouped = [x[1] for x in activity_df.groupby('ground_truth')]
type_grouped_activity = [np.stack(x['activity'].values) for x in type_grouped]
mean_type_activity = [np.mean(x, axis=0) for x in type_grouped_activity]

vmin = min(np.min(x) for x in mean_type_activity)
vmax = max(np.max(x) for x in mean_type_activity)

fig, ax = plt.subplots(1, 2, figsize=(8, 4),
                       sharex=True, sharey=True)
for i, activity in enumerate(mean_type_activity):
    # ax[i].imshow(zscore(activity.T, axis=-1), aspect='auto',
    ax[i].imshow(activity.T, aspect='auto',)
                 # vmin=vmin, vmax=vmax, cmap='viridis')
    ax[i].set_title('Ground truth = {}'.format(i))
cax = fig.add_axes([0.95, 0.2, 0.01, 0.6])
plt.colorbar(cm.ScalarMappable(cmap='viridis'), cax=cax)
plt.show()

# [x.sum(axis=-1) for x in mean_type_activity]

##############################
# Apply PCA, boilerplate sklearn

# Concatenate activity for PCA
activity = np.concatenate(list(activity_dict[i] for i in range(num_trial)), axis=0)
print('Shape of the neural activity: (Time points, Neurons): ', activity.shape)

pca = PCA(n_components=2)
pca.fit(activity)  # activity (Time points, Neurons)
activity_pc = pca.transform(activity)  # transform to low-dimension
print('Shape of the projected activity: (Time points, PCs): ', activity_pc.shape)

# Project each trial and visualize activity


# Plot all trials in ax1, plot fewer trials in ax2
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6, 3))

for i in range(100):
    # Transform and plot each trial
    activity_pc = pca.transform(activity_dict[i])  # (Time points, PCs)

    trial = trial_infos[i]
    color = 'red' if trial['ground_truth'] == 0 else 'blue'

    _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color,
                 alpha = 0.3)
    if i < 3:
        _ = ax2.plot(activity_pc[:,
                                 0], activity_pc[:, 1], 'o-', color=color)

    # Plot the beginning of a trial with a special symbol
    _ = ax1.plot(activity_pc[0, 0], activity_pc[0, 1], '^', color='black')

ax1.set_title('{:d} Trials'.format(100))
ax2.set_title('{:d} Trials'.format(3))
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
plt.show()

##############################
# Plot PCs over time
fig, ax = plt.subplots(2, 1, figsize=(6, 3))
for i in range(100):
    # Transform and plot each trial
    activity_pc = pca.transform(activity_dict[i])  # (Time points, PCs)

    trial = trial_infos[i]
    color = 'red' if trial['ground_truth'] == 0 else 'blue'

    ax[0].plot(activity_pc[:, 0], 'o-', color=color, alpha = 0.3)
    ax[1].plot(activity_pc[:, 1], 'o-', color=color, alpha = 0.3)

ax[0].set_title('PC 1')
ax[1].set_title('PC 2')
plt.show()
