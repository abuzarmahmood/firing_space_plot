import os
from glob import glob
import numpy as np
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from itertools import product
import tables
import pylab as plt

import sys
discrim_path = '/media/bigdata/firing_space_plot/firing_analyses/single_trial_discrimination'
sys.path.append(discrim_path)
from single_trial_discrim_test import template_classifier

ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
sys.path.append(ephys_data_dir)
from ephys_data import ephys_data

class discrim_handler():
    def __init__(self, dir_name, epoch):
        self.dir_name = dir_name
        self.epoch = epoch
        self.dat = ephys_data(dir_name)
        self.dat.get_spikes()
        self.dat.get_info_dict()
        self.taste_names = self.dat.info_dict['taste_params']['tastes']

    def get_epoch_firing_rates(self):
        spike_array = np.stack(self.dat.spikes)
        epoch_spike_trains = spike_array[...,self.epoch[0]:self.epoch[1]]
        # epoch_spike_counts.shape = (n_tastes, n_trials, n_neurons)
        epoch_spike_counts = np.sum(epoch_spike_trains,axis=-1)
        epoch_duration = np.abs(np.diff(self.epoch))[0]/1000
        epoch_firing_rates = epoch_spike_counts / epoch_duration
        return epoch_firing_rates

    def get_epoch_normal_rates(self):
        epoch_firing_rates = self.get_epoch_firing_rates()
        # Normalize firing rates for each neurons for each epoch
        epoch_normal_rates = epoch_firing_rates - \
                epoch_firing_rates.mean(axis=(0,1), keepdims=True)
        epoch_normal_rates = epoch_normal_rates / \
                epoch_normal_rates.std(axis=(0,1), keepdims=True)
        return epoch_normal_rates

    def get_epoch_flat_rates(self):
        epoch_normal_rates = self.get_epoch_normal_rates()
        epoch_flat_rates = epoch_normal_rates.reshape(\
                (-1,epoch_normal_rates.shape[-1]))
        return epoch_flat_rates

    def perform_classification(self):
        epoch_flat_rates = self.get_epoch_flat_rates()
        self.y = np.repeat(
                np.arange(len(self.taste_names)),
                epoch_flat_rates.shape[0]/len(self.taste_names))
        clf = template_classifier()
        clf.fit(epoch_flat_rates,self.y)
        return clf

    def return_pred_proba_and_entropy(self):
        clf = self.perform_classification()
        epoch_flat_rates = self.get_epoch_flat_rates()
        pred = clf.predict(epoch_flat_rates)
        pred_proba = clf.predict_proba(epoch_flat_rates).T
        pred_entropy = clf.prediction_entropy(epoch_flat_rates)
        return pred, pred_proba, pred_entropy

if __name__ == '__main__':

    epoch_lims = [[300,800], [800,1300]]
    epoch_names = ['middle', 'late']

    dir_list_path = \
            '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
    with open(dir_list_path, 'r') as f:
        dir_list = f.read().splitlines()

    epoch_inds = np.arange(len(epoch_lims))
    dir_inds = np.arange(len(dir_list))

    iter_inds = list(product(epoch_inds, dir_inds))

    for this_iter in tqdm(iter_inds):
        #this_iter = iter_inds[0]

        epoch_ind = this_iter[0]
        dir_ind = this_iter[1]

        this_epoch = epoch_lims[epoch_ind]
        dir_name = dir_list[dir_ind]

        print(f'Processing {dir_name}')
        print(f'Epoch : {this_epoch}')

        this_epoch_name = epoch_names[epoch_ind]
        basename = dir_name.split('/')[-1]

        #dat = ephys_data(dir_name)
        #dat.get_spikes()
        #dat.get_info_dict()
        #taste_names = dat.info_dict['taste_params']['tastes']

        #spike_array = np.stack(dat.spikes)
        #epoch_spike_trains = spike_array[...,this_epoch[0]:this_epoch[1]]
        ## epoch_spike_counts.shape = (n_tastes, n_trials, n_neurons)
        #epoch_spike_counts = np.sum(epoch_spike_trains,axis=-1)
        #epoch_duration = np.abs(np.diff(this_epoch))[0]/1000
        #epoch_firing_rates = epoch_spike_counts / epoch_duration

        ## Normalize firing rates for each neurons for each epoch
        #epoch_normal_rates = epoch_firing_rates - \
        #        epoch_firing_rates.mean(axis=(0,1), keepdims=True)
        #epoch_normal_rates = epoch_normal_rates / \
        #        epoch_normal_rates.std(axis=(0,1), keepdims=True)

        ## Flatten across taste
        #epoch_normal_flat = epoch_normal_rates.reshape( \
        #        (-1, epoch_normal_rates.shape[-1]))

        ## Perform classification
        #y = np.tile(
        #        np.arange(len(taste_names)),
        #        (epoch_normal_rates.shape[1],1)).T.flatten()

        #clf = template_classifier().fit(epoch_normal_flat, y)
        #pred = clf.predict(epoch_normal_flat)
        #pred_proba = clf.predict_proba(epoch_normal_flat).T
        #pred_entropy = clf.prediction_entropy(epoch_normal_flat)

        this_discrim_handler = discrim_handler(
                dir_name, this_epoch)

        epoch_flat_rates = this_discrim_handler.get_epoch_flat_rates()

        # # Use xgboost to perform classification
        # import xgboost as xgb
        # clf = xgb.XGBClassifier()
        # clf.fit(epoch_flat_rates, this_discrim_handler.y)
        # clf.score(epoch_flat_rates, this_discrim_handler.y)

        # # Use sklearn to perform leave-one-out cross validation
        # from sklearn.model_selection import LeaveOneOut, cross_val_score
        # loo = LeaveOneOut()
        # clf = xgb.XGBClassifier()
        # scores = cross_val_score(
        #         clf,
        #         epoch_flat_rates,
        #         this_discrim_handler.y,
        #         cv=loo)
        # # Print mean +/- std
        # print(f'Accuracy : {np.round(np.mean(scores),2)}'\
        #         f'+/- {np.round(np.std(scores),2)}')


        pred, pred_proba, pred_entropy = \
                this_discrim_handler.return_pred_proba_and_entropy()
        y = this_discrim_handler.y
        taste_names = this_discrim_handler.taste_names

        accuracy = np.sum(pred == y) / len(y)

        # Accuracy per taste
        accuracy_per_taste = [np.mean(pred[y==t] == y[y==t]) \
                for t in range(len(taste_names))]
        accuracy_per_taste = np.round(np.array(accuracy_per_taste),2)

        # # Generate dataframe for saving
        # save_dict = {
        #         'epoch_name' : this_epoch_name,
        #         'epoch_lims' : this_epoch,
        #         'pred_entropy' : pred_entropy,
        #         'pred_proba' : pred_proba,
        #         'pred' : pred,
        #         'y' : y,
        #         'trials' : np.arange(len(y)),
        #         'taste_names' : np.array(taste_names)[y],
        #         'basename' : basename}

        # save_df = pd.DataFrame(save_dict)

        # Plot classifier predictions
        plot_dir = '/media/bigdata/firing_space_plot/lfp_analyses/granger_causality/plots/disrimination_plots'
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        imshow_kwargs = {'aspect' : 'auto', 'interpolation' : 'none'}

        even_probs = np.ones(len(taste_names)) / len(taste_names)
        max_entropy = -np.sum(even_probs * np.log2(even_probs)) 

        fig, ax = plt.subplots(1,4, sharey=True)
        im = ax[0].imshow(pred_entropy[:,None], **imshow_kwargs)
        cax = fig.add_axes([0.05, -0.1, 0.2, 0.05])
        fig.colorbar(im, ax=ax[0], cax = cax, 
                     orientation='horizontal', label='Entropy')
        im = ax[1].imshow(pred_proba, **imshow_kwargs)
        cax = fig.add_axes([0.3, -0.1, 0.2, 0.05])
        fig.colorbar(im, ax=ax[1], cax = cax, 
                     orientation='horizontal', label='Probability')
        ax[2].imshow(pred[:,None], **imshow_kwargs)
        #accuracy_per_taste_str = [str(x) for x in accuracy_per_taste]
        #ax[2].text(0, -0.5,
        #           'Accuracy per taste:\n' + '\n'.join(accuracy_per_taste_str),
        #        transform=ax[2].transAxes)
        ax[3].imshow(y[:,None], **imshow_kwargs) 
        ax[0].set_title('Entropy\nMax Entropy = {:.2f}'.format(max_entropy))
        ax[1].set_title('Probability')
        ax[2].set_title('Prediction\nAccuracy = {:.2f}'.format(accuracy))
        ax[3].set_title('True')
        fig.suptitle(f'{basename} {this_epoch_name}')
        plt.tight_layout()
        fig.savefig(f'{plot_dir}/{basename}_{this_epoch_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        #plt.show()

