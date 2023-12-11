"""
Look at separation during both identity and palatability epochs
Also look at separation between all tastes and simply palatability groups
Entropy of classifier predictions can be used to classify certainty
"""
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder

class template_classifier():
    """
    Classifier for template matching
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.class_templates = np.stack([self.X[self.y==i].mean(axis=0) \
                for i in self.classes])
        return self

    def predict(self, X):
        dists = np.stack([np.linalg.norm(X - self.class_templates[i],axis=1) \
                for i in range(len(self.classes))])
        return np.argmin(dists,axis=0)

    def predict_proba(self, X):
        dists = np.stack([np.linalg.norm(X - self.class_templates[i],axis=1) \
                for i in range(len(self.classes))])
        return softmax(-dists,axis=0)

    def prediction_entropy(self, X):
        return -np.sum(self.predict_proba(X) * \
                np.log(self.predict_proba(X)),axis=0)

class gaussian_classifier():
    """
    Classifier for gaussian mixture model
    """
    def __init__(self, variance_explained = 0.95):
        self.variance_explained = variance_explained

    def fit(self, X, y):
        self.X = X
        #self.preprocessed_X = self.preprocess(X)
        self.y = y
        self.classes = np.unique(y)
        self.gmm = GaussianMixture(n_components = len(self.classes))
        self.gmm.fit(self.X, self.y)
        return self

    def preprocess(self, X):
        """
        Get components which preserve 95% of variance
        """
        pca_model = PCA()
        pca_model.fit(X)
        self.n_components = np.where(np.cumsum(pca_model.explained_variance_ratio_) \
                > self.variance_explained)[0][0]
        preprocessed_data = pca_model.transform(X)[:,:self.n_components]
        return preprocessed_data

    def predict(self, X):
        return self.gmm.predict(X)

    def predict_proba(self, X):
        return self.gmm.predict_proba(X)

    def prediction_entropy(self, X):
        return -np.sum(self.predict_proba(X) * \
                np.log(self.predict_proba(X)),axis=-1)

############################################################
if __name__ == '__main__':
    import sys
    import pylab as plt
    ephys_data_dir = '/media/bigdata/firing_space_plot/ephys_data'
    sys.path.append(ephys_data_dir)
    from ephys_data import ephys_data
    import visualize as vz

    ## Log all stdout and stderr to a log file in results folder
    #sys.stdout = open(os.path.join(granger_causality_path, 'stdout.txt'), 'w')
    #sys.stderr = open(os.path.join(granger_causality_path, 'stderr.txt'), 'w')

    ############################################################
    # Load Data
    ############################################################

    dir_list_path = \
            '/media/bigdata/projects/pytau/pytau/data/fin_inter_list_3_14_22.txt'
    with open(dir_list_path, 'r') as f:
        dir_list = f.read().splitlines()

    #for dir_name in dir_list:
    dir_name = dir_list[0]
    basename = dir_name.split('/')[-1]
    # Assumes only one h5 file per dir
    print(f'Processing {basename}')
    dat = ephys_data(dir_name)
    dat.get_spikes()
    dat.firing_rate_params = dat.default_firing_params
    dat.get_firing_rates()

    vz.firing_overview(dat.all_firing_array)
    plt.show()

    dat.get_info_dict()
    taste_names = dat.info_dict['taste_params']['tastes']

    ############################################################
    # Template Matching Entropy
    ############################################################

    # Preprocessing

    # Summed normalized firing rates per epoch
    epoch_names = ['pre_stim','det','iden','pal']
    epoch_bounds = [[1500,2000],[2000,2300],[2300,2800],[2800,3500]]
    epoch_durations = np.array([epoch_bounds[i][1] - epoch_bounds[i][0] \
            for i in range(len(epoch_bounds))])
    epoch_durations = epoch_durations / 1000 # Convert to seconds

    spike_array = np.stack(dat.spikes)
    epoch_spike_trains = [spike_array[...,epoch_bounds[i][0]:epoch_bounds[i][1]] \
            for i in range(len(epoch_bounds))]
    # epoch_spike_counts.shape = (n_epochs, n_tastes, n_trials, n_neurons)
    epoch_spike_counts = np.stack([np.sum(epoch_spike_trains[i],axis=-1) \
            for i in range(len(epoch_spike_trains))])
    epoch_firing_rates = epoch_spike_counts / np.expand_dims(epoch_durations,(1,2,3))

    # Normalize firing rates for each neurons for each epoch
    epoch_normal_rates = epoch_firing_rates - \
            epoch_firing_rates.mean(axis=(0,-1), keepdims=True)
    epoch_normal_rates = epoch_normal_rates / \
            epoch_normal_rates.std(axis=(0,-1), keepdims=True)

    # Flatten across taste
    epoch_normal_flat = epoch_normal_rates.reshape( \
            (epoch_normal_rates.shape[0],-1, epoch_normal_rates.shape[-1]))

    # For each epoch, plot firing rates
    fig, ax = plt.subplots(1, len(epoch_names), figsize=(20,5))
    for i in range(len(epoch_names)):
        ax[i].imshow(epoch_normal_flat[i], aspect='auto')
        ax[i].set_title(epoch_names[i])
    plt.show()

    # Perform classification
    y = np.tile(
            np.arange(len(taste_names)),
            (epoch_normal_rates.shape[2],1)).T.flatten()

    this_epoch = epoch_normal_flat[0]
    clf = template_classifier().fit(this_epoch, y)
    pred = clf.predict(this_epoch)
    pred_proba = clf.predict_proba(this_epoch).T
    pred_entropy = clf.prediction_entropy(this_epoch)

    preprocessed_epoch = gaussian_classifier().preprocess(this_epoch)
    clf = gaussian_classifier().fit(preprocessed_epoch, y)
    pred = clf.predict(preprocessed_epoch)
    pred_proba = clf.predict_proba(preprocessed_epoch)
    pred_entropy = clf.prediction_entropy(preprocessed_epoch)

    one_hot_y = OneHotEncoder(sparse=False).fit_transform(y[:,None])
    out_corr = np.cov(pred_proba.T, one_hot_y.T)
    plt.imshow(out_corr); plt.show()

    # Plot classifier predictions
    fig, ax = plt.subplots(1,4, sharey=True)
    ax[0].imshow(pred_entropy[:,None], aspect='auto', interpolation = 'none')
    ax[1].imshow(pred_proba, aspect='auto', interpolation = 'none')
    ax[2].imshow(pred[:,None], aspect='auto', interpolation = 'none')
    ax[3].imshow(y[:,None], aspect='auto', interpolation = 'none')
    plt.show()

    # Entrop per epoch
    epoch_entropies = []
    for i in range(len(epoch_normal_flat)):
        this_epoch = epoch_normal_flat[i]
        clf = template_classifier().fit(this_epoch, y)
        pred_entropy = clf.prediction_entropy(this_epoch)
        epoch_entropies.append(pred_entropy)
    epoch_entropies = np.stack(epoch_entropies)

    # Plot entropies
    cmap = plt.get_cmap('tab10')
    for i, this_entropy in enumerate(epoch_entropies):
        plt.hist(this_entropy, bins = 10, 
                 alpha = 0.5, 
                 color=cmap(i))
        plt.axvline(this_entropy.mean(), color=cmap(i),
                    label = epoch_names[i], linewidth=3)
    plt.legend()
    plt.show()
