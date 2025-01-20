"""
"""

import os
import pickle
from sklearn.cluster import KMeans
from kneed import KneeLocator

from .base_monitor import BaseMonitor

from Params.params_networks import *
from Params.params_monitors import *
from Utils.utils_monitors import Box, Boxes


class OutsideTheBoxMonitor(BaseMonitor):
    def __init__(self, dataset, network, layer_index, n_clusters=1, is_cv=True):
        """
        :param dataset:     The dataset to fit upon
        :param network:     The network to monitor
        :param layer_index: The layer index(es) to extract the features from
        :param n_clusters:  Either an integer or 'auto'
        :param is_cv:   If n_clusters is automatically tuned, is_cv will determine whether 
                        the training set should be split for the hyperparameters tuning, 
                        i.e. if cross-validation should be used.
        """
        super().__init__()

        self.dataset = dataset
        self._check_accepted_datasets()

        self.n_clusters_type = n_clusters
        self._check_accepted_params()

        self.n_clusters = None
        self.n_classes = n_classes_dataset[dataset]
        self.is_cv = is_cv
        self.boxes = [Boxes() for _ in range(self.n_classes)]
        layer_name = list(layers[network].items())[layer_index][0]
        self.filename_to_save_monitor = f"{path_to_saved_monitors}otb_{n_clusters}_{dataset}_{network}_{layer_name}.p"

    def fit(self, X, y_pred=None, y_true=None, use_only_correct=True, save=True):
        """
        :param X:       The feature to fit upon
        :param y_pred:  The predictions of the network
        :param y_true:  The true labels of the dataset
        :param use_only_correct: 
        :param save:    If True, the OTB models will be saved and if they exist, they will be loaded instead of retrained.
        :return:
        """
        if use_only_correct and (y_pred is not None) and (y_true is not None):
            correct_indices = (y_true == y_pred)
            X = X[correct_indices]
            y_true = y_true[correct_indices]
            y_pred = y_pred[correct_indices]
        
        if not os.path.exists(path_to_saved_monitors):
            os.makedirs(path_to_saved_monitors)

        if os.path.exists(self.filename_to_save_monitor) and save:
            self.boxes = self._load_params(self.filename_to_save_monitor)
        else:
            self.n_clusters = self._tune_hyperparameters(X, y_true)

            for i in range(self.n_classes):
                km = KMeans(self.n_clusters[i])
                clusters = km.fit_predict(X[y_true == i])

                for j in range(self.n_clusters[i]):
                    min_X = np.min(X[y_true == i][clusters == j], axis=0)
                    max_X = np.min(X[y_true == i][clusters == j], axis=0)
                    self.boxes[i].add_box(Box(min_X, max_X))

    def predict(self, X, y_pred=None):
        scores = np.zeros(X.shape[0])
        for i in range(self.n_classes):
            scores[y_pred == i] = self.boxes[i].score(X[y_pred == i])
        return scores

    def _check_accepted_datasets(self):
        datasets_lst = list(n_classes_dataset.keys())
        if self.dataset not in datasets_lst:
            raise ValueError(f"Accepted datasets are : {str(datasets_lst)[1:-1]}")

    def _check_accepted_params(self):
        if not np.issubdtype(type(self.n_clusters_type), np.integer):
            if self.n_clusters_type != "auto":
                raise ValueError(f"Accepted n_clusters values are either 'int' or \"auto\".")

    def _tune_hyperparameters(self, X, y_true):
        if np.issubdtype(type(self.n_clusters_type), np.integer):
            n_clusters = [self.n_clusters_type] * self.n_classes
        else:
            n_clusters = []
            n_clust_values = otb_n_clust_values
            for i in range(self.n_classes):
                score = []
                for n in n_clust_values:
                    km = KMeans(n)
                    if self.is_cv:
                        value_split = int(4 * X[y_true == i].shape[0] / 5)
                        km.fit(X[y_true == i][:value_split])
                        score.append(km.score(X[y_true == i][value_split:]))
                    else:
                        km.fit(X[y_true == i])
                        score.append(km.score(X[y_true == i]))
                kneedle = KneeLocator(n_clust_values, score)
                n_clusters.append(kneedle.knee)
        return n_clusters

    @staticmethod
    def _save_params(boxes, filename):
        pf = open(filename)
        pickle.dump(boxes, 'wb')
        pf.close()

    @staticmethod
    def _load_params(filename):
        pf = open(filename, 'rb')
        boxes = pickle.load(pf)
        pf.close()
        return boxes