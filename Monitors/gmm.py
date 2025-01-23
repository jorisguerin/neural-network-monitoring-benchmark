import numpy as np
import os
import pickle
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

from .base_monitor import BaseMonitor

from Params.params_monitors import *
from Params.params_networks import *


class GMMMonitor(BaseMonitor):
    def __init__(self, dataset, network, layer_index, n_components="auto_knee", t_covariance="diag", is_cv=True):
        """
        :param dataset:
        :param network:
        :param layer_index:
        :param n_components: Either 
            - an integer (common value for all GMs), 
            - a list of integers (same size as the number of classes), 
            - "auto_aic" (AIC optim),
            - "auto_bic" (BIC optim),
            - "auto_knee".
        :param t_covariance: Either 
            - "full" (each component has its own general cov matrix), 
            - "diag" (each component has its own diagonal cov matrix), 
            - "tied" (all components share the same general cov matrix), 
            - "spherical" (each component has its own single variance),
            - "auto_bic",
            - a list of string constraints
        :param is_cv: if n_components or constraint are tuned automatically, is_cv will determine whether
                      the training set should be split for the hyperparameter tuning procedure
        """
        self.dataset = dataset
        self._check_accepted_datasets()

        self.n_components_type = n_components
        self.t_covariance_type = t_covariance
        self._check_accepted_params()

        self.n_classes = n_classes_dataset[dataset]
        self.n_components = None
        self.t_covariance = None
        self.is_cv = is_cv
        self.gmm = None

        layer_name = list(layers[network].items())[layer_index][0]
        self.file_name = path_to_saved_monitors + "gmm_%s_%s_%s_%s_%s.p" % (
            n_components, 
            t_covariance, 
            dataset, 
            network, 
            layer_name
        )

    def fit(self, X, y_pred=None, y_true=None, use_only_correct=True, save=True):
        """
        :param X:       The features 
        :param y_pred:  The predictions of the network
        :param y_true:  The true labels 
        :param use_only_correct:
        :param save: If True, the GMM models will be saved and if they exist they will be loaded instead of retrained.
        :return:
        """
        if use_only_correct and (y_pred is not None) and (y_true is not None):
            correct_indices = (y_true == y_pred)
            X = X[correct_indices]
            y_true = y_true[correct_indices]
            y_pred = y_pred[correct_indices]

        if not os.path.exists(path_to_saved_monitors):
            os.makedirs(path_to_saved_monitors)

        if os.path.exists(self.file_name) and save:
            self.gmm = self._load_params(self.file_name)
        else:
            self.n_components, self.t_covariance = self._tune_hyperparameters(X, y_pred)
            self.gmm = []
            for i in range(self.n_classes):
                gm = GaussianMixture(n_components=self.n_components[i], covariance_type=self.t_covariance[i])
                self.gmm.append(gm.fit(X[y_pred == i]))
            if save:
                self._save_params(self.gmm, self.file_name)

    def predict(self, X, y_pred):
        scores = np.zeros([X.shape[0]])
        for i in range(self.n_classes):
            if np.count_nonzero(y_pred == i) > 0:
                scores[y_pred == i] = self.gmm[i].score_samples(X[y_pred == i])
        return -scores

    def _check_accepted_datasets(self):
        accepted_dataset = list(n_classes_dataset.keys())
        if self.dataset not in accepted_dataset:
            raise ValueError("Accepted datasets are: %s" % str(accepted_dataset)[1:-1])

    def _check_accepted_params(self):
        if not np.issubdtype(type(self.n_components_type), np.integer):
            if (type(self.n_components_type) is not list) and (self.n_components_type not in accepted_n_components):
                raise ValueError("Accepted n_components values are either int, list of ints or one of: %s"
                                 % str(accepted_n_components)[1:-1])
        if type(self.t_covariance_type) is not list:
            if self.t_covariance_type not in accepted_t_covariance:
                raise ValueError("Accepted t_covariance values are : %s"
                                 % str(accepted_t_covariance)[1:-1])

    def _tune_hyperparameters(self, features, predictions):
        """
        Optimize the GMM parameters w.r.t. the features and predictions,
        depending on the optimization criterion selected.
        """
        # Initialize the params for the number of components
        if type(self.n_components_type) is list and len(self.n_components_type) == len(self.n_classes):
            n_components = self.n_components_type
        elif np.issubdtype(type(self.n_components_type), np.integer):
            values_n_components = [self.n_components_type]
            n_components = [self.n_components_type] * self.n_classes
        else: # auto_something
            values_n_components = gmm_n_components_values
            n_components = []

        # Initialize the params for the type of covariance
        if type(self.t_covariance_type) is list and len(self.t_covariance_type) == len(self.n_classes):
            t_covariance = self.t_covariance_type
        elif "auto" not in self.t_covariance_type:
            values_t_covariance = [self.t_covariance_type]
            t_covariance = [self.t_covariance_type] * self.n_classes
        else: # auto_something
            values_t_covariance = gmm_t_covariance_values
            t_covariance = []

        # If there is a need to optimize either one of the params
        if min(len(n_components), len(t_covariance)) == 0:
            if self.n_components_type == "auto_aic":
                select_type = "auto_aic"
            elif self.n_components_type == "auto_bic" or self.t_covariance_type == "auto_bic":
                select_type = "auto_bic"
            else:
                select_type = "auto_knee"         

            for i in range(self.n_classes):
                combination = []
                aic = []  # To store the AIC scores
                bic = []  # To store the BIC scores
                total_score = []  # To store the LogLikelihood scores

                # Grid-search selection 'alamano'.
                for tcov in values_t_covariance:
                    for ncmp in values_n_components:
                        combination.append([tcov, ncmp])
                        gm = GaussianMixture(n_components=ncmp, covariance_type=tcov)

                        if self.is_cv:
                            val_split = int(4 * features[predictions == i].shape[0] / 5) # Train/test cut at 4/5 of the dataset
                            gm.fit(features[predictions == i][:val_split])
                            aic.append(gm.aic(features[predictions == i][val_split:]))
                            bic.append(gm.bic(features[predictions == i][val_split:]))
                            total_score.append(gm.score(features[predictions == i][val_split:]))
                        else:
                            gm.fit(features[predictions == i])
                            aic.append(gm.aic(features[predictions == i]))
                            bic.append(gm.bic(features[predictions == i]))
                            total_score.append(gm.score(features[predictions == i]))

                if select_type == "auto_bic":
                    selected_t_covariance, selected_n_components = combination[np.argmin(bic)]
                if select_type == "auto_aic":
                    selected_t_covariance, selected_n_components = combination[np.argmin(aic)]
                if select_type == "auto_knee":
                    cov_idx = 0 if len(values_t_covariance) == 1 else i
                    selected_t_covariance = values_t_covariance[cov_idx]
                    kneedle = KneeLocator(values_n_components, total_score)
                    selected_n_components = kneedle.knee

                n_components.append(selected_n_components)
                t_covariance.append(selected_t_covariance)

        return n_components, t_covariance

    @staticmethod
    def _save_params(gmm, file_name):
        pf = open(file_name, 'wb')
        pickle.dump(gmm, pf)
        pf.close()

    @staticmethod
    def _load_params(file_name):
        pf = open(file_name, 'rb')
        gmm = pickle.load(pf)
        pf.close()
        return gmm
