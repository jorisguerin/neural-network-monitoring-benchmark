import numpy as np
import os
import pickle
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator

from .base_monitor import BaseMonitor

from Params.params_monitors import *
from Params.params_networks import *


class GaussianMixtureMonitor(BaseMonitor):
    def __init__(self, dataset, network, layer_index, n_components="auto_knee", constraint="full", is_cv=True):
        """
        :param dataset:
        :param network:
        :param layer_index:
        :param n_components: Either an integer or "auto_knee" or "auto_bic" or
                             a list of integer of the same size as the number of classes
        :param constraint: Either "full", "diag", "tied", "spherical" or "auto_bic" or a list of string constraints
        :param is_cv: if n_components or constraint are tuned automatically, is_cv will determine whether
                      the training set should be split for the hyperparameter tuning procedure
        """
        self.dataset = dataset
        self._check_accepted_datasets()

        self.n_comp_type = n_components
        self.constr_type = constraint
        self._check_accepted_params()

        self.n_classes = n_classes_dataset[dataset]
        self.n_comp = None
        self.constr = None
        self.is_cv = is_cv
        self.gmm = None

        layer_name = list(layers[network].items())[layer_index][0]
        self.file_name = f"{path_to_saved_monitors}gmm_{n_components}_{constraint}_{dataset}_{network}_{layer_name}.p"


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
            self.n_comp, self.constr = self._tune_hyperparameters(X, y_pred)
            self.gmm = []
            for i in range(self.n_classes):
                gm = GaussianMixture(n_components=self.n_comp[i], covariance_type=self.constr[i])
                self.gmm.append(gm.fit(X[y_pred == i]))
            if save:
                self._save_params(self.gmm, self.file_name)

    def predict(self, X, y_pred):
        scores = np.zeros([X.shape[0]])
        for i in range(self.n_classes):
            if np.count_nonzero(y_pred == i) > 0:
                scores[y_pred == i] = self.gmm[i].score_samples(X[y_pred == i])
        return scores

    def _check_accepted_datasets(self):
        accepted_dataset = list(n_classes_dataset.keys())
        if self.dataset not in accepted_dataset:
            raise ValueError("Accepted datasets are: %s" % str(accepted_dataset)[1:-1])

    def _check_accepted_params(self):
        if not np.issubdtype(type(self.n_comp_type), np.integer):
            if (type(self.n_comp_type) is not list) and (self.n_comp_type not in accepted_n_comp):
                raise ValueError("Accepted n_components values are either int, list of ints or one of: %s"
                                 % str(accepted_n_comp)[1:-1])
        if type(self.constr_type) is not list:
            if self.constr_type not in accepted_constr:
                raise ValueError("Accepted constraint values are : %s"
                                 % str(accepted_constr)[1:-1])

    def _tune_hyperparameters(self, features, predictions):
        if type(self.n_comp_type) is list:
            n_components = self.n_comp_type
        elif np.issubdtype(type(self.n_comp_type), np.integer):
            values_n_comp = [self.n_comp_type]
            n_components = [self.n_comp_type] * self.n_classes
        else:
            values_n_comp = gmm_n_comp_values
            n_components = []

        if type(self.constr_type) is list:
            constraints = self.constr_type
        elif "auto" not in self.constr_type:
            values_constraints = [self.constr_type]
            constraints = [self.constr_type] * self.n_classes
        else:
            values_constraints = gmm_constr_values
            constraints = []

        if min(len(n_components), len(constraints)) == 0:
            if "auto_bic" in [self.constr_type, self.n_comp_type]:
                select_type = "auto_bic"
            else:
                select_type = "auto_knee"
            for i in range(self.n_classes):
                # print("\n", i, "\n")
                combination = []
                bic, total_score = [], []
                for vc in values_constraints:
                    for n in values_n_comp:
                        combination.append([vc, n])
                        gmm = GaussianMixture(n_components=n, covariance_type=vc)
                        if self.is_cv:
                            val_split = int(4 * features[predictions == i].shape[0] / 5)
                            gmm.fit(features[predictions == i][:val_split])
                            bic.append(gmm.bic(features[predictions == i][val_split:]))
                            total_score.append(gmm.score(features[predictions == i][val_split:]))
                        else:
                            gmm.fit(features[predictions == i])
                            bic.append(gmm.bic(features[predictions == i]))
                            total_score.append(gmm.score(features[predictions == i]))
                if select_type == "auto_knee":
                    selected_constraint = values_constraints[0]
                    kneedle = KneeLocator(values_n_comp, total_score)
                    selected_n_comp = kneedle.knee
                    # print("total score", total_score)
                    # print("ncomp", selected_n_comp)
                else:
                    selected_constraint, selected_n_comp = combination[np.argmin(bic)]
                    # print(selected_constraint, selected_n_comp)

                n_components.append(selected_n_comp)
                constraints.append(selected_constraint)

        return n_components, constraints

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
