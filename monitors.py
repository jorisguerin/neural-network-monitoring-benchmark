import os
import h5py
import pickle

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from kneed import KneeLocator

from Params.params_network import *
from Params.params_monitors import *

from Utils.utils_monitors import Box, Boxes


class MaxSoftmaxProbabilityMonitor:
    def __init__(self):
        pass

    def fit(self):
        pass

    @staticmethod
    def predict(softmax_values):
        return np.max(softmax_values, axis=1)


class MaxLogitMonitor:
    def __init__(self):
        pass

    def fit(self):
        pass

    @staticmethod
    def predict(logits):
        return np.max(logits, axis=1)


class EnergyMonitor:
    def __init__(self, temperature=1):
        self.T = temperature

    def fit(self):
        pass

    def predict(self, logits):
        return self.T * logsumexp(logits / self.T, axis=-1)


class ReActMonitor:
    def __init__(self, quantile_value=0.9, mode="energy"):
        self.W = None
        self.b = None
        self.clip_value = None

        self.quantile_value = quantile_value
        self.mode = mode

    def fit(self, model, features):
        """

        :param model:
        :param features: Must be the features just before the last linear layer
        :return:
        """

        self.W = model.linear_weights
        self.b = model.linear_bias
        self.clip_value = np.quantile(features, self.quantile_value)

    def predict(self, features):
        clipped_features = np.clip(features, a_min=None, a_max=self.clip_value)
        modified_logits = clipped_features.dot(self.W.T) + self.b

        if self.mode == "energy":
            monitor = EnergyMonitor()
            inputs = modified_logits
        if self.mode == "msp":
            monitor = MaxSoftmaxProbabilityMonitor()
            inputs = softmax(modified_logits, axis=1)
        monitor.fit()
        scores = monitor.predict(inputs)

        return scores


class OutsideTheBoxMonitor:
    def __init__(self, dataset, network, layer_index, n_clusters=1, is_cv=True):
        """

        :param dataset:
        :param network:
        :param layer_index:
        :param n_clusters: Either an integer or "auto"
        :param is_cv: if n_clusters is tuned automatically, is_cv will determine whether
                      the training set should be split for the hyperparameter tuning procedure
        """

        self.dataset = dataset
        self._check_accepted_dataset()
        self.n_classes = n_classes_dataset[dataset]

        self.n_clusters_type = n_clusters
        self._check_accepted_params()
        self.n_clusters = None
        self.is_cv = is_cv

        layer_name = list(layers[network].items())[layer_index][0]
        self.file_name = save_monitors_path + "otb_%s_%s_%s_%s.p" % (n_clusters, dataset, network, layer_name)

        self.boxes = [Boxes() for _ in range(self.n_classes)]

    def fit(self, features, predictions=None, labels=None,
            use_only_correct=True, save=True):
        """

        :param features:
        :param predictions:
        :param labels:
        :param use_only_correct:
        :param save: If True, the GMM models will be saved and if they exist they will be loaded instead of retrained.
        :return:
        """
        if use_only_correct and (predictions is not None) and (labels is not None):
            correct_indices = (labels == predictions)
            features = features[correct_indices]
            labels = labels[correct_indices]
            predictions = predictions[correct_indices]

        if not os.path.exists(save_monitors_path):
            os.makedirs(save_monitors_path)

        if os.path.exists(self.file_name) and save:
            self.boxes = self._load_params(self.file_name)
        else:
            self.n_clusters = self._tune_hyperparameters(features, labels)

            for i in range(self.n_classes):
                km = KMeans(self.n_clusters[i])
                clusters = km.fit_predict(features[labels == i])
                for j in range(self.n_clusters[i]):
                    min_features = np.min(features[labels == i][clusters == j], axis=0)
                    max_features = np.max(features[labels == i][clusters == j], axis=0)

                    self.boxes[i].add_box(Box(min_features, max_features))

    def predict(self, features, predictions=None):
        scores = np.zeros(features.shape[0])
        for i in range(self.n_classes):
            scores_i = self.boxes[i].score(features[predictions == i])
            scores[predictions == i] = scores_i
        return scores

    def _check_accepted_dataset(self):
        accepted_dataset = list(n_classes_dataset.keys())
        if self.dataset not in accepted_dataset:
            raise ValueError("Accepted datasets are: %s" % str(accepted_dataset)[1:-1])

    def _check_accepted_params(self):
        if not np.issubdtype(type(self.n_clusters_type), np.integer):
            if self.n_clusters_type is not "auto":
                raise ValueError("Accepted n_clusters values are either int or \"auto\"")

    def _tune_hyperparameters(self, features, labels):
        if np.issubdtype(type(self.n_clusters_type), np.integer):
            n_clusters = [self.n_clusters_type] * self.n_classes
        else:
            values_n_clust = otb_n_clust_values
            n_clusters = []
            for i in range(self.n_classes):
                score = []
                for n in values_n_clust:
                    km = KMeans(n)
                    if self.is_cv:
                        val_split = int(4 * features[labels == i].shape[0] / 5)
                        km.fit(features[labels == i][:val_split])
                        score.append(km.score(features[labels == i][val_split:]))
                    else:
                        km.fit(features[labels == i])
                        score.append(km.score(features[labels == i]))
                kneedle = KneeLocator(values_n_clust, score)
                selected_n_clust = kneedle.knee
                n_clusters.append(selected_n_clust)

        return n_clusters

    @staticmethod
    def _save_params(boxes, file_name):
        pf = open(file_name, 'wb')
        pickle.dump(boxes, pf)
        pf.close()

    @staticmethod
    def _load_params(file_name):
        pf = open(file_name, 'rb')
        boxes = pickle.load(pf)
        pf.close()

        return boxes


class MahalanobisMonitor:
    def __init__(self, dataset, network, layer_index, is_tied=True):
        self.dataset = dataset
        self.cov_calculator = EmpiricalCovariance()
        self.is_tied = is_tied

        layer_name = list(layers[network].items())[layer_index][0]
        if is_tied:
            self.file_name = save_monitors_path + "mahalanobisTied_%s_%s_%s.h5" % (dataset, network, layer_name)
        else:
            self.file_name = save_monitors_path + "mahalanobisFree_%s_%s_%s.h5" % (dataset, network, layer_name)

        self._check_accepted_dataset()
        self.n_classes = n_classes_dataset[dataset]

        self.precision = None
        self.mean = []

    def fit(self, features, predictions=None, labels=None,
            use_only_correct=True, save=True):
        """

        :param features:
        :param predictions:
        :param labels:
        :param use_only_correct:
        :param save: If True, the GMM models will be saved and if they exist they will be loaded instead of retrained.
        :return:
        """
        if use_only_correct and (predictions is not None) and (labels is not None):
            correct_indices = (labels == predictions)
            features = features[correct_indices]
            labels = labels[correct_indices]
            predictions = predictions[correct_indices]

        if not os.path.exists(save_monitors_path):
            os.makedirs(save_monitors_path)

        if os.path.exists(self.file_name) and save:
            self.mean, self.precision = self._load_params(self.file_name)

        else:
            if self.is_tied:
                self.cov_calculator.fit(features)
                self.precision = self.cov_calculator.precision_
            else:
                self.precision = []
                for i in range(self.n_classes):
                    self.cov_calculator.fit(features[predictions == i])
                    self.precision.append(self.cov_calculator.precision_)
            self.mean = []
            for i in range(self.n_classes):
                self.mean.append(features[predictions == i].mean(axis=0))

            if save:
                self._save_params(self.mean, self.precision, self.file_name)

    def predict(self, features, predictions):
        scores = np.zeros([features.shape[0]])
        max_len = 1000
        for k in range(0, len(features), max_len):
            indices = range(k, min(k + max_len, len(features)))
            scores_int = np.zeros([len(indices)])
            for i in range(self.n_classes):
                if self.is_tied:
                    maha_squared = self._compute_mahalanobis(features[indices][predictions[indices] == i],
                                                             self.mean[i], self.precision)
                else:
                    maha_squared = self._compute_mahalanobis(features[indices][predictions[indices] == i],
                                                             self.mean[i], self.precision[i])
                scores_int[predictions[indices] == i] = maha_squared
            scores[indices] = scores_int
        return -scores

    def _check_accepted_dataset(self):
        accepted_dataset = list(n_classes_dataset.keys())
        if self.dataset not in accepted_dataset:
            raise ValueError("Accepted datasets are: %s" % str(accepted_dataset)[1:-1])

    @staticmethod
    def _save_params(mean, precision, file_name):
        hf = h5py.File(file_name, 'w')
        hf.create_dataset('mean', data=mean)
        hf.create_dataset('precision', data=precision)
        hf.close()

    @staticmethod
    def _load_params(file_name):
        hf = h5py.File(file_name, 'r')
        mean = np.array(hf.get("mean"))
        precision = np.array(hf.get("precision"))
        hf.close()

        return mean, precision

    @staticmethod
    def _compute_mahalanobis(features, mean, precision):
        residual = features - mean
        maha_squared = np.diag(np.matmul(np.matmul(residual, precision), residual.T))
        return maha_squared


class GaussianMixtureMonitor:
    def __init__(self, dataset, network, layer_index, n_components="auto_knee", constraint="full", is_cv=True):
        """

        :param dataset:
        :param network:
        :param layer_index:
        :param n_components: Either an integer or "auto_knee" or "auto_bic"
        :param constraint: Either "full", "diag", "tied", "spherical" or "auto_bic"
        :param is_cv: if n_components or constraint are tuned automatically, is_cv will determine whether
                      the training set should be split for the hyperparameter tuning procedure
        """

        self.dataset = dataset
        self._check_accepted_dataset()
        self.n_classes = n_classes_dataset[dataset]

        self.n_components_type = n_components
        self.constraint_type = constraint
        self._check_accepted_params()
        self.n_components, self.constraints = None, None
        self.is_cv = is_cv

        layer_name = list(layers[network].items())[layer_index][0]
        self.file_name = save_monitors_path + "gmm_%s_%s_%s_%s_%s.p" % (n_components, constraint,
                                                                        dataset, network, layer_name)

        self.gmm = None

    def fit(self, features, predictions=None, labels=None,
            use_only_correct=True, save=True):
        """

        :param features:
        :param predictions:
        :param labels:
        :param use_only_correct:
        :param save: If True, the GMM models will be saved and if they exist they will be loaded instead of retrained.
        :return:
        """
        if use_only_correct and (predictions is not None) and (labels is not None):
            correct_indices = (labels == predictions)
            features = features[correct_indices]
            labels = labels[correct_indices]
            predictions = predictions[correct_indices]

        if not os.path.exists(save_monitors_path):
            os.makedirs(save_monitors_path)

        if os.path.exists(self.file_name) and save:
            self.gmm = self._load_params(self.file_name)
        else:
            self.n_components, self.constraints = self._tune_hyperparameters(features, predictions)
            self.gmm = []
            for i in range(self.n_classes):
                gmm = GaussianMixture(n_components=self.n_components[i], covariance_type=self.constraints[i])
                self.gmm.append(gmm.fit(features[predictions == i]))
            self._save_params(self.gmm, self.file_name)

    def predict(self, features, predictions):
        scores = np.zeros([features.shape[0]])
        for i in range(self.n_classes):
            if np.count_nonzero(predictions == i) > 0:
                scores[predictions == i] = self.gmm[i].score_samples(features[predictions == i])
        return scores

    def _check_accepted_dataset(self):
        accepted_dataset = list(n_classes_dataset.keys())
        if self.dataset not in accepted_dataset:
            raise ValueError("Accepted datasets are: %s" % str(accepted_dataset)[1:-1])

    def _check_accepted_params(self):
        if not np.issubdtype(type(self.n_components_type), np.integer):
            if self.n_components_type not in accepted_n_comp:
                raise ValueError("Accepted n_components values are either int or one of: %s"
                                 % str(accepted_n_comp)[1:-1])
        if self.constraint_type not in accepted_constraints:
            raise ValueError("Accepted constraint values are : %s"
                             % str(accepted_constraints)[1:-1])

    def _tune_hyperparameters(self, features, predictions):
        if np.issubdtype(type(self.n_components_type), np.integer):
            values_n_comp = [self.n_components_type]
            n_components = [self.n_components_type] * self.n_classes
        else:
            values_n_comp = gmm_n_comp_values
            n_components = []
        if "auto" not in self.constraint_type:
            values_constraints = [self.constraint_type]
            constraints = [self.constraint_type] * self.n_classes
        else:
            values_constraints = gmm_constraints_values
            constraints = []

        if min(len(n_components), len(constraints)) == 0:
            if "auto_bic" in [self.constraint_type, self.n_components_type]:
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
