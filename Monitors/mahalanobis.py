import h5py
import os
from sklearn.covariance import EmpiricalCovariance

from .base_monitor import BaseMonitor

from Params.params_monitors import *
from Params.params_networks import *


class MahalanobisMonitor(BaseMonitor):
    def __init__(self, dataset, network, layer_index, is_tied=True):
        super().__init__()

        self.dataset = dataset
        self.is_tied = is_tied
        self.cov_calculator = EmpiricalCovariance()

        layer_name = list(layers[network].items())[layer_index][0]
        if is_tied:
            self.file_name = path_to_saved_monitors + "mahalanobisTied_%s_%s_%s.h5" % (dataset, network, layer_name)
        else:
            self.file_name = path_to_saved_monitors + "mahalanobisFree_%s_%s_%s.h5" % (dataset, network, layer_name)

        self._check_accepted_datasets()
        self.n_classes = n_classes_dataset[dataset]
        self.precision = None
        self.mean = []

    def fit(self, X, y_pred=None, y_true=None, use_only_correct=True, save=True):
        """
        :param X:
        :param y_pred:
        :param y_true:
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
            self.mean, self.precision = self._load_params(self.file_name)
        else:
            if self.is_tied:
                self.cov_calculator.fit(X)
                self.precision = self.cov_calculator.precision_
            else:
                self.precision = []
                for i in range(self.n_classes):
                    self.cov_calculator.fit(X[y_pred == i])
                    self.precision.append(self.cov_calculator.precision_)
            self.mean = []
            for i in range(self.n_classes):
                self.mean.append(X[y_pred == i].mean(axis=0))

            if save:
                self._save_params(self.mean, self.precision, self.file_name)

    def predict(self, X, y_pred):
        scores = np.zeros([X.shape[0]])
        maxlen = 1000
        for k in range(0, len(X), maxlen):
            indices = range(k, min(k + maxlen, len(X)))
            scores_int = np.zeros([len(indices)])
            for i in range(self.n_classes):
                if self.is_tied:
                    maha_squared = self._compute_mahalanobis(
                        X[indices][y_pred[indices] == i], self.mean[i], self.precision)
                else:
                    maha_squared = self._compute_mahalanobis(
                        X[indices][y_pred[indices] == i], self.mean[i], self.precision[i])
                scores_int[y_pred[indices] == i] = maha_squared
            scores[indices] = scores_int
        return -scores

    def _check_accepted_datasets(self):
        datasets_lst = list(n_classes_dataset.keys())
        if self.dataset not in datasets_lst:
            raise ValueError(f"Accepted datasets are : {str(datasets_lst)[1:-1]}")

    @staticmethod
    def _save_params(mean, prec, filename):
        hf = h5py.File(filename, 'w')
        hf.create_dataset('mean', data=mean)
        hf.create_dataset('precision', data=prec)
        hf.close()

    @staticmethod
    def _load_params(filename):
        hf = h5py.File(filename, 'r')
        mean = np.array(hf.get("mean"))
        prec = np.array(hf.get("precision"))
        hf.close()
        return mean, prec
    
    @staticmethod
    def _compute_mahalanobis(X, mean, prec):
        residual = X - mean
        maha_squared = np.diag(np.matmul(np.matmul(residual, prec), residual.T))
        return maha_squared