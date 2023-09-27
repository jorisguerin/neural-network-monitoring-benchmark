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
        return 1-np.max(softmax_values, axis=1)


class MaxLogitMonitor:
    def __init__(self):
        pass

    def fit(self):
        pass

    @staticmethod
    def predict(logits):
        return 1-np.max(logits, axis=1)


class EnergyMonitor:
    def __init__(self, temperature=1):
        self.T = temperature

    def fit(self):
        pass

    def predict(self, logits):
        return 1-self.T * logsumexp(logits / self.T, axis=-1)

    
class ODINMonitor:
    def __init__(self, temperature=50):
        self.temperature = temperature

    def fit(self):
        pass

    def predict(self, logits):
        confidence_scores = softmax(logits / self.temperature, axis=1)
        scores = np.max(confidence_scores, axis=1)
        return 1-scores
    
    
class Doctor:
    def __init__(self, mode="alpha", alpha=2):
        self.mode = mode
        self.alpha = alpha

    def fit(self):
        pass

    def _calculate_Pe(self, softmax):
        Pe_beta = []
        for row in softmax:
            max_softmax = np.max(row)
            Pe_beta.append(1 - max_softmax)
        return Pe_beta

    def _calculate_g_x(self, softmax):
        g_x = []
        for row in softmax:
            sum_r = np.sum(row ** self.alpha)  # Utilisation de self.alpha
            g_x.append(1 - sum_r)
        return g_x

    def _calculate_F(self, softmax):
        if self.mode == "alpha":
            F = self._calculate_g_x(softmax)
        elif self.mode == "beta":
            F = self._calculate_Pe(softmax)
        else:
            raise ValueError("Mode invalide. Veuillez choisir 'alpha' ou 'beta'.")
        return F

    @staticmethod
    def _doctor_ratio(F):
        return [f / (1 - f) for f in F]

    def predict(self, softmax):
        F = self._calculate_F(softmax)
        scores = self._doctor_ratio(F)
        return np.array(scores)
    
    
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
        if self.mode == "doctor alpha":
            monitor = Doctor(mode="alpha")
            inputs = softmax(modified_logits, axis=1)
        if self.mode == "ODIN":
            monitor = ODINMonitor()
            inputs = modified_logits
        if self.mode == "maxlogits":  
            monitor = MaxLogitMonitor()
            inputs = modified_logits
      
        monitor.fit()
        scores = monitor.predict(inputs)

        return scores
