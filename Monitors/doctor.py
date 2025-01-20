
import numpy as np
from .base_monitor import BaseMonitor


class DOCTOR(BaseMonitor):
    def __init__(self, mode="alpha", alpha=2):
        super().__init__()
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
   