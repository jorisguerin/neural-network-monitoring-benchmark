import numpy as np
from .base_monitor import BaseMonitor
from scipy.special import softmax


class ODIN(BaseMonitor):
    def __init__(self, T=50):
        super().__init__()
        self.T = T

    def fit(self):
        pass

    def predict(self, logits):
        confidence_scores = softmax(logits / self.T, axis=1)
        scores = np.max(confidence_scores, axis=1)
        return 1 - scores
