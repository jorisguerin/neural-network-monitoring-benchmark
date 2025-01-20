from .base_monitor import BaseMonitor
from scipy.special import logsumexp


class EnergyMonitor(BaseMonitor):
    def __init__(self, T=1):
        super().__init__()
        self.T = T

    def fit(self):
        pass

    def predict(self, logits):
        return 1 - self.T * logsumexp(logits / self.T, axis=-1)

