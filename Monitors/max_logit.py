import numpy as np
from .base_monitor import BaseMonitor


class MaxLogitMonitor(BaseMonitor):
    def __init__(self):
        super().__init__()

    def fit(self):
        pass

    def predict(self, logits):
        return 1 - np.max(logits, axis=1)