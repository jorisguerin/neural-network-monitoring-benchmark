import numpy as np
from .base_monitor import BaseMonitor


class MSPMonitor(BaseMonitor):
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self, softmax):
        return 1 - np.max(softmax, axis=1)