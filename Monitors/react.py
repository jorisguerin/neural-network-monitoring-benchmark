import numpy as np
from scipy.special import softmax
from .base_monitor import BaseMonitor
from .energy import EnergyMonitor
from .doctor import DOCTOR
from .max_logit import MaxLogitMonitor
from .max_softmax import MaxSoftmaxProbabilityMonitor
from .odin import ODIN


class ReActMonitor(BaseMonitor):
    def __init__(self, quantile_value=.9, mode="energy"):
        super().__init__()
        self.W = None
        self.b = None
        self.clip_value = None

        self.qval = quantile_value
        self.mode = mode

    def fit(self, model, features):
        """
        :param model   :    The model to monitor 
        :param features:    The features just before the last linear layer.
        """
        self.W = model.linear_weights
        self.b = model.linear_bias
        self.clip_value = np.quantile(features, self.qval)

    def predict(self, features):
        clipped_features = np.clip(features, a_min=None, a_max=self.clip_value)
        modified_logits  = clipped_features.dot(self.W.T) + self.b

        match self.mode:
            case "energy":
                monitor = EnergyMonitor()
                inputs = modified_logits
            case "MSP":
                monitor = MaxSoftmaxProbabilityMonitor()
                inputs = softmax(modified_logits, axis=1)
            case "DOCTOR alpha":
                monitor = DOCTOR(mode="alpha")
                inputs = softmax(modified_logits, axis=1)
            case "ODIN":
                monitor = ODIN()
                inputs = modified_logits
            case "MaxLogits":
                monitor = MaxLogitMonitor()
                inputs = modified_logits
            case _:
                raise AttributeError("Unexpected value set to attribute 'mode' of ReActMonitor().")
        
        monitor.fit()
        scores = monitor.predict(inputs)
        return scores