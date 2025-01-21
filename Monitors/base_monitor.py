from abc import ABC, abstractmethod


class BaseMonitor(ABC):
    """
    Abstract class for a Base Monitor, in order to acheive 
    a general interface for the use of monitors.
    """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass