from abc import ABC, abstractmethod


class Evaluation(ABC):
    def __init__(self, cfg, model, cfg_custom=None, dataset=None):
        self.cfg = cfg
        self.model = model
        self.cfg_custom = cfg_custom
        self.dataset = dataset

    @abstractmethod
    def inference(self):
        """Model prediction on test data"""

    @abstractmethod
    def calculate_metrics(self):
        """Calculate evaluation metrics on output of inference data"""

    @abstractmethod
    def evaluate(self):
        """Run evaluation pipeline"""
