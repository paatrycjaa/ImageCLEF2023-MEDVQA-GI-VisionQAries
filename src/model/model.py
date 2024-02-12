from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def build_model(self):
        """Build specific model"""

    @abstractmethod
    def get_model(self):
        """Get builded model"""
    
    @abstractmethod
    def get_predictor(self):
        """Get predictor"""
    
    @abstractmethod
    def predict(self):
        """Get predictor"""
