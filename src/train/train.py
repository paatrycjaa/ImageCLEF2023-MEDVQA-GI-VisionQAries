from abc import ABC, abstractmethod


class Train(ABC):
    def __init__(self, cfg, cfg_custom, model, train_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.cfg_custom = cfg_custom

    @abstractmethod
    def get_model(self):
        """Get trained model"""

    @abstractmethod
    def train_model(self):
        """Implementaion of train pipeline"""
