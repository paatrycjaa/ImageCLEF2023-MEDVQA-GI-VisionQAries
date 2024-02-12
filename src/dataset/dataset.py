from abc import ABC, abstractmethod


class DatasetObject(ABC):
    def __init__(self, cfg, cfg_custom):
        self.cfg = cfg
        self.cfg_custom = cfg_custom

    @abstractmethod
    def get_train_dataloader(self):
        """Get train dataloader"""

    @abstractmethod
    def get_test_dataloader(self):
        """Get test dataloader"""

    @abstractmethod
    def get_valid_dataloader(self):
        """Get valid dataloader"""
