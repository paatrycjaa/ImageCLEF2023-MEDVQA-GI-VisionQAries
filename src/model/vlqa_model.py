import cv2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model as build_detectron_model

from model.model import Model


class DetectronModel(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self.build_model()

    def build_model(self):
        model = build_detectron_model(self.cfg)
        DetectionCheckpointer(model).load(self.cfg.MODEL.WEIGHTS)
        return model

    def get_model(self):
        return self.model

    def get_predictor(self):
        return DefaultPredictor(self.cfg)
    
    def predict(self, path_to_image: str):
        im = cv2.imread(path_to_image)
        predictor = self.get_predictor()
        return predictor(im)
