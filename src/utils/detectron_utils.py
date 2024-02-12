import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import CfgNode

def draw_predictions(cfg: CfgNode, path_to_original_image: str, outputs: dict):
    image_with_predictions = add_predictions_to_image(cfg, path_to_original_image,outputs)
    cv2.imshow("prediction", image_with_predictions)
    cv2.waitKey(0)

def add_predictions_to_image(cfg: CfgNode, path_to_original_image: str, outputs: dict):
    orignal_image = cv2.imread(path_to_original_image)
    v = Visualizer(
        orignal_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2
    )
    instance_predictions = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return instance_predictions.get_image()[:, :, ::-1]

