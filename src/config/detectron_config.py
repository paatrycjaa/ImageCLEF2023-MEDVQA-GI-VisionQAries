import os
from datetime import datetime

import yaml
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg

# DETECTRON_CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# DETECTRON_MODEL_WEIGHTS = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# BATCH_SIZE = 4
# NUM_CLASSES = 2


def load_custom_config():
    with open("./src/config/detectron_config.yaml") as stream:
        cfg_custom = yaml.safe_load(stream)
    return cfg_custom


def load_custom_config_from_file(path):
    with open(path) as stream:
        cfg_custom = yaml.safe_load(stream)
    return cfg_custom


def create_train_config(cfg_custom) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(cfg_custom["MODEL"]["DETECTRON_CONFIG_FILE"])
    )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        cfg_custom["MODEL"]["DETECTRON_MODEL_WEIGHTS"]
    )
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = cfg_custom["HYPERPARAMETERS"][
        "BATCH_SIZE"
    ]  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_custom["MODEL"]["NUM_CLASSES"]
    cfg.SOLVER.IMS_PER_BATCH = (
        2  # This is the real "batch size" commonly known to deep learning people
    )
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = cfg_custom["HYPERPARAMETERS"]["EPOCHS"]
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.INPUT.MASK_FORMAT = "bitmask"
    today = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if len(cfg_custom["MODEL"]["THING_CLASSES"]) > 1:
        cfg.OUTPUT_DIR = "./output/vlqa_" + today
    else:
        cfg.OUTPUT_DIR = (
            "./output/vlqa_" + cfg_custom["MODEL"]["THING_CLASSES"][0] + "_" + today
        )
    # cfg.TEST.EVAL_PERIOD = 100
    return cfg


def create_test_config(path_to_config_after_train: str, path_to_model: str) -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(path_to_config_after_train)
    cfg.MODEL.WEIGHTS = path_to_model  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    return cfg


def save_config_to_yaml_file(cfg: CfgNode, custom_cfg):
    config = cfg.dump()
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as file:
        file.write(config)
    custom_config = yaml.dump(custom_cfg)
    with open(os.path.join(cfg.OUTPUT_DIR, "custom_config.yaml"), "w") as file:
        file.write(custom_config)
