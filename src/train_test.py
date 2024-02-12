import os
import shutil

import torch

from config.detectron_config import create_train_config, load_custom_config
from dataset.vlqa_dataset import DetectronDataset
from model.vlqa_model import DetectronModel
from train.vlqa_train import DetectronTrain

if __name__ == "__main__":
    TORCH_VERSION = torch.__version__
    CUDA_VERSION = torch.version.cuda
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

    try:
        shutil.rmtree("./image_clef_valid")
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    custom_cfg = load_custom_config()
    cfg = create_train_config(custom_cfg)
    detectron = DetectronModel(cfg)
    model = detectron.get_model()

    dataset = DetectronDataset(cfg, custom_cfg).get_train_dataloader()

    train = DetectronTrain(cfg, custom_cfg, model, dataset)
    train.train_model()
