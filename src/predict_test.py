import torch

from config.detectron_config import create_test_config, load_custom_config_from_file
from dataset.vlqa_dataset import DetectronDataset
from model.vlqa_model import DetectronModel
from utils.detectron_utils import draw_predictions

PATH_TO_CONFIG_AFTER_TRAIN = "output/vlqa_INSTRUMENT_2023_04_24_23_31_52/config.yaml"
PATH_TO_MODEL = "output/vlqa_INSTRUMENT_2023_04_24_23_31_52/model_final.pth"
PATH_TO_CUSTOM_CONFIG_AFTER_TRAIN = "output/vlqa_INSTRUMENT_2023_04_24_23_31_52/custom_config.yaml"
PATH_TO_EXAMPLE_IMAGE = "data/images/cl8k2u1qg1euf0832gmua4rbq.jpg"
# PATH_TO_EXAMPLE_IMAGE = "data/images/cl8k2u1pr1dzf08320rla73o6.jpg"
# PATH_TO_EXAMPLE_IMAGE = "data/images/cl8k2u1qq1f4b08323ks0c6ob.jpg"

if __name__ == "__main__":
    TORCH_VERSION = torch.__version__
    CUDA_VERSION = torch.version.cuda
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

    cfg = create_test_config(
        path_to_config_after_train=PATH_TO_CONFIG_AFTER_TRAIN,
        path_to_model=PATH_TO_MODEL,
    )
    custom_cfg = load_custom_config_from_file(PATH_TO_CUSTOM_CONFIG_AFTER_TRAIN)
    detectron = DetectronModel(cfg)
    dataset = DetectronDataset(cfg, custom_cfg)
    outputs = detectron.predict(path_to_image=PATH_TO_EXAMPLE_IMAGE)

    draw_predictions(cfg, PATH_TO_EXAMPLE_IMAGE, outputs)
