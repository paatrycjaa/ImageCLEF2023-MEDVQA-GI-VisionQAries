from pprint import pprint

import torch

from config.detectron_config import create_test_config, load_custom_config_from_file
from dataset.vlqa_dataset import DetectronDataset
from eval.vlqa_evaluation import DetectronEvaluation
from model.vlqa_model import DetectronModel

MODEL_VERSION = "vlqa_POLYP_2023_04_27_23_12_27"
PATH_TO_CONFIG_AFTER_TRAIN = "output/" + MODEL_VERSION + "/config.yaml"
PATH_TO_MODEL = "output/" + MODEL_VERSION + "/model_final.pth"
PATH_TO_CUSTOM_CONFIG_AFTER_TRAIN = "output/"+ MODEL_VERSION +"/custom_config.yaml"
PATH_TO_SAVE = "output/" + MODEL_VERSION + "/evaluation_results.json"

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
    detectron_dataset = DetectronDataset(cfg, custom_cfg)
    test_dataset = detectron_dataset.get_valid_dataloader()

    custom_cfg = load_custom_config_from_file(PATH_TO_CUSTOM_CONFIG_AFTER_TRAIN)
    
    evaluation = DetectronEvaluation(cfg, detectron, custom_cfg, test_dataset)
    results = evaluation.custom_evaluate(detectron_dataset.get_annoations())
    pprint(results)

    evaluation.save_results_to_file(PATH_TO_SAVE)
