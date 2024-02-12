import logging
import os
import sys

import torch

from config.vgqa_config import (
    VQGAConfig,
    create_traning_arguments_config,
    get_vqga_params_from_yaml,
    save_config_to_yaml_file,
)
from eval.vqga_evaluation import (
    VQGAEvaluation,
    calculate_metrics,
    get_inference_data,
    write_inference_to_json,
)
from dataset.vqa_vqg_dataset import VQACollator, VQADataset, VQGCollator, VQGDataset
from model.vgqa_model import VGQAModel
from train.vgqa_train import VGQATrain
from utils.pipeline_utils import check_if_checkpoint_exists


logger = logging.getLogger("vqga_pipeline")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    PATH_DATA = sys.argv[1]
    PATH_MODELS = sys.argv[2]
    TRAIN = sys.argv[3]
    INPUT_CONFIG = sys.argv[4]
    INFERENCE = sys.argv[5]
    PATH_INFERENCE_DATA = sys.argv[6]
    PATH_INFERENCE_TEXTS = sys.argv[7]
    PATH_INFERENCE_OUTPUT = sys.argv[8]
    logger.info(INPUT_CONFIG)

    CONFIG_NAME = INPUT_CONFIG.split(".", maxsplit=1)[0].split("/")[-1]
    INPUT_PARAMS = get_vqga_params_from_yaml(INPUT_CONFIG)
    logger.info(INPUT_PARAMS)

    TASK_TYPE = INPUT_PARAMS["model_config"]["type"]
    logger.info(f"Task type: {TASK_TYPE}")

    COLLATOR = VQGCollator if TASK_TYPE == "VQG" else VQACollator
    DATASET = VQGDataset if TASK_TYPE == "VQG" else VQADataset

    DEVICE_TYPE = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(DEVICE_TYPE)
    logger.info(f"Device: {DEVICE_TYPE}")

    CONFIG = VQGAConfig(
        config_name=CONFIG_NAME,
        collator=COLLATOR,
        dataset_generator=DATASET,
        path_data=PATH_DATA,
        path_models=PATH_MODELS,
        encoder_text_name=INPUT_PARAMS["model_config"]["encoder_text_name"],
        encoder_image_name=INPUT_PARAMS["model_config"]["encoder_image_name"],
        device_type=DEVICE_TYPE,
        device=device,
        test_data_size=INPUT_PARAMS["model_config"]["test_data_size"],
        model_intermediate_dim=INPUT_PARAMS["model_config"]["model_intermediate_dim"],
        model_intermediate_dim_dense=INPUT_PARAMS["model_config"][
            "model_intermediate_dim_dense"
        ],
        model_dropout=INPUT_PARAMS["model_config"]["model_dropout"],
    )

    logger.info(f"Config for {TASK_TYPE} initialized")
    logger.info(f"Path data: {CONFIG.path_data}")
    logger.info(f"Path models: {CONFIG.path_models}")
    logger.info(f"Path created model: {CONFIG.path_created_model}")

    # SET CACHE FOR HUGGINGFACE TRANSFORMERS + DATASETS
    os.environ["HF_HOME"] = os.path.join(".", "cache")
    # SET ONLY 1 GPU DEVICE
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Model
    VQGA_MODEL = VGQAModel(
        encoder_text_name=CONFIG.encoder_text_name,
        encoder_image_name=CONFIG.encoder_image_name,
        num_labels=CONFIG.num_labels,
        intermediate_dim=CONFIG.model_intermediate_dim,
        intermediate_dim_dense=CONFIG.model_intermediate_dim_dense,
        dropout=CONFIG.model_dropout,
    ).to(device)
    logger.info("Model initialized")

    # Training arguments
    training_arguments_config = create_traning_arguments_config(
        CONFIG.path_created_model, INPUT_PARAMS["training_config"]
    )
    logger.info("Config created")
    save_config_to_yaml_file(config=training_arguments_config, vqga_config=CONFIG)
    # Trainer
    trainer = VGQATrain(
        cfg=training_arguments_config,
        model=VQGA_MODEL,
        train_dataset=CONFIG.dataset["train"],
        test_dataset=CONFIG.dataset["test"],
        collator=CONFIG.collator,
        compute_metrics=calculate_metrics,
    )
    logger.info("Trainer initialized")

    # Train model
    if TRAIN != "false":
        logger.info("Starting training")
        if check_if_checkpoint_exists(CONFIG.path_created_model):
            logger.info("Training from checkpoint")
            trainer.train_model_from_checkpoint()
        else:
            logger.info("Training from start, no checkpoint found")
            trainer.train_model()
        logger.info("Model trained")

        # Save model
        torch.save(
            VQGA_MODEL.state_dict(), os.path.join(CONFIG.path_created_model, "model")
        )
        logger.info("Model saved")

    # Inference
    if INFERENCE != "false":
        evaluator = VQGAEvaluation(CONFIG)
        logger.info("Evaluator initialized, running inference")
        texts, paths, map_input_id = get_inference_data(
            PATH_INFERENCE_TEXTS, PATH_INFERENCE_DATA
        )
        infer = evaluator.inference(texts, paths)
        logger.info(f"Writing to file {PATH_INFERENCE_OUTPUT}")
        write_inference_to_json(
            texts,
            paths,
            map_input_id,
            infer,
            PATH_INFERENCE_OUTPUT,
            CONFIG.dataset_generator.input_text,
            CONFIG.dataset_generator.target_text,
        )
        logger.info("Written to file")
