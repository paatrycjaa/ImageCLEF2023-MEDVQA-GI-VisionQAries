import logging
import os
from dataclasses import dataclass, field
from typing import List, Union

import yaml

import torch
from transformers import (  # Preprocessing / Common
    AutoFeatureExtractor,
    AutoTokenizer,
    TrainingArguments,
)

from dataset.vqa_vqg_dataset import VQACollator, VQADataset, VQGCollator, VQGDataset

logger = logging.getLogger("vqga_config")
logging.basicConfig(level=logging.INFO)


@dataclass
class VQGAConfig:
    config_name: str
    collator: Union[VQGCollator, VQACollator]
    dataset_generator: Union[VQADataset, VQGDataset]
    path_data: str
    path_models: str
    device_type: str
    device: torch.device
    encoder_text_name: str
    encoder_image_name: str
    encoder_text: AutoTokenizer = field(init=False)
    encoder_image: AutoFeatureExtractor = field(init=False)
    num_labels: int = field(init=False)
    target_space: List[str] = field(init=False)
    test_data_size: float = 0.2
    model_intermediate_dim_dense: int = 512
    model_intermediate_dim: int = 512
    model_dropout: float = 0.5

    def __post_init__(self):
        self._init_models()
        self._init_collator()
        self.path_created_model = os.path.join(self.path_models, self.config_name)

    def _init_models(self):
        self.encoder_text = AutoTokenizer.from_pretrained(self.encoder_text_name)
        logger.info(f"Initialized model for text: {self.encoder_text_name}")
        self.encoder_image = AutoFeatureExtractor.from_pretrained(
            self.encoder_image_name
        )
        logger.info(f"Initialized model for image: {self.encoder_image_name}")

    def _init_collator(self):
        logger.info(f"Initializing collator {self.collator.__name__}")
        self.collator = self.collator(
            self.path_data,
            self.encoder_text,
            self.encoder_image,
        )
        logger.info("Collator initializer")

        logger.info(f"Initializing dataset {self.dataset_generator.__name__}")
        self.dataset_generator = self.dataset_generator(
            self.path_data, self.test_data_size
        )
        self.dataset = self.dataset_generator.dataset
        self.num_labels = self.dataset_generator.num_labels()
        self.target_space = self.dataset_generator.target_space
        logger.info(f"Dataset initialized with {self.num_labels} labels")


def create_traning_arguments_config(path_model, training_config):
    config = TrainingArguments(
        output_dir=path_model,
        seed=training_config["seed"],
        evaluation_strategy=training_config["evaluation_strategy"],
        eval_steps=training_config["eval_steps"],
        logging_strategy=training_config["logging_strategy"],
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config[
            "save_total_limit"
        ],  # Since models are large, save only the last 3 checkpoints at any given time while training
        metric_for_best_model=training_config["metric_for_best_model"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        remove_unused_columns=training_config["remove_unused_columns"],
        num_train_epochs=training_config["num_train_epochs"],
        fp16=training_config["fp16"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
    )  ## +image text and version of model

    return config


def save_config_to_yaml_file(config: TrainingArguments, vqga_config: VQGAConfig):
    model_type = (
        "VQA"
        if type(vqga_config.collator) == VQACollator
        and type(vqga_config.dataset) == VQADataset
        else "VQG"
    )
    training_dict = config.to_dict()
    vqga_config_dict = {
        "type": model_type,
        "path_data": vqga_config.path_data,
        "path_models": vqga_config.path_models,
        "device_type": vqga_config.device_type,
        "device": vqga_config.device,
        "encoder_text_name": vqga_config.encoder_text_name,
        "encoder_image_name": vqga_config.encoder_image_name,
        "num_labels": vqga_config.num_labels,
        "target_space": vqga_config.target_space,
        "test_data_size": vqga_config.test_data_size,
        "model_intermediate_dim": vqga_config.model_intermediate_dim,
        "model_dropout": vqga_config.model_dropout,
    }
    result = {"training_config": training_dict, "model_config": vqga_config_dict}
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    with open(os.path.join(config.output_dir, "config.yaml"), "w") as file:
        yaml.dump(result, file)


def get_vqga_params_from_yaml(input_file: str) -> dict:
    with open(input_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
