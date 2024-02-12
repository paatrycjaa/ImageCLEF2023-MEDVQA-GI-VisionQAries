import logging
import os
import json
from typing import Dict, Tuple
import numpy as np

import torch
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config.vgqa_config import VQGAConfig
from dataset.vqa_vqg_dataset import MAP_STRING_TO_NUMBER
from eval.evaluation import Evaluation
from model.vgqa_model import VGQAModel

logger = logging.getLogger("vqga_eval")
logging.basicConfig(level=logging.INFO)


def calculate_metrics(eval_tuple: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    thresholds = [0.3, 0.5]
    res = {}
    for threshold in thresholds:
        suffix = "_" + str(threshold) if threshold != 0.5 else ""
        pred, target = eval_tuple
        pred = np.array(pred > threshold, dtype=float)
        res = {
            **res,
            f"acc{suffix}": accuracy_score(y_true=target, y_pred=pred),
            f"micro/precision{suffix}": precision_score(
                y_true=target, y_pred=pred, average="micro"
            ),
            f"micro/recall{suffix}": recall_score(
                y_true=target, y_pred=pred, average="micro"
            ),
            f"micro/f1{suffix}": f1_score(y_true=target, y_pred=pred, average="micro"),
            f"macro/precision{suffix}": precision_score(
                y_true=target, y_pred=pred, average="macro"
            ),
            f"macro/recall{suffix}": recall_score(
                y_true=target, y_pred=pred, average="macro"
            ),
            f"macro/f1{suffix}": f1_score(y_true=target, y_pred=pred, average="macro"),
            f"samples/precision{suffix}": precision_score(
                y_true=target, y_pred=pred, average="samples"
            ),
            f"samples/recall{suffix}": recall_score(
                y_true=target, y_pred=pred, average="samples"
            ),
            f"samples/f1{suffix}": f1_score(
                y_true=target, y_pred=pred, average="samples"
            ),
        }
    return res


def get_inference_data(path_texts, path_data):
    with open(path_texts) as file:
        texts_raw = file.read().split("\n")
    map_input_id = {}
    for i, text in enumerate(texts_raw):
        map_input_id[text] = i
    paths_raw = [os.path.join(path_data, img) for img in os.listdir(path_data)]

    paths, texts = [], []
    for path in paths_raw:
        for text in texts_raw:
            paths.append(path)
            texts.append(text)
    return texts, paths, map_input_id


def write_inference_to_json(
    texts, paths, map_input_id, inferences, output_path, input_text, target_text
):
    res = {}
    input_text_cap = input_text.capitalize()
    target_text_cap = target_text.capitalize()
    for path, text, infer in zip(paths, texts, inferences):
        img_id = path.split("/")[-1]
        res[img_id] = [
            *res.get(img_id, []),
            {
                f"{input_text_cap}ID": map_input_id[text],
                input_text_cap: text,
                target_text_cap: infer,
            },
        ]

    with open(output_path, "w") as file:
        json.dump(res, file)


class VQGAEvaluation(Evaluation):
    def __init__(self, cfg):
        self.model = None
        self.cfg: VQGAConfig = cfg
        self._load_model(os.path.join(self.cfg.path_created_model, "model"))
        logger.info("Model loaded")

        super().__init__(cfg, self.model)

    def inference(self, texts, img_paths):
        res = []
        n_texts = len(texts)
        for i, (text, img_path) in enumerate(zip(texts, img_paths)):
            res.append(self.inference_single(text, img_path))
            if i % 100 == 0 and i > 0:
                logger.info(f"Inference {i}/{n_texts}")
        return res

    def inference_single(self, text, img_path, threshold=0.5):
        tokenized_text = self._tokenize_question(text)
        featurized_img = self._featurize_image(img_path)
        input_ids = tokenized_text["input_ids"].to(self.cfg.device)
        token_type_ids = tokenized_text["token_type_ids"].to(self.cfg.device)
        attention_mask = tokenized_text["attention_mask"].to(self.cfg.device)
        pixel_values = featurized_img["pixel_values"].to(self.cfg.device)
        output = self.model(input_ids, pixel_values, attention_mask, token_type_ids)

        preds = np.array(output["logits"].cpu().detach().numpy() > threshold)
        res = [
            MAP_STRING_TO_NUMBER.get(self.cfg.target_space[i], self.cfg.target_space[i])
            for i, pred in enumerate(preds[0])
            if pred
        ]
        return res

    def inference_single_text(self, text, img_path):
        inference = self.inference_single(text, img_path)
        return self.cfg.target_space[inference]

    def calculate_metrics(self):
        pass

    def evaluate(self):
        pass

    def _load_model(self, model_path):
        self.model = VGQAModel(
            encoder_text_name=self.cfg.encoder_text_name,
            encoder_image_name=self.cfg.encoder_image_name,
            num_labels=self.cfg.num_labels,
            intermediate_dim=self.cfg.model_intermediate_dim,
            intermediate_dim_dense=self.cfg.model_intermediate_dim_dense,
            dropout=self.cfg.model_dropout,
        ).to(self.cfg.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.cfg.device)

    def _tokenize_question(self, question) -> Dict:
        tokenizer = self.cfg.encoder_text
        encoded_text = tokenizer(
            text=[question],
            padding="longest",
            max_length=24,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text["input_ids"].to(self.cfg.device),
            "token_type_ids": encoded_text["token_type_ids"].to(self.cfg.device),
            "attention_mask": encoded_text["attention_mask"].to(self.cfg.device),
        }

    def _featurize_image(self, img_path) -> Dict:
        featurizer = self.cfg.encoder_image
        processed_images = featurizer(
            images=[Image.open(img_path).convert("RGB")],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images["pixel_values"].to(self.cfg.device),
        }
