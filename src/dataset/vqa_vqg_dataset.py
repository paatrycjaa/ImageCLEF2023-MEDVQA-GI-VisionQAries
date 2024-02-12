import json
import os
import ast
from abc import ABC
from dataclasses import dataclass

import pandas as pd
import torch
from datasets import enable_caching, load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, AutoTokenizer

enable_caching()

MAP_NUMBER_TO_STRING = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
    "10": "Ten",
    "11": "Eleven",
    "12": "Twelve",
    "13": "Thirteen",
    "14": "Fourteen",
    "15": "Fifteen",
    "16": "Sixteen",
}
MAP_STRING_TO_NUMBER = {value: key for key, value in MAP_NUMBER_TO_STRING.items()}


def convert_number_to_string(row):
    if row["answer_type"] != "Number":
        return row["answer"]
    return MAP_NUMBER_TO_STRING.get(row["answer"], row["answer"])


class Dataset(ABC):
    def __init__(self, path_data, test_size=0.2):
        self.path_data = path_data
        self.test_size = test_size
        self._json = None
        self._df = None
        self.df_train = None
        self.df_test = None
        self.dataset = None
        self.input_text = None
        self.target_text = None
        self.task = None
        self.target_space = []

    def _read_json(self):
        with open(
            os.path.join(self.path_data, "gt.json"), "r", encoding="utf-8"
        ) as file:
            self._json = json.load(file)

    def _create_df(self):
        items = []
        for image in self._json:
            for label in image["Labels"]:
                for answer in label["Answer"]:
                    item = {
                        "image_id": image["ImageID"],
                        "question": label["Question"],
                        "answer_type": label["AnswerType"],
                        "answer": answer,
                    }
                    items.append(item)
        df = pd.DataFrame.from_dict(items)
        df = df[(df["answer_type"] != "segmentation") & (df["answer"] != "")].copy()
        df["answer"] = df.apply(convert_number_to_string, axis=1)
        df_multiple_target_text = (
            df.groupby(["image_id", self.input_text, "answer_type"])[self.target_text]
            .unique()
            .reset_index()
        )
        df_multiple_target_text[self.target_text] = df_multiple_target_text[
            self.target_text
        ].apply(lambda x: x.tolist())
        self._df = df_multiple_target_text

    def _create_target_space(self):
        self.target_space = []
        for ans in self._df[self.target_text].sum():
            self.target_space = (
                self.target_space + [ans]
                if "," not in ans
                else self.target_space + ans.replace(" ", "").split(",")
            )
        self.target_space = list(set(self.target_space + ["None"]))
        self.target_space.sort()

    def _train_test_split(self, test_size, random_state=42):
        self.df_train, self.df_test = train_test_split(
            self._df, test_size=test_size, random_state=random_state
        )

    def _save_dataset(self):
        with open(
            os.path.join(self.path_data, f"{self.task}_target_space.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            file.writelines("\n".join(self.target_space))
        self.df_train.to_csv(
            os.path.join(self.path_data, f"{self.task}_data_train.csv"),
            index=False,
        )
        self.df_test.to_csv(
            os.path.join(self.path_data, f"{self.task}_data_test.csv"),
            index=False,
        )

    def create_dataset(self):
        self._read_json()
        self._create_df()
        self._create_target_space()
        self._train_test_split(self.test_size)
        self._save_dataset()

    def check_if_dataset_created(self):
        return (
            os.path.isfile(os.path.join(self.path_data, f"{self.task}_data_train.csv"))
            and os.path.isfile(
                os.path.join(self.path_data, f"{self.task}_data_test.csv")
            )
            and os.path.isfile(
                os.path.join(self.path_data, f"{self.task}_target_space.txt")
            )
        )

    def load_dataset(self):
        self.dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(self.path_data, f"{self.task}_data_train.csv"),
                "test": os.path.join(self.path_data, f"{self.task}_data_test.csv"),
            },
        )
        with open(
            os.path.join(self.path_data, f"{self.task}_target_space.txt"),
            encoding="UTF-8",
        ) as file:
            self.target_space = file.read().splitlines()
        self.dataset = self.dataset.map(
            lambda examples: {"label": self.transform_to_target_space(examples)}
        )

    def transform_to_target_space(self, examples):
        ans_list = [0] * len(self.target_space)
        examples[self.target_text] = ast.literal_eval(examples[self.target_text])
        for ex in examples[self.target_text]:
            ans_list[self.target_space.index(ex)] = 1
        return ans_list

    def create_load_dataset(self):
        if not self.check_if_dataset_created():
            self.create_dataset()
        if not self.dataset:
            self.load_dataset()

    def num_labels(self):
        return len(self.target_space)


class VQGDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_text = "question"
        self.input_text = "answer"
        self.task = "vqg"
        self.create_load_dataset()


class VQADataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_text = "answer"
        self.input_text = "question"
        self.task = "vqa"
        self.create_load_dataset()


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    preprocessor: AutoFeatureExtractor

    def __init__(self, path_data, tokenizer, preprocessor):
        self.path_data = path_data
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.input_text = None

    def tokenize_text(self, texts):
        encoded_text = self.tokenizer(
            text=texts,
            padding="longest",
            max_length=24,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded_text["input_ids"].squeeze(),
            "token_type_ids": encoded_text["token_type_ids"].squeeze(),
            "attention_mask": encoded_text["attention_mask"].squeeze(),
        }

    def preprocess_images(self, images):
        processed_images = self.preprocessor(
            images=[
                Image.open(
                    os.path.join(self.path_data, "images", image_id + ".jpg")
                ).convert("RGB")
                for image_id in images
            ],
            return_tensors="pt",
        )
        return {
            "pixel_values": processed_images["pixel_values"].squeeze(),
        }

    def __call__(self, raw_batch_dict):
        return {
            **self.tokenize_text(
                raw_batch_dict[self.input_text]
                if isinstance(raw_batch_dict, dict)
                else [i[self.input_text] for i in raw_batch_dict]
            ),
            **self.preprocess_images(
                raw_batch_dict["image_id"]
                if isinstance(raw_batch_dict, dict)
                else [i["image_id"] for i in raw_batch_dict]
            ),
            "labels": torch.tensor(
                raw_batch_dict["label"]
                if isinstance(raw_batch_dict, dict)
                else [i["label"] for i in raw_batch_dict],
                dtype=torch.int64,
            ),
        }


class VQGCollator(Collator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_text = "answer"


class VQACollator(Collator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_text = "question"
