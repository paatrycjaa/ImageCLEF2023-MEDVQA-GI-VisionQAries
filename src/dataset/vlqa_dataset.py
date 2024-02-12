import json
from enum import Enum

import cv2
import detectron2.data.transforms as T
import numpy as np
import pandas as pd
import pycocotools
import torch
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split

from dataset.dataset import DatasetObject


class Categories(Enum):
    INSTRUMENT = 0
    POLYP = 1
    NON_CATEGORY = 2

    def __str__(self):
        return self.name

def read_mask(cfg_custom, image_id):
    mask_path = f"{cfg_custom['DATASET']['DATASET_DIR']}/masks/{image_id}_mask.jpg"
    mask = cv2.imread(mask_path)
    if mask is None:
        mask_path = f"{cfg_custom['DATASET']['DATASET_DIR']}/masks/{image_id}.jpg"
        mask = cv2.imread(mask_path)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)[1]
    return mask, gray_mask, thresh


class DetectronDataset(DatasetObject):
    def __init__(self, cfg, cfg_custom):
        super().__init__(cfg, cfg_custom)
        self.mask_ann = self.preprocess_annotations()
        self.update_cfg(cfg)
        self.data_part = "train"

    def update_cfg(self, cfg):
        self.register_train_dataset()
        self.register_valid_dataset()
        self.register_test_dataset()
        self.cfg.DATASETS.TRAIN = (self.cfg_custom["DATASET"]["DATASET_TRAIN_NAME"],)
        self.cfg.DATASETS.TEST = (self.cfg_custom["DATASET"]["DATASET_VALID_NAME"],)
        return cfg

    def preprocess_annotations(self):
        with open(
            f"{self.cfg_custom['DATASET']['DATASET_DIR']}/gt.json", encoding="utf-8"
        ) as file:
            vqa_json = json.load(file)
        vqa = []
        for image in vqa_json:
            image_id = image["ImageID"]
            for qa in image["Labels"]:
                vqa_item = {
                    "image_id": image_id,
                    "question": qa["Question"],
                    "answer_type": qa["AnswerType"],
                    "answer": qa["Answer"][0],
                }
                vqa.append(vqa_item)
        df_vqa = pd.DataFrame.from_dict(vqa)

        # images with category
        df_vqa_masks = df_vqa[df_vqa["answer_type"] == "segmentation"].copy()
        df_vqa_masks["class"] = df_vqa_masks["question"].apply(self.choose_class)
        df_vqa_masks = df_vqa_masks[["image_id", "class"]].copy().reset_index()
        del df_vqa_masks["index"]

        if self.cfg_custom["DATASET"]["NON_CATEGORY_DATA"]:
            # images with none category
            df_vqa_not_instrument = df_vqa[
                (df_vqa["question"] == "How many instrumnets are in the image?")
                & (df_vqa["answer"] == "0")
            ].copy()
            df_vqa_not_polyp = df_vqa[
                (df_vqa["question"] == "How many polyps are in the image?")
                & (df_vqa["answer"] == "0")
            ].copy()

            not_category_images = set(df_vqa_not_instrument.image_id.unique()) & set(
                df_vqa_not_polyp.image_id.unique()
            )

            not_category_dict = []
            for image in not_category_images:
                obj = {
                    "image_id": image,
                    "class": 2,
                }
                not_category_dict.append(obj)
            df_not_category = pd.DataFrame.from_dict(not_category_dict)
            # concatenate df with category and no category masks
            df_vqa_result = pd.concat(
                [df_vqa_masks, df_not_category], ignore_index=True
            )
        else:
            df_vqa_result = df_vqa_masks

        df_vqa_divide_results = self.divide_data(df_vqa_result)
        return df_vqa_divide_results

    def divide_data(self, df_vqa_result):
        df_0_category = df_vqa_result[df_vqa_result["class"] == 0]
        df_1_category = df_vqa_result[df_vqa_result["class"] == 1]
        df_2_category = df_vqa_result[df_vqa_result["class"] == 2][:150]

        df_0_category_train, df_0_category_test = train_test_split(
            df_0_category, test_size=0.1, shuffle=True
        )
        df_1_category_train, df_1_category_test = train_test_split(
            df_1_category, test_size=0.1, shuffle=True
        )
        df_2_category_train, df_2_category_test = train_test_split(
            df_2_category, test_size=0.1, shuffle=True
        )

        df_train_data = pd.concat(
            [df_0_category_train, df_1_category_train, df_2_category_train]
        )
        df_train_data["data"] = "train"
        df_test_data = pd.concat(
            [df_0_category_test, df_1_category_test, df_2_category_test]
        )
        df_test_data["data"] = "valid"
        df_data = pd.concat([df_train_data, df_test_data])
        df_shuffle = df_data.sample(frac=1).reset_index(drop=True)
        return df_shuffle

    def choose_class(self, question):
        if "instrument" in question:
            return 0
        return 1

    def register_train_dataset(self):
        self.data_part = "train"
        DatasetCatalog.register(
            self.cfg_custom["DATASET"]["DATASET_TRAIN_NAME"], self.get_colon_dicts_train
        )
        MetadataCatalog.get(self.cfg_custom["DATASET"]["DATASET_TRAIN_NAME"]).set(
            thing_classes=self.cfg_custom["MODEL"]["THING_CLASSES"]
        )

    def register_valid_dataset(self):
        self.data_part = "valid"
        DatasetCatalog.register(
            self.cfg_custom["DATASET"]["DATASET_VALID_NAME"], self.get_colon_dicts_valid
        )
        MetadataCatalog.get(self.cfg_custom["DATASET"]["DATASET_VALID_NAME"]).set(
            thing_classes=self.cfg_custom["MODEL"]["THING_CLASSES"]
        )

    def register_test_dataset(self):
        self.data_part = "test"
        DatasetCatalog.register(
            self.cfg_custom["DATASET"]["DATASET_TEST_NAME"], self.get_colon_dicts_test
        )
        MetadataCatalog.get(self.cfg_custom["DATASET"]["DATASET_TEST_NAME"]).set(
            thing_classes=self.cfg_custom["MODEL"]["THING_CLASSES"]
        )

    def get_colon_dicts_valid(self):
        self.data_part = "valid"
        return self.get_colon_dicts()

    def get_colon_dicts_train(self):
        self.data_part = "train"
        return self.get_colon_dicts()

    def get_colon_dicts_test(self):
        self.data_part = "test"
        return self.get_colon_dicts()

    def get_colon_dicts(self):
        dataset_dicts = []

        # filter valid, test or train data
        if self.data_part != "test":
            ann = self.mask_ann[self.mask_ann.data == self.data_part].copy()
        else:
            ann = self.mask_ann.copy()

        if len(self.cfg_custom["MODEL"]["THING_CLASSES"]) == 1:
            class_number = getattr(
                        Categories,
                        self.cfg_custom["MODEL"]["THING_CLASSES"][0] ).value
            ann = ann[ ann["class"].isin([2, class_number]) ].copy()

        print(f"Dataset: {self.data_part}")
        print(f"Number of images: {len(ann)}")

        for _, row in ann.iterrows():
            record = {}

            image_filename = f"{self.cfg_custom['DATASET']['DATASET_DIR']}/images/{row['image_id']}.jpg"
            height, width = cv2.imread(image_filename).shape[:2]
            objs = []
            if row["class"] != 2:
                _, _, thresh = read_mask(self.cfg_custom, row['image_id'])
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    bbox, instance_mask = self.get_bbox_and_instance_mask(
                        thresh, contour
                    )
                    if len(self.cfg_custom["MODEL"]["THING_CLASSES"]) == 2:
                        category_id = row["class"]
                    elif len(self.cfg_custom["MODEL"]["THING_CLASSES"]) == 1:
                        category_id = 0
                    else:
                        raise Exception("Wrong number of classes")
                    obj = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": pycocotools.mask.encode(
                            np.asarray(instance_mask, order="F")
                        ),
                        "category_id": category_id,
                    }
                    objs.append(obj)

            record["file_name"] = image_filename
            record["image_id"] = row["image_id"]
            record["height"] = height
            record["width"] = width
            record["annotations"] = objs

            dataset_dicts.append(record)

        return dataset_dicts

    def get_bbox_and_instance_mask(self, thresh, contour):
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [x, y, x + w, y + h]
        roi = thresh[y : y + h, x : x + w]
        contour_mask = (roi == 255).astype(np.uint8)
        instance_mask = np.zeros_like(thresh)
        instance_mask[y : y + h, x : x + w] = contour_mask * 255
        return bbox, instance_mask

    def get_augumentations(self):
        # we can probably add RandomRotation, ShiftScaleRotate and RandomCrop
        augs = [
            T.ResizeShortestEdge(
                self.cfg.INPUT.MIN_SIZE_TRAIN,
                self.cfg.INPUT.MAX_SIZE_TRAIN,
                self.cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        augs.append(T.RandomFlip())
        return augs

    def get_train_dataloader(self):
        mapper = DatasetMapper(
            self.cfg, is_train=True, augmentations=self.get_augumentations()
        )
        data_train_loader = build_detection_train_loader(
            self.cfg,
            mapper=mapper,
            total_batch_size=self.cfg_custom["HYPERPARAMETERS"]["TOTAL_BATCH_SIZE"],
        )
        return data_train_loader

    def get_test_dataloader(self):
        self.cfg.DATASETS.TEST = (self.cfg_custom["DATASET"]["DATASET_TEST_NAME"],)
        data_test_loader = build_detection_test_loader(
            self.cfg, self.cfg_custom["DATASET"]["DATASET_TEST_NAME"]
        )
        self.cfg.DATASETS.TEST = (self.cfg_custom["DATASET"]["DATASET_VALID_NAME"],)
        return data_test_loader

    def get_valid_dataloader(self):
        data_test_loader = build_detection_test_loader(
            self.cfg, self.cfg.DATASETS.TEST[0]
        )
        return data_test_loader

    def get_annoations(self):
        return self.mask_ann