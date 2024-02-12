import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision.io import read_image

DATASET_DIR = "data/"
TRESHOLD = 177

def choose_class(question):
    if "instrument" in question:
        return 0
    return 1


def preprocess_annotations():
    with open(f"{DATASET_DIR}/gt.json", encoding="utf-8") as file:
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
    df_vqa_masks["class"] = df_vqa_masks["question"].apply(choose_class)
    df_vqa_masks = df_vqa_masks[["image_id", "class"]].copy().reset_index()
    del df_vqa_masks["index"]

    # images with none category
    df_vqa_not_instrument = df_vqa[
        (df_vqa["question"] == "How many instrumnets are in the image?")
        & (df_vqa["answer"] == '0')
    ].copy()

    df_vqa_not_polyp = df_vqa[
        (df_vqa["question"] == "How many polyps are in the image?")
        & (df_vqa["answer"] == '0')
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
    df_vqa_result = pd.concat([df_vqa_masks, df_not_category], ignore_index=True)

    return df_vqa_result


def divide_data(df_vqa_result):
    df_0_category = df_vqa_result[df_vqa_result["class"] == 0]
    df_1_category = df_vqa_result[df_vqa_result["class"] == 1]
    df_2_category = df_vqa_result[df_vqa_result["class"] == 2]

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
    df_test_data["data"] = "test"
    df_data = pd.concat([df_train_data, df_test_data])
    return df_data


def df_semantic_segmenation_data(df_data):
    for idx, row in df_data.iterrows():
        image_filename = f"{DATASET_DIR}/images/{row['image_id']}.jpg"
        if row["class"] != 2:
            mask_path = f"{DATASET_DIR}/masks/{row['image_id']}_mask.jpg"
            try:
                mask_torch = read_image(mask_path)[0]
            except Exception:
                # TODO : determine name of exception
                mask_path = f"{DATASET_DIR}/masks/{row['image_id']}.jpg"
                mask_torch = read_image(mask_path)[0]
            
            mask_segmentation = mask_torch.numpy().astype("uint8")
            mask_segmentation[mask_segmentation <= TRESHOLD] = 0
            mask_segmentation[mask_segmentation > TRESHOLD] = 255
    
            # class 2 - background (pixel value 0)
            # class 0 - background (pixel value 1) - instrument
            # class 1 - background (pixel value 2) - polyp
            if row['class'] == 0 :
                mask_segmentation[mask_segmentation > TRESHOLD] = 1
            if row['class'] == 1 :
                mask_segmentation[mask_segmentation > TRESHOLD] = 2

        if row["class"] == 2:
            height, width = cv2.imread(image_filename).shape[:2]
            mask_segmentation = np.zeros([height, width])

        mask_filename = f"{DATASET_DIR}/masks_semantic/{row['image_id']}_mask.jpg"
        cv2.imwrite(mask_filename, mask_segmentation)   

if __name__ == '__main__':
    df = preprocess_annotations()
    df_divide = divide_data(df)
    df_semantic_segmenation_data(df_divide)
