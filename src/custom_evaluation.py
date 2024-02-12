""" Sources:
https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
https://github.com/frankkramer-lab/miseval
"""

from pathlib import Path

import cv2
import numpy as np

from config.detectron_config import create_test_config, load_custom_config_from_file
from dataset.vlqa_dataset import DetectronDataset
from model.vlqa_model import DetectronModel
from utils.detectron_utils import add_predictions_to_image

MODEL_VERSION = "vlqa_2023_04_24_21_06_26"
DETECTRON_CONFIG_PATH = "output/" + MODEL_VERSION + "/config.yaml"
DETECTRON_MODEL_PATH = "output/" + MODEL_VERSION + "/model_final.pth"
PATH_TO_CUSTOM_CONFIG_AFTER_TRAIN = "output/" + MODEL_VERSION + "/custom_config.yaml"
IMAGE_NAME = "cl8k2u1qg1euf0832gmua4rbq"

PATH_TO_EXAMPLE_IMAGE = f"data/images/{IMAGE_NAME}.jpg"
PATH_TO_EXAMPLE_MASK = f"data/masks/{IMAGE_NAME}_mask.jpg"

TARGET_CLASS = 1


def convert_detectron_outputs_to_predicted_mask(outputs, target_class, shape):
    predicted_instances = outputs["instances"].to("cpu")
    predicted_masks = predicted_instances._fields["pred_masks"].numpy()
    predicted_classes = predicted_instances._fields["pred_classes"].numpy()

    masks_of_target_class = [
        predicted_mask
        for predicted_mask, predicted_class in zip(predicted_masks, predicted_classes)
        if predicted_class == target_class
    ]

    result = np.zeros(shape, dtype=bool)
    for mask in masks_of_target_class:
        result = np.logical_or(result, mask)
    return result.astype(np.uint8) * 255


def convert_detectron_outputs_to_rgb_representation(outputs, shape):
    predicted_instances = outputs["instances"].to("cpu")
    predicted_masks = predicted_instances._fields["pred_masks"].numpy()
    predicted_classes = predicted_instances._fields["pred_classes"].numpy()

    result = np.zeros(shape).astype(np.uint8)
    is_foreground = np.zeros(shape[:2], dtype=bool)
    for predicted_mask, predicted_class in zip(predicted_masks, predicted_classes):
        result[:, :, 2 - predicted_class] += predicted_mask.astype(np.uint8) * 255
        is_foreground = np.logical_or(is_foreground, predicted_mask)
    background = np.logical_not(is_foreground)
    result[:, :, 0] = background.astype(np.uint8) * 255
    return result


def render_predictions(cfg, outputs):
    orignal_image = cv2.imread(PATH_TO_EXAMPLE_IMAGE)
    cv2.imwrite(f"{IMAGE_NAME}/original_image.jpg", orignal_image)

    prediction_visualization = add_predictions_to_image(
        cfg, PATH_TO_EXAMPLE_IMAGE, outputs
    )
    cv2.imwrite(f"{IMAGE_NAME}/prediction_visualization.jpg", prediction_visualization)

    mask_ground_truth = cv2.imread(PATH_TO_EXAMPLE_MASK, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(f"{IMAGE_NAME}/mask_ground_truth.jpg", mask_ground_truth)

    mask_prediction = convert_detectron_outputs_to_predicted_mask(
        outputs, TARGET_CLASS, mask_ground_truth.shape
    )
    cv2.imwrite(f"{IMAGE_NAME}/mask_target_class_prediction.jpg", mask_prediction)

    mask_rgb = convert_detectron_outputs_to_rgb_representation(
        outputs, orignal_image.shape
    )
    cv2.imwrite(f"{IMAGE_NAME}/mask_full_prediction_rgb.jpg", mask_rgb)


def generate_sets(ground_truth, prediction):
    gt = ground_truth
    pd = prediction
    not_gt = np.logical_not(gt)
    not_pd = np.logical_not(pd)
    return gt, pd, not_gt, not_pd


def calculate_accuracy(ground_truth, prediction):
    gt, pd, not_gt, not_pd = generate_sets(ground_truth, prediction)
    acc = (
        np.logical_and(pd, gt).sum() + np.logical_and(not_pd, not_gt).sum()
    ) / gt.size
    return acc


def calculate_iou(ground_truth, prediction):
    gt, pd, _, _ = generate_sets(ground_truth, prediction)
    acc = np.logical_and(pd, gt).sum() / np.logical_or(pd, gt).sum()
    return acc


def calculate_dice_coefficient(ground_truth, prediction):
    gt, pd, _, _ = generate_sets(ground_truth, prediction)
    acc = 2 * np.logical_and(pd, gt).sum() / (gt.sum() + pd.sum())
    return acc


if __name__ == "__main__":

    cfg = create_test_config(
        path_to_config_after_train=DETECTRON_CONFIG_PATH,
        path_to_model=DETECTRON_MODEL_PATH,
    )
    custom_cfg = load_custom_config_from_file(PATH_TO_CUSTOM_CONFIG_AFTER_TRAIN)
    detectron = DetectronModel(cfg)
    dataset = DetectronDataset(cfg, custom_cfg)

    directory = Path(f"{IMAGE_NAME}")
    if not directory.exists():
        directory.mkdir()

    outputs = detectron.predict(path_to_image=PATH_TO_EXAMPLE_IMAGE)

    mask_ground_truth = cv2.imread(PATH_TO_EXAMPLE_MASK, cv2.IMREAD_GRAYSCALE)
    mask_ground_truth_binary = cv2.threshold(
        mask_ground_truth, 127, 255, cv2.THRESH_BINARY
    )[1].astype(bool)

    mask_prediction = convert_detectron_outputs_to_predicted_mask(
        outputs, TARGET_CLASS, mask_ground_truth.shape
    )
    mask_prediction_binary = mask_prediction.astype(bool)

    print(
        f"Accuracy: {calculate_accuracy(mask_ground_truth_binary, mask_prediction_binary):.3}"
    )
    print(
        f"IoU (Jaccard Index): {calculate_iou(mask_ground_truth_binary, mask_prediction_binary):.3}"
    )
    print(
        f"Dice Coefficient (F1 Score): {calculate_dice_coefficient(mask_ground_truth_binary, mask_prediction_binary):.3}"
    )

    render_predictions(cfg, outputs)
