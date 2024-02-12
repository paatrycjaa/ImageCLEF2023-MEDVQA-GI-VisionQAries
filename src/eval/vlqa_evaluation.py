import json
from collections import OrderedDict

import numpy as np
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
)

from dataset.vlqa_dataset import read_mask
from eval.evaluation import Evaluation


class DetectronEvaluation(Evaluation):
    def __init__(self, cfg, model, cfg_custom, dataset):
        super().__init__(cfg, model, cfg_custom, dataset)

    def inference(self):
        evaluator = COCOEvaluator(
            self.cfg.DATASETS.TEST[0], output_dir=self.cfg.OUTPUT_DIR
        )
        self.results = inference_on_dataset(
            self.model.get_predictor().model, self.dataset, evaluator
        )

    def calculate_metrics(self) -> OrderedDict:
        return self.results

    def evaluate(self) -> OrderedDict:
        self.inference()
        return self.results

    def custom_evaluate(self, annotations):
        accuracies, ious, dice_coeffs = [[], []], [[], []], [[], []]

        for data in self.dataset:
            image_id = data[0]["image_id"]
            file_name = data[0]["file_name"]
            target_class = annotations[annotations["image_id"] == image_id][
                "class"
            ].item()
            if target_class != 2:
                _, _, thresh = read_mask(self.cfg_custom, image_id)
                ground_truth_binary_mask = thresh.astype(bool)

                outputs = self.model.predict(path_to_image=file_name)
                prediction_mask = self.convert_detectron_outputs_to_predicted_mask(
                    outputs, target_class, ground_truth_binary_mask.shape
                )
                prediction_binary_mask = prediction_mask.astype(bool)

                accuracies[target_class].append(
                    self.calculate_accuracy(
                        ground_truth_binary_mask, prediction_binary_mask
                    )
                )
                ious[target_class].append(
                    self.calculate_iou(
                        ground_truth_binary_mask, prediction_binary_mask
                    )
                )
                dice_coeffs[target_class].append(
                    self.calculate_dice_coefficient(
                        ground_truth_binary_mask, prediction_binary_mask
                    )
                )
        result = {}
        result['classes_accuracy'] = [np.mean(accuracies[0]), np.mean(accuracies[1])]
        result['mean-accuracy'] = np.mean(result['classes_accuracy'])
        result['classes_iou'] = [np.mean(ious[0]), np.mean(ious[1])]
        result['mean_iou'] = np.mean(result['classes_iou'])
        result['classes_dice_coeff'] = [np.mean(dice_coeffs[0]), np.mean(dice_coeffs[1])]
        result['mean_dice_coeff'] = np.mean(result['classes_dice_coeff'])
        self.results = result
        return self.results

    def convert_detectron_outputs_to_predicted_mask(self, outputs, target_class, shape):
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

    def generate_sets(self, ground_truth, prediction):
        gt = ground_truth
        pd = prediction
        not_gt = np.logical_not(gt)
        not_pd = np.logical_not(pd)
        return gt, pd, not_gt, not_pd

    def calculate_accuracy(self, ground_truth, prediction):
        gt, pd, not_gt, not_pd = self.generate_sets(ground_truth, prediction)
        acc = (
            np.logical_and(pd, gt).sum() + np.logical_and(not_pd, not_gt).sum()
        ) / gt.size
        return acc

    def calculate_iou(self, ground_truth, prediction):
        gt, pd, _, _ = self.generate_sets(ground_truth, prediction)
        acc = np.logical_and(pd, gt).sum() / np.logical_or(pd, gt).sum()
        return acc

    def calculate_dice_coefficient(self, ground_truth, prediction):
        gt, pd, _, _ = self.generate_sets(ground_truth, prediction)
        acc = 2 * np.logical_and(pd, gt).sum() / (gt.sum() + pd.sum())
        return acc

    def save_results_to_file(self, output_path: str) -> None:
        results = dict(self.results)
        with open(output_path, "w") as f:
            json.dump(results, f)
