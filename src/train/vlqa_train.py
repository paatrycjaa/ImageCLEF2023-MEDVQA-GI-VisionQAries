import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import DefaultTrainer, default_writers
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

from config.detectron_config import save_config_to_yaml_file
from train.train import Train


def build_evaluator(cfg, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    """
    print(f"Evaluation output: {cfg.OUTPUT_DIR}")
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=output_folder))
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.
    """

    @classmethod
    def build_evaluator(cls, cfg, output_folder=None):
        return build_evaluator(cfg, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        print("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class DetectronTrain(Train):
    def __init__(self, cfg, custom_cfg, model, train_dataset):
        super().__init__(cfg, custom_cfg, model, train_dataset)

    def get_model(self):
        return self.model()

    def train_model(self, custom=False, resume=False):
        if custom:
            self.custom_train(resume)
        else:
            self.default_train(resume)
        save_config_to_yaml_file(self.cfg, self.cfg_custom)

    def default_train(self, resume=False):
        trainer = Trainer(self.cfg)
        trainer.resume_or_load(resume)
        trainer.train()

    def custom_train(self, resume=False):
        self.model.train()
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_lr_scheduler(self.cfg, optimizer)

        checkpointer = DetectionCheckpointer(
            self.model, self.cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
        )
        start_iter = (
            checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                "iteration", -1
            )
            + 1
        )
        max_iter = self.cfg.SOLVER.MAX_ITER

        periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

        writers = (
            default_writers(self.cfg.OUTPUT_DIR, max_iter)
            if comm.is_main_process()
            else []
        )

        # compared to "train_net.py", we do not support accurate timing and
        # precise BN here, because they are not trivial to implement in a small training loop
        print("Starting training from iteration {}".format(start_iter))
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(self.train_dataset, range(start_iter, max_iter)):
                storage.iter = iteration

                loss_dict = self.model(data)
                losses = sum(loss_dict.values())

                # TODO: logging
                print("Iteration {}".format(iteration))
                print("Losses: {}".format(losses))

                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {
                    k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
                }
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar(
                    "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
                )
                scheduler.step()

                if (
                    self.cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % self.cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
                ):
                    # TODO: Add evaluation on validation dataset
                    # do_test(cfg, model)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()

                if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)
