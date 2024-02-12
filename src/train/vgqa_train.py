from transformers import Trainer

from train.train import Train


class VGQATrain(Train):
    def __init__(
        self, cfg, model, train_dataset, test_dataset, collator, compute_metrics
    ):
        super().__init__(cfg, model, train_dataset)
        self.multi_trainer = Trainer(
            model,
            cfg,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
            ## todo: waiting  test dataset.py
        )

    def get_model(self):
        return self.multi_trainer()

    def train_model_from_checkpoint(self):
        self.multi_trainer.train(resume_from_checkpoint=True)

    def train_model(self):
        self.multi_trainer.train()
