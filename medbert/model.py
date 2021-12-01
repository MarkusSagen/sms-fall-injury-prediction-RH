#!/usr/bin/env ipython

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import transformers
from losses import CrossEntropyLoss, LabelSmoothing
from pytorch_lightning.metrics import (
    F1,
    Accuracy,
    MetricCollection,
    Precision,
    Recall,
)
from torch import FloatTensor, LongTensor, Tensor
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from utils import is_not_empty, tolist


class BinaryClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        lr: float,
        label_smoothing: float,
        batch_size: int,
        weight_decay: float,
        warmup_steps: int = 0,
        num_workers: int = 4,
        destroyed_indices: set = set(),

    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_workers = num_workers
        self.destroyed_indices = destroyed_indices

        self.loss = LabelSmoothing(num_labels, smoothing=label_smoothing)
        self.train_metrics = MetricCollection(
            [
                Accuracy(),
                Precision(num_classes=self.num_labels),
                Recall(num_classes=self.num_labels),
                F1(num_classes=self.num_labels),
            ]
        )
        self.val_metrics = self.train_metrics.clone()
        self.test_metrics = self.train_metrics.clone()

        self.config: PretrainedConfig
        self.model: PreTrainedModel

        # For debugging
        self.iter_counter = 0
        self.prediction_dict = {
            "dataset": "imdb",
            "split": "train",
            "epochs": [],
            "idx": [],
            "max_epochs": 0,
        }

        self.save_hyperparameters()


    def prepare_data(self):
        """Set logging level and download model on main process."""
        transformers.utils.logging.set_verbosity_warning()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        AutoModelForSequenceClassification.from_pretrained(self.model_name)


    def setup(self, stage: Optional[str] = None):
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=self.config
        )

        self.set_prediction_dict_value("max_epochs", self.trainer.max_epochs)


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids, attention_mask=attention_mask, return_dict=True
        )
        return output


    def log_predictions(
        self,
        data_idx_t: LongTensor,
        labels_t: LongTensor,
        probs_t: FloatTensor,
        preds_t: FloatTensor,
        batch_idx: int,
    ) -> None:
        """Capture and store the results from the training.

        For the logits, predicted labels, softmax values etc.,
        store the values to a global dict. This is only used for storing
        and verifying the predictions.

        The intention was to investigate if data points had been miss-labeled
        and if it was possible to detect from the softmax probably distributions.

        Args:
            data_idx_t - Index for all data samples in the batch
            labels_t - The expected correct labels
            probs_t - The softmax output for each batch
            preds_t - The predicted labels
            batch_idx - Index of the batch samples

        Returns:
            Updates global list with all batch predictions, labels etc.
        """

        # Store which predictions the model predicted correct and not
        match = tolist((preds_t == labels_t))
        indices_correct = [i for i, x in enumerate(match) if x]
        indices_incorrect = [i for i, x in enumerate(match) if not x]

        data_idx: List[int] = tolist(data_idx_t)
        probs: List[float] = tolist(probs_t)
        preds: List[float] = tolist(preds_t)
        labels: List[int] = tolist(labels_t)
        is_corrupted = self.get_matching_destroyed_indices(data_idx)

        def insert_batch_pred(indices_list: List):
            # Inserts predictions for the current batch
            # Based on the index (predicts softmax correct or not)
            batch_pred: Dict[str, Any]
            batch_pred = {
                "idx": batch_idx,
                "data_idx": [],
                "probs": [],
                "preds": [],
                "labels": [],
                "is_corrupted": [],
            }

            while is_not_empty(indices_list):
                index = indices_list.pop(0)
                batch_pred["data_idx"].append(data_idx[index])
                batch_pred["probs"].append(probs[index])
                batch_pred["preds"].append(preds[index])
                batch_pred["labels"].append(labels[index])
                batch_pred["is_corrupted"].append(is_corrupted[index])
            return batch_pred

        batch_pred = {
            "correct": insert_batch_pred(indices_correct),
            "incorrect": insert_batch_pred(indices_incorrect),
        }

        self.insert_batch_predictions(self.current_epoch, batch_pred)
        return None


    def get_matching_destroyed_indices(self, indices: List[int]) -> List[int]:
        """Check if data points belong to the destroyed labels."""
        return [des_idx in self.destroyed_indices for des_idx in indices]


    def insert_batch_predictions(self, current_epoch, batch_pred):
        """Insert the predictions for the current batch to the prediction list.

        Find the correct epoch and insert the predictions.
        """
        pred_list = self.get_prediction_dict()

        if current_epoch not in pred_list["idx"]:
            pred_list["idx"].append(current_epoch)
            pred_list["epochs"].append(
                {"idx": current_epoch, "batches": [batch_pred]}
            )
        else:
            for idx, d in enumerate(pred_list["epochs"]):
                if d["idx"] == current_epoch:
                    pred_list["epochs"][idx]["batches"].append(batch_pred)

        # Update the predictions list
        self.set_prediction_dict(pred_list)


    def get_prediction_dict(self):
        """Get the predicitons list and return a copy of it."""
        return self.prediction_dict


    def set_prediction_dict(self, updated_predictions):
        """Update the predicitons list with a new list"""
        self.prediction_dict = updated_predictions


    def set_prediction_dict_value(self, key: str, value: str):
        """Set a key value pair in"""
        self.prediction_dict[key] = value


    def get_all_predictions(self):
        print("Returning all predictions made during training")
        return self.get_prediction_dict()


    def log_each_step(
        self,
        prefix: str,
        metric_name: str,
        metric: Tensor,
        show_progress=False,
    ) -> None:
        """Log the metric value used in each step."""
        if show_progress:
            self.log(
                f"{prefix}_{metric_name}",
                metric,
                prog_bar=True,
                on_step=True,
                logger=True,
            )
        else:
            self.log(
                f"{prefix}_{metric_name}", metric, on_step=True, logger=True
            )


    def step(self, batch, batch_idx, stage):
        input_ids = batch["input_ids"]
        labels = batch["label"].type(torch.int16) # HACK
        attention_mask = batch["attention_mask"]
        output = self(input_ids, attention_mask=attention_mask)

        logits = output["logits"]
        loss = self.loss(logits, labels)

        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)

        acc, prec, recall, f1 = eval(
            f"self.{stage}_metrics(preds, labels)"
        ).values()

        self.log_each_step(stage, "loss", loss, show_progress=True)
        self.log_each_step(stage, "acc", acc)
        self.log_each_step(stage, "f1", f1)
        self.log_each_step(stage, "prec", prec)
        self.log_each_step(stage, "recall", recall)

        if stage == "train":
            self.iter_counter += 1
            self.log_predictions(batch["idx"], labels, probs, preds, batch_idx)
            self.log("learning_rate", self.optim.param_groups[0]["lr"])

        return loss, acc, prec, recall, f1


    def training_step(self, batch, batch_idx):
        loss, acc, prec, rec, f1 = self.step(batch, batch_idx, stage="train")
        return {"loss": loss, "acc": acc}


    def validation_step(self, batch, batch_idx):
        loss, acc, prec, recall, f1 = self.step(batch, batch_idx, stage="val")
        return {"val_loss": loss, "val_acc": acc}


    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.log("avg_val_acc", avg_val_acc, prog_bar=True)
        return {"avg_val_loss": avg_val_loss, "avg_val_acc": avg_val_acc}


    def on_batch_end(self):
        # This is needed to use the One Cycle learning rate.
        # Without this, the learning rate will only change after every epoch
        if self.optim is not None:
            self.optim.step()

        if self.sched is not None:
            self.sched.step()


    def on_epoch_end(self):
        if self.optim is not None:
            self.optim.step()

        if self.sched is not None:
            self.sched.step()


    def test_step(self, batch, batch_idx):
        loss, acc, prec, recall, f1 = self.step(batch, batch_idx, stage="test")
        return {"test_loss": loss, "test_acc": acc}


    def test_epoch_end(self, outputs):
        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        tensorboard_logs = {
            "avg_test_loss": avg_test_loss,
            "avg_test_acc": avg_test_acc,
        }
        return {
            "avg_test_acc": avg_test_acc,
            "log": tensorboard_logs,
            "progress_bar": tensorboard_logs,
        }


    def configure_optimizers(self):
        """Return optimizers and schedulers."""
        no_decay = ["bias", "LayerNorm.weight"]
        optim_params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optim_params,
            lr=self.lr,
            eps=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=2e-5, total_steps=2000
        )

        self.sched = scheduler
        self.optim = optimizer
        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs


    def save_pretrained_model(self, path: str):
        self.model.save_pretrained(path)




def save_all_predictions(_config, predictions):
    dataset_name = _config['dataset']
    exp_id = _config['experiment_id']

    p = _config["corrupt_percentage"]
    p = 0 if p < 0 else p
    p = str(p).zfill(3)

    seed = _config["seed"]
    s = _config["label_smoothing"]
    dataname = f"seed-{seed}-{dataset_name}-p-{p}-smoothing-{s}-{exp_id}"

    output_dir = Path(_config["output_dir"])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = f"{output_dir}/{dataname}.json"

    with open(file_path, "w") as f:
        json.dump(predictions, f)
