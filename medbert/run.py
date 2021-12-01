
import argparse
import datetime
import logging
import os
from pathlib import Path
from pprint import PrettyPrinter
from typing import List

import pytorch_lightning as pl
import torch
import urllib3
from dataset.imdb import ImdbCorruptDataModule
from dataset.region_halland import RegionHallandDataModule
from model import BinaryClassificationModel, save_all_predictions
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import neptune.new as neptune
from pytorch_lightning.loggers.neptune import NeptuneLogger

# Hack since running  on remote server may not enable secure connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


logger = logging.getLogger(__name__)
pprint = PrettyPrinter(indent=4, width=56, compact=True).pprint

# Gives error for running in potentiall parallism in PL else
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    def is_valid_percentage(p):
        """Check if value in range [0.0, 1.0]."""
        try:
            p = float(p)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%r not a floating-point literal" % (p,)
            )
        if p < 0.0 or p > 1.0:
            raise argparse.ArgumentTypeError(
                "%r not in range [0.0, 1.0]" % (p,)
            )
        return p

    parser = argparse.ArgumentParser()

    # PL trainer args
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["imdb", "rh", "mimic-iii", "none", "None"],
        help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--model_dir",
        default="models/medbert-rh",
        type=str,
        help="path for pre-trained models",
    )
    parser.add_argument(
        "--experiment_name",
        default="medbert-rh-label-reconstruction",
        type=str,
        help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment_id",
        default=str(datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")),
        type=str,
        help="unique identifier for the particular experiment"
    )
    parser.add_argument(
        "--loggers",
        default=None,
        choices=["wandb", "tb", "neptune", "mlflow", "all", "none", "None"],
        help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--log_every_n_steps",
        default=50,
        type=int,
        help="Determine after how many iterations PL logs the results.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default=None,
        type=str,
        help="The base directory where checkpoints will be written.",
    )
    # General
    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_batch_size",
        default=16,
        type=int,
        help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--max_sequence_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization."
        "Applies truncation and padding to max length.",
    )
    parser.add_argument(
        "--corrupt_percentage",
        type=int,
        default=0,
        help="Percentage of how many of the labels to corrupt from 1 to 0."
        "Overwritten if --corrupt_num_samples is set. "
        "Default: Don't change the labels",
    )
    parser.add_argument(
        "--corrupt_num_samples",
        type=int,
        default=0,
        help="How many of the labels to corrupt from 1 to 0."
        "Overrides the parameters --corrupt_percentage."
        "Defualt: Don't change the labels",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=10000,
        help="Defines how many training examples to restrict the dataset to.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Set folder for where predictions will be saved.",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        default=-1,
        help="UNUSED!!",
    )
    parser.add_argument(
        "--num_label_split",
        type=int,
        default=-1,
        help="UNUSED!!",
    )
    parser.add_argument(
        "--val_is_test",
        action="store_true",
        default=False,
        help="Defines if test set will be used for validation.",
    )

    # Model
    parser.add_argument(
        "--lr", default=2e-5, type=float, help="Initial learning rate for Adam"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="type (default: %(default)s)"
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="type (default: %(default)s)",
    )
    parser.add_argument(
        "--max_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--training_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override "
        "num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before "
        "performing a backward pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex)."
        "32-bit",
    )
    parser.add_argument(
        "--label_smoothing",
        default=0,
        type=is_valid_percentage,
        help="Amount of label smoothing to be applied in ranges [0.0, 1.0]."
        "Defaults to using no label smoothing (0.0)",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="If training should be stopped when validation loss decreases.",
    )
    parser.add_argument(
        "--lr_monitor",
        action="store_true",
        default=False,
        help="If learning rate should be monitored during training.",
    )
    return parser.parse_args()


def get_logging_callback(_config):
    loggers = []
    logger_choice = _config["loggers"]

    project_name = _config["experiment_name"]
    dataset = _config["dataset"]
    corruption = f"p={_config['corrupt_percentage']}"
    label_smoothing = f"smoothing={_config['label_smoothing']}"

    if logger_choice in ["neptune", "all"]:
        neptune_logger = NeptuneLogger(
            api_key = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwY2I2ODZiMS05YjliLTQxYWMtYjE2OC00Yzk4MjdkN2M1YmUifQ==',
            project_name="markussagen/medbert-label-reconstruction",
            experiment_name="rh-small-data",  # Optional,
            params= {
                "batch_size": _config['gpu_batch_size'],
                "max_epochs": _config['max_epochs'],
                "lr": _config['lr'],
                "early_stopping": _config["early_stopping"],
            },  # Optional,
            tags=["classifier", "rh", "medbert"],  # Optional,
            #upload_source_files=["**/*.py"]  # Optional,
        )
        loggers.append(neptune_logger)
    if logger_choice in ["none", "None", None]:
        loggers = None

    return loggers


def start_training(_config):
    seed = _config["seed"]
    pl.seed_everything(seed)

    callbacks: List[pl.callbacks.Callback] = []
    loggers = get_logging_callback(_config)

    lr_monitor = (LearningRateMonitor(logging_interval="step"),)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="auto",
        strict=True,
    )

    if _config["early_stopping"]:
        callbacks.append(early_stopping)

    if _config["lr_monitor"]:
        callbacks.append(lr_monitor)

    if _config["checkpoints_dir"] is not None:
        Path(_config["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
        checkpoint_dir = _config["checkpoints_dir"]

        logger.info(f"Storing checkpoints in: {checkpoint_dir}")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir
        )
        callbacks.append(checkpoint_callback)
    else:
        logger.warning(
            "No checkpoint directory specified: Defaults a predefined path...!"
        )
        checkpoint_dir = _config['model_dir'] + "-" + _config['experiment_id']
        _config['checkpoints_dir'] = checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Storing checkpoints in: {checkpoint_dir}")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir
        )
        callbacks.append(checkpoint_callback)


    if _config["dataset"] == "imdb":
        print("\n\n loading imdb\n\n")
        dataset = ImdbCorruptDataModule(
            tokenizer_name=_config["model_name"],
            max_sequence_length=_config["max_sequence_length"],
            batch_size=_config["gpu_batch_size"],
            corrupt_percentage=_config["corrupt_percentage"],
            corrupt_num_samples=_config["corrupt_num_samples"],
            num_train_samples=_config["num_train_samples"],
            num_val_samples=_config["num_val_samples"],
            val_is_test=True,
            seed=seed,
        )
    elif _config["dataset"] == "rh":
        print("loading rh dataset")
        dataset = RegionHallandDataModule(
            tokenizer_name=_config["model_name"],
            max_sequence_length=_config["max_sequence_length"],
            batch_size=_config["gpu_batch_size"],
            val_is_test=True,
            seed=seed,
        )
    else:
        print("\nNot implemented for other datasets")
        raise NotADirectoryError()

    destroyed_indices = dataset.get_destroyed_labels()
    model = BinaryClassificationModel(
        model_name=_config["model_name"],
        num_labels=2,
        lr=_config["lr"],
        label_smoothing=_config["label_smoothing"],
        batch_size=_config["gpu_batch_size"],
        weight_decay=_config["weight_decay"],
        warmup_steps=_config["warmup_steps"],
        num_workers=_config["num_workers"],
        destroyed_indices=destroyed_indices,
    )
    trainer = pl.Trainer(
        profiler=True,
        deterministic=True,
        distributed_backend=None,
        fast_dev_run=False,
        precision=16 if _config["fp16"] else 32,
        logger=loggers,
        max_epochs=_config["max_epochs"],
        gradient_clip_val=1.0,
        gpus=1 if torch.cuda.is_available() else 0,
        log_every_n_steps=_config["log_every_n_steps"],
        callbacks=callbacks,
    )
    trainer.fit(model=model, datamodule=dataset)

    pre_trained_transformer_dir = Path(f"{checkpoint_dir}/saved")
    pre_trained_transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_model(pre_trained_transformer_dir)

    all_predictions = model.get_all_predictions()
    save_all_predictions(_config, all_predictions)

    print("\n\nSaved model path:")
    print(f"{checkpoint_dir}\n\n")


def main():
    # Start training
    _config = vars(get_args())
    exp_id = _config["experiment_id"]
    start_training(_config)


if __name__ == "__main__":
    main()
