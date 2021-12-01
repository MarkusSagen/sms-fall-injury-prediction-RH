#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional, Set

import pandas as pd
import pytorch_lightning as pl
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class ImdbCorruptDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str,
        max_sequence_length: int,
        batch_size: int,
        seed: int,
        corrupt_percentage: int = 0,
        corrupt_num_samples: int = 0,
        num_train_samples: int = 10000,
        num_val_samples: int = -1,  # TODO not used (num_train == num_val)
        num_label_split: int = -1,  # TODO not used
        val_is_test: bool = True,  # TODO fix for other dataset
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self.corrupt_percentage = corrupt_percentage
        self.corrupt_num_samples = corrupt_num_samples
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_label_split = num_label_split
        self.val_is_test = val_is_test

        self.data_loader_cache: Dict[str, DataLoader] = dict()
        self.dataset: Dataset
        self.destroyed_indices: Set[int] = set()


    # (required) Runs first on only one GPU
    def prepare_data(self):
        """Download files to cache first once per node."""
        AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        split = self.get_train_test_split()
        load_dataset(path="imdb", split=split)


    def sync_destroyed_num_samples_and_percentages(self) -> None:
        """Depending on which param sent in.
        Sync how many labels destroyed"""

        # N/2 since we are using equally many of label 0 and label 1.
        N = self.num_train_samples // 2

        if self.corrupt_num_samples > 0:
            self.corrupt_percentage = self.corrupt_num_samples // N
        elif self.corrupt_percentage > 0:
            self.corrupt_num_samples = (N // 100) * self.corrupt_percentage


    def get_train_test_split(self):
        """Return train test split for HF datasets."""
        split = {
            "train": "train",
            "test": "test",
        }

        if self.num_train_samples > 0:
            num_samples = self.num_train_samples // 2
            split = {
                "train": f"train[:{num_samples}]+train[-{num_samples}:]",
                "test": f"test[:{num_samples}]+test[-{num_samples}:]",
            }
        return split


    def get_and_process_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, use_fast=True
        )

        # tokenize the dataset
        def preprocess(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
            encoding = tokenizer(
                example["text"],
                max_length=self.max_sequence_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            encoding.update({"idx": idx})
            return encoding

        dataset: DatasetDict = DatasetDict()
        split = self.get_train_test_split()
        dataset = load_dataset(path="imdb", split=split)

        self.sync_destroyed_num_samples_and_percentages()
        num_destroyed = self.corrupt_num_samples

        if num_destroyed > 0:
            dataset["train"] = self.destroy_labels(
                dataset["train"], num_destroyed
            )

        return dataset.map(preprocess, batched=True, with_indices=True)


    def get_destroyed_labels(self):
        self.sync_destroyed_num_samples_and_percentages()
        self.set_destroyed_indices(self.corrupt_num_samples)
        return self.get_destroyed_indices()


    def destroy_labels(
        self,
        dataset: DatasetDict,
        num_destroyed_labels: int,
    ) -> DatasetDict:
        """Change positve label (1s) to false labels (0s)."""

        tmp_file = "./destroyed_labels.csv"
        dataset.to_csv(f"{tmp_file}", index=False)
        df = pd.read_csv(f"{tmp_file}")

        pos_csv = self.get_dataframe_maching_label(df, 1)
        neg_csv = self.get_dataframe_maching_label(df, 0)

        pos_csv.loc[: num_destroyed_labels - 1, "label"] = 0
        self.set_destroyed_indices(num_destroyed_labels)

        df = pd.concat([pos_csv, neg_csv], ignore_index=True)
        df["label"] = df["label"].astype(int)
        df.to_csv(f"{tmp_file}", index=False)

        # Load from csv creates a dict with the data split
        dataset_split = "train"
        dataset = load_dataset("csv", data_files=tmp_file)
        os.remove(tmp_file)
        return dataset[dataset_split]


    def get_dataframe_maching_label(self, df, label: int):
        return df.where(df["label"] == label).dropna().reset_index(drop=True)


    def get_destroyed_indices(self):
        """Get all indeces that have been destroyed."""
        return self.destroyed_indices


    def set_destroyed_indices(self, num_destroyed):
        """Set which indices in the dataset have been destroyed."""
        self.destroyed_indices = set(range(0, num_destroyed))


    def setup(self, stage: Optional[str] = None):
        dataset = self.get_and_process_dataset()
        dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "token_type_ids",
                "attention_mask",
                "label",
                "idx",  # To find destroyed labels
            ],
        )
        self.dataset = dataset.shuffle(seed=self.seed)


    def _create_dataloader(
        self,
        dataset_partitions: Dataset,
        split: str,
        batch_size: int,
    ):
        if split == "validation" and self.val_is_test:
            split = "test"

        dataset = dataset_partitions[split]
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
        )


    def train_dataloader(self):
        self.data_loader_cache["train"] = self._create_dataloader(
            self.dataset, split="train", batch_size=self.batch_size
        )
        return self.data_loader_cache["train"]


    def val_dataloader(self):
        self.data_loader_cache["validation"] = self._create_dataloader(
            self.dataset, split="validation", batch_size=self.batch_size
        )
        return self.data_loader_cache["validation"]


    def test_dataloader(self):
        self.data_loader_cache["test"] = self._create_dataloader(
            self.dataset, split="test", batch_size=self.batch_size
        )
        return self.data_loader_cache["test"]
