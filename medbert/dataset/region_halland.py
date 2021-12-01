#!/usr/bin/env python3

from typing import Any, Dict, Optional, Set

import pytorch_lightning as pl
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer


class RegionHallandDataModule(pl.LightningDataModule):
    def __init__(
        self,
        # dataset_path: str,
        tokenizer_name: str,
        max_sequence_length: int,
        batch_size: int,
        seed: int,
        val_is_test: bool = True,
        train_data_path: str = "data/rh_train.csv",
        test_data_path: str = "data/rh_test.csv",
    ):
        super().__init__()
        # self.dataset_path = dataset_path # TODO
        self.tokenizer_name = tokenizer_name
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.seed = seed
        self.val_is_test = val_is_test
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        self.data_loader_cache: Dict[str, DataLoader] = dict()
        self.dataset: Dataset
        self.destroyed_indices: Set[int] = set()


    def get_destroyed_labels(self):
        return []


    def prepare_data(self):
        """Download files to cache first once per node."""
        AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        split = self.get_train_test_split()
        load_dataset(
            "csv",
            split=split,
            data_files={
                'train': [self.train_data_path],
                'test': [self.test_data_path],
            },
        )


    def get_train_test_split(self):
        """Return train test split for HF datasets."""
        split = {"train": "train", "test": "test"}
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

        dataset = load_dataset(
            "csv",
            split=split,
            data_files={
                'train': [self.train_data_path],
                'test': [self.test_data_path],
            },
        )

        return dataset.map(preprocess, batched=True, with_indices=True)


    def setup(self):
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

        # Get frequency of the labels and convert to weights
        targets = dataset['label'].type(torch.long) # index needs to be long, byte or bool (byte == uint8, long==int64)
        class_freq = torch.bincount(targets)
        class_weights = 1. / class_freq
        class_weights_all = class_weights[targets]

        sampler = WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            # shuffle=False,
            num_workers=4,
            sampler=sampler,
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
