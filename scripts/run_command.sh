#!/usr/bin/env bash

python medbert/run.py --dataset=rh --loggers=neptune --model_name=KB/bert-base-swedish-cased --max_sequence_length=512 --fp16 --seed=42 --max_epochs=50

