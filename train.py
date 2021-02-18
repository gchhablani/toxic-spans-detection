"""Train File."""
## Imports
import argparse

# import itertools
import copy
import os
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from copy import deepcopy
from datasets import load_metric
from evaluation.semeval2021 import f1
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    default_data_collator,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import f1_score
from src.utils.configuration import Config

from src.datasets import *
from src.models import *

from src.modules.preprocessors import *
from src.utils.mapper import configmapper

import os
import gc


def compute_metrics_token(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  ## batch_size, seq_length

    offset_wise_scores = []
    # print(len(predictions))
    for i, prediction in enumerate(predictions):
        ## Batch Wise
        # print(len(prediction))
        ground_spans = eval(validation_spans[i])
        predicted_spans = []
        for j, tokenwise_prediction in enumerate(
            prediction[: len(validation_offsets_mapping[i])]
        ):
            if tokenwise_prediction == 1:
                predicted_spans += list(
                    range(
                        validation_offsets_mapping[i][j][0],
                        validation_offsets_mapping[i][j][1],
                    )
                )
        offset_wise_scores.append(f1(predicted_spans, ground_spans))
    results_offset = np.mean(offset_wise_scores)

    true_predictions = [
        [p for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = np.mean(
        [
            f1_score(true_label, true_preds)
            for true_label, true_preds in zip(true_labels, true_predictions)
        ]
    )
    return {"Token-Wise F1": results, "Offset-Wise F1": results_offset}


dirname = os.path.dirname(__file__)  ## For Paths Relative to Current File

## Config
parser = argparse.ArgumentParser(prog="train.py", description="Train a model.")
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The configuration for model training/evaluation",
)
parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
)

args = parser.parse_args()
# print(vars(args))
train_config = OmegaConf.load(args.train)
data_config = OmegaConf.load(args.data)

print(data_config.train_files)
dataset = configmapper.get_object("datasets", data_config.name)(data_config)
untokenized_train_dataset = dataset.dataset
tokenized_train_dataset = dataset.tokenized_inputs
tokenized_test_dataset = dataset.test_tokenized_inputs


model_class = configmapper.get_object("models", train_config.model_name)

if "toxic-bert" in train_config.pretrained_args.pretrained_model_name_or_path:
    toxicbert_model = AutoModelForSequenceClassification.from_pretrained(
        train_config.pretrained_args.pretrained_model_name_or_path
    )
    train_config.pretrained_args.pretrained_model_name_or_path = "bert-base-uncased"
    model = model_class.from_pretrained(**train_config.pretrained_args)
    model.bert = deepcopy(toxicbert_model.bert)
    gc.collect()
else:
    model = model_class.from_pretrained(**train_config.pretrained_args)

tokenizer = AutoTokenizer.from_pretrained(data_config.model_checkpoint_name)
if "token" in train_config.model_name:
    validation_spans = untokenized_train_dataset["validation"]["spans"]
    validation_offsets_mapping = tokenized_train_dataset["validation"]["offset_mapping"]
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = compute_metrics_token


else:
    data_collator = default_data_collator
    compute_metrics = None

## Need to place data_collator
args = TrainingArguments(**train_config.args)
if not os.path.exists(train_config.args.output_dir):
    os.makedirs(train_config.args.output_dir)
checkpoints = sorted(
    os.listdir(train_config.args.output_dir), key=lambda x: int(x.split("-")[1])
)
if len(checkpoints) != 0:
    print("Found Checkpoints:")
    print(checkpoints)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset["train"],
    eval_dataset=tokenized_train_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if len(checkpoints) != 0:
    trainer.train(
        os.path.join(train_config.args.output_dir, checkpoints[-1])
    )  ## Load from checkpoint
else:
    trainer.train()
if not os.path.exists(train_config.save_model_path):
    os.makedirs(train_config.save_model_path)
trainer.save_model(train_config.save_model_path)