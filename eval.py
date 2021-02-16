"""Train File."""
## Imports
import argparse

# import itertools
import copy
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from evaluation.semeval2021 import f1
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
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
    "--model",
    type=str,
    action="store",
    help="The configuration for model",
    default=os.path.join(dirname, "./configs/models/forty/default.yaml"),
)
parser.add_argument(
    "--eval",
    type=str,
    action="store",
    help="The configuration for model training/evaluation",
    default=os.path.join(dirname, "./configs/trainers/forty/train.yaml"),
)
parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
    default=os.path.join(dirname, "./configs/datasets/forty/default.yaml"),
)

args = parser.parse_args()
# print(vars(args))
model_config = OmegaConf.load(args.model)
eval_config = OmegaConf.load(args.eval)
data_config = OmegaConf.load(args.data)

dataset = configmapper.get_object("datasets", data_config.name)(data_config)
untokenized_train_dataset = dataset.dataset
tokenized_train_dataset = dataset.tokenized_inputs
tokenized_test_dataset = dataset.test_tokenized_inputs

validation_spans = untokenized_train_dataset["validation"]["spans"]
validation_offsets_mapping = tokenized_train_dataset["validation"]["offset_mapping"]

model_class = configmapper.get_object("models", eval_config.model_name)
model = model_class.from_pretrained(**eval_config.pretrained_args)

tokenizer = AutoTokenizer.from_pretrained(data_config.model_checkpoint_name)
if "token" in eval_config.model_name:
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = compute_metrics_token

else:
    data_collator = default_data_collator

## Need to place data_collator
args = TrainingArguments(**eval_config.args)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset["train"],
    eval_dataset=tokenized_train_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if eval_config.with_ground:
    for key in tokenized_train_dataset.keys():
        predictions = trainer.predict(tokenized_train_dataset[key])

        preds = predictions.predictions
        preds = np.argmax(preds, axis=2)
        f1_scores = []
        with open(
            os.path.join(eval_config.save_dir, f"spans-pred_{key}.txt"), "w"
        ) as f:
            for i, pred in enumerate(preds):
                ## Batch Wise
                # print(len(prediction))
                predicted_spans = []
                for j, tokenwise_prediction in enumerate(
                    pred[: len(tokenized_train_dataset[key]["offset_mapping"][i])]
                ):
                    if tokenwise_prediction == 1:
                        predicted_spans += list(
                            range(
                                tokenized_train_dataset[key]["offset_mapping"][i][j][0],
                                tokenized_train_dataset[key]["offset_mapping"][i][j][1],
                            )
                        )
                if i == len(preds) - 1:
                    f.write(f"{i}\t{str(predicted_spans)}")
                else:
                    f.write(f"{i}\t{str(predicted_spans)}\n")
                f1_scores.append(
                    f1(
                        predicted_spans,
                        eval(untokenized_train_dataset[key]["spans"][i]),
                    )
                )
        with open(os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt")) as f:
            f.write(np.mean(f1_scores))
else:
    for key in tokenized_test_dataset.keys():
        predictions = trainer.predict(tokenized_test_dataset[key])

        preds = predictions.predictions
        preds = np.argmax(preds, axis=2)
        f1_scores = []
        with open(
            os.path.join(eval_config.save_dir, f"spans-pred_{key}.txt"), "w"
        ) as f:
            for i, pred in enumerate(preds):
                ## Batch Wise
                # print(len(prediction))
                predicted_spans = []
                for j, tokenwise_prediction in enumerate(
                    pred[: len(tokenized_test_dataset[key]["offset_mapping"][i])]
                ):
                    if tokenwise_prediction == 1:
                        predicted_spans += list(
                            range(
                                tokenized_test_dataset[key]["offset_mapping"][i][j][0],
                                tokenized_test_dataset[key]["offset_mapping"][i][j][1],
                            )
                        )
                if i == len(preds) - 1:
                    f.write(f"{i}\t{str(predicted_spans)}")
                else:
                    f.write(f"{i}\t{str(predicted_spans)}\n")
