"""
The train and predict script.

This script uses datasets, omegaconf and transformers libraries.
Please install them in order to run this script.

Usage:
    $python train.py --train ./configs/train/civil_comments/default.yaml --dataset ./configs/datasets/civil_comments/default.yaml

"""
import os
import argparse
import json
import pickle as pkl

from datasets import load_metric
from omegaconf import OmegaConf

from sklearn.metrics import roc_auc_score
import torch
from transformers import (
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
)

from src.models import BertForSequenceMultilabelClassification
from src.datasets import CivilCommentsDataset
from src.utils.mapper import configmapper
from src.utils.misc import seed


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return roc_auc_score(labels, predictions)


dirname = os.path.dirname(__file__)
## Config
parser = argparse.ArgumentParser(
    prog="train_bert_sequence.py", description="Train a sequence model and predict."
)
parser.add_argument(
    "--dataset",
    type=str,
    action="store",
    help="The configuration for dataset",
    default=os.path.join(dirname, "./configs/datasets/civil_comments/default.yaml"),
)
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The configuration for trainer",
    default=os.path.join(dirname, "./configs/train/civil_comments/default.yaml"),
)

args = parser.parse_args()
train_config = OmegaConf.load(args.train)
dataset_config = OmegaConf.load(args.dataset)

seed(train_config.args.seed)

# Load datasets
print("### Loading Datasets ###")
datasets = configmapper.get_object("datasets", dataset_config.dataset_name)(
    dataset_config
)
print(datasets.get_datasets())
training_datasets = datasets.get_datasets()


# Train

print("### Getting Training Args ###")
train_args = TrainingArguments(**train_config.args)

print(train_args)


print("### Loading Tokenizer for Trainer ###")
tokenizer = AutoTokenizer.from_pretrained(
    train_config.trainer.pretrained_tokenizer_name
)


print("### Loading Model ###")
model = BertForSequenceMultilabelClassification.from_pretrained(
    train_config.model.pretrained_model_name,
    num_labels=train_config.model.num_labels,
)

print("### Loading Trainer ###")
trainer = Trainer(
    model,
    train_args,
    default_data_collator,
    training_datasets["train"],
    training_datasets["validation"],
    tokenizer,
)

print("### Training ###")
trainer.train()
trainer.save_model(train_config.trainer.save_model_name)