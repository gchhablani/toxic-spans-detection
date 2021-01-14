import math
import os
import torch
from src.modules.optimizers import *
from src.modules.embeddings import *
from src.modules.schedulers import *
from src.modules.tokenizers import *
from src.modules.metrics import *
from src.modules.losses import *
from src.utils.misc import *
from src.utils.logger import Logger
from src.utils.mapper import configmapper
from src.utils.configuration import Config

from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer, default_data_collator


@configmapper.map("trainers", "qa_trainer")
class QATrainer:
    def __init__(self, config):
        self.config = config
        self.train_config = self.config.train
        # self.val_config = self.config.val
        self.tokenizer = configmapper.get_object(
            "tokenizers", self.config.train.tokenizer.name
        ).from_pretrained(**self.config.train.tokenizer.init_params.as_dict())

    def train(self, model, train_dataset, val_dataset=None):
        args = TrainingArguments(**self.train_config.trainer_args.as_dict())
        data_collator = default_data_collator
        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        trainer.train()
        trainer.save_model(self.config.train.save_path)
