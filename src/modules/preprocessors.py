import pandas as pd
from src.modules.tokenizers import *
from src.modules.embeddings import *
from src.utils.mapper import configmapper

import transformers

from datasets import Dataset, DatasetDict


class Preprocessor:
    def preprocess(self):
        pass


@configmapper.map("preprocessors", "glove")
class GlovePreprocessor(Preprocessor):
    """GlovePreprocessor."""

    def __init__(self, config):
        """
        Args:
            config (src.utils.module.Config): configuration for preprocessor
        """
        super(GlovePreprocessor, self).__init__()
        self.config = config
        self.tokenizer = configmapper.get_object(
            "tokenizers", self.config.main.preprocessor.tokenizer.name
        )(**self.config.main.preprocessor.tokenizer.init_params.as_dict())
        self.tokenizer_params = (
            self.config.main.preprocessor.tokenizer.init_vector_params.as_dict()
        )

        self.tokenizer.initialize_vectors(**self.tokenizer_params)
        self.embeddings = configmapper.get_object(
            "embeddings", self.config.main.preprocessor.embedding.name
        )(
            self.tokenizer.text_field.vocab.vectors,
            self.tokenizer.text_field.vocab.stoi[self.tokenizer.text_field.pad_token],
        )

    def preprocess(self, model_config, data_config):
        train_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.train, self.tokenizer
        )
        val_dataset = configmapper.get_object("datasets", data_config.main.name)(
            data_config.val, self.tokenizer
        )
        model = configmapper.get_object("models", model_config.name)(
            self.embeddings, **model_config.params.as_dict()
        )

        return model, train_dataset, val_dataset


@configmapper.map("preprocessors", "qa_preprocessor")
class QAPreprocessor(Preprocessor):
    """QAPreprocessor."""

    def __init__(self, config):
        """
        Args:
            config (src.utils.module.Config): configuration for preprocessor
        """
        super(QAPreprocessor, self).__init__()
        self.custom_config = config

        self.tokenizer = configmapper.get_object(
            "tokenizers", self.custom_config.main.preprocessor.tokenizer.name
        ).from_pretrained(
            **self.custom_config.main.preprocessor.tokenizer.init_params.as_dict()
        )

        # # assert isinstance(
        # #     self.tokenizer, transformers.PreTrainedTokenizerFast
        # # ), "Tokenizer should be of type PreTrainedTokenizerFast"
        # self.max_length = None
        self.doc_stride = None
        self.pad_on_right = None

    def preprocess(self, model_config, data_config):

        train_data = configmapper.get_object("datasets", data_config.main.name)(
            data_config.train, self.tokenizer
        ).get_dataset()
        val_data = configmapper.get_object("datasets", data_config.main.name)(
            data_config.val, self.tokenizer
        ).get_dataset()
        model = configmapper.get_object("models", model_config.name).from_pretrained(
            **model_config.params.as_dict()
        )
        print("Total Training Samples: ", len(train_data))
        print("Total Validation Samples: ", len(val_data))
        print("An example of the train data: ", train_data[0])
        return model, train_data, val_data