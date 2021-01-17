"""Class to load and process Civil Comments Dataset.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from src.utils.mapper import configmapper


@configmapper.map("datasets", "civil_comments")
class CivilCommentsDataset:
    """Implement CivilCommentsDataset dataset class."""

    def __init__(self, config):
        """Initialize the object.

        Args:
            config (omegaconf.dictconfig.DictConfig): Configuration for the dataset.

        Raises:
            AssertionError: If the tokenizer in config.model_checkpoint does not
                            belong to the PreTrainedTokenizerFast.
        """
        self.config = config
        self.datasets = load_dataset(config.dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
        self.tokenized_datasets = self.process()

    def process(self):
        """Map prepare training features to datasets.

        Returns:
            datasets.dataset_dict.DatasetDict:
                The datasets with tokenized examples
        """
        tokenized_datasets = self.datasets.map(
            self.prepare_train_features,
            batched=True,
            remove_columns=self.datasets["train"].column_names,
        )
        return tokenized_datasets

    def prepare_train_features(self, examples):
        """Generate tokenized features from examples.

        Args:
            examples (dict): The examples to be tokenized.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding:
                The tokenized features/examples after processing.
        """
        pad_on_right = self.tokenizer.padding_side == "right"
        print("### Batch Tokenizing Examples ###")
        tokenized_examples = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
        )
        tokenized_examples["labels"] = []
        column_names = sorted(list(examples.keys()))

        for i, example_column in enumerate(examples[column_names[0]]):
            tokenized_examples["labels"].append([])
            for column_name in column_names:
                if column_name != "text":
                    tokenized_examples["labels"][-1].append(
                        1 if examples[column_name][i] > 0.5 else 0
                    )

        return tokenized_examples

    def get_datasets(self):
        """Get processed datasets.

        Returns:
            datasets.dataset_dict.DatasetDict :
                The DatasetDict containing processed train and validation
        """
        return self.tokenized_datasets


if __name__ == "__main__":
    ccd = CivilCommentsDataset()