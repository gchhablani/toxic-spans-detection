from src.utils.mapper import configmapper
from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset, Dataset
from evaluation.fix_spans import _contiguous_ranges


@configmapper.map("datasets", "toxic_spans_spans")
class ToxicSpansSpansDataset:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint_name
        )

        self.dataset = load_dataset("csv", data_files=dict(self.config.train_files))
        self.test_dataset = load_dataset("csv", data_files=dict(self.config.eval_files))

        temp_key_train = list(self.dataset.keys())[0]
        self.intermediate_dataset = self.dataset.map(
            self.create_train_features,
            batched=True,
            batch_size=1000000,  ##Unusually Large Batch Size ## Needed For Correct ID mapping
            remove_columns=self.dataset[temp_key_train].column_names,
        )

        temp_key_test = list(self.test_dataset.keys())[0]
        self.intermediate_test_dataset = self.test_dataset.map(
            self.create_test_features,
            batched=True,
            batch_size=1000000,  ##Unusually Large Batch Size ## Needed For Correct ID mapping
            remove_columns=self.test_dataset[temp_key_test].column_names,
        )

        self.tokenized_inputs = self.intermediate_dataset.map(
            self.prepare_train_features,
            batched=True,
            remove_columns=self.intermediate_dataset[temp_key_train].column_names,
        )
        self.test_tokenized_inputs = self.intermediate_test_dataset.map(
            self.prepare_test_features,
            batched=True,
            remove_columns=self.intermediate_test_dataset[temp_key_test].column_names,
        )

    def create_train_features(self, examples):
        features = {"context": [], "id": [], "question": [], "title": []}
        id = 0
        # print(examples)
        for row_number in range(len(examples["text"])):
            context = examples["text"][row_number]
            question = "offense"
            title = context.split(" ")[0]
            span = eval(examples["spans"][row_number])
            contiguous_spans = _contiguous_ranges(span)
            for lst in contiguous_spans:
                lst = list(lst)
                dict_to_write = {}

                dict_to_write["answer_start"] = [lst[0]]
                dict_to_write["text"] = [context[lst[0] : lst[-1] + 1]]
                # print(dict_to_write)
                if "answers" in features.keys():
                    features["answers"].append(dict_to_write)
                else:
                    features["answers"] = [
                        dict_to_write,
                    ]
                features["context"].append(context)
                features["id"].append(str(id))
                features["question"].append(question)
                features["title"].append(title)
                id += 1

        return features

    def create_test_features(self, examples):
        features = {"context": [], "id": [], "question": [], "title": []}
        id = 0
        for row_number in range(len(examples["text"])):
            context = examples["text"][row_number]
            question = "offense"
            title = context.split(" ")[0]
            features["context"].append(context)
            features["id"].append(str(id))
            features["question"].append(question)
            features["title"].append(title)
            id += 1
        return features

    def prepare_train_features(self, examples):
        """Generate tokenized features from examples.

        Args:
            examples (dict): The examples to be tokenized.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding:
                The tokenized features/examples after processing.
        """
        # Tokenize our examples with truncation and padding, but keep the
        # overflows using a stride. This results in one example possible
        # giving several features when a context is long, each of those
        # features having a context that overlaps a bit the context
        # of the previous feature.
        pad_on_right = self.tokenizer.padding_side == "right"
        print("### Batch Tokenizing Examples ###")
        tokenized_examples = self.tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            **dict(self.config.tokenizer_params),
        )

        # Since one example might give us several features if it has
        # a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to
        # character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of
            # the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span
                # (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and
                    # stoken_end_index to the two ends of the answer.
                    # Note: we could go after the last offset
                    # if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_test_features(self, examples):

        """Generate tokenized validation features from examples.

        Args:
            examples (dict): The validation examples to be tokenized.

        Returns:
            transformers.tokenization_utils_base.BatchEncoding:
                The tokenized features/examples for validation set after processing.
        """

        # Tokenize our examples with truncation and maybe
        # padding, but keep the overflows using a stride.
        # This results in one example possible giving several features
        # when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        print("### Tokenizing Validation Examples")
        pad_on_right = self.tokenizer.padding_side == "right"
        tokenized_examples = self.tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            **dict(self.config.tokenizer_params),
        )

        # Since one example might give us several features if it has a long context,
        #  we need a map from a feature to its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans,
            # this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(str(examples["id"][sample_index]))

            # Set to None the offset_mapping that are not part
            # of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
