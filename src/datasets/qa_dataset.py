import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from src.utils.mapper import configmapper

from evaluation.fix_spans import _contiguous_ranges


@configmapper.map("datasets", "qa_dataset")
class QADataset:
    def __init__(self, config, tokenizer):
        self.config = config

        self.tokenizer = tokenizer

        df_old = pd.read_csv(self.config.file_path)
        self.df = self.create_dataframe(df_old)
        self.max_length = config.max_length
        self.doc_stride = config.doc_stride
        self.pad_on_right = self.tokenizer.padding_side == "right"

        self.dataset = Dataset.from_dict(dict(self.get_tokenized_examples()))

    def create_dataframe(self, df):
        df_new = pd.DataFrame(columns=["answers", "context", "id", "question", "title"])
        id = 0
        for row_number in range(df.shape[0]):
            row = df.iloc[row_number]
            context = row["text"]
            if row["spans"][0] == "[" and row["spans"][1] == "]":
                continue
            span = row["spans"].strip("][").split(", ")
            span = [int(i) for i in span]
            question = "find offensive spans"
            title = context.split(" ")[0]
            contiguous_spans = _contiguous_ranges(span)

            for lst in contiguous_spans:
                lst = list(lst)
                dict_to_write = {}

                dict_to_write["answer_start"] = [lst[0]]
                dict_to_write["text"] = [context[lst[0] : lst[-1] + 1]]
                # print(dict_to_write)
                df_new = df_new.append(
                    {
                        "answers": dict_to_write,
                        "context": context,
                        "id": str(id),
                        "question": question,
                        "title": title,
                    },
                    ignore_index=True,
                )
                id += 1
        return df_new

    def get_tokenized_examples(self):
        # print(list(self.df["context"].values))
        tokenized_examples = self.tokenizer(
            list(self.df["question" if self.pad_on_right else "context"].values),
            list(self.df["context" if self.pad_on_right else "question"].values),
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = self.df["answers"][sample_index]
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
                while sequence_ids[token_start_index] != (
                    1 if self.pad_on_right else 0
                ):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
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

    def get_dataset(self):
        return self.dataset
