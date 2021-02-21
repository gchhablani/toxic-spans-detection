from src.utils.mapper import configmapper
from transformers import AutoTokenizer
from datasets import load_dataset


@configmapper.map("datasets", "toxic_spans_tokens")
class ToxicSpansTokenDataset:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint_name
        )
        self.dataset = load_dataset("csv", data_files=dict(self.config.train_files))
        self.test_dataset = load_dataset("csv", data_files=dict(self.config.eval_files))

        self.tokenized_inputs = self.dataset.map(
            self.tokenize_and_align_labels_for_train, batched=True
        )
        self.test_tokenized_inputs = self.test_dataset.map(
            self.tokenize_for_test, batched=True
        )

    def tokenize_and_align_labels_for_train(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"], **self.config.tokenizer_params
        )

        # tokenized_inputs["text"] = examples["text"]
        example_spans = []
        labels = []
        offsets_mapping = tokenized_inputs["offset_mapping"]
        for i, offset_mapping in enumerate(offsets_mapping):
            labels.append([])

            spans = eval(examples["spans"][i])
            example_spans.append(spans)
            if self.config.label_cls:
                cls_label = (
                    1
                    if (
                        len(examples["text"][i]) > 0
                        and len(spans) / len(examples["text"][i])
                        > self.config.cls_threshold
                    )
                    else 0
                )  ## Make class label based on threshold
            else:
                cls_label = -100
            for j, offsets in enumerate(offset_mapping):
                if tokenized_inputs["input_ids"][i][j] == self.tokenizer.cls_token_id:
                    labels[-1].append(cls_label)
                elif offsets[0] == offsets[1] and offsets[0] == 0:
                    labels[-1].append(-100)  ## SPECIAL TOKEN
                else:
                    toxic_offsets = [x in spans for x in range(offsets[0], offsets[1])]
                    ## If any part of the the token is in span, mark it as Toxic
                    if (
                        len(toxic_offsets) > 0
                        and sum(toxic_offsets) / len(toxic_offsets)
                        > self.config.token_threshold
                    ):
                        labels[-1].append(1)
                    else:
                        labels[-1].append(0)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_for_test(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["text"], **self.config.tokenizer_params
        )
        return tokenized_inputs
