import gc
import os
import pickle as pkl
from captum import attr

import numpy as np
from captum.attr import IntegratedGradients
from datasets import Dataset

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import collections
import numpy as np

from transformers import Trainer
import argparse
from omegaconf import OmegaConf
from src.datasets import *
from src.models import *
from src.utils.mapper import configmapper
import pickle as pkl


def postprocess_spans_with_index(
    features,
    examples,
    raw_predictions,
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
    squad_v2=False,
):

    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(list(examples["id"]))}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # Let's loop over all the examples!
    for example_index in tqdm(range(len(examples))):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = examples[example_index]["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions
            # in our logits to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers,
                    # either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that
                    # is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                            "start": start_char,
                            "end": end_char,
                            "start_index": start_index,
                            "end_index": end_index,
                        }
                    )

        if len(valid_answers) > 0:
            sorted_answers = sorted(
                valid_answers, key=lambda x: x["score"], reverse=True
            )
        else:
            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid failure.
            sorted_answers = [{"text": "", "score": 0.0, "start": None, "end": None}]

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)

        if sorted_answers[0]["score"] <= min_null_score:
            sorted_answers = [
                {"text": "", "score": min_null_score, "start": None, "end": None},
            ] + sorted_answers

        predictions[examples[example_index]["id"]] = sorted_answers

    return predictions


def get_spans_token_indices_above_threshold(
    model, feature, example, threshold, tokenizer
):
    trainer = Trainer(
        model,
    )
    raw_predictions = trainer.predict(feature)
    feature.set_format(
        type=feature.format["type"], columns=list(feature.features.keys())
    )
    predictions = postprocess_spans_with_index(
        feature, example, raw_predictions.predictions, tokenizer
    )
    start_end_indices = []
    for span in predictions[0]:  ## Should Contain Only One Example
        if torch.sigmoid(torch.tensor(span["score"])) > threshold:
            start_end_indices.append(span["start_index"], span["end_index"])
    return start_end_indices


def get_token_token_indices(model, feature):
    trainer = Trainer(model)
    predictions = trainer.predict(feature)
    preds = predictions.predictions
    preds = np.argmax(preds, axis=2)
    token_indices = []
    for j, pred in enumerate(preds[0]):  ## Should Contain Only One Example
        if pred == 1:  ## Toxic
            token_indices.append(j)
    return list(set(token_indices))


def get_token_model_output(
    model, embedding_outputs, attention_masks, name="bert", position=None
):
    if name == "bert":
        extended_attention_masks = model.bert.get_extended_attention_mask(
            attention_masks, embedding_outputs.shape, torch.device("cuda")
        )
        out = model.bert.encoder(
            embedding_outputs, extended_attention_masks, return_dict=None
        )[0]

    else:
        extended_attention_masks = model.roberta.get_extended_attention_mask(
            attention_masks, embedding_outputs.shape, torch.device("cuda")
        )
        out = model.roberta.encoder(
            embedding_outputs, extended_attention_masks, return_dict=None
        )[0]

    out = model.dropout(out)
    logits = model.classifier(out)
    return F.softmax(logits, dim=1)


def get_spans_model_output(
    model, embedding_outputs, attention_masks, name="bert", position="start"
):
    if name == "bert":
        extended_attention_masks = model.bert.get_extended_attention_mask(
            attention_masks, embedding_outputs.shape, torch.device("cuda")
        )
        out = model.bert.encoder(
            embedding_outputs, extended_attention_masks, return_dict=None
        )[0]
    else:
        extended_attention_masks = model.roberta.get_extended_attention_mask(
            attention_masks, embedding_outputs.shape, torch.device("cuda")
        )
        out = model.roberta.encoder(
            embedding_outputs, extended_attention_masks, return_dict=None
        )[0]
    out = model.qa_outputs(out)
    start_logits, end_logits = out.split(1, dim=-1)
    pred = (
        F.softmax(start_logits, dim=1)
        if position == "start"
        else F.softmax(end_logits, dim=1)
    )
    return pred.reshape(-1, embedding_outputs.size(-2))


def get_embedding_outputs(self, input_ids, name="bert"):
    if name == "bert":
        return self.model.bert.embeddings(input_ids)
    else:
        return self.model.roberta.embeddings(input_ids)


def get_token_wise_attributions(
    fn,
    embedding_outputs,
    attention_masks,
    name,
    position,
    token_index,
    n_steps,
    internal_batch_size=4,
    method="riemann_right",
):
    int_grad = IntegratedGradients(
        fn,
        multiply_by_inputs=True,
    )
    attributions, approximation_error = int_grad.attribute(
        embedding_outputs,
        target=token_index,
        n_steps=n_steps,
        method=method,
        additional_forward_args=(attention_masks, name, position),
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    return {
        "attributions": attributions,
        "delta": approximation_error,
    }


def get_token_wise_importances(input_ids, attributions, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    token_wise_attributions = torch.linalg.norm(attributions, dim=1)
    token_wise_importances = token_wise_attributions / torch.sum(
        token_wise_attributions, dim=0
    ).reshape(
        -1, 1
    )  # Normalize by sum across seq_length

    return (
        tokens,
        token_wise_importances.squeeze(0).detach().cpu().numpy(),
    )


def get_word_wise_importances(
    input_ids, offset_mapping, importances, text, tokenizer, name="bert"
):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offset_mapping = offset_mapping
    word_wise_importances = []
    word_wise_offsets = []
    words = []
    if name == "bert":
        for i, token in enumerate(tokens):
            if token == "[SEP]":
                is_context = not is_context
                continue
            if token == "[CLS]":
                is_context = False
                continue

            if token == "[PAD]":
                continue

            if token.startswith("##"):
                if (
                    tokens[i - 1] == "[SEP]"
                ):  # Tokens can be broked due to stride after the [SEP]
                    word_wise_importances.append(
                        importances[i]
                    )  # We just make new entries for them
                    word_wise_offsets.append(offset_mapping[i])

                    words.append(
                        text[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                    )

                else:
                    word_wise_importances[-1] += importances[i]
                    word_wise_offsets[-1] = (
                        word_wise_offsets[-1][0],
                        offset_mapping[i][1],
                    )
                    words[-1] = text[
                        word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                    ]

    else:
        for i, token in enumerate(tokens):
            if token in ["<s>", "</s>", "<pad>"]:
                continue

            if tokens[i - 1] not in ["<s>", "</s>"] and not token.startswith("Ġ"):
                if (
                    tokens[i - 1] == "[SEP]"
                ):  # Tokens can be broked due to stride after the [SEP]
                    word_wise_importances.append(
                        importances[i]
                    )  # We just make new entries for them
                    word_wise_offsets.append(offset_mapping[i])

                    words.append(
                        text[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                    )

                else:
                    word_wise_importances[-1] += importances[i]
                    word_wise_offsets[-1] = (
                        word_wise_offsets[-1][0],
                        offset_mapping[i][1],
                    )
                    words[-1] = text[
                        word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                    ]

            else:
                word_wise_importances.append(importances[i])
                word_wise_offsets.append(offset_mapping[i])
                words.append(text[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]])
    return (
        words,
        word_wise_importances / np.sum(word_wise_importances),
    )


def get_importances(
    model,
    name,
    feature,
    example,
    fn,
    tokenizer,
    text,
    n_steps,
    type="spans",
    threshold=None,
):

    columns = ["input_ids", "attention_mask"]

    for key in columns:
        example[key] = torch.tensor(example[key], device=torch.device("cuda"))
    embedding_outputs = get_embedding_outputs(feature["input_ids"])

    if type == "spans":
        start_end_indices = get_spans_token_indices_above_threshold(
            model, feature, example, threshold, tokenizer
        )

        all_token_importances = []
        for indices in start_end_indices:
            start_pos = [indices[0]]
            end_pos = [indices[1]]
            start_attributions = get_token_wise_attributions(
                fn,
                embedding_outputs,
                feature["attention_mask"],
                name,
                "start",
                start_pos,
                n_steps,
            )
            end_attributions = get_token_wise_attributions(
                fn,
                embedding_outputs,
                feature["attention_mask"],
                name,
                "end",
                end_pos,
                n_steps,
            )
            total_attributions = (
                start_attributions["attributions"] + end_attributions["attributions"]
            )
            tokens, total_importance_scores = get_token_wise_importances(
                feature["input_ids"], total_attributions, tokenizer
            )
            all_token_importances.append(total_importance_scores)
        avg_token_importances = np.mean(all_token_importances, axis=1)
    else:
        token_indices = get_token_token_indices(model, feature)
        all_token_importances = []
        for index in token_indices:
            pos = [index]
            attributions = get_token_wise_attributions(
                fn,
                embedding_outputs,
                feature["attention_mask"],
                name,
                None,
                pos,
                n_steps,
            )
            tokens, importance_scores = get_token_wise_importances(
                feature["input_ids"], attributions, tokenizer
            )
            all_token_importances.append(importance_scores, axis=1)
        avg_token_importances = np.mean(all_token_importances, axis=1)

    word_importances = get_word_wise_importances(
        feature["input_ids"],
        feature["offset_mapping"],
        avg_token_importances,
        text,
        tokenizer,
        name,
    )

    return {
        "word_importances": word_importances,
        # batches, batch_size, len of examples
        "token_importances": (tokens, avg_token_importances),
        # batches,len of layers, batch_size, len of examples
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="integrated_gradients.py",
        description="Script to run IG on a model and an example.",
    )
    parser.add_argument(
        "--config",
        type=str,
        action="store",
        help="The configuration for IG",
    )
    args = parser.parse_args()
    ig_config = OmegaConf.load(args.config)
    data_config = ig_config.data_config
    dataset = configmapper.get_object("datasets", data_config.name)(data_config)

    if ig_config.type == "spans":
        example = Dataset.from_dict(
            [
                dataset.intermediate_test_dataset[data_config.eval_files[0]][
                    ig_config.sample_index
                ]
            ]
        )
        feature = Dataset.from_dict(
            [
                dataset.test_tokenized_inputs[data_config.eval_files[0]][
                    ig_config.sample_index
                ]
            ]
        )
        fn = get_spans_model_output
        with open(ig_config.thresh_file, "r") as f:
            thresh = float(f.read().split()[0])
        text = example[0]["context"]

    else:
        example = Dataset.from_dict(
            [dataset.test_dataset[data_config.eval_files[0]][ig_config.sample_index]]
        )
        feature = Dataset.from_dict(
            [
                dataset.test_tokenized_inputs[data_config.eval_files[0]][
                    ig_config.sample_index
                ]
            ]
        )
        fn = get_token_model_output
        thresh = None
        text = example[0]["text"]

    model_class = configmapper.get_object("models", ig_config.model_name)
    model = model_class.from_pretrained(**ig_config.pretrained_args)
    tokenizer = AutoTokenizer.from_pretrained(data_config.model_checkpoint_name)

    importances = get_importances(
        model,
        ig_config.name,  # bert or roberta
        feature,
        example,
        fn,
        tokenizer,
        text,
        ig_config.n_steps,
        ig_config.type,  # 'spans' or 'token'
        thresh,
    )
    with open(ig_config.word_out_file, "w") as f:
        pkl.dump(importances["word_importances"], f)
    with open(ig_config.token_out_file, "w") as f:
        pkl.dump(importances["token_importances"], f)
