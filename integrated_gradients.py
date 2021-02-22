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
from IPython.core.display import HTML
from src.utils.viz import format_word_importances, save_to_file
from evaluation.fix_spans import _contiguous_ranges


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
    columns = ["input_ids", "attention_mask", "token_type_ids"]

    features.set_format(type="torch", columns=columns, output_all_columns=True)
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
            cls_index = list(features[feature_index]["input_ids"]).index(
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
    # print(feature)
    trainer = Trainer(
        model,
    )
    # print(feature)
    raw_predictions = trainer.predict(feature)
    feature.set_format(
        type=feature.format["type"], columns=list(feature.features.keys())
    )
    # print(feature)
    predictions = postprocess_spans_with_index(
        feature, example, raw_predictions.predictions, tokenizer
    )
    start_end_indices = []
    for span in list(predictions.values())[0]:  ## Should Contain Only One Example
        if torch.sigmoid(torch.tensor(span["score"])) > threshold:
            start_end_indices.append((span["start_index"], span["end_index"]))
    return start_end_indices


def get_token_token_indices(model, feature, tokenizer):
    trainer = Trainer(model)
    predictions = trainer.predict(feature)
    preds = predictions.predictions
    preds = np.argmax(preds, axis=2)
    token_indices = []
    input_ids = feature["input_ids"][0]
    for j, pred in enumerate(preds[0]):  ## Should Contain Only One Example
        if pred == 1 and input_ids[j] != tokenizer.pad_token_id:  ## Toxic
            token_indices.append(j)
    return sorted(list(set(token_indices)))


def get_token_model_output(
    embedding_outputs, model, attention_masks, name="bert", position=None
):

    if name == "bert":
        extended_attention_masks = model.bert.get_extended_attention_mask(
            attention_masks, embedding_outputs.shape, torch.device("cuda")
        )
        # print(embedding_outputs,attention_masks,extended_attention_masks)
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
    return F.softmax(logits, dim=2)[:, :, 1]  ## Select only Toxic Logits


def get_spans_model_output(
    embedding_outputs, model, attention_masks, name="bert", position="start"
):
    if name == "bert":
        extended_attention_masks = model.bert.get_extended_attention_mask(
            attention_masks, embedding_outputs.shape, torch.device("cuda")
        ).cuda()

        out = model.bert.encoder(
            embedding_outputs, extended_attention_masks, return_dict=None
        )[0]
    else:
        extended_attention_masks = model.roberta.get_extended_attention_mask(
            attention_masks, embedding_outputs.shape, torch.device("cuda")
        ).cuda()
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


def get_embedding_outputs(model, input_ids, name="bert"):
    if name == "bert":
        return model.bert.embeddings(input_ids)
    else:
        return model.roberta.embeddings(input_ids)


def get_token_wise_attributions(
    fn,
    model,
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
        additional_forward_args=(model, attention_masks, name, position),
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    return {
        "attributions": attributions,
        "delta": approximation_error,
    }


def get_token_wise_importances(input_ids, attributions, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
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


def get_word_wise_importances_spans(
    input_ids, offset_mapping, importances, text, tokenizer, name="bert"
):
    question = text[0]
    context = text[1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    offset_mapping = offset_mapping[0]
    question_offsets = tokenizer(
        "offense", add_special_tokens=False, return_offsets_mapping=True
    )["offset_mapping"]
    i = 1
    while i < len(offset_mapping) and tokens[i] != "[SEP]":
        offset_mapping[i] = question_offsets[i - 1]
        i += 1
    word_wise_importances = []
    word_wise_offsets = []
    words = []
    is_context = False
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
                    if is_context:
                        words.append(
                            context[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                        )
                    else:
                        words.append(
                            question[
                                word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                            ]
                        )

                else:
                    word_wise_importances[-1] += importances[i]
                    word_wise_offsets[-1] = (
                        word_wise_offsets[-1][0],
                        offset_mapping[i][1],
                    )  ## Expand the offsets
                    if is_context:
                        words[-1] = context[
                            word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                        ]
                    else:
                        words[-1] = question[
                            word_wise_offsets[-1][0] : word_wise_offsets[-1][1]
                        ]

            else:
                word_wise_importances.append(
                    importances[i]
                )  # We just make new entries for them
                word_wise_offsets.append(offset_mapping[i])
                if is_context:
                    words.append(
                        context[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                    )
                else:
                    words.append(
                        question[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]
                    )
    else:
        raise NotImplementedError("Not defined for any other model name than 'bert'")
    return (
        words,
        word_wise_importances / np.sum(word_wise_importances),
        word_wise_offsets,
    )


def get_word_wise_importances(
    input_ids, offset_mapping, importances, text, tokenizer, name="bert"
):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    offset_mapping = offset_mapping[0]
    print(offset_mapping)
    word_wise_importances = []
    word_wise_offsets = []
    words = []
    if name == "bert":
        for i, token in enumerate(tokens):
            if token in ["[SEP]", "[PAD]", "[CLS]"]:
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
                word_wise_importances.append(
                    importances[i]
                )  # We just make new entries for them
                word_wise_offsets.append(offset_mapping[i])
                words.append(text[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]])

    else:
        for i, token in enumerate(tokens):
            if token in ["<s>", "</s>", "<pad>"]:
                continue

            if (
                tokens[i - 1] in ["<s>", "</s>"] and token[i] not in ["<s>", "</s>"]
            ) or token.startswith("Ġ"):
                word_wise_importances.append(
                    importances[i]
                )  # We just make new entries for them
                word_wise_offsets.append(offset_mapping[i])

                words.append(text[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]])

            else:
                word_wise_importances[-1] += importances[i]
                word_wise_offsets[-1] = (
                    word_wise_offsets[-1][0],
                    offset_mapping[i][1],
                )
                words[-1] = text[word_wise_offsets[-1][0] : word_wise_offsets[-1][1]]

    return (
        words,
        word_wise_importances / np.sum(word_wise_importances),
        word_wise_offsets,
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
    typ="spans",
    threshold=None,
):

    columns = ["input_ids", "attention_mask", "token_type_ids"]

    feature.set_format(
        type="torch", columns=columns, device="cuda", output_all_columns=True
    )
    embedding_outputs = get_embedding_outputs(model, feature["input_ids"], name)

    if typ == "spans":
        start_end_indices = get_spans_token_indices_above_threshold(
            model, feature, example, threshold, tokenizer
        )

        print(start_end_indices)
        feature.set_format(
            type="torch", columns=columns, device="cuda", output_all_columns=True
        )
        start_indices = list(set([temp[0] for temp in start_end_indices]))
        end_indices = list(set([temp[1] for temp in start_end_indices]))
        all_token_importances = np.array([])
        start_attributions_maps = {}
        end_attributions_maps = {}

        for start_index in start_indices:
            start_attributions = get_token_wise_attributions(
                fn,
                model,
                embedding_outputs,
                feature["attention_mask"],
                name,
                "start",
                start_index,
                n_steps,
            )
            start_attributions_maps[start_index] = start_attributions
        for end_index in end_indices:
            end_attributions = get_token_wise_attributions(
                fn,
                model,
                embedding_outputs,
                feature["attention_mask"],
                name,
                "end",
                end_index,
                n_steps,
            )
            end_attributions_maps[end_index] = end_attributions

        for indices in start_end_indices:
            start_pos = indices[0]
            end_pos = indices[1]
            total_attributions = (
                start_attributions_maps[start_pos]["attributions"][0]
                + end_attributions_maps[end_pos]["attributions"][0]
            )
            tokens, total_importance_scores = get_token_wise_importances(
                feature["input_ids"], total_attributions, tokenizer
            )
            all_token_importances = np.append(
                all_token_importances, total_importance_scores
            )
        all_token_importances = all_token_importances.reshape(
            len(start_end_indices), -1
        )
        avg_token_importances = np.mean(all_token_importances, axis=0)
        word_importances = get_word_wise_importances_spans(
            feature["input_ids"],
            feature["offset_mapping"],
            avg_token_importances,
            text,
            tokenizer,
            name,
        )
    else:
        token_indices = get_token_token_indices(model, feature, tokenizer)
        print(token_indices)

        feature.set_format(
            type="torch", columns=columns, device="cuda", output_all_columns=True
        )
        all_token_importances = np.array([])
        for index in token_indices:
            pos = [index]
            attributions = get_token_wise_attributions(
                fn,
                model,
                embedding_outputs,
                feature["attention_mask"],
                name,
                None,
                pos,
                n_steps,
            )
            attributions = attributions["attributions"][0]
            tokens, importance_scores = get_token_wise_importances(
                feature["input_ids"], attributions, tokenizer
            )
            all_token_importances = np.append(all_token_importances, importance_scores)
        all_token_importances = all_token_importances.reshape(len(token_indices), -1)
        avg_token_importances = np.mean(all_token_importances, axis=0)
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
        example_intermediate = dataset.intermediate_test_dataset["test"][
            ig_config.sample_index
        ]
        for key in example_intermediate.keys():
            example_intermediate[key] = [example_intermediate[key]]
        example = Dataset.from_dict(example_intermediate)
        # print(example)
        potential_feature_indices = [
            i
            for i, feature in enumerate(dataset.test_tokenized_inputs["test"])
            if feature["example_id"] == example[0]["id"]
        ]
        feature_intermediate = dataset.test_tokenized_inputs["test"][
            potential_feature_indices[0]
        ]  # Take First Feature
        for key in feature_intermediate.keys():
            feature_intermediate[key] = [feature_intermediate[key]]
        feature = Dataset.from_dict(feature_intermediate)

        fn = get_spans_model_output
        with open(ig_config.thresh_file, "r") as f:
            thresh = float(f.read().split()[0])

        text = (example["question"][0], example["context"][0])
        ignore_first_word = True

    else:
        example_intermediate = dataset.test_dataset["test"][ig_config.sample_index]
        for key in example_intermediate.keys():
            example_intermediate[key] = [example_intermediate[key]]
        example = Dataset.from_dict(example_intermediate)
        # print(example)

        feature_intermediate = dataset.test_tokenized_inputs["test"][
            ig_config.sample_index
        ]
        for key in feature_intermediate.keys():
            feature_intermediate[key] = [feature_intermediate[key]]
        feature = Dataset.from_dict(feature_intermediate)
        # print(feature)
        fn = get_token_model_output
        thresh = None
        text = example["text"][0]
        ignore_first_word = False

    if not os.path.exists(ig_config.word_out_file):

        model_class = configmapper.get_object("models", ig_config.model_name)
        model = model_class.from_pretrained(**ig_config.pretrained_args)
        model.cuda()
        model.eval()
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

        if not os.path.exists(ig_config.out_dir + "/" + str(ig_config.sample_index)):
            os.makedirs(ig_config.out_dir + "/" + str(ig_config.sample_index))
        with open(ig_config.word_out_file, "wb") as f:
            pkl.dump(importances["word_importances"], f)
        with open(ig_config.token_out_file, "wb") as f:
            pkl.dump(importances["token_importances"], f)

        words, importances, word_wise_offsets = importances["word_importances"]

    else:
        with open(ig_config.word_out_file, "rb") as f:
            words, importances, word_wise_offsets = pkl.load(f)

    ground_spans = _contiguous_ranges(
        eval(pd.read_csv(ig_config.ground_truths_file)["spans"][ig_config.sample_index])
    )

    predicted_spans = _contiguous_ranges(
        eval(
            pd.read_csv(ig_config.predictions_file, header=None, sep="\t")[1][
                ig_config.sample_index
            ]
        )
    )

    ground_text_spans = []
    predicted_text_spans = []
    if ignore_first_word:
        for span in ground_spans:
            ground_text_spans.append(text[1][span[0] : span[1] + 1])
        for span in predicted_spans:
            predicted_text_spans.append(text[1][span[0] : span[1] + 1])
    else:
        for span in ground_spans:
            ground_text_spans.append(text[span[0] : span[1] + 1])
        for span in predicted_spans:
            predicted_text_spans.append(text[span[0] : span[1] + 1])

    # print(words)
    # print(importances)
    # print(ground_text_spans)
    # print(predicted_text_spans)

    html = format_word_importances(
        words, importances, ground_text_spans, predicted_text_spans
    )

    save_to_file(html, ig_config.viz_out_file)
