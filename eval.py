"""Eval File."""
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
    pipeline,
    Trainer,
)
from sklearn.metrics import f1_score
from src.utils.configuration import Config

from src.datasets import *
from src.models import *

from src.modules.preprocessors import *
from src.utils.mapper import configmapper
from src.utils.postprocess_predictions import postprocess_token_span_predictions

from tqdm.auto import tqdm
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


def predict_tokens_spans(model, dataset, examples, tokenizer):
    trainer = Trainer(
        model,
    )
    raw_predictions = trainer.predict(dataset)
    dataset.set_format(
        type=dataset.format["type"], columns=list(dataset.features.keys())
    )
    final_predictions = postprocess_token_span_predictions(
        dataset, examples, raw_predictions.predictions, tokenizer
    )
    return final_predictions


dirname = os.path.dirname(__file__)  ## For Paths Relative to Current File

## Config
parser = argparse.ArgumentParser(prog="eval.py", description="Evaluate a model.")
parser.add_argument(
    "--eval",
    type=str,
    action="store",
    help="The configuration for model training/evaluation",
)

args = parser.parse_args()
# print(vars(args))
eval_config = OmegaConf.load(args.eval)
data_config = eval_config.dataset

dataset = configmapper.get_object("datasets", data_config.name)(data_config)
untokenized_train_dataset = dataset.dataset
untokenized_test_dataset = dataset.test_dataset
tokenized_train_dataset = dataset.tokenized_inputs
tokenized_test_dataset = dataset.test_tokenized_inputs

model_class = configmapper.get_object("models", eval_config.model_name)
model = model_class.from_pretrained(**eval_config.pretrained_args)
tokenizer = AutoTokenizer.from_pretrained(data_config.model_checkpoint_name)

if "tokenspans" in eval_config.model_name:
    data_collator = default_data_collator

elif "token" in eval_config.model_name:
    validation_spans = untokenized_train_dataset["validation"]["spans"]
    validation_offsets_mapping = tokenized_train_dataset["validation"]["offset_mapping"]
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = compute_metrics_token

else:
    nlp = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
    data_collator = default_data_collator

## Need to place data_collator

if not os.path.exists(eval_config.save_dir):
    os.makedirs(eval_config.save_dir)

if "tokenspans" in eval_config.model_name:
    intermediate_eval = untokenized_train_dataset["validation"].map(
        dataset.create_test_features,
        batched=True,
        remove_columns=untokenized_train_dataset["validation"].column_names,
    )
    tokenized_eval = intermediate_eval.map(
        dataset.prepare_test_features,
        batched=True,
        remove_columns=intermediate_eval.column_names,
    )

    validation_predictions = predict_tokens_spans(
        model, tokenized_eval, intermediate_eval, tokenizer
    )

    val_original = untokenized_train_dataset["validation"]
    best_threshold = -1
    best_macro_f1 = -1
    thresholds = np.linspace(0, 1, 100)
    for threshold in tqdm(thresholds):
        macro_f1 = 0
        for row_number in range(len(val_original)):
            row = val_original[row_number]
            ground_spans = eval(row["spans"])
            predicted_spans = validation_predictions[str(row_number)]
            predicted_spans = [
                span
                for span in predicted_spans
                if torch.sigmoid(torch.tensor(span["score"])) > threshold
            ]

            final_predicted_spans = []
            for span in predicted_spans:
                # print(span['start'])
                if span["start"] is not None and span["end"] is not None:
                    final_predicted_spans += list(range(span["start"], span["end"]))

            final_predicted_spans = sorted(final_predicted_spans)
            macro_f1 += f1(final_predicted_spans, ground_spans)
        avg = macro_f1 / len(val_original)
        if avg > best_macro_f1:
            best_macro_f1 = avg
            best_threshold = threshold
    with open(os.path.join(eval_config.save_dir, f"thresh.txt"), "w") as f:
        f.write(str(best_threshold) + "\n")
        f.write(str(best_macro_f1))

    topk = eval_config.topk

    if eval_config.with_ground:
        for key in untokenized_train_dataset.keys():
            f1_scores = []
            intermediate_test = untokenized_train_dataset[key].map(
                dataset.create_test_features,
                batched=True,
                remove_columns=untokenized_train_dataset[key].column_names,
            )
            tokenized_test = intermediate_test.map(
                dataset.prepare_test_features,
                batched=True,
                remove_columns=intermediate_eval.column_names,
            )

            test_predictions = predict_tokens_spans(
                model, tokenized_test, intermediate_test, tokenizer
            )

            test_original = untokenized_train_dataset[key]
            with open(
                os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
            ) as f:
                for row_number in range(len(test_original)):
                    row = val_original[row_number]
                    ground_spans = eval(row["spans"])
                    predicted_spans = test_predictions[str(row_number)]
                    predicted_spans = [
                        span
                        for span in predicted_spans
                        if torch.sigmoid(torch.tensor(span["score"])) > threshold
                    ]

                    final_predicted_spans = []
                    for span in predicted_spans:
                        # print(span['start'])
                        if span["start"] is not None and span["end"] is not None:
                            final_predicted_spans += list(
                                range(span["start"], span["end"])
                            )

                    final_predicted_spans = sorted(final_predicted_spans)
                    if row_number != len(test_original) - 1:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
                    else:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}")
                    f1_scores.append(f1(final_predicted_spans, eval(row["spans"])))
            with open(
                os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt")
            ) as f:
                f.write(str(np.mean(f1_scores)))

    else:
        for key in untokenized_test_dataset.keys():
            intermediate_test = untokenized_test_dataset[key].map(
                dataset.create_test_features,
                batched=True,
                remove_columns=untokenized_test_dataset[key].column_names,
            )
            tokenized_test = intermediate_test.map(
                dataset.prepare_test_features,
                batched=True,
                remove_columns=intermediate_eval.column_names,
            )

            test_predictions = predict_tokens_spans(
                model, tokenized_test, intermediate_test, tokenizer
            )

            test_original = untokenized_test_dataset[key]
            with open(
                os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
            ) as f:
                for row_number in range(len(test_original)):
                    row = val_original[row_number]
                    ground_spans = eval(row["spans"])
                    predicted_spans = test_predictions[str(row_number)]
                    predicted_spans = [
                        span
                        for span in predicted_spans
                        if torch.sigmoid(torch.tensor(span["score"])) > threshold
                    ]

                    final_predicted_spans = []
                    for span in predicted_spans:
                        # print(span['start'])
                        if span["start"] is not None and span["end"] is not None:
                            final_predicted_spans += list(
                                range(span["start"], span["end"])
                            )

                    final_predicted_spans = sorted(final_predicted_spans)
                    if row_number != len(test_original) - 1:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
                    else:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}")

elif "token" in eval_config.model_name:
    trainer = Trainer(
        model=model,
    )
    if eval_config.with_ground:
        for key in tokenized_train_dataset.keys():
            temp_offset_mapping = tokenized_train_dataset[key]["offset_mapping"]
            predictions = trainer.predict(tokenized_train_dataset[key])
            temp_untokenized_spans = untokenized_train_dataset[key]["spans"]

            preds = predictions.predictions
            preds = np.argmax(preds, axis=2)
            f1_scores = []
            with open(
                os.path.join(eval_config.save_dir, f"spans-pred_{key}.txt"), "w"
            ) as f:
                for i, pred in tqdm(enumerate(preds)):
                    # print(key,i)
                    ## Batch Wise
                    # print(len(prediction))
                    predicted_spans = []
                    for j, tokenwise_prediction in enumerate(
                        pred[: len(temp_offset_mapping[i])]
                    ):
                        if tokenwise_prediction == 1:
                            predicted_spans += list(
                                range(
                                    temp_offset_mapping[i][j][0],
                                    temp_offset_mapping[i][j][1],
                                )
                            )
                    if i == len(preds) - 1:
                        f.write(f"{i}\t{str(predicted_spans)}")
                    else:
                        f.write(f"{i}\t{str(predicted_spans)}\n")
                    f1_scores.append(
                        f1(
                            predicted_spans,
                            eval(temp_untokenized_spans[i]),
                        )
                    )
            with open(
                os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "w"
            ) as f:
                f.write(np.mean(f1_scores))
    else:
        for key in tokenized_test_dataset.keys():
            temp_offset_mapping = tokenized_test_dataset[key]["offset_mapping"]
            predictions = trainer.predict(tokenized_test_dataset[key])
            preds = predictions.predictions
            preds = np.argmax(preds, axis=2)
            f1_scores = []
            with open(
                os.path.join(eval_config.save_dir, f"spans-pred_{key}.txt"), "w"
            ) as f:
                for i, pred in tqdm(enumerate(preds)):
                    # print(key,i)
                    ## Batch Wise
                    # print(len(prediction))
                    predicted_spans = []
                    for j, tokenwise_prediction in enumerate(
                        pred[: len(temp_offset_mapping[i])]
                    ):
                        if tokenwise_prediction == 1:
                            predicted_spans += list(
                                range(
                                    temp_offset_mapping[i][j][0],
                                    temp_offset_mapping[i][j][1],
                                )
                            )
                    if i == len(preds) - 1:
                        f.write(f"{i}\t{str(predicted_spans)}")
                    else:
                        f.write(f"{i}\t{str(predicted_spans)}\n")
else:
    # QA Eval
    val_original = untokenized_train_dataset["validation"]

    topk = eval_config.topk
    all_predicted_spans = []
    best_threshold = -1
    best_macro_f1 = -1
    for row_number in range(len(val_original)):
        row = val_original[row_number]
        context = row["text"]
        question = "offense"
        while True and topk > 0:
            try:
                if topk == 1:
                    spans = [nlp(question=question, context=context, topk=topk)]
                else:
                    spans = nlp(question=question, context=context, topk=topk)
                break
            except:
                topk -= 1
                if topk == 0:
                    break
        all_predicted_spans.append(spans)  # [examples,topk]
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        macro_f1 = 0
        for row_number in range(len(val_original)):
            row = val_original.iloc[row_number]
            ground_spans = eval(row["spans"])
            predicted_spans = all_predicted_spans[row_number]
            predicted_spans = [
                span
                for span in predicted_spans
                if torch.sigmoid(torch.tensor(span["score"])) > threshold
            ]

            final_predicted_spans = []
            for span in predicted_spans:
                final_predicted_spans += list(range(span["start"], span["end"]))

            final_predicted_spans = sorted(spans)
            macro_f1 += f1(final_predicted_spans, ground_spans)
        avg = macro_f1 / len(val_original)
        if avg > best_macro_f1:
            best_macro_f1 = avg
            best_threshold = threshold

    with open(os.path.join(eval_config.save_dir, f"thresh.txt"), "w") as f:
        f.write(str(best_threshold) + "\n")
        f.write(str(best_macro_f1))

    if eval_config.with_ground:
        for key in untokenized_train_dataset.keys():
            f1_scores = []
            temp_test_dataset = untokenized_train_dataset[key]
            with open(
                os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
            ) as f:
                for row_number in range(len(temp_test_dataset)):
                    row = temp_test_dataset[row_number]
                    context = row["text"]
                    question = "offense"
                    while True and topk > 0:
                        try:
                            if topk == 1:
                                spans = [
                                    nlp(question=question, context=context, topk=topk)
                                ]
                            else:
                                spans = nlp(
                                    question=question, context=context, topk=topk
                                )
                            break
                        except:
                            topk -= 1
                            if topk == 0:
                                break
                    predicted_spans = spans
                    predicted_spans = [
                        span
                        for span in predicted_spans
                        if torch.sigmoid(torch.tensor(span["score"])) > best_threshold
                    ]

                    final_predicted_spans = []
                    for span in predicted_spans:
                        final_predicted_spans += list(range(span["start"], span["end"]))

                    final_predicted_spans = sorted(spans)
                    if row_number != len(temp_test_dataset) - 1:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
                    else:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}")
                    f1_scores.append(f1(final_predicted_spans, eval(row["spans"])))
            with open(
                os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt")
            ) as f:
                f.write(str(np.mean(f1_scores)))

    else:
        for key in untokenized_test_dataset.keys():
            temp_test_dataset = untokenized_test_dataset[key]
            with open(
                os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
            ) as f:
                for row_number in range(len(temp_test_dataset)):
                    row = temp_test_dataset[row_number]
                    context = row["text"]
                    question = "offense"
                    while True and topk > 0:
                        try:
                            if topk == 1:
                                spans = [
                                    nlp(question=question, context=context, topk=topk)
                                ]
                            else:
                                spans = nlp(
                                    question=question, context=context, topk=topk
                                )
                            break
                        except:
                            topk -= 1
                            if topk == 0:
                                break
                    predicted_spans = spans
                    predicted_spans = [
                        span
                        for span in predicted_spans
                        if torch.sigmoid(torch.tensor(span["score"])) > best_threshold
                    ]

                    final_predicted_spans = []
                    for span in predicted_spans:
                        final_predicted_spans += list(range(span["start"], span["end"]))

                    final_predicted_spans = sorted(spans)
                    if row_number != len(temp_test_dataset) - 1:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
                    else:
                        f.write(f"{row_number}\t{str(final_predicted_spans)}")
