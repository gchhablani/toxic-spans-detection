import collections
import numpy as np
from tqdm.auto import tqdm


def postprocess_token_span_predictions(
    features,
    examples,
    raw_predictions,
    tokenizer,
    n_best_size=20,
    max_answer_length=30,
    squad_v2=False,
):
    all_start_logits, all_end_logits, token_logits = raw_predictions
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
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
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
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "qa_score": (
                                start_logits[start_index] + end_logits[end_index]
                            )
                            / 2,
                            "token_score": np.mean(
                                [
                                    token_logits[example_index][token_index][1]
                                    for token_index in range(start_index, end_index + 1)
                                ]
                            ),
                            "score": (start_logits[start_index] + end_logits[end_index])
                            / 2
                            + np.mean(
                                [
                                    token_logits[example_index][token_index][1]
                                    for token_index in range(start_index, end_index + 1)
                                ]
                            ),
                            "text": context[start_char:end_char],
                            "start": start_char,
                            "end": end_char,
                        }
                    )

        if len(valid_answers) > 0:
            sorted_answers = sorted(
                valid_answers, key=lambda x: x["score"], reverse=True
            )
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            sorted_answers = [{"text": "", "score": 0.0, "start": None, "end": None}]
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if sorted_answers[0]["score"] <= min_null_score:
            sorted_answers = [
                {"text": "", "score": min_null_score, "start": None, "end": None},
            ] + sorted_answers
        predictions[examples[example_index]["id"]] = sorted_answers

    return predictions


def postprocess_multi_span_predictions(
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