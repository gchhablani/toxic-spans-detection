import string
import numpy as np
from baselines.models import RNNSL
from baselines.spacy_tagging import read_datafile

from evaluation.semeval2021 import f1  ## WRONG F1, ONLY USED FOR OFFSETS
from sklearn.metrics import f1_score
from evaluation.fix_spans import _contiguous_ranges
from keras.utils import to_categorical


def convert_spans_to_token_labels(text, spans=None, test=False):
    token_labels = []
    token_to_offsets_map = []
    i = 0
    text = " ".join(text.split())
    while i < len(text):
        if text[i] in string.whitespace:
            i += 1
            continue
        else:
            # print(i,text[i])
            token_to_offsets_map.append(
                [
                    i,
                ]
            )
            if not test:
                if i in spans:
                    token_labels.append(2)  ##Toxic
                else:
                    token_labels.append(1)  ##Non-Toxi
            while i < len(text) and (text[i] not in string.whitespace):
                i += 1
            token_to_offsets_map[-1].append(i)  ##Not Inclusive
    if not test:
        assert len(text.split()) == len(token_labels)
        return token_labels, token_to_offsets_map
    else:
        return token_to_offsets_map


def dev():
    train_file = "./data/modified_train.csv"
    test_file = "./data/tsd_trial.csv"
    train = read_datafile(train_file)
    test = read_datafile(test_file)
    train_token_labels = np.array(
        [convert_spans_to_token_labels(text, spans) for spans, text in train]
    )
    test_token_labels = np.array(
        [convert_spans_to_token_labels(text, spans) for spans, text in test]
    )
    train_tokens = np.array([text.split() for spans, text in train])
    test_tokens = np.array([text.split() for spans, text in test])
    train_token_labels_oh = [
        to_categorical(train_token_label, num_classes=3)
        for train_token_label in train_token_labels
    ]
    test_token_labels_oh = [
        to_categorical(test_token_label, num_classes=3)
        for test_token_label in test_token_labels
    ]

    rnnsl = RNNSL()

    run_df = rnnsl.fit(
        train_tokens,
        train_token_labels_oh,
        validation_data=(test_tokens, test_token_labels_oh),
    )
    run_df.to_csv("RNNSL_Run.csv", index=False)
    # rnnsl.set_up_preprocessing(train_tokens)
    # rnnsl.model = rnnsl.build()

    val_data = (test_tokens, test_token_labels)
    rnnsl.tune_threshold(val_data, f1_score)
    print("Threshold: ", rnnsl.threshold)

    predictions = rnnsl.get_toxic_offsets(val_data[0])
    print(
        "F1_score :",
        np.mean(
            [
                f1_score(predictions[i], val_data[1][i][:128])
                for i in range(len(val_data[1]))
            ]
        ),
    )


def predict():
    train_file = "./data/tsd_train.csv"
    dev_file = "./data/tsd_trial.csv"
    test_file = "./data/tsd_test.csv"

    train = read_datafile(train_file)
    dev = read_datafile(dev_file)

    reduced_train = []
    for i in train:
        if i not in dev:
            reduced_train.append(i)

    ## Tune Threshold on Dev
    reduced_train_token_labels, reduced_train_offset_mapping = list(
        zip(
            *[
                convert_spans_to_token_labels(text, spans)
                for spans, text in reduced_train
            ]
        )
    )

    dev_token_labels, dev_offset_mapping = list(
        zip(*[convert_spans_to_token_labels(text, spans) for spans, text in dev])
    )
    # print(dev_token_labels)
    # print(dev_offset_mapping)
    reduced_train_tokens = [
        [
            word.lower().translate(
                str.maketrans("", "", string.punctuation)
            )  ## Remove Punctuation and make into lower case
            for word in text.split()
        ]
        for spans, text in reduced_train
    ]
    dev_tokens = [
        [
            word.lower().translate(
                str.maketrans("", "", string.punctuation)
            )  ## Remove Punctuation and make into lower case
            for word in text.split()
        ]
        for spans, text in dev
    ]
    reduced_train_token_labels_oh = [
        to_categorical(train_token_label, num_classes=3)
        for train_token_label in reduced_train_token_labels
    ]
    dev_token_labels_oh = [
        to_categorical(dev_token_label, num_classes=3)
        for dev_token_label in dev_token_labels
    ]

    rnnsl = RNNSL()

    # run_df = rnnsl.fit(
    #     reduced_train_tokens,
    #     reduced_train_token_labels_oh,
    #     validation_data=(dev_tokens, dev_token_labels_oh),
    # )
    # run_df.to_csv("RNNSL_Run.csv", index=False)
    rnnsl.set_up_preprocessing(reduced_train_tokens)
    rnnsl.model = rnnsl.build()

    val_data = (dev_tokens, dev_token_labels)
    # rnnsl.tune_threshold(val_data, f1_score)
    print("Threshold: ", rnnsl.threshold)
    token_predictions = rnnsl.get_toxic_offsets(
        val_data[0],
    )  ## Word Level Toxic Offsets
    print(
        "F1_score Word Wise on Dev Tokens :",
        np.mean(
            [
                f1_score(token_predictions[i], val_data[1][i][:128])
                for i in range(len(val_data[1]))
            ]
        ),
    )

    # print(reduced_train_tokens)
    # print(reduced_train_offset_mapping)

    for example in range(len(reduced_train_tokens)):
        print(
            reduced_train_tokens[example],
        )

        print(
            [
                reduced_train[example][1][
                    reduced_train_offset_mapping[example][token][
                        0
                    ] : reduced_train_offset_mapping[example][token][1]
                ]
                .lower()
                .translate(str.maketrans("", "", string.punctuation))
                for token in range(len(reduced_train_tokens[example]))
            ]
        )
    exit()

    # dev_offset_mapping #map token index to offsets
    offset_predictions = []
    for i in range(len(dev_tokens)):
        offset_predictions.append([])
        pred_spans = list(range(dev_offset_mapping[i][0], dev_offset_mapping[i][1]))
        contiguous_spans = _contiguous_ranges(pred_spans)
        print("Text: ", dev_tokens[i])
        for j in contiguous_spans:
            print(
                "Span: ",
                j,
                " ",
                dev[i][1][contiguous_spans[j][0] : contiguous_spans[i][1] + 1],
            )

        if token_predictions[i] == rnnsl.toxic_label:
            offset_predictions[-1] += list(
                range(dev_offset_mapping[i][0], dev_offset_mapping[i][1])
            )
    dev_spans = [spans for spans, text in dev]
    avg_dice_score = np.mean(
        [f1(preds, gold) for preds, gold in zip(offset_predictions, dev_spans)]
    )

    print("Avg Dice Score on Dev: ", avg_dice_score)

    ## Test predictions
    print("Training on both train and dev for predictions!")
    combo = reduced_train + dev

    combo_token_labels, combo_offset_mapping = list(
        zip(*[convert_spans_to_token_labels(text, spans) for spans, text in combo])
    )
    combo_tokens = [
        [
            word.lower().translate(
                str.maketrans("", "", string.punctuation)
            )  ## Remove Punctuation and make into lower case
            for word in text.split()
        ]
        for spans, text in combo_tokens
    ]
    combo_token_labels_oh = [
        to_categorical(combo_token_label, num_classes=3)
        for combo_token_label in combo_token_labels
    ]

    rnnsl_2 = RNNSL()
    # pred_df = rnnsl_2.fit(combo_tokens, combo_token_labels_oh)
    # pred_df.to_csv("RNNSL_Pred.csv", index=False)
    # rnnsl_2.threshold = rnnsl.threshold  ##Replace with tuned threshold
    rnnsl_2.set_up_preprocessing(combo_tokens)
    rnnsl_2.model = rnnsl_2.build()

    print("Predicting on Test")
    test = read_datafile(test_file, test=True)
    test_tokens = [
        [
            word.lower().translate(
                str.maketrans("", "", string.punctuation)
            )  ## Remove Punctuation and make into lower case
            for word in text.split()
        ]
        for spans, text in test
    ]
    test_offset_mapping = convert_spans_to_token_labels(test_tokens, test=True)
    final_token_predictions = rnnsl_2.get_toxic_offsets(test_tokens)

    final_offset_predictions = []
    for i in range(len(test_tokens)):
        final_offset_predictions.append([])
        pred_spans = list(range(test_offset_mapping[i][0], test_offset_mapping[i][1]))
        contiguous_spans = _contiguous_ranges(pred_spans)
        print("Text: ", test_tokens[i])
        for j in contiguous_spans:
            print(
                "Span: ",
                j,
                " ",
                test[i][1][contiguous_spans[j][0] : contiguous_spans[i][1] + 1],
            )

        if final_token_predictions[i] == rnnsl.toxic_label:
            final_offset_predictions[-1] += list(
                range(test_offset_mapping[i][0], test_offset_mapping[i][1])
            )
    # 0\t[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]\n
    with open("./test_predictions.txt", "w") as f:
        for i, spans in enumerate(final_offset_predictions):
            f.write(f"{i}\t{str(spans)}\n")


if __name__ == "__main__":
    predict()
