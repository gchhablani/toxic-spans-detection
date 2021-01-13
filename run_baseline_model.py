import string
import numpy as np
from baselines.models import RNNSL
from baselines.spacy_tagging import read_datafile
from evaluation.semeval2021 import f1
from keras.utils import to_categorical


def convert_spans_to_token_labels(text, spans):
    token_labels = []
    i = 0
    text = ' '.join(text.split())
    while i < len(text):
        if text[i] in string.whitespace:
            i += 1
            continue
        else:
            # print(i,text[i])
            if i in spans:
                token_labels.append(2)
            else:
                token_labels.append(1)
            while(i < len(text) and (text[i] not in string.whitespace)):
                i += 1
    assert len(text.split()) == len(token_labels)
    return token_labels

def main():
    train_file = './data/modified_train.csv'
    test_file = './data/tsd_trial.csv'
    train = read_datafile(train_file)
    test = read_datafile(test_file)
    train_token_labels = np.array([convert_spans_to_token_labels(text,spans) for spans,text in train])
    test_token_labels = np.array([convert_spans_to_token_labels(text, spans) for spans, text in test])
    train_tokens = np.array([text.split() for spans, text in train])
    test_tokens = np.array([text.split() for spans, text in test])
    train_token_labels_oh = [to_categorical(
    train_token_label, num_classes=3) for train_token_label in train_token_labels]
    test_token_labels_oh = [to_categorical(
        test_token_label, num_classes=3) for test_token_label in test_token_labels]

    rnnsl = RNNSL()
    val_data = (test_tokens, test_token_labels)
    rnnsl.fit(train_tokens,train_token_labels_oh,validation_data=val_data)
    rnnsl.tune_threshold(val_data, f1)
    print(rnnsl.threshold)
    predictions = rnnsl.get_toxic_offsets(val_data[0])
    print(np.mean([f1(predictions[i], val_data[1][i][:128])
                   for i in range(len(val_data[1]))]))
    

