""" Script to print test_predictions"""
test_file = "./data/tsd_test.csv"
predictions_file = "./spans-pred_spanbert_qa_gs.txt"
output_file = "./output_spans-pred_spanbert_qa_gs.txt"

from evaluation.fix_spans import _contiguous_ranges
from baselines.spacy_tagging import read_datafile

import pandas as pd
import numpy as np

texts = read_datafile(test_file, test=True)
with open(predictions_file, "r") as f:
    all_preds = f.readlines()

preds = [sorted(np.unique(eval(pred.split("\t")[1]))) for pred in all_preds]


def get_text_from_preds(text, pred):
    text_spans = []
    ranges = _contiguous_ranges(pred)
    for _range in ranges:
        text_spans.append(text[_range[0] : _range[1] + 1])
    return text_spans


text_spans = [get_text_from_preds(text, pred) for text, pred in zip(texts, preds)]


with open(output_file, "w") as f:
    for i in range(len(texts)):
        f.write("Text:\n" + str(texts[i]) + "\nSpans:\n" + str(text_spans[i]) + "\n")
