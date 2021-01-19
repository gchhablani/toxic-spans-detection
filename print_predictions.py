""" Function to print test_predictions"""
test_file = "./data/tsd_test.csv"
predictions_file = "./spans-predroberta3ckpt.txt"
output_file = "./output_roberta3ckpt.txt"

from evaluation.fix_spans import _contiguous_ranges
from baselines.spacy_tagging import read_datafile

import pandas as pd
import numpy as np

texts = read_datafile(test_file, test=True)
with open(predictions_file, "r") as f:
    all_preds = f.read()
preds = [eval(pred.split("\t")[1]) for pred in all_preds.split("\n")]


# ## Refining and Cleaning for Spans Pred Multi Thresh 0.5
# import string


# def is_not_regular_whitespace(c):  ##From google-research/bert run_squad.py
#     if (
#         c == "\t"
#         or c == "\r"
#         or c == "\n"
#         or c == "."
#         or ord(c) == 0x202F
#         or ord(c) == 160
#         or ord(c) == 8196
#     ):
#         return True
#     return False


# new_preds = []
# for idx, pred in enumerate(preds):
#     new_preds.append([])
#     offset_idx = 0
#     while offset_idx < len(pred):
#         if is_not_regular_whitespace(texts[idx][pred[offset_idx]]):
#             offset_idx += 1
#         elif (
#             texts[idx][pred[offset_idx]].isalnum()
#             or texts[idx][pred[offset_idx]] == "*"
#             or texts[idx][pred[offset_idx]] == "#"
#         ):
#             new_preds[-1].append(pred[offset_idx])
#             j = 1
#             while (
#                 pred[offset_idx] + j < len(texts[idx])
#                 and texts[idx][pred[offset_idx] + j].isalnum()
#             ):
#                 if (
#                     offset_idx + j < len(pred)
#                     and pred[offset_idx + 1] == pred[offset_idx] + j
#                 ):
#                     new_preds[-1].append(pred[offset_idx + 1])
#                     offset_idx += 1

#                 else:
#                     new_preds[-1].append(pred[offset_idx] + j)
#                     j += 1
#             else:
#                 offset_idx += 1

#         else:
#             new_preds[-1].append(pred[offset_idx])
#             offset_idx += 1


# preds = new_preds

# final_new_preds = []
# for idx, text in enumerate(texts):
#     text_spans = []
#     final_new_preds.append([])
#     ranges = _contiguous_ranges(preds[idx])

#     for _range in ranges:
#         start = _range[0]
#         end = _range[1]  ## Inclusive
#         while start < len(text) and text[start] == " ":
#             start = start + 1
#         while end >= 0 and text[end] == " ":
#             end = end - 1
#         final_new_preds[-1] += list(range(start, end + 1))

# preds = final_new_preds


with open("./spans-pred_spanbert_qatoken_cleaned.txt", "w") as f:
    for i in range(len(preds)):
        if i != len(preds) - 1:
            f.write(f"{i}\t{str(list(preds[i]))}\n")
        else:
            f.write(f"{i}\t{str(list(preds[i]))}")

# predictions = pd.read_csv("output (27).csv")
# preds = [eval(pred) for pred in predictions["spans"].values]


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
