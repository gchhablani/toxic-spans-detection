from run_baseline_model import clean_predicted_text
import pandas as pd
from evaluation.fix_spans import _contiguous_ranges

predictions = pd.read_csv("output (27).csv")
preds = [sorted(list(set(eval(pred)))) for pred in predictions["spans"].values]
texts = [text for text in predictions["text"].values]

new_offsets = []
for offsets, text in zip(preds, texts):
    new_offsets.append([])
    pred_ranges = _contiguous_ranges(offsets)
    for _range in pred_ranges:
        new_offsets[-1] += list(range(_range[0], _range[1]))

new_preds = [
    clean_predicted_text(text, offsets) for text, offsets in zip(texts, new_offsets)
]

with open("spans-pred.txt", "w") as f:
    for i, pred in enumerate(new_preds):
        if i == len(new_preds) - 1:
            f.write(f"{i}\t{str(pred)}")
        else:
            f.write(f"{i}\t{str(pred)}\n")
