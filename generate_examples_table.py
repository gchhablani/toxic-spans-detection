from evaluation.fix_spans import _contiguous_ranges
import pandas as pd
import numpy as np
from ast import literal_eval
import os
import sys
from evaluation.metrics import f1


def get_spans_from_offsets(text, offsets):
    text_spans = []
    ranges = _contiguous_ranges(offsets)
    for _range in ranges:
        text_spans.append(text[_range[0] : _range[1] + 1])
    return text_spans


test_file = "./data/tsd_test_spans.csv"
test_df = pd.read_csv(test_file)
test_df["spans"] = test_df["spans"].apply(lambda x: literal_eval(x))
output_file = "./example_table.txt"


all_preds = []
for predictions_file in sorted(os.listdir("results/test_predictions")):
    with open(os.path.join("results/test_predictions", predictions_file), "r") as f:
        all_preds.append(f.readlines())

# best_example_id = -1
# best_f1_std = 0
# second_example_id = -1
# second_f1_std = 0
# third_example_id = -1
# third_f1_std = 0
example_f1_variance = {}
for example_id in range(2000):
    # example_id = int(sys.argv[1])  # something between 0 and 1999
    f1_scores = []
    example_text = test_df.iloc[example_id]["text"]
    gold = test_df.iloc[example_id]["spans"]

    for idx, predictions_file in enumerate(
        sorted(os.listdir("results/test_predictions"))
    ):
        # print(predictions_file)
        pred = all_preds[idx][example_id]
        pred = sorted(np.unique(literal_eval(pred.split("\t")[1])))

        f1_scores.append(f1(pred, gold))

    example_f1_variance[example_id] = np.std(f1_scores)
    # if np.std(f1_scores) > best_f1_std:
    #     third_example_id = second_example_id
    #     third_f1_std = second_f1_std

    #     second_example_id = best_example_id
    #     second_f1_std = best_f1_std

    #     best_example_id = example_id
    #     best_f1_std = np.std(f1_scores)
    # elif np.std(f1_scores) > second_f1_std:
    #     third_example_id = second_example_id
    #     third_f1_std = second_f1_std

    #     second_example_id = example_id
    #     second_f1_std = np.std(f1_scores)

    # elif np.std(f1_scores) > third_f1_std:
    #     third_example_id = example_id
    #     third_f1_std = np.std(f1_scores)


# from scipy.stats import mode

val = np.median(np.array(list(example_f1_variance.values())))
print(val)
for k, v in example_f1_variance.items():
    if v == val:
        example_id = k
        break
# sorted_example_f1_variance = [
#     k for k, v in sorted(example_f1_variance.items(), key=lambda x: x[1], reverse=True)
# ]
# example_id = sorted_example_f1_variance[15]
example_text = test_df.iloc[example_id]["text"]
gold = test_df.iloc[example_id]["spans"]
example_spans = get_spans_from_offsets(example_text, gold)
file_names = ["text_" + str(example_id), "ground"]
spans = [example_text, example_spans]

for idx, predictions_file in enumerate(sorted(os.listdir("results/test_predictions"))):
    pred = all_preds[idx][example_id]
    pred = sorted(np.unique(literal_eval(pred.split("\t")[1])))

    text_spans = get_spans_from_offsets(example_text, pred)
    file_names.append(
        predictions_file.replace("_spans-pred.txt", "").replace("original_test-", "")
    )
    spans.append(text_spans)

df = pd.DataFrame.from_dict({"file_names": file_names, "spans": spans})
df.to_latex(output_file, index=False)
s = df.to_markdown(tablefmt="github")
with open("example_markdown_table.md", "w") as f:
    f.write(s)