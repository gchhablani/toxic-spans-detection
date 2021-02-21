import argparse
from evaluation.semeval2021 import f1
import pandas as pd
import numpy as np


def calculate_f1(preds_file, ground_file, out_file):
    ground_spans = pd.read_csv(ground_file)["spans"].apply(lambda x: eval(x)).values
    pred_spans = []
    with open(preds_file, "r") as f:
        for line in f:
            line_split = line.split("\t")
            pred_spans.append(eval(line_split[1]))

    f1_score = np.mean([f1(pred, gold) for pred, gold in zip(pred_spans, ground_spans)])
    with open(out_file, "w") as f:
        f.write(str(f1_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="calculate_f1_scores.py", description="Calculate F1 scores."
    )
    parser.add_argument(
        "--preds",
        type=str,
        action="store",
        help="The path for predictions.",
    )
    parser.add_argument(
        "--ground",
        type=str,
        action="store",
        help="The path for ground truths.",
    )
    parser.add_argument(
        "--out",
        type=str,
        action="store",
        help="The path for score output.",
    )
    args = parser.parse_args()
    calculate_f1(args.preds, args.ground, args.out)
