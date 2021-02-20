import os
import argparse
from omegaconf import OmegaConf


def binary_intersection(lst1, lst2):
    lst3 = list(set([value for value in lst1 if value in lst2]))
    return lst3


def binary_union(lst1, lst2):
    lst3 = list(set(lst1 + lst2))
    return lst3


def combine(files, type="union"):
    text = {}
    if type == "union":
        fn = binary_union
    else:
        fn = binary_intersection
    for fil in files:
        with open(fil, "r") as f:
            for line in f:
                line_split = line.split("\t")
                if int(line_split[0]) in text:
                    text[int(line_split[0])] = fn(
                        text[int(line_split[0])], eval(line_split[1])
                    )
                else:
                    text[int(line_split[0])] = eval(line_split[1])
    return text


def write_dict_to_file(text, path):
    with open(path, "w") as f:
        for id, spans in text.items():
            if id != len(text) - 1:
                f.write(f"{id}\t{str(spans)}\n")
            else:
                f.write(f"{id}\t{str(spans)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="combine_preds.py", description="Combine span predictions."
    )
    parser.add_argument(
        "--config",
        type=str,
        action="store",
        help="The configuration for combining predictions.",
    )
    args = parser.parse_args()
    combine_config = OmegaConf.load(args.config)
    text = combine(combine_config.files, combine_config.type)
    dir = "/".join(combine_config.path.split("/")[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    write_dict_to_file(text, combine_config.path)
