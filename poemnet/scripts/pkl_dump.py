#!/usr/bin/env python3
"""Usage: pkl_dump.py [-h] PKLFILE

Uses simple heuristics to determine, from the saiil's model output,
class names used, label for the ground truth, and labels for different
models whose performance was evaluated. Can use these labels in other
scripts that require you to know their labels in the pkl to generate
output.

Arguments:
    PKLFILE    pkl produced after model runs

Options:
    -h --help

"""
import pickle
from docopt import docopt
import numpy as np


def main(pkl_filename):
    """Dumps pertinent info from model output pkl file"""
    with open(pkl_filename, "rb") as pkl_file:
        pkl = pickle.load(pkl_file)

    # pkl structure is a dictionary with keys:
    # 1) class names, ground
    #   - Value: list of class names
    # 2) Ground truth
    #   - Value: dict with (k, v), where k is video number, v is list of
    #   ground truths
    # 3) Model 1
    #    - Value: dict with (k, v), where is video number, v is
    #    numpy.ndarray of likelihoods for each class name
    # ....
    # N) Model N
    # Since each has a different value, use this as a heuristic to
    # detect the different items and print pertinent info
    for k, v in pkl.items():
        # 'list' means we found the class_names
        value_type = type(v)
        if value_type == list:
            print(f"Class names found in '{k}' are:")
            for class_name in v:
                print("\t" + class_name)
        # dict means we found either gt or model
        elif value_type == dict:
            # each k in the dict is a different video
            num_videos = len(v)
            # save type of first item (popitem returns tuple of (k,v))
            contents_type = type(v.popitem()[1])
            if contents_type == list:
                print(f"Ground truth for {num_videos} videos found in: '{k}'")
            elif contents_type == np.ndarray:
                print(f"A model's results for {num_videos} videos found in: '{k}'")


if __name__ == "__main__":
    ARGS = docopt(__doc__)
    main(ARGS["PKLFILE"])
