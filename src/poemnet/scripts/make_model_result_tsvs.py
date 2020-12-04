#!/usr/bin/env python3
"""
Takes a pkl for the poem videos and generates a directory full of TSV's,
one for each video, that contains rows with:
    (second,vid_num,block,gt,predicted,likelihood_step_1...likelihood_step_N)

Usage:
    make_model_result_tsvs.py [-h] -c CLASSNAMES -p PKLFILE -g GT -m MODEL TSVDIR

Options:
    -h             Show this screen and exit.
    -c CLASSNAMES  Specifies label for class names in the pkl file.
    -p PKLFILE     Specifies pkl results file.
    -g GT          Specifies label for ground truth in pkl file.
    -m MODEL       Specifies label for model to output results for in pkl file.

Arguments:
    TSVDIR         Directory in which to output tsvs. Cannot already exist.
"""

from itertools import chain, groupby, islice, repeat
import os
import pickle
from docopt import docopt
import numpy as np


def get_likeliest_step(likelihoods, labels):
    """Finds model's likeliest identified step for each second.

    Model outputs likelihood of step for each second in an array,
    with each index corresponding to the index of the class name
    (aka label) in labels. Use np.argmax to find index of most likely
    step and return that label.
    """
    for likelihood in likelihoods:
        yield labels[np.argmax(likelihood)]


def make_blocks(gt):
    """Divvies up gt into temporal continuous blocks

    Uses groupby() which functions like posix uniq, not SQL group by, so
    it generates a break everytime the gt annotation changes rather than
    aggregating all elements regardless of their input order."""
    # groupby gives (key, groups)
    # enumerate (starting at 1) therefore gives the block number for
    # that annotated segment.
    # We repeat that block number for the len(groups) (length of
    # the annotated block) to generate an iterable of the number for
    # that block for the length of the annotated block
    for i, grouper in enumerate(groupby(gt), 1):
        yield repeat(i, len(list(grouper[1])))


def get_blocks(gt):
    """Generates an iterable of gt block numbers.

    Will call make_blocks() to generate iterables of the number of a
    block with appropriate length for that block, then this method will
    chain those together, which generates a long iterable for block
    number (this ultimately will be the block_num column in the TSV).
    """
    return chain.from_iterable(make_blocks(gt))


def make_header(labels):
    """Creates header for TSV file of:
        (second,vid_num,block,gt,predicted,step_1,...,step_N) with step
       Step names come from the labels.
    """
    return (
        "\t".join(chain(("second", "vid_num", "block", "gt", "predicted"), labels))
        + "\n"
    )


def make_line(vid_sec, line):
    """Generates line/row for TSV.

    Parenthesis dance. Walking through (inside-to-out):
        1. combine vid_sec and line into a single flattened interable. To
        do this, use chain. chain expects everything to be an iterable,
        so make that possible with:
            1. Iterable length one of 'vid_sec' made with (vid_sec,)
            2. Iterable made with islice of items 0, 1, 2, 3 from 'line'
               (corresponds to vid_num, block, gt, predicted step)
            3. line[4] which is a list of the likelihoods for each step
        2. Make everything in the above flattened iterable a string with
        map(str, combined_iterable)
        3. Join the combined iterable together with '\t' to make a TSV
        4. End it with a '\n'
    """
    return "\t".join(map(str, chain((vid_sec,), islice(line, 4), line[4]))) + "\n"


def make_vid_maps(pkl, gt_name, labels, model_name):
    """Makes video maps, with each item being the information for a video.
    Each video's info is stored in vid_zips, corresponding to the info
    for each second (what will ultimately be a row of the tsv, that
    holds (vid_num,block, gt, model_best_prediction, [likelihoods])"""

    # Creates vid_num column which will be populated with the number of
    # the video via an infinite iterator for each video number (the gt
    # dict has keys that are just the videos number)
    vid_nums = map(repeat, pkl.get(gt_name).keys())
    # Create gt column. gt is stored in a dict, with each value being a
    # list of the gt, so get all the gt values
    gts = pkl.get(gt_name).values()
    # create block column
    blocks = map(get_blocks, gts)
    # create what will ultimately (once the list is split) the columns
    # for each step's likelihood
    model_likelihoods = pkl.get(model_name).values()
    # create column for predicted (aka most likely) step. Needs labels
    # to be able to correlate index of likeliest step to human readable
    # label
    model_best_predictions = map(get_likeliest_step, model_likelihoods, repeat(labels))
    return map(zip, vid_nums, blocks, gts, model_best_predictions, model_likelihoods)


def make_vidname(tsv_dir, num):
    """Generates video name of 'tsv_dir/video_NN.tsv'"""
    return os.path.join(tsv_dir, "video_" + f"{num:02}" + ".tsv")


def read_pkl(pkl_filename):
    """Loads pickle file."""
    with open(pkl_filename, "rb") as pkl_file:
        return pickle.load(pkl_file)


def write_tsv(tsv_file, header, vid_map):
    """Write TSV for video map"""
    with open(tsv_file, "w") as vid_file:
        vid_file.write(header)
        # use enumerate to generate a value for the second for each
        # gt/prediction
        for i, line in enumerate(vid_map):
            vid_file.write(make_line(i, line))


def write_tsvs(tsv_dir, labels, vid_maps):
    """Writes a TSV for each video map"""
    for i, vid_map in enumerate(vid_maps, 1):
        # use enumerate to generate video number
        write_tsv(make_vidname(tsv_dir, i), make_header(labels), vid_map)


def main(args):
    """Generate TSV for each video in pkl holding GT and prediction info."""
    pkl_file = args.get("-p")
    gt_name = args.get("-g")
    model_name = args.get("-m")
    class_names = args.get("-c")
    tsv_dir = args.get("TSVDIR")

    os.mkdir(tsv_dir)

    pkl = read_pkl(pkl_file)
    labels = pkl.get(class_names)
    vid_maps = make_vid_maps(pkl, gt_name, labels, model_name)

    write_tsvs(tsv_dir, labels, vid_maps)


if __name__ == "__main__":
    ARGS = docopt(__doc__)
    main(ARGS)
