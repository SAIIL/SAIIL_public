#!/usr/bin/env python3
"""Takes a directory of annotations and changes coder to a single name.

Useful when trying to compute phase durations statistics from the ground
truth for a bunch of videos. Allows us to repurpose the
anvil_to_frame.csv, which puts the annotations in for each coder.
However we aren't comparing multiple annotators, just want to translate
all the annotation info into one column so this makes a temporary set of
annotations with a standardized annotator. Can then do:
    anvil_to_frame_csv.py 1 annotations csvfile to generate a csv to do
    stats in R on
"""

import os
import re
import sys


def standardize_annotator(annotation, annotator="annotator"):
    """Substitute coder name in annotation for annotator."""
    # Anvil line for coder: <info key="coder" type="String">TMW</info>
    # regex matches line prior to the coder name in group 1, coder name
    # in group 2, and remainder in group 3, so return string with group
    # 2 substituted as 'annotator'
    return re.sub(
        r"(<.+key=\"coder\".+>)(.+)(</info>)", r"\1" + annotator + r"\3", annotation
    )


def main():
    """Takes a directory of annotations and standardizes the annotator.

    Expects the new directory to already exist.
    """
    if len(sys.argv) != 3:
        print("Usage: make_anvil_same_annotator.py [dir with annos] [new dir]")
        sys.exit(2)

    old_dir = sys.argv[1]
    new_dir = sys.argv[2]
    os.makedirs(new_dir, exist_ok=True)

    for filename in os.listdir(old_dir):
        if not filename.casefold().endswith((".anvil", ".xml")):
            continue
        new_filename = os.path.join(new_dir, filename)
        with open(os.path.join(old_dir, filename), "r") as old_file, open(
            new_filename, "w"
        ) as new_file:
            annotation = old_file.read()
            new_file.write(standardize_annotator(annotation))


if __name__ == "__main__":
    main()
