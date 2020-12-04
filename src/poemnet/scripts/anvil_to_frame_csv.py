#!/usr/bin/env python3

"""
Converts anvil annotation to a per-frame CSV annotation.

Anvil annotations are xml files with information on the annotator and
video stored in the <head> and operative steps stored in the <body>.

This script takes a directory which contains anvil files, then for each
file determines the annotator, the video annotated, and a list of
operative steps that contains the steps name and when it started and
ended.

It then creates a CSV with the following header:
    video name,frame number, annotator 1 name,... annotator n name

Then each row contains:
    name,frame,annotator 1 label,... annotator 2 label

CAVEAT: You should have the same annotators for each video otherwise the labels
could be scrambled (for example annotator 1 is missing, so annotator 2
is put in for those videos frames).
"""

from collections import namedtuple, defaultdict
from itertools import zip_longest
from operator import attrgetter
import os
import sys
import xml.etree.ElementTree as ET


def extract_annotator_name(anvil_filename):
    """ Extracts the annotator name from the <head> """
    tree = ET.parse(anvil_filename)
    root = tree.getroot()
    head = root.find("head")
    # The element with tag 'info' has an element with attribute coder
    # whose text stores the annotator name
    for el in head.iterfind("info"):
        if el.get("key") == "coder":
            return el.text
    raise KeyError(f"No annotator name specified in '{anvil_filename}'")


def extract_video_name(anvil_filename):
    """
    Takes an anvil file and returns the video name. Video name stored
    within the 'head' element of the xml root
    """
    tree = ET.parse(anvil_filename)
    root = tree.getroot()
    head = root.find("head")
    # The element with tag 'video' has an element with attribute src
    # whose text stores the video name
    for el in head.iterfind("video"):
        return el.get("src")
    raise KeyError(f"No video source specified in '{anvil_filename}'")


def extract_steps(anvil_filename):
    """
    Takes an anvil file and extracts a namedtuple "Step" which holds the
    name of the step (eg 'Port placement'), annotated start time, and
    annotated end time. Units by default are seconds in the anvil files.
    """
    tree = ET.parse(anvil_filename)
    root = tree.getroot()
    body = root.find("body")
    Step = namedtuple("Step", ["name", "start", "end", "unit"])
    unit = "seconds"
    # caveat, may not work if you are interested in other than the first
    # track that it contains (I made this script for ones I pruned all
    # unnecessary tracks/steps out of the file first)
    for step in body.find("track").iterfind("el"):
        start = float(step.get("start"))
        end = float(step.get("end"))
        for attribute in step.iterfind("attribute"):
            if attribute.get("name") == "step":
                name = attribute.text
        step = Step(name, start, end, unit)
        # sanity check so we only take steps with complete info
        if None in step:
            continue
        yield step


def convert_to_frames(steps, fps=30):
    """
    Converts time from seconds to frames in each Step namedtuple in list
    of steps
    """
    Step = namedtuple("Step", ["name", "start", "end", "unit"])
    for step in steps:
        yield Step(step.name, int(step.start * fps), int(step.end * fps), "frames")


def get_annotations(directory):
    """ Returns rel path for all anvil files in a directory"""
    return (
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".anvil")
    )


def make_column(steps):
    """
    Takes a list of Step namedtuples and creates a list that holds,
    where the list index correlates to the video frame, either
    the step name or 'NA'. This will become the column in the csv.
    """
    column = []
    for step in steps:
        # checks to see if the step is after the last annotated frame,
        # and if so, fills in the non-annotated frames with 'NA'
        if step.start > len(column):
            column.extend("NA" for _ in range(len(column), step.start))
        # populates the indices between start and end with the step name
        # (eg Port placement)
        column.extend(step.name for _ in range(step.start, step.end))
    return column


def main():
    """
    Takes arguments for directory with annotations and a new file to
    write a csv to, then finds all the anvil files in that directory, extracts
    from each file their annotator, video file annotated, and all the
    operative step info, converts step info into a column for a future
    dataframe, then writes a csv with all the vid infos
    """
    if len(sys.argv) != 4:
        print("Usage: ./anvil_to_frame_csv.py [fps] [anvilDir] [newCSVName]")
        sys.exit(0)
    fps = int(sys.argv[1])
    anvil_dir = sys.argv[2]
    csv_filename = sys.argv[3]

    VidInfo = namedtuple("VidInfo", ["annotator", "frame_column"])
    # key is video filename, value is list of VidInfos, which allows you
    # to have multiple annotations per video
    vid_infos = defaultdict(list)
    annotators = set()
    for anvil_filename in get_annotations(anvil_dir):
        annotator = extract_annotator_name(anvil_filename)
        video = extract_video_name(anvil_filename)
        steps = convert_to_frames(extract_steps(anvil_filename), fps)
        frame_column = make_column(steps)
        vid_infos[video].append(VidInfo(annotator, frame_column))
        annotators.add(annotator)

    with open(csv_filename, "w") as csv_file:
        # in order to ensure the correct annotations go under the
        # correct column, always work with the annotators sorted
        # alphabetically

        # write the header
        csv_file.write("video,frame," + ",".join(sorted(annotators)) + "\n")
        for video, anno_list in vid_infos.items():
            # sort the annotations for each video alphabetically by the
            # annotator
            sorted_columns = (
                vid_info.frame_column
                for vid_info in sorted(anno_list, key=attrgetter("annotator"))
            )
            # transpose the frames so we can write a row easily. Use
            # zip_longest because some annotators stopped at earlier
            # frames than others and fill with NA since they didn't
            # annotate those frames
            for num, frame_row in enumerate(
                zip_longest(*sorted_columns, fillvalue="NA")
            ):
                csv_file.write(
                    video + "," + str(num) + "," + ",".join(frame_row) + "\n"
                )


if __name__ == "__main__":
    main()
