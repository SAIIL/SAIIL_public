#!/usr/bin/env python3
"""
Takes a directory of videos and prints statistics on the videos'
duration. Requires ffprobe (utility from ffmpeg) to be installed.

Usage:
    video_lengths.py [-h] VIDEODIR

Options:
    -h         Show this screen and exit.

Arguments:
    VIDEODIR   Directory containing videos.
"""
import os
from statistics import mean, median, pstdev
import subprocess

from docopt import docopt


def get_videos(vid_dir):
    """Yield files with video extensions in vid_dir"""
    vid_exts = (".mp4", ".avi")
    with os.scandir(vid_dir) as it:
        for entry in it:
            if entry.name.casefold().endswith(vid_exts) and entry.is_file():
                yield entry.path


def get_video_length(video_file):
    """Return, in seconds, length of video_file."""
    command = [
        "ffprobe",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "compact=nokey=1:print_section=0",
        video_file,
    ]
    try:
        p = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError:
        raise ValueError(f"ffprobe could not calculate length of '{video_file}'.")
    # ffprobe outputs to stdout a string of vidlength in seconds as a
    # float, just want to nearest second (eg it returns 3.00000) so cast
    # string to float then int
    return int(float(p.stdout.decode().strip()))


def get_stat_funcs():
    """Return list of statistics functions"""
    return [mean, pstdev, min, median, max]


def main(args):
    """Output video length statistics."""
    vid_dir = args.get("VIDEODIR")
    lengths = list(map(get_video_length, get_videos(vid_dir)))
    print(f"For the videos in '{vid_dir}', in seconds:")
    for stat in get_stat_funcs():
        print(f"The {stat.__name__} is: {round(stat(lengths), 2)}")


if __name__ == "__main__":
    ARGS = docopt(__doc__)
    main(ARGS)
