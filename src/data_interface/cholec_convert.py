import argparse
import csv
import glob
import os
import uuid

import google
import numpy as np

from data_interface import sages_pb2


def parse_args():
    """Parse arguments

    Returns:
        params: arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="The output filename")
    parser.add_argument("--phase-folder", type=str, default=None, help="The location of the phases text files")
    parser.add_argument("--tool-folder", type=str, default=None, help="The location of the tools text files")
    parser.add_argument("--frames-per-second", type=float, default=25.0, help="The FPS of the video files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose flag")
    params = parser.parse_args()
    return params


def seconds_to_timestamp(seconds):
    """Convert seconds to a protobuf timestamp.

    Args:
        seconds (float): seconds timespan

    Returns:
        timestamp: google.protobuf.timestamp_pb2.Timestamp
    """
    timestamp = google.protobuf.timestamp_pb2.Timestamp()
    timestamp.seconds = np.long(seconds) // 1
    timestamp.nanos = np.long((seconds - timestamp.seconds) * 1e9)
    return timestamp


def create_track_groups(track_type, folder):
    """read a folder with txt files from camma annotation, convert to protobuf annotation track group

    Args:
        track_type (str): The track type / name.
        folder (str): The folder where the txt files are.

    Returns:
        dict: mapping from surgery id to a track group for that surgery.
    """
    results_dict = {}
    folder_name = os.path.expandvars(os.path.expanduser(folder))
    txt_files = glob.glob(os.path.join(folder_name, "*.txt"))
    labels_set = set()
    surgical_entities = []
    for filename in txt_files:
        if params["verbose"]:
            print("annotation filename: {}".format(filename))
        with open(filename, "r") as fp:
            csvreader = csv.reader(fp, delimiter="\t")
            last_label = None
            intervals = []
            for i_row, row in enumerate(csvreader):
                if i_row == 0:
                    continue
                row_id = int(row[0])
                row_label = row[1]
                labels_set.add(row_label)
                interval_start_id = row_id
                if not row_label == last_label:
                    if last_label is None:
                        last_label = row_label
                        interval_start_id = row_id
                    else:
                        interval_end_id = row_id
                        start_time = interval_start_id / params["frames_per_second"]
                        end_time = interval_end_id / params["frames_per_second"]
                        label = last_label

                        # save interval
                        new_interval_info = {
                            "start_time": start_time,
                            "end_time": end_time,
                            "label": label,
                        }
                        intervals.append(new_interval_info)
                        last_label = row_label
                        # print(new_interval_info)
                        video_filename, _ = os.path.splitext(filename)
                        video_filename = video_filename[: -len("-phase")] + ".mp4"
                        new_id = str(uuid.uuid4())
                        protobuf_start_time = seconds_to_timestamp(start_time)
                        protobuf_end_time = seconds_to_timestamp(end_time)
                        protobuf_interval = sages_pb2.TemporalInterval(
                            start=protobuf_start_time, end=protobuf_end_time, start_exact=True, end_exact=True
                        )
                        video_id = video_filename
                        new_event = sages_pb2.Event(
                            type="phase", temporal_interval=protobuf_interval, video_id=video_id, annotator_id="camma"
                        )
                        new_entity = sages_pb2.SurgeryEntity(event=new_event, entity_id=new_id)

                        surgical_entities.append(new_entity)

            print(video_filename)
        track = sages_pb2.Track(name=track_type, entities=surgical_entities)
        tg = sages_pb2.TracksGroup(name=track_type + "_group", tracks=[track])

        results_dict[video_filename] = tg
    print(str(labels_set))
    return results_dict


if __name__ == "__main__":
    args = parse_args()
    params = vars(args)
    annotation_sets = []
    annotation_set_dicts = {}
    if params["phase_folder"] is not None:
        tg_dicts = create_track_groups(track_type="phases", folder=params["phase_folder"])
        for key in tg_dicts:
            if key not in annotation_set_dicts:
                annotation_set_dicts[key] = []
            annotation_set_dicts[key].append(tg_dicts[key])

    if params["tool_folder"] is not None:
        # TODO(guy.rosman): modify tool groups creation to match txt csv format.
        tg_dicts = create_track_groups(track_type="tools", folder=params["tool_folder"])
        for key in tg_dicts:
            if key not in annotation_set_dicts:
                annotation_set_dicts[key] = []
            annotation_set_dicts[key].append(tg_dicts[key])
    for key in annotation_set_dicts:
        annotation_set = sages_pb2.AnnotationSet(tracks_groups=annotation_set_dicts[key])
        annotation_sets.append(annotation_set)
    output_pathname = os.path.expandvars(os.path.expanduser(params['output']))
    with open(output_pathname,'wb') as fp:
        for aset in annotation_sets:
            fp.write(aset.SerializeToString())
        fp.close()
    
