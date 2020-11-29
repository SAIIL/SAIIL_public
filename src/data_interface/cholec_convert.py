import argparse
import csv
import glob
import os
import uuid

import google
import numpy as np

from data_interface import sages_pb2
import google.protobuf.timestamp_pb2

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


def isLast(itr):
    """wraps an iterator and returns whether we're at the last element.
    https://stackoverflow.com/questions/970983/have-csv-reader-tell-when-it-is-on-the-last-line
    """
    old = itr.__next__()
    for new in itr:
        yield False, old
        old = new
    yield True, old


def create_track_groups(track_type, folder, label_type):
    """read a folder with txt files from camma annotation, convert to protobuf annotation track group

    Args:
        track_type (str): The track type / name.
        folder (str): The folder where the txt files are.
        label_type (str): Either label_name or binary_labels

    Returns:
        dict: mapping from surgery id to a track group for that surgery.
    """
    assert(label_type in ['label_name','binary_labels'])
    results_dict = {}
    folder_name = os.path.expandvars(os.path.expanduser(folder))
    txt_files = glob.glob(os.path.join(folder_name, "*.txt"))
    labels_set = set()
    labels_list = list()
    for filename in txt_files:
        surgical_entities = []
        if params["verbose"]:
            print("annotation filename: {}".format(filename))
        video_filename, _ = os.path.splitext(filename)
        _,video_filename = os.path.split(video_filename)
        # import IPython;IPython.embed()
        video_id = video_filename[: -len("-"+track_type)] + ".mp4"
        video_filename = video_filename[: -len("-"+track_type)]
        print(video_filename)
        with open(filename, "r") as fp:
            csvreader = csv.reader(fp, delimiter="\t")
            segments_status={}
            last_label = None
            interval_start_id = 1
            intervals = []
            for i_row, (is_last,row) in enumerate(isLast(csvreader)):
                if i_row == 0:
                    if (label_type == 'binary_labels'):
                        labels_list = row[1:]
                        labels_set = set(labels_list)
                        segments_status={}
                        for label in labels_list:
                            segments_status[label]={}
                            segments_status[label]['state']=0
                        continue
                    else:
                        continue
                row_id = int(row[0])
                if label_type == 'binary_labels':
                    for row_value,label_id in zip(row[1:],labels_list):
                        # import IPython;IPython.embed(header='populate new segments0: {}'.format(row_id))
                        if int(row_value)==1 and segments_status[label_id]['state']==0:
                            # start new segment
                            segments_status[label_id]['state'] = 1
                            segments_status[label_id]['start_time']=(row_id)/ params["frames_per_second"]
                        elif (int(row_value)==0 or is_last) and segments_status[label_id]['state']==1:
                            segments_status[label_id]['state'] = 0
                            segments_status[label_id]['end_time']=(row_id)/ params["frames_per_second"]

                            new_id = str(uuid.uuid4())
                            protobuf_start_time = seconds_to_timestamp(segments_status[label_id]['start_time'])
                            protobuf_end_time = seconds_to_timestamp(segments_status[label_id]['end_time'])
                            protobuf_interval = sages_pb2.TemporalInterval(
                                start=protobuf_start_time, end=protobuf_end_time, start_exact=True, end_exact=True
                            )
                            new_event = sages_pb2.Event(
                                type=label_id, temporal_interval=protobuf_interval, video_id=video_id, annotator_id="camma"
                            )
                            new_entity = sages_pb2.SurgeryEntity(event=new_event, entity_id=new_id)

                            surgical_entities.append(new_entity)
                            del segments_status[label_id]['start_time']
                            del segments_status[label_id]['end_time']
                            # import IPython;IPython.embed(header='populate new segments')
                else:
                    row_label = row[1]
                    labels_set.add(row_label)
                    
                    if not row_label == last_label:
                        if last_label is None:
                            last_label = row_label
                            interval_start_id = row_id
                        else:
                            interval_end_id = row_id
                            start_time = (interval_start_id) / params["frames_per_second"]
                            end_time = (interval_end_id) / params["frames_per_second"]
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
                            new_id = str(uuid.uuid4())
                            protobuf_start_time = seconds_to_timestamp(start_time)
                            protobuf_end_time = seconds_to_timestamp(end_time)
                            protobuf_interval = sages_pb2.TemporalInterval(
                                start=protobuf_start_time, end=protobuf_end_time, start_exact=True, end_exact=True
                            )
                            new_event = sages_pb2.Event(
                                type=label, temporal_interval=protobuf_interval, video_id=video_id, annotator_id="camma"
                            )
                            new_entity = sages_pb2.SurgeryEntity(event=new_event, entity_id=new_id)

                            surgical_entities.append(new_entity)
                            interval_start_id = row_id

        track = sages_pb2.Track(name=track_type, entities=surgical_entities)
        tg = sages_pb2.TracksGroup(name=track_type + "_group", tracks=[track])

        results_dict[video_filename] = tg
    print(str(labels_set))
    return results_dict


if __name__ == "__main__":
    args = parse_args()
    params = vars(args)
    annotation_set_dicts = {}
    if params["phase_folder"] is not None:
        tg_dicts = create_track_groups(track_type="phase", folder=params["phase_folder"], label_type='label_name')
        for key in tg_dicts:
            if key not in annotation_set_dicts:
                annotation_set_dicts[key] = []
            annotation_set_dicts[key].append(tg_dicts[key])

    if params["tool_folder"] is not None:
        # TODO(guy.rosman): modify tool groups creation to match txt csv format.
        tg_dicts = create_track_groups(track_type="tool", folder=params["tool_folder"], label_type='binary_labels')
        for key in tg_dicts:
            if key not in annotation_set_dicts:
                annotation_set_dicts[key] = []
            annotation_set_dicts[key].append(tg_dicts[key])
    for key in annotation_set_dicts:
        annotation_set = sages_pb2.AnnotationSet(tracks_groups=annotation_set_dicts[key])
        folder_name = os.path.expandvars(os.path.expanduser(params['output']))
        os.makedirs(folder_name, exist_ok=True)

        output_pathname = os.path.join(folder_name,key + '.pb')
        with open(output_pathname,'wb') as fp:
            fp.write(annotation_set.SerializeToString())
            fp.close()
    
