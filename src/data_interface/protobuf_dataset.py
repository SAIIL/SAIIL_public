import fnmatch
import os
import data_interface.sages_pb2
from collections import OrderedDict
import glob
import multidict
from torch.utils.data import Dataset, dataloader
from data_interface.sages_pb2 import *

def load_protobuf_dir(annotation_dir,
                       missing_annotations_is_okay=False,
                       prefix='',
                       verbose=True,phase_translation_file=None, allowed_track_names=None):
    '''
    load annotation information from protobuf dataset
    :param annotation_dir: a folder contains .pb files
    :param missing_annotations_is_okay:
    :param prefix: prefix string if any
    :param verbose:
    :param phase_translation_file: the phase translation filename
    :param allowed_track_names: the track names that will actually be read. Use None to allow all tracks.
    :return:
    '''
    steps = {}
    annotations = {}

    files_list = sorted(glob.glob(os.path.join(annotation_dir, '*.pb')))
    if phase_translation_file is not None:
        from data_interface.utils import read_phase_mapping
        phase_name_map = read_phase_mapping(phase_translation_file)
    else:
        phase_name_map = {}

    for filename in files_list:
        try:
            tracks = load_protobuf_file(filename)
        except:
            print('Error in ' + filename)
            raise

        video_file = os.path.split(filename)[-1].split('.')[0]
        annotations[os.path.join(prefix,
                                 video_file)] = multidict.MultiDict()
        total_time = 0
        pre_phase_name = ''
        for trk in tracks:
            if trk['track_name'] == 'point' or trk[
                'track_name'] == 'other steps':
                continue
            track_name = trk['track_name']
            if allowed_track_names is not None and track_name not in allowed_track_names:
                continue

            props = trk['entity'].event
            name = props.type
            if (name in phase_name_map.keys()):
                name = phase_name_map[name]

            start = float(str(props.temporal_interval.start.seconds) + '.'+ str(props.temporal_interval.start.nanos))
            end = float(str(props.temporal_interval.end.seconds) + '.'+ str(props.temporal_interval.end.nanos))
            temporal_attr = 'interval'
            if start == 0 and end == 0:
                temporal_attr = 'point'
                start = float(str(props.temporal_point.point.seconds) + '.'+ str(props.temporal_point.point.nanos))
                end = start
            video_id = props.video_id
            annotator_id = props.annotator_id
            if (name is None):
                if verbose:
                    print('Attribute name is missing, ' + filename)
                continue
            if track_name not in steps:
                steps[track_name] = set()
            steps[track_name].add(name)
            annotations[os.path.join(prefix, video_file)].add(
                name, {
                    'start': start,
                    'end': end,
                    'temporal_attr': temporal_attr,
                    'annotator_id': annotator_id,
                    'track_name': track_name,
                    'video_id': video_id
                })
            total_time += (float(end) - float(start))
            if track_name == 'major operative phases':
                pre_phase_name = name

    if len(annotations) == 0 and not missing_annotations_is_okay:
        print('Missing annotations: ' + annotation_dir)
        from data_interface.utils import MissingAnnotationsError
        raise MissingAnnotationsError(annotation_dir)
    # sort the class_name in alphabet order, otherwise each training has different class names
    class_names = {}
    for key in steps:
        class_names[key] = list(steps[key])
        class_names[key].sort()
    return class_names, annotations

def load_protobuf_file(filename):
    with open(filename, 'rb') as fp:
        annotation_set = AnnotationSet()
        annotation_set.ParseFromString(fp.read())
        all_entities = []
        for tracks_group in annotation_set.tracks_groups:
            for tr in tracks_group.tracks:
                for entity in tr.entities:
                    all_entities.append({'track_name': tr.name, 'entity': entity})
        return all_entities






