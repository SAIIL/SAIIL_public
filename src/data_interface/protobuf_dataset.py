import fnmatch
import glob
import os
from collections import OrderedDict

import cv2
import multidict
import numpy as np
import tqdm
from torch.utils.data import Dataset, dataloader

import data_interface.sages_pb2
from data_interface.base_utils import MissingAnnotationsError
from data_interface.sages_pb2 import *
from data_interface.surgical_dataset import SurgicalDataset


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
        from data_interface.base_utils import read_phase_mapping
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


def process_data_directory_surgery(data_dir,
                                     fractions=[],
                                     width=224,
                                     height=224,
                                     max_video=80,
                                     batch_size=32,
                                     num_workers=4,
                                     train_transform=None,
                                     shuffle=True,
                                     segment_ratio=1.0,
                                     patient_factor_list = [],
                                     past_length = 10,
                                     train_ratio=0.75,
                                     default_fps=25,
                                     sampler=None,
                                     verbose=True,
                                     annotation_filename=None,
                                     temporal_len=None,
                                     sampling_rate = 1,
                                     avoid_annotations=False,
                                     seed=1234,
                                     skip_nan = True,
                                     phase_translation_file=None, cache_dir='', params={}):
    '''
    Read a data directory, and can handle multiple annotators.
    :param data_dir: the root dir for the data
    :param fractions:
    :param width:
    :param height:
    :param max_video:
    :param batch_size:
    :param num_workers:
    :param train_transform:
    :param shuffle:
    :param segment_ratio:
    :param train_ratio:
    :param default_fps:
    :param sampler:
    :param verbose:
    :param annotation_filename: - the folder w/ annotations files (from anvil)
    :param temporal_len:
    :param avoid_annotations:
    :param seed:
    :param sampling_rate: the sampling rate of creating the dataset from videos, unit: fps
    :param avoid_annotation #TODO - complete this one
    :param skip_nan if add nan label into the dataset
    :param params - a dictionary of additional parameters:
    'track_name' - the track name to generate datasets for.
    :param params: a dictionary for new parameters
    #TODO: move arguments into params dictionary
    :return:
    '''
    print("sampling rate:  " + str(sampling_rate))
    train_images = []
    train_labels = []
    train_video_idx = multidict.MultiDict()
    test_images = []
    test_labels = []
    test_video_idx = multidict.MultiDict()
    track_name=params.get('track_name',None)
    # make sure there's a trailing separator for consistency
    data_dir=os.path.join(data_dir,'')
    class_names, annotations = load_protobuf_dir(

    annotation_dir=annotation_filename, verbose=verbose,phase_translation_file=phase_translation_file, allowed_track_names=[track_name])


    if track_name is None:
        track_name=list(class_names.keys())[0]
    training_per_phase = False
    training_frames = 0
    test_frames = 0
    video_surgeon_translation_file = params.get('video_surgeon_translation_file','')
    if not os.path.exists(data_dir):
        raise MissingVideoDirError(data_dir)
    np.random.seed(seed)
    all_video_files=glob.glob(os.path.join(data_dir, '**/*.mp4'))+glob.glob(os.path.join(data_dir, '*.mp4'))+glob.glob(os.path.join(data_dir, '**/*.avi'))+glob.glob(os.path.join(data_dir, '*.avi'))
    for filename in tqdm.tqdm(sorted(all_video_files),desc='reading videos'):
        video_filename = filename[len(data_dir):].split('.')[0]
        video_pathname = filename
        try:
            phases_info = annotations[video_filename]
        except:
            continue

        video = cv2.VideoCapture(video_pathname)
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if (fps == 0.0):
            if default_fps is not None:
                fps = default_fps
            else:
                raise 'fps missing ' + video_pathname
        if avoid_annotations:
            phases_info = multidict.MultiDict()
            phases_info.add('all', {
                'start': 0,
                'end': int(num_frames / fps) - 1
            })

        if not (training_per_phase):
            if np.random.uniform(0.0, 1.0) < train_ratio:
                training_data = True
            else:
                training_data = False

            if verbose:
                print(video_pathname + ", training_data=" + str(training_data))

        if training_data:
            train_video_idx[video_pathname] = multidict.MultiDict()
            train_video_idx[video_pathname]['start_idx'] = len(train_images)
        else:
            test_video_idx[video_pathname] = multidict.MultiDict()
            test_video_idx[video_pathname]['start_idx'] = len(test_images)


        time_step = round(fps / sampling_rate)
        current_phase = 0
        phases_info = list(phases_info.items())
        start_pre = -1
        end_pre = -1
        fraction = 1.0
        if fractions is not None:
            if type(fractions) == float:
                fraction = fractions
            elif isinstance(fractions, list):
                raise Exception('Need to fix fractions based on the old dataset reading')
        else:
            fraction = 1.0
        fraction_draw = np.random.uniform()
        if fraction < fraction_draw:
            continue

        for frame_idx in range(0, num_frames, time_step):
            add_frame_to_dataset = True
            current_time_idx =  frame_idx/fps
            phase_step = phases_info[current_phase]
            start = float(phase_step[1]['start'])
            end = float(phase_step[1]['end'])
            label = None

            if (training_data):
                training_frames += 1
            else:
                test_frames += 1
            while label is None:
                if  current_time_idx >= start and current_time_idx <= end:
                    # if the frame_idx fall in the segment
                    # add the image to the train and test dataset
                    label = class_names[track_name].index(phase_step[0])
                elif current_time_idx < start and current_time_idx > end_pre:
                    if end_pre == -1:
                        # skip the frames before the first annotation
                        add_frame_to_dataset = False
                        label = float('NaN')
                    else:
                        # add empty label (nans) to the dataset
                        label = float('NaN')
                elif current_time_idx > end:
                    if current_phase < (len(phases_info)-1):
                        # move to the next phase segment
                        current_phase += 1
                        start_pre = start
                        end_pre = end
                        phase_step = phases_info[current_phase]
                        start = float(phase_step[1]['start'])
                        end = float(phase_step[1]['end'])
                    elif current_phase >= (len(phases_info) -1):
                        # skip the frames after the last annotation segment
                        add_frame_to_dataset = False
                        label = float('NaN')

            if np.isnan(label) and skip_nan:
                add_frame_to_dataset = False

            if add_frame_to_dataset is False:
                continue

            img = (video_pathname, frame_idx)

            try:
                if (training_data):
                    train_images.append(img)
                    train_labels.append(label)
                    for patient_factor in patient_factor_list:
                        train_patient_factor[patient_factor].append(patient_factor_sample[patient_factor])
                else:
                    test_images.append(img)
                    test_labels.append(label)
                    for patient_factor in patient_factor_list:
                        test_patient_factor[patient_factor].append(patient_factor_sample[patient_factor])
            except:
                pass

        if training_data:
            train_video_idx[video_pathname]['end_idx'] = len(train_images) - 1
        else:
            test_video_idx[video_pathname]['end_idx'] = len(test_images) - 1

    if verbose:
        print('Collected: %d training, %d test examples, segments: %d,%d ' %
              (len(train_images), len(test_images), training_frames,
               test_frames))

    if any(np.isnan(train_labels)):
        print('there is nan in labels')


    train_dataset = SurgicalDataset(train_images,
                                         train_labels,
                                         train_video_idx,
                                         past_length=past_length,
                                         fps = fps,
                                         width=width,
                                         height=height,
                                         transform=train_transform,
                                         class_names=class_names, cache_dir = cache_dir, params=params)



    val_dataset = SurgicalDataset(test_images,
                                       test_labels,
                                       test_video_idx,
                                       past_length=past_length,
                                       fps = fps,
                                       width=width,
                                       height=height,
                                       class_names=class_names, cache_dir = cache_dir, params=params)


    dataloaders = {'train': train_dataset, 'val': val_dataset}
    return dataloaders




