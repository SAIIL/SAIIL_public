import os
import glob
import torch
import multidict
import warnings
import unicodedata
import numpy as np
import cv2
from torch.utils.data import Dataset, dataloader
import torch.utils.data.sampler
import xml.etree.ElementTree as ET
import collections
import tqdm
import json
from data_interface.surgical_dataset import SurgicalDataset
from data_interface.protobuf_dataset import load_protobuf_dir


def collate_filter_empty_elements(batch):
    #filter the empty elements in the batch
    batch = list(filter(lambda x:x is not None, batch))
    return dataloader.default_collate(batch)

class MissingAnnotationsError(BaseException):
    def __init__(self, dir):
        self.directory = dir


class MissingVideoDirError(BaseException):
    def __init__(self, dir):
        self.directory = dir


def create_onehot_nan(path, num_classes):
    batch_size = path.shape[0]
    time_size = path.shape[1]

    y_onehot = torch.zeros(batch_size, time_size,
                           num_classes).float().to(path.device)

    y_onehot.zero_()
    for batch_idx in range(batch_size):
        path_batch = path[batch_idx].squeeze(0).cpu().numpy()
        for idx, path_sample in enumerate(path_batch):
            if not np.isnan(path_sample):
                y_onehot[batch_idx, idx, path_sample] = 1.0
    return y_onehot



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
        annotation_dir=annotation_filename, verbose=verbose,phase_translation_file=phase_translation_file)

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



def write_results_to_txt(estimate_list=None,
                         class_names=None,
                         res_dir='results_txt/camma/',
                         video_name_list_filename=None,
                         estimate_fps=1,
                         write_fps=30):
    '''
    Write the formatted estimation result into txt files
    :param estimate_list:
    :param class_names:
    :param res_dir:
    :param video_name_list_filename:
    :param estimate_fps: the estimation frame rate
    :param write_fps: the output frame rate
    :return:
    '''

    time_ratio = int(write_fps / estimate_fps)
    os.makedirs(res_dir, exist_ok=True)
    with open(video_name_list_filename) as f_video_names:
        video_name_list = list(json.load(f_video_names).values())

    for video_idx in estimate_list.keys():
        video_name = video_name_list[video_idx - 1].split('_')[0]
        estimate = np.argmax(estimate_list[video_idx], axis=1).astype(int)
        write_file_name = res_dir + video_name + '.txt'
        f_out = open(write_file_name, "w")
        f_out.write('Frame\tPhase\n')
        for t in range(estimate.shape[0]):
            for idx1 in range(time_ratio):
                t_idx = t * time_ratio + idx1
                f_out.write(str(t_idx) + '\t' + class_names[estimate[t]] + '\n')

        f_out.close()
    return




