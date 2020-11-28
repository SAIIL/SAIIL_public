import collections
import copy
import csv
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from data_interface.surgical_video_dataset_utils import (make_numeric_pre_post,
                                                         merge_columns, create_onehot_nan)
#TODO(guy.rosman): move create_onehot_nan into surgical_video_dataset_utils, or merge into similar non-gan-specific file.

DATASET_KEY_IMAGES = 'imgs'
DATASET_KEY_PHASE_TRAJECTORY = 'phase_trajectory'
DATASET_KEY_TARGETS_PHASE_TRAJECTORY = 'tgts_phase_trajectory'
#TODO(guy.rosman) MOVING_DAY Yutong, Add a doc on what this constant is used for, as opposed to phase trajectory.
DATASET_KEY_CANDIDATE_TARGETS_PHASE_TRAJECTORY = 'tgts_candidate_phase_trajectory'
DATASET_KEY_VIDEO_NAME = 'video_name'
DATASET_KEY_SURGEON_NAME = 'surgeon_id'
DATASET_KEY_VALID_LIST = 'valid_list'
DATASET_KEY_EMR_PREOP = 'past_variables'
DATASET_KEY_EMR_POSTOP = 'future_variables'


class SurgicalPredictionDataset(Dataset):
    """Dataset for gan trajectory generation"""


    def __init__(self,
                 images_list = [],
                 labels_list = [],
                 patient_factor_list = [],
                 video_idx_list = {},
                 width = None,
                 height = None,
                 transform = None,
                 time_step = 1,
                 past_length = 10,
                 future_length = 10,
                 class_names = None,
                 track_name = None,
                 cache_dir = None,
                 patient_factor_name_list = [],
                 fps = 30,
                 params=None):
        '''
        Dataset with past images, phase trajectories, patient factors
                and  future phase trajectories, patient factors
        :param images_list:
        :param labels_list:
        :param video_idx_list: the idx range for each video in images_list and labels_list
        :param width:
        :param height:
        :param transform:
        :param time_step: the gap between two different trajectories
        :param past_length:
        :param predict_length:
        :param class_names:
        :param patient_factor_list:
        :param cache_dir:
        :param params:
        '''
        self.images_list = images_list
        self.labels_list = labels_list
        self.patient_factor_list = patient_factor_list
        self.video_idx_list = video_idx_list
        self.width = width
        self.height = height
        self.transform = transform
        self.time_step = time_step
        self.past_length = past_length
        self.future_length = future_length
        self.cache_dir = cache_dir
        self.patient_factor_name_list = patient_factor_name_list
        self.class_name_patient_factors = class_names
        if track_name is None:
            track_name = list(class_names.keys())[0]
        self.track_name = track_name
        self.class_names = class_names[track_name]
        self.num_classes = len(self.class_names)
        self.n_candidate_trajectory = params.get('n_decoding_samples', 5)
        self.fps = fps


        self.video_to_emr_map = params.get('video_redcap_translation_file')
        if self.video_to_emr_map is not None and len(self.video_to_emr_map)>0:
            self.video_to_emr_map = os.path.expanduser(self.video_to_emr_map)
            with open(self.video_to_emr_map, 'r') as fp:
                reader = csv.reader(fp)
                self.video_to_emr_map={}
                for i_row, row in enumerate(reader):
                    video_id, record_id = row
                    if i_row == 0:
                        continue
                    self.video_to_emr_map[video_id.replace('-','')] = {'record_id':int(record_id)-1}
        self.video_to_surgeon_map = params.get('video_surgeon_translation_file')
        if self.video_to_surgeon_map is not None and len(self.video_to_surgeon_map)>0:
            print('self.video_to_surgeon_map: '+str(self.video_to_surgeon_map))
            self.video_to_surgeon_map = os.path.expanduser(self.video_to_surgeon_map)
            with open(self.video_to_surgeon_map, 'r') as fp:
                self.video_to_surgeon_map = {}
                reader = csv.reader(fp)
                for i_row, row in enumerate(reader):
                    video_id_, surgeon_id, last_name = row
                    if i_row == 0:
                        continue
                    self.video_to_surgeon_map[video_id_.replace('-','')] = {'surgeon_id': surgeon_id, 'last_name': last_name}
        self.redcap_data = params.get('video_redcap_data_file')
        if self.redcap_data is not None and len(self.redcap_data)>0:
            with open(self.redcap_data, 'r',encoding = 'latin-1') as fp:
                self.redcap_data = collections.OrderedDict()
                #TODO(guy.rosman): get from argument
                video_redcap_data_delimiter = '\t'
                reader = csv.reader(fp,delimiter = video_redcap_data_delimiter)
                column_names=[]

                row_transforms=[merge_columns]
                for i_row, row in enumerate(reader):
                    if i_row == 0:
                        self.emr_column_names=[x.lower().strip() for x in row]
                        continue
                    video_id_=row[0]
                    tformed_row = copy.copy(row)
                    tformed_column_names = copy.copy(self.emr_column_names)

                    for tform in row_transforms:
                        tformed_row,tformed_column_names=tform(tformed_row,tformed_column_names)
                    self.redcap_data[video_id_] = {'transformed_row':tformed_row,'column_names':tformed_column_names,'full_row':row}

        self.past_emr_columns=params.get('past_emr_column_names')
        if self.past_emr_columns is not None and len(self.past_emr_columns)>0:
            with open(self.past_emr_columns, 'r',encoding = 'latin-1') as fp:
                tmp_jsn=json.load(fp)
                self.past_emr_columns=[x.lower().strip() for x in tmp_jsn]
                self.past_emr_size = 0
                missing_columns=[]
                for emr_name in self.past_emr_columns:
                    if emr_name in self.emr_column_names:
                        self.past_emr_size += 1
                    else:
                        if not emr_name[0]=='_':
                            missing_columns.append(emr_name)
                if len(missing_columns)>0:
                    import IPython;IPython.embed(header='missing pre-op variables')
                    raise Exception('Missing pre-op emr columns: '+str(missing_columns))

            print('past_emr_columns: '+str(self.past_emr_columns))

        self.future_emr_columns = params.get('future_emr_column_names')
        if self.future_emr_columns is not None and len(self.future_emr_columns)>0:
            with open(self.future_emr_columns, 'r',encoding = 'latin-1') as fp:
                tmp_jsn=json.load(fp)
                self.future_emr_columns=[x.lower().strip() for x in tmp_jsn]
                self.future_emr_size = 0
                missing_columns = []
                for emr_name in self.future_emr_columns:
                    if emr_name in self.emr_column_names:
                        self.future_emr_size += 1
                    else:
                        if not emr_name[0]=='_':
                            missing_columns.append(emr_name)
                if len(missing_columns)>0:
                    import IPython;IPython.embed(header='missing post-op variables')
                    raise Exception('Missing post-op emr columns: '+str(missing_columns))

        self.create_phase_trajectories()
        # create weights for imbalanced data classes
        self.class_weights = torch.tensor([1/x if x!=0 else 0 for x in self.class_count])

    def __len__(self):
        return len(self.input_img_list)

    def __getitem__(self, idx):
        sample_idx = self.current_idx_list[idx]
        sample_sequence_length = self.video_length_list[idx]
        sample_phase_trajectory = self.input_phase_trajectory[idx]
        sample_tgts_phase_trajectory = self.tgts_phase_trajectory[idx]
        sample_candidate_tgts_phase_trajectories = self.get_possible_trajectories(idx)
        sample_img_name_list = self.input_img_list[idx]
        sample_video_name = self.video_name_list[idx]
        sample_surgeon_id = self.get_surgeon_info(sample_video_name)
        sample_patient_factor = dict()
        sample_gt_patient_factor = dict()
        for patient_factor in self.patient_factor_name_list:
            sample_patient_factor[patient_factor] = self.inputs_patient_factor[patient_factor][idx]
            sample_gt_patient_factor[patient_factor] = self.tgts_patient_factor[patient_factor][idx]

        # load images
        sample_img_list = []
        sample_validity_list = []
        opened_video=False
        file_ptr = -1
        video_name = None
        for idx_img, video_info in enumerate(sample_img_name_list):
            if type(sample_img_name_list[idx_img]) is not tuple:
                img = np.zeros((3,self.height,self.width))
                valid_flag = False
            else:
                video_name = video_info[0]
                frame_no = int(video_info[1])
                image = None
                cache_pathname = None
                time_idx = frame_no / self.fps

                if not (self.cache_dir == ''):
                    # cache img in separate video folder e.g. cache_dir/video_name/*.jpg
                    cache_dir_video = os.path.join(self.cache_dir, os.path.split(video_name)[-1].split('.')[0])
                    os.makedirs(self.cache_dir, exist_ok=True)
                    os.makedirs(cache_dir_video, exist_ok=True)

                    if any(sample_phase_trajectory[idx_img,:]>0):
                        class_name = self.class_names[sample_phase_trajectory[idx_img, :].argmax()]
                        valid_flag = True
                    else:
                        class_name = 'nan'
                        valid_flag = False

                    cache_img_name = str(frame_no) + '_' + str(round(time_idx, 2)) + '_' + class_name + '_' + str(self.height) + '_' + str(self.width)
                    cache_pathname = os.path.join(cache_dir_video, cache_img_name) + '.jpg'

                if cache_pathname is not None and os.path.exists(cache_pathname):
                    try:
                        im_pil = Image.open(cache_pathname)
                        image = np.asarray(im_pil)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except:
                        print('error in ' + cache_pathname + ', image = ' + str(image))

                if image is None:
                    # print(video_name)
                    try:
                        if not opened_video:
                            cap = cv2.VideoCapture(video_name)
                            if (cap == 0):
                                print('Failed to open ' + video_name)
                                return [None, -1]
                            opened_video = True
                        if not (file_ptr == frame_no):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                            file_ptr = frame_no

                        total_video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        ret, image = cap.read()
                    except:
                        print("video loading error")
                    if image is None:
                        img = np.float('NaN')
                    else:
                        if (not image.shape[0] == self.height
                                or not image.shape[1] == self.width):
                            image = cv2.resize(image, (self.height, self.width))
                        file_ptr += 1
                        if cache_pathname is not None:
                            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            im_pil = Image.fromarray(img)
                            im_pil.save(cache_pathname)

                if not (image is None):
                    try:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except:
                        print('Error in cache_pathname' + str(cache_pathname) + ', image = ' + str(image))
                    if self.transform:
                        try:
                            pil_im = Image.fromarray(image)
                        except:
                            import IPython
                            IPython.embed(header='handle bad type')
                        scale_tmp = np.random.uniform(0.7, 1.3)
                        transform_apply = transforms.Compose([
                            self.transform,
                            transforms.CenterCrop(scale_tmp * self.width)
                        ])
                        img = transform_apply(pil_im)
                        img = img.resize((self.height, self.width))
                        img = np.float32(img).transpose((2, 0, 1))
                    else:
                        try:
                            pil_im = Image.fromarray(image)
                        except:
                            print("error: " + str(image))
                        img = np.float32(pil_im).transpose((2, 0, 1))

            sample_img_list.append(np.expand_dims(img, 0))
            sample_validity_list.append(valid_flag)
            # finish load images


        sample_img_list = np.concatenate(sample_img_list, axis=0)
        sample_validity_list = torch.Tensor(sample_validity_list)

        #if all nan, return empty sample
        if all(valid_sample == False for valid_sample in sample_validity_list):
            return None

        sample = {'idx':time_idx,
                  'sequence_length':sample_sequence_length,
                  DATASET_KEY_IMAGES:sample_img_list,
                  DATASET_KEY_VALID_LIST:sample_validity_list,
                  DATASET_KEY_PHASE_TRAJECTORY: sample_phase_trajectory,
                  DATASET_KEY_TARGETS_PHASE_TRAJECTORY: sample_tgts_phase_trajectory,
                  DATASET_KEY_CANDIDATE_TARGETS_PHASE_TRAJECTORY: sample_candidate_tgts_phase_trajectories,
                  DATASET_KEY_VIDEO_NAME:sample_video_name,
                  DATASET_KEY_SURGEON_NAME:sample_surgeon_id
                  }

        emr = self.get_emr(self.get_video_id_from_video_name(video_name))
        sample = self.add_emr_to_sample(sample,emr)
        for patient_factor in self.patient_factor_name_list:
            sample['patient_factor'] = sample_patient_factor[patient_factor]
            sample['tgts_patient_factor'] = sample_gt_patient_factor[patient_factor]

        if opened_video:
            cap.release()
            opened_video = False

        return sample

    def get_possible_trajectories(self, idx):
        '''
        get n possible future trajectories from data
        n = self.n_candidate_trajectory
        :param idx:
        :return:
        '''
        candidate_trajectories = dict()
        idx_candidate = 0
        sample_phase_trajectory = self.input_phase_trajectory[idx]
        sample_tgts_phase_trajectory = self.tgts_phase_trajectory[idx]
        len_past = len(sample_phase_trajectory)
        len_future = len(sample_tgts_phase_trajectory)
        len_dataset = len(self.input_phase_trajectory)
        len_compare_past = int(len_past/3)
        len_compare_future= int(len_future / 3 * 2)
        search_start = max(0,idx-400)
        search_end = min(len(self.input_phase_trajectory), idx + 400)
        search_step = 5
        candidate = []

        while len(candidate) < self.n_candidate_trajectory and len_compare_future <= len_future:
            candidate = [x for x in range(search_start,search_end,search_step) if torch.all(torch.eq(self.input_phase_trajectory[x][-len_compare_past:,:], sample_phase_trajectory[-len_compare_past:,:]))
                         and torch.sum(torch.eq(self.tgts_phase_trajectory[x], sample_tgts_phase_trajectory)) <= len_compare_future]
            len_compare_future += 1


        while idx_candidate < self.n_candidate_trajectory:
            if len(candidate) > 0:
                idx_rand = np.random.randint(len(candidate))
                candidate_trajectories[idx_candidate] = self.tgts_phase_trajectory[candidate[idx_rand]]
            else:
                candidate_trajectories[idx_candidate] = self.tgts_phase_trajectory[idx]
            idx_candidate += 1
            if len(candidate)>1:
                candidate.pop(idx_rand)

        return candidate_trajectories


    def add_emr_to_sample(self,sample,emr):
        if emr is None:
            emr = []
        sample[DATASET_KEY_EMR_PREOP] = {}
        sample[DATASET_KEY_EMR_POSTOP] = {}
        if emr is not None:
            if len(self.past_emr_columns) > 0:
                for key in self.past_emr_columns:
                    if key.lower() in emr:
                        sample[DATASET_KEY_EMR_PREOP][key] = emr[key]
            if len(self.future_emr_columns) > 0:
                for key in self.future_emr_columns:
                    if key.lower() in emr:
                        sample[DATASET_KEY_EMR_POSTOP][key] = emr[key]

        sample = make_numeric_pre_post(sample, self.past_emr_columns, self.future_emr_columns)
        return sample

    def create_phase_trajectories(self):
        self.input_phase_trajectory = []
        self.input_img_list = []
        self.tgts_phase_trajectory = []
        self.current_idx_list = []
        self.video_name_list = []
        self.segment_weights_list = []
        self.video_length_list = []
        self.surgeon_id_list = []
        self.inputs_patient_factor = dict()
        self.tgts_patient_factor = dict()
        self.class_count = np.zeros(self.num_classes)
        for patient_factor in self.patient_factor_name_list:
            self.inputs_patient_factor[patient_factor] = []
            self.tgts_patient_factor[patient_factor] = []

        for video in tqdm(self.video_idx_list.items(), desc='preparing file list'):
            video_name = video[0].split('/')[-1].split('.')[0]
            video_start_idx = video[1]['start_idx']
            video_end_idx = video[1]['end_idx']
            video_total_length = video_end_idx - video_start_idx
            surgeon_id = self.get_surgeon_info(video_name)
            if surgeon_id not in self.surgeon_id_list:
                self.surgeon_id_list.append(surgeon_id)

            for idx in range(video_start_idx,video_end_idx,self.time_step):


                inputs_patient_factor_tmp = dict()
                tgts_patient_factor_tmp = dict()

                if idx >= (video_start_idx + self.past_length) and idx <= (video_end_idx - self.future_length):
                    # copy the slice to the trajectory
                    input_phase_trajectory_tmp = self.labels_list[idx-self.past_length+1:idx+1]
                    img_list_tmp = self.images_list[idx-self.past_length+1:idx+1]
                    tgts_phase_trajectory_tmp = self.labels_list[idx+1:idx+self.future_length+1]

                    for patient_factor in self.patient_factor_name_list:
                        # TODO: add the patient factors
                        inputs_patient_factor_tmp[patient_factor] = self.patient_factor_list[patient_factor][idx - self.past_length + 1:idx + 1]
                        tgts_patient_factor_tmp[patient_factor] = self.patient_factor_list[patient_factor][idx + 1:idx + self.future_length + 1]


                elif idx < (video_start_idx + self.past_length):
                    # padding nan to the begining of the sequence
                    padding_length = self.past_length - (idx - video_start_idx) - 1
                    nan_vector = []
                    for idx_nan in range(padding_length):
                        nan_vector.append(float('NaN'))

                    input_phase_trajectory_tmp = nan_vector.copy()
                    input_phase_trajectory_tmp.extend(self.labels_list[video_start_idx:idx + 1])

                    img_list_tmp = nan_vector.copy()
                    img_list_tmp.extend(self.images_list[video_start_idx:idx + 1])

                    tgts_phase_trajectory_tmp = self.labels_list[idx + 1:idx + self.future_length + 1]

                    for patient_factor in self.patient_factor_name_list:
                        # TODO: add the patient factors
                        nan_vector_patient_factor = []
                        for idx_nan in range(padding_length):
                            nan_vector_patient_factor.append(np.array([float('NaN')]))
                        inputs_patient_factor_tmp[patient_factor] = nan_vector_patient_factor.copy()
                        inputs_patient_factor_tmp[patient_factor].extend(self.patient_factor_list[patient_factor][video_start_idx:idx + 1])

                        tgts_patient_factor_tmp[patient_factor] = self.patient_factor_list[patient_factor][idx + 1:idx + self.future_length + 1]

                elif idx > (video_end_idx - self.future_length):
                    # padding nan to the end of the sequence
                    input_phase_trajectory_tmp = self.labels_list[idx - self.past_length + 1:idx + 1]
                    img_list_tmp = self.images_list[idx - self.past_length + 1:idx + 1]

                    nan_vector = []
                    padding_length = idx + self.future_length + 1 - video_end_idx
                    for idx_nan in range(padding_length):
                        nan_vector.append(float('NaN'))

                    tgts_phase_trajectory_tmp = self.labels_list[idx + 1:video_end_idx]
                    tgts_phase_trajectory_tmp.extend(nan_vector)

                    for patient_factor in self.patient_factor_name_list:
                        nan_vector_patient_factor = []
                        for idx_nan in range(padding_length):
                            nan_vector_patient_factor.append(np.array([float('NaN')]))
                        inputs_patient_factor_tmp[patient_factor] = self.patient_factor_list[patient_factor][idx - self.past_length + 1:idx + 1]
                        tgts_patient_factor_tmp[patient_factor] = self.patient_factor_list[patient_factor][idx + 1:video_end_idx]
                        tgts_patient_factor_tmp[patient_factor].extend(nan_vector_patient_factor)

                #no nan labels in the dataset
                if any(np.isnan(sample) for sample in tgts_phase_trajectory_tmp):
                    continue
                if any(np.isnan(sample) for sample in input_phase_trajectory_tmp):
                    continue


                #count phases:
                count_tmp = np.zeros(self.num_classes)
                for idx_tmp in range(len(input_phase_trajectory_tmp)):
                    count_tmp[input_phase_trajectory_tmp[idx_tmp]] += 1
                for idx_tmp in range(len(tgts_phase_trajectory_tmp)):
                    count_tmp[tgts_phase_trajectory_tmp[idx_tmp]] += 1
                self.class_count += count_tmp

                weights_segment = calculateSegmentsWeight(count_tmp)
                self.segment_weights_list.append(weights_segment)

                # convert list to Tensor
                input_phase_trajectory_tmp = torch.Tensor(input_phase_trajectory_tmp).unsqueeze(0).unsqueeze(2)
                input_phase_trajectory_tmp = create_onehot_nan(input_phase_trajectory_tmp, self.num_classes)
                input_phase_trajectory_tmp = input_phase_trajectory_tmp.squeeze(0)


                tgts_phase_trajectory_tmp = torch.Tensor(tgts_phase_trajectory_tmp).unsqueeze(1)


                self.current_idx_list.append(idx)
                self.video_length_list.append(video_total_length)
                self.video_name_list.append(video_name)
                self.input_phase_trajectory.append(input_phase_trajectory_tmp)
                self.input_img_list.append(img_list_tmp)
                self.tgts_phase_trajectory.append(tgts_phase_trajectory_tmp)
                for patient_factor in self.patient_factor_name_list:
                    if patient_factor in self.class_name_patient_factors:
                        n_class_patient_factor = len(self.class_name_patient_factors[patient_factor])
                    else:
                        n_class_patient_factor = 2
                    inputs_patient_factor_tmp[patient_factor] = torch.Tensor(inputs_patient_factor_tmp[patient_factor]).unsqueeze(0).unsqueeze(2)
                    inputs_patient_factor_tmp[patient_factor] = create_onehot_nan(inputs_patient_factor_tmp[patient_factor], n_class_patient_factor)
                    inputs_patient_factor_tmp[patient_factor] = inputs_patient_factor_tmp[patient_factor].squeeze(0)

                    tgts_patient_factor_tmp[patient_factor] = torch.Tensor(tgts_patient_factor_tmp[patient_factor])

                    self.inputs_patient_factor[patient_factor].append(inputs_patient_factor_tmp)
                    self.tgts_patient_factor[patient_factor].append(tgts_patient_factor_tmp)

        # normalize the segment weights
        self.segment_weights_list = np.array(self.segment_weights_list)
        self.segment_weights_list = self.segment_weights_list/ np.sum(self.segment_weights_list)
        print('finish creating trajectories, in total '+ str(len(self.input_phase_trajectory)) + 'trajectories')

    def get_video_id_from_video_name(self,video_name):
        if self.video_to_emr_map is None:
            return None
        else:
            _,filename = os.path.split(video_name)
            filename,_ = os.path.splitext(filename)
            return filename

    def get_surgeon_info(self,video_name):
        video_id = self.get_video_id_from_video_name(video_name)
        if (self.video_to_surgeon_map is None or video_id is None or video_id not in self.video_to_surgeon_map):
            return 'None'
        else:
            return self.video_to_surgeon_map[video_id]['surgeon_id']

    def get_emr(self,video_id):
        if (self.video_to_emr_map is None or self.redcap_data==''):
            return None
        else:
            emr_key = video_id
            if emr_key in self.video_to_emr_map:
                emr_id = self.video_to_emr_map[emr_key]['record_id']
                value = list(self.redcap_data.values())[emr_id]
                value = dict(zip(self.emr_column_names,value['full_row']))
                value = dict((k.lower(), v) for k, v in value.items())
                #TODO(by guy.rosman):remove entries that are empty via a "cleanup" function.
            else:
                value = None
            return value


#TODO(guy.rosman) MOVING_DAY Yutong, Fix the doc string for factor default. Also, not clear what does factor mean.
def calculateSegmentsWeight(count = None, factor = 1.2):
    '''
    compute the sampling weight for each segment
    :param count: the number of phases in the segment
    :param factor:  with factor = 1.2 the effective number over the number of sampled segments is around 4
    :return: the weight of the segment
    '''
    n_class = sum(count>0)
    weight = np.exp(factor * n_class)
    return weight