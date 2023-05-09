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

DATASET_KEY_IMAGES = "imgs"
DATASET_KEY_PHASE_TRAJECTORY = "phase_trajectory"
DATASET_KEY_VIDEO_NAME = "video_name"
DATASET_KEY_SURGEON_NAME = "surgeon_id"
DATASET_KEY_VALID_LIST = "valid_list"


class SurgicalDataset(Dataset):
    """Dataset for images and corresponding labels"""

    def __init__(
        self,
        images_list=[],
        labels_list=[],
        video_idx_list={},
        width=None,
        height=None,
        transform=None,
        time_step=1,
        past_length=10,
        class_names=None,
        track_name=None,
        cache_dir=None,
        video_data_type="video",
        fps=30,
        params=None,
    ):
        """
        Dataset with past images, phase trajectories
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
        """
        self.images_list = images_list
        self.labels_list = labels_list
        self.video_idx_list = video_idx_list
        self.width = width
        self.height = height
        self.transform = transform
        self.time_step = time_step
        self.past_length = past_length
        self.cache_dir = cache_dir
        self.class_name_patient_factors = class_names
        if track_name is None:
            track_name = list(class_names.keys())[0]
        self.track_name = track_name
        self.class_names = class_names[track_name]
        self.num_classes = len(self.class_names)
        self.fps = fps
        self.video_data_type = video_data_type

        self.create_phase_trajectories()
        # create weights for imbalanced data classes
        self.class_weights = torch.tensor([1/x if x!=0 else 0 for x in self.class_count])
        self.skip_nan = params.get('skip_nan', True)

    def __len__(self):
        return len(self.input_img_list)

    def __getitem__(self, idx):
        sample_idx = self.current_idx_list[idx]
        sample_sequence_length = self.video_length_list[idx]
        sample_phase_trajectory = self.input_phase_trajectory[idx]
        sample_img_name_list = self.input_img_list[idx]
        sample_video_name = self.video_name_list[idx]

        # load images
        sample_img_list = []
        sample_validity_list = []
        opened_video = False
        file_ptr = -1
        video_name = None

        # Go over images for each subsequence.
        for idx_img, video_info in enumerate(sample_img_name_list):
            if type(sample_img_name_list[idx_img]) is not tuple:
                img = np.zeros((3, self.height, self.width))
                valid_flag = False
            else:
                video_name = video_info[0]
                frame_no = int(video_info[1])
                image = None
                cache_pathname = None
                time_idx = frame_no / self.fps

                # Obtain cache pathname.
                if not (self.cache_dir == ""):
                    # cache img in separate video folder e.g. cache_dir/video_name/*.jpg
                    cache_dir_video = os.path.join(self.cache_dir, os.path.split(video_name)[-1].split(".")[0])
                    os.makedirs(self.cache_dir, exist_ok=True)
                    os.makedirs(cache_dir_video, exist_ok=True)

                    if any(sample_phase_trajectory[idx_img, :] > 0):
                        class_name = self.class_names[sample_phase_trajectory[idx_img, :].argmax()]
                        valid_flag = True
                    else:
                        class_name = "nan"
                        valid_flag = False

                    cache_img_name = (
                        str(frame_no)
                        + "_"
                        + str(round(time_idx, 2))
                        + "_"
                        + class_name
                        + "_"
                        + str(self.height)
                        + "_"
                        + str(self.width)
                    )
                    cache_pathname = os.path.join(cache_dir_video, cache_img_name) + ".jpg"

                # Load cached file if it exits.
                if cache_pathname is not None and os.path.exists(cache_pathname):
                    try:
                        im_pil = Image.open(cache_pathname)
                        image = np.asarray(im_pil)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except:
                        print("error in " + cache_pathname + ", image = " + str(image))

                if image is None:
                    if self.video_data_type == "video":
                        try:
                            if not opened_video:
                                cap = cv2.VideoCapture(video_name)
                                if cap == 0:
                                    print("Failed to open " + video_name)
                                    return [None, -1]
                                opened_video = True
                            if not (file_ptr == frame_no):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                                file_ptr = frame_no

                            total_video_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            ret, image = cap.read()
                        except:
                            print("video loading error")
                    elif self.video_data_type == "images":
                        img_filename = os.path.join(video_name, str(frame_no).zfill(6) + ".png")
                        image = cv2.imread(img_filename)

                    if image is None:
                        img = np.float("NaN")
                    else:
                        if not image.shape[0] == self.height or not image.shape[1] == self.width:
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
                        print("Error in cache_pathname" + str(cache_pathname) + ", image = " + str(image))
                    if self.transform:
                        try:
                            pil_im = Image.fromarray(image)
                        except:
                            import IPython

                            IPython.embed(header="handle bad type")

                        transform_apply = self.transform
                        img = transform_apply(pil_im)
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
        if self.skip_nan:
            if all(valid_sample == False for valid_sample in sample_validity_list):
                return None

        sample = {
            "idx": time_idx,
            "sequence_length": sample_sequence_length,
            DATASET_KEY_IMAGES: sample_img_list,
            DATASET_KEY_VALID_LIST: sample_validity_list,
            DATASET_KEY_PHASE_TRAJECTORY: sample_phase_trajectory,
            DATASET_KEY_VIDEO_NAME: sample_video_name,
        }

        if opened_video:
            cap.release()
            opened_video = False

        return sample

    def create_phase_trajectories(self):
        self.input_phase_trajectory = []
        self.input_img_list = []
        self.current_idx_list = []
        self.video_name_list = []
        self.segment_weights_list = []
        self.video_length_list = []
        self.class_count = np.zeros(self.num_classes)

        for video in tqdm(self.video_idx_list.items(), desc="preparing file list"):
            video_name = video[0].split("/")[-1].split(".")[0]
            video_start_idx = video[1]["start_idx"]
            video_end_idx = video[1]["end_idx"]
            video_total_length = video_end_idx - video_start_idx

            for idx in range(video_start_idx, video_end_idx, self.time_step):
                if idx >= (video_start_idx + self.past_length):
                    # copy the slice to the trajectory
                    input_phase_trajectory_tmp = self.labels_list[idx - self.past_length + 1 : idx + 1]
                    img_list_tmp = self.images_list[idx - self.past_length + 1 : idx + 1]

                elif idx < (video_start_idx + self.past_length):
                    # padding nan to the begining of the sequence
                    padding_length = self.past_length - (idx - video_start_idx) - 1
                    nan_vector = []
                    for idx_nan in range(padding_length):
                        nan_vector.append(float("NaN"))

                    input_phase_trajectory_tmp = nan_vector.copy()
                    input_phase_trajectory_tmp.extend(self.labels_list[video_start_idx : idx + 1])

                    img_list_tmp = nan_vector.copy()
                    img_list_tmp.extend(self.images_list[video_start_idx : idx + 1])

                # no nan labels in the dataset
                if any(np.isnan(sample) for sample in input_phase_trajectory_tmp):
                    continue
                
                # count phases:
                count_tmp = np.zeros(self.num_classes)
                for idx_tmp in range(len(input_phase_trajectory_tmp)):
                    count_tmp[input_phase_trajectory_tmp[idx_tmp]] += 1
                self.class_count += count_tmp

                weights_segment = calculateSegmentsWeight(count_tmp)
                self.segment_weights_list.append(weights_segment)

                # convert list to Tensor
                input_phase_trajectory_tmp = torch.Tensor(input_phase_trajectory_tmp).unsqueeze(0).unsqueeze(2).type(torch.int)
                #TODO: move to a different folder
                from data_interface.base_utils import create_onehot_nan

                input_phase_trajectory_tmp = create_onehot_nan(input_phase_trajectory_tmp, self.num_classes)
                input_phase_trajectory_tmp = input_phase_trajectory_tmp.squeeze(0)

                self.current_idx_list.append(idx)
                self.video_length_list.append(video_total_length)
                self.video_name_list.append(video_name)
                self.input_phase_trajectory.append(input_phase_trajectory_tmp)
                self.input_img_list.append(img_list_tmp)

        # normalize the segment weights
        self.segment_weights_list = np.array(self.segment_weights_list)
        self.segment_weights_list = self.segment_weights_list / np.sum(self.segment_weights_list)
        print("finish creating trajectories, in total " + str(len(self.input_phase_trajectory)) + "trajectories")

    def get_video_id_from_video_name(self, video_name):
        if self.video_to_emr_map is None:
            return None
        else:
            _, filename = os.path.split(video_name)
            filename, _ = os.path.splitext(filename)
            return filename


# TODO(guy.rosman) MOVING_DAY Yutong, Fix the doc string for factor default. Also, not clear what does factor mean.
def calculateSegmentsWeight(count=None, factor=1.2):
    """
    compute the sampling weight for each segment
    :param count: the number of phases in the segment
    :param factor:  with factor = 1.2 the effective number over the number of sampled segments is around 4
    :return: the weight of the segment
    """
    n_class = sum(count > 0)
    weight = np.exp(factor * n_class)
    return weight
