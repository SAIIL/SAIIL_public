import collections
import glob
import json
import os
import unicodedata
import warnings
import xml.etree.ElementTree as ET

import cv2
import multidict
import numpy as np
import torch
import torch.utils.data.sampler
import tqdm
from torch.utils.data import Dataset, dataloader

from data_interface.surgical_dataset import SurgicalDataset


def collate_filter_empty_elements(batch):
    # filter the empty elements in the batch
    batch = list(filter(lambda x: x is not None, batch))
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

    y_onehot = torch.zeros(batch_size, time_size, num_classes).float().to(path.device)

    y_onehot.zero_()
    for batch_idx in range(batch_size):
        path_batch = path[batch_idx].squeeze(0).cpu().numpy()
        for idx, path_sample in enumerate(path_batch):
            if not np.isnan(path_sample):
                y_onehot[batch_idx, idx, path_sample] = 1.0
    return y_onehot


def write_results_to_txt(
    estimate_list=None,
    class_names=None,
    res_dir="results_txt/camma/",
    video_name_list_filename=None,
    estimate_fps=1,
    write_fps=30,
):
    """
    Write the formatted estimation result into txt files
    :param estimate_list:
    :param class_names:
    :param res_dir:
    :param video_name_list_filename:
    :param estimate_fps: the estimation frame rate
    :param write_fps: the outp
    ut frame rate
    :return:
    """

    time_ratio = int(write_fps / estimate_fps)
    os.makedirs(res_dir, exist_ok=True)
    with open(video_name_list_filename) as f_video_names:
        video_name_list = list(json.load(f_video_names).values())

    for video_idx in estimate_list.keys():
        video_name = video_name_list[video_idx - 1].split("_")[0]
        estimate = np.argmax(estimate_list[video_idx], axis=1).astype(int)
        write_file_name = res_dir + video_name + ".txt"
        f_out = open(write_file_name, "w")
        f_out.write("Frame\tPhase\n")
        for t in range(estimate.shape[0]):
            for idx1 in range(time_ratio):
                t_idx = t * time_ratio + idx1
                f_out.write(str(t_idx) + "\t" + class_names[estimate[t]] + "\n")

        f_out.close()
    return
