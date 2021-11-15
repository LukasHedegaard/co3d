# Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/ssv2.py

import random
from itertools import chain as chain
from pathlib import Path
from typing import List
import json
import pandas as pd
import torch
import torch.utils.data
from ride.utils.logging import getLogger
from torchvision.io import read_image
from . import utils as utils

# from ride.utils.env import CACHE_PATH
# from joblib import Memory
# cache = Memory(CACHE_PATH, verbose=1).cache

logger = getLogger(__name__)


class Ssv2(torch.utils.data.Dataset):
    """
    Something-something v2 video loader. Construct the video loader, then sample
    clips from the videos. For training and validation, a single clip is randomly
    sampled from every video with random cropping, scaling, and flipping. For
    testing, multiple clips are uniformaly sampled from every video with uniform
    cropping. For uniform cropping, we take the left, center, and right crop if
    the width is larger than height, or take top, center, and bottom crop if the
    height is larger than the width.
    """

    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips=1,
        temporal_downsampling=None,
        split="train",
        video_transform=None,
        audio_transform=None,
        label_transform=None,
        global_transform=None,
        num_ensemble_views=1,
        num_spatial_crops=1,
        *args,
        **kwargs,
    ):
        """
        Load Something-Something V2 data (frame paths, labels, etc. ) to a given
        Dataset object. The dataset could be downloaded from Something-Something
        official website (https://20bn.com/datasets/something-something).


        This dataset consider every video as a collection of video clips of fixed size, specified
        by ``frames_per_clip``, where the step in frames between each clip is given by
        ``step_between_clips``.

        Args:
            root (str): Root directory of the Kinetics Data.
            annotation_path (str): path to the folder containing the split files
            frames_per_clip (int): number of frames in a clip
            step_between_clips (int): number of frames between each clip
            split (str, optional): Which split to use (Options are ["train", "val", "test"])
            video_transform (callable, optional): A function/transform that  takes in a TxHxWxC video
                and returns a transformed version.
            audio_transform (callable, optional): A function that applies a transformation to the audio input.
            label_transform (callable, optional): A function that applies a transformation to the labels.
            global_transform (callable, optional): A function that applies a transformation to the
                (video, audio, label, video_inds) tuple.

        Returns:
            video (Tensor[T, H, W, C]): the `T` video frames
            audio(None): Currently not supported
            label (int): class of the video clip
            video_ind (int): index of the originating video
        """
        # Only support train, val, and test mode.
        assert split in [
            "train",
            "val",
            "validate",
            "test",
        ], "Split '{}' not supported for Kinetics".format(split)
        if split == "validate":
            split = "val"
        if split == "test":
            logger.info("Using Something-Something V2 val set for testing")
        self.split = split
        self.root = root
        self.data_root = Path(self.root) / "frames"
        self.annotation_path = annotation_path
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.temporal_downsampling = temporal_downsampling
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.label_transform = label_transform
        self.global_transform = global_transform
        self.num_ensemble_views = num_ensemble_views
        assert num_spatial_crops in {1, 2, 3}
        self.num_spatial_crops = num_spatial_crops
        # self.target_fps = 30
        # self.num_retries = num_retries

        logger.info("Loading Something-Something V2 {}".format(self.split))
        (
            self.classes,
            self.path_to_videos,
            self.labels,
            self.video_inds,
            self.spatial_temporal_idx,
        ) = load_annotations(
            self.annotation_path,
            split="train" if self.split == "train" else "val",
            num_crops=num_ensemble_views * num_spatial_crops,
        )

    def get_seq_frames(self, index: int) -> List[int]:
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        temporal_clip_index = (
            -1
            if self.split in ["train", "val"]
            else self.spatial_temporal_idx[index] // self.num_spatial_crops
        )

        video_length = len(self.path_to_videos[index])
        assert video_length == len(self.labels[index])

        clip_length = (self.frames_per_clip - 1) * self.temporal_downsampling + 1
        if temporal_clip_index == -1:
            if clip_length > video_length:
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        else:
            if self.num_ensemble_views > 1:
                gap = float(max(video_length - clip_length, 0)) / (
                    self.num_ensemble_views - 1
                )
                start = int(round(gap * temporal_clip_index))
            else:
                # Select video center
                start = max(video_length - clip_length, 0) // 2

        seq = [
            max(min(start + i * self.temporal_downsampling, video_length - 1), 0)
            for i in range(self.frames_per_clip)
        ]

        return seq

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """

        frame_inds = self.get_seq_frames(index)

        # Load image frames
        video = torch.stack(
            [
                read_image(str(self.data_root / self.path_to_videos[index][frame_ind]))
                for frame_ind in frame_inds
            ]
        )
        video = video.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)

        # Load labels for selection and aggregate to clip-level labels
        # Use video-level labels for testing
        label_selection = (
            slice(frame_inds[0], frame_inds[-1])
            if self.split == "train"
            else slice(None)
        )
        label_set = set(self.labels[index][label_selection])
        label_inds = torch.tensor(
            [
                int(lbl)
                for lbl in set(
                    chain.from_iterable([str(lbls).split(",") for lbls in label_set])
                )
                if lbl != "nan"
            ],
            dtype=torch.long,
        )
        label = torch.zeros(len(self.classes))  # multi-hot encoding
        label[label_inds] = 1

        audio = None

        if self.video_transform is not None:
            video = self.video_transform(video)

        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        if self.label_transform is not None:
            label = self.label_transform(label)

        sample = (video, audio, label, self.video_inds[index])

        if self.global_transform is not None:
            sample = self.global_transform(sample)

        return sample

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.path_to_videos)


# @cache
def load_annotations(
    annotation_path: str,
    split: str,
    num_crops: int,
):
    """
    Construct the video loader.
    """
    annotation_path = Path(annotation_path)

    # Load label names
    with open(str(annotation_path / "something-something-v2-labels.json")) as f:
        label_dict = json.load(f)

    ld_inv = {int(v): k for k, v in label_dict.items()}
    classes = [ld_inv[i] for i in sorted(ld_inv.keys())]

    # Load labels
    label_path = (
        annotation_path
        / f"something-something-v2-{'train' if split == 'train' else 'validation'}.json"
    )
    with open(str(label_path)) as f:
        label_json = json.load(f)

    video_names = []
    labels = []
    for video in label_json:
        video_name = video["id"]
        template = video["template"]
        template = template.replace("[", "")
        template = template.replace("]", "")
        label = int(label_dict[template])
        video_names.append(video_name)
        labels.append(label)

    # Load image paths and labels from csv with the format:
    #   original_vido_id video_id frame_id path labels
    #   46GP8 0 0 46GP8/46GP8-000001.jpg "147"
    #   46GP8 0 1 46GP8/46GP8-000002.jpg "59,147"
    #   ...
    csv_path = annotation_path / f'{"train" if split == "train" else "val"}.csv'
    path_to_videos, _ = utils.load_image_lists(str(csv_path))

    assert len(path_to_videos) == len(video_names)

    # Sort to match loaded `video_names`
    new_paths, new_labels = [], []
    for index in range(len(video_names)):
        if video_names[index] in path_to_videos:
            new_paths.append(path_to_videos[video_names[index]])
            new_labels.append(labels[index])

    labels = new_labels
    path_to_videos = new_paths

    path_to_videos = list(
        chain.from_iterable([[x] * num_crops for x in path_to_videos])
    )
    labels = list(chain.from_iterable([[x] * num_crops for x in labels]))
    spatial_temporal_idx = list(
        chain.from_iterable([range(num_crops) for _ in range(len(labels))])
    )
    video_inds = list(
        chain.from_iterable([[i] * num_crops for i in range(len(labels))])
    )

    logger.debug(
        f"Something-Something-V2 {split} loaded (size: {len(path_to_videos)}) from {str(annotation_path)}"
    )

    return (
        classes,
        path_to_videos,
        labels,
        video_inds,
        spatial_temporal_idx,
    )
