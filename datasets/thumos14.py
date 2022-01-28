import math
import pickle
import random
from functools import partial
from math import inf
from pathlib import Path
from typing import Optional

try:
    import av
except Exception:
    pass
import numpy as np
import torch
import torch.utils.data
from joblib import Memory
from ride.utils.env import CACHE_PATH, NUM_CPU
from ride.utils.logging import getLogger
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from datasets.decoder import pyav_decode_stream

logger = getLogger(__name__)
cache = Memory(CACHE_PATH, verbose=1).cache

CLASSES = [
    "None",
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking",
    "Ambiguous",
]


class Thumos14(torch.utils.data.Dataset):
    """
    Thumos14 activity detection video loader.
    """

    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips=1,
        temporal_downsampling=6,
        split="train",
        video_transform=None,
        audio_transform=None,
        label_transform=None,
        global_transform=None,
        num_retries=10,
        num_spatial_crops=1,
        skip_short_videos=True,
        skip_clips_with_no_actions=False,
        *args,
        **kwargs,
    ):
        """
        `Thumos14 <http://crcv.ucf.edu/THUMOS14/>` activity detection dataset.

        This dataset considers every video as a collection of video clips of fixed size, specified
        by ``frames_per_clip``, where the step in frames between each clip is given by
        ``step_between_clips``.

        Args:
            root (str): Root directory of the Kinetics Data.
            annotation_path (str): path to the folder containing the split files
            frames_per_clip (int): number of frames in a clip
            step_between_clips (int): number of frames between each clip
            temporal_downsampling (int): the downsampling factor to use (default: 6)
            split (str, optional): Which split to use (Options are ["train", "val", "test"])
            video_transform (callable, optional): A function that takes in a TxHxWxC video
                and returns a transformed version.
            audio_transform (callable, optional): A function that applies a transformation to the audio input.
                Since this dataset doesn't load audio data, this function is a dummy.
            label_transform (callable, optional): A function that applies a transformation to the labels.
            global_transform (callable, optional): A function that applies a transformation to the
                (video, audio, label, video_inds) tuple.

        Returns:
            video (Tensor[T, H, W, C]): the `T` video frames
            audio(None): Currently not supported
            label (Tensor[T]): frame-by-frame classes
            video_ind (int): index of the originating video
        """
        # Only support train, val, and test split.
        assert split in [
            "train",
            "val",
            "validate",
            "test",
        ], "Split '{}' not supported for Thumos14".format(split)
        if split == "validate":
            split = "val"
            logger.info(
                "Using test set for validation as is customary for THUMOS14 Action Detection."
            )
        elif split == "train":
            logger.info(
                "Using validation set for training as is customary for THUMOS14 Action Detection."
            )
        self.split = split
        used_ds_split = {"train": "val", "val": "test", "test": "test"}[self.split]
        self.data_path = Path(root) / used_ds_split
        assert self.data_path.is_dir()
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.temporal_downsampling = temporal_downsampling
        self.target_fps = 30 / temporal_downsampling
        self.video_transform = video_transform
        # self.audio_transform = audio_transform
        self.label_transform = label_transform
        self.global_transform = global_transform
        self.num_retries = num_retries
        assert num_spatial_crops in {1, 2, 3}
        self.num_spatial_crops = num_spatial_crops
        self.num_retries = num_retries

        prepare_labels(used_ds_split, root, annotation_path, self.target_fps)

        annotation_file = (
            Path(annotation_path) / f"{used_ds_split}_{self.target_fps}fps.pickle"
        )
        assert annotation_file.exists()

        with open(annotation_file, "rb") as f:
            annotations = pickle.load(f)

        # Validate annotations file contents
        test_key = next(iter(annotations))
        test_value = annotations[test_key]
        assert type(test_value) == list

        # Used for multi-crop testing
        num_clips = num_spatial_crops if self.split in ["test"] else 1

        self.class_counts = [0 for _ in range(len(self.classes))]
        self._clip_meta = []  # (video_id, start_idx, end_idx, clip_targets)
        for video_idx, video_id in enumerate(annotations.keys()):
            # Densely sample all sub-videos of with `frames_per_clip`
            # Use positional encording instead of one-hot, since there is only one action per time-step
            vid_target = annotations[video_id]

            if skip_short_videos and len(vid_target) < frames_per_clip:
                continue

            for start_idx, end_idx in zip(
                range(0, len(vid_target) - frames_per_clip, step_between_clips),
                range(frames_per_clip, len(vid_target), step_between_clips),
            ):
                clip_targets = vid_target[start_idx:end_idx]

                if skip_clips_with_no_actions:
                    if all([ct == 0 for ct in clip_targets]):
                        continue

                for ct in clip_targets:
                    self.class_counts[ct] += 1

                for _ in range(num_clips):
                    self._clip_meta.append(
                        (video_idx, video_id, start_idx, end_idx, clip_targets)
                    )

    def __len__(self):
        """
        Returns:
            (int): the number of clips in the dataset.
        """
        return len(self._clip_meta)

    @property
    def classes(self):
        return CLASSES

    def __getitem__(self, index):
        video = None
        for _ in range(self.num_retries):
            video_idx, video_id, start_idx, end_idx, labels = self._clip_meta[index]
            video_path = str(self.data_path / (video_id + ".mp4"))

            video = decode_video(
                video_path,
                start_idx,
                end_idx,
                self.target_fps,
            )

            # If decoding failed (wrong format, video is too short, and etc), select another video.
            if video is None:
                index = random.randint(0, len(self._clip_meta) - 1)
            else:
                break

        if video is None:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self.num_retries)
            )

        labels = torch.tensor(labels)
        assert len(labels) == len(video)

        audio = None

        if self.video_transform is not None:
            video = self.video_transform(video)

        # if self.audio_transform is not None:
        #     audio = self.audio_transform(audio)

        if self.label_transform is not None:
            labels = self.label_transform(labels)

        sample = (video, audio, labels, video_idx)

        if self.global_transform is not None:
            sample = self.global_transform(sample)

        return sample


def decode_video(
    video_path: str,
    start_idx: int,
    end_idx: int,
    target_fps: float,
) -> Optional[torch.Tensor]:
    try:
        container = av.open(video_path)
    except Exception as e:
        logger.info("Failed to load video from {} with error {}".format(video_path, e))
        return None

    video_stream = container.streams.video[0]
    video_fps = float(video_stream.average_rate)
    video_duration = container.duration / 1e6  # seconds

    # Some fractions are noted as 30/1, others as 30000/1001.
    # video pts are stepped according to the denominator of the frame_rate (e.g. 1 or 1001)
    pts_base = (video_stream.average_rate * video_stream.time_base).denominator

    if video_duration is None:
        # Decode the entire video.
        video_start_pts, video_end_pts = 0, inf
    else:
        # Decode video selectively
        timebase = video_fps / target_fps * pts_base
        video_start_pts = int(start_idx * timebase)

        if end_idx == -1:
            end_idx = int(((video_stream.duration - 1) / timebase) + 1)

        video_end_pts = int((end_idx - 1) * timebase) + 1

    frames, _ = pyav_decode_stream(
        container,
        video_start_pts,
        video_end_pts,
        video_stream,
        seek_margin=100 * pts_base,
    )
    num_decoded_frames = len(frames)
    num_wanted_frames = int(end_idx) - start_idx

    # Select frame indices
    frame_inds = torch.linspace(
        0, target_fps * (num_wanted_frames - 1), num_wanted_frames
    )

    if num_decoded_frames * 1.5 < target_fps * (num_wanted_frames - 1):
        container.close()
        return None

    if num_decoded_frames < target_fps * (num_wanted_frames - 1):
        frame_inds = torch.clamp(frame_inds, 0, num_decoded_frames - 1)

    frame_inds = frame_inds.long()

    # Convert to torch Tensor
    frames = torch.as_tensor(
        np.stack([frames[i].to_rgb().to_ndarray() for i in frame_inds])
    )

    container.close()

    return frames


def get_default_labels(video_path, target_fps: float):
    try:
        container = av.open(str(video_path))
    except Exception as e:
        logger.info("Failed to load video from {} with error {}".format(video_path, e))
        return None

    video_duration = container.duration / 1e6  # seconds
    container.close()

    video_id = video_path.stem
    default_label = [0 for _ in range(math.ceil(video_duration * target_fps))]
    return (video_id, default_label)


def prepare_labels(split: str, data_path: str, annotation_path: str, target_fps: float):

    parsed_annotation_path = Path(annotation_path) / f"{split}_{target_fps}fps.pickle"
    if parsed_annotation_path.exists():
        return

    logger.info(f"Preparing THUMOS-14 labels for {split} split")

    # Pre-fill labels with zeros
    video_paths = list((Path(data_path) / split).glob("*.mp4"))

    # Single thread
    # label_list = []
    # for vp in tqdm(video_paths, desc="Checking video durations"):
    #     lbls = get_default_labels(vp, target_fps)
    #     if lbls is not None:
    #         label_list.append(lbls)
    # label_dict = dict(label_list)

    # Multi-thread
    label_dict = dict(
        [
            res
            for res in process_map(
                partial(
                    get_default_labels,
                    target_fps=target_fps,
                ),
                video_paths,
                chunksize=max(10, NUM_CPU // 4),
                desc="Checking video durations",
            )
            if res is not None
        ]
    )

    # Iterate through annotations and update
    class2idx = {c: i for i, c in enumerate(CLASSES)}

    # NB: If diving is added after cliff-diving, there will be no labels for the former
    anno_paths = sorted((Path(annotation_path) / split).glob(f"*_{split}.txt"))[::-1]
    for anno_path in tqdm(anno_paths, desc="Parsing annotations"):
        class_name = anno_path.stem.split("_")[0]
        class_index = class2idx[class_name]
        with open(anno_path, mode="r") as f:
            for line in f.readlines():
                line = line.split(" ")
                video_id = line[0]
                start = round(float(line[-2]) * target_fps)
                stop = min(
                    round(float(line[-1].strip()) * target_fps),
                    len(label_dict[video_id]),
                )
                label_dict[video_id][start:stop] = [
                    class_index for _ in range(stop - start)
                ]

    # Save annotatio-data
    with open(parsed_annotation_path, "wb") as file:
        logger.info(f"Saving parsed annotations to {parsed_annotation_path}")
        pickle.dump(label_dict, file)
