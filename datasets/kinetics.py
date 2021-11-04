# Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/kinetics.py

import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import av
import torch
import torch.utils.data
from joblib import Memory
from ride.utils.env import CACHE_PATH, NUM_CPU
from ride.utils.io import load_json
from ride.utils.logging import getLogger
from tqdm.contrib.concurrent import process_map

from . import decoder as decoder
from . import video_container as container

logger = getLogger(__name__)
cache = Memory(CACHE_PATH, verbose=1).cache


class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
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
        num_retries=10,
        num_ensemble_views=1,
        num_spatial_crops=1,
        *args,
        **kwargs,
    ):
        """
        `Kinetics <https://deepmind.com/research/open-source/open-source-datasets/kinetics/>`_
        dataset.

        Kinetics is an action recognition video dataset.
        It has multiple versions available which are identified by their number of classes (400, 600, 700)

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
        # Only support train, val, and test split.
        assert split in [
            "train",
            "val",
            "validate",
            "test",
        ], "Split '{}' not supported for Kinetics".format(split)
        if split == "val":
            split = "validate"
        self.split = split
        self.root = root
        self.annotation_path = annotation_path
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.temporal_downsampling = temporal_downsampling
        self.target_fps = 30
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.label_transform = label_transform
        self.global_transform = global_transform
        self.num_retries = num_retries
        self.num_ensemble_views = num_ensemble_views
        assert num_spatial_crops in {1, 2, 3}
        self.num_spatial_crops = num_spatial_crops
        self.num_retries = num_retries

        logger.info("Loading Kinetics {}".format(self.split))
        (
            self.labels,
            self.file_paths,
            self.spatial_temporal_idx,
            self.video_meta,
            self.classes,
            _,  # num_not_found,
            self.video_inds,
        ) = validate_splits(
            root,
            annotation_path,
            split,
            num_ensemble_views,
            num_spatial_crops,
        )
        if len(self.classes) not in {400, 600, 700}:
            logger.warning(
                f"Only found {len(self.classes)} classes for {split} set, but expected either 400, 600, or 700 for Kinetics."
            )

    def __getitem__(self, index):
        video_container = None
        frames = None
        for _ in range(self.num_retries):
            try:
                video_container = container.get_video_container(
                    self.file_paths[index],
                    multi_thread_decode=False,
                    backend="pyav",
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self.file_paths[index], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                index = random.randint(0, len(self.file_paths) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            temporal_clip_idx = {
                "train": -1,  # pick random
                "validate": 0,  # pick middle of clip
                "test": (  # pick temporal position cf. current temporal index
                    self.spatial_temporal_idx[index] // self.num_spatial_crops
                ),
            }[self.split]
            frames = decoder.decode(
                video_container,
                self.temporal_downsampling,
                self.frames_per_clip,
                clip_idx=temporal_clip_idx,
                num_clips=self.num_ensemble_views if self.split == "test" else 1,
                video_meta=self.video_meta[index],
                target_fps=self.target_fps,
                backend="pyav",
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self.file_paths) - 1)
            else:
                break

        if frames is None:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self.num_retries)
            )

        video = frames
        audio = None
        label = self.labels[index]

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
        return len(self.file_paths)


def map_segments(segment_arr: List[float]) -> Tuple[str, str]:
    assert len(segment_arr) == 2
    return (
        str(int(segment_arr[0])).zfill(6),
        str(int(segment_arr[1])).zfill(6),
    )


def make_path_name(
    key_and_annotations: Tuple[str, dict], root_path: Path, extention="mp4"
) -> Optional[Tuple[Path, Optional[str]]]:
    key, annotation = key_and_annotations
    start, stop = map_segments(annotation["annotations"]["segment"])
    if "label" in annotation["annotations"] and annotation["annotations"]["label"]:
        label = annotation["annotations"]["label"]
        p = (
            root_path
            / annotation["subset"]
            / label
            / f"{key}_{start}_{stop}.{extention}"
        ).resolve()
    else:
        p = (
            root_path / annotation["subset"] / f"{key}_{start}_{stop}.{extention}"
        ).resolve()
        label = None

    return (p, label) if p.exists() else None


def validate_and_make_path(
    key_and_annotations: Tuple[str, dict],
    root_path: Path,
    min_frames: int,
    target_fps: float,
    sampling_rate: float,
):
    maybe_path = make_path_name(key_and_annotations, root_path)
    if not maybe_path:
        return None

    path_to_vid, label = maybe_path

    try:
        video_container = av.open(str(path_to_vid))

        if video_container is None:
            return None

        fps = float(video_container.streams.video[0].average_rate)
        video_size = video_container.streams.video[0].frames
        clip_size = sampling_rate * min_frames / target_fps * fps
        if clip_size > video_size:
            return None

    except Exception:
        return None

    # If we made it thus far, we should be good!
    return path_to_vid, label


@cache
def validate_splits(
    # min_frames: int,
    # target_fps: float,
    # sampling_rate: float,
    root: str,
    annotation_path: str,
    split: str,
    num_ensemble_views=1,
    num_spatial_crops=1,
):
    root_path = Path(root)
    assert root_path.is_dir()

    annotation_file = Path(annotation_path) / f"{split}.json"
    assert annotation_file.exists()

    annotations: Dict = load_json(annotation_file)

    # Validate annotations file
    test_key = next(iter(annotations))
    test_value = annotations[test_key]
    assert type(test_value) == dict
    assert "subset" in test_value
    assert "annotations" in test_value
    assert "label" in test_value["annotations"]
    assert "segment" in test_value["annotations"]

    # Determine valid files
    maybe_results = process_map(
        partial(
            make_path_name,
            root_path=root_path,
        ),
        annotations.items(),
        chunksize=max(10, NUM_CPU // 2),
        desc=f"Validating Kinetics {split}",
    )

    results: List[Tuple[Path, str]] = list(filter(bool, maybe_results))  # type: ignore
    num_not_found = len(maybe_results) - len(results)
    logger.info(
        f"{len(results)} / {len(maybe_results)} ({100*len(results)/len(maybe_results):.1f}%) of videos were valid"
    )

    assert len(results) > 0
    file_paths, labels = zip(*results)
    file_paths, labels = list(file_paths), list(labels)

    classes = sorted(set(labels))  # type: ignore
    class_to_idx = {n: i for i, n in enumerate(classes)}
    labels = [class_to_idx[lbl] for lbl in labels]
    file_paths = [str(p) for p in file_paths]

    num_clips = num_ensemble_views * num_spatial_crops if split in ["test"] else 1

    _labels = []
    _file_paths = []
    _spatial_temporal_idx = []
    _video_meta = {}
    _video_idx = []
    for i in range(len(file_paths)):
        for j in range(num_clips):
            _labels.append(labels[i])
            _file_paths.append(file_paths[i])
            _spatial_temporal_idx.append(j)
            _video_meta[i * num_clips + j] = {}
            _video_idx.append(i)

    return (
        _labels,
        _file_paths,
        _spatial_temporal_idx,
        _video_meta,
        classes,
        num_not_found,
        _video_idx,
    )
