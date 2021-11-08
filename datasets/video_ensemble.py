from functools import partial
from typing import Callable, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.interfaces import VideoClassificationDataset
from datasets.transforms import CropVideo
from datasets.videoclips import get_inds_for_video, get_num_videos


def backfilling(fn: Callable):
    def wrapped(population, k, *args, **kwargs):
        if len(population) <= k:
            return population + fn(population, len(population) - k, *args, **kwargs)
        else:
            return fn(population, k, *args, **kwargs)

    return wrapped


def if_enough(fn: Callable):
    def wrapped(population, k, *args, **kwargs):
        if len(population) <= k:
            return population
        else:
            return fn(population, k, *args, **kwargs)

    return wrapped


def uniform_subset(population: Sequence[int], k: int) -> List[int]:
    m = len(population)
    inds = [*np.round(np.array(range(0, k)) * (m - 1) / k).astype(int), m - 1]
    l2 = list(np.array(population)[inds])
    return l2


class SpatiallySamplingVideoEnsemble(Dataset):
    """Wraps a video dataset ensemble view.

    For each input video clip, a spatial_sampling_strategy is used to extract multiple clips.
    The return of this dataset is an example index in addition to the original data.
    """

    def __init__(
        self,
        dataset: VideoClassificationDataset,
        crop_size: Union[int, Tuple[int, int]],
        spatial_sampling_strategy: str = "diagonal",
        video_transform=None,
    ):
        """Wraps a video dataset such as Hmdb51 or Ucf101 to create an ensemble view.

        Args:
            video_clips (VideoClips): Source object from which videos will be sampled
            spatial_sampling_strategy (str, optional):
                Spatial sampling strategy for clips.
                Suppoerted strategies: ["center","vertical","diagonal","horizontal"]
                Defaults to "diagonal".
            video_transform ([type], optional): Transform function applied to video. Defaults to None.
            index_map_fn (Callable[[int],int]):
                if the wrapped dataset is itself an ensemble, pass a function for mapping its item index to video index
        """
        super(SpatiallySamplingVideoEnsemble, self).__init__()
        self.dataset = dataset
        if hasattr(self.dataset, "classes"):
            self.classes = self.dataset.classes

        crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

        self.sampling_multiplier: int
        self.spatial_sampling_fn: Selector
        self.sampling_multiplier, self.spatial_sampling_fn = {
            "center": (1, _spatial_center_sampling(crop_size)),
            "vertical": (3, _spatial_vertical_sampling(crop_size)),
            "horizontal": (3, _spatial_horizontal_sampling(crop_size)),
            "diagonal": (3, _spatial_diagonal_sampling(crop_size)),
        }[spatial_sampling_strategy]

        self.len = len(dataset) * self.sampling_multiplier

        self.video_transform = video_transform

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        dataset_idx = index // self.sampling_multiplier
        # Assuming a sample has (video, ..., video_index)
        sample = self.dataset[dataset_idx]
        assert isinstance(
            sample[-1], int
        ), "Last index in a sample should be the video_index"
        video, rest = sample[0], sample[1:]
        video = self.spatial_sampling_fn[index](video)
        if self.video_transform:
            video = self.video_transform(video)
        return (video, *rest)




class TemporallySamplingVideoEnsemble(Dataset):
    """Wraps a video dataset which uses the torchvision VideoClips object (e.g. Hmdb51 and Ucf101)
    to create an ensemble view, uniformly sampling videos through time.
    For each video, num_temporal_clips are uniformly sampled from the temporal dimension.
    """

    def __init__(
        self,
        dataset: VideoClassificationDataset,
        num_temporal_clips: int = 10,
        video_transform=None,
    ):
        """Wraps a video dataset such as Hmdb51 or Ucf101 to create an ensemble view.

        Args:
            video_clips (VideoClips): Source object from which videos will be sampled
            num_temporal_clips (int, optional): Number of temporal clips, that are sampled uniformly along the temporal dimension. Defaults to 10.
            spatial_sampling_strategy (str, optional):
                Spatial sampling strategy for clips.
                Suppoerted strategies: ["vertical","diagonal","horizontal"]
                Defaults to "diagonal".
            video_transform ([type], optional): Transform function applied to video. Defaults to None.
        """
        super(TemporallySamplingVideoEnsemble, self).__init__()
        self.dataset = dataset
        self.num_temporal_clips = num_temporal_clips
        self.video_transform = video_transform
        sampling_fn = partial(if_enough(uniform_subset), k=num_temporal_clips)

        # Build a dict of clip inds -> video inds
        self._clip2video = {
            c: v
            for v in range(get_num_videos(self.dataset.video_clips))
            for c in sampling_fn(get_inds_for_video(self.dataset.video_clips, v))
        }
        self._index2clip = list(self._clip2video.keys())

    def __len__(self) -> int:
        return len(self._clip2video)

    def get_video_index(self, item_idx: int) -> int:
        return self._clip2video[item_idx]

    def __getitem__(self, index: int):
        dataset_index = self._index2clip[index]
        # Assuming a sample has (video, ..., video_index)
        sample = self.dataset[dataset_index]
        assert isinstance(
            sample[-1], int
        ), "Last index in a sample should be the video_index"
        video, rest = sample[0], sample[1:]
        if self.video_transform:
            video = self.video_transform(video)
        return (video, *rest)


T = TypeVar("T")


class Selector:
    def __init__(self, selectables: Sequence[T]):
        self.selectables = selectables

    def __len__(self) -> int:
        return len(self.selectables)

    def __getitem__(self, index: int) -> T:
        i = index % len(self)
        return self.selectables[i]

    def __call__(self, index: int) -> T:
        return self.__getitem__(index)


def _spatial_center_sampling(
    crop_size: Tuple[int, int]
) -> Callable[[torch.Tensor], Sequence[torch.Tensor]]:
    return Selector([CropVideo(crop_size, "center")])


def _spatial_diagonal_sampling(
    crop_size: Tuple[int, int]
) -> Callable[[torch.Tensor], Sequence[torch.Tensor]]:
    return Selector(
        [
            CropVideo(crop_size, "top_left"),
            CropVideo(crop_size, "center"),
            CropVideo(crop_size, "bottom_right"),
        ]
    )


def _spatial_vertical_sampling(
    crop_size: Tuple[int, int]
) -> Callable[[torch.Tensor], Sequence[torch.Tensor]]:
    return Selector(
        [
            CropVideo(crop_size, "top_center"),
            CropVideo(crop_size, "center"),
            CropVideo(crop_size, "bottom_center"),
        ]
    )


def _spatial_horizontal_sampling(
    crop_size: Tuple[int, int]
) -> Callable[[torch.Tensor], Sequence[torch.Tensor]]:
    return Selector(
        [
            CropVideo(crop_size, "center_left"),
            CropVideo(crop_size, "center"),
            CropVideo(crop_size, "center_right"),
        ]
    )
