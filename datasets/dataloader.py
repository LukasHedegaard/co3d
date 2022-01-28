from argparse import ArgumentParser, Namespace
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Tuple, Union

import torch
# from pytorchvideo.transforms import RandAugment
from ride import Configs, RideClassificationDataset
from ride.utils.env import DATASETS_PATH, NUM_CPU
from ride.utils.logging import getLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    ToTensorVideo,
)

from datasets.ava import Ava
from datasets.charades import Charades
from datasets.ssv2 import Ssv2
from datasets.kinetics import Kinetics
from datasets.thumos14 import Thumos14
from datasets.transforms import RandomShortSideScaleJitterVideo, discard_audio
from datasets.tvseries import TvSeries
from datasets.video_ensemble import SpatiallySamplingVideoEnsemble

logger = getLogger("datasets")


class ActionRecognitionDatasets(RideClassificationDataset):
    """Adds a dataloader

    Side-effects:
        Adds self.dataloader
             self.input_shape: Tuple[int,...]
             self.output_shape: Tuple[int,...]
             self.classes: List[str]
             self.task: str
    """

    dataloader: ...

    @staticmethod
    def configs() -> Configs:
        c = RideClassificationDataset.configs()
        c.add(
            name="dataset",
            type=str,
            default="kinetics400",
            choices=[
                "kinetics400",
                "kinetics600",
                "kinetics3",
                "thumos14",
                "tvseries",
                "charades",
                "ssv2",
                "ava",
            ],
            strategy="constant",
            description=f"Dataset name. It is assumed that these datasets are available in the DATASETS_PATH env variable ({str(DATASETS_PATH)})",
        )
        c.add(
            name="dataset_path",
            type=str,
            default=None,
            strategy="constant",
            description="Dataset path. If None, a default path will be inferred from the choice of the 'dataset' and 'dataset_version' parameters.",
        )
        c.add(
            name="dataloader_prefetch_factor",
            type=int,
            default=2,
            strategy="constant",
            description="Dataloader prefetch_factor.",
        )
        c.add(
            name="frames_per_clip",
            type=int,
            default=32,
            strategy="constant",
            description=(
                "Number of frames per clip. "
                "In conjunction with 'temporal_downsampling', this parameter has a large effect on the number of samples generated. "
                "If too large, a given video may not produce any clips."
            ),
        )
        c.add(
            name="temporal_downsampling",
            type=int,
            default=1,
            strategy="constant",
            description="Resampled frame rate for video clips.",
        )
        c.add(
            name="step_between_clips",
            type=int,
            default=2,
            strategy="constant",
            description="Number of steps between video clips.",
        )
        c.add(
            name="dataset_fold",
            type=int,
            default=1,
            choices=[1, 2, 3],
            strategy="constant",
            description="Dataset fold.",
        )
        c.add(
            name="image_size",
            type=int,
            default=160,
            strategy="constant",
            description="Target image size.",
        )
        c.add(
            name="val_split_pct",
            type=float,
            default=0.15,
            strategy="constant",
            description=(
                "Percentage of train set to be used for validation. "
                "Only applied if dataset supports it"
            ),
        )
        c.add(
            name="test_ensemble",
            type=int,
            default=0,
            strategy="constant",
            description="Flag indicating whether the test dataset should yield a clip ensemble.",
        )
        c.add(
            name="test_ensemble_temporal_clips",
            type=int,
            default=10,
            strategy="constant",
            description="Number of temporal clips to use for ensemble testing.",
        )
        c.add(
            name="test_ensemble_spatial_sampling_strategy",
            type=str,
            default="center",
            strategy="constant",
            choices=["center", "vertical", "horizontal", "diagonal"],
            description="How to spatially sample clips.",
        )
        c.add(
            name="normalisation_style",
            type=str,
            default="feichtenhofer",
            strategy="choice",
            choices=["imagenet", "feichtenhofer", "kopuklu"],
            description="The style of normalisation, i.e. choice of mean and std.",
        )
        c.add(
            name="rand_augment_magnitude",
            type=int,
            default=7,
            strategy="uniform",
            choices=[0, 25],
            description="RandAugment magnitude.",
        )
        c.add(
            name="rand_augment_num_layers",
            type=int,
            default=2,
            strategy="uniform",
            choices=[1, 6],
            description="RandAugment number of transorm layers.",
        )
        return c

    def __init__(self, hparams):
        self.prepare_data()

    def on_init_end(self, hparams):
        self.prepare_data()

    def prepare_data(self):
        self.dataloader = ActionRecognitionDatasetLoader.from_argparse_args(
            self.hparams
        )
        self.input_shape = (
            3,  # Input channels
            self.hparams.frames_per_clip,
            self.hparams.image_size,
            self.hparams.image_size,
        )
        self.classes = self.dataloader.classes
        self.output_shape = self.num_classes

        self.task = {
            "kinetics400": "classification",
            "kinetics600": "classification",
            "kinetics3": "classification",
            "charades": "classification",
            "ssv2": "classification",
            "ava": "classification",
            "thumos14": "detection",
            "tvseries": "detection",
        }[self.hparams.dataset]

        return self.dataloader

    def train_dataloader(self):
        return self.dataloader.train_dataloader

    def val_dataloader(self):
        return self.dataloader.val_dataloader

    def test_dataloader(self):
        return self.dataloader.test_dataloader


class ActionRecognitionDatasetLoader:
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    test_dataloader: DataLoader

    def __init__(
        self,
        dataset: str,
        image_size: int,
        frames_per_clip: int,
        step_between_clips: int,
        temporal_downsampling: int,
        fold: int,
        batch_size: int,
        val_split_pct: float = None,
        dataset_path: str = None,
        num_workers=NUM_CPU,
        test_ensemble=True,
        test_ensemble_temporal_clips=10,
        test_ensemble_spatial_sampling_strategy="diagonal",
        normalisation_style="imagenet",
        dataloader_prefetch_factor=2,
        rand_augment_magnitude=9,
        rand_augment_num_layers=2,
    ):
        if dataset_path:
            root_path = Path(dataset_path)
        else:
            root_path = DATASETS_PATH / dataset

        data_path = root_path / "data"
        annotation_path = root_path / "splits"

        assert root_path.is_dir(), f"{root_path} is not a valid directory"
        assert data_path.is_dir(), f"{data_path} is not a valid directory"
        assert annotation_path.is_dir(), f"{annotation_path} is not a valid directory"

        train_ds: Dataset
        val_ds: Dataset
        test_ds: Dataset

        # Grab train, val and test data
        train_ds, val_ds, test_ds = train_val_test(
            data_path=str(data_path),
            annotation_path=str(annotation_path),
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            temporal_downsampling=temporal_downsampling,
            fold=fold,
            val_split_pct=val_split_pct,
            image_size=image_size,
            test_ensemble=test_ensemble,
            test_ensemble_temporal_clips=test_ensemble_temporal_clips,
            test_ensemble_spatial_sampling_strategy=test_ensemble_spatial_sampling_strategy,
            normalisation_style=normalisation_style,
            rand_augment_magnitude=rand_augment_magnitude,
            rand_augment_num_layers=rand_augment_num_layers,
        )

        self.classes: List[str] = getattr(test_ds, "classes") or []
        self.num_classes = len(self.classes)

        # Pass to loaders
        self.train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=dataset != "ava",
            pin_memory=num_workers > 1,
            drop_last=True,
            prefetch_factor=dataloader_prefetch_factor,
            persistent_workers=True,
        )
        self.val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
            drop_last=True,
            prefetch_factor=dataloader_prefetch_factor,
            persistent_workers=True,
        )
        self.test_dataloader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
            drop_last=False,
            prefetch_factor=dataloader_prefetch_factor,
            persistent_workers=True,
        )

    @classmethod
    def from_argparse_args(
        cls,
        args: Union[Namespace, ArgumentParser],
    ) -> "ActionRecognitionDatasetLoader":
        if isinstance(args, ArgumentParser):
            args = args.parse_args()
        return ActionRecognitionDatasetLoader(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            image_size=args.image_size,
            frames_per_clip=args.frames_per_clip,
            step_between_clips=args.step_between_clips,
            temporal_downsampling=args.temporal_downsampling,
            fold=args.dataset_fold,
            val_split_pct=args.val_split_pct,
            batch_size=args.batch_size,
            num_workers=args.num_workers or NUM_CPU,
            test_ensemble=args.test_ensemble,
            test_ensemble_temporal_clips=args.test_ensemble_temporal_clips,
            test_ensemble_spatial_sampling_strategy=args.test_ensemble_spatial_sampling_strategy,
            normalisation_style=args.normalisation_style,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            rand_augment_magnitude=args.rand_augment_magnitude,
            rand_augment_num_layers=args.rand_augment_num_layers,
        )


@lru_cache()
def train_val_test(
    data_path: str,
    annotation_path: str,
    temporal_downsampling=None,
    step_between_clips=1,
    val_split_pct=0.15,
    frames_per_clip=16,
    fold=1,
    image_size: int = 240,
    test_ensemble=True,
    test_ensemble_temporal_clips=10,
    test_ensemble_spatial_sampling_strategy="diagonal",
    image_train_scale=(
        0.875,
        1.25,
    ),  # corresponds to 0.875 (crop) : 1 (min scale) : 1.25 (max scale)
    normalisation_style="imagenet",
    rand_augment_magnitude=9,
    rand_augment_num_layers=2,
) -> Tuple[Dataset, Dataset, Dataset]:

    MEAN, STD = {
        "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "feichtenhofer": ((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        "kopuklu": (
            (0.4338692858, 0.4045515923, 0.37760875),
            (0.1519876776, 0.14855877375, 0.1569763971),
        ),
        "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    }[normalisation_style]

    if "kinetics" in data_path.lower():
        Ds = Kinetics
    elif "thumos14" in data_path.lower():
        Ds = Thumos14
    elif "tvseries" in data_path.lower():
        Ds = TvSeries
    elif "charades" in data_path.lower():
        Ds = Charades
    elif "ssv2" in data_path.lower():
        Ds = Ssv2
    elif "ava" in data_path.lower():
        Ds = Ava
    else:
        raise ValueError(
            "'root_path' must contain either 'kinetics', 'thumos14', or 'tvseries'"
        )

    train_crop_pix = floor2(image_size * image_train_scale[0])
    train_scale_pix_min = image_size
    train_scale_pix_max = floor2(image_size * image_train_scale[1])
    assert (
        train_scale_pix_max > train_scale_pix_min
        and train_scale_pix_min > train_crop_pix
    )

    # swap_axes = partial(torch.swapaxes, axis0=0, axis1=1)
    train_transforms = Compose(
        [
            ToTensorVideo(),
            RandomShortSideScaleJitterVideo(
                min_size=train_scale_pix_min, max_size=train_scale_pix_max
            ),
            # swap_axes,  # (C, T, H, W) -> (T, C, H, W)
            # RandAugment(rand_augment_magnitude, rand_augment_num_layers),
            # swap_axes,  # (T, C, H, W) -> (C, T, H, W)
            CenterCropVideo(image_size),
            NormalizeVideo(mean=MEAN, std=STD),
        ]
    )

    eval_transforms = Compose(
        [
            ToTensorVideo(),
            RandomShortSideScaleJitterVideo(min_size=image_size, max_size=image_size),
            CenterCropVideo(image_size),
            NormalizeVideo(mean=MEAN, std=STD),
        ]
    )

    train = Ds(
        root=data_path,
        annotation_path=annotation_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        temporal_downsampling=temporal_downsampling,
        fold=fold,
        split="train",
        val_split=val_split_pct,
        video_transform=train_transforms,
        label_transform=None,
        global_transform=discard_audio,
    )

    val = Ds(
        root=data_path,
        annotation_path=annotation_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        temporal_downsampling=temporal_downsampling,
        fold=fold,
        split="val",
        val_split=val_split_pct,
        video_transform=eval_transforms,
        label_transform=None,
        global_transform=discard_audio,
    )

    if not test_ensemble:
        test = Ds(
            root=data_path,
            annotation_path=annotation_path,
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            temporal_downsampling=temporal_downsampling,
            fold=fold,
            split="test",
            video_transform=eval_transforms,
            label_transform=None,
            global_transform=discard_audio,
        )
    else:
        if Ds in {Kinetics}:
            test = Ds(
                root=data_path,
                annotation_path=annotation_path,
                frames_per_clip=frames_per_clip,
                step_between_clips=step_between_clips,
                temporal_downsampling=temporal_downsampling,
                fold=fold,
                split="test",
                video_transform=Compose(
                    [
                        ToTensorVideo(),
                        RandomShortSideScaleJitterVideo(
                            min_size=image_size, max_size=image_size
                        ),
                    ]
                ),
                global_transform=discard_audio,
                num_ensemble_views=test_ensemble_temporal_clips,
            )
        else:
            test = Ds(
                root=data_path,
                annotation_path=annotation_path,
                frames_per_clip=frames_per_clip,
                step_between_clips=step_between_clips,
                temporal_downsampling=temporal_downsampling,
                fold=fold,
                split="test",
                video_transform=Compose(
                    [
                        ToTensorVideo(),
                        RandomShortSideScaleJitterVideo(
                            min_size=image_size, max_size=image_size
                        ),
                    ]
                ),
                global_transform=discard_audio,
            )
            # if test_ensemble_temporal_clips > 1:
            #     test = TemporallySamplingVideoEnsemble(
            #         dataset=test,
            #         num_temporal_clips=test_ensemble_temporal_clips,
            #     )

        test = SpatiallySamplingVideoEnsemble(
            dataset=test,
            crop_size=image_size,
            spatial_sampling_strategy=test_ensemble_spatial_sampling_strategy,
            video_transform=NormalizeVideo(mean=MEAN, std=STD),
        )

    return train, val, test


def floor2(x):
    return x // 2 * 2
