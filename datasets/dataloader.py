from argparse import ArgumentParser, Namespace
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Union

from ride import Configs, RideClassificationDataset
from ride.utils.env import DATASETS_PATH, NUM_CPU
from ride.utils.logging import getLogger
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
    ToTensorVideo,
)

from datasets.kinetics import Kinetics
from datasets.thumos14 import Thumos14
from datasets.transforms import RandomShortSideScaleJitterVideo, discard_audio
from datasets.tvseries import TvSeries
from datasets.video_ensemble import (
    SpatiallySamplingVideoEnsemble,
    TemporallySamplingVideoEnsemble,
)

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
            choices=["kinetics400", "kinetics600", "kinetics3", "thumos14", "tvseries"],
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
            name="num_workers",
            type=int,
            default=min(10, NUM_CPU),
            strategy="constant",
            description="Number of CPU workers to use for dataloading.",
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
            default=1,
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
        return c

    def __init__(self, hparams):
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
            "thumos14": "detection",
            "tvseries": "detection",
        }[self.hparams.dataset]

        return self.dataloader

    def train_dataloader(self):
        self.prepare_data()
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
        )

        self.classes: List[str] = getattr(test_ds, "classes") or []
        self.num_classes = len(self.classes)

        # Pass to loaders
        self.train_dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=num_workers > 1,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        self.val_dataloader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True,
        )
        self.test_dataloader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=num_workers > 1,
            drop_last=False,
            prefetch_factor=4,
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
        1 / 0.7 * 0.8,
        1 / 0.7,
    ),  # corresponds to 0.875 (crop) : 1 (min scale) : 1.25 (max scale)
    normalisation_style="imagenet",
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
    else:
        raise ValueError(
            "'root_path' must contain either 'kinetics', 'thumos14', or 'tvseries'"
        )

    scaled_pix_min = floor2(image_size * image_train_scale[0])
    scaled_pix_max = floor2(image_size * image_train_scale[1])
    assert scaled_pix_max > scaled_pix_min and scaled_pix_min > image_size

    train_transforms = Compose(
        [
            ToTensorVideo(),
            RandomShortSideScaleJitterVideo(
                min_size=scaled_pix_min, max_size=scaled_pix_max
            ),
            RandomCropVideo(image_size),
            RandomHorizontalFlipVideo(),
            NormalizeVideo(mean=MEAN, std=STD),
        ]
    )

    test_transforms = Compose(
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
        video_transform=test_transforms,
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
            video_transform=test_transforms,
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
            test = TemporallySamplingVideoEnsemble(
                dataset=test,
                num_temporal_clips=test_ensemble_temporal_clips,
            )

        test = SpatiallySamplingVideoEnsemble(
            dataset=test,
            crop_size=image_size,
            spatial_sampling_strategy=test_ensemble_spatial_sampling_strategy,
            video_transform=NormalizeVideo(mean=MEAN, std=STD),
        )

    return train, val, test


def floor2(x):
    return x // 2 * 2
