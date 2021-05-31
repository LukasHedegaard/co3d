from abc import abstractmethod

from torchvision.datasets.video_utils import VideoClips


class VideoClassificationDataset:
    """Abstract base class for VideoClassificationDataset
    """

    video_clips: VideoClips

    @abstractmethod
    def __getitem__(self, idx):
        ...
