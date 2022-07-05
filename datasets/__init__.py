import os

from ride.utils.logging import getLogger

from .dataloader import ActionRecognitionDatasets  # noqa: F401
from .interfaces import VideoClassificationDataset  # noqa: F401
from .video_ensemble import (  # noqa: F401
    SpatiallySamplingVideoEnsemble,
    TemporallySamplingVideoEnsemble,
)
from .videoclips import videoclips  # noqa: F401

logger = getLogger("datasets")

# Ensure ffmpeg is accessible
os.environ["PATH"] += os.pathsep + "/usr/local/bin"
os.environ["PATH"] += os.pathsep + "/usr/bin"
ffmpeg_path = os.popen("which ffmpeg").read()[:-1]
if not ffmpeg_path:
    logger.error("No ffmpeg installation found")
