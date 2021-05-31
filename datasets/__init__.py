import os

from .dataloader import ActionRecognitionDatasets  # noqa: F401
from .interfaces import VideoClassificationDataset  # noqa: F401
from .video_ensemble import (  # noqa: F401
    SpatiallySamplingVideoEnsemble,
    TemporallySamplingVideoEnsemble,
)
from .videoclips import videoclips  # noqa: F401

# Ensure ffmpeg is accessible
os.environ["PATH"] += os.pathsep + "/usr/local/bin"
os.environ["PATH"] += os.pathsep + "/usr/bin"
ffmpeg_path = os.popen("which ffmpeg").read()[:-1]
assert ffmpeg_path, "No ffmpeg installation found"
