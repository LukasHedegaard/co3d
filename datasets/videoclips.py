from typing import Sequence

from joblib import Memory
from ride.utils.env import CACHE_PATH, NUM_CPU
from torchvision.datasets.video_utils import VideoClips

cache = Memory(CACHE_PATH, verbose=1).cache


@cache
def videoclips(
    video_paths,
    clip_length_in_frames=16,
    frames_between_clips=1,
    temporal_downsampling=None,
    _precomputed_metadata=None,
    num_workers=0,
    _video_width=0,
    _video_height=0,
    _video_min_dimension=0,
    _video_max_dimension=0,
    _audio_samples=0,
    _audio_channels=0,
):
    return VideoClips(
        video_paths=video_paths,
        clip_length_in_frames=clip_length_in_frames,
        frames_between_clips=frames_between_clips,
        temporal_downsampling=temporal_downsampling,
        _precomputed_metadata=_precomputed_metadata,
        num_workers=num_workers or max(1, NUM_CPU - 4),
        _video_width=_video_width,
        _video_height=_video_height,
        _video_min_dimension=_video_min_dimension,
        _video_max_dimension=_video_max_dimension,
        _audio_samples=_audio_samples,
        _audio_channels=_audio_channels,
    )


def get_num_videos(vc: VideoClips):
    return len(vc.clips)


def get_video_clip_idx(vc: VideoClips, video_idx: int, clip_idx: int) -> int:
    # The get_clip_location uses a 1-indexed video_idx; We'll use a 0-indexed version
    return vc.cumulative_sizes[video_idx] + clip_idx


def get_inds_for_video(vc: VideoClips, video_idx: int) -> Sequence[int]:
    num_clips = len(vc.clips[video_idx])
    cum_clips = vc.cumulative_sizes[video_idx]
    return list(range(cum_clips - num_clips, cum_clips))
