import math
import numbers
import random
from functools import partial
from typing import Callable, Iterable, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms._functional_video as V
from torchvision.transforms.functional import _is_pil_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_size(img) -> Tuple[int, int]:
    """Returns tuple(width, height)"""
    if _is_pil_image(img):
        return img.size  # type: ignore
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]  # type: ignore
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Parallel:
    """Splits input and passes to several transforms in parallel.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms: Sequence):
        self.transforms = transforms

    def __call__(self, x):
        return tuple(t(x) for t in self.transforms)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class OneHot:
    """Produces a one-hot tensor from a categorical one

    Args:
        num_classes { int }
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, x):
        return F.one_hot(F.relu(x), self.num_classes)

    def __repr__(self):
        return self.__class__.__name__ + f"(num_classes={self.num_classes})"


def discard_audio(
    sample: Tuple[torch.Tensor, torch.Tensor, Union[int, torch.Tensor]]
) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
    # Assuming sample is (video, audio, ...)
    return (sample[0], *sample[2:])


class Permute:
    def __init__(self, *axes):
        self.axes = axes

    def __call__(self, tensor: torch.Tensor):
        return tensor.permute(*self.axes)


class CropVideo(object):
    def __init__(self, crop_size: Tuple[int, int], position: str):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

        assert position in {
            "top_left",
            "top_center",
            "top_right",
            "center_left",
            "center",
            "center_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
            "random",
        }
        self.position = position
        self.sampling_fn: Callable[[int, int, int, int], Tuple[int, int]] = {
            "top_left": partial(
                self._sampling_fn, vertical=self._start, horizontal=self._start
            ),
            "top_center": partial(
                self._sampling_fn, vertical=self._start, horizontal=self._center
            ),
            "top_right": partial(
                self._sampling_fn, vertical=self._start, horizontal=self._end
            ),
            "center_left": partial(
                self._sampling_fn, vertical=self._center, horizontal=self._start
            ),
            "center": partial(
                self._sampling_fn, vertical=self._center, horizontal=self._center
            ),
            "center_right": partial(
                self._sampling_fn, vertical=self._center, horizontal=self._end
            ),
            "bottom_left": partial(
                self._sampling_fn, vertical=self._end, horizontal=self._start
            ),
            "bottom_center": partial(
                self._sampling_fn, vertical=self._end, horizontal=self._center
            ),
            "bottom_right": partial(
                self._sampling_fn, vertical=self._end, horizontal=self._end
            ),
            "random": partial(
                self._sampling_fn, vertical=self._rand, horizontal=self._rand
            ),
        }[position]

    @staticmethod
    def _start(x: int, tx: int) -> int:
        return 0

    @staticmethod
    def _center(x: int, tx: int) -> int:
        return int(round((x - tx) / 2.0))

    @staticmethod
    def _end(x: int, tx: int) -> int:
        return x - tx

    @staticmethod
    def _rand(x: int, tx: int) -> int:
        return random.randint(0, x - tx)

    @staticmethod
    def _sampling_fn(
        h: int,
        w: int,
        th: int,
        tw: int,
        vertical: Callable[[int, int], int],
        horizontal: Callable[[int, int], int],
    ) -> Tuple[int, int]:
        i = 0 if th >= h else vertical(h, th)
        j = 0 if tw >= w else horizontal(w, tw)
        return i, j

    def get_positions(self, clip: torch.Tensor) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = image_size(clip)  # type:ignore
        th, tw = self.crop_size
        if w == tw and h == th:
            return 0, 0, h, w

        i, j = self.sampling_fn(h, w, th, tw)
        return i, j, th, tw

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        i, j, h, w = self.get_positions(clip)
        return V.crop(clip, i, j, h, w)

    def __repr__(self):
        return f"{self.__class__.__name__} (crop_size={self.crop_size}, position='{self.position}')"


class ResizeVideo:
    """Resize the input clip to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``bilinear``
    """

    def __init__(
        self, size: Union[int, Sequence[int]], interpolation: str = "bilinear"
    ):
        if not (
            isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        ):
            try:
                size = int(size)
            except Exception:
                raise RuntimeError(
                    f"size must be either int or list of ints (got {type(size)})"
                )

        self.size = (size, size) if isinstance(size, int) else size
        self.interpolation = interpolation

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: Rescaled video clip.
                size is (C, T, H, W)
        """
        return V.resize(clip, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, interpolation={1})".format(
            self.size, self.interpolation
        )


class DeNormalizeVideo:
    """Produces the inverse of its `NormalizeVideo` counterparts,
    scaling to the original mean and std, and placing the channel last
    """

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)
        Returns:
            denormalized clip (torch.tensor): Size is (C, T, H, W)
        """
        assert V._is_tensor_video_clip(clip), "clip should be a 4D torch.tensor"
        if not self.inplace:
            clip = clip.clone()
        mean = torch.as_tensor(self.mean, dtype=clip.dtype, device=clip.device)
        std = torch.as_tensor(self.std, dtype=clip.dtype, device=clip.device)
        clip.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
        return clip

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, inplace={2})".format(
            self.mean, self.std, self.inplace
        )


class RandomShortSideScaleJitterVideo:
    def __init__(self, min_size: int, max_size: int, inverse_uniform_sampling=False):
        """
        Args:
            min_size (int): the minimal size to scale the frames.
            max_size (int): the maximal size to scale the frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale, max_scale].
        """
        self.min_size = min_size
        self.max_size = max_size
        self.inverse_uniform_sampling = inverse_uniform_sampling

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform a spatial short scale jittering on the given images and
        corresponding boxes.
        Args:
            images (tensor): images to perform scale jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
        Returns:
            (tensor): the scaled images with dimension of
                `num frames` x `channel` x `new height` x `new width`.
        """
        if self.inverse_uniform_sampling:
            size = int(
                round(1.0 / np.random.uniform(1.0 / self.max_size, 1.0 / self.min_size))
            )
        else:
            size = int(round(np.random.uniform(self.min_size, self.max_size)))

        height = images.shape[2]
        width = images.shape[3]
        if (width <= height and width == size) or (height <= width and height == size):
            return images
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        return torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
