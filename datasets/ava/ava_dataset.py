# Modified from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/ava_dataset.py

from functools import wraps

import numpy as np
import torch
from ride.utils.logging import getLogger

from . import ava_helper as ava_helper
from . import cv2_transform as cv2_transform
from .. import transform as transform
from .. import utils as utils

logger = getLogger(__name__)


class Ava(torch.utils.data.Dataset):
    """
    AVA Dataset
    """

    def __init__(
        self,
        root: str,
        annotation_path: str,
        frames_per_clip: int,
        step_between_clips=1,
        temporal_downsampling=None,
        split="train",
        label_transform=None,
        *args,
        **kwargs,
    ):
        self._split = split
        self._sample_rate = temporal_downsampling
        self._video_length = frames_per_clip
        self._seq_len = self._video_length * self._sample_rate
        self.classes = CLASSES
        self._num_classes = len(self.classes)
        # Augmentation params.
        self._data_mean = [0.45, 0.45, 0.45]
        self._data_std = [0.225, 0.225, 0.225]
        self._use_bgr = False
        self.random_horizontal_flip = True
        if self._split == "train":
            self._crop_size = 224
            self._jitter_min_scale = 256
            self._jitter_max_scale = 320
            self._use_color_augmentation = False
            self._pca_jitter_only = True
            self._pca_eigval = [0.225, 0.224, 0.229]
            self._pca_eigvec = [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        else:
            self._crop_size = 256
            self._test_force_flip = False

        # Loading frame paths.
        logger.info(f"Loading AVA {self._split}")
        (self._image_paths, self._video_idx_to_name,) = ava_helper.load_image_lists(
            frame_list_dir=annotation_path + "/frame_lists",
            train_lists=["train.csv"],
            test_lists=["val.csv"],
            frame_dir=root + "/frames",
            is_train=(self._split == "train"),
        )

        # Loading annotations for boxes and labels.
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            train_gt_box_lists=["ava_train_v2.2.csv"],
            train_predict_box_lists=[
                "ava_train_v2.2.csv",
                "person_box_67091280_iou90/ava_detection_train_boxes_and_labels_include_negative_v2.2.csv",
            ],
            test_predict_box_list=[
                "person_box_67091280_iou90/ava_detection_val_boxes_and_labels.csv"
            ],
            annotation_dir=annotation_path,
            detection_score_thresh=0.9,  # TODO: Expose as hparam
            full_test_on_eval=False,
            mode=self._split,
        )

        assert len(boxes_and_labels) == len(self._image_paths)

        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used, self._max_boxes_per_clip = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self.print_summary()

    def print_summary(self):
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info(
            f"Loaded AVA {self._split} with "
            f"{len(self._image_paths)} videos, "
            f"{total_frames} frames, "
            f"{len(self)} key frames, "
            f"{self._num_boxes_used} total boxes, "
            f"and max {self._max_boxes_per_clip} boxes per clip"
        )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._keyframe_indices)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError("Unsupported split mode {}".format(self._split))

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(imgs, self._crop_size, boxes=boxes)

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError("{} split not supported yet!".format(self._split))

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(boxes, self._crop_size, self._crop_size)

        return imgs, boxes

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )

        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
        boxes = np.array(boxes)
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        img_proc_backend = "cv2"  # or "cv2"
        imgs = utils.retry_load_images(image_paths, backend=img_proc_backend)
        if img_proc_backend == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(imgs, boxes=boxes)
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(imgs, boxes=boxes)

        # Construct label arrays.
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
                label_arrs[i][label - 1] = 1

        # if reverse_input_channel:
        #     imgs = imgs[[2, 1, 0], :, :, :]

        if True:  # batch_size > 1
            # Pad boxes and labels with dummies to have the same shape
            # There are maximum 25 boxes per clip
            M = 25
            missing = M - len(boxes)

            dummy_boxes = _DUMMY_VAL * np.ones_like(boxes[0])[None, :].repeat(
                missing, axis=0
            )
            dummy_label_arrs = _DUMMY_VAL * np.ones_like(label_arrs[0])[None, :].repeat(
                missing, axis=0
            )

            boxes = np.concatenate([boxes, dummy_boxes])
            ori_boxes = np.concatenate([ori_boxes, dummy_boxes])
            label_arrs = np.concatenate([label_arrs, dummy_label_arrs])

        metadata = np.array([[video_idx, sec]] * len(boxes))

        return (
            {
                "images": imgs,
                "boxes": boxes,  # (batch_id, x1, y1, x2, y2)
            },
            {
                "labels": label_arrs,
                "ori_boxes": ori_boxes,
                "metadata": metadata,
            },
            idx,
        )


_DUMMY_VAL = -1


def preprocess_ava_batch(batch):
    # Discard padded dummy values.
    images = batch[0]["images"]
    boxes = batch[0]["boxes"]
    labels = batch[1]["labels"]
    ori_boxes = batch[1]["ori_boxes"]
    metadata = batch[1]["metadata"]
    idx = batch[2]

    # Split into to lists
    sel = [x[:, 0] != _DUMMY_VAL for x in list(boxes)]
    boxes = [x[sel[i]].float() for i, x in enumerate(boxes)]
    labels = torch.cat([x[sel[i]].float() for i, x in enumerate(labels)])
    ori_boxes = torch.cat([x[sel[i]].float() for i, x in enumerate(ori_boxes)])
    metadata = torch.cat([x[sel[i]].float() for i, x in enumerate(metadata)])

    # inds =
    return (
        {
            "images": images,
            "boxes": boxes,
        },
        {
            "labels": labels,
            "ori_boxes": ori_boxes,
            "metadata": metadata,
        },
        idx,
    )


def ava_loss(loss_fn):
    @wraps(loss_fn)
    def wrapped(preds, outs):
        # outs =  { "labels": label_arrs, "ori_boxes": ori_boxes, "metadata": metadata}
        return loss_fn(preds, outs["labels"])

    return wrapped


CLASSES = [
    "bend/bow (at the waist)",
    "crawl",
    "crouch/kneel",
    "dance",
    "fall down",
    "get up",
    "jump/leap",
    "lie/sleep",
    "martial art",
    "run/jog",
    "sit",
    "stand",
    "swim",
    "walk",
    "answer phone",
    "brush teeth",
    "carry/hold (an object)",
    "catch (an object)",
    "chop",
    "climb (e.g., a mountain)",
    "clink glass",
    "close (e.g., a door, a box)",
    "cook",
    "cut",
    "dig",
    "dress/put on clothing",
    "drink",
    "drive (e.g., a car, a truck)",
    "eat",
    "enter",
    "exit",
    "extract",
    "fishing",
    "hit (an object)",
    "kick (an object)",
    "lift/pick up",
    "listen (e.g., to music)",
    "open (e.g., a window, a car door)",
    "paint",
    "play board game",
    "play musical instrument",
    "play with pets",
    "point to (an object)",
    "press",
    "pull (an object)",
    "push (an object)",
    "put down",
    "read",
    "ride (e.g., a bike, a car, a horse)",
    "row boat",
    "sail boat",
    "shoot",
    "shovel",
    "smoke",
    "stir",
    "take a photo",
    "text on/look at a cellphone",
    "throw",
    "touch (an object)",
    "turn (e.g., a screwdriver)",
    "watch (e.g., TV)",
    "work on a computer",
    "write",
    "fight/hit (a person)",
    "give/serve (an object) to (a person)",
    "grab (a person)",
    "hand clap",
    "hand shake",
    "hand wave",
    "hug (a person)",
    "kick (a person)",
    "kiss (a person)",
    "lift (a person)",
    "listen to (a person)",
    "play with kids",
    "push (another person)",
    "sing to (e.g., self, a person, a group)",
    "take (an object) from (a person)",
    "talk to (e.g., self, a person, a group)",
    "watch (a person)",
]
