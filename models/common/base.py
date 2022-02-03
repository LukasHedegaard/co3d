from operator import attrgetter
from typing import Sequence

import torch
from pytorch_lightning.utilities.parsing import AttributeDict
from ride.core import Configs, RideMixin
from ride.utils.logging import getLogger

from continual import CoModule, TensorPlaceholder
from ride.utils.utils import name

logger = getLogger("co3d")


class Co3dBase(RideMixin):
    hparams: AttributeDict
    module: CoModule

    def validate_attributes(self):
        for hparam in self.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="co3d_temporal_fill",
            type=str,
            default="zeros",
            choices=["zeros", "replicate"],
            strategy="choice",
            description="Fill mode for samples along temporal dimension. This is used at state initialisation and in `forward_steps` as padding along the temporal dimension.",
        )
        c.add(
            name="co3d_forward_mode",
            type=str,
            default="init_frame",
            choices=["clip", "frame", "init_frame"],
            strategy="choice",
            description="Whether to compute clip or frame during forward. If 'clip_init_frame', the network is initialised with a clip and then frame forwards are applied.",
        )
        c.add(
            name="co3d_num_forward_frames",
            type=int,
            default=1,
            description="The number of frames to predict over",
        )
        c.add(
            name="co3d_forward_frame_delay",
            type=int,
            default=-1,
            strategy="choice",
            description="Number of frames forwards prior to final prediction in 'clip_init_frame' mode. If '-1', a delay of clip_length - 1 is used",
        )
        c.add(
            name="co3d_forward_prediction_delay",
            type=int,
            default=0,
            strategy="choice",
            description="Number of steps to delay the prediction relative to the frames",
        )
        c.add(
            name="temporal_window_size",
            type=int,
            default=8,
            strategy="choice",
            description="Temporal window size for global average pool.",
        )
        c.add(
            name="enable_detection",
            type=int,
            default=0,
            strategy="choice",
            choices=[0, 1],
            description="Whether to enable detection head.",
        )
        c.add(
            name="align_detection",
            type=int,
            default=0,
            strategy="choice",
            choices=[0, 1],
            description="Whether to utilise alignment in detection head.",
        )
        return c

    def __init__(self, hparams: AttributeDict, *args, **kwargs):
        self.dim_in = 3
        self.hparams.frames_per_clip = self.hparams.temporal_window_size

    def on_init_end(self, hparams: AttributeDict, *args, **kwargs):
        # Determine the frames_per_clip to ask from dataloader
        self.hparams.frames_per_clip = self.hparams.temporal_window_size

        if "init" in self.hparams.co3d_forward_mode:
            num_init_frames = max(
                self.module.receptive_field - self.module.padding - 1,
                self.hparams.co3d_forward_frame_delay - 1,
            )
            self.hparams.frames_per_clip = (
                num_init_frames
                + self.hparams.co3d_num_forward_frames
                + self.hparams.co3d_forward_prediction_delay
            )
        elif "clip" in self.hparams.co3d_forward_mode:
            self.hparams.frames_per_clip = (
                (self.module.receptive_field - 2 * self.module.padding - 1)
                + self.hparams.co3d_num_forward_frames
                + self.hparams.co3d_forward_prediction_delay
            )

        # From ActionRecognitionDatasets
        if self.hparams.co3d_forward_mode == "frame":
            self.hparams.frames_per_clip = 1

        self.input_shape = (
            self.dim_in,
            self.hparams.frames_per_clip,
            self.hparams.image_size,
            self.hparams.image_size,
        )

        # Decide inference mode
        if "frame" in self.hparams.co3d_forward_mode:
            self.module.call_mode = "forward_steps"  # default = "forward"

        logger.info(f"Model receptive field: {self.module.receptive_field} frames")
        logger.info(f"Training loss: {name(self.loss)}")

        # If conducting profiling, ensure that the model has been warmed up
        # so that it doesn't output placeholder values
        if self.hparams.profile_model and self.hparams.enable_detection:
            # Pass in bounding boxes to RoIHead for AVA dataset.
            dummy_boxes = [torch.tensor([[11.5200, 25.0880, 176.0000, 250.6240]])]
            self.module[-1].set_boxes(dummy_boxes)

    def warm_up(self, data_shape: Sequence[int] = None):
        data_shape = data_shape or (self.hparams.batch_size, *self.module.input_shape)
        self.module.clean_state()
        prevously_training = self.module.training
        self.module.eval()
        with torch.no_grad():
            zeros = torch.zeros(data_shape, dtype=torch.float)
            for _ in range(self.module.receptive_field - self.module.padding - 1):
                self.module(zeros)
        if prevously_training:
            self.module.train()

    def forward(self, x):
        result = None

        if self.hparams.enable_detection and not self.hparams.profile_model:
            # Pass in bounding boxes to RoIHead for AVA dataset.
            self.module[-1].set_boxes(x["boxes"])
            x = x["images"]

        if "init" in self.hparams.co3d_forward_mode:
            self.module.clean_state()
            num_init_frames = max(
                self.module.receptive_field - self.module.padding - 1,
                self.hparams.co3d_forward_frame_delay - 1,
            )
            assert isinstance(self.module(x[:, :, :num_init_frames]), TensorPlaceholder)

            result = self.module(x[:, :, num_init_frames:])
        else:
            result = self.module(x)

        if self.task == "classification":
            result = result.mean(dim=-1)
        elif self.task == "detection":
            result = result.permute(0, 2, 1).reshape(-1, self.num_classes)
        else:
            raise ValueError(f"Unknown task {self.task}")

        return result
