from operator import attrgetter

import torch
from continual import CoModule
from pytorch_lightning.utilities.parsing import AttributeDict
from ride.core import Configs, RideMixin
from ride.utils.logging import getLogger
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
            choices=["clip", "frame", "init_frame", "init_clip", "clip_init_frame"],
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
        return c

    def __init__(self, hparams: AttributeDict, *args, **kwargs):
        self.dim_in = 3

        # Determine the frames_per_clip to ask from dataloader
        self.hparams.frames_per_clip = self.hparams.temporal_window_size

        if "init" in self.hparams.co3d_forward_mode:
            num_init_frames = max(
                self.module.receptive_field - 1,
                self.hparams.co3d_forward_frame_delay - 1,
            )
            self.hparams.frames_per_clip = (
                num_init_frames
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

    def on_init_end(self, hparams: AttributeDict, *args, **kwargs):
        # Decide inference mode
        if "frame" in self.hparams.co3d_forward_mode:
            self.module.call_mode = "forward_steps"  # default = "forward"

        logger.info(f"Model receptive field: {self.module.receptive_field} frames")
        logger.info(f"Training loss: {name(self.loss)}")

        # If conducting profiling, ensure that the model has been warmed up
        # so that it doesn't output placeholder values
        if "profile_model" in self.hparams:
            logger.info("Warming model up")
            self.module(
                torch.randn(
                    (
                        self.hparams.batch_size,
                        self.dim_in,
                        self.module.receptive_field,
                        self.hparams.image_size,
                        self.hparams.image_size,
                    )
                )
            )
            for m in self.module.modules():
                if hasattr(m, "state_index"):
                    m.state_index = 0
                if hasattr(m, "stride_index"):
                    m.stride_index = 0

    def forward(self, x):
        result = None

        if "init" in self.hparams.co3d_forward_mode:
            self.warm_up(tuple(x[:, :, 0].shape))

            num_init_frames = max(
                self.module.receptive_field - 1,
                self.hparams.co3d_forward_frame_delay - 1,
            )
            self.module(x[:, :, :num_init_frames])  # = forward_steps don't save

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
