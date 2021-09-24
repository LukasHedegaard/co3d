""" CoX3D main """
from ride import Main  # isort:skip
from functools import partial
from typing import Sequence

import torch
from ride import Configs, RideModule
from ride.metrics import TopKAccuracyMetric, MeanAveragePrecisionMetric
from ride.optimizers import SgdCyclicLrOptimizer
from ride.utils.logging import getLogger

from datasets import ActionRecognitionDatasets
from models.cox3d.modules.x3d import CoX3D

logger = getLogger("CoX3D")


class CoX3DRide(
    RideModule,
    ActionRecognitionDatasets,
    SgdCyclicLrOptimizer,
    TopKAccuracyMetric(1, 3, 5),
    MeanAveragePrecisionMetric,
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="x3d_num_groups",
            type=int,
            default=1,
            strategy="constant",
            description="Number of groups.",
        )
        c.add(
            name="x3d_width_per_group",
            type=int,
            default=64,
            strategy="constant",
            description="Width of each group.",
        )
        c.add(
            name="x3d_width_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Width expansion factor.",
        )
        c.add(
            name="x3d_depth_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Depth expansion factor.",
        )
        c.add(
            name="x3d_bottleneck_factor",
            type=float,
            default=1.0,
            strategy="constant",
            description="Bottleneck expansion factor for the 3x3x3 conv.",
        )
        c.add(
            name="x3d_conv1_dim",
            type=int,
            default=12,
            strategy="constant",
            description="Dimensions of the first 3x3 conv layer.",
        )
        c.add(
            name="x3d_conv5_dim",
            type=int,
            default=2048,
            strategy="constant",
            description="Dimensions of the last linear layer before classificaiton.",
        )
        c.add(
            name="x3d_use_channelwise_3x3x3",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="Whether to use channelwise (=depthwise) convolution in the center (3x3x3) convolution operation of the residual blocks.",
        )
        c.add(
            name="x3d_dropout_rate",
            type=float,
            default=0.5,
            choices=[0.0, 1.0],
            strategy="uniform",
            description="Dropout rate before final projection in the backbone.",
        )
        c.add(
            name="x3d_head_activation",
            type=str,
            default="softmax",
            choices=["softmax", "sigmoid"],
            strategy="choice",
            description="Activation layer for the output head.",
        )
        c.add(
            name="x3d_head_batchnorm",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="choice",
            description="Whether to use a BatchNorm layer before the classifier.",
        )
        c.add(
            name="x3d_fc_std_init",
            type=float,
            default=0.01,
            strategy="choice",
            description="The std to initialize the fc layer(s).",
        )
        c.add(
            name="x3d_final_batchnorm_zero_init",
            type=int,
            default=1,
            choices=[0, 1],
            strategy="choice",
            description="If true, initialize the gamma of the final BN of each block to zero.",
        )
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

        return c

    def __init__(self, hparams):
        # Inflate frames_per_clip of dataset, to have some frames for initialization
        self.temporal_window_size = self.hparams.frames_per_clip
        image_size = self.hparams.image_size
        dim_in = 3

        self.module = CoX3D(
            dim_in,
            image_size,
            self.temporal_window_size,
            self.dataloader.num_classes,  # from ActionRecognitionDatasets
            self.hparams.x3d_conv1_dim,
            self.hparams.x3d_conv5_dim,
            self.hparams.x3d_num_groups,
            self.hparams.x3d_width_per_group,
            self.hparams.x3d_width_factor,
            self.hparams.x3d_depth_factor,
            self.hparams.x3d_bottleneck_factor,
            self.hparams.x3d_use_channelwise_3x3x3,
            self.hparams.x3d_dropout_rate,
            self.hparams.x3d_head_activation,
            self.hparams.x3d_head_batchnorm,
            self.hparams.x3d_fc_std_init,
            self.hparams.x3d_final_batchnorm_zero_init,
            self.hparams.co3d_temporal_fill,
            se_scope="frame",
        )

        # Ensure that state-dict is flattened
        self.load_state_dict = partial(self.module.load_state_dict, flatten=True)
        self.state_dict = partial(self.module.state_dict, flatten=True)

        num_init_frames = max(
            self.module.receptive_field - 1, self.hparams.co3d_forward_frame_delay - 1
        )
        self.hparams.frames_per_clip = (
            num_init_frames + self.hparams.co3d_num_forward_frames
        )

        # if "clip" not in self.hparams.co3d_forward_mode:
        #     self.hparams.frames_per_clip = 0
        #     self.module.call_mode = "forward"

        # if "frame" in self.hparams.co3d_forward_mode:
        #     assert self.hparams.co3d_num_forward_frames > 0
        #     self.hparams.frames_per_clip += self.hparams.co3d_num_forward_frames
        #     self.module.call_mode = "forward_step"

        # if "init" in self.hparams.co3d_forward_mode:
        #     if self.hparams.co3d_forward_frame_delay < 0:
        #         self.hparams.co3d_forward_frame_delay = (
        #             self.temporal_window_size
        #             + 1
        #             + self.hparams.co3d_forward_frame_delay
        #         )
        #     self.hparams.frames_per_clip += self.hparams.co3d_forward_frame_delay

        # From ActionRecognitionDatasets
        frames_per_clip = (
            1
            if self.hparams.co3d_forward_mode == "frame"
            else self.hparams.frames_per_clip
        )
        self.input_shape = (dim_in, frames_per_clip, image_size, image_size)

        # Assuming Conv3d have stride = 1 and dilation = 1, and that no other modules delay the network.
        logger.info(f"Model receptive field: {self.module.receptive_field} frames")

        if self.hparams.dataset == "thumos14":
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=21)
            self.mAP_ignore_classes = [0, 21]

    def preprocess_batch(self, batch):
        """Overloads method in ride.Lifecycle"""

        x, y = batch[0], batch[1]

        # Ensure that there are enough frames
        num_init_frames = max(
            self.module.receptive_field - 1, self.hparams.co3d_forward_frame_delay - 1
        )
        num_needed_frames = num_init_frames + self.hparams.co3d_num_forward_frames
        num_missing_frames = num_needed_frames - x.shape[2]
        if num_missing_frames > 0:
            # Repeat the last frame
            append = (
                x[:, :, -1]
                .repeat((num_missing_frames, 1, 1, 1, 1))
                .permute(1, 2, 0, 3, 4)
            )
            x = torch.cat([x, append], dim=2)
            assert x.shape[2] == num_needed_frames

        if self.task == "detection":
            # Remove labels for initialisation frames
            y = y[:, -self.hparams.co3d_num_forward_frames :]

            # Collapse into batch dimension
            y = y.reshape(-1)

        batch = [x, y]
        return batch

    def clean_state_on_shape_change(self, shape):
        if not hasattr(self, "_current_input_shape"):
            self._current_input_shape = shape

        if self._current_input_shape != shape:
            self.module.clean_state()

    def forward(self, x):
        if self.training:
            self.module.clean_state()
        else:
            self.clean_state_on_shape_change(x.shape)

        result = None

        if "init" in self.hparams.co3d_forward_mode:
            self.module.warm_up(tuple(x[:, :, 0].shape))

        num_init_frames = max(
            self.module.receptive_field - 1, self.hparams.co3d_forward_frame_delay - 1
        )
        self.module.forward_steps(x[:, :, :num_init_frames])  # don't save
        result = self.module.forward_steps(x[:, :, num_init_frames:])  # don't save

        if self.task == "classification":
            result = result.mean(dim=-1)
        elif self.task == "detection":
            result = result.permute(0, 2, 1).reshape(-1, self.num_classes)
        else:
            raise ValueError(f"Unknown task {self.task}")

        return result

    def warm_up(self, input_shape: Sequence[int], *args, **kwargs):
        for m in self.modules.modules():
            if hasattr(m, "state_index"):
                m.state_index = 0


if __name__ == "__main__":
    Main(CoX3DRide).argparse()
