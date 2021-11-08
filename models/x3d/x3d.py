import math

import torch
from ride import Configs, RideModule, TopKAccuracyMetric
from ride.optimizers import SgdOneCycleOptimizer
from torch import Tensor

from datasets import ActionRecognitionDatasets

from .head_helper import X3DHead
from .resnet_helper import ResStage
from .stem_helper import VideoModelStem
from .weight_init_helper import init_weights


class X3D(torch.nn.Module):
    """
    X3D model,
    adapted from https://github.com/facebookresearch/SlowFast

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(
        self,
        dim_in: int,
        image_size: int,
        temporal_window_size: int,
        num_classes: int,
        x3d_conv1_dim: int,
        x3d_conv5_dim: int,
        x3d_num_groups: int,
        x3d_width_per_group: int,
        x3d_width_factor: float,
        x3d_depth_factor: float,
        x3d_bottleneck_factor: float,
        x3d_use_channelwise_3x3x3: bool,
        x3d_dropout_rate: float,
        x3d_head_activation: str,
        x3d_head_batchnorm: bool,
        x3d_fc_std_init: float,
        x3d_final_batchnorm_zero_init: bool,
        headless=False,
    ):
        torch.nn.Module.__init__(self)
        self.norm_module = torch.nn.BatchNorm3d

        exp_stage = 2.0
        self.dim_conv1 = x3d_conv1_dim

        self.dim_res2 = (
            _round_width(self.dim_conv1, exp_stage, divisor=8)
            if False  # hparams.X3D.SCALE_RES2
            else self.dim_conv1
        )
        self.dim_res3 = _round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = _round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = _round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]

        num_groups = x3d_num_groups
        width_per_group = x3d_width_per_group
        dim_inner = num_groups * width_per_group

        w_mul = x3d_width_factor
        d_mul = x3d_depth_factor
        dim_res1 = _round_width(self.dim_conv1, w_mul)

        # Basis of temporal kernel sizes for each of the stage.
        temp_kernel = [
            [[5]],  # conv1 temporal kernels.
            [[3]],  # res2 temporal kernels.
            [[3]],  # res3 temporal kernels.
            [[3]],  # res4 temporal kernels.
            [[3]],  # res5 temporal kernels.
        ]

        self.s1 = VideoModelStem(
            dim_in=[dim_in],
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        dim_out = dim_in
        for stage, block in enumerate(self.block_basis):
            dim_out = _round_width(block[1], w_mul)
            dim_inner = int(x3d_bottleneck_factor * dim_out)

            n_rep = _round_repeats(block[0], d_mul)
            prefix = "s{}".format(stage + 2)  # start w res2 to follow convention

            s = ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner] if x3d_use_channelwise_3x3x3 else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=[[]],
                nonlocal_group=[1],
                nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
                instantiation="dot_product",
                trans_func_name="x3d_transform",
                stride_1x1=False,
                norm_module=self.norm_module,
                dilation=[1],
                drop_connect_rate=0.0,
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if not headless:
            spat_sz = int(math.ceil(image_size / 32.0))
            self.head = X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=x3d_conv5_dim,
                num_classes=num_classes,
                pool_size=(temporal_window_size, spat_sz, spat_sz),
                dropout_rate=x3d_dropout_rate,
                act_func=x3d_head_activation,
                bn_lin5_on=bool(x3d_head_batchnorm),
            )
        init_weights(self, x3d_fc_std_init, bool(x3d_final_batchnorm_zero_init))

    def forward(self, x: Tensor):
        # The original slowfast code was set up to use multiple paths, wrap the input
        x = [x]  # type:ignore
        for module in self.children():
            x = module(x)
        return x


class X3DRide(
    RideModule,
    X3D,
    ActionRecognitionDatasets,
    SgdOneCycleOptimizer,
    TopKAccuracyMetric(1, 3, 5),
):
    @staticmethod
    def configs() -> Configs:
        c = Configs()  # type: ignore
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
            name="forward_frame_delay",
            type=int,
            default=0,
            strategy="choice",
            description="Number of frames to skip before feeding clip to network.",
        )
        c.add(
            name="temporal_window_size",
            type=int,
            default=8,
            strategy="choice",
            description="Temporal window size for global average pool.",
        )
        return c

    def __init__(self, hparams):
        # Ask for more frames in ActionRecognitionDatasets
        assert self.hparams.forward_frame_delay >= 0
        self.hparams.frames_per_clip = self.hparams.temporal_window_size + self.hparams.forward_frame_delay

        image_size = self.hparams.image_size
        dim_in = 3
        self.input_shape = (dim_in, self.hparams.temporal_window_size, image_size, image_size)

        X3D.__init__(
            self,
            dim_in,
            image_size,
            self.hparams.temporal_window_size,
            self.dataloader.num_classes,  # from ActionRecognitionDatasets
            hparams.x3d_conv1_dim,
            hparams.x3d_conv5_dim,
            hparams.x3d_num_groups,
            hparams.x3d_width_per_group,
            hparams.x3d_width_factor,
            hparams.x3d_depth_factor,
            hparams.x3d_bottleneck_factor,
            hparams.x3d_use_channelwise_3x3x3,
            hparams.x3d_dropout_rate,
            hparams.x3d_head_activation,
            hparams.x3d_head_batchnorm,
            hparams.x3d_fc_std_init,
            hparams.x3d_final_batchnorm_zero_init,
        )

    def forward(self, x: Tensor):
        x = X3D.forward(self, x[:, :, self.hparams.forward_frame_delay :])
        return x


def _round_width(width, multiplier, min_depth=8, divisor=8):
    """Round width of filters based on width multiplier."""
    if not multiplier:
        return width

    width *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(width + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * width:
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats, multiplier):
    """Round number of layers based on depth multiplier."""
    multiplier = multiplier
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))
