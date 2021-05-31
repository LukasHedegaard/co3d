#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified from https://github.com/facebookresearch/SlowFast

"""Video models."""

import torch
import torch.nn as nn

from . import head_helper, resnet_helper, stem_helper
from .batchnorm_helper import get_norm
from .weight_init_helper import init_weights

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}

NONLOCAL_POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        slowfast_alpha: float,
        slowfast_beta_inv: float,
        slowfast_fusion_conv_channel_ratio: float,
        slowfast_fusion_kernel_size: int,
        resnet_depth: int,
        image_size: int,
        frames_per_clip: int,
        num_classes: int,
        dropout_rate: float,
        head_activation: str,
        num_groups: int = 1,
        width_per_group: int = 64,
        fc_std_init: float = 0.01,
        dim_in: int = 3,
        final_batchnorm_zero_init: bool = True,
        detection_enable: bool = False,
        norm_type: str = "batchnorm",
        norm_num_splits: int = 1,
        multigrid_short_cycle: bool = False,
        use_nonlocal: bool = False,
        detection_aligned=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        nn.Module.__init__(self)
        self.norm_module = get_norm(norm_type, norm_num_splits)
        self.enable_detection = detection_enable
        self.num_pathways = 2

        model_arch = "slowfast"
        pool_size = _POOL1[model_arch]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert resnet_depth in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[resnet_depth]

        dim_inner = num_groups * width_per_group
        out_dim_ratio = slowfast_beta_inv // slowfast_fusion_conv_channel_ratio

        temp_kernel = _TEMPORAL_KERNEL_BASIS[model_arch]

        NUM_BLOCK_TEMP_KERNEL = [[3, 3], [4, 4], [6, 6], [3, 3]]
        SPATIAL_STRIDES = [[1, 1], [2, 2], [2, 2], [2, 2]]
        SPATIAL_DILATIONS = [[1, 1], [1, 1], [1, 1], [1, 1]]

        NONLOCAL_GROUP = [[1, 1], [1, 1], [1, 1], [1, 1]]
        NONLOCAL_LOCATION = (
            [[[], []], [[1, 3], []], [[1, 3, 5], []], [[], []]]
            if use_nonlocal
            else [[[], []], [[], []], [[], []], [[], []]]
        )
        NONLOCAL_INSTANTIATION = "dot_product"

        TRANS_FUNC = "bottleneck_transform"

        DETECTION_ROI_XFORM_RESOLUTION = 7
        DETECTION_SPATIAL_SCALE_FACTOR = 16

        self.s1 = stem_helper.VideoModelStem(
            dim_in=dim_in,
            dim_out=[width_per_group, width_per_group // slowfast_beta_inv],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // slowfast_beta_inv,
            slowfast_fusion_conv_channel_ratio,
            slowfast_fusion_kernel_size,
            slowfast_alpha,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // slowfast_beta_inv,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // slowfast_beta_inv,
            ],
            dim_inner=[dim_inner, dim_inner // slowfast_beta_inv],
            temp_kernel_sizes=temp_kernel[1],
            stride=SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=NONLOCAL_LOCATION[0],
            nonlocal_group=NONLOCAL_GROUP[0],
            nonlocal_pool=NONLOCAL_POOL[0],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name="bottleneck_transform",
            dilation=SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // slowfast_beta_inv,
            slowfast_fusion_conv_channel_ratio,
            slowfast_fusion_kernel_size,
            slowfast_alpha,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // slowfast_beta_inv,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // slowfast_beta_inv,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // slowfast_beta_inv],
            temp_kernel_sizes=temp_kernel[2],
            stride=SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=NONLOCAL_LOCATION[1],
            nonlocal_group=NONLOCAL_GROUP[1],
            nonlocal_pool=NONLOCAL_POOL[1],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name=TRANS_FUNC,
            dilation=SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // slowfast_beta_inv,
            slowfast_fusion_conv_channel_ratio,
            slowfast_fusion_kernel_size,
            slowfast_alpha,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // slowfast_beta_inv,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // slowfast_beta_inv,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // slowfast_beta_inv],
            temp_kernel_sizes=temp_kernel[3],
            stride=SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=NONLOCAL_LOCATION[2],
            nonlocal_group=NONLOCAL_GROUP[2],
            nonlocal_pool=NONLOCAL_POOL[2],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name=TRANS_FUNC,
            dilation=SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // slowfast_beta_inv,
            slowfast_fusion_conv_channel_ratio,
            slowfast_fusion_kernel_size,
            slowfast_alpha,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // slowfast_beta_inv,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // slowfast_beta_inv,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // slowfast_beta_inv],
            temp_kernel_sizes=temp_kernel[4],
            stride=SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=NONLOCAL_LOCATION[3],
            nonlocal_group=NONLOCAL_GROUP[3],
            nonlocal_pool=NONLOCAL_POOL[3],
            instantiation=NONLOCAL_INSTANTIATION,
            trans_func_name=TRANS_FUNC,
            dilation=SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if detection_enable:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // slowfast_beta_inv,
                ],
                num_classes=num_classes,
                pool_size=[
                    [
                        frames_per_clip // slowfast_alpha // pool_size[0][0],
                        1,
                        1,
                    ],
                    [frames_per_clip // pool_size[1][0], 1, 1],
                ],
                resolution=[[DETECTION_ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[DETECTION_SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=dropout_rate,
                act_func=head_activation,
                aligned=detection_aligned,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // slowfast_beta_inv,
                ],
                num_classes=num_classes,
                pool_size=[None, None]
                if multigrid_short_cycle
                else [
                    [
                        frames_per_clip // slowfast_alpha // pool_size[0][0],
                        image_size // 32 // pool_size[0][1],
                        image_size // 32 // pool_size[0][2],
                    ],
                    [
                        frames_per_clip // pool_size[1][0],
                        image_size // 32 // pool_size[1][1],
                        image_size // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=dropout_rate,
                act_func=head_activation,
            )

        init_weights(self, fc_std_init, bool(final_batchnorm_zero_init))

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(
        self,
        model_arch: str,
        resnet_depth: int,
        image_size: int,
        frames_per_clip: int,
        num_classes: int,
        dropout_rate: float,
        head_activation: str,
        width_per_group: int = 64,
        num_groups: int = 1,
        fc_std_init: float = 0.01,
        dim_in: int = 3,
        final_batchnorm_zero_init: bool = True,
        detection_enable: bool = False,
        norm_type: str = "batchnorm",
        norm_num_splits: int = 1,
        multigrid_short_cycle: bool = False,
        use_nonlocal: bool = False,
        detection_aligned=False,
        num_block_temp_kernel=[[3], [4], [6], [3]],  # Correct for ResNet50-based models
        nonlocal_location=[[[]], [[]], [[]], [[]]],
        nonlocal_group=[[1], [1], [1], [1]],
        nonlocal_instantiation="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        torch.nn.Module.__init__(self)
        self.norm_module = get_norm(norm_type, norm_num_splits)
        self.enable_detection = detection_enable
        self.num_pathways = 1

        assert model_arch in _POOL1.keys()
        pool_size = _POOL1[model_arch]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert resnet_depth in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[resnet_depth]

        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[model_arch]

        RESNET_SPATIAL_STRIDES = [[1], [2], [2], [2]]
        RESNET_TRANS_FUNC = "bottleneck_transform"
        RESNET_STRIDE_1X1 = False
        RESNET_INPLACE_RELU = True
        RESNET_SPATIAL_DILATIONS = [[1], [1], [1], [1]]

        DETECTION_ROI_XFORM_RESOLUTION = 7
        DETECTION_SPATIAL_SCALE_FACTOR = 16

        self.s1 = stem_helper.VideoModelStem(
            dim_in=dim_in,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=RESNET_SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[0],
            nonlocal_inds=nonlocal_location[0],
            nonlocal_group=nonlocal_group[0],
            nonlocal_pool=NONLOCAL_POOL[0],
            instantiation=nonlocal_instantiation,
            trans_func_name=RESNET_TRANS_FUNC,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
            dilation=RESNET_SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=RESNET_SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[1],
            nonlocal_inds=nonlocal_location[1],
            nonlocal_group=nonlocal_group[1],
            nonlocal_pool=NONLOCAL_POOL[1],
            instantiation=nonlocal_instantiation,
            trans_func_name=RESNET_TRANS_FUNC,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
            dilation=RESNET_SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=RESNET_SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[2],
            nonlocal_inds=nonlocal_location[2],
            nonlocal_group=nonlocal_group[2],
            nonlocal_pool=NONLOCAL_POOL[2],
            instantiation=nonlocal_instantiation,
            trans_func_name=RESNET_TRANS_FUNC,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
            dilation=RESNET_SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=RESNET_SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=num_block_temp_kernel[3],
            nonlocal_inds=nonlocal_location[3],
            nonlocal_group=nonlocal_group[3],
            nonlocal_pool=NONLOCAL_POOL[3],
            instantiation=nonlocal_instantiation,
            trans_func_name=RESNET_TRANS_FUNC,
            stride_1x1=RESNET_STRIDE_1X1,
            inplace_relu=RESNET_INPLACE_RELU,
            dilation=RESNET_SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=num_classes,
                pool_size=[[frames_per_clip // pool_size[0][0], 1, 1]],
                resolution=[[DETECTION_ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[DETECTION_SPATIAL_SCALE_FACTOR],
                dropout_rate=dropout_rate,
                act_func=head_activation,
                aligned=detection_aligned,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=num_classes,
                pool_size=[None, None]
                if multigrid_short_cycle
                else [
                    [
                        frames_per_clip // pool_size[0][0],
                        image_size // 32 // pool_size[0][1],
                        image_size // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=dropout_rate,
                act_func=head_activation,
            )

        init_weights(self, fc_std_init, bool(final_batchnorm_zero_init))

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x
