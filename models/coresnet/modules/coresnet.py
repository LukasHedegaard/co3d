from collections import OrderedDict
from typing import Sequence, Tuple

import continual as co
from torch import nn
from models.common.res import CoResStage, init_weights

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
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
}

# Size of stride on different res stages.
_SPATIAL_STRIDES = {False: [1, 2, 2, 2], True: [1, 1, 1, 2]}  # key: enable_detection

# Size of dilation on different res stages.
_SPATIAL_DILATIONS = {False: [1, 1, 1, 1], True: [1, 2, 2, 1]}  # key: enable_detection

_POOL1 = {
    "2d": [1, 1, 1],
    "c2d": [2, 1, 1],
    "c2d_nopool": [1, 1, 1],
    "i3d": [2, 1, 1],
    "i3d_nopool": [1, 1, 1],
    "slow": [1, 1, 1],
}


def CoResNetBasicStem(
    dim_in: int,
    dim_out: int,
    kernel: int,
    stride: int,
    padding: int,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    norm_module=nn.BatchNorm3d,
    temporal_fill: co.PaddingMode = "zeros",
    *args,
    **kwargs,
):
    """
    Basic 3D Resnet stem module.

    Args:
        dim_in (int): the channel dimension of the input. Normally 3 is used
            for rgb input, and 2 or 3 is used for optical flow input.
        dim_out (int): the output dimension of the convolution in the stem
            layer.
        kernel (list): the kernel size of the convolution in the stem layer.
            temporal kernel size, height kernel size, width kernel size in
            order.
        stride (list): the stride size of the convolution in the stem layer.
            temporal kernel stride, height kernel size, width kernel size in
            order.
        padding (int): the padding size of the convolution in the stem
            layer, temporal padding size, height padding size, width
            padding size in order.
        inplace_relu (bool): calculate the relu on the original input
            without allocating new memory.
        eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
            default is torch.nn.BatchNorm3d.
    """

    conv = co.Conv3d(
        dim_in,
        dim_out,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=False,
    )

    bn = norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt)

    relu = nn.ReLU(inplace_relu)

    pool_layer = nn.MaxPool3d(
        kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]
    )

    # Wrap in sequential to match weight specification
    return co.Sequential(
        OrderedDict(
            [
                ("conv", conv),
                ("norm", bn),
                ("relu", relu),
                ("pool_layer", pool_layer),
            ]
        )
    )


def CoBottleneckTransform(
    dim_in: int,
    dim_out: int,
    temp_kernel_size: int,
    stride: int,
    dim_inner: int,
    num_groups: int,
    stride_1x1=False,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    dilation=1,
    norm_module=nn.BatchNorm3d,
    temporal_fill: co.PaddingMode = "zeros",
    *args,
    **kwargs,
):
    """
    Args:
        dim_in (int): the channel dimensions of the input.
        dim_out (int): the channel dimension of the output.
        temp_kernel_size (int): the temporal kernel sizes of the middle
            convolution in the bottleneck.
        stride (int): the stride of the bottleneck.
        dim_inner (int): the inner dimension of the block.
        num_groups (int): number of groups for the convolution. num_groups=1
            is for standard ResNet like networks, and num_groups>1 is for
            ResNeXt like networks.
        stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
            apply stride to the 3x3 conv.
        inplace_relu (bool): if True, calculate the relu on the original
            input without allocating new memory.
        eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        dilation (int): size of dilation.
        norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
            default is torch.nn.BatchNorm3d.
    """
    (str1x1, str3x3) = (stride, 1) if stride_1x1 else (1, stride)

    a = co.Conv3d(
        dim_in,
        dim_inner,
        kernel_size=(temp_kernel_size, 1, 1),
        stride=(1, str1x1, str1x1),
        padding=(int(temp_kernel_size // 2), 0, 0),
        bias=False,
    )

    a_bn = co.forward_stepping(
        norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt)
    )

    a_relu = nn.ReLU(inplace=inplace_relu)

    # Tx3x3, BN, ReLU.
    b = co.Conv3d(
        dim_inner,
        dim_inner,
        kernel_size=(1, 3, 3),
        stride=(1, str3x3, str3x3),
        padding=(0, dilation, dilation),
        groups=num_groups,
        bias=False,
        dilation=(1, dilation, dilation),
        temporal_fill=temporal_fill,
    )

    b_bn = co.forward_stepping(
        norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt)
    )

    b_relu = nn.ReLU(inplace=inplace_relu)

    # 1x1x1, BN.
    c = co.Conv3d(
        dim_inner,
        dim_out,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        bias=False,
    )

    c_bn = co.forward_stepping(
        norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt)
    )

    c_bn.transform_final_bn = True

    return co.Sequential(
        OrderedDict(
            [
                ("conv_a", a),
                ("norm_a", a_bn),
                ("relu_a", a_relu),
                ("conv_b", b),
                ("norm_b", b_bn),
                ("relu_b", b_relu),
                ("conv_c", c),
                ("norm_c", c_bn),
            ]
        )
    )


def CoResNetBasicHead(
    dim_in: int,
    num_classes: int,
    pool_size: Tuple[int, int, int],
    dropout_rate=0.0,
    act_func="softmax",
    temporal_window_size: int = 4,
    temporal_fill: co.PaddingMode = "zeros",
):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1.

    Args:
        dim_in (list): the list of channel dimensions of the p inputs to the
            ResNetHead.
        num_classes (int): the channel dimensions of the p outputs to the
            ResNetHead.
        pool_size (list): the list of kernel sizes of p spatial temporal
            poolings, temporal pool kernel size, spatial pool kernel size,
            spatial pool kernel size in order.
        dropout_rate (float): dropout rate. If equal to 0.0, perform no
            dropout.
        act_func (string): activation function to use. 'softmax': applies
            softmax on the output. 'sigmoid': applies sigmoid on the output.

    """
    modules = []
    if pool_size is None:
        avg_pool = co.AdaptiveAvgPool3d(
            (1, 1, 1), kernel_size=temporal_window_size, temporal_fill=temporal_fill
        )
    else:
        avg_pool = co.AvgPool3d(pool_size, stride=1, temporal_fill=temporal_fill)

    modules.append(("avg_pool", avg_pool))

    if dropout_rate > 0.0:
        modules.append(("dropout", nn.Dropout(dropout_rate)))

    # Perform FC in a fully convolutional manner. The FC layer will be
    # initialized with a different std comparing to convolutional layers.
    modules.append(
        ("projection", co.Linear(dim_in, num_classes, bias=True, channel_dim=-4))
    )

    def not_training(module, *args):
        return not module.training

    modules.append(
        (
            "act",
            co.Conditional(
                not_training,
                {
                    "softmax": nn.Softmax(dim=1),
                    "sigmoid": nn.Sigmoid(),
                }[act_func],
            ),
        )
    )

    def view(x):
        return x.view(x.shape[0], -1)

    modules.append(("view", co.Lambda(view)))

    return co.Sequential(OrderedDict(modules))


def CoResNetRoIHead(
    dim_in: int,
    num_classes: int,
    pool_size: Sequence[int],
    resolution: Sequence[int],
    scale_factor: Sequence[int],
    dropout_rate=0.0,
    act_func="softmax",
    aligned=True,
):
    """
    ResNe(X)t RoI head.

    Args:
        dim_in (list): the list of channel dimensions of the p inputs to the
            ResNetHead.
        num_classes (int): the channel dimensions of the p outputs to the
            ResNetHead.
        pool_size (list): the list of kernel sizes of p spatial temporal
            poolings, temporal pool kernel size, spatial pool kernel size,
            spatial pool kernel size in order.
        resolution (list): the list of spatial output size from the ROIAlign.
        scale_factor (list): the list of ratio to the input boxes by this
            number.
        dropout_rate (float): dropout rate. If equal to 0.0, perform no
            dropout.
        act_func (string): activation function to use. 'softmax': applies
            softmax on the output. 'sigmoid': applies sigmoid on the output.
        aligned (bool): if False, use the legacy implementation. If True,
            align the results more perfectly.
    Note:
        Given a continuous coordinate c, its two neighboring pixel indices
        (in our pixel model) are computed by floor (c - 0.5) and ceil
        (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal at
        continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing neighboring
        pixel indices and therefore it uses pixels with a slightly incorrect
        alignment (relative to our pixel model) when performing bilinear
        interpolation.
        With `aligned=True`, we first appropriately scale the ROI and then
        shift it by -0.5 prior to calling roi_align. This produces the
        correct neighbors; It makes negligible differences to the model's
        performance if ROIAlign is used together with conv layers.
    """
    ...  # TODO: impl


def CoResNet(
    arch: str,
    dim_in: int,
    image_size: int,
    frames_per_clip: int,
    num_classes: int,
    resnet_depth: int,
    resnet_num_groups: int,
    resnet_width_per_group: int,
    resnet_dropout_rate: float,
    resnet_fc_std_init: float,
    resnet_final_batchnorm_zero_init: bool,
    resnet_head_act: str = "softmax",
    enable_detection=False,
    align_detection=False,
    temporal_fill: co.PaddingMode = "zeros",
):
    """
    Continual 3D ResNet model (CoC2D, CoI3D, CoSlow),
    adapted from https://github.com/facebookresearch/SlowFast

    The "Slow" network was originally proposed by
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """
    norm_module = nn.BatchNorm3d

    (dep2, dep3, dep4, dep5) = _MODEL_STAGE_DEPTH[resnet_depth]
    (stride2, stride3, stride4, stride5) = _SPATIAL_STRIDES[enable_detection]
    (dil2, dil3, dil4, dil5) = _SPATIAL_DILATIONS[enable_detection]
    pool_size = _POOL1[arch]

    dim_inner = resnet_num_groups * resnet_width_per_group

    temp_kernel = _TEMPORAL_KERNEL_BASIS[arch]

    s1 = CoResNetBasicStem(
        dim_in=dim_in,
        dim_out=resnet_width_per_group,
        kernel=temp_kernel[0][0] + [7, 7],
        stride=[1, 2, 2],
        padding=[temp_kernel[0][0][0] // 2, 3, 3],
        norm_module=norm_module,
    )

    s2 = CoResStage(
        dim_in=resnet_width_per_group,
        dim_out=resnet_width_per_group * 4,
        dim_inner=dim_inner,
        temp_kernel_sizes=temp_kernel[1][0],
        stride=stride2,
        num_blocks=dep2,
        num_groups=resnet_num_groups,
        num_block_temp_kernel=dep2,
        trans_func=CoBottleneckTransform,
        stride_1x1=False,
        inplace_relu=True,
        dilation=dil2,
        norm_module=norm_module,
    )

    pathway0_pool = co.MaxPool3d(
        kernel_size=pool_size,
        stride=pool_size,
        padding=[0, 0, 0],
    )

    s3 = CoResStage(
        dim_in=resnet_width_per_group * 4,
        dim_out=resnet_width_per_group * 8,
        dim_inner=dim_inner * 2,
        temp_kernel_sizes=temp_kernel[2][0],
        stride=stride3,
        num_blocks=dep3,
        num_groups=resnet_num_groups,
        num_block_temp_kernel=dep3,
        trans_func=CoBottleneckTransform,
        stride_1x1=False,
        inplace_relu=True,
        dilation=dil3,
        norm_module=norm_module,
    )

    s4 = CoResStage(
        dim_in=resnet_width_per_group * 8,
        dim_out=resnet_width_per_group * 16,
        dim_inner=dim_inner * 4,
        temp_kernel_sizes=temp_kernel[3][0],
        stride=stride4,
        num_blocks=dep4,
        num_groups=resnet_num_groups,
        num_block_temp_kernel=dep4,
        trans_func=CoBottleneckTransform,
        stride_1x1=False,
        inplace_relu=True,
        dilation=dil4,
        norm_module=norm_module,
    )

    s5 = CoResStage(
        dim_in=resnet_width_per_group * 16,
        dim_out=resnet_width_per_group * 32,
        dim_inner=dim_inner * 8,
        temp_kernel_sizes=temp_kernel[4][0],
        stride=stride5,
        num_blocks=dep5,
        num_groups=resnet_num_groups,
        num_block_temp_kernel=dep5,
        trans_func=CoBottleneckTransform,
        stride_1x1=False,
        inplace_relu=True,
        dilation=dil5,
        norm_module=norm_module,
    )

    if enable_detection:
        head = CoResNetRoIHead(
            dim_in=resnet_width_per_group * 32,
            num_classes=num_classes,
            pool_size=(frames_per_clip // pool_size[0], 1, 1),
            resolution=[7] * 2,
            scale_factor=16,
            dropout_rate=resnet_dropout_rate,
            act_func=resnet_head_act,
            aligned=align_detection,
        )
    else:
        head = CoResNetBasicHead(
            dim_in=resnet_width_per_group * 32,
            num_classes=num_classes,
            pool_size=(
                frames_per_clip // pool_size[0],
                image_size // 32 // pool_size[1],
                image_size // 32 // pool_size[2],
            ),
            dropout_rate=resnet_dropout_rate,
            act_func=resnet_head_act,
            temporal_window_size=frames_per_clip,
            temporal_fill=temporal_fill,
        )

    seq = co.Sequential(
        OrderedDict(
            [
                ("0", s1),
                ("1", s2),
                ("pathway0_pool", pathway0_pool),  # Not actually utilized
                ("2", s3),
                ("3", s4),
                ("4", s5),
                ("5", head),
            ]
        )
    )
    init_weights(seq, resnet_fc_std_init, bool(resnet_final_batchnorm_zero_init))
    return seq
