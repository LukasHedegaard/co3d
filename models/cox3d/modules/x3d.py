import math
from collections import OrderedDict

import continual as co
import torch
from continual import PaddingMode
from torch import nn

from .activation import Swish
from .se import CoSe


def CoX3DTransform(
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
    norm_module=torch.nn.BatchNorm3d,
    se_ratio=0.0625,
    swish_inner=True,
    block_idx=0,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",  # "frame" or "clip"
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
        se_ratio (float): if > 0, apply SE to the Tx3x3 conv, with the SE
            channel dimensionality being se_ratio times the Tx3x3 conv dim.
        swish_inner (bool): if True, apply swish to the Tx3x3 conv, otherwise
            apply ReLU to the Tx3x3 conv.
    """
    (str1x1, str3x3) = (stride, 1) if stride_1x1 else (1, stride)

    a = co.Conv3d(
        dim_in,
        dim_inner,
        kernel_size=(1, 1, 1),
        stride=(1, str1x1, str1x1),
        padding=(0, 0, 0),
        bias=False,
    )

    a_bn = co.forward_stepping(
        norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt)
    )

    a_relu = torch.nn.ReLU(inplace=inplace_relu)

    # Tx3x3, BN, ReLU.
    b = co.Conv3d(
        dim_inner,
        dim_inner,
        kernel_size=(temp_kernel_size, 3, 3),
        stride=(1, str3x3, str3x3),
        padding=(int(temp_kernel_size // 2), dilation, dilation),
        groups=num_groups,
        bias=False,
        dilation=(1, dilation, dilation),
        temporal_fill=temporal_fill,
    )

    b_bn = co.forward_stepping(
        norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt)
    )

    # Apply SE attention or not
    use_se = True if (block_idx + 1) % 2 else False
    if se_ratio > 0.0 and use_se:
        se = CoSe(
            temporal_window_size,
            dim_in=dim_inner,
            ratio=se_ratio,
            temporal_fill=temporal_fill,
            scope=se_scope,
        )

    b_relu = co.forward_stepping(
        Swish()  # nn.SELU is the same as Swish
        if swish_inner
        else nn.ReLU(inplace=inplace_relu)
    )

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
                ("a", a),
                ("a_bn", a_bn),
                ("a_relu", a_relu),
                ("b", b),
                ("b_bn", b_bn),
                *([("se", se)] if use_se else []),
                ("b_relu", b_relu),
                ("c", c),
                ("c_bn", c_bn),
            ]
        )
    )


def CoResBlock(
    dim_in,
    dim_out,
    temp_kernel_size,
    stride,
    trans_func,
    dim_inner,
    num_groups=1,
    stride_1x1=False,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    dilation=1,
    norm_module=torch.nn.BatchNorm3d,
    block_idx=0,
    drop_connect_rate=0.0,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",  # "clip" or "frame"
):
    """
    ResBlock class constructs redisual blocks. More details can be found in:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
        "Deep residual learning for image recognition."
        https://arxiv.org/abs/1512.03385
    Args:
        dim_in (int): the channel dimensions of the input.
        dim_out (int): the channel dimension of the output.
        temp_kernel_size (int): the temporal kernel sizes of the middle
            convolution in the bottleneck.
        stride (int): the stride of the bottleneck.
        trans_func (string): transform function to be used to construct the
            bottleneck.
        dim_inner (int): the inner dimension of the block.
        num_groups (int): number of groups for the convolution. num_groups=1
            is for standard ResNet like networks, and num_groups>1 is for
            ResNeXt like networks.
        stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
            apply stride to the 3x3 conv.
        inplace_relu (bool): calculate the relu on the original input
            without allocating new memory.
        eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        dilation (int): size of dilation.
        norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
            default is torch.nn.BatchNorm3d.
        drop_connect_rate (float): basic rate at which blocks are dropped,
            linearly increases from input to output blocks.
    """
    branch2 = trans_func(
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=stride_1x1,
        inplace_relu=inplace_relu,
        dilation=dilation,
        norm_module=norm_module,
        block_idx=block_idx,
        temporal_window_size=temporal_window_size,
        temporal_fill=temporal_fill,
        se_scope=se_scope,
    )

    def _is_training(module: nn.Module) -> bool:
        return module.training

    def _drop_connect(x, drop_ratio):
        """Apply dropconnect to x"""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x

    if drop_connect_rate > 0:
        drop = [("drop", co.Conditional(_is_training, co.Lambda(_drop_connect)))]
    else:
        drop = []

    main_stream = co.Sequential(
        OrderedDict(
            [
                ("branch2", branch2),
                *drop,
            ]
        )
    )

    if (dim_in == dim_out) and (stride == 1):
        residual_stream = co.Delay(main_stream.delay)
    else:
        residual_stream = co.Sequential(
            OrderedDict(
                [
                    (
                        "branch1",
                        co.Conv3d(
                            dim_in,
                            dim_out,
                            kernel_size=1,
                            stride=(1, stride, stride),
                            padding=0,
                            bias=False,
                            dilation=1,
                        ),
                    ),
                    (
                        "branch1_bn",
                        norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt),
                    ),
                ]
            )
        )

    return co.Sequential(
        co.BroadcastReduce(residual_stream, main_stream, reduce="sum"),
        nn.ReLU(),
    )


def CoResStage(
    dim_in: int,
    dim_out: int,
    stride: int,
    temp_kernel_sizes: int,
    num_blocks: int,
    dim_inner: int,
    num_groups: int,
    num_block_temp_kernel: int,
    dilation: int,
    trans_func_name="x3d_transform",
    stride_1x1=False,
    inplace_relu=True,
    norm_module=torch.nn.BatchNorm3d,
    drop_connect_rate=0.0,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",
    *args,
    **kwargs,
):
    """
    Create a Continual Residual X3D Stage.

    Note: Compared to the original implementation of X3D, we discard the
    obsolete handling of the multiple pathways and the non-local mehcanism.

    Args:
        dim_in (int): channel dimensions of the input.
        dim_out (int): channel dimensions of the output.
        temp_kernel_sizes (int): temporal kernel sizes of the
            convolution in the bottleneck.
        stride (int): stride of the bottleneck.
        num_blocks (int): numbers of blocks.
        dim_inner (int): inner channel dimensions of the input.
        num_groups (int): number of roups for the convolution.
            num_groups=1 is for standard ResNet like networks, and
            num_groups>1 is for ResNeXt like networks.
        num_block_temp_kernel (int): extent the temp_kernel_sizes to
            num_block_temp_kernel blocks, then fill temporal kernel size
            of 1 for the rest of the layers.
        dilation (int): size of dilation.
        trans_func_name (string): name of the the transformation function apply
            on the network.
        norm_module (nn.Module): nn.Module for the normalization layer. The
            default is nn.BatchNorm3d.
        drop_connect_rate (float): basic rate at which blocks are dropped,
            linearly increases from input to output blocks.
    """

    assert trans_func_name == "x3d_transform"
    assert num_block_temp_kernel <= num_blocks

    temp_kernel_sizes = (temp_kernel_sizes * num_blocks)[:num_block_temp_kernel] + (
        [1] * (num_blocks - num_block_temp_kernel)
    )

    return co.Sequential(
        OrderedDict(
            [
                (
                    f"pathway0_res{i}",
                    CoResBlock(
                        dim_in=dim_in if i == 0 else dim_out,
                        dim_out=dim_out,
                        temp_kernel_size=temp_kernel_sizes[i],
                        stride=stride if i == 0 else 1,
                        trans_func=CoX3DTransform,
                        dim_inner=dim_inner,
                        num_groups=num_groups,
                        stride_1x1=stride_1x1,
                        inplace_relu=inplace_relu,
                        dilation=dilation,
                        norm_module=norm_module,
                        block_idx=i,
                        drop_connect_rate=drop_connect_rate,
                        temporal_window_size=temporal_window_size,
                        temporal_fill=temporal_fill,
                        se_scope=se_scope,
                    ),
                )
                for i in range(num_blocks)
            ]
        )
    )


def CoX3DHead(
    dim_in: int,
    dim_inner: int,
    dim_out: int,
    num_classes: int,
    pool_size: int,
    dropout_rate=0.0,
    act_func="softmax",
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    norm_module=torch.nn.BatchNorm3d,
    bn_lin5_on=False,
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
):
    """
    Continual X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """
    modules = []
    modules.append(
        (
            "conv_5",
            co.Conv3d(
                dim_in,
                dim_inner,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),
        )
    )
    modules.append(
        ("conv_5_bn", norm_module(num_features=dim_inner, eps=eps, momentum=bn_mmt))
    )
    modules.append(("conv_5_relu", torch.nn.ReLU(inplace_relu)))

    if pool_size is None:
        avg_pool = co.AdaptiveAvgPool3d(
            (1, 1, 1), kernel_size=temporal_window_size, temporal_fill=temporal_fill
        )
    else:
        avg_pool = co.AvgPool3d(pool_size, stride=1, temporal_fill=temporal_fill)
    modules.append(("avg_pool", avg_pool))

    modules.append(
        (
            "lin_5",
            co.Conv3d(
                dim_inner,
                dim_out,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=False,
            ),
        )
    )
    if bn_lin5_on:
        modules.append(
            ("lin_5_bn", norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt))
        )

    modules.append(("lin_5_relu", torch.nn.ReLU(inplace_relu)))

    if dropout_rate > 0.0:
        modules.append(("dropout", torch.nn.Dropout(dropout_rate)))

    # Perform FC in a fully convolutional manner. The FC layer will be
    # initialized with a different std comparing to convolutional layers.
    modules.append(
        ("projection", co.Linear(dim_out, num_classes, bias=True, channel_dim=1))
    )

    modules.append(
        (
            "act",
            {
                "softmax": torch.nn.Softmax(dim=1),
                "sigmoid": torch.nn.Sigmoid(),
            }[act_func],
        )
    )

    def view(x):
        return x.view(x.shape[0], -1)

    modules.append(("view", co.Lambda(view)))

    return co.Sequential(OrderedDict(modules))


def CoX3DStem(
    dim_in: int,
    dim_out: int,
    kernel: int,
    stride: int,
    padding: int,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    norm_module=torch.nn.BatchNorm3d,
    temporal_fill: PaddingMode = "zeros",
    *args,
    **kwargs,
):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu followed by a
    spatiotemporal pooling.

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
    conv_xy = co.Conv3d(
        dim_in,
        dim_out,
        kernel_size=(1, kernel[1], kernel[2]),
        stride=(1, stride[1], stride[2]),
        padding=(0, padding[1], padding[2]),
        bias=False,
    )

    conv = co.Conv3d(
        dim_out,
        dim_out,
        kernel_size=(kernel[0], 1, 1),
        stride=(stride[0], 1, 1),
        padding=(padding[0], 0, 0),
        bias=False,
        groups=dim_out,
        temporal_fill=temporal_fill,
    )

    bn = norm_module(num_features=dim_out, eps=eps, momentum=bn_mmt)

    relu = torch.nn.ReLU(inplace_relu)

    return co.Sequential(
        OrderedDict(
            [
                ("conv_xy", conv_xy),
                ("conv", conv),
                ("bn", bn),
                ("relu", relu),
            ]
        )
    )


def CoVideoModelStem(
    dim_in: int,
    dim_out: int,
    kernel: int,
    stride: int,
    padding: int,
    inplace_relu=True,
    eps=1e-5,
    bn_mmt=0.1,
    norm_module=torch.nn.BatchNorm3d,
    stem_func_name="x3d_stem",
    temporal_window_size: int = 4,
    temporal_fill: PaddingMode = "zeros",
):
    """
    Args:
        dim_in (int): channel dimensions of the inputs.
        dim_out (int): output dimension of the convolution in the stem
            layer.
        kernel (int): kernel size of the convolutions in the stem
            layers. Temporal kernel size, height kernel size, width kernel
            size in order.
        stride (int): stride sizes of the convolutions in the stem
            layer. Temporal kernel stride, height kernel size, width kernel
            size in order.
        padding (int): paddings sizes of the convolutions in the stem
            layer. Temporal padding size, height padding size, width padding
            size in order.
        inplace_relu (bool): calculate the relu on the original input
            without allocating new memory.
        eps (float): epsilon for batch norm.
        bn_mmt (float): momentum for batch norm. Noted that BN momentum in
            PyTorch = 1 - BN momentum in Caffe2.
        norm_module (torch.nn.Module): torch.nn.Module for the normalization layer. The
            default is torch.nn.BatchNorm3d.
        stem_func_name (string): name of the the stem function applied on
            input to the network.
    """
    assert (
        stem_func_name == "x3d_stem"
    ), "Currently, only 'x3d_stem' stem func is implemented."

    return co.Sequential(
        OrderedDict(
            [
                (
                    "pathway0_stem",
                    CoX3DStem(
                        dim_in,
                        dim_out,
                        kernel,
                        stride,
                        padding,
                        inplace_relu,
                        eps,
                        bn_mmt,
                        norm_module,
                        temporal_window_size=temporal_window_size,
                        temporal_fill=temporal_fill,
                    ),
                )
            ]
        )
    )


def CoX3D(
    dim_in: int,
    image_size: int,
    frames_per_clip: int,
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
    temporal_fill: PaddingMode = "zeros",
    se_scope="frame",
) -> co.Sequential:
    """
    Continual X3D model,
    adapted from https://github.com/facebookresearch/SlowFast

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """
    norm_module = torch.nn.BatchNorm3d
    exp_stage = 2.0
    dim_conv1 = x3d_conv1_dim

    num_groups = x3d_num_groups
    width_per_group = x3d_width_per_group
    dim_inner = num_groups * width_per_group

    w_mul = x3d_width_factor
    d_mul = x3d_depth_factor

    dim_res1 = _round_width(dim_conv1, w_mul)
    dim_res2 = dim_conv1
    dim_res3 = _round_width(dim_res2, exp_stage, divisor=8)
    dim_res4 = _round_width(dim_res3, exp_stage, divisor=8)
    dim_res5 = _round_width(dim_res4, exp_stage, divisor=8)

    block_basis = [
        # blocks, c, stride
        [1, dim_res2, 2],
        [2, dim_res3, 2],
        [5, dim_res4, 2],
        [3, dim_res5, 2],
    ]

    # Basis of temporal kernel sizes for each of the stage.
    temp_kernel = [
        [5],  # conv1 temporal kernels.
        [3],  # res2 temporal kernels.
        [3],  # res3 temporal kernels.
        [3],  # res4 temporal kernels.
        [3],  # res5 temporal kernels.
    ]

    modules = []

    s1 = CoVideoModelStem(
        dim_in=dim_in,
        dim_out=dim_res1,
        kernel=temp_kernel[0] + [3, 3],
        stride=[1, 2, 2],
        padding=[temp_kernel[0][0] // 2, 1, 1],
        norm_module=norm_module,
        stem_func_name="x3d_stem",
        temporal_window_size=frames_per_clip,
        temporal_fill=temporal_fill,
    )
    modules.append(("s1", s1))

    # blob_in = s1
    dim_in = dim_res1
    dim_out = dim_in
    for stage, block in enumerate(block_basis):
        dim_out = _round_width(block[1], w_mul)
        dim_inner = int(x3d_bottleneck_factor * dim_out)

        n_rep = _round_repeats(block[0], d_mul)
        prefix = "s{}".format(stage + 2)  # start w res2 to follow convention

        s = CoResStage(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_inner=dim_inner,
            temp_kernel_sizes=temp_kernel[1],
            stride=block[2],
            num_blocks=n_rep,
            num_groups=dim_inner if x3d_use_channelwise_3x3x3 else num_groups,
            num_block_temp_kernel=n_rep,
            trans_func_name="x3d_transform",
            stride_1x1=False,
            norm_module=norm_module,
            dilation=1,
            drop_connect_rate=0.0,
            temporal_window_size=frames_per_clip,
            temporal_fill=temporal_fill,
            se_scope=se_scope,
        )
        dim_in = dim_out
        modules.append((prefix, s))

    spat_sz = int(math.ceil(image_size / 32.0))
    head = CoX3DHead(
        dim_in=dim_out,
        dim_inner=dim_inner,
        dim_out=x3d_conv5_dim,
        num_classes=num_classes,
        pool_size=(frames_per_clip, spat_sz, spat_sz),
        dropout_rate=x3d_dropout_rate,
        act_func=x3d_head_activation,
        bn_lin5_on=bool(x3d_head_batchnorm),
        temporal_window_size=frames_per_clip,
        temporal_fill=temporal_fill,
    )
    modules.append(("head", head))
    seq = co.Sequential(OrderedDict(modules))
    init_weights(seq, x3d_fc_std_init, bool(x3d_final_batchnorm_zero_init))
    return seq


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


def c2_msra_fill(module: torch.nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        torch.nn.init.constant_(module.bias, 0)


def init_weights(model, fc_init_std=0.01, zero_init_final_bn=True):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d) or isinstance(m, co.Conv3d):
            """
            Follow the initialization method proposed in:
            {He, Kaiming, et al.
            "Delving deep into rectifiers: Surpassing human-level
            performance on imagenet classification."
            arXiv preprint arXiv:1502.01852 (2015)}
            """
            c2_msra_fill(m)
        elif isinstance(m, torch.nn.BatchNorm3d) or isinstance(m, torch.nn.BatchNorm2d):
            if (
                hasattr(m, "transform_final_bn")
                and m.transform_final_bn
                and zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()
