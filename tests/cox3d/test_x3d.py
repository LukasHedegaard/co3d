from pathlib import Path
from urllib.request import urlretrieve

import continual as co
import numpy as np
import pytest
import torch
from continual import TensorPlaceholder
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    ToTensorVideo,
)

from datasets.transforms import RandomShortSideScaleJitterVideo
from models.cox3d.modules.x3d import (
    CoResBlock,
    CoResStage,
    CoVideoModelStem,
    CoX3D,
    CoX3DHead,
    CoX3DTransform,
)
from models.x3d.head_helper import X3DHead
from models.x3d.resnet_helper import ResBlock, ResStage, X3DTransform
from models.x3d.stem_helper import VideoModelStem
from models.x3d.x3d import X3D

torch.manual_seed(42)


def boring_video(
    image_size=160, frames_per_clip=4, save_dir=Path(__file__).parent / "downloads"
):
    save_dir.mkdir(exist_ok=True)

    # Download image
    IMAGE_URL = "https://kids.kiddle.co/images/thumb/8/81/Pacific_Yew_Selfbow.jpg/300px-Pacific_Yew_Selfbow.jpg"
    # IMAGE_URL = "https://www.thetimes.co.uk/imageserver/image/%2Fmethode%2Ftimes%2Fprod%2Fweb%2Fbin%2F5eafdd48-22b6-11e7-bbe5-53dfe0d91782.jpg?crop=3429%2C1929%2C105%2C241&resize=1180"
    # IMAGE_URL = "https://www.lincolnshirelife.co.uk/images/uploads/main-images/_main/1_Michael_Willrich_competing_at_Bonneville_Flats_this_summer.jpg"
    image_name = "archer.jpg"
    image_path = save_dir / image_name
    if not image_path.exists():
        download_path = Path(image_name)
        if not download_path.exists():
            urlretrieve(IMAGE_URL, image_name)
        assert download_path.exists()
        download_path.rename(image_path)

    # Preprocess it (as a "boring" video)
    im = np.asarray(Image.open(image_path))
    vid = torch.tensor(
        np.repeat(im[None, :, :], frames_per_clip, axis=0)
    )  # .transpose(1, 2)
    transforms = Compose(
        [
            ToTensorVideo(),
            RandomShortSideScaleJitterVideo(min_size=image_size, max_size=image_size),
            CenterCropVideo(image_size),
            NormalizeVideo(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
        ]
    )
    vid = transforms(vid).unsqueeze(0)
    return vid


example_s = boring_video(image_size=20)
example_l = boring_video(image_size=160)


def forward_partial(model: torch.nn.Module, input: torch.Tensor, num_modules: int):
    x = [input]
    for i, module in enumerate(model.children()):
        if i < num_modules:
            x = module(x)
        else:
            return x
    return x


def test_VideoModelStem():
    trans = VideoModelStem(
        dim_in=[2],
        dim_out=[2],
        kernel=[[5, 3, 3]],
        stride=[[1, 2, 2]],
        padding=[[2, 1, 1]],
        norm_module=torch.nn.BatchNorm3d,
        stem_func_name="x3d_stem",
    )

    cotrans = CoVideoModelStem(
        dim_in=2,
        dim_out=2,
        kernel=[5, 3, 3],
        stride=[1, 2, 2],
        padding=[2, 1, 1],
        norm_module=torch.nn.BatchNorm3d,
        stem_func_name="x3d_stem",
        temporal_fill="zeros",
    )
    cotrans.load_state_dict(trans.state_dict(), flatten=True)
    assert cotrans.delay == 2

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    cotrans.eval()

    sample = torch.randn((1, 2, 4, 4, 4))

    # Forward through models
    target = trans.forward([sample])[0]

    # forward
    output = cotrans.forward(sample)
    assert torch.allclose(target, output)

    # forward_steps
    output = cotrans.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output)

    # forward_steps - broken up
    cotrans.clean_state()
    nothing = cotrans.forward_steps(sample[:, :, :-2], pad_end=False)  # init
    assert isinstance(nothing, TensorPlaceholder)

    mid = cotrans.forward_step(sample[:, :, -2])
    assert torch.allclose(mid, target[:, :, 0])

    lasts = cotrans.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(lasts, target[:, :, 1:])


def test_CoX3DTransform():
    sample = torch.randn((1, 2, 4, 4, 4))

    # Regular block
    trans = X3DTransform(
        dim_in=2,
        dim_out=2,
        temp_kernel_size=3,
        stride=2,
        dim_inner=5,
        num_groups=5,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=torch.nn.BatchNorm3d,
        se_ratio=0.1,
        swish_inner=True,
        block_idx=0,
    )
    trans.eval()  # This has a major effect on BatchNorm result
    target = trans.forward(sample)

    # Recurrent block
    cotrans = CoX3DTransform(
        dim_in=2,
        dim_out=2,
        temp_kernel_size=3,
        stride=2,
        dim_inner=5,
        num_groups=5,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=torch.nn.BatchNorm3d,
        se_ratio=0.1,
        swish_inner=True,
        block_idx=0,
        temporal_window_size=sample.shape[2],  # +2 from padding
        temporal_fill="zeros",
        se_scope="clip",
    )
    co.load_state_dict(cotrans, trans.state_dict(), flatten=True)
    cotrans.eval()  # This has a major effect on BatchNorm result
    assert cotrans.delay == 4

    # forward
    output = cotrans.forward(sample)
    assert torch.allclose(target, output)

    # forward_steps
    output = cotrans.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output)

    # forward_steps - broken up
    cotrans.clean_state()
    nothing = cotrans.forward_steps(sample[:, :, :-1], pad_end=False)  # init
    assert isinstance(nothing, TensorPlaceholder)

    lasts = cotrans.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(lasts, target)

    # forward_step
    cotrans.clean_state()
    nothing = cotrans.forward_steps(sample[:, :, :], pad_end=False)  # init
    assert isinstance(nothing, TensorPlaceholder)

    # Manual pad end.
    zeros = torch.zeros_like(sample[:, :, 0])
    step = cotrans.forward_step(zeros)
    assert torch.allclose(step, target[:, :, 0])  # (4/4) correct in SE pool
    step = cotrans.forward_step(zeros)
    assert torch.allclose(step, target[:, :, 1], atol=5e-3)  # (3/4) correct in pool
    step = cotrans.forward_step(zeros)
    assert torch.allclose(step, target[:, :, 2], atol=5e-3)  # (2/4) correct in pool
    step = cotrans.forward_step(zeros)
    assert torch.allclose(step, target[:, :, 3], atol=5e-3)  # (1/4) correct in pool


def test_CoX3DTransform_boring_input():

    # Same input across all time-slices
    sample = torch.randn((1, 2, 1, 4, 4)).repeat(1, 1, 4, 1, 1)
    assert sample.shape == (1, 2, 4, 4, 4)

    # Regular block
    trans = X3DTransform(
        dim_in=2,
        dim_out=2,
        temp_kernel_size=3,
        stride=2,
        dim_inner=5,
        num_groups=5,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=torch.nn.BatchNorm3d,
        se_ratio=0.1,
        swish_inner=True,
        block_idx=0,
    )
    trans.eval()  # This has a major effect on BatchNorm result
    target = trans.forward(sample)

    # Recurrent block
    cotrans = CoX3DTransform(
        dim_in=2,
        dim_out=2,
        temp_kernel_size=3,
        stride=2,
        dim_inner=5,
        num_groups=5,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=torch.nn.BatchNorm3d,
        se_ratio=0.1,
        swish_inner=True,
        block_idx=0,
        temporal_window_size=sample.shape[2],  # +2 from padding
        temporal_fill="zeros",
        se_scope="frame",  # <-- Simplifies SE
    )
    co.load_state_dict(cotrans, trans.state_dict(), flatten=True)
    cotrans.eval()  # This has a major effect on BatchNorm result
    assert cotrans.delay == 1  # Less delay

    # forward
    output = cotrans.forward(sample)
    assert torch.allclose(target, output)

    # forward_steps
    output = cotrans.forward_steps(sample, pad_end=True)
    assert torch.allclose(target, output, atol=5e-4)  # approximate

    # Broken up
    cotrans.clean_state()
    firsts = cotrans.forward_steps(sample[:, :, :-2], pad_end=False)  # init
    assert torch.allclose(firsts, target[:, :, :-3], atol=5e-4)

    mid = cotrans.forward_step(sample[:, :, -2])
    assert torch.allclose(mid, target[:, :, -2], atol=5e-4)

    lasts = cotrans.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(lasts, target[:, :, -2:], atol=5e-4)


def test_ResBlock():
    sample = torch.randn((1, 2, 4, 4, 4))

    # Regular block
    trans = ResBlock(
        dim_in=2,
        dim_out=2,
        temp_kernel_size=3,
        stride=2,
        trans_func=X3DTransform,
        dim_inner=5,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=torch.nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
    )

    # Converted block
    cotrans = CoResBlock(
        dim_in=2,
        dim_out=2,
        temp_kernel_size=3,
        stride=2,
        trans_func=CoX3DTransform,
        dim_inner=5,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=torch.nn.BatchNorm3d,
        block_idx=0,
        drop_connect_rate=0.0,
        temporal_window_size=sample.shape[2],
        temporal_fill="zeros",
        se_scope="clip",
    )
    cotrans.load_state_dict(trans.state_dict(), flatten=True)

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    cotrans.eval()

    # Forward through models
    target = trans.forward(sample)

    # forward
    output = cotrans.forward(sample)
    assert torch.allclose(target, output)

    # Broken up
    cotrans.clean_state()
    nothing = cotrans.forward_steps(sample[:, :, :-1], pad_end=False)  # init
    assert isinstance(nothing, TensorPlaceholder)

    lasts = cotrans.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(lasts, target)


def test_ResStage_single():
    # Regular
    trans = ResStage(
        dim_in=[2],
        dim_out=[2],
        dim_inner=[5],
        temp_kernel_sizes=[[3]],
        stride=[2],
        num_blocks=[1],  # <--
        num_groups=[5],
        num_block_temp_kernel=[1],  # <--
        nonlocal_inds=[[]],  # Dummy args used in original impl
        nonlocal_group=[1],  # Dummy args used in original impl
        nonlocal_pool=[[1, 2, 2], [1, 2, 2]],  # Dummy args used in original impl
        instantiation="dot_product",  # Dummy args used in original impl
        trans_func_name="x3d_transform",
        stride_1x1=False,
        norm_module=torch.nn.BatchNorm3d,
        dilation=[1],
        drop_connect_rate=0.0,
    )

    # Converted block
    cotrans = CoResStage(
        dim_in=2,
        dim_out=2,
        dim_inner=5,
        temp_kernel_sizes=[3],
        stride=2,
        num_blocks=1,  # <-
        num_groups=5,
        num_block_temp_kernel=1,  # <-
        trans_func_name="x3d_transform",
        stride_1x1=False,
        norm_module=torch.nn.BatchNorm3d,
        dilation=1,
        drop_connect_rate=0.0,
        temporal_window_size=example_clip.shape[2],
        temporal_fill="zeros",
        se_scope="clip",
    )
    cotrans.load_state_dict(trans.state_dict(), flatten=True)

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    cotrans.eval()

    sample = torch.randn((1, 2, 4, 4, 4))

    # Forward through models
    target = trans.forward([sample])[0]  # needs to be packed due to multi-path support

    # forward
    output = cotrans.forward(sample)
    assert torch.allclose(target, output)

    # Broken up
    cotrans.clean_state()
    nothing = cotrans.forward_steps(sample[:, :, :-1], pad_end=False)  # init
    assert isinstance(nothing, TensorPlaceholder)

    lasts = cotrans.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(lasts, target)


def test_ResStage_multi():
    # Regular
    trans = ResStage(
        dim_in=[2],
        dim_out=[2],
        dim_inner=[5],
        temp_kernel_sizes=[[3]],
        stride=[2],
        num_blocks=[3],  # <--
        num_groups=[5],
        num_block_temp_kernel=[3],  # <--
        nonlocal_inds=[[]],
        nonlocal_group=[1],
        nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
        instantiation="dot_product",
        trans_func_name="x3d_transform",
        stride_1x1=False,
        norm_module=torch.nn.BatchNorm3d,
        dilation=[1],
        drop_connect_rate=0.0,
    )

    # Converted block
    cotrans = CoResStage(
        dim_in=2,
        dim_out=2,
        dim_inner=5,
        temp_kernel_sizes=[3],
        stride=2,
        num_blocks=3,  # <-
        num_groups=5,
        num_block_temp_kernel=3,  # <-
        trans_func_name="x3d_transform",
        stride_1x1=False,
        norm_module=torch.nn.BatchNorm3d,
        dilation=1,
        drop_connect_rate=0.0,
        temporal_window_size=example_clip.shape[2],
        temporal_fill="zeros",
        se_scope="clip",
    )
    cotrans.load_state_dict(trans.state_dict(), flatten=True)

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    cotrans.eval()

    sample = torch.randn((1, 2, 4, 4, 4))

    # Forward through models
    target = trans.forward([sample])[0]  # needs to be packed due to multi-path support

    # forward
    output = cotrans.forward(sample)
    assert torch.allclose(target, output)

    # Broken up
    cotrans.clean_state()
    nothing = cotrans.forward_steps(sample[:, :, :-1], pad_end=False)  # init
    assert isinstance(nothing, TensorPlaceholder)

    lasts = cotrans.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(lasts, target)


def test_CoX3DHead():
    trans = X3DHead(
        dim_in=2,
        dim_inner=5,
        dim_out=3,
        num_classes=3,
        pool_size=(4, 4, 4),
        dropout_rate=0.0,
        act_func="softmax",
        bn_lin5_on=False,
    )

    cotrans = CoX3DHead(
        dim_in=2,
        dim_inner=5,
        dim_out=3,
        num_classes=3,
        pool_size=(4, 4, 4),
        dropout_rate=0.0,
        act_func="softmax",
        bn_lin5_on=False,
        temporal_window_size=4,
        temporal_fill="zeros",
    )
    cotrans.load_state_dict(trans.state_dict())

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    cotrans.eval()

    sample = torch.randn((1, 2, 4, 4, 4))

    # Forward through models
    target = trans.forward([sample])[0]  # needs to be packed due to multi-path support

    # forward
    output = cotrans.forward(sample)
    assert torch.allclose(target, output)

    # Broken up
    cotrans.clean_state()
    nothing = cotrans.forward_steps(sample[:, :, :-1], pad_end=False)  # init
    assert isinstance(nothing, TensorPlaceholder)

    lasts = cotrans.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(lasts, target)


example_clip = torch.normal(mean=torch.zeros(2 * 4 * 4 * 4)).reshape((1, 2, 4, 4, 4))
next_example_frame = torch.normal(mean=torch.zeros(2 * 1 * 4 * 4)).reshape((1, 2, 4, 4))
next_example_clip = torch.stack(
    [
        example_clip[:, :, 1],
        example_clip[:, :, 2],
        example_clip[:, :, 3],
        next_example_frame,
    ],
    dim=2,
)


def download_weights(
    url="https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_s.pyth",
    save_dir=Path(__file__).parent / "downloads",
):
    save_dir.mkdir(exist_ok=True)

    # Download pretrained model
    weight_name = url.split("/")[-1]
    weights_path = save_dir / weight_name
    if not weights_path.exists():
        download_path = Path(weight_name)
        if not download_path.exists():
            urlretrieve(url, weight_name)
        assert download_path.exists()
        download_path.rename(weights_path)

    return weights_path


def x3d_feature_example(num_modules: int):
    weights_path = download_weights()
    video_input = boring_video(image_size=160)

    # Regular model
    model = X3D(
        dim_in=3,
        image_size=160,
        frames_per_clip=4,
        num_classes=400,
        x3d_conv1_dim=12,
        x3d_conv5_dim=2048,
        x3d_num_groups=1,
        x3d_width_per_group=64,
        x3d_width_factor=2.0,
        x3d_depth_factor=2.2,
        x3d_bottleneck_factor=2.25,
        x3d_use_channelwise_3x3x3=1,
        x3d_dropout_rate=0.5,
        x3d_head_activation="softmax",
        x3d_head_batchnorm=0,
        x3d_fc_std_init=0.01,
        x3d_final_batchnorm_zero_init=1,
    )

    model_state = torch.load(weights_path, map_location="cpu")["model_state"]
    model.load_state_dict(model_state)
    model.eval()
    result = forward_partial(model, video_input, num_modules)
    return result


@pytest.mark.skip()
def test_CoX3D():
    weights_path = download_weights()
    frames_per_clip = 13
    input = boring_video(image_size=160, frames_per_clip=frames_per_clip)

    # Regular model
    model = X3D(
        dim_in=3,
        image_size=160,
        frames_per_clip=4,
        num_classes=400,
        x3d_conv1_dim=12,
        x3d_conv5_dim=2048,
        x3d_num_groups=1,
        x3d_width_per_group=64,
        x3d_width_factor=2.0,
        x3d_depth_factor=2.2,
        x3d_bottleneck_factor=2.25,
        x3d_use_channelwise_3x3x3=1,
        x3d_dropout_rate=0.5,
        x3d_head_activation="softmax",
        x3d_head_batchnorm=0,
        x3d_fc_std_init=0.01,
        x3d_final_batchnorm_zero_init=1,
    )

    # Continual model
    comodel = CoX3D(
        dim_in=3,
        image_size=160,
        frames_per_clip=frames_per_clip,
        num_classes=400,
        x3d_conv1_dim=12,
        x3d_conv5_dim=2048,
        x3d_num_groups=1,
        x3d_width_per_group=64,
        x3d_width_factor=2.0,
        x3d_depth_factor=2.2,
        x3d_bottleneck_factor=2.25,
        x3d_use_channelwise_3x3x3=1,
        x3d_dropout_rate=0.5,
        x3d_head_activation="softmax",
        x3d_head_batchnorm=0,
        x3d_fc_std_init=0.01,
        x3d_final_batchnorm_zero_init=1,
        temporal_fill="zeros",
        se_scope="clip",
    )

    # Load weights
    model_state = torch.load(weights_path, map_location="cpu")["model_state"]
    model.load_state_dict(model_state)
    comodel.load_state_dict(model_state)

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    model.eval()
    comodel.eval()

    # Forward
    target = model(input)
    target_top10 = torch.topk(target, k=10)[1][0].tolist()

    # forward_steps produces same outputs
    output = comodel.forward_steps(input)
    assert torch.allclose(target, output, atol=5e-3)

    # Continuation is not exact due to zero paddings (even for boring video)
    output_next_frame = comodel.forward(input[:, :, -1])

    # assert torch.allclose(target, output_next_frame, atol=5e-4)

    next_frame_top10 = torch.topk(output_next_frame, k=10)[1][0].tolist()
    # Top 2 is the same
    assert len(set(target_top10[:2]) - set(next_frame_top10[:2])) == 0
    # Top 4 is overlapping
    assert len(set(target_top10[:4]) - set(next_frame_top10[:4])) == 0
    # Top 10 is mostly overlapping
    assert len(set(target_top10) - set(next_frame_top10)) == 1

    # Check output if we continue for a long time
    for _ in range(20):
        output_next_frame = comodel.forward(input[:, :, -1])

    # They are further apart now
    assert torch.allclose(target, output_next_frame, atol=5e-2)
    _, next_frame_top10 = torch.topk(output_next_frame, k=10)

    next_frame_top10 = torch.topk(output_next_frame, k=10)[1][0].tolist()
    # Top 1 is the same
    assert len(set(target_top10[:1]) - set(next_frame_top10[:1])) == 0
    # Top 10 is half-way overlapping
    assert len(set(target_top10) - set(next_frame_top10)) == 5
