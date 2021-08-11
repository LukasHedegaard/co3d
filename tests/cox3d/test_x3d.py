from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
    ToTensorVideo,
)

from datasets.transforms import RandomShortSideScaleJitterVideo
from models.cox3d.modules.x3d import (
    CoX3D,
    CoX3DHead,
    CoX3DTransform,
    ReResBlock,
    ReResStage,
    ReVideoModelStem,
)
from models.x3d.head_helper import X3DHead
from models.x3d.resnet_helper import ResBlock, ResStage, X3DTransform
from models.x3d.stem_helper import VideoModelStem
from models.x3d.x3d import X3D

torch.manual_seed(42)

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
    url="https://dl.fbaipublicfiles.com/pyslowfast/x3d_models/x3d_xs.pyth",
    save_dir=Path(Path(__file__).parent / "downloads"),
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


def boring_video(image_size=160, save_dir=Path(Path(__file__).parent / "downloads")):
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
    vid = torch.tensor(np.repeat(im[None, :, :], 4, axis=0))  # .transpose(1, 2)
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


def forward_partial(model: torch.nn.Module, input: torch.Tensor, num_modules: int):
    x = [input]
    for i, module in enumerate(model.children()):
        if i < num_modules:
            x = module(x)
        else:
            return x
    return x


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


def test_CoX3D():
    weights_path = download_weights()
    input = boring_video(image_size=160)

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
    remodel = CoX3D(
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
        temporal_fill="zeros",
    )

    # Load weights
    model_state = torch.load(weights_path, map_location="cpu")["model_state"]
    model.load_state_dict(model_state)
    remodel.load_state_dict(model_state)

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    model.eval()
    remodel.eval()

    # Forward
    target = model(input)
    target_top10 = torch.topk(target, k=10)[1][0].tolist()

    # forward_regular produces same outputs
    output = remodel.forward_regular(input)
    assert torch.allclose(target, output, atol=1e-7)

    # Continuation is not exact due to zero paddings (even for boring video)
    output_next_frame = remodel.forward(input[:, :, -1])

    assert torch.allclose(target, output_next_frame, atol=5e-4)

    next_frame_top10 = torch.topk(output_next_frame, k=10)[1][0].tolist()
    # Top 2 is the same
    assert len(set(target_top10[:2]) - set(next_frame_top10[:2])) == 0
    # Top 4 is overlapping
    assert len(set(target_top10[:4]) - set(next_frame_top10[:4])) == 0
    # Top 10 is mostly overlapping
    assert len(set(target_top10) - set(next_frame_top10)) == 1

    # Check output if we continue for a long time
    for _ in range(20):
        output_next_frame = remodel.forward(input[:, :, -1])

    # They are further apart now
    assert torch.allclose(target, output_next_frame, atol=5e-2)
    _, next_frame_top10 = torch.topk(output_next_frame, k=10)

    next_frame_top10 = torch.topk(output_next_frame, k=10)[1][0].tolist()
    # Top 1 is the same
    assert len(set(target_top10[:1]) - set(next_frame_top10[:1])) == 0
    # Top 10 is half-way overlapping
    assert len(set(target_top10) - set(next_frame_top10)) == 5


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

    rtrans = ReVideoModelStem(
        dim_in=[2],
        dim_out=[2],
        kernel=[[5, 3, 3]],
        stride=[[1, 2, 2]],
        padding=[[2, 1, 1]],
        norm_module=torch.nn.BatchNorm3d,
        stem_func_name="x3d_stem",
        temporal_fill="zeros",
    )
    rtrans.load_state_dict(trans.state_dict())

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    rtrans.eval()

    # Forward through models
    target = trans([example_clip])[0]

    outputs = []
    # Manual zero pad (due to padding=1 in trans.b Conv3d)
    zeros = torch.zeros_like(example_clip[:, :, 0])
    outputs.append(rtrans.forward([zeros])[0])
    outputs.append(rtrans.forward([zeros])[0])
    for i in range(example_clip.shape[2]):
        outputs.append(rtrans.forward([example_clip[:, :, i]])[0])
    outputs.append(rtrans.forward([zeros])[0])
    outputs.append(rtrans.forward([zeros])[0])

    # For debugging:
    close = []
    for t in range(target.shape[2]):
        for i in range(len(outputs)):
            if torch.allclose(target[:, :, t], outputs[i], atol=5e-4):
                close.append(f"t = {t}, o = {i}")

    shift = 4  # kernel[0] - 1

    for t in range(1, target.shape[2]):
        assert torch.allclose(target[:, :, t], outputs[t + shift])

    # forward_regular also works
    outputs2 = rtrans.forward_regular([example_clip])[0]
    assert torch.allclose(target, outputs2)


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

    rtrans = CoX3DHead(
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
    rtrans.load_state_dict(trans.state_dict())

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    rtrans.eval()

    # Forward through models
    target = trans([example_clip])[0]

    outputs = []
    for i in range(example_clip.shape[2]):
        outputs.append(rtrans.forward([example_clip[:, :, i]]))

    # For debugging:
    # close = []
    # for i in range(len(outputs)):
    #     if torch.allclose(target, outputs[i], atol=5e-4):
    #         close.append(f"o = {i}")

    assert torch.allclose(target, outputs[3])

    # forward_regular also works
    outputs2 = rtrans.forward_regular([example_clip])[0]
    assert torch.allclose(target, outputs2)


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
    rtrans = ReResStage(
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
        temporal_window_size=example_clip.shape[2],
        temporal_fill="zeros",
    )
    rtrans.load_state_dict(trans.state_dict())

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    rtrans.eval()

    # Forward through models
    target = trans([example_clip])[0]

    outputs = []
    # Manual zero pad (due to padding=1 in trans.b Conv3d)
    zeros = torch.zeros_like(example_clip[:, :, 0])
    outputs.append(rtrans.forward([zeros])[0])
    for i in range(example_clip.shape[2]):
        outputs.append(rtrans.forward([example_clip[:, :, i]])[0])
    outputs.append(rtrans.forward([zeros])[0])
    outputs.append(rtrans.forward([zeros])[0])
    outputs.append(rtrans.forward([zeros])[0])

    # For debugging:
    close = []
    for t in range(target.shape[2]):
        for i in range(len(outputs)):
            if torch.allclose(target[:, :, t], outputs[i], atol=5e-4):
                close.append(f"t = {t}, o = {i}")

    shift = 1 + 3 * 1  # 1 from zero-padding in Conv3d, 3*1 from delay in each block

    # SE block global average pool is still "filling up"
    # This takes "temporal_window_size" which is 4 for each block
    for t in range(1, target.shape[2]):
        assert torch.allclose(target[:, :, t], outputs[t + shift], atol=5e-4)

    # forward_regular also works
    outputs2 = rtrans.forward_regular([example_clip])[0]
    assert torch.allclose(target, outputs2)


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
    rtrans = ReResStage(
        dim_in=[2],
        dim_out=[2],
        dim_inner=[5],
        temp_kernel_sizes=[[3]],
        stride=[2],
        num_blocks=[1],  # <--
        num_groups=[5],
        num_block_temp_kernel=[1],  # <--
        nonlocal_inds=[[]],
        nonlocal_group=[1],
        nonlocal_pool=[[1, 2, 2], [1, 2, 2]],
        instantiation="dot_product",
        trans_func_name="x3d_transform",
        stride_1x1=False,
        norm_module=torch.nn.BatchNorm3d,
        dilation=[1],
        drop_connect_rate=0.0,
        temporal_window_size=example_clip.shape[2],
        temporal_fill="zeros",
    )
    rtrans.load_state_dict(trans.state_dict())

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    rtrans.eval()

    # Forward through models
    target = trans([example_clip])[0]

    outputs = []
    # Manual zero pad (due to padding=1 in trans.b Conv3d)
    zeros = torch.zeros_like(example_clip[:, :, 0])
    outputs.append(rtrans.forward([zeros])[0])
    for i in range(example_clip.shape[2]):
        outputs.append(rtrans.forward([example_clip[:, :, i]])[0])
    outputs.append(rtrans.forward([zeros])[0])
    outputs.append(rtrans.forward([zeros])[0])
    outputs.append(rtrans.forward([zeros])[0])

    # For debugging:
    # close = []
    # for t in range(target.shape[2]):
    #     for i in range(len(outputs)):
    #         if torch.allclose(target[:, :, t], outputs[i], atol=5e-4):
    #             close.append(f"t = {t}, o = {i}")

    shift = 1 + 1  # 1 from zero-padding in Conv3d, 1 from delay

    # SE block global average pool is still "filling up"
    # This takes "temporal_window_size" which is 4 here
    for t in range(1, target.shape[2]):
        assert torch.allclose(target[:, :, t], outputs[t + shift], atol=5e-4)

    # After temporal_window_size inputs it also produces precise computation
    torch.allclose(target[:, :, 3], outputs[3 + shift], atol=1e-9)

    # forward_regular also works
    outputs2 = rtrans.forward_regular([example_clip])[0]
    assert torch.allclose(target, outputs2)


def test_ResBlock():
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
    rtrans = ReResBlock(
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
        temporal_window_size=example_clip.shape[2],
        temporal_fill="zeros",
    )
    rtrans.load_state_dict(trans.state_dict())

    # Training mode has large effect on BatchNorm result - test will fail otherwise
    trans.eval()
    rtrans.eval()

    # Forward through models
    target = trans(example_clip)

    outputs = []
    # Manual zero pad (due to padding=1 in trans.b Conv3d)
    zeros = torch.zeros_like(example_clip[:, :, 0])
    outputs.append(rtrans.forward(zeros))
    for i in range(example_clip.shape[2]):
        outputs.append(rtrans.forward(example_clip[:, :, i]))
    outputs.append(rtrans.forward(zeros))
    outputs.append(rtrans.forward(zeros))
    outputs.append(rtrans.forward(zeros))

    # For debugging:
    # close = []
    # for t in range(target.shape[2]):
    #     for i in range(len(outputs)):
    #         if torch.allclose(target[:, :, t], outputs[i], atol=5e-2):
    #             close.append(f"t = {t}, o = {i}")

    shift = 3 - 1  # From Conv3d

    # Not very precise because SE block still initializes
    for t in range(1, target.shape[2]):
        assert torch.allclose(target[:, :, t], outputs[t + shift], rtol=5e-3)

    # After temporal_window_size inputs it also produces precise computation
    torch.allclose(target[:, :, 3], outputs[3 + shift], atol=1e-9)

    # forward_regular also works as expected
    outputs2 = rtrans.forward_regular(example_clip)
    assert torch.allclose(target, outputs2)


def test_CoX3DTransform():
    # Regular bloack
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
    target = trans(example_clip)

    # Recurrent block
    rtrans = CoX3DTransform(
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
        temporal_window_size=example_clip.shape[2],
        temporal_fill="zeros",
    )
    rtrans.load_state_dict(trans.state_dict())
    rtrans.eval()  # This has a major effect on BatchNorm result

    # Forward 3D works like the original
    output3d = rtrans.forward_regular(example_clip)
    assert torch.allclose(target, output3d)

    o = []
    # Manual zero pad (due to padding=1 in Conv3d)
    zeros = torch.zeros_like(example_clip[:, :, 0])
    o.append(rtrans.forward(zeros))
    for i in range(example_clip.shape[2]):
        o.append(rtrans.forward(example_clip[:, :, i]))
    o.append(rtrans.forward(zeros))

    # For debugging:
    # close = []
    # equal = []
    # for t in range(target.shape[2]):
    #     for i in range(len(o)):
    #         if torch.allclose(target[:, :, t], o[i], atol=5e-3):
    #             close.append(f"t = {t}, o = {i}")
    #         if torch.equal(target[:, :, t], o[i]):
    #             equal.append(f"t = {t}, o = {i}")

    shift = 3 - 1  # From Conv3d
    # Not very precise because SE block still initializes
    for t in range(target.shape[2]):
        assert torch.allclose(target[:, :, t], o[t + shift], atol=5e-3)

    # After temporal_window_size inputs it also produces precise computation
    torch.allclose(target[:, :, 3], o[3 + shift], atol=1e-9)

    # Forward 3D works like the original
    output3d = rtrans.forward_regular(example_clip)
    assert torch.allclose(target, output3d)
