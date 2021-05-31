import torch
import pytest

from models.cox3d.modules.conv_fix import FixedConvCo3d

torch.manual_seed(42)

T = S = 3
example_clip = torch.normal(mean=torch.zeros(4 * 3 * 3)).reshape((1, 1, 4, 3, 3))
next_example_frame = torch.normal(mean=torch.zeros(3 * 3)).reshape((1, 1, 3, 3))
next_example_clip = torch.stack(
    [
        example_clip[:, :, 1],
        example_clip[:, :, 2],
        example_clip[:, :, 3],
        next_example_frame,
    ],
    dim=2,
)
# Long example clip
long_example_clip = torch.normal(mean=torch.zeros(8 * 3 * 3)).reshape((1, 1, 8, 3, 3))
long_next_example_clip = torch.stack(
    [
        *[long_example_clip[:, :, i] for i in range(1, 8)],
        next_example_frame,
    ],
    dim=2,
)


example_clip_large = torch.normal(mean=torch.zeros(2 * 2 * 4 * 8 * 8)).reshape(
    (2, 2, 4, 8, 8)
)


@pytest.mark.skip(reason="Not yet done.")
def test_simple_fixed():
    # Without initialisation using forward3D, the output has no delay

    # Init regular
    conv = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(5, S, S),
        bias=True,
        padding=(0, 1, 1),
        padding_mode="zeros",
    )

    # Init continual
    rconv = FixedConvCo3d.from_3d(conv, temporal_fill="zeros")

    # Targets
    target1 = conv(long_example_clip)
    target2 = conv(long_next_example_clip)

    # Test 3D mode
    output1 = rconv.forward3d(long_example_clip)
    torch.allclose(target1, output1, atol=5e-8)

    # Next 2D forward
    output2 = rconv.forward(next_example_frame.unsqueeze(2))

    # Correct result is output
    assert torch.allclose(target2[:, :, -1], output2, atol=5e-8)


@pytest.mark.skip(reason="Not yet done.")
def test_stacked_pad():
    # Without initialisation using forward3D, the output has no delay

    # Init regular
    conv1 = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(5, S, S),
        bias=True,
        padding=(2, 1, 1),
        padding_mode="zeros",
    )
    conv2 = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=(3, S, S),
        bias=True,
        padding=(1, 1, 1),
        padding_mode="zeros",
    )

    # Init continual
    rconv1 = FixedConvCo3d.from_3d(conv1, temporal_fill="zeros")
    rconv2 = FixedConvCo3d.from_3d(conv2, temporal_fill="zeros")

    # Targets
    target11 = conv1(long_example_clip)
    target12 = conv2(target11)

    target21 = conv1(long_next_example_clip)
    target22 = conv2(target21)

    # Test 3D mode
    output11 = rconv1.forward3d(long_example_clip)
    output12 = rconv2.forward3d(output11)
    torch.allclose(target12, output12, atol=5e-8)

    # Next 2D forward
    output21 = rconv1.forward(next_example_frame)
    output22 = rconv2.forward(output21)

    # Correct result is output
    assert torch.allclose(target22[:, :, -1], output22, atol=5e-8)
