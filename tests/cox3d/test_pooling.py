import torch
from torch.nn.modules.pooling import (
    AdaptiveAvgPool3d,
    AdaptiveMaxPool3d,
    AvgPool3d,
    MaxPool3d,
)

from models.cox3d.modules.pooling import (
    AdaptiveAvgPoolCo3d,
    AdaptiveMaxPoolCo3d,
    AvgPoolCo3d,
    MaxPoolCo3d,
)

torch.manual_seed(42)

example_clip = torch.normal(mean=torch.zeros(2 * 4 * 4 * 4)).reshape((1, 2, 4, 4, 4))
example_long = torch.normal(mean=torch.zeros(2 * 8 * 4 * 4)).reshape((1, 2, 8, 4, 4))
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


def test_avg_pool():
    target = AvgPool3d((2, 2, 2))(example_clip)
    output = AvgPoolCo3d(window_size=2, kernel_size=(2, 2)).forward_regular(
        example_clip
    )
    sub_output = torch.stack(
        [
            output[:, :, 0],
            output[:, :, 2],
        ],
        dim=2,
    )
    assert torch.allclose(sub_output, target)


def test_global_avg_pool():
    pool = AdaptiveAvgPool3d((1, 1, 1))
    rpool = AdaptiveAvgPoolCo3d(window_size=4, output_size=(1, 1))

    target = pool(example_clip)
    output = rpool.forward_regular(example_clip)
    assert torch.allclose(output, target)

    # Now that memory is full (via `forward_regular`), pooling works as expected for subsequent frames
    target_next = pool(next_example_clip).squeeze(2)
    output_frame_next = rpool(next_example_frame)
    assert torch.allclose(target_next, output_frame_next)


def test_max_pool():
    target = MaxPool3d((2, 2, 2))(example_clip)
    output = MaxPoolCo3d(window_size=2, kernel_size=(2, 2)).forward_regular(
        example_clip
    )
    sub_output = torch.stack(
        [
            output[:, :, 0],
            output[:, :, 2],
        ],
        dim=2,
    )
    assert torch.allclose(sub_output, target)


def test_global_max_pool():
    target = AdaptiveMaxPool3d((1, 1, 1))(example_clip)
    output = AdaptiveMaxPoolCo3d(window_size=4, output_size=(1, 1)).forward_regular(
        example_clip
    )
    assert torch.allclose(output, target)


def test_dilation():
    target = MaxPool3d((2, 2, 2), dilation=(2, 1, 1))(example_long)
    output = MaxPoolCo3d(
        window_size=4, kernel_size=(2, 2), temporal_dilation=2
    ).forward_regular(example_long)
    assert torch.allclose(target, output.index_select(2, torch.tensor([0, 2, 4])))
