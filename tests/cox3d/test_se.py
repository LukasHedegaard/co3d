import torch

from models.cox3d.modules.se import SE, CoSe

torch.manual_seed(42)

example_input = torch.normal(mean=torch.zeros(2 * 4 * 4 * 4)).reshape((1, 2, 4, 4, 4))


def test_se_block():

    # Regular SE block
    se = SE(dim_in=2, ratio=10)
    target1 = se(example_input)

    # Recursive SE block
    rse = CoSe(window_size=4, dim_in=2, ratio=10, scope="clip")
    rse.load_state_dict(se.state_dict())

    # Frame-wise
    _ = rse.forward(example_input[:, :, 0])
    _ = rse.forward(example_input[:, :, 1])
    _ = rse.forward(example_input[:, :, 2])
    output1 = rse.forward(example_input[:, :, 3])

    assert torch.allclose(target1[:, :, 3], output1)

    # Clip
    output2 = rse.forward_regular(example_input)
    assert torch.allclose(target1, output2)

    # After initialising with whole clip, next frame works as expected
    next_frame = torch.normal(mean=torch.zeros(2 * 1 * 4 * 4)).reshape((1, 2, 4, 4))
    next_clip = torch.stack(
        [
            example_input[:, :, 1],
            example_input[:, :, 2],
            example_input[:, :, 3],
            next_frame,
        ],
        dim=2,
    )

    target2 = se(next_clip)
    output3 = rse.forward(next_frame)
    assert torch.allclose(target2[:, :, -1], output3)

    # NB: prior frames in yield different result because a
    # "lookahead" is used in regular global avg pool wrt. those frames!
    assert not torch.allclose(target2[:, :, 0], output2[:, :, 1])
    assert not torch.allclose(target2[:, :, 1], output2[:, :, 2])
    assert not torch.allclose(target2[:, :, 2], output2[:, :, 3])
