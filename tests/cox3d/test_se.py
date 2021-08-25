from continual import TensorPlaceholder
import torch

from models.cox3d.modules.se import SE, CoSe, CoSeAlt

torch.manual_seed(42)


def test_se_block():
    sample = torch.randn((1, 2, 4, 4, 4))

    # Regular SE block
    se = SE(dim_in=2, ratio=10)
    target1 = se(sample)

    # Recursive SE block - first impl
    cose = CoSeAlt.build_from(se, window_size=4, scope="clip")

    # Frame-wise
    _ = cose.forward_step(sample[:, :, 0])
    _ = cose.forward_step(sample[:, :, 1])
    _ = cose.forward_step(sample[:, :, 2])
    output1 = cose.forward_step(sample[:, :, 3])

    assert torch.allclose(target1[:, :, 3], output1)

    # Clip
    output2 = cose.forward_steps(sample)
    assert torch.allclose(target1, output2)

    # After initialising with whole clip, next frame works as expected
    next_frame = torch.normal(mean=torch.zeros(2 * 1 * 4 * 4)).reshape((1, 2, 4, 4))
    next_clip = torch.stack(
        [
            sample[:, :, 1],
            sample[:, :, 2],
            sample[:, :, 3],
            next_frame,
        ],
        dim=2,
    )

    target2 = se(next_clip)
    output3 = cose.forward_step(next_frame)
    assert torch.allclose(target2[:, :, -1], output3)

    # NB: prior frames in yield different result because a
    # "lookahead" is used in regular global avg pool wrt. those frames!
    assert not torch.allclose(target2[:, :, 0], output2[:, :, 1])
    assert not torch.allclose(target2[:, :, 1], output2[:, :, 2])
    assert not torch.allclose(target2[:, :, 2], output2[:, :, 3])

    # New implementation:
    cose = CoSe(window_size=4, dim_in=2, ratio=10, scope="clip")
    cose.load_state_dict(se.state_dict(), flatten=True)
    output = cose.forward_steps(sample, pad_end=True)
    assert torch.allclose(target1, output)

    # broken up
    cose.clean_state()
    nothing = cose.forward_steps(sample[:, :, :-1], pad_end=False)
    assert isinstance(nothing, TensorPlaceholder)

    lasts = cose.forward_steps(sample[:, :, -1:], pad_end=True)
    assert torch.allclose(target1, lasts)
