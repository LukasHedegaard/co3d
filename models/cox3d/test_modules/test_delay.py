import torch

from models.cox3d.modules.delay import Delay

torch.manual_seed(42)

example_input = torch.normal(mean=torch.zeros(4 * 3 * 3)).reshape((1, 1, 4, 3, 3))


def test_delay():
    delay = Delay(window_size=3, temporal_fill="zeros")

    zeros = torch.zeros_like(example_input[:, :, 0])
    ones = torch.ones_like(example_input[:, :, 0])

    assert torch.equal(delay(example_input[:, :, 0]), zeros)

    assert torch.equal(delay(example_input[:, :, 1]), zeros)

    assert torch.equal(delay(example_input[:, :, 2]), example_input[:, :, 0])

    assert torch.equal(delay(example_input[:, :, 3]), example_input[:, :, 1])

    assert torch.equal(delay(ones), example_input[:, :, 2])

    assert torch.equal(delay(ones), example_input[:, :, 3])

    assert torch.equal(delay(ones), ones)
