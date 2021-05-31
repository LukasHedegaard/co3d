import torch
from torch.nn.modules.activation import Softmax

from .utils import FillMode


def convert_softmax(
    instance: torch.nn.Softmax,
    window_size: int = None,  # Not used: only there to satisfy interface
    temporal_fill: FillMode = "replicate",  # Not used: only there to satisfy interface
):
    assert instance.dim <= 4, "Cannot convert Softmax with dim > 4."
    return Softmax(dim=3 if instance.dim == 4 else instance.dim)


class Swish(torch.nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishEfficient.apply(x)


class SwishEfficient(torch.autograd.Function):
    """Swish activation function: x * sigmoid(x)."""

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
