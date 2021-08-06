import functools

import numpy as np
import torch


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def nice_weights(w: torch.Tensor, keep_sign=True) -> torch.Tensor:
    total = np.prod(list(w.shape))
    vals = np.arange(total)
    if keep_sign:
        if w.max() <= 0:
            vals = -vals
        elif w.min() < 0:
            vals = 2 * (vals - total) + 1

    return torch.nn.Parameter(torch.tensor(vals).float().reshape(w.shape))


def nice_module(m: torch.nn.Module) -> torch.nn.Module:
    for n, p in m.named_parameters():
        rsetattr(m, n, nice_weights(p))
    return m
