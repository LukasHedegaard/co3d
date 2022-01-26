from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


class MultiCrossEntropyLoss(torch.nn.Module):
    """
    Multi-instance cross-entropy.
    Adapted from: https://github.com/xumingze0308/TRN.pytorch
    """

    def __init__(self, ignore_index: int = None):
        super(MultiCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        target = target.to(dtype=torch.int)

        if self.ignore_index is not None:
            notice_index = [
                i for i in range(target.shape[-1]) if i != self.ignore_index
            ]
            output = torch.sum(
                -target[:, notice_index] * self.logsoftmax(input[:, notice_index]), 1
            )
            if output.sum() == 0:  # ignore result
                loss_ce = torch.tensor(0.0).to(input.device).type_as(target)
            else:
                loss_ce = torch.mean(output[target[:, self.ignore_index] != 1])
        else:
            output = torch.sum(-target * self.logsoftmax(input), 1)
            loss_ce = torch.sum(output)

        return loss_ce


class LabelSmoothingCrossEntropy(nn.Module):
    """NLL loss with label smoothing.
    Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/cross_entropy.py
    to include ignore_classes
    """

    def __init__(self, smoothing=0.1, ignore_classes: Union[int, Sequence[int]] = []):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        if isinstance(ignore_classes, int):
            ignore_classes = [ignore_classes]
        assert all(
            isinstance(i, int) for i in ignore_classes
        ), "Ignore_classes should only contain integers."
        self.ignore_classes = ignore_classes

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ignore_classes:
            sel = torch.tensor(
                list(set(range(x.shape[-1])) - set(self.ignore_classes)),
                device=x.device,
            )
            x = x.index_select(-1, sel)
            target = target.index_select(-1, sel)

        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
