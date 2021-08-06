import torch


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
