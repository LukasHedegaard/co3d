import numpy as np
import torch
from metrics import mean_calibrated_average_precision
from torchmetrics import AveragePrecision, RetrievalMAP


def frame_level_map_n_cap(all_probs, all_labels):
    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(0, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0.0, 0.0
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        this_cls_ap = psum / np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    mAP = sum(all_cls_ap) / len(all_cls_ap)
    cap = sum(all_cls_acp) / len(all_cls_acp)
    return mAP, all_cls_ap, cap, all_cls_acp


def test_map():
    pred = torch.tensor(
        [
            [0.75, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.75, 0.05, 0.05, 0.05],
            [0.05, 0.75, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.75, 0.05, 0.05],
            [0.05, 0.05, 0.75, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.75, 0.05],
            [0.75, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.75, 0.05, 0.05, 0.05],
            [0.05, 0.30, 0.05, 0.05, 0.55],
            [0.05, 0.30, 0.35, 0.05, 0.25],
        ]
    )
    target = torch.tensor([0, 0, 1, 3, 2, 2, 1, 1, 4, 4])
    target_oh = torch.nn.functional.one_hot(target, 5)

    # TorchMetric v1
    average_precision = AveragePrecision(num_classes=5)
    ap_tm_1 = average_precision(pred, target)
    mAP_tm_1 = torch.tensor([t for t in ap_tm_1 if not t.isnan()]).mean()

    # TorchMetric v2
    rmap = RetrievalMAP()
    mAP_tm_2 = rmap(
        pred, target_oh, indexes=torch.arange(pred.shape[1]).repeat(pred.shape[0], 1)
    )

    assert mAP_tm_1 != mAP_tm_2

    # OadTR implementation
    mAP_tr, all_cls_ap, cap, all_cls_acp = frame_level_map_n_cap(
        np.array(pred.T), np.array(torch.nn.functional.one_hot(target, 5).T)
    )
    assert mAP_tm_2 == mAP_tr

    # New impl
    mAP_new, cmAP_new, _, _ = mean_calibrated_average_precision(pred, target)
    assert mAP_new - mAP_tr < 1e-9
    assert cmAP_new - cap < 1e-9
