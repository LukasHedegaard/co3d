import math
from typing import Sequence

import torch
from ride.metrics import (
    Configs,
    MetricDict,
    MetricMixin,
    OptimisationDirection,
    attrgetter,
)
from torch import Tensor


def mean_calibrated_average_precision(
    preds: Tensor,
    targets: Tensor,
    skip_classes: Sequence[int] = [],
    empty_target_action="skip",
):
    """Computes the mean calibrated average precision as defined in "Online Action Detection" [De Geest et al.]

    This implementation was adapted from https://github.com/wangxiang1230/OadTR.
    and uses a class-wise retrieval.

    return tuple of
        mAP (float): mean average precision
        mcAP (float): mean calibrated average precision
        classwise_AP (Dict[int, float]): class-wise average precision
        classwise_cAP (Dict[int, float]): class-wise calibrated average precision
    """
    N, C = preds.shape

    # Ensure one-hot encoding for targets
    if len(targets.shape) == 1:
        targets = torch.nn.functional.one_hot(targets, C)

    # Select subset of classes
    cids = torch.arange(C)
    sel = torch.tensor([True for _ in range(C)])
    for c in skip_classes:
        sel.logical_and_(cids != c)
    preds = preds[:, sel]
    targets = targets[:, sel]
    C = preds.shape[1]

    empty = {"skip": float("nan"), "neg": 0.0, "pos": 1.0}[empty_target_action]
    classwise_AP = [empty for _ in range(C)]
    classwise_cAP = [empty for _ in range(C)]

    for i in range(C):
        cls_preds = preds[:, i]
        cls_targets = targets[:, i]

        # Class-wise weighting
        cls_sum = float(torch.sum(cls_targets))
        if cls_sum == 0.0:
            continue

        w = float(torch.sum(cls_targets == 0) / torch.sum(cls_targets == 1))

        indices = torch.argsort(-cls_preds)
        tp, psum, cpsum = 0, 0.0, 0.0
        for k, idx in enumerate(indices):
            if cls_targets[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp) if wtp > 0.0 else 0.0
        cls_AP = psum / cls_sum
        cls_cAP = cpsum / cls_sum

        classwise_AP[i] = cls_AP
        classwise_cAP[i] = cls_cAP

    classwise_AP_dict = {int(k): v for k, v in zip(list(cids[sel]), classwise_AP)}
    classwise_cAP_dict = {int(k): v for k, v in zip(list(cids[sel]), classwise_cAP)}

    if empty_target_action == "skip":
        classwise_AP = [c for c in classwise_AP if not math.isnan(c)]
        classwise_cAP = [c for c in classwise_cAP if not math.isnan(c)]

    if len(classwise_AP) > 0:
        mAP = sum(classwise_AP) / len(classwise_AP)
        mcAP = sum(classwise_cAP) / len(classwise_cAP)
    else:
        mAP = empty
        mcAP = empty

    return mAP, mcAP, classwise_AP_dict, classwise_cAP_dict


def _formatted_mean_calibrated_average_precision(
    preds: Tensor,
    targets: Tensor,
    skip_classes: Sequence[int] = [],
    empty_target_action="skip",
    class_names=[],
):
    (
        mAP,
        mcAP,
        classwise_AP_dict,
        classwise_cAP_dict,
    ) = mean_calibrated_average_precision(
        preds,
        targets,
        skip_classes,
        empty_target_action,
    )

    if len(class_names):

        def name(k):
            return f"{class_names[k]} ({k})"

    else:

        def name(k):
            return k

    return {
        "mAP": torch.tensor(mAP),
        "mcAP": torch.tensor(mcAP),
        **{f"AP/{name(k)}": torch.tensor(v) for k, v in classwise_AP_dict.items()},
        **{f"cAP/{name(k)}": torch.tensor(v) for k, v in classwise_cAP_dict.items()},
    }


class CalibratedMeanAveragePrecisionMetric(MetricMixin):
    """Mean Average Precision (mAP) and Calibrated Mean Average Precision (mcAP)
    for class-wise retrieval as used in "Online Action Detection" [De Geest et al.]"""

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="mean_average_precision_skip_classes",
            type=str,
            default="",
            strategy="constant",
            description="Comma-seperated class indexes to skip during computaion of mAP and cmAP",
        )
        return c

    def validate_attributes(self):
        for hparam in CalibratedMeanAveragePrecisionMetric.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    def __init__(self, *args, **kwargs):
        if isinstance(self.hparams.mean_average_precision_skip_classes, str):
            self.hparams.mean_average_precision_skip_classes = [
                int(i)
                for i in self.hparams.mean_average_precision_skip_classes.split(",")
                if len(i)
            ]

    @classmethod
    def _metrics(cls):
        return {
            "mAP": OptimisationDirection.MAX,
            "mcAP": OptimisationDirection.MAX,
        }

    def metrics_step(
        self, preds: Tensor, targets: Tensor, *args, **kwargs
    ) -> MetricDict:
        return _formatted_mean_calibrated_average_precision(
            preds,
            targets,
            skip_classes=self.hparams.mean_average_precision_skip_classes,
            class_names=getattr(self, "classes", []),
        )

    def metrics_epoch(
        self, preds: Tensor, targets: Tensor, *args, **kwargs
    ) -> MetricDict:
        return _formatted_mean_calibrated_average_precision(
            preds,
            targets,
            skip_classes=self.hparams.mean_average_precision_skip_classes,
            class_names=getattr(self, "classes", []),
        )
