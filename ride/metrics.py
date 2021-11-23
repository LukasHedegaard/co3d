from collections import abc
from enum import Enum
from operator import attrgetter
from typing import Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from ptflops import get_model_complexity_info
from supers import supers
from torch import Tensor
from torchmetrics.functional.classification import average_precision
from torchmetrics.functional.classification.confusion_matrix import confusion_matrix

from ride.core import Configs, RideMixin
from ride.utils.logging import getLogger
from ride.utils.utils import merge_dicts, name

ExtendedMetricDict = Dict[str, Union[Tensor, Figure]]
MetricDict = Dict[str, Tensor]
FigureDict = Dict[str, Figure]
StepOutputs = List[Dict[str, Tensor]]

logger = getLogger(__name__, log_once=True)


def sort_out_figures(d: ExtendedMetricDict) -> Tuple[MetricDict, FigureDict]:
    mets, figs = {}, {}
    for k, v in d.items():
        if type(v) == Figure:
            figs[k] = v
        else:
            mets[k] = v
    return mets, figs


class OptimisationDirection(Enum):
    MIN = "min"
    MAX = "max"


class MetricMixin(RideMixin):
    """Abstract base class for Ride modules"""

    def __init_subclass__(cls):
        if not hasattr(cls, "_metrics"):
            logger.error(
                f"Subclasses of MetricMixin should define a `_metrics` classmethod, but none was found in {name(cls)}"
            )

    @classmethod
    def metrics(cls) -> Dict[str, str]:
        ms = merge_dicts(
            *[c._metrics() for c in cls.__bases__ if issubclass(c, MetricMixin)]
        )
        return ms

    @classmethod
    def metric_names(cls) -> List[str]:
        return list(sorted(cls.metrics().keys()))

    def metrics_step(self, *args, **kwargs) -> MetricDict:
        return {}  # pragma: no cover

    def metrics_epoch(
        self, preds: Tensor, targets: Tensor, prefix: str = "", *args, **kwargs
    ) -> MetricDict:
        return {}

    def collect_metrics(self, preds: Tensor, targets: Tensor) -> MetricDict:
        mdlist: List[MetricDict] = supers(self).metrics_step(preds, targets)  # type: ignore
        return {
            k: v.to(device=self.device) if hasattr(v, "to") else v
            for md in mdlist
            for k, v in md.items()
        }

    def collect_epoch_metrics(
        self, preds: Tensor, targets: Tensor, prefix: str = None
    ) -> ExtendedMetricDict:
        mdlist: List[ExtendedMetricDict] = supers(self).metrics_epoch(preds, targets, prefix=prefix)  # type: ignore
        return {
            k: v.to(device=self.device) if hasattr(v, "to") else v
            for md in mdlist
            for k, v in md.items()
        }


def MetricSelector(  # noqa: C901
    mapping: Dict[str, Union[MetricMixin, Iterable[MetricMixin]]] = None,
    default_config: str = "",
    **kwargs: Union[MetricMixin, Iterable[MetricMixin]],
) -> MetricMixin:
    if not isinstance(mapping, dict):
        mapping = {}

    mapping = {**mapping, **kwargs}
    # Ensure mapping is Dict[str, List[MetricMixin]]
    mapping = {
        k: (list(v) if isinstance(v, abc.Iterable) else [v]) for k, v in mapping.items()
    }
    metric_set = set([item for sublist in mapping.values() for item in sublist])
    assert all(
        issubclass(M, MetricMixin) for M in metric_set
    ), "All passed values should be of type ride.metrics.MetricMixin"

    class MetricSelectorMixin(MetricMixin):
        @staticmethod
        def configs() -> Configs:
            c = Configs()
            c.add(
                name="metric_selection",
                default=default_config,
                type=str,
                strategy="constant",
                description="Selection key for MetricSelector.",
                choices=list(mapping.keys()),
            )
            for Metric in metric_set:
                if hasattr(Metric, "configs"):
                    c += Metric.configs()
            return c

        @classmethod
        def _metrics(cls):
            ms = {}
            for Metric in metric_set:
                ms = {**ms, **Metric._metrics()}
            return ms

        def __init__(self, hparams, *args, **kwargs):
            assert (
                self.hparams.metric_selection in mapping
            ), f"You must specify a `metric_selection` hyperparameter. Choices: {list(mapping.keys())}"
            self.metrics_selection = mapping[self.hparams.metric_selection]
            for m in self.metrics_selection:
                m.__init__(self, hparams, *args, **kwargs)

        def on_init_end(self, *args, **kwargs):
            for m in self.metrics_selection:
                m.on_init_end(self, *args, **kwargs)

        def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
            res = {}
            for m in self.metrics_selection:
                res = {**res, **m.metrics_step(self, preds, targets, **kwargs)}
            return res

        def metrics_epoch(
            self, preds: Tensor, targets: Tensor, prefix: str = "", *args, **kwargs
        ) -> MetricDict:
            res = {}
            for m in self.metrics_selection:
                res = {
                    **res,
                    **m.metrics_epoch(self, preds, targets, prefix, *args, **kwargs),
                }
            return res

    return MetricSelectorMixin


class MeanAveragePrecisionMetric(MetricMixin):
    """Mean Average Precision (mAP) metric"""

    def validate_attributes(self):
        for attribute in ["hparams.loss", "classes"]:
            attrgetter(attribute)(self)

    def _compute_mean_average_precision(self, preds, targets):
        try:
            ap = average_precision(
                preds,
                targets,
                num_classes=targets.shape[-1],
            )
        except RuntimeError as e:  # pragma: no cover
            logger.error("Unable to compute Average Precision: ", e)
            return torch.tensor(float("nan"))

        if isinstance(getattr(self, "ignore_classes", None), list):
            ap = [t for i, t in enumerate(ap) if i not in self.ignore_classes]

        if isinstance(ap, list):
            ap = torch.tensor([t for t in ap if not t.isnan()])

        return ap.mean()

    @classmethod
    def _metrics(cls):
        return {"mAP": OptimisationDirection.MAX}

    def metrics_step(
        self, preds: Tensor, targets: Tensor, *args, **kwargs
    ) -> MetricDict:
        return {"mAP": self._compute_mean_average_precision(preds, targets)}

    def metrics_epoch(
        self, preds: Tensor, targets: Tensor, *args, **kwargs
    ) -> MetricDict:
        return {"mAP": self._compute_mean_average_precision(preds, targets)}


def TopKAccuracyMetric(*Ks) -> MetricMixin:
    if not Ks:
        Ks = [1, 3, 5]

    for k in Ks:
        assert type(k) == int and k > 0

    class TopKAccuracyMetricClass(MetricMixin):
        """Top K accuracy metrics: top1acc, top3acc, top5acc"""

        @classmethod
        def _metrics(cls):
            return {f"top{k}acc": OptimisationDirection.MAX for k in Ks}

        def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
            ks = [k for k in Ks]
            accs = [torch.tensor(-1.0) for _ in ks]
            try:
                accs = topk_accuracies(preds, targets, ks)
            except RuntimeError:  # pragma: no cover
                logger.error("Unable to compute top-k accuracy.")
            return {f"top{k}acc": accs[i] for i, k in enumerate(ks)}

    return TopKAccuracyMetricClass


class FlopsMetric(MetricMixin):
    """Computes Floating Point Operations (FLOPs) for the model and adds it as metric"""

    @classmethod
    def _metrics(cls):
        return {"flops": OptimisationDirection.MIN}

    def on_init_end(self, *args, **kwargs):
        assert isinstance(self, torch.nn.Module)
        self.flops = flops(self)  # type: ignore

    def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
        return {"flops": torch.tensor(self.flops)}


class FlopsWeightedAccuracyMetric(FlopsMetric):
    """Computes acc * (flops / target_gflops) ** (-0.07)"""

    @classmethod
    def _metrics(cls):
        return {
            **{"flops_weighted_acc": OptimisationDirection.MAX},
            **FlopsMetric._metrics(),
        }

    def validate_attributes(self):
        for hparam in FlopsWeightedAccuracyMetric.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="target_gflops",
            type=float,
            default=2.0,
            strategy="constant",
            description="Target (Giga) Floating Point Operations per Second.",
        )
        return c

    def metrics_step(self, preds: Tensor, targets: Tensor, **kwargs) -> MetricDict:
        acc = topk_accuracies(preds, targets, ks=[1])[0]
        return {
            **FlopsMetric.metrics_step(self, preds, targets, **kwargs),
            "flops_weighted_acc": acc
            * (self.flops * 1e-9 / self.hparams.target_gflops) ** (-0.07),
        }


def topks_correct(preds: Tensor, labels: Tensor, ks: List[int]) -> List[Tensor]:
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"

    max_k = int(preds.shape[-1])

    # Find the top max_k predictions for each sample
    _, top_max_k_inds = torch.topk(preds, max_k, dim=1, largest=True, sorted=True)
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct: List[Tensor] = [
        top_max_k_correct[: min(k, max_k), :].reshape(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds: Tensor, labels: Tensor, ks: List[int]):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) for x in num_topks_correct]


def topk_accuracies(preds: Tensor, labels: Tensor, ks: List[int]):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) for x in num_topks_correct]


def flops(model: torch.nn.Module):
    """Compute the Floating Point Operations per Second for the model"""
    return get_model_complexity_info(
        model,
        model.input_shape,
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True,
    )[0]


def params_count(model: torch.nn.Module):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()])


def make_confusion_matrix(
    preds: Tensor,
    targets: Tensor,
    classes: List[str],
) -> Figure:
    sns.set_theme()
    fig = plt.figure()
    z = (
        confusion_matrix(
            preds.argmax(1), targets, num_classes=len(classes), normalize="true"
        )
        .cpu()
        .numpy()
    )
    ax = sns.heatmap(
        z, annot=len(classes) <= 20, fmt=".2f", vmin=0, vmax=1, cmap="Blues"
    )
    for x, y in zip(ax.get_xticklabels(), ax.get_yticklabels()):
        x.set_text(f"{classes[int(x._text)]} ({x._text})")
        y.set_text(f"{classes[int(y._text)]} ({y._text})")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    return fig
