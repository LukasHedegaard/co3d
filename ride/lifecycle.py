import collections
from itertools import groupby
from operator import attrgetter
from typing import Any, Callable, Dict, Sequence, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from supers import supers
from torch import Tensor

from ride.core import Configs
from ride.logging import log_figures
from ride.metrics import (
    MetricDict,
    MetricMixin,
    OptimisationDirection,
    sort_out_figures,
)
from ride.utils.gpus import parse_num_gpus
from ride.utils.logging import getLogger
from ride.utils.utils import some_callable

loss_names = {
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "poisson_nll_loss",
    # cosine_embedding_loss(input1, input2, target)
    "cross_entropy",
    # ctc_loss(log_probs, targets, input_lengths, target_lengths)
    "hinge_embedding_loss",
    "kl_div",
    "l1_loss",
    "mse_loss",
    # margin_ranking_loss(input1, input2, target)
    "multilabel_margin_loss",
    "multilabel_soft_margin_loss",
    "multi_margin_loss",
    "nll_loss",
    "smooth_l1_loss",
    "soft_margin_loss",
    # triplet_margin_loss(anchor, positive, negative)
    # triplet_margin_with_distance_loss(anchor, positive, negative)
}

logger = getLogger(__name__)


class Lifecycle(MetricMixin):
    """Adds train, val, and test lifecycle methods with cross_entropy loss

    During its `traning_epoch_end(epoch)` lifecycle method,
    it will call `on_traning_epoch_end` for all superclasses of
    its child class
    """

    hparams: ...
    forward: Callable[[torch.Tensor], torch.Tensor]
    _epoch: int

    @classmethod
    def _metrics(cls):
        return {"loss": OptimisationDirection.MIN}

    def __init__(self, hparams=None, *args, **kwargs):
        self._epoch = 0
        self.hparams.num_gpus = parse_num_gpus(hparams.gpus)
        self._sync_dist = self.hparams.num_gpus > 1
        if not some_callable(self, "loss", min_num_args=2):
            self.loss = attrgetter(hparams.loss)(F)

    def validate_attributes(self):
        for attribute in [
            "forward",
            "training_step",
            "validation_step",
            "test_step",
        ]:
            attrgetter(attribute)(self)

        for hparam in Lifecycle.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="optimization_metric",
            default="loss",
            type=str,
            choices=MetricMixin.metric_names(),
            description="Name of the performance metric that should be optimized",
        )
        c.add(
            name="test_ensemble",
            type=int,
            default=0,
            strategy="constant",
            description="Flag indicating whether the test dataset should yield a clip ensemble.",
        )
        c.add(
            name="gpus",
            type=str,
            default=None,
            strategy="constant",
            description="Which gpus should be used. Can be either the number of gpus (e.g. '2') or a list of gpus (e.g. ('0,1').",
        )
        c.add(
            name="loss",
            type=str,
            default="cross_entropy",
            choices=loss_names,
            strategy="constant",
            description="Loss function used during optimisation.",
        )
        return c

    def metrics_step(
        self, preds: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> MetricDict:
        return {"loss": self.loss(preds, targets)}

    def common_step(self, pred, target, prefix="train/", log=False):
        opt_key = prefix + self.hparams.optimization_metric
        loss_key = prefix + "loss"
        metrics = prefix_keys(prefix, self.collect_metrics(pred, target))

        if log:
            LightningModule.log(
                self,
                name=f"step_{loss_key}",
                value=metrics[loss_key],
                prog_bar=True,
                logger=True,
                sync_dist=self._sync_dist,
            )
            if opt_key != loss_key and opt_key in metrics:
                LightningModule.log(
                    self,
                    name=f"step_{opt_key}",
                    value=metrics[opt_key],
                    prog_bar=True,
                    logger=True,
                    sync_dist=self._sync_dist,
                )
        return {
            "loss": metrics[loss_key],
            **detach_to_cpu(metrics),
            "pred": detach_to_cpu(pred),
            "target": detach_to_cpu(target),
        }

    def common_epoch_end(
        self, step_outputs, prefix="train/", exclude_keys={"pred", "target"}
    ):
        keys = list(step_outputs[0].keys())
        mean_step_metrics = {
            k: torch.mean(torch.stack([x[k] for x in step_outputs]))
            for k in keys
            if k not in exclude_keys
        }
        preds, targets = zip(*[(s["pred"], s["target"]) for s in step_outputs])
        preds = cat_steps(preds)
        targets = cat_steps(targets)
        epoch_metrics = prefix_keys(
            prefix,
            self.collect_epoch_metrics(preds, targets, prefix.replace("/", "")),
        )
        epoch_metrics, epoch_figures = sort_out_figures(epoch_metrics)
        all_metrics = {**mean_step_metrics, **epoch_metrics}
        LightningModule.log_dict(self, all_metrics, sync_dist=self._sync_dist)
        log_figures(self, epoch_figures)

    def preprocess_batch(self, batch):
        return batch

    def training_step(self, batch, batch_idx=None):
        if batch_idx == 0:
            supers(self).on_traning_epoch_start(self._epoch)
        batch = self.preprocess_batch(batch)
        x, target = batch[0], batch[1]
        pred = self.forward(x)
        return self.common_step(pred, target, prefix="train/", log=True)

    def training_epoch_end(self, step_outputs):
        self._epoch += 1
        self.common_epoch_end(step_outputs, prefix="train/")

    def validation_step(self, batch, batch_idx=None):
        batch = self.preprocess_batch(batch)
        x, target = batch[0], batch[1]
        pred = self.forward(x)
        return self.common_step(pred, target, prefix="val/")

    def validation_epoch_end(self, step_outputs):
        self.common_epoch_end(step_outputs, prefix="val/")

    def test_step(self, batch, batch_idx=None):
        if batch is None:
            return None

        batch = self.preprocess_batch(batch)
        x, target = batch[0], batch[1]
        pred = self.forward(x)

        if not self.hparams.test_ensemble:
            return {
                **self.common_step(pred, target, prefix="test/"),
                "pred": detach_to_cpu(pred),
                "target": detach_to_cpu(target),
            }

        identifier = batch[-1]
        # Delay computation of metrics to epoch end
        return {"pred": pred, "target": target, "identifier": identifier}

    def test_epoch_end(self, step_outputs):
        if self.hparams.test_ensemble:
            keys = set(step_outputs[0].keys())
            assert keys == {"pred", "target", "identifier"}

            # Unbatching step_outputs
            step_outputs = [
                {k: batch_step[k][i] for k in keys}
                for batch_step in step_outputs
                for i in range(len(batch_step["identifier"]))
            ]

            # Grouping samples by identifier
            steps_by_key = [
                (i, list(group))
                for i, group in groupby(
                    step_outputs, key=lambda x: int(x["identifier"])
                )
            ]

            # Averaging predictions
            preds_and_targets = [
                (
                    torch.mean(
                        torch.stack([s["pred"] for s in steps]), dim=0
                    ).unsqueeze(0),
                    steps[0]["target"].unsqueeze(0),  # unqueeze to add batch dim
                )
                for _, steps in steps_by_key
            ]

            # Computing steps
            step_outputs = [
                {
                    **self.common_step(pred, target, prefix="test/"),
                    "pred": pred,
                    "target": target,
                }
                for pred, target in preds_and_targets
            ]

        self.common_epoch_end(step_outputs, prefix="test/")


def prefix_keys(prefix: str, dictionary: Dict) -> Dict:
    return {f"{prefix}{k}": v for k, v in dictionary.items()}


def detach_to_cpu(x: Union[Tensor, Sequence[Tensor], Dict[Any, Tensor]]):
    if isinstance(x, Tensor):
        return x.detach().cpu()
    if isinstance(x, collections.abc.Sequence):
        return [detach_to_cpu(t) for t in x]
    if isinstance(x, dict):
        return {k: detach_to_cpu(v) for k, v in x.items()}
    return x


def cat_steps(steps: Sequence[Union[Tensor, Sequence[Tensor], Dict[Any, Tensor]]]):
    if len(steps) == 0:
        return steps
    step = steps[0]
    if isinstance(step, Tensor):
        return torch.cat(steps)
    if isinstance(step, collections.abc.Sequence):
        return [cat_steps([s[i] for s in steps]) for i in range(len(step))]
    if isinstance(step, dict):
        return {k: cat_steps([s[k] for s in steps]) for k in step.keys()}
    raise ValueError("Steps should contain either a Tensor of Dict")
