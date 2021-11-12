from ride.metrics import MetricDict, MetricMixin, OptimisationDirection
from torch import Tensor


class AvaMetric(MetricMixin):

    # def validate_attributes(self):
    #     for attribute in ["hparams.loss", "classes"]:
    #         attrgetter(attribute)(self)

    @classmethod
    def _metrics(cls):
        return {"ava": OptimisationDirection.MAX}

    # def metrics_step(
    #     self, preds: Tensor, targets: Tensor, *args, **kwargs
    # ) -> MetricDict:
    #     return {"ava": self._compute_mean_average_precision(preds, targets)}

    # def metrics_epoch(
    #     self, preds: Tensor, targets: Tensor, *args, **kwargs
    # ) -> MetricDict:
    #     return {"ava": self._compute_mean_average_precision(preds, targets)}
