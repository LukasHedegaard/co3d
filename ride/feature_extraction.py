from operator import attrgetter
from pathlib import Path

import numpy
import torch

from ride.core import Configs, RideMixin
from ride.logging import get_log_dir
from ride.metrics import MetricDict
from ride.utils.io import bump_version
from ride.utils.logging import getLogger
from ride.utils.utils import rgetattr

logger = getLogger(__name__)


class FeatureExtractable(RideMixin):
    """Adds feature extraction capabilities to model"""

    hparams: ...

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="extract_features_after_layer",
            default="",
            type=str,
            description=(
                "Layer name after which to extract features. "
                "Nested layers may be selected using dot-notation, "
                "e.g. `block.subblock.layer1`"
            ),
        )
        return c

    def validate_attributes(self):
        for hparam in FeatureExtractable.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    def on_init_end(self, hparams, *args, **kwargs):
        if not self.hparams.extract_features_after_layer:
            return

        available_layers = [k for k, _ in self.named_modules() if k != ""]

        assert self.hparams.extract_features_after_layer in available_layers, (
            f"Invalid `extract_features_after_layer` ({self.hparams.extract_features_after_layer}). "
            f"Available layers are: {available_layers}"
        )

        layer = rgetattr(self, self.hparams.extract_features_after_layer)

        self.extracted_features = []

        def store_features(sself, input, output):
            nonlocal self
            for o in output:
                self.extracted_features.append(o.detach().cpu().numpy())

        layer.register_forward_hook(store_features)

    def metrics_epoch(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        prefix: str = None,
        clear_extracted_features=True,
        *args,
        **kwargs,
    ) -> MetricDict:
        if not hasattr(self, "extracted_features"):
            return {}

        # Save
        save_path = bump_version(
            Path(get_log_dir(self))
            / "features"
            / (prefix or "")
            / f"{self.hparams.extract_features_after_layer.replace('.','_')}.npy"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"💾 Saving extracted features to {str(save_path)}")
        numpy.save(save_path, self.extracted_features)

        if clear_extracted_features and prefix != "test":
            self.extracted_features = []

        return {}
