from operator import attrgetter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ride.core import Configs, RideClassificationDataset
from ride.feature_extraction import FeatureExtractable
from ride.logging import get_log_dir
from ride.metrics import FigureDict
from ride.utils.io import bump_version
from ride.utils.logging import getLogger

logger = getLogger(__name__)


class FeatureVisualisable(FeatureExtractable):
    """Adds feature visualisation capabilities to model"""

    hparams: ...

    @staticmethod
    def configs() -> Configs:
        c = FeatureExtractable.configs()
        c.add(
            name="visualise_features",
            default="",
            type=str,
            choices=["", "umap", "tsne", "pca"],
            description=(
                "Visualise extracted features using selected dimensionality reduction method. "
                "Visualisations are created only during evaluation."
            ),
        )
        return c

    def validate_attributes(self):
        for hparam in FeatureVisualisable.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    def __init__(self, hparams, *args, **kwargs):
        self.dimensionality_reduction = None
        if self.hparams.visualise_features == "umap":
            try:
                from umap import UMAP

                self.dimensionality_reduction = UMAP(n_components=2)
            except ModuleNotFoundError as e:  # pragma: no cover
                logger.error(
                    "To visualise features with UMAP, first install Umap via `pip install umap-learn` or `pip install 'ride[extras]'`"
                )
                raise e

        elif self.hparams.visualise_features == "tsne":
            try:
                from sklearn.manifold import TSNE

                self.dimensionality_reduction = TSNE(n_components=2)
            except ModuleNotFoundError as e:  # pragma: no cover
                logger.error(
                    "To visualise features with TSNE, first install Scikit-learn via `pip install scikit-learn` or `pip install 'ride[extras]'`"
                )
                raise e
        elif self.hparams.visualise_features == "pca":
            try:
                from sklearn.decomposition import PCA

                self.dimensionality_reduction = PCA(n_components=2)
            except ModuleNotFoundError as e:  # pragma: no cover
                logger.error(
                    "To visualise features with PCA, first install Scikit-learn via `pip install scikit-learn` or `pip install 'ride[extras]'`"
                )
                raise e

        if (
            self.dimensionality_reduction
            and not self.hparams.extract_features_after_layer
        ):
            logger.error(
                "Unable to visualise features if no layer is specified using `extract_features_after_layer`."
            )

    def metrics_epoch(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        prefix: str = None,
        *args,
        **kwargs,
    ) -> FigureDict:
        if not hasattr(self, "extracted_features"):
            return {}

        FeatureExtractable.metrics_epoch(
            self, preds, targets, prefix, clear_extracted_features=(prefix != "test")
        )
        if (
            prefix != "test"
            or not self.dimensionality_reduction
            or not len(self.extracted_features) > 0
        ):
            return {}

        # Dimensionality reduction
        try:
            feat = np.stack(self.extracted_features)
            if len(feat.shape) > 2:
                logger.debug(
                    f"🔧 Flattening extracted_features ({feat.shape[0]} -> {np.prod(feat.shape[1:])}) prior to dimensionality reduction."
                )
                feat = feat.reshape(feat.shape[0], -1)

            logger.info(
                f"👁 Performing dimensionality reduction using {self.hparams.visualise_features.upper()}"
            )
            features = self.dimensionality_reduction.fit_transform(feat)

            # Save features
            base_path = Path(get_log_dir(self)) / "features" / (prefix or "")
            base_path.mkdir(parents=True, exist_ok=True)
            base_name = self.hparams.extract_features_after_layer.replace(".", "_")
            save_path = bump_version(
                base_path / f"{base_name}_{self.hparams.visualise_features}.npy"
            )
            logger.info(
                f"💾 Saving {self.hparams.visualise_features.upper()} features to {str(save_path)}"
            )
            np.save(save_path, features)

            # Create scatterplot
            fig = (
                scatter_plot(features, np.array(targets), self.classes)
                if issubclass(type(self), RideClassificationDataset)
                else scatter_plot(features)
            )
            return {
                f"{self.hparams.extract_features_after_layer}_{self.hparams.visualise_features}": fig
            }

        except Exception as e:
            logger.error(f"Caught exception during feature visualisation: {e}")
            return {}


def scatter_plot(
    features: np.array, labels: np.array = None, classes: List[str] = None
):
    sns.set_theme()

    fig = plt.figure(figsize=(6, 6))

    if labels is not None:
        palette = sns.color_palette("hls", n_colors=len(classes))
        legend = True
        plt.legend(loc="center left", bbox_to_anchor=(0.997, 0.5))
    else:
        palette = None
        legend = False

    # Marker size according to number of features (heuristic choice)
    s = max(3, round((20 - len(features) / 150)))

    g = sns.scatterplot(
        x=features[:, 0],
        y=features[:, 1],
        hue=labels,
        linewidth=0,
        alpha=0.7,
        palette=palette,
        s=s,
        legend=legend,
    )

    if legend:
        fig.axes[0].legend(loc="center left", bbox_to_anchor=(0.997, 0.5))
        for t, l in zip(g.legend_.texts, classes):
            t.set_text(l)
    g.set(xticklabels=[])
    g.set(yticklabels=[])
    plt.axis("equal")
    plt.tight_layout()

    return fig
