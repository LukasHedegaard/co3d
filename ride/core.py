# from ride.profile import Profileable
import inspect
from abc import ABC
from typing import Any, List, Sequence, Union

import pytorch_lightning as pl
from corider import Configs as _Configs
from pytorch_lightning.utilities.parsing import AttributeDict
from supers import supers
from torch import Tensor
from torch.utils.data import DataLoader

from ride.utils.logging import getLogger
from ride.utils.utils import (
    DictLike,
    attributedict,
    is_shape,
    merge_attributedicts,
    missing_or_not_in_other,
    name,
    some,
)

logger = getLogger(__name__)


DataShape = Union[int, Sequence[int], Sequence[Sequence[int]]]


class Configs(_Configs):
    """Configs module for holding project configurations."""

    @staticmethod
    def collect(cls: "RideModule") -> "Configs":
        """Collect the configs from all class bases

        Returns:
            Configs: Aggregated configurations
        """
        c: Configs = sum([c.configs() for c in cls.__bases__ if hasattr(c, "configs")])  # type: ignore
        return c

    def default_values(self):
        return attributedict({k: v.default for k, v in self.values.items()})


def _init_subclass(cls):
    # Validate inheritance order
    assert (
        cls.__bases__[0] == RideModule or cls.__bases__[0].__bases__[0] == RideModule
    ), """RideModule must come first in inheritance order, e.g.:
    class YourModule(RideModule, OtherMixin):
        ..."""

    add_bases = []

    # Extend funtionality with additional base-classes
    from ride.feature_visualisation import FeatureVisualisable
    from ride.finetune import Finetunable
    from ride.lifecycle import Lifecycle  # Break cyclical dependencies

    # Ensure pl.LightningModule is the lowest-priority parent
    if not cls.__bases__[-1] == pl.LightningModule:
        add_bases.append(pl.LightningModule)

    if not issubclass(cls, DefaultMethods):
        add_bases.append(DefaultMethods)

    if not issubclass(cls, Lifecycle):
        add_bases.append(Lifecycle)

    if not issubclass(cls, Finetunable):
        add_bases.append(Finetunable)

    if not issubclass(cls, FeatureVisualisable):
        add_bases.append(FeatureVisualisable)

    # Warn if there is no forward
    if missing_or_not_in_other(
        cls, pl.LightningModule, {"forward"}, must_be_callable=True
    ):
        logger.warning(
            f"No `forward` function found in {name(cls)}. Did you forget to define it?"
        )

    # Ensure dataset
    dataset_steps = {"train_dataloader", "val_dataloader", "test_dataloader"}
    missing_dataset_steps = missing_or_not_in_other(
        cls, pl.LightningModule, dataset_steps
    )
    if missing_dataset_steps:
        logger.warning(
            f"No dataloader funcions {missing_dataset_steps} found in {name(cls)}"
        )
        logger.info(
            "🔧 Adding ride.RideDataset automatically and assuming that `self.datamodule`, `self.input_shape`, and `self.output_shape` will be provided by user"
        )
        add_bases.append(RideDataset)

    # Ensure optimizer
    if missing_or_not_in_other(cls, pl.LightningModule, {"configure_optimizers"}):
        logger.info(f"`configure_optimizers` not found in in {name(cls)}")
        logger.info("🔧 Adding ride.SgdOptimizer automatically")

        from ride.optimizers import SgdOptimizer  # Avoid cyclical import

        add_bases.append(SgdOptimizer)

    # Update class bases with pl.LightningModule as lowest rank
    cls.__bases__ = (*cls.__bases__, *add_bases[::-1])

    # Monkeypatch derived module init
    cls._orig_init = cls.__init__

    def init(self, hparams: DictLike = {}, *args, **kwargs):
        pl.LightningModule.__init__(self)
        self.hparams = merge_attributedicts(self.configs().default_values(), hparams)
        supers(self)[1:-1].__init__(self.hparams)
        apply_init_args(cls._orig_init, self, self.hparams, *args, **kwargs)
        supers(self).on_init_end(self.hparams, *args, **kwargs)
        supers(self).validate_attributes()

    cls.__init__ = init

    # Monkeypatch derived module configs
    orig_configs = getattr(cls, "configs", None)

    @staticmethod
    def configs():
        c = Configs.collect(cls)
        if orig_configs:
            c += orig_configs()
        return c

    cls.configs = configs


def apply_init_args(fn, self, hparams, *args, **kwargs):
    spec = inspect.getfullargspec(fn)
    valid_kwargs = (
        kwargs
        if spec.varkw == "kwargs"
        else {k: v for k, v in kwargs.items() if k in spec.args}
    )
    if len(spec.args) == 1:
        return fn(self)
    return fn(self, hparams, *args, **valid_kwargs)


class RideModule:
    """
    Base-class for modules using the Ride ecosystem.

    This module should be inherited as the highest-priority parent (first in sequence).

    Example::

        class MyModule(ride.RideModule, ride.SgdOneCycleOptimizer):
            def __init__(self, hparams):
                ...

    It handles proper initialisation of `RideMixin` parents and adds automatic attribute validation.

    If `pytorch_lightning.LightningModule` is omitted as lowest-priority parent, `RideModule` will automatically add it.

    If `training_step`, `validation_step`, and `test_step` methods are not found, the `ride.Lifecycle` will be automatically mixed in by this module.
    """

    def __init_subclass__(cls):
        # Only initialise immediate children
        if cls.__bases__[0] == RideModule:
            _init_subclass(cls)

    @property
    def hparams(self) -> AttributeDict:
        if not hasattr(self, "_hparams"):
            self._hparams = AttributeDict()
        return self._hparams

    @hparams.setter
    def hparams(self, hp: Union[dict, AttributeDict, Any]):
        # Overload the version in pytorch_lightning core to omit DeprecationWarning
        self._hparams = attributedict(hp)

    @classmethod
    def with_dataset(cls, ds: "RideDataset"):
        new_bases = [b for b in cls.__bases__ if not issubclass(b, RideDataset)]
        old_dataset = [b for b in cls.__bases__ if issubclass(b, RideDataset)]
        assert len(old_dataset) <= 1, "`RideModule` should only have one `RideDataset`"
        if old_dataset and issubclass(old_dataset[0], RideClassificationDataset):
            assert issubclass(
                ds, RideClassificationDataset
            ), "A `RideClassificationDataset` should be replaced by a `RideClassificationDataset`"
        new_bases.insert(-1, ds)
        cls.__init__ = cls._orig_init  # Revert to orig init
        DerivedRideModule = type(
            f"{name(cls)}With{name(ds)}", tuple(new_bases), dict(cls.__dict__)
        )

        return DerivedRideModule


class RideMixin(ABC):
    """Abstract base-class for Ride mixins"""

    def __init__(self, hparams: AttributeDict, *args, **kwargs):
        ...

    def on_init_end(self, hparams: AttributeDict, *args, **kwargs):
        ...

    def validate_attributes(self):
        ...


class DefaultMethods(RideMixin):
    def warm_up(self, input_shape: Sequence[int], *args, **kwargs):
        """Warms up the model state with a dummy input of shape `input_shape`.
        This method is called prior to model profiling.

        Args:
            input_shape (Sequence[int]): input shape with which to warm the model up, including batch size.
        """
        ...


class OptimizerMixin(RideMixin):
    """Abstract base-class for Optimizer mixins"""

    ...


class RideDataset(RideMixin):
    """Base-class for Ride datasets.

    If no dataset is specified otherwise, this mixin is automatically add as a base of RideModule childen.

    User-specified datasets must inherit from this class, and specify the following:
    - `self.input_shape`: Union[int, Sequence[int], Sequence[Sequence[int]]]
    - `self.output_shape`: Union[int, Sequence[int], Sequence[Sequence[int]]]

    and either the functions:
    - `train_dataloader`: Callable[[Any], DataLoader]
    - `val_dataloader`: Callable[[Any], DataLoader]
    - `test_dataloader`: Callable[[Any], DataLoader]

    or:
    - `self.datamodule`, which has `train_dataloader`, `val_dataloader`, and `test_dataloader` attributes.
    """

    input_shape: DataShape
    output_shape: DataShape

    def validate_attributes(self):
        assert is_shape(
            getattr(self, "input_shape", None)
        ), "RideDataset should define an `input_shape` of type int, list, tuple, or namedtuple."
        assert is_shape(
            getattr(self, "output_shape", None)
        ), "RideDataset should define `output_shape` of type int, list, tuple, or namedtuple."

        for n in RideDataset.configs().names:
            assert some(
                self, f"hparams.{n}"
            ), "`self.hparams.{n}` not found in Dataset. Did you forget to include its `configs`?"

    @staticmethod
    def configs() -> Configs:
        c = Configs()
        c.add(
            name="batch_size",
            type=int,
            default=16,
            strategy="constant",
            description="Batch size for dataset.",
        )
        c.add(
            name="num_workers",
            type=int,
            default=0,
            strategy="constant",
            description="Number of workers in dataloader.",
        )
        return c

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader"""
        assert some(
            self, "datamodule.train_dataloader"
        ), f"{name(self)} should either have a `self.datamodule: pl.LightningDataModule` or overload the `train_dataloader` function."
        return self.datamodule.train_dataloader

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader"""
        assert some(
            self, "datamodule.val_dataloader"
        ), f"{name(self)} should either have a `self.datamodule: pl.LightningDataModule` or overload the `val_dataloader` function."
        return self.datamodule.val_dataloader

    def test_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader"""
        assert some(
            self, "datamodule.test_dataloader"
        ), f"{name(self)} should either have a `self.datamodule: pl.LightningDataModule` or overload the `test_dataloader` function."
        return self.datamodule.test_dataloader


class RideClassificationDataset(RideDataset):
    """Base-class for Ride classification datasets.

    If no dataset is specified otherwise, this mixin is automatically add as a base of RideModule childen.

    User-specified datasets must inherit from this class, and specify the following:
    - `self.input_shape`: Union[int, Sequence[int], Sequence[Sequence[int]]]
    - `self.output_shape`: Union[int, Sequence[int], Sequence[Sequence[int]]]
    - `self.classes`: List[str]

    and either the functions:
    - `train_dataloader`: Callable[[Any], DataLoader]
    - `val_dataloader`: Callable[[Any], DataLoader]
    - `test_dataloader`: Callable[[Any], DataLoader]

    or:
    - `self.datamodule`, which has `train_dataloader`, `val_dataloader`, and `test_dataloader` attributes.
    """

    classes: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @staticmethod
    def configs() -> Configs:
        c = RideDataset.configs()
        c.add(
            name="test_confusion_matrix",
            type=int,
            default=0,
            choices=[0, 1],
            strategy="constant",
            description="Create and save confusion matrix for test data.",
        )
        return c

    def validate_attributes(self):
        RideDataset.validate_attributes(self)
        assert type(getattr(self, "classes", None)) in {
            list,
            tuple,
        }, "Ride RideClassificationDataset should define `classes` but none was found."

    def metrics_epoch(
        self,
        preds: Tensor,
        targets: Tensor,
        prefix: str = None,
        *args,
        **kwargs,
    ):  # -> "FigureDict":
        if prefix != "test" or not self.hparams.test_confusion_matrix:
            return {}
        from ride.metrics import make_confusion_matrix

        fig = make_confusion_matrix(preds, targets, self.classes)
        return {"confusion_matrix": fig}
