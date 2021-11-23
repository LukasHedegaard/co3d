from operator import attrgetter
from typing import Dict, Sequence, Tuple

import torch

from ride.core import Configs, RideMixin
from ride.utils.logging import getLogger

logger = getLogger(__name__)


class Unfreezable(RideMixin):
    hparams: ...

    def validate_attributes(self):
        for hparam in self.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = Configs()

        c.add(
            name="unfreeze_from_epoch",
            type=int,
            default=-1,
            description="Number of epochs to wait before starting gradual unfreeze. If -1, unfreeze is omitted.",
        )
        c.add(
            name="unfreeze_layers_must_include",
            type=str,
            default="",
            description="String that must be contained in layer names which should be unfrozen. If empty, this feature is disabled.",
        )
        c.add(
            name="unfreeze_epoch_step",
            type=int,
            default=1,
            description="Number of epochs to train before next unfreeze.",
        )
        c.add(
            name="unfreeze_layers_initial",
            type=int,
            default=1,
            strategy="choice",
            description="Number layers to unfreeze initially. If `-1`, it will be equal to total_layers",
        )
        c.add(
            name="unfreeze_layer_step",
            type=int,
            default=1,
            description="Number additional layers to unfreeze at each unfreeze step. If `-1`, all layers are unfrozon after a step",
        )
        c.add(
            name="unfreeze_layers_max",
            type=int,
            default=-1,
            description="Maximum number of layers to unfreeze. If `-1`, it will be equal to total_layers",
        )
        return c

    def on_init_end(
        self,
        hparams,
        layers_to_unfreeze: Sequence[Tuple[str, torch.nn.Module]] = None,
        names_to_unfreeze: Sequence[str] = None,
        *args,
        **kwargs,
    ):
        self.layers_to_unfreeze = (
            layers_to_unfreeze
            if layers_to_unfreeze is not None
            else get_modules_to_unfreeze(
                self, self.hparams.unfreeze_layers_must_include
            )
        )

        self.unfreeze_schedule = {}

        if self.hparams.unfreeze_from_epoch > -1:
            # Gradual unfreeze linear schedule
            self.unfreeze_schedule = linear_unfreeze_schedule(
                initial_epoch=self.hparams.unfreeze_from_epoch,
                total_layers=len(self.layers_to_unfreeze),
                step_size=self.hparams.unfreeze_layer_step,
                init_layers=self.hparams.unfreeze_layers_initial,
                max_layers=self.hparams.unfreeze_layers_max,
                epoch_step=self.hparams.unfreeze_epoch_step,
            )
            freeze_layers_except_names(self, names_to_unfreeze)

    def on_traning_epoch_start(self, epoch: int):
        # Called by TrainValTestStepsMixin
        if epoch in self.unfreeze_schedule:
            num_layers = self.unfreeze_schedule[epoch]
            logger.info(f"Epoch {epoch}, unfreezing {num_layers} layer(s)")
            unfrozen_layers = unfreeze_from_end(self.layers_to_unfreeze, num_layers)
            logger.debug(f"Unfrozen layers: {', '.join(unfrozen_layers)}")


def freeze_layers_except_names(
    parent_module: torch.nn.Module, names_to_unfreeze: Sequence[str]
):
    names_to_unfreeze = names_to_unfreeze or []
    names_to_unfreeze = [
        n.replace(".weight", "").replace(".bias", "") for n in names_to_unfreeze
    ]
    unfrozen_names = []
    for (name, layer) in parent_module.named_modules():
        requires_grad = any(n in name for n in names_to_unfreeze)
        if requires_grad:
            unfrozen_names.append(name)
        for param in layer.parameters():
            param.requires_grad = requires_grad

    if unfrozen_names:
        logger.debug(f"Unfrozen layers: {', '.join(unfrozen_names)}")


def get_modules_to_unfreeze(
    parent_module: torch.nn.Module, name_must_include=""
) -> Sequence[Tuple[str, torch.nn.Module]]:
    return [
        (n, m)
        for (n, m) in parent_module.named_modules()
        if (name_must_include in n) and (hasattr(m, "weight") or hasattr(m, "bias"))
    ]


def unfreeze_from_end(
    layers: Sequence[Tuple[str, torch.nn.Module]],
    num_layers_from_end: int,
    freeze_others=False,
):
    if freeze_others:
        for _, layer in layers[:-num_layers_from_end:]:
            for param in layer.parameters():
                param.requires_grad = False

    unfrozen_names = []
    layer_iter = iter(layers[::-1])
    while len(unfrozen_names) < num_layers_from_end:
        try:
            name, layer = next(layer_iter)
            for param in layer.parameters():
                param.requires_grad = True
            unfrozen_names.append(name)
        except StopIteration:
            break
    return unfrozen_names


def linear_unfreeze_schedule(
    initial_epoch: int,
    total_layers: int,
    step_size: int = 1,
    init_layers: int = 0,
    max_layers: int = -1,
    epoch_step: int = 1,
) -> Dict[int, int]:
    if init_layers == -1:
        init_layers = total_layers
    if max_layers == -1:
        max_layers = total_layers
    if step_size == -1:
        step_size = total_layers - init_layers
    return {
        **{
            epoch * epoch_step + initial_epoch: i * step_size + init_layers
            for i, epoch in enumerate(range(total_layers * step_size))
            if i * step_size + init_layers <= max_layers
        },
        **{0: init_layers},
    }
