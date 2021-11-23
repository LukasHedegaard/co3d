import pickle
import re
from argparse import ArgumentError
from operator import attrgetter
from pathlib import Path

import pytorch_lightning as pl
import torch

from ride.core import Configs
from ride.unfreeze import Unfreezable
from ride.utils.logging import getLogger
from ride.utils.utils import attributedict, to_dict

logger = getLogger(__name__)


class Finetunable(Unfreezable):
    """Adds finetune capabilities to model

    This module is automatically added when RideModule is inherited
    """

    hparams: ...

    @staticmethod
    def configs() -> Configs:
        c: Configs = Unfreezable.configs()
        c.add(
            name="finetune_from_weights",
            default="",
            type=str,
            description=(
                "Path to weights to finetune from. "
                "Allowed extension include {'.ckpt', '.pyth', '.pth', '.pkl', '.pickle'}."
            ),
        )
        c.add(
            name="finetune_from_weights_key",
            default="",
            type=str,
            description="Key in weights-file, which should contains model state_dict in case of '.pyth' or '.pth' files.",
        )
        c.add(
            name="finetune_params_skip",
            default="",
            type=str,
            description=(
                "Regex for matching parameter names between fintune source and target model."
                "The parameter is not copied if `finetune_params_skip` is in paramer name."
            ),
        )
        return c

    def validate_attributes(self):
        for hparam in Finetunable.configs().names:
            attrgetter(f"hparams.{hparam}")(self)
        Unfreezable.validate_attributes(self)

    def map_loaded_weights(self, file, loaded_state_dict):
        return loaded_state_dict

    def on_init_end(self, hparams, *args, **kwargs):
        self.hparams.finetune_params_skip = (
            f".*({self.hparams.finetune_params_skip}).*"
            if self.hparams.finetune_params_skip
            else None
        )

        if not self.hparams.finetune_from_weights:
            Unfreezable.on_init_end(self, hparams, *args, **kwargs)
            return

        # Load model
        new_model_state = self.state_dict()  # type: ignore

        # Load hparams
        default_ft_hparams = to_dict(Finetunable.configs().default_values())
        hparams_passed = attributedict(
            {
                k: (default_ft_hparams[k] if k in default_ft_hparams else v)
                for k, v in self.hparams.items()
            }
        )

        state_dict = load_model_weights(
            self.hparams.finetune_from_weights,
            hparams_passed,
            self.hparams.finetune_from_weights_key,
        )
        state_dict = self.map_loaded_weights(
            self.hparams.finetune_from_weights, state_dict
        )

        def key_ok(k):
            return (
                k in new_model_state
                and k in state_dict
                and not (
                    self.hparams.finetune_params_skip
                    and re.match(self.hparams.finetune_params_skip, k)
                )
            )

        def size_ok(k):
            return new_model_state[k].size() == state_dict[k].size()

        to_load = {k: v for k, v in state_dict.items() if key_ok(k) and size_ok(k)}

        self.load_state_dict(to_load, strict=False)  # type: ignore

        names_missing = set(new_model_state.keys()) - set(to_load.keys())
        names_missing = {n for n in names_missing if "num_batches_tracked" not in n}

        if names_missing:
            logger.debug(f"missing keys: {sorted(names_missing)}")

        names_unexpected = set(state_dict.keys()) - set(new_model_state.keys())
        names_unexpected = {
            n for n in names_unexpected if "num_batches_tracked" not in n
        }

        if names_unexpected:
            logger.debug(f"unexpected keys: {sorted(names_unexpected)}")

        Unfreezable.on_init_end(
            self, hparams, names_to_unfreeze=names_missing, *args, **kwargs
        )


def load_model_weights(file: str, hparams_passed, model_state_key):
    path = Path(file)
    suffix = path.suffix
    assert (
        path.exists()
    ), f"Unable to load model weights from non-existing file ({file})"
    logger.info(f"Loading model weights from {path}")

    if suffix in {".ckpt"}:
        return pl.utilities.cloud_io.load(
            file, map_location=lambda storage, loc: storage
        )["state_dict"]
    elif suffix in {".pyth", ".pth", ".pt"}:
        return try_pyth_load(file, model_state_key)
    elif suffix in {".pkl", ".pickle"}:
        return try_pickle_load(file)
    else:
        raise ArgumentError(
            f"Unable to load model weights with suffix '{suffix}'. Suffix must be one of {'.ckpt', '.pyth', '.pth', '.pt', '.pkl', '.pickle'}"
        )


def try_pyth_load(file, model_state_key):
    loaded = torch.load(file, map_location="cpu")
    if issubclass(type(loaded), torch.nn.Module):
        return loaded.state_dict()

    assert issubclass(
        type(loaded), dict
    ), "pyth checkpoint should either be a model or a dict of weights"

    guesses = [
        model_state_key,
        "state_dict",
        "model_state",
    ]
    for g in guesses:
        if g in loaded.keys():
            state_dict = loaded[g]
            break

    # Check if we already have a state_dict
    if all([isinstance(v, torch.Tensor) for v in loaded.values()]):
        state_dict = loaded

    if not state_dict:  # pragma: no cover
        raise KeyError(
            f"None of the tried keys {guesses} fits loaded model state {loaded.keys()}. You can try another key using the `finetune_from_weights_key` hparam."
        )

    return state_dict


def try_pickle_load(file):
    with open(file, "r") as f:
        try:
            return pickle.load(file)
        except Exception:
            pass

        try:
            return pickle.load(f, encoding="latin1")
        except Exception:
            pass

    with open(file, "rb") as f:
        try:
            return pickle.load(file)
        except Exception:
            pass

        try:
            return pickle.load(f, encoding="latin1")
        except Exception:  # pragma: no cover
            pass

    raise ValueError(f"Unable to load file {file}")  # pragma: no cover
