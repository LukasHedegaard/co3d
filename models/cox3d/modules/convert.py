from copy import deepcopy

import torch
from ride.utils.logging import getLogger
from torch.nn import (
    AdaptiveAvgPool3d,
    AdaptiveMaxPool3d,
    AvgPool3d,
    BatchNorm3d,
    Conv3d,
    Linear,
    MaxPool3d,
    Module,
)
from torch.nn.modules.activation import ReLU, Sigmoid, Softmax

from .activation import Swish, convert_softmax
from .batchnorm import convert_batchnorm3d
from .conv import convert_conv3d
from .pooling import (
    convert_adaptiveavgpool3d,
    convert_adaptivemaxpool3d,
    convert_avgpool3d,
    convert_maxpool3d,
)
from .utils import FillMode

logger = getLogger(__name__)


CONVERT = {
    BatchNorm3d: convert_batchnorm3d,
    Conv3d: convert_conv3d,
    AdaptiveAvgPool3d: convert_adaptiveavgpool3d,
    AvgPool3d: convert_avgpool3d,
    MaxPool3d: convert_maxpool3d,
    AdaptiveMaxPool3d: convert_adaptivemaxpool3d,
    Softmax: convert_softmax,
}

SKIP = {ReLU, Sigmoid, Swish, Linear}


def convert_simple_module(
    instance: Module, window_size: int, temporal_fill: FillMode = "replicate"
):
    t = type(instance)
    if t in CONVERT.keys():
        return CONVERT[t](instance, window_size, temporal_fill)

    if t in SKIP:
        return instance

    logger.warning(
        f"No rule for converting {t.__name__}. This may lead to unexpected behavior. Leaving module as is."
    )
    return instance


def convert_module(
    instance: Module, window_size: int, temporal_fill: FillMode = "replicate"
):
    with torch.no_grad():
        copied = deepcopy(instance)
        if not list(copied.children()):
            return convert_simple_module(copied, window_size, temporal_fill)
        attrs = dir(copied)
        for attr in attrs:
            if isinstance(getattr(copied, attr), Module):
                setattr(
                    copied,
                    attr,
                    convert_module(getattr(copied, attr), window_size, temporal_fill),
                )
    return copied


# # Remnant of a previous approach, trying to modify the source-code for a Module
# import inspect
# import re
# from types import ModuleType, Type

# def from_class(Class: Type[Module]):
#     module = ModuleType("from_class")
#     mod_init_code = (
#         deindent(inspect.getsource(Class.__init__), tab="    ")
#         .replace("def __init__", "def _rinit")
#         .replace(f"super({Class.__name__}", "super(Class")
#     )

#     compiled = compile(mod_init_code, "", "exec")
#     exec(compiled, module.__dict__)

#     class RecursiveModule(Class):
#         __init__ = module._rinit

#         # def forward(self, input: Tensor) -> Tensor:
#         #     ...

#     return RecursiveModule


# def deindent(input, tab: str):
#     return re.sub(f"({tab})(?!{tab})", "", input)


# def indent(input, tab: str):
#     output = re.sub(r"().(\n)", "\n{tab}", input)
#     output = tab + output
#     return output


# def replace_super(input, name: str, new_name):
#     return re.sub(
#         r"(super\(NAME[, self]*\))".replace("NAME", name),
#         "super(self.__class__, self)",
#         input,
#     )
