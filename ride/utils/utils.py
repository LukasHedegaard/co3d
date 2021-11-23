import functools
import inspect
import math
import re
from argparse import Namespace
from contextlib import contextmanager
from functools import wraps
from operator import attrgetter
from typing import Any, Callable, Collection, Dict, Set, Union

from pytorch_lightning.utilities.parsing import AttributeDict

DictLike = Union[AttributeDict, Dict[str, Any], Namespace]


def is_shape(x: Any):
    """Tests whether `x` is a shape, i.e. one of
    - int
    - List[int]
    - Tuple[int]
    - Namedtuple[int]

    Args:
        x (Any): instance to check
    """
    Type = type(x)
    if Type == int:
        return True
    if not (Type in {list, tuple} or issubclass(Type, tuple)):
        return False
    return all(type(y) == int for y in x)


def once(fn: Callable):
    mem = set()

    @wraps(fn)
    def wrapped(*args, **kwargs):
        h = hash((args, str(kwargs)))
        if h in mem:
            return
        mem.add(h)
        return fn(*args, **kwargs)

    return wrapped


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def attributedict(dict_like: DictLike) -> AttributeDict:
    """If given a dict, it is converted it to an argparse.AttributeDict. Otherwise, no change is made"""
    if isinstance(dict_like, AttributeDict):
        return dict_like
    elif isinstance(dict_like, Namespace):
        return AttributeDict(vars(dict_like))
    elif isinstance(dict_like, dict):
        return AttributeDict(**dict_like)

    raise ValueError(f"Unable to convert type {type(dict_like)} to AttributeDict")


def to_dict(d):
    if type(d) == Namespace:
        return vars(d)
    return dict(d)


def merge_dicts(*args):
    if len(args) == 0:
        return {}
    if len(args) == 1:
        return args[0]
    acc = to_dict(args[0])
    for a in args[1:]:
        acc = {**acc, **to_dict(a)}
    return acc


def merge_attributedicts(*args):
    return attributedict(merge_dicts(*args))


def some(self, attr: str):
    try:
        a = attrgetter(attr)(self)
        return a is not None
    except Exception:
        return False


def some_callable(self, attr: str, min_num_args=0, max_num_args=math.inf):
    try:
        fn = attrgetter(attr)(self)
        if not callable(fn):
            return False
        num_args = len(inspect.getfullargspec(fn).args)
        return min_num_args <= num_args and num_args <= max_num_args
    except Exception:
        return False


def get(self, attr: str):
    try:
        a = attrgetter(attr)(self)
        return a
    except KeyError:
        return None


def differ_and_exist(a, b):
    return a is b and a is not None


def missing(self, attrs: Collection[str]) -> Set[str]:
    return {a for a in attrs if not some(self, a)}


def missing_or_not_in_other(
    first, other, attrs: Collection[str], must_be_callable=False
) -> Set[str]:
    some_ = some_callable if must_be_callable else some
    return {
        a
        for a in attrs
        if not some_(first, a) or differ_and_exist(get(first, a), get(other, a))
    }


def name(thing):
    if isinstance(thing, str):
        return thing
    elif hasattr(thing, "__name__"):
        return thing.__name__
    return thing.__class__.__name__


def prefix_keys(prefix: str, dictionary: Dict) -> Dict:
    return {f"{prefix}{k}": v for k, v in dictionary.items()}


def camel_to_snake(s: str) -> str:
    """Convert from camel-case to snake-case
    Source: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


@contextmanager
def temporary_parameter(obj, attr, val):
    prev_val = rgetattr(obj, attr)
    rsetattr(obj, attr, val)
    yield obj
    rsetattr(obj, attr, prev_val)
