import typing
from argparse import ArgumentParser, _ArgumentGroup, _StoreAction
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Type, TypeVar, Union
from warnings import warn

from .utils import load_structured_data


class Strategy(Enum):
    CONSTANT = "constant"  # default, not inteded for hparamsearch
    CHOICE = "choice"
    UNIFORM = "uniform"
    LOGUNIFORM = "loguniform"


T = TypeVar("T")


@dataclass
class Config:
    name: str
    alias: Optional[str]
    type: Type[T]
    default: T
    choices: Optional[Sequence[T]]
    strategy: Strategy
    description: str


class Configs:
    def __init__(self):
        self.values = {}

    def __add__(self, other: "Configs") -> "Configs":
        c = Configs()
        c.values = {**self.values, **other.values}
        return c

    def __radd__(self, other: "Configs") -> "Configs":
        if other == 0:
            return self
        return self.__add__(other)

    def __iadd__(self, other: "Configs") -> "Configs":
        return self.__radd__(other)

    def __sub__(self, other: "Configs") -> "Configs":
        c = Configs()
        other_names = set(other.names)
        for name, val in self.values.items():
            if name not in other_names:
                c.values[name] = val
        return c

    def __repr__(self):
        return repr(self.values)

    @property
    def names(self):
        return list(self.values.keys())

    def add(
        self,
        name: str,
        type: Type[T],
        default: T,
        strategy: Union[Strategy, str] = "constant",
        description: str = "",
        choices: Sequence[T] = None,
        alias: str = None,
    ) -> "Configs":
        if name in self.values:
            warn(f"Configs already include {name} (old value will be overridden).")

        self.values[name] = Config(
            name,
            alias,
            type,
            default,
            list(choices) if choices else None,
            Strategy(strategy),
            description,
        )

        return self

    @typing.overload
    def add_argparse_args(self, parser: ArgumentParser) -> ArgumentParser:
        ...

    @typing.overload
    def add_argparse_args(self, parser: _ArgumentGroup) -> _ArgumentGroup:
        ...

    def add_argparse_args(self, parser: ArgumentParser) -> ArgumentParser:
        # parser = ArgumentParser(parents=[parser], add_help=False)
        for h in self.values.values():
            self._add_argparse_arg(parser, h)
        return parser

    def _add_argparse_arg(
        self, parser: ArgumentParser, config: "Config"
    ) -> ArgumentParser:
        names = [f"--{config.name}"]
        if config.alias:
            names.append(f"--{config.alias}")

        parser.add_argument(
            *names,
            type=config.type,
            default=config.default,
            help=f"{config.description} (Default: {config.default})",
            choices=(
                config.choices
                if config.strategy in {Strategy.CHOICE, Strategy.CONSTANT}
                else None  # type:ignore
            ),
        )
        return parser

    def tune_config(self):
        from ray import tune  # type: ignore

        def map_space(h: Config):
            if h.choices is None:
                return
            if h.strategy == Strategy.CHOICE:
                return tune.choice(h.choices)
            elif h.strategy == Strategy.UNIFORM:
                return tune.uniform(*h.choices)
            elif h.strategy == Strategy.LOGUNIFORM:
                return tune.loguniform(*h.choices)
            else:
                # Strategy.CONSTANT: h.default, # Should be passed as argument instead
                raise ValueError(
                    f"hyperparameter strategy should be one of {[Strategy.CHOICE, Strategy.UNIFORM, Strategy.LOGUNIFORM]}"
                )

        config = {
            n: map_space(h)
            for n, h in self.values.items()
            if h.strategy != Strategy.CONSTANT
        }
        return config

    def default_values(self):
        return {k: v.default for k, v in self.values.items()}

    def add_tune_argparse_args(self, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        for h in self.values.values():
            if h.strategy == Strategy.CONSTANT:
                self._add_argparse_arg(parser, h)

        return parser

    @staticmethod
    def from_argument_parser(parser: ArgumentParser) -> "Configs":
        c = Configs()

        for action in parser._actions:
            if type(action) == _StoreAction:
                try:
                    c.add(
                        name=action.dest,  # type: ignore
                        type=action.type,  # type: ignore
                        default=action.default,  # type: ignore
                        choices=action.choices,  # type: ignore
                        description=action.help,  # type: ignore
                        strategy=Strategy.CONSTANT,
                    )
                except Exception:
                    pass

        return c

    @staticmethod
    def from_file(path: Union[str, Path]) -> "Configs":
        path = Path(path)
        assert path.exists()
        d = load_structured_data(path)
        assert type(d) == dict
        type_dict = {
            "int": int,
            "float": float,
            "str": str,
        }
        c = Configs()
        for k, v in d.items():
            c.add(
                name=k,
                type=type_dict[v["type"]] if "type" in v else str,
                default=v["default"] if "default" in v else None,
                strategy=v["strategy"] if "strategy" in v else "constant",
                description=v["description"] if "description" in v else "",
                choices=v["choices"] if "choices" in v else None,
                alias=None,
            )
        return c
