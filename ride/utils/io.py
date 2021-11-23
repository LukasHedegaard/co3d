import json
import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml
from torch import Tensor


def is_nonempty_file(path: Union[str, Path]) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def bump_version(path: Union[str, Path]) -> Path:
    """Bumps the version number for a path if it already exists

    Example::

        bump_version("folder/new_file.json") == Path("folder/new_file.json)
        bump_version("folder/old_file.json") == Path("folder/old_file_1.json)
        bump_version("folder/old_file_1.json") == Path("folder/old_file_2.json)
    """
    path = Path(path)
    if not path.exists():
        return path

    # Check for already bumped versions
    prev_version = None
    try:
        prev_version = max(
            map(
                int,
                filter(
                    lambda s: s.isdigit(),
                    [f.stem.split("_")[-1] for f in path.parent.glob(f"{path.stem}*")],
                ),
            )
        )
        new_version = prev_version + 1
    except ValueError:  # max() arg is an empty sequence
        new_version = 1

    if prev_version and path.stem.endswith(f"_{prev_version}"):
        suffix = f"_{prev_version}"
        new_name = f"{path.stem[:-len(suffix)]}_{new_version}{path.suffix}"
    else:
        new_name = f"{path.stem}_{new_version}{path.suffix}"
    return path.parent / new_name


def load_structured_data(path: Path):
    suffix = path.suffix
    assert suffix in {
        ".json",
        ".yml",
        ".yaml",
    }, f"The supplied file ({str(path)}) should be of type 'json' or 'yaml"
    if suffix == ".yml":
        suffix = ".yaml"
    d = {".json": load_json, ".yaml": load_yaml}[suffix](path)
    return d


def dump_yaml(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        yaml.dump(data, f, sort_keys=True)


def load_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def dump_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(data, f, cls=NpJsonEncoder, sort_keys=True, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        data = json.load(f)
    return data


class NpJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  # type:ignore
            return int(obj)
        elif isinstance(obj, np.floating):  # type:ignore
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpJsonEncoder, self).default(obj)


def float_representer(dumper: yaml.Dumper, value: float):
    text = "{0:.9f}".format(value)
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


def tensor_representer(dumper: yaml.Dumper, data: Tensor):
    assert type(data) == Tensor
    if data.shape:
        return dumper.represent_sequence("tag:yaml.org,2002:float", data.tolist())
    else:
        return float_representer(dumper, data.item())


yaml.add_representer(float, float_representer)
yaml.add_representer(Tensor, tensor_representer)
