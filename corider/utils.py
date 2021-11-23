import json
from pathlib import Path
from typing import Any

import yaml

# from typing import Dict, Union
# from argparse import Namespace

# def namespace(namespace_or_dict: Union[Namespace, Dict[str, Any]]) -> Namespace:
#     """If given a dict, it is converted it to an argparse.Namespace.Otherwise, no change is made"""
#     if isinstance(namespace_or_dict, dict):
#         return Namespace(**namespace_or_dict)
#     else:
#         return namespace_or_dict


def load_structured_data(path: Path):
    suffix = path.suffix
    assert suffix in {".json", ".yml", ".yaml"}
    if suffix == ".yml":
        suffix = ".yaml"
    d = {".json": load_json, ".yaml": load_yaml}[suffix](path)
    return d


def load_yaml(path: Path) -> Any:
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        data = json.load(f)
    return data
