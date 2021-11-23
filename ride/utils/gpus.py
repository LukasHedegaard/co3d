from typing import List, Optional, Union

from pytorch_lightning.utilities import device_parser


def parse_gpus(args_gpus: Optional[Union[int, str, List[int]]]) -> List[int]:
    return device_parser.parse_gpu_ids(args_gpus) or []


def parse_num_gpus(args_gpus: Optional[Union[int, str, List[int]]]) -> int:
    return len(parse_gpus(args_gpus))
